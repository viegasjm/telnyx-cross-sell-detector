"""
Multi-Account Scanner for Cross-Sell Detection.

Two-phase strategy:
  Phase 1 — Collect: Paginate through users, collecting non-blocked/non-cancelled
            accounts. Stop after `collect_limit` candidates gathered.
  Phase 2 — Enrich & Filter: Fetch connections, phone numbers, and CDR for each
            candidate. Only keep accounts with actual product usage (≥1 connection
            or ≥1 phone number). This is the real "qualified" filter.

The old approach required KYC="approved" which doesn't exist (the API returns
"accepted") and filtered out most real accounts. The asset-based filter is both
more accurate (we only care about accounts using the platform) and more inclusive.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from services.mcp_client import MCPClient

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
CDR_API_URL = "http://detail-record-search-ui.query.prod.telnyx.io:3000/api/search-dynamic"
CDR_TIMEOUT = 10.0
MCP_ENDPOINT = "https://private-users-mcp.query.prod.telnyx.io:3000/mcp"


@dataclass
class AccountData:
    """Enriched data for a single Telnyx account."""
    user_id: str
    email: str
    business_name: str = ""
    user_type: str = ""
    country: str = ""
    kyc_status: str = ""
    account_blocked: bool = False
    account_dormant: bool = False
    account_cancelled: bool = False
    created_at: str = ""
    is_account_manager: bool = False
    connections: List[Dict[str, Any]] = field(default_factory=list)
    phone_numbers: List[Dict[str, Any]] = field(default_factory=list)
    cdr_summary: Dict[str, Any] = field(default_factory=dict)
    raw_user: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def has_usage(self) -> bool:
        """True if this account has at least one connection or phone number."""
        return len(self.connections) > 0 or len(self.phone_numbers) > 0

    @property
    def connection_types(self) -> List[str]:
        """Unique connection types across all connections."""
        return list({c.get("record_type", "unknown") for c in self.connections})


class AccountScanner:
    """Scans Telnyx accounts via MCP + CDR.

    Usage:
        scanner = AccountScanner(max_accounts=20)
        accounts = scanner.scan_all()
    """

    def __init__(
        self,
        cache_dir: str = "data",
        max_accounts: Optional[int] = None,
        # How many raw candidates to collect before enrichment.
        # We over-collect because many will have zero assets.
        collect_limit: int = 300,
    ) -> None:
        self.cache_dir = PROJECT_DIR / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_accounts = max_accounts
        self.collect_limit = collect_limit
        self._cache_path = self.cache_dir / "scan_results.json"
        self._mcp: Optional[MCPClient] = None

    def _get_mcp(self) -> MCPClient:
        if self._mcp is None:
            token = self._get_auth_token()
            self._mcp = MCPClient(endpoint=MCP_ENDPOINT, auth_token=token, timeout=30.0)
            self._mcp.initialize()
            logger.info("MCP session established")
        return self._mcp

    @staticmethod
    def _get_auth_token() -> str:
        try:
            return subprocess.check_output(
                ["security", "find-generic-password", "-s", "TELNYX_PROD_API_KEY", "-w"],
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            raise RuntimeError("Could not get TELNYX_PROD_API_KEY from keychain")

    # -- Public API ----------------------------------------------------------

    def scan_all(self) -> List[AccountData]:
        """Two-phase scan: collect candidates, enrich, filter by usage."""
        logger.info("Starting account scan (collect_limit=%d, max_accounts=%s)",
                     self.collect_limit, self.max_accounts)

        # Phase 1: Collect raw candidates
        candidates = self._collect_candidates()
        logger.info("Collected %d raw candidate(s)", len(candidates))

        # Phase 2: Enrich each candidate and filter
        accounts: List[AccountData] = []
        enriched_with_usage = 0
        for idx, user in enumerate(candidates, 1):
            user_id = user.get("id", "")
            email = str(user.get("email", "unknown"))[:40]
            t0 = time.time()
            try:
                acct = self._enrich_account(user)
                if acct.has_usage:
                    accounts.append(acct)
                    enriched_with_usage += 1
                    logger.info(
                        "[%d/%d] ✓ %s — %d conns, %d nums, CDR=%s (%.1fs)",
                        idx, len(candidates), email,
                        len(acct.connections), len(acct.phone_numbers),
                        acct.cdr_summary.get("total_results", "?") if acct.cdr_summary else "?",
                        time.time() - t0,
                    )
                else:
                    logger.debug("[%d/%d] ✗ %s — no assets, skipping", idx, len(candidates), email)
            except Exception as exc:
                logger.warning("[%d/%d] Error enriching %s: %s", idx, len(candidates), email, exc)
                continue

            if self.max_accounts and enriched_with_usage >= self.max_accounts:
                logger.info("Reached max_accounts (%d), stopping enrichment", self.max_accounts)
                break

        self.save_cache(accounts)
        logger.info("Scan complete — %d account(s) with usage out of %d candidates, cached",
                     len(accounts), len(candidates))
        return accounts

    def scan_from_cache(self) -> List[AccountData]:
        if not self._cache_path.exists():
            logger.warning("Cache not found: %s", self._cache_path)
            return []
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Cache read error: %s", exc)
            return []
        accounts = []
        for item in raw:
            try:
                accounts.append(AccountData(**item))
            except Exception as exc:
                logger.warning("Bad cache entry: %s", exc)
        logger.info("Loaded %d account(s) from cache", len(accounts))
        return accounts

    def save_cache(self, accounts: List[AccountData]) -> None:
        data = [a.to_dict() for a in accounts]
        try:
            self._cache_path.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
            logger.info("Cached %d account(s) → %s", len(accounts), self._cache_path)
        except OSError as exc:
            logger.error("Cache write error: %s", exc)

    def close(self) -> None:
        if self._mcp:
            self._mcp.close()
            self._mcp = None

    # -- Phase 1: Candidate Collection ---------------------------------------

    def _collect_candidates(self) -> List[Dict[str, Any]]:
        """Paginate through users, collecting non-blocked/non-cancelled ones."""
        mcp = self._get_mcp()
        candidates: List[Dict[str, Any]] = []
        page = 1
        page_size = 100

        while True:
            result = mcp.call_tool(
                "listPrivateUsers",
                {"page_size": page_size, "page_number": page},
            )

            if not isinstance(result, dict):
                logger.warning("Unexpected response type: %s", type(result))
                break

            data = result.get("data", [])
            if not data:
                break

            page_candidates = 0
            for user in data:
                if self._is_collectible(user):
                    candidates.append(user)
                    page_candidates += 1

            logger.info("Page %d: fetched %d users, %d collectible, %d total candidates",
                        page, len(data), page_candidates, len(candidates))

            # Stop if we have enough candidates
            if len(candidates) >= self.collect_limit:
                logger.info("Reached collect_limit (%d), stopping", self.collect_limit)
                break

            # Check if more pages
            meta = result.get("meta", {})
            total_pages = meta.get("total_pages")
            if total_pages is not None and page >= total_pages:
                break
            if len(data) < page_size:
                break

            page += 1

        return candidates[:self.collect_limit]

    @staticmethod
    def _is_collectible(user: Dict[str, Any]) -> bool:
        """Quick pre-filter before enrichment. Only excludes clearly dead accounts."""
        email = (user.get("email") or "").strip()
        # Skip # prefixed test accounts
        if email.startswith("#"):
            return False
        # Skip clearly dead accounts
        if user.get("account_blocked"):
            return False
        if user.get("account_cancelled"):
            return False
        return True

    # -- Phase 2: Enrichment -------------------------------------------------

    def _enrich_account(self, user: Dict[str, Any]) -> AccountData:
        """Fetch connections, phone numbers, and CDR for a single user."""
        user_id = str(user.get("id", ""))
        mcp = self._get_mcp()

        # Fetch assets in parallel-ish (sequential MCP calls but same session)
        connections = mcp.call_tool(
            "listPrivateConnections",
            {"filter_user_id": user_id, "page_size": 100},
        )
        phone_numbers = mcp.call_tool(
            "listPrivatePhoneNumbers",
            {"filter_user_id": user_id, "page_size": 100},
        )
        cdr = self._fetch_cdr_sample(user_id)

        conn_data = connections.get("data", []) if isinstance(connections, dict) else []
        num_data = phone_numbers.get("data", []) if isinstance(phone_numbers, dict) else []

        profile = user.get("user_profile") or {}
        if isinstance(profile, str):
            profile = {}

        return AccountData(
            user_id=user_id,
            email=str(user.get("email", "")),
            business_name=str(profile.get("business_name", "")),
            user_type=str(profile.get("user_type", "")),
            country=str(profile.get("country", "")),
            kyc_status=str(user.get("kyc_status", "")),
            account_blocked=bool(user.get("account_blocked", False)),
            account_dormant=bool(user.get("account_dormant", False)),
            account_cancelled=bool(user.get("account_cancelled", False)),
            created_at=str(user.get("created_at", "")),
            is_account_manager=bool(user.get("is_account_manager", False)),
            connections=conn_data,
            phone_numbers=num_data,
            cdr_summary=cdr,
            raw_user=user,
        )

    def _fetch_cdr_sample(self, user_id: str) -> Dict[str, Any]:
        payload = {
            "recordType": "sip-trunking",
            "timePreset": "last_30d",
            "primaryFilters": {"user_id": user_id},
            "pageSize": 5,
            "page": 1,
            "sort": "started_at",
            "sortDirection": "desc",
            "region": "USA",
        }
        try:
            with httpx.Client(timeout=CDR_TIMEOUT) as client:
                resp = client.post(CDR_API_URL, json=payload)
                resp.raise_for_status()
                body = resp.json()
                return {
                    "total_results": body.get("meta", {}).get("total_results", 0),
                    "records": body.get("data", []),
                }
        except Exception as exc:
            logger.debug("CDR fetch failed for %s: %s", user_id, exc)
            return {"total_results": 0, "records": []}


def main() -> None:
    """CLI: scan N accounts and print summary."""
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    max_accounts = None
    if len(sys.argv) > 1:
        try:
            max_accounts = int(sys.argv[1])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [max_accounts]", file=sys.stderr)
            sys.exit(1)

    scanner = AccountScanner(max_accounts=max_accounts)
    try:
        accounts = scanner.scan_all()
        print(f"\n{'='*70}")
        print(f"Scan Results: {len(accounts)} account(s) with product usage")
        print(f"{'='*70}")
        for acct in accounts:
            cdr_total = acct.cdr_summary.get("total_results", 0) if acct.cdr_summary else 0
            print(
                f"  {acct.email[:40]:40s}  "
                f"conns={len(acct.connections):3d}  "
                f"nums={len(acct.phone_numbers):3d}  "
                f"cdr={cdr_total:5d}  "
                f"kyc={acct.kyc_status:10s}  "
                f"biz={acct.business_name[:20] or '—'}"
            )
    finally:
        scanner.close()


if __name__ == "__main__":
    main()
