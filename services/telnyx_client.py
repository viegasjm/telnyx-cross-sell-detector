"""
Telnyx API v2 Client — fetches live account data with pagination.

Requires the macOS keychain entry `TELNYX_AI_API_KEY` (set via
`security add-generic-password -s TELNYX_AI_API_KEY -w <key>`).

All public methods return plain dicts/lists and never raise on
individual-endpoint failures (empty list is returned instead).
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.telnyx.com"
TIMEOUT = 15.0  # seconds
PAGE_SIZE = 250  # maximum allowed by Telnyx


# ---------------------------------------------------------------------------
# Keychain helper
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Retrieve the Telnyx API key from the macOS keychain."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "TELNYX_AI_API_KEY", "-w"],
            capture_output=True,
            text=True,
            check=True,
        )
        key = result.stdout.strip()
        if not key:
            raise RuntimeError("Keychain returned an empty value for TELNYX_AI_API_KEY")
        return key
    except FileNotFoundError:
        raise RuntimeError(
            "The `security` command is not available — this module requires macOS."
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to read TELNYX_AI_API_KEY from keychain: {exc.stderr.strip()}"
        ) from exc


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class TelnyxClient:
    """Async-compatible (via httpx) client that paginates through Telnyx v2 endpoints."""

    def __init__(self, api_key: str | None = None):
        self._api_key: str = api_key or _get_api_key()
        self._base_url: str = BASE_URL
        self._timeout: float = TIMEOUT
        self._headers: Dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

    # -- internal helpers ---------------------------------------------------

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
        )

    def _paginate(self, client: httpx.Client, path: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """Walk through every page of a Telnyx list endpoint and return all records."""
        items: List[Dict[str, Any]] = []
        query: Dict[str, Any] = dict(params or {})
        query.setdefault("page[size]", PAGE_SIZE)
        query["page[number]"] = 1

        while True:
            try:
                resp = client.get(path, params=query)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.error("Error fetching %s (page %s): %s", path, query["page[number]"], exc)
                break

            body = resp.json()
            data = body.get("data", [])
            items.extend(data)

            # Telnyx uses cursor-based or page-number pagination depending on
            # the endpoint.  We check for `meta.page_number` / `meta.total_pages`
            # as well as the absence of a next page in cursor style.
            meta = body.get("meta") or {}

            # Cursor-based: if no more pages, we're done.
            if "next_page_token" in meta and not meta.get("next_page_token"):
                break

            total_pages = meta.get("total_pages")
            current_page = meta.get("page_number", query["page[number]"])

            if total_pages is not None:
                if current_page >= total_pages:
                    break
                query["page[number]"] = current_page + 1
            else:
                # Page-number style but total_pages not reported — stop when
                # a page returns fewer items than requested.
                if len(data) < query["page[size]"]:
                    break
                query["page[number]"] += 1

        return items

    def _fetch_list(self, path: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        """Convenience wrapper: open a client, paginate, return list (empty on error)."""
        try:
            with self._client() as client:
                return self._paginate(client, path, params)
        except Exception as exc:
            logger.error("Unhandled error fetching %s: %s", path, exc)
            return []

    # -- public endpoint methods --------------------------------------------

    def get_connections(self) -> List[Dict[str, Any]]:
        """GET /v2/connections — all connection objects."""
        return self._fetch_list("/v2/connections")

    def get_phone_numbers(self) -> List[Dict[str, Any]]:
        """GET /v2/phone_numbers"""
        return self._fetch_list("/v2/phone_numbers")

    def get_messaging_phone_numbers(self) -> List[Dict[str, Any]]:
        """GET /v2/messaging_phone_numbers"""
        return self._fetch_list("/v2/messaging_phone_numbers")

    def get_call_control_applications(self) -> List[Dict[str, Any]]:
        """GET /v2/call_control_applications"""
        return self._fetch_list("/v2/call_control_applications")

    def get_texml_applications(self) -> List[Dict[str, Any]]:
        """GET /v2/texml_applications"""
        return self._fetch_list("/v2/texml_applications")

    def get_ai_assistants(self) -> List[Dict[str, Any]]:
        """GET /v2/ai/assistants"""
        return self._fetch_list("/v2/ai/assistants")

    def get_outbound_voice_profiles(self) -> List[Dict[str, Any]]:
        """GET /v2/outbound_voice_profiles"""
        return self._fetch_list("/v2/outbound_voice_profiles")

    def get_credential_connections(self) -> List[Dict[str, Any]]:
        """GET /v2/credential_connections"""
        return self._fetch_list("/v2/credential_connections")

    def get_ip_connections(self) -> List[Dict[str, Any]]:
        """GET /v2/ip_connections"""
        return self._fetch_list("/v2/ip_connections")

    def get_fqdn_connections(self) -> List[Dict[str, Any]]:
        """GET /v2/fqdn_connections"""
        return self._fetch_list("/v2/fqdn_connections")

    def get_messaging_profiles(self) -> List[Dict[str, Any]]:
        """GET /v2/messaging_profiles"""
        return self._fetch_list("/v2/messaging_profiles")

    def get_verify_profiles(self) -> List[Dict[str, Any]]:
        """GET /v2/verify_profiles"""
        return self._fetch_list("/v2/verify_profiles")

    # -- aggregate summary --------------------------------------------------

    def get_account_summary(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch every endpoint and return a single summary dict.

        Each key maps to the list returned by the corresponding method.
        Individual failures are swallowed and replaced with an empty list
        so the caller always gets a complete (if partial) picture.
        """
        fetch_methods = [
            ("connections", self.get_connections),
            ("phone_numbers", self.get_phone_numbers),
            ("messaging_phone_numbers", self.get_messaging_phone_numbers),
            ("call_control_applications", self.get_call_control_applications),
            ("texml_applications", self.get_texml_applications),
            ("ai_assistants", self.get_ai_assistants),
            ("outbound_voice_profiles", self.get_outbound_voice_profiles),
            ("credential_connections", self.get_credential_connections),
            ("ip_connections", self.get_ip_connections),
            ("fqdn_connections", self.get_fqdn_connections),
            ("messaging_profiles", self.get_messaging_profiles),
            ("verify_profiles", self.get_verify_profiles),
        ]

        summary: Dict[str, List[Dict[str, Any]]] = {}
        for name, method in fetch_methods:
            try:
                summary[name] = method()
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", name, exc)
                summary[name] = []

        return summary


# ---------------------------------------------------------------------------
# Quick-standalone helper
# ---------------------------------------------------------------------------

def fetch_summary(api_key: str | None = None) -> Dict[str, List[Dict[str, Any]]]:
    """One-liner to get a full account summary dict."""
    return TelnyxClient(api_key=api_key).get_account_summary()


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("Fetching Telnyx account summary …")
    data = fetch_summary()
    for key, items in data.items():
        print(f"  {key}: {len(items)} record(s)")
    print(json.dumps(data, indent=2, default=str))
