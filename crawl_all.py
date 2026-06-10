#!/usr/bin/env python3
"""Full customer-base crawler for Cross-Sell Signal Detector.

Read-only crawler:
1. Paginates through all collectible customers via Private Users MCP.
2. Enriches accounts with product assets.
3. Runs cross-sell signal detection and recommendations.
4. Writes checked-vs-signal metrics to data/crawl_summary.json.

Use:
  .venv/bin/python crawl_all.py --full
  .venv/bin/python crawl_all.py --full --loop --sleep-seconds 21600
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from services.account_scanner import AccountScanner, AccountData
from services.customer_enrichment import CustomerEnrichmentProvider
from services.signal_detector import detect_signals
from services.inference_engine import generate_recommendations

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
ENRICHED_DIR = DATA_DIR / "enriched"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("cross-sell-crawler")


def jsonify(value: Any) -> Any:
    if is_dataclass(value):
        return jsonify(asdict(value))
    if isinstance(value, dict):
        return {k: jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonify(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


def signal_to_dict(sig: Any) -> Dict[str, Any]:
    if hasattr(sig, "to_dict"):
        return sig.to_dict()
    return jsonify(sig)


def rec_to_dict(rec: Any) -> Dict[str, Any]:
    data = jsonify(rec)
    data.setdefault("product_name", data.get("target_product"))
    data.setdefault("feature_description", data.get("target_feature"))
    data.setdefault("evidence", data.get("evidence_bullets", []))
    data.setdefault("next_steps", data.get("integration_path", []))
    if "integration_effort" in data:
        data.setdefault("effort", str(data.get("integration_effort", "")).lower())
    return data


def account_to_signal_input(account: AccountData) -> Dict[str, Any]:
    d = account.to_dict()
    cdr_total = account.cdr_summary.get("total_results", 0) if isinstance(account.cdr_summary, dict) else 0
    connection_types = account.connection_types
    phone_countries = sorted({str(pn.get("country_code")) for pn in account.phone_numbers if isinstance(pn, dict) and pn.get("country_code")})
    has_voice = len(account.connections) > 0 or cdr_total > 0
    has_messaging = len(account.messaging_profiles) > 0
    has_ai = any("ai" in str(c).lower() or "assistant" in str(c).lower() for c in account.connections)
    d.update({
        "has_voice": has_voice,
        "has_messaging": has_messaging,
        "has_ai": has_ai,
        "has_numbers": len(account.phone_numbers) > 0,
        "total_connections": len(account.connections),
        "total_numbers": len(account.phone_numbers),
        "total_messaging_profiles": len(account.messaging_profiles),
        "total_ai_assistants": 0,
        "total_calls": cdr_total,
        "voice_conn_types": {t: connection_types.count(t) for t in set(connection_types)},
        "number_features": phone_countries,
        "verify_profiles": account.verify_profiles,
        "messaging_profiles": account.messaging_profiles,
        "support_complaints_by_product": account.support_complaints_by_product,
    })
    return d


def build_account_record(account: AccountData, enrichment_provider: CustomerEnrichmentProvider) -> Dict[str, Any]:
    base = account_to_signal_input(account)
    enrichment = enrichment_provider.enrich_account({
        "user_id": account.user_id,
        "email": account.email,
        "business_name": account.business_name,
        "country": account.country,
        "has_voice": base["has_voice"],
        "has_messaging": base["has_messaging"],
        "has_ai": base["has_ai"],
        "has_numbers": base["has_numbers"],
        "total_messaging_profiles": base["total_messaging_profiles"],
        "numbers_count": base["total_numbers"],
        "connections_count": base["total_connections"],
    })
    signals = detect_signals(base)
    recs = generate_recommendations(signals, {"customer_enrichment": enrichment})
    sig_dicts = [signal_to_dict(s) for s in signals]
    rec_dicts = [rec_to_dict(r) for r in recs]
    score = sum(float(s.get("confidence", 0)) * float(s.get("strength", 0)) for s in sig_dicts)
    return {
        "user_id": account.user_id,
        "email": account.email,
        "business_name": account.business_name,
        "country": account.country,
        "kyc_status": account.kyc_status,
        "created_at": account.created_at,
        "has_voice": base["has_voice"],
        "has_messaging": base["has_messaging"],
        "has_ai": base["has_ai"],
        "has_numbers": base["has_numbers"],
        "total_connections": base["total_connections"],
        "total_numbers": base["total_numbers"],
        "total_messaging_profiles": base["total_messaging_profiles"],
        "total_ai_assistants": base["total_ai_assistants"],
        "voice_conn_types": base["voice_conn_types"],
        "number_features": base["number_features"],
        "support_complaints_by_product": base["support_complaints_by_product"],
        "customer_enrichment": enrichment,
        "industry": enrichment.get("industry", "unknown"),
        "product_fit_hints": enrichment.get("product_fit_hints", []),
        "opportunity_score": round(score, 3),
        "signal_count": len(sig_dicts),
        "signals": sig_dicts,
        "recommendations": rec_dicts,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def run_once(args: argparse.Namespace) -> Dict[str, Any]:
    DATA_DIR.mkdir(exist_ok=True)
    ENRICHED_DIR.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc)
    progress_path = DATA_DIR / "crawl_progress.json"
    write_json(progress_path, {"status": "running", "started_at": started.isoformat(), "phase": "scan"})

    scanner = AccountScanner(
        max_accounts=args.max_accounts,
        collect_limit=None if args.full else args.collect_limit,
    )
    enrichment_provider = CustomerEnrichmentProvider(DATA_DIR)
    try:
        accounts = scanner.scan_all()
        write_json(DATA_DIR / "enriched_accounts_cdr.json", [a.to_dict() for a in accounts])
        for account in accounts:
            write_json(ENRICHED_DIR / f"{account.user_id}.json", account.to_dict())

        write_json(progress_path, {
            "status": "running",
            "started_at": started.isoformat(),
            "phase": "signals",
            **scanner.scan_stats,
        })

        records: List[Dict[str, Any]] = []
        signal_accounts = 0
        for idx, account in enumerate(accounts, 1):
            record = build_account_record(account, enrichment_provider)
            records.append(record)
            if record["signal_count"] > 0:
                signal_accounts += 1
            if idx % args.checkpoint_every == 0:
                write_json(DATA_DIR / "all_accounts_signals.json", records)
                write_json(progress_path, {
                    "status": "running",
                    "started_at": started.isoformat(),
                    "phase": "signals",
                    "processed_usage_accounts": idx,
                    "accounts_with_signals": signal_accounts,
                    **scanner.scan_stats,
                })

        records.sort(key=lambda r: r["opportunity_score"], reverse=True)
        write_json(DATA_DIR / "all_accounts_signals.json", records)

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "started_at": started.isoformat(),
            "duration_seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 2),
            "accounts_checked": scanner.scan_stats.get("accounts_checked", 0),
            "candidates_collected": scanner.scan_stats.get("candidates_collected", 0),
            "accounts_with_usage": scanner.scan_stats.get("accounts_with_usage", len(accounts)),
            "accounts_with_signals": signal_accounts,
            "accounts_without_signals": max(0, len(accounts) - signal_accounts),
            "signal_yield_pct": round((signal_accounts / scanner.scan_stats.get("accounts_checked", 1)) * 100, 2) if scanner.scan_stats.get("accounts_checked") else 0,
            "errors": scanner.scan_stats.get("errors", 0),
            "top_products": {},
            "top_20": records[:20],
        }
        for record in records:
            for rec in record.get("recommendations", []):
                product = rec.get("target_product") or rec.get("product_name") or "Unknown"
                summary["top_products"][product] = summary["top_products"].get(product, 0) + 1
        write_json(DATA_DIR / "crawl_summary.json", summary)
        write_json(progress_path, {"status": "complete", **summary})
        logger.info(
            "Crawl complete: checked=%s usage=%s signals=%s yield=%s%%",
            summary["accounts_checked"], summary["accounts_with_usage"], summary["accounts_with_signals"], summary["signal_yield_pct"],
        )
        return summary
    finally:
        scanner.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Full cross-sell customer-base crawler")
    parser.add_argument("--full", action="store_true", help="crawl entire collectible customer base")
    parser.add_argument("--collect-limit", type=int, default=300, help="candidate limit when not using --full")
    parser.add_argument("--max-accounts", type=int, default=None, help="stop after this many accounts with usage")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="write partial signal file every N usage accounts")
    parser.add_argument("--loop", action="store_true", help="keep crawling forever")
    parser.add_argument("--sleep-seconds", type=int, default=21600, help="sleep between loop runs; default 6h")
    args = parser.parse_args()

    while True:
        try:
            run_once(args)
        except Exception as exc:
            logger.exception("Crawl failed: %s", exc)
            write_json(DATA_DIR / "crawl_progress.json", {
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            })
        if not args.loop:
            break
        logger.info("Sleeping %ss before next crawl", args.sleep_seconds)
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
