"""
FastAPI Web Application — Cross-Sell Signal Detector Dashboard

Loads pre-computed pipeline data from JSON files (output of run_demo.py).
Routes:
    GET /              Dashboard home (summary + top accounts)
    GET /accounts      Full account list with filters
    GET /account/{uid} Account detail view
    GET /api/accounts  JSON — all accounts + signals + recommendations
    GET /refresh       Reload data from JSON files
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from services.customer_enrichment import CustomerEnrichmentProvider
from services.signal_detector import detect_signals
from services.inference_engine import generate_recommendations

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DEMO_DIR = DATA_DIR / "demo"  # Anonymized data for Bot Week

# Prefer anonymized demo data when available; fall back to raw data
PIPELINE_FILE = (DEMO_DIR / "demo_accounts.json") if (DEMO_DIR / "demo_accounts.json").exists() else (DATA_DIR / "demo_accounts.json")
ENRICHED_FILE = (DEMO_DIR / "enriched_accounts_cdr.json") if (DEMO_DIR / "enriched_accounts_cdr.json").exists() else (DATA_DIR / "enriched_accounts_cdr.json")
FULL_SCAN_FILE = (DEMO_DIR / "all_accounts_signals.json") if (DEMO_DIR / "all_accounts_signals.json").exists() else (DATA_DIR / "all_accounts_signals.json")
CUSTOMER_ENRICHMENT = CustomerEnrichmentProvider(DATA_DIR)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("cross-sell-dashboard")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Cross-Sell Signal Detector", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# Add custom Jinja filters
def fmt_date(value: str) -> str:
    """Format ISO date to human-readable."""
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y")
    except Exception:
        return value

def fmt_pct(value: float) -> str:
    """Format 0-1 float as percentage."""
    if value is None:
        return "0%"
    return f"{value * 100:.0f}%"

def fmt_money(value: Any) -> str:
    """Format as dollar amount."""
    try:
        return f"${float(value):,.0f}"
    except (TypeError, ValueError):
        return "$0"

templates.env.filters["fmt_date"] = fmt_date
templates.env.filters["fmt_pct"] = fmt_pct
templates.env.filters["fmt_money"] = fmt_money


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _safe_parse_estimated_value(raw: Any) -> dict:
    """Parse estimated_value which may be a string representation of a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except Exception:
            return {}
    return {}


def _is_dormant(acct: dict) -> bool:
    """An account is dormant if it has zero real product usage."""
    return (
        acct.get("connections_count", 0) == 0
        and acct.get("numbers_count", 0) == 0
        and acct.get("cdr_voice_30d", 0) == 0
        and acct.get("cdr_callcontrol_30d", 0) == 0
    )


def _compute_score(signals: list) -> float:
    """Opportunity score = sum(confidence * strength)."""
    return sum(s.get("confidence", 0) * s.get("strength", 0) for s in signals)


def _classify_tier(score: float) -> str:
    if score >= 1.5:
        return "hot"
    elif score >= 0.7:
        return "warm"
    return "cool"


def _format_date(value: str) -> str:
    return fmt_date(value)




def _signal_to_dict(sig: Any) -> Dict[str, Any]:
    if hasattr(sig, "to_dict"):
        return sig.to_dict()
    return dict(sig)


def _jsonify_dataclass(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonify_dataclass(asdict(value))
    if isinstance(value, dict):
        return {k: _jsonify_dataclass(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify_dataclass(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _rec_to_dict(rec: Any) -> Dict[str, Any]:
    if hasattr(rec, "to_dict"):
        data = rec.to_dict()
    else:
        data = _jsonify_dataclass(rec)
    if isinstance(data, dict):
        # Normalize inference-engine dataclass recommendations to the older
        # dashboard template schema used by the Bot Week demo.
        data.setdefault("product_name", data.get("target_product"))
        data.setdefault("feature_description", data.get("target_feature"))
        data.setdefault("evidence", data.get("evidence_bullets", []))
        data.setdefault("next_steps", data.get("integration_path", []))
        if "integration_effort" in data:
            data.setdefault("effort", str(data.get("integration_effort", "")).lower())
    return data


def _augment_account_opportunities(acct: Dict[str, Any], customer_enrichment: Dict[str, Any]) -> Dict[str, Any]:
    """Add newly-supported cross-sell signals/recs to precomputed demo data.

    The repo ships precomputed Bot Week JSON. This keeps old data useful while
    letting newly-added detectors (Messaging, Verify, Voice/SIP, support pain)
    show up immediately without rerunning a prod crawl.
    """
    existing_signals = list(acct.get("signals") or [])
    existing_names = {s.get("name") for s in existing_signals if isinstance(s, dict)}
    dynamic_signal_objs = detect_signals(acct)
    dynamic_signals = [_signal_to_dict(s) for s in dynamic_signal_objs]
    new_signal_objs = []
    for obj, sig in zip(dynamic_signal_objs, dynamic_signals):
        if sig.get("name") not in existing_names:
            existing_signals.append(sig)
            existing_names.add(sig.get("name"))
            new_signal_objs.append(obj)

    existing_recs = list(acct.get("recommendations") or [])
    existing_products = {r.get("target_product") or r.get("product_name") for r in existing_recs if isinstance(r, dict)}
    # Generate recommendations from dynamic signals only; legacy precomputed
    # dict signals do not share the inference dataclass type.
    rec_objs = generate_recommendations(dynamic_signal_objs, {"customer_enrichment": customer_enrichment})
    for rec in [_rec_to_dict(r) for r in rec_objs]:
        product = rec.get("target_product") or rec.get("product_name")
        if product not in existing_products:
            existing_recs.append(rec)
            existing_products.add(product)

    score = sum(float(s.get("confidence", 0)) * float(s.get("strength", 0)) for s in existing_signals if isinstance(s, dict))
    return {
        "signals": existing_signals,
        "recommendations": existing_recs,
        "opportunity_score": score,
    }

def _current_products(acct: Dict[str, Any]) -> List[str]:
    products: List[str] = []
    if acct.get("has_voice") or acct.get("connections_count", 0) or acct.get("cdr_voice_30d_raw", 0):
        products.append("Voice")
    if acct.get("has_messaging") or acct.get("total_messaging_profiles", 0):
        products.append("Messaging")
    if acct.get("has_ai") or acct.get("total_ai_assistants", 0):
        products.append("AI")
    if acct.get("has_numbers") or acct.get("numbers_count", 0):
        products.append("Numbers")
    return products


# ---------------------------------------------------------------------------
# Data store
# ---------------------------------------------------------------------------

class DashboardData:
    """Container for all processed dashboard data."""

    def __init__(self) -> None:
        self.accounts: List[Dict[str, Any]] = []
        self._loaded_at: Optional[str] = None

    def load(self) -> None:
        """Load and merge data from JSON files.

        Priority:
          1. Full-scan file (all_accounts_signals.json) — produced by crawler.py
          2. Pipeline + enriched files — produced by run_demo.py
        """
        pipeline_data = []

        # Try full-scan first (produced by crawler.py)
        if FULL_SCAN_FILE.exists():
            with open(FULL_SCAN_FILE) as f:
                pipeline_data = json.load(f)
            logger.info("Loaded %d accounts from FULL SCAN %s", len(pipeline_data), FULL_SCAN_FILE.name)
        elif PIPELINE_FILE.exists():
            with open(PIPELINE_FILE) as f:
                pipeline_data = json.load(f)
            logger.info("Loaded %d accounts from %s", len(pipeline_data), PIPELINE_FILE.name)
        else:
            logger.warning("No pipeline file at %s or %s", FULL_SCAN_FILE, PIPELINE_FILE)

        # Load enriched data (profile info)
        enriched_by_id: Dict[str, dict] = {}
        if ENRICHED_FILE.exists():
            with open(ENRICHED_FILE) as f:
                enriched_list = json.load(f)
            enriched_by_id = {a["user_id"]: a for a in enriched_list if "user_id" in a}
            logger.info("Loaded %d enriched accounts from %s", len(enriched_by_id), ENRICHED_FILE.name)

        # Merge
        is_full_scan = FULL_SCAN_FILE.exists()
        merged = []
        for acct in pipeline_data:
            uid = acct.get("user_id", "")

            if is_full_scan:
                # Full-scan format — data is already at top level
                enriched = enriched_by_id.get(uid, {})
                raw_user = enriched.get("raw_user", {})
                profile = raw_user.get("user_profile", {}) if isinstance(raw_user, dict) else {}

                first_name = profile.get("first_name", "") or ""
                last_name = profile.get("last_name", "") or ""
                full_name = f"{first_name} {last_name}".strip()

                _biz_raw = profile.get("business_name") or enriched.get("business_name")
                biz_name = (_biz_raw if _biz_raw and str(_biz_raw).strip().lower() not in ("none", "", "null") else None) or full_name or f"Account {uid[:8]}"

                signals = acct.get("signals", [])
                recommendations = acct.get("recommendations", [])
                score = acct.get("opportunity_score", 0)
                tier = _classify_tier(score)

                top_target = ""
                if recommendations:
                    top_target = recommendations[0].get("target_product", "")

                est_revenue = 0.0
                for rec in recommendations:
                    val = _safe_parse_estimated_value(rec.get("estimated_value"))
                    est_revenue += float(val.get("telnyx_revenue_monthly", 0))

                connections_count = acct.get("total_connections", 0) or 0
                numbers_count = acct.get("total_numbers", 0) or 0
                customer_enrichment = CUSTOMER_ENRICHMENT.enrich_account({**acct, **enriched, "user_id": uid, "business_name": biz_name, "full_name": full_name})
                augmented = _augment_account_opportunities({**acct, "connections_count": connections_count, "numbers_count": numbers_count}, customer_enrichment)
                signals = augmented["signals"]
                recommendations = augmented["recommendations"]
                score = augmented["opportunity_score"]
                tier = _classify_tier(score)
                top_target = recommendations[0].get("target_product", "") if recommendations else ""
                est_revenue = 0.0
                for rec in recommendations:
                    val = _safe_parse_estimated_value(rec.get("estimated_value"))
                    est_revenue += float(val.get("telnyx_revenue_monthly", 0))

                merged.append({
                    "user_id": uid,
                    "business_name": biz_name,
                    "full_name": full_name,
                    "email": acct.get("email") or enriched.get("email") or "—",
                    "country": acct.get("country") or "—",
                    "user_type": profile.get("user_type") or "—",
                    "kyc_status": acct.get("kyc_status") or "—",
                    "created_at": acct.get("created_at") or "—",
                    "primary_use_case": profile.get("primary_use_case") or "—",
                    "data_locality": profile.get("data_locality") or "—",
                    "connections_count": connections_count,
                    "connections_by_type": acct.get("voice_conn_types", {}),
                    "numbers_count": numbers_count,
                    "phone_numbers": [],
                    "cdr_voice_30d": "—",
                    "cdr_voice_30d_raw": 0,
                    "cdr_callcontrol_30d": "—",
                    "cdr_callcontrol_30d_raw": 0,
                    "signals": signals,
                    "recommendations": recommendations,
                    "opportunity_score": round(score, 2),
                    "opportunity_tier": tier,
                    "top_cross_sell_target": top_target,
                    "signal_count": len(signals),
                    "rec_count": len(recommendations),
                    "est_revenue_monthly": round(est_revenue, 2),
                    "is_dormant": False,
                    "enrichment_time_s": 0,
                    "customer_enrichment": customer_enrichment,
                    "industry": customer_enrichment.get("industry", "unknown"),
                    "product_fit_hints": customer_enrichment.get("product_fit_hints", []),
                    "products": _current_products({**acct, "connections_count": connections_count, "numbers_count": numbers_count}),
                    # Full-scan extras
                    "has_voice": acct.get("has_voice", False),
                    "has_messaging": acct.get("has_messaging", False),
                    "has_ai": acct.get("has_ai", False),
                    "has_numbers": acct.get("has_numbers", False),
                    "total_messaging_profiles": acct.get("total_messaging_profiles", 0),
                    "total_ai_assistants": acct.get("total_ai_assistants", 0),
                    "number_features": acct.get("number_features", []),
                })
            else:
                # Legacy demo-pipeline format
                enriched = enriched_by_id.get(uid, {})
                raw_user = enriched.get("raw_user", {})
                profile = raw_user.get("user_profile", {}) if isinstance(raw_user, dict) else {}

                first_name = profile.get("first_name", "") or ""
                last_name = profile.get("last_name", "") or ""
                full_name = f"{first_name} {last_name}".strip()

                _biz_raw = profile.get("business_name") or enriched.get("business_name")
                biz_name = (_biz_raw if _biz_raw and str(_biz_raw).strip().lower() not in ("none", "", "null") else None) or full_name or f"Account {uid[:8]}"

                dormant = _is_dormant(acct)
                signals = acct.get("signals", []) if not dormant else []
                recommendations = acct.get("recommendations", []) if not dormant else []
                score = _compute_score(signals)
                tier = _classify_tier(score) if not dormant else "dormant"

                top_target = ""
                if recommendations:
                    top_target = recommendations[0].get("target_product", "")

                est_revenue = 0.0
                for rec in recommendations:
                    val = _safe_parse_estimated_value(rec.get("estimated_value"))
                    est_revenue += float(val.get("telnyx_revenue_monthly", 0))

                connections_count = acct.get("connections_count", 0) or 0
                numbers_count = acct.get("numbers_count") or acct.get("phone_numbers_count", 0) or 0
                cdr_voice_raw = acct.get("cdr_voice_30d") or acct.get("cdr_total", 0) or 0
                cdr_cc_raw = acct.get("cdr_callcontrol_30d") or 0
                cdr_voice_capped = cdr_voice_raw >= 10000
                cdr_cc_capped = cdr_cc_raw >= 10000
                cdr_voice = f"{cdr_voice_raw:,}+" if cdr_voice_capped else cdr_voice_raw
                cdr_cc = f"{cdr_cc_raw:,}+" if cdr_cc_capped else cdr_cc_raw
                customer_enrichment = CUSTOMER_ENRICHMENT.enrich_account({**acct, **enriched, "user_id": uid, "business_name": biz_name, "full_name": full_name})
                augmented = _augment_account_opportunities({**acct, "connections_count": connections_count, "numbers_count": numbers_count}, customer_enrichment)
                signals = augmented["signals"]
                recommendations = augmented["recommendations"]
                score = augmented["opportunity_score"]
                tier = _classify_tier(score)
                top_target = recommendations[0].get("target_product", "") if recommendations else ""
                est_revenue = 0.0
                for rec in recommendations:
                    val = _safe_parse_estimated_value(rec.get("estimated_value"))
                    est_revenue += float(val.get("telnyx_revenue_monthly", 0))

                merged.append({
                    "user_id": uid,
                    "business_name": biz_name,
                    "full_name": full_name,
                    "email": enriched.get("email") or profile.get("verified_phone_number") or "—",
                    "country": profile.get("country") or raw_user.get("registration_ip_country") or "—",
                    "user_type": profile.get("user_type") or "—",
                    "kyc_status": enriched.get("kyc_status") or raw_user.get("kyc_status") or "—",
                    "created_at": raw_user.get("created_at") or acct.get("created_at") or "—",
                    "primary_use_case": profile.get("primary_use_case") or "—",
                    "data_locality": profile.get("data_locality") or "—",
                    "connections_count": connections_count,
                    "connections_by_type": {
                        k: len(v) if isinstance(v, list) else v
                        for k, v in (acct.get("connections_by_type") or {}).items()
                    },
                    "numbers_count": numbers_count,
                    "phone_numbers": acct.get("phone_numbers") or [],
                    "cdr_voice_30d": cdr_voice,
                    "cdr_voice_30d_raw": cdr_voice_raw,
                    "cdr_callcontrol_30d": cdr_cc,
                    "cdr_callcontrol_30d_raw": cdr_cc_raw,
                    "signals": signals,
                    "recommendations": recommendations,
                    "opportunity_score": round(score, 2),
                    "opportunity_tier": tier,
                    "top_cross_sell_target": top_target,
                    "signal_count": len(signals),
                    "rec_count": len(recommendations),
                    "est_revenue_monthly": round(est_revenue, 2),
                    "is_dormant": dormant,
                    "enrichment_time_s": acct.get("enrichment_time_s", 0),
                    "customer_enrichment": customer_enrichment,
                    "industry": customer_enrichment.get("industry", "unknown"),
                    "product_fit_hints": customer_enrichment.get("product_fit_hints", []),
                    "products": _current_products({**acct, "connections_count": connections_count, "numbers_count": numbers_count, "cdr_voice_30d_raw": cdr_voice_raw, "cdr_callcontrol_30d_raw": cdr_cc_raw}),
                })

        # Sort by opportunity_score descending, dormant last
        merged.sort(key=lambda a: (a["is_dormant"], -a["opportunity_score"]))

        self.accounts = merged
        self._loaded_at = datetime.now().strftime("%H:%M:%S")
        logger.info(
            "Dashboard ready: %d accounts, %d active, %d dormant",
            len(merged),
            sum(1 for a in merged if not a["is_dormant"]),
            sum(1 for a in merged if a["is_dormant"]),
        )

    # Convenience properties
    @property
    def total_accounts(self) -> int:
        return len(self.accounts)

    @property
    def active_accounts(self) -> int:
        return sum(1 for a in self.accounts if not a["is_dormant"])

    @property
    def total_signals(self) -> int:
        return sum(a["signal_count"] for a in self.accounts)

    @property
    def total_recommendations(self) -> int:
        return sum(a["rec_count"] for a in self.accounts)

    @property
    def hot_count(self) -> int:
        return sum(1 for a in self.accounts if a["opportunity_tier"] == "hot")

    @property
    def warm_count(self) -> int:
        return sum(1 for a in self.accounts if a["opportunity_tier"] == "warm")

    @property
    def cool_count(self) -> int:
        return sum(1 for a in self.accounts if a["opportunity_tier"] == "cool")

    @property
    def dormant_count(self) -> int:
        return sum(1 for a in self.accounts if a["is_dormant"])

    @property
    def total_est_revenue(self) -> float:
        return sum(a["est_revenue_monthly"] for a in self.accounts)


# Global data store
data = DashboardData()


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _signal_distribution() -> List[Dict[str, Any]]:
    """Count signals by id across all accounts."""
    dist: Dict[str, Dict[str, Any]] = {}
    for acct in data.accounts:
        for sig in acct["signals"]:
            sid = sig.get("id", "unknown")
            if sid not in dist:
                dist[sid] = {"name": sig.get("name", sid), "id": sid, "count": 0, "category": sig.get("category", "")}
            dist[sid]["count"] += 1
    return sorted(dist.values(), key=lambda x: x["count"], reverse=True)


def _rec_summary() -> List[Dict[str, Any]]:
    """Aggregate recommendations by target_product."""
    summary: Dict[str, Dict[str, Any]] = {}
    for acct in data.accounts:
        for rec in acct["recommendations"]:
            product = rec.get("target_product", "Unknown")
            val = _safe_parse_estimated_value(rec.get("estimated_value"))
            if product not in summary:
                summary[product] = {"product": product, "count": 0, "total_revenue": 0.0, "confidences": []}
            summary[product]["count"] += 1
            summary[product]["total_revenue"] += float(val.get("telnyx_revenue_monthly", 0))
            summary[product]["confidences"].append(rec.get("confidence", 0))

    result = []
    for entry in summary.values():
        confs = entry.pop("confidences")
        entry["avg_confidence"] = sum(confs) / len(confs) if confs else 0
        result.append(entry)
    return sorted(result, key=lambda x: x["total_revenue"], reverse=True)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting dashboard — loading data …")
    data.load()


# ---------------------------------------------------------------------------
# HTML Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard home."""
    signal_dist = _signal_distribution()
    max_sig_count = max((s["count"] for s in signal_dist), default=1)

    context = {
        "request": request,
        "data": data,
        "signal_distribution": signal_dist,
        "max_sig_count": max_sig_count,
        "rec_summary": _rec_summary(),
        "top_accounts": [a for a in data.accounts if not a["is_dormant"]][:10],
    }
    return templates.TemplateResponse("dashboard.html", context)


@app.get("/accounts", response_class=HTMLResponse)
async def accounts_list(
    request: Request,
    tier: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
):
    """Full account list with filters."""
    accounts = list(data.accounts)

    if tier and tier in ("hot", "warm", "cool", "dormant"):
        accounts = [a for a in accounts if a["opportunity_tier"] == tier]

    if q:
        q_lower = q.lower()
        accounts = [
            a for a in accounts
            if q_lower in (a.get("business_name", "") or "").lower()
            or q_lower in (a.get("email", "") or "").lower()
            or q_lower in (a.get("user_id", "") or "").lower()
        ]

    context = {
        "request": request,
        "accounts": accounts,
        "tier_filter": tier or "all",
        "search_query": q or "",
        "total_accounts": len(accounts),
        "data": data,
    }
    return templates.TemplateResponse("accounts.html", context)


@app.get("/account/{user_id}", response_class=HTMLResponse)
async def account_detail(request: Request, user_id: str):
    """Account detail view."""
    account = None
    for a in data.accounts:
        if a["user_id"] == user_id or a["user_id"].startswith(user_id):
            account = a
            break

    if account is None:
        return templates.TemplateResponse("account_detail.html", {
            "request": request, "account": None, "error": f"Account {user_id} not found"
        })

    # Parse estimated_value strings in recommendations for template use
    recs_parsed = []
    for rec in account["recommendations"]:
        r = dict(rec)
        r["estimated_value_parsed"] = _safe_parse_estimated_value(rec.get("estimated_value"))
        recs_parsed.append(r)

    # Connection type display names
    conn_type_names = {
        "credential_connection": "SIP Credential",
        "uac_connection": "User-Agent Config",
        "call_control_connection": "Call Control",
        "ip_connection": "IP Access",
        "fqdn_connection": "FQDN",
        "call_control_xml_connection": "TeXML",
        "connection": "Generic",
    }

    context = {
        "request": request,
        "account": account,
        "signals": account["signals"],
        "recommendations": recs_parsed,
        "conn_type_names": conn_type_names,
        "error": None,
    }
    return templates.TemplateResponse("account_detail.html", context)


@app.get("/refresh")
async def refresh_data():
    """Reload data from JSON files."""
    data.load()
    return RedirectResponse(url="/?msg=Data+refreshed", status_code=303)


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@app.get("/api/accounts")
async def api_accounts(
    tier: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
):
    """JSON API — all accounts with signals and recommendations."""
    accounts = list(data.accounts)
    if tier and tier in ("hot", "warm", "cool", "dormant"):
        accounts = [a for a in accounts if a["opportunity_tier"] == tier]
    if q:
        q_lower = q.lower()
        accounts = [a for a in accounts if q_lower in (a.get("business_name", "") or "").lower() or q_lower in (a.get("email", "") or "").lower()]
    return {"accounts": accounts, "total": len(accounts)}


@app.get("/api/account/{user_id}")
async def api_account(user_id: str):
    """JSON API — single account."""
    for a in data.accounts:
        if a["user_id"] == user_id or a["user_id"].startswith(user_id):
            return a
    return {"error": f"Account {user_id} not found"}, 404


@app.get("/api/enrichment/providers")
async def api_enrichment_providers():
    """JSON API — customer enrichment provider ranking/capabilities."""
    return {"providers": CUSTOMER_ENRICHMENT.provider_capabilities()}


@app.get("/api/signals")
async def api_signals():
    """JSON API — signal distribution."""
    return {"distribution": _signal_distribution()}


@app.get("/api/recommendations")
async def api_recommendations():
    """JSON API — recommendation summary."""
    return {"summary": _rec_summary()}
