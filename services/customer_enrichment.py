"""Customer enrichment adapter for cross-sell recommendations.

The best upstream source identified from ACP is Hermes Prospect Bot, which can
research prospects via Quinn, OpenFunnel, Cognism, and Salesloft. We keep this
module read-only and cache-first for the Bot Week demo: it defines the target
schema and enriches accounts from local cache/heuristics today, while making the
provider ranking explicit for later ACP/A2A wiring.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


PROVIDER_RANKING = [
    {
        "agent_id": "alexsobin@telnyx.com:acp:hermes-prospect-bot",
        "name": "Hermes Prospect Bot",
        "fit": "best",
        "provides": ["prospect research", "company enrichment", "personalized outreach", "Quinn/OpenFunnel/Cognism/Salesloft context"],
        "notes": "Best match for customer/prospect enrichment and outbound sales context.",
    },
    {
        "agent_id": "rodolfoi@telnyx.com:acp:hermes-prospect-bot",
        "name": "Hermes Prospect Bot",
        "fit": "best",
        "provides": ["prospect research", "company enrichment", "personalized outreach", "Quinn/OpenFunnel/Cognism/Salesloft context"],
        "notes": "Same role/capability; healthy K8S status in directory snapshot.",
    },
    {
        "agent_id": "salvador@telnyx.com:acp:salvadors-hermes-bot",
        "name": "Salvador's HermesBot",
        "fit": "fallback",
        "provides": ["deal management", "prospection", "purchase intent", "customer info collection", "Quinn/OpenFunnel context"],
        "notes": "Good fallback for sales/deal context.",
    },
    {
        "agent_id": "zachs:acp:delphi",
        "name": "Delphi",
        "fit": "fallback",
        "provides": ["account research", "contact intelligence", "deal prep"],
        "notes": "Executive account research/deal-prep agent.",
    },
]

INDUSTRY_KEYWORDS = [
    ("healthcare", ["health", "clinic", "medical", "dental", "patient", "hospital", "care"]),
    ("financial_services", ["bank", "finance", "loan", "mortgage", "credit", "payment", "insurance", "capital"]),
    ("real_estate", ["realty", "real estate", "property", "mortgage", "broker"]),
    ("logistics", ["logistics", "delivery", "fleet", "transport", "courier", "shipping"]),
    ("retail", ["shop", "store", "retail", "commerce", "market"]),
    ("software_saas", ["software", "saas", "app", "platform", "cloud", "tech"]),
    ("telecom", ["telecom", "communications", "voip", "sip", "carrier", "wireless"]),
]

INDUSTRY_CROSS_SELL_HINTS = {
    "healthcare": ["Verify", "Messaging", "AI Voice Assistant", "HIPAA-aware workflows"],
    "financial_services": ["Verify", "Fraud controls", "Number Lookup", "Messaging compliance"],
    "real_estate": ["Messaging", "Branded Calling", "AI Voice Assistant"],
    "logistics": ["Messaging", "Voice failover", "Wireless/IoT"],
    "retail": ["Messaging", "Verify", "WhatsApp"],
    "software_saas": ["Verify", "Number Lookup", "Webhooks/Observability"],
    "telecom": ["SIP Trunking", "Voice analytics", "Messaging"],
}


@dataclass
class CustomerEnrichment:
    user_id: str
    provider: str = "heuristic"
    company_name: str = ""
    domain: str = ""
    industry: str = "unknown"
    company_size: str = "unknown"
    geo_footprint: List[str] = field(default_factory=list)
    likely_use_cases: List[str] = field(default_factory=list)
    buyer_personas: List[str] = field(default_factory=list)
    product_fit_hints: List[str] = field(default_factory=list)
    confidence: float = 0.25
    evidence: List[str] = field(default_factory=list)
    source_agent: Optional[str] = None
    unavailable_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CustomerEnrichmentProvider:
    """Cache-first customer enrichment provider."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.cache_path = data_dir / "customer_enrichment.json"
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        try:
            raw = json.loads(self.cache_path.read_text())
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, list):
                return {str(item.get("user_id")): item for item in raw if item.get("user_id")}
        except Exception:
            return {}
        return {}

    @staticmethod
    def provider_capabilities() -> List[Dict[str, Any]]:
        return PROVIDER_RANKING

    def enrich_account(self, account: Dict[str, Any]) -> Dict[str, Any]:
        uid = str(account.get("user_id") or "")
        if uid and uid in self.cache:
            cached = dict(self.cache[uid])
            cached.setdefault("provider", "cache")
            return cached
        return self._heuristic_enrichment(account).to_dict()

    def _heuristic_enrichment(self, account: Dict[str, Any]) -> CustomerEnrichment:
        uid = str(account.get("user_id") or "")
        company = str(account.get("business_name") or account.get("full_name") or "")
        email = str(account.get("email") or "")
        domain = self._domain_from_email(email)
        text = " ".join([
            company,
            domain,
            str(account.get("primary_use_case") or ""),
            str(account.get("country") or ""),
        ]).lower()
        industry = "unknown"
        evidence: List[str] = []
        for candidate, keywords in INDUSTRY_KEYWORDS:
            matched = [kw for kw in keywords if kw in text]
            if matched:
                industry = candidate
                evidence.append(f"matched industry keywords: {', '.join(matched[:3])}")
                break

        likely_use_cases: List[str] = []
        if account.get("has_voice") or account.get("connections_count", 0) > 0:
            likely_use_cases.append("voice communications")
        if account.get("has_messaging") or account.get("total_messaging_profiles", 0) > 0:
            likely_use_cases.append("messaging/customer notifications")
        if account.get("has_numbers") or account.get("numbers_count", 0) > 0:
            likely_use_cases.append("phone number inventory/local presence")
        if account.get("has_ai") or account.get("total_ai_assistants", 0) > 0:
            likely_use_cases.append("AI voice workflows")

        personas = ["RevOps", "Product", "Engineering"]
        if industry in {"healthcare", "financial_services"}:
            personas.append("Compliance")
        if account.get("cdr_voice_30d_raw", 0) or account.get("cdr_callcontrol_30d_raw", 0):
            personas.append("Contact Center / CX")

        hints = list(INDUSTRY_CROSS_SELL_HINTS.get(industry, []))
        if account.get("has_voice") and not account.get("has_messaging"):
            hints.extend(["Messaging", "Branded Calling", "AI Transcription"])
        if account.get("has_messaging") and not account.get("has_voice"):
            hints.extend(["Voice", "SIP Trunking", "Call Control"])
        if account.get("has_messaging") and "Verify" not in hints:
            hints.append("Verify")

        geo = []
        country = account.get("country")
        if country and country != "—":
            geo.append(str(country))

        confidence = 0.35 if industry != "unknown" else 0.2
        if domain:
            evidence.append(f"email domain: {domain}")

        return CustomerEnrichment(
            user_id=uid,
            provider="heuristic_fallback",
            company_name=company,
            domain=domain,
            industry=industry,
            geo_footprint=geo,
            likely_use_cases=likely_use_cases,
            buyer_personas=personas,
            product_fit_hints=sorted(set(hints)),
            confidence=confidence,
            evidence=evidence or ["no external enrichment cache; inferred from account fields"],
            source_agent=PROVIDER_RANKING[0]["agent_id"],
            unavailable_reason="ACP prospect enrichment not wired in this demo; using cache/heuristics",
        )

    @staticmethod
    def _domain_from_email(email: str) -> str:
        if "@" not in email or email.strip() in {"—", ""}:
            return ""
        domain = email.split("@", 1)[1].lower().strip()
        if domain in {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"}:
            return ""
        return re.sub(r"[^a-z0-9.-]", "", domain)
