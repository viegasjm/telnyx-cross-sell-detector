"""
Signal Detector for Cross-Sell Detection — v2 (aligned with inference_engine templates)

Produces Signal objects whose IDs, names, and categories match
the InferenceEngine's RecommendationTemplate requirements exactly.

Signal → Template matching rules (in _match_and_build):
  - name_match: signal.name in template.required_signal_names
  - category_match: signal.category in template.required_signal_categories (enum)
  - Match requires: name_match OR (category_match AND ≥2 signals in cluster)

So each signal's `name` must be one of the template's `required_signal_names`,
and `category` must be a `SignalCategory` enum value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from services.inference_engine import SignalCategory


# ---------------------------------------------------------------------------
# Data model — uses the SAME Signal that inference_engine expects
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """A detected cross-sell signal for an account."""
    id: str
    name: str           # Must match a required_signal_names entry in a template
    description: str
    confidence: float   # 0.0–1.0
    strength: float      # 0.0–1.0
    evidence: str
    cross_sell_target: str
    category: SignalCategory  # Must be SignalCategory enum

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "confidence": self.confidence,
            "strength": self.strength,
            "evidence": self.evidence,
            "cross_sell_target": self.cross_sell_target,
            "category": self.category.value if isinstance(self.category, SignalCategory) else self.category,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_keyword(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)

_AI_KW = ["ai", "assistant", "gpt", "llm", "agent", "intelligent", "bot"]
_IVR_KW = ["ivr", "menu", "dtmf", "press", "extension", "auto-attendant"]
_TEXML_KW = ["texml", "twiml", "xml dial"]
_MSG_KW = ["sms", "mms", "messaging", "message", "text"]
_VERIFY_KW = ["verify", "verification", "2fa", "two-factor", "otp"]
_RECORD_KW = ["record", "recording", "transcri"]


def _safe_len(val: Any) -> int:
    if val is None:
        return 0
    return len(val)


def _connection_text(conn: dict) -> str:
    parts = []
    if conn.get("connection_name"):
        parts.append(conn["connection_name"])
    if conn.get("user_agent"):
        parts.append(conn["user_agent"])
    if conn.get("type"):
        parts.append(conn["type"])
    return " ".join(parts)


def _phone_number_text(pn: dict) -> str:
    parts = []
    if pn.get("phone_number"):
        parts.append(str(pn["phone_number"]))
    if pn.get("country_code"):
        parts.append(pn["country_code"])
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Individual signal detectors — names match template required_signal_names
# ---------------------------------------------------------------------------

def _detect_has_call_control(data: dict) -> Optional[Signal]:
    """has_call_control: Account has Call Control applications."""
    cc_apps = data.get("call_control_apps", [])
    connections = data.get("connections", [])
    has_cc = len(cc_apps) > 0 or any(
        c.get("type") == "call-control" or _has_keyword(_connection_text(c), ["call control", "call-control"])
        for c in connections
    )
    if has_cc:
        return Signal(
            id="has_call_control",
            name="has_call_control",
            description="Account has Call Control applications configured",
            confidence=0.95,
            strength=0.9,
            evidence=f"{len(cc_apps)} Call Control app(s), {len(connections)} connection(s)",
            cross_sell_target="AI Call Routing",
            category=SignalCategory.VOICE_USAGE,
        )
    return None


def _detect_has_voice_product(data: dict) -> Optional[Signal]:
    """has_voice_product: Account has voice/SIP connections."""
    connections = data.get("connections", [])
    voice_types = {"call-control", "credential", "ip", "fqdn"}
    voice_conns = [c for c in connections if c.get("type", "").lower() in voice_types
                   or _has_keyword(_connection_text(c), ["sip", "voice", "call", "pbx"])]
    if voice_conns:
        return Signal(
            id="has_voice_product",
            name="has_voice_product",
            description="Account has voice/SIP connections",
            confidence=0.95,
            strength=0.9,
            evidence=f"{len(voice_conns)} voice-type connection(s) out of {len(connections)} total",
            cross_sell_target="AI Voice Assistant",
            category=SignalCategory.VOICE_USAGE,
        )
    return None


def _detect_has_voice_calls(data: dict) -> Optional[Signal]:
    """has_voice_calls: Account has made or received voice calls (CDR evidence)."""
    cdr = data.get("cdr_summary", {})
    total_calls = cdr.get("total_calls", 0) if isinstance(cdr, dict) else 0
    if total_calls > 0:
        return Signal(
            id="has_voice_calls",
            name="has_voice_calls",
            description=f"Account has {total_calls} calls in recent period",
            confidence=0.95,
            strength=0.85,
            evidence=f"{total_calls} calls recorded in CDR",
            cross_sell_target="AI Transcription & Analysis",
            category=SignalCategory.VOICE_USAGE,
        )
    # Also check connections as proxy
    connections = data.get("connections", [])
    if len(connections) >= 3:
        return Signal(
            id="has_voice_calls",
            name="has_voice_calls",
            description=f"Account has {len(connections)} connections (call volume inferred)",
            confidence=0.6,
            strength=0.5,
            evidence=f"No CDR data, but {len(connections)} connections suggest voice usage",
            cross_sell_target="AI Transcription & Analysis",
            category=SignalCategory.VOICE_USAGE,
        )
    return None


def _detect_high_call_volume(data: dict) -> Optional[Signal]:
    """high_call_volume: Account handles high call volumes."""
    cdr = data.get("cdr_summary", {})
    total_calls = cdr.get("total_calls", 0) if isinstance(cdr, dict) else 0
    if total_calls >= 100:
        return Signal(
            id="high_call_volume",
            name="high_call_volume",
            description=f"High call volume: {total_calls} calls in period",
            confidence=0.95,
            strength=0.9,
            evidence=f"{total_calls} calls in CDR",
            cross_sell_target="AI Call Routing",
            category=SignalCategory.VOICE_USAGE,
        )
    elif total_calls >= 30:
        return Signal(
            id="high_call_volume",
            name="high_call_volume",
            description=f"Moderate call volume: {total_calls} calls in period",
            confidence=0.7,
            strength=0.6,
            evidence=f"{total_calls} calls in CDR",
            cross_sell_target="AI Call Routing",
            category=SignalCategory.VOICE_USAGE,
        )
    return None


def _detect_high_inbound_call_volume(data: dict) -> Optional[Signal]:
    """high_inbound_call_volume: Account receives many inbound calls."""
    cdr = data.get("cdr_summary", {})
    inbound = cdr.get("inbound_count", 0) if isinstance(cdr, dict) else 0
    if inbound >= 20:
        return Signal(
            id="high_inbound_call_volume",
            name="high_inbound_call_volume",
            description=f"High inbound call volume: {inbound} inbound calls",
            confidence=0.9,
            strength=0.85,
            evidence=f"{inbound} inbound calls in period",
            cross_sell_target="AI Voice Assistant",
            category=SignalCategory.VOICE_USAGE,
        )
    return None


def _detect_no_ai_usage(data: dict) -> Optional[Signal]:
    """no_ai_usage: Account has voice but no AI services."""
    connections = data.get("connections", [])
    has_ai = any(_has_keyword(_connection_text(c), _AI_KW) for c in connections)
    texml_apps = data.get("texml_apps", [])
    has_ai_texml = any(_has_keyword(a.get("name", "") + " " + a.get("xml", ""), _AI_KW) for a in texml_apps)
    ai_assistants = data.get("ai_assistants", [])

    if not has_ai and not has_ai_texml and len(ai_assistants) == 0:
        return Signal(
            id="no_ai_usage",
            name="no_ai_usage",
            description="Account has voice products but no AI services enabled",
            confidence=0.85,
            strength=0.8,
            evidence="No AI assistants, no AI-named connections, no AI TeXML apps found",
            cross_sell_target="AI Call Routing",
            category=SignalCategory.TECH_STACK,
        )
    return None


def _detect_no_ai_assistant(data: dict) -> Optional[Signal]:
    """no_ai_assistant: Account doesn't use AI Voice Assistant specifically."""
    ai_assistants = data.get("ai_assistants", [])
    connections = data.get("connections", [])
    has_assistant_conn = any(_has_keyword(_connection_text(c), ["assistant", "voice-agent", "conversational"]) for c in connections)

    if len(ai_assistants) == 0 and not has_assistant_conn:
        return Signal(
            id="no_ai_assistant",
            name="no_ai_assistant",
            description="No AI Voice Assistant configured",
            confidence=0.9,
            strength=0.85,
            evidence="0 AI assistants found",
            cross_sell_target="AI Voice Assistant",
            category=SignalCategory.VOICE_USAGE,
        )
    return None


def _detect_no_transcription(data: dict) -> Optional[Signal]:
    """no_transcription: Account has voice calls but no recording/transcription."""
    cdr = data.get("cdr_summary", {})
    total_calls = cdr.get("total_calls", 0) if isinstance(cdr, dict) else 0
    connections = data.get("connections", [])

    has_recording = any(_has_keyword(_connection_text(c), _RECORD_KW) for c in connections)
    texml_apps = data.get("texml_apps", [])
    has_record_texml = any(_has_keyword(a.get("xml", ""), ["<record", "<dial", "recording"]) for a in texml_apps)

    if (total_calls > 0 or len(connections) >= 3) and not has_recording and not has_record_texml:
        return Signal(
            id="no_transcription",
            name="no_transcription",
            description="Voice calls present but no recording/transcription configured",
            confidence=0.8,
            strength=0.7,
            evidence=f"{total_calls} calls, no recording/transcription setup detected",
            cross_sell_target="AI Transcription & Analysis",
            category=SignalCategory.VOICE_USAGE,
        )
    return None


def _detect_no_verify(data: dict) -> Optional[Signal]:
    """no_verify: Account has voice but no Verify product."""
    verify_profiles = data.get("verify_profiles", [])
    connections = data.get("connections", [])
    has_verify_conn = any(_has_keyword(_connection_text(c), _VERIFY_KW) for c in connections)

    if len(verify_profiles) == 0 and not has_verify_conn and len(connections) >= 1:
        return Signal(
            id="no_verify",
            name="no_verify",
            description="Voice account without Verify/2FA product",
            confidence=0.75,
            strength=0.6,
            evidence="0 verify profiles, voice connections present",
            cross_sell_target="AI Verification",
            category=SignalCategory.COMPLIANCE,
        )
    return None


def _detect_has_texml_or_call_control(data: dict) -> Optional[Signal]:
    """has_texml_or_call_control: Account uses TeXML or Call Control for call flows."""
    cc_apps = data.get("call_control_apps", [])
    texml_apps = data.get("texml_apps", [])

    if len(cc_apps) > 0 or len(texml_apps) > 0:
        return Signal(
            id="has_texml_or_call_control",
            name="has_texml_or_call_control",
            description=f"Account uses {len(cc_apps)} Call Control + {len(texml_apps)} TeXML apps",
            confidence=0.95,
            strength=0.9,
            evidence=f"{len(cc_apps)} CC apps, {len(texml_apps)} TeXML apps",
            cross_sell_target="AI-Powered IVR",
            category=SignalCategory.TECH_STACK,
        )
    return None


def _detect_dtmf_ivr_detected(data: dict) -> Optional[Signal]:
    """dtmf_ivr_detected: TeXML apps with DTMF/gather patterns (non-AI IVR)."""
    texml_apps = data.get("texml_apps", [])

    for app in texml_apps:
        xml = app.get("xml", "") or ""
        name = app.get("name", "") or ""
        has_gather = "<gather" in xml.lower() or "gather" in name.lower()
        has_say = "<say" in xml.lower() or "<play" in xml.lower()
        has_no_ai = not _has_keyword(xml + " " + name, _AI_KW)

        if has_gather and has_no_ai:
            return Signal(
                id="dtmf_ivr_detected",
                name="dtmf_ivr_detected",
                description=f"DTMF-based IVR detected in TeXML app '{name}'",
                confidence=0.85,
                strength=0.8,
                evidence=f"TeXML app '{name}' uses <Gather> without AI integration",
                cross_sell_target="AI-Powered IVR",
                category=SignalCategory.TECH_STACK,
            )
    return None


def _detect_no_ai_ivr(data: dict) -> Optional[Signal]:
    """no_ai_ivr: Has IVR infrastructure but no AI IVR."""
    texml_apps = data.get("texml_apps", [])
    cc_apps = data.get("call_control_apps", [])

    has_ivr = len(texml_apps) > 0 or any(
        _has_keyword(_connection_text(c), _IVR_KW) for c in data.get("connections", [])
    )
    has_ai_ivr = any(_has_keyword(a.get("name", "") + " " + a.get("xml", ""), _AI_KW) for a in texml_apps)

    if has_ivr and not has_ai_ivr:
        return Signal(
            id="no_ai_ivr",
            name="no_ai_ivr",
            description="IVR infrastructure present but no AI-powered IVR",
            confidence=0.8,
            strength=0.75,
            evidence=f"{len(texml_apps)} TeXML app(s), none with AI integration",
            cross_sell_target="AI-Powered IVR",
            category=SignalCategory.TECH_STACK,
        )
    return None


def _detect_manual_routing_detected(data: dict) -> Optional[Signal]:
    """manual_routing_detected: Call Control without intelligent routing."""
    cc_apps = data.get("call_control_apps", [])
    connections = data.get("connections", [])

    has_cc = len(cc_apps) > 0
    has_ai_routing = any(_has_keyword(_connection_text(c), ["ai", "intelligent", "smart", "router"]) for c in connections)

    if has_cc and not has_ai_routing:
        return Signal(
            id="manual_routing_detected",
            name="manual_routing_detected",
            description="Call Control present without AI routing",
            confidence=0.75,
            strength=0.7,
            evidence=f"{len(cc_apps)} CC app(s), no AI routing patterns detected",
            cross_sell_target="AI Call Routing",
            category=SignalCategory.TECH_STACK,
        )
    return None


def _detect_fraud_risk_indicators(data: dict) -> Optional[Signal]:
    """fraud_risk_indicators: Voice account with high failure rate or suspicious patterns."""
    cdr = data.get("cdr_summary", {})
    if not isinstance(cdr, dict):
        return None

    failure_rate = cdr.get("failure_rate", 0)
    total_calls = cdr.get("total_calls", 0)

    if total_calls >= 10 and failure_rate >= 0.15:
        return Signal(
            id="fraud_risk_indicators",
            name="fraud_risk_indicators",
            description=f"Elevated failure rate ({failure_rate:.0%}) across {total_calls} calls",
            confidence=0.7,
            strength=0.6,
            evidence=f"Failure rate: {failure_rate:.0%} across {total_calls} calls",
            cross_sell_target="AI Verification",
            category=SignalCategory.COMPLIANCE,
        )
    return None


def _detect_compliance_monitoring_need(data: dict) -> Optional[Signal]:
    """compliance_monitoring_need: Voice calls without compliance monitoring."""
    cdr = data.get("cdr_summary", {})
    total_calls = cdr.get("total_calls", 0) if isinstance(cdr, dict) else 0
    connections = data.get("connections", [])
    numbers = data.get("phone_numbers", [])

    # International numbers without compliance monitoring
    intl_numbers = [n for n in numbers if n.get("country_code") and n["country_code"] != "US"]

    if (total_calls >= 20 or len(intl_numbers) >= 2) and not any(
        _has_keyword(_connection_text(c), _RECORD_KW + ["compliance", "monitor", "audit"]) for c in connections
    ):
        return Signal(
            id="compliance_monitoring_need",
            name="compliance_monitoring_need",
            description="Call volume or international presence without compliance monitoring",
            confidence=0.7,
            strength=0.6,
            evidence=f"{total_calls} calls, {len(intl_numbers)} intl numbers, no compliance monitoring",
            cross_sell_target="AI Transcription & Analysis",
            category=SignalCategory.COMPLIANCE,
        )
    return None


def _detect_quality_assurance_need(data: dict) -> Optional[Signal]:
    """quality_assurance_need: High call volume without QA tools."""
    cdr = data.get("cdr_summary", {})
    total_calls = cdr.get("total_calls", 0) if isinstance(cdr, dict) else 0
    connections = data.get("connections", [])

    has_qa = any(_has_keyword(_connection_text(c), ["quality", "qa", "monitor", "score", "evaluat"]) for c in connections)

    if total_calls >= 50 and not has_qa:
        return Signal(
            id="quality_assurance_need",
            name="quality_assurance_need",
            description=f"{total_calls} calls without quality assurance tooling",
            confidence=0.7,
            strength=0.65,
            evidence=f"{total_calls} calls, no QA tooling detected",
            cross_sell_target="AI Transcription & Analysis",
            category=SignalCategory.VOICE_USAGE,
        )
    return None


def _detect_long_queue_wait_times(data: dict) -> Optional[Signal]:
    """long_queue_wait_times: Inbound-heavy with no queue/IVR optimization."""
    cdr = data.get("cdr_summary", {})
    if not isinstance(cdr, dict):
        return None
    inbound = cdr.get("inbound_count", 0)
    outbound = cdr.get("outbound_count", 1)

    if inbound > outbound * 2 and inbound >= 20:
        texml_apps = data.get("texml_apps", [])
        has_queue = any(_has_keyword(a.get("xml", "") + " " + a.get("name", ""), ["queue", "enqueue", "wait"]) for a in texml_apps)

        if not has_queue:
            return Signal(
                id="long_queue_wait_times",
                name="long_queue_wait_times",
                description=f"Inbound-heavy ({inbound} in vs {outbound} out) without queue management",
                confidence=0.7,
                strength=0.65,
                evidence=f"Inbound/outbound ratio: {inbound/outbound:.1f}x, no queue management",
                cross_sell_target="AI Voice Assistant",
                category=SignalCategory.VOICE_USAGE,
            )
    return None


def _detect_high_abandonment_rate(data: dict) -> Optional[Signal]:
    """high_abandonment_rate: Evidence of calls being abandoned (high failure + inbound heavy)."""
    cdr = data.get("cdr_summary", {})
    if not isinstance(cdr, dict):
        return None
    failure_rate = cdr.get("failure_rate", 0)
    inbound = cdr.get("inbound_count", 0)

    if failure_rate > 0.1 and inbound >= 10:
        return Signal(
            id="high_abandonment_rate",
            name="high_abandonment_rate",
            description=f"High abandonment indicators: {failure_rate:.0%} failure rate on {inbound} inbound calls",
            confidence=0.65,
            strength=0.6,
            evidence=f"Failure rate: {failure_rate:.0%}, {inbound} inbound calls",
            cross_sell_target="AI-Powered IVR",
            category=SignalCategory.VOICE_USAGE,
        )
    return None


# ---------------------------------------------------------------------------
# Negative signals (exclude templates when these are present)
# ---------------------------------------------------------------------------

def _detect_has_ai_routing(data: dict) -> Optional[Signal]:
    """has_ai_routing: Account already uses AI for call routing."""
    connections = data.get("connections", [])
    if any(_has_keyword(_connection_text(c), ["ai", "intelligent", "smart", "router"] + _AI_KW) for c in connections):
        return Signal(
            id="has_ai_routing",
            name="has_ai_routing",
            description="Account already has AI routing capabilities",
            confidence=0.9,
            strength=0.9,
            evidence="AI routing keywords detected in connections",
            cross_sell_target="",
            category=SignalCategory.AI_USAGE,
        )
    return None


def _detect_has_ai_assistant(data: dict) -> Optional[Signal]:
    """has_ai_assistant: Account already uses AI Voice Assistant."""
    ai_assistants = data.get("ai_assistants", [])
    if len(ai_assistants) > 0:
        return Signal(
            id="has_ai_assistant",
            name="has_ai_assistant",
            description=f"Account already has {len(ai_assistants)} AI assistant(s)",
            confidence=0.95,
            strength=0.95,
            evidence=f"{len(ai_assistants)} AI assistant(s) configured",
            cross_sell_target="",
            category=SignalCategory.AI_USAGE,
        )
    return None


def _detect_has_transcription(data: dict) -> Optional[Signal]:
    """has_transcription: Account already uses recording/transcription."""
    connections = data.get("connections", [])
    if any(_has_keyword(_connection_text(c), _RECORD_KW + ["transcri"]) for c in connections):
        return Signal(
            id="has_transcription",
            name="has_transcription",
            description="Account already has recording/transcription",
            confidence=0.9,
            strength=0.9,
            evidence="Recording/transcription keywords detected in connections",
            cross_sell_target="",
            category=SignalCategory.AI_USAGE,
        )
    return None


def _detect_has_ai_ivr(data: dict) -> Optional[Signal]:
    """has_ai_ivr: Account already has AI-powered IVR."""
    texml_apps = data.get("texml_apps", [])
    if any(_has_keyword(a.get("name", "") + " " + a.get("xml", ""), _AI_KW + ["assistant", "agent"]) for a in texml_apps):
        return Signal(
            id="has_ai_ivr",
            name="has_ai_ivr",
            description="Account already has AI-powered IVR",
            confidence=0.85,
            strength=0.85,
            evidence="AI IVR patterns detected in TeXML apps",
            cross_sell_target="",
            category=SignalCategory.AI_USAGE,
        )
    return None


def _detect_has_verify(data: dict) -> Optional[Signal]:
    """has_verify: Account already uses Verify product."""
    verify_profiles = data.get("verify_profiles", [])
    if len(verify_profiles) > 0:
        return Signal(
            id="has_verify",
            name="has_verify",
            description=f"Account already has {len(verify_profiles)} verify profile(s)",
            confidence=0.95,
            strength=0.95,
            evidence=f"{len(verify_profiles)} verify profile(s) configured",
            cross_sell_target="",
            category=SignalCategory.COMPLIANCE,
        )
    return None


# ---------------------------------------------------------------------------
# Main detection pipeline
# ---------------------------------------------------------------------------

# Ordered: positive signals first, then negative/exclusion signals
_DETECTORS = [
    # --- Positive signals (trigger template matching) ---
    _detect_has_call_control,
    _detect_has_voice_product,
    _detect_has_voice_calls,
    _detect_high_call_volume,
    _detect_high_inbound_call_volume,
    _detect_no_ai_usage,
    _detect_no_ai_assistant,
    _detect_no_transcription,
    _detect_no_verify,
    _detect_has_texml_or_call_control,
    _detect_dtmf_ivr_detected,
    _detect_no_ai_ivr,
    _detect_manual_routing_detected,
    _detect_fraud_risk_indicators,
    _detect_compliance_monitoring_need,
    _detect_quality_assurance_need,
    _detect_long_queue_wait_times,
    _detect_high_abandonment_rate,
    # --- Negative signals (suppress template matching) ---
    _detect_has_ai_routing,
    _detect_has_ai_assistant,
    _detect_has_transcription,
    _detect_has_ai_ivr,
    _detect_has_verify,
]


def _is_account_dormant(data: dict) -> bool:
    """Return True if the account has no meaningful activity.
    
    An account is dormant when it has zero calls, zero messages,
    zero connections, and zero phone numbers — there's nothing to
    cross-sell into.
    """
    return (
        data.get("total_calls", 0) == 0
        and data.get("total_messages", 0) == 0
        and data.get("voice_connection_count", 0) == 0
        and data.get("messaging_profile_count", 0) == 0
        and data.get("phone_number_count", 0) == 0
    )


def detect_signals(account_data: dict[str, Any]) -> list[Signal]:
    """Run all signal detectors on account data and return non-None results.
    
    Dormant accounts (no calls, messages, connections, or numbers) are
    suppressed entirely — there's nothing to cross-sell into.
    """
    if _is_account_dormant(account_data):
        return []
    
    signals: list[Signal] = []
    for detector in _DETECTORS:
        try:
            sig = detector(account_data)
            if sig is not None:
                signals.append(sig)
        except Exception:
            # Never let one detector crash the whole pipeline
            continue
    return signals


def detect_signals_for_accounts(accounts: list[dict]) -> dict[str, list[Signal]]:
    """Run signal detection across multiple accounts."""
    results: dict[str, list[Signal]] = {}
    for acct in accounts:
        uid = acct.get("user_id", "unknown")
        results[uid] = detect_signals(acct)
    return results
