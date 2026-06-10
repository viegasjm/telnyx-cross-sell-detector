"""
Inference & Recommendation Engine for Cross-Sell Detection

Takes detected signals (Voice usage, API patterns, tech stack signals) and produces
actionable cross-sell Recommendations with reasoning chains, value quantification,
integration effort estimates, and prioritized scoring.

Focus: Voice → AI cross-sell recommendations for Telnyx product suite.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SignalStrength(str, Enum):
    """How strongly a signal indicates an opportunity."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CRITICAL = "critical"


class IntegrationEffort(str, Enum):
    """Estimated integration complexity."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class SignalCategory(str, Enum):
    """Broad signal categories."""
    VOICE_USAGE = "voice_usage"
    MESSAGING_USAGE = "messaging_usage"
    AI_USAGE = "ai_usage"
    TELEMETRY = "telemetry"
    TECH_STACK = "tech_stack"
    BILLING = "billing"
    SUPPORT = "support"
    COMPLIANCE = "compliance"


@dataclass
class Signal:
    """A detected signal that may indicate a cross-sell opportunity.

    Attributes:
        id: Unique identifier for the signal.
        name: Human-readable short name (e.g. "High Call Volume").
        description: One-sentence description of what was detected.
        confidence: 0-1 confidence that this signal is real / meaningful.
        strength: Qualitative strength indicator.
        evidence: Supporting data or metric that triggered this signal.
        cross_sell_target: The product / feature this signal points toward.
        category: Broad signal classification.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    confidence: float = 0.5
    strength: SignalStrength = SignalStrength.MODERATE
    evidence: str = ""
    cross_sell_target: str = ""
    category: SignalCategory = SignalCategory.TELEMETRY


@dataclass
class Recommendation:
    """A cross-sell recommendation produced by the inference engine.

    Attributes:
        id: Unique recommendation identifier.
        title: Human-readable recommendation title.
        target_product: Telnyx product to cross-sell (e.g. "AI Call Routing").
        target_feature: Specific feature within the product.
        reasoning_chain: Step-by-step logic from signal → inference → recommendation.
        evidence_bullets: Bullet-point evidence supporting this recommendation.
        estimated_value: Dict with dollar/month estimates and retention uplift %.
        integration_effort: Low / Medium / High.
        integration_path: Ordered steps to adopt the recommended product.
        priority: 1 (highest) – 5 (lowest).
        confidence: 0-1 confidence in this recommendation.
        demo_highlight: What makes this impressive in a live demo.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    target_product: str = ""
    target_feature: str = ""
    reasoning_chain: List[str] = field(default_factory=list)
    evidence_bullets: List[str] = field(default_factory=list)
    estimated_value: Dict = field(default_factory=dict)
    integration_effort: IntegrationEffort = IntegrationEffort.MEDIUM
    integration_path: List[str] = field(default_factory=list)
    priority: int = 3
    confidence: float = 0.5
    demo_highlight: str = ""


@dataclass
class OpportunityCluster:
    """A group of related signals that form one coherent opportunity."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    signals: List[Signal] = field(default_factory=list)
    combined_confidence: float = 0.0
    target_product: str = ""


# ---------------------------------------------------------------------------
# Value-quantification constants (simple model)
# ---------------------------------------------------------------------------

# Agent cost assumptions
AGENT_COST_PER_MINUTE = 0.30          # USD – blended agent cost
AI_SAVINGS_PCT_CLASSIFICATION = 0.40  # AI classification saves 40 % of handling
AI_SAVINGS_PCT_ROUTING = 0.35         # Smart routing saves 35 % of transfer time
AI_SAVINGS_PCT_TRANSCRIPTION = 0.50   # AI transcription eliminates 50 % manual review
AI_SAVINGS_PCT_IVR = 0.30             # AI IVR deflects 30 % of inbound calls
AI_SAVINGS_PCT_VOICEMAIL = 0.25       # AI voicemail auto-action on 25 % of messages
VERIFICATION_FRAUD_REDUCTION = 0.60    # AI verification reduces 60 % of fraud losses

# Telnyx revenue assumptions (per-customer, monthly)
TELNYX_AI_ROUTING_REVENUE = 150.0
TELNYX_AI_ASSISTANT_REVENUE = 200.0
TELNYX_AI_TRANSCRIPTION_REVENUE = 120.0
TELNYX_AI_IVR_REVENUE = 180.0
TELNYX_AI_VERIFICATION_REVENUE = 100.0

# Retention uplift per recommendation type (percentage-point increase)
RETENTION_UPLIFT = {
    "AI Call Routing": 8,
    "AI Voice Assistant": 12,
    "AI Transcription & Analysis": 10,
    "AI-Powered IVR": 9,
    "AI Verification": 7,
}

# Integration effort lookup (product → effort)
INTEGRATION_EFFORT_MAP = {
    "AI Call Routing": IntegrationEffort.LOW,
    "AI Voice Assistant": IntegrationEffort.MEDIUM,
    "AI Transcription & Analysis": IntegrationEffort.LOW,
    "AI-Powered IVR": IntegrationEffort.MEDIUM,
    "AI Verification": IntegrationEffort.LOW,
}

# Integration path templates (product → ordered steps)
INTEGRATION_PATHS = {
    "AI Call Routing": [
        "Enable AI Classification API on Telnyx portal",
        "Configure call routing rules with ML model selection",
        "Map existing call flows to AI-assisted decision tree",
        "A/B test AI routing vs. current routing on 20 % of traffic",
        "Roll out to 100 % after accuracy validation",
    ],
    "AI Voice Assistant": [
        "Provision AI Assistant add-on in Telnyx portal",
        "Define assistant persona, knowledge base, and escalation rules",
        "Integrate assistant webhook with existing Call Control flow",
        "Run pilot on after-hours / overflow calls",
        "Expand to full queue with live-agent fallback",
    ],
    "AI Transcription & Analysis": [
        "Enable Transcription API on Telnyx portal",
        "Configure real-time vs. post-call transcription preference",
        "Set up sentiment analysis and keyword extraction pipelines",
        "Build dashboards for call quality and compliance monitoring",
        "Automate alerts on negative sentiment / compliance keywords",
    ],
    "AI-Powered IVR": [
        "Enable AI IVR module in Telnyx portal",
        "Design conversational IVR flows replacing DTMF menus",
        "Map existing TeXML / Call Control routing to AI IVR intents",
        "Deploy with human-agent escalation path",
        "Iterate on intent model based on call analytics",
    ],
    "AI Verification": [
        "Enable Verify API with AI-enhanced fraud detection",
        "Configure verification channels (SMS, voice, WhatsApp)",
        "Integrate verification step into call flow or onboarding",
        "Set up fraud-score thresholds and auto-block rules",
        "Monitor verification success rates and adjust",
    ],
}


# Additional Bot Week cross-sell products beyond Voice → AI.
RETENTION_UPLIFT.update({
    "Messaging": 9,
    "Verify": 8,
    "Voice / SIP Trunking": 10,
    "Global Messaging / Local Presence": 7,
    "Observability / Support Intelligence": 6,
})
INTEGRATION_EFFORT_MAP.update({
    "Messaging": IntegrationEffort.LOW,
    "Verify": IntegrationEffort.LOW,
    "Voice / SIP Trunking": IntegrationEffort.MEDIUM,
    "Global Messaging / Local Presence": IntegrationEffort.MEDIUM,
    "Observability / Support Intelligence": IntegrationEffort.MEDIUM,
})
INTEGRATION_PATHS.update({
    "Messaging": [
        "Create or select a Messaging Profile",
        "Enable SMS/MMS-capable numbers already in inventory",
        "Configure webhooks and delivery receipt handling",
        "Run low-volume notification pilot",
        "Expand to customer notification / reminder workflows",
    ],
    "Verify": [
        "Create Verify profile",
        "Choose verification channels: SMS, voice, WhatsApp",
        "Integrate OTP verification into signup/login or transaction flow",
        "Monitor conversion and fraud reduction metrics",
    ],
    "Voice / SIP Trunking": [
        "Create SIP credential or IP-auth connection",
        "Assign existing/local numbers to voice routing",
        "Run test inbound and outbound call flows",
        "Add failover and monitoring before production rollout",
    ],
    "Global Messaging / Local Presence": [
        "Map current countries and destination mix",
        "Identify countries where local sender/number presence improves conversion",
        "Provision compliant local numbers/profiles",
        "Pilot country-specific routing and delivery monitoring",
    ],
    "Observability / Support Intelligence": [
        "Map recurring support categories to product workflows",
        "Enable delivery/call/debug observability for affected product",
        "Create account-facing health checks and alerts",
        "Review support trend after two weeks",
    ],
})


# ---------------------------------------------------------------------------
# Recommendation template definitions
# ---------------------------------------------------------------------------

@dataclass
class RecommendationTemplate:
    """Template from which concrete Recommendations are instantiated."""
    product: str
    feature: str
    title: str
    demo_highlight: str
    required_signal_categories: List[SignalCategory]
    required_signal_names: List[str]         # Signal names that trigger this template
    excluded_signal_names: List[str]         # If present, suppress this template
    reasoning_template: List[str]            # {signal_name}, {evidence} placeholders
    default_evidence_bullets: List[str]


TEMPLATES: List[RecommendationTemplate] = [
    # (a) AI Call Routing — Call Control without AI
    RecommendationTemplate(
        product="AI Call Routing",
        feature="AI-Powered Call Classification & Routing",
        title="AI Call Routing — Automate Inbound Call Classification",
        demo_highlight="Live AI re-classifies a misrouted call in <2 s and transfers to the right queue",
        required_signal_categories=[SignalCategory.VOICE_USAGE, SignalCategory.TECH_STACK],
        required_signal_names=[
            "has_call_control",
            "high_call_volume",
            "no_ai_usage",
            "manual_routing_detected",
        ],
        excluded_signal_names=["has_ai_routing"],
        reasoning_template=[
            "Signal detected: {signal_name} — {evidence}",
            "Inference: Customer is routing calls manually, incurring agent handling overhead",
            "Opportunity: AI Call Routing can classify and route calls automatically, reducing transfer time",
            "Value: At {agent_cost}/min agent cost, a 40 % reduction in handling saves an estimated ${savings}/mo",
            "Recommendation: Deploy AI Call Routing to automate classification and reduce transfer overhead",
        ],
        default_evidence_bullets=[
            "Customer uses Call Control API for inbound routing",
            "No AI services currently enabled on the account",
            "Call volume is high enough to justify automation ROI",
        ],
    ),

    # (b) AI Voice Assistant — Voice without AI assistant
    RecommendationTemplate(
        product="AI Voice Assistant",
        feature="Conversational AI Voice Agent",
        title="AI Voice Assistant — Automate Tier-1 Voice Interactions",
        demo_highlight="AI assistant handles a live caller's FAQ, books an appointment, and escalates complex issue — all in one call",
        required_signal_categories=[SignalCategory.VOICE_USAGE],
        required_signal_names=[
            "has_voice_product",
            "high_inbound_call_volume",
            "no_ai_assistant",
            "long_queue_wait_times",
        ],
        excluded_signal_names=["has_ai_assistant"],
        reasoning_template=[
            "Signal detected: {signal_name} — {evidence}",
            "Inference: Customers experience long queue wait times with no AI-assisted self-service",
            "Opportunity: AI Voice Assistant can deflect routine inquiries and handle Tier-1 tasks autonomously",
            "Value: Assistant-driven deflection reduces call volume by ~35 %, saving ${savings}/mo in agent cost",
            "Recommendation: Deploy AI Voice Assistant to automate repetitive voice interactions",
        ],
        default_evidence_bullets=[
            "Customer has active voice product but no AI assistant",
            "Queue wait times exceed threshold, indicating capacity issues",
            "High inbound call volume suggests repetitive / Tier-1 queries",
        ],
    ),

    # (c) AI Transcription & Analysis — Voice calls needing intelligence
    RecommendationTemplate(
        product="AI Transcription & Analysis",
        feature="Real-Time Transcription & Sentiment Analysis",
        title="AI Transcription & Analysis — Add Intelligence to Voice Calls",
        demo_highlight="Live transcript appears word-by-word; sentiment turns red on a frustrated caller and auto-triggers supervisor alert",
        required_signal_categories=[SignalCategory.VOICE_USAGE],
        required_signal_names=[
            "has_voice_calls",
            "no_transcription",
            "compliance_monitoring_need",
            "quality_assurance_need",
        ],
        excluded_signal_names=["has_transcription"],
        reasoning_template=[
            "Signal detected: {signal_name} — {evidence}",
            "Inference: Customer processes voice calls without transcription or analytics, limiting visibility",
            "Opportunity: AI Transcription provides searchable records, sentiment analysis, and compliance flags",
            "Value: Eliminating 50 % of manual call review saves ~${savings}/mo; compliance risk is also reduced",
            "Recommendation: Enable AI Transcription & Analysis for call intelligence and compliance automation",
        ],
        default_evidence_bullets=[
            "Customer has voice calls but no transcription or AI analysis",
            "Compliance or QA needs indicate manual review overhead",
            "Real-time transcription enables proactive issue resolution",
        ],
    ),

    # (d) AI-Powered IVR — TeXML/Call Control with manual routing
    RecommendationTemplate(
        product="AI-Powered IVR",
        feature="Conversational AI-Driven IVR",
        title="AI-Powered IVR — Replace DTMF Menus with Natural Language",
        demo_highlight="Caller speaks naturally: 'I need to change my appointment' — AI understands, confirms, and reschedules — zero button presses",
        required_signal_categories=[SignalCategory.VOICE_USAGE, SignalCategory.TECH_STACK],
        required_signal_names=[
            "has_texml_or_call_control",
            "dtmf_ivr_detected",
            "no_ai_ivr",
            "high_abandonment_rate",
        ],
        excluded_signal_names=["has_ai_ivr"],
        reasoning_template=[
            "Signal detected: {signal_name} — {evidence}",
            "Inference: Customer uses legacy DTMF IVR, contributing to caller frustration and abandonment",
            "Opportunity: AI-Powered IVR understands natural language, reducing abandonment and improving CSAT",
            "Value: 30 % call deflection saves ${savings}/mo; CSAT improvement reduces churn by ~9 pp",
            "Recommendation: Upgrade to AI-Powered IVR for natural-language call handling",
        ],
        default_evidence_bullets=[
            "Customer uses TeXML or Call Control with DTMF-based IVR",
            "No conversational AI IVR detected",
            "High abandonment rate suggests poor IVR experience",
        ],
    ),

    # (e) AI Verification — Voice without verify
    RecommendationTemplate(
        product="AI Verification",
        feature="AI-Enhanced Identity Verification",
        title="AI Verification — Strengthen Fraud Prevention on Voice Channels",
        demo_highlight="AI detects a synthetic voice attempt during verification, blocks it instantly, and logs the fraud score",
        required_signal_categories=[SignalCategory.VOICE_USAGE, SignalCategory.COMPLIANCE],
        required_signal_names=[
            "has_voice_product",
            "no_verify",
            "fraud_risk_indicators",
        ],
        excluded_signal_names=["has_verify"],
        reasoning_template=[
            "Signal detected: {signal_name} — {evidence}",
            "Inference: Customer has voice channels without AI-enhanced verification, exposing fraud risk",
            "Opportunity: AI Verification adds multi-channel identity checks and synthetic-voice detection",
            "Value: 60 % fraud reduction lowers loss exposure; verification revenue adds ${savings}/mo",
            "Recommendation: Deploy AI Verification to protect voice channels from fraud",
        ],
        default_evidence_bullets=[
            "Customer has voice product but no Verify API enabled",
            "Fraud risk indicators detected (e.g. suspicious call patterns)",
            "AI-enhanced verification reduces account takeover risk",
        ],
    ),
]

TEMPLATES.extend([
    RecommendationTemplate(
        product="Messaging",
        feature="SMS/MMS Customer Notifications",
        title="Messaging — Activate SMS/MMS on Existing Number Footprint",
        demo_highlight="Customer starts sending appointment/order notifications from existing Telnyx numbers",
        required_signal_categories=[SignalCategory.MESSAGING_USAGE],
        required_signal_names=["has_numbers_no_messaging"],
        excluded_signal_names=["has_messaging"],
        reasoning_template=[
            "Signal detected: {signal_name} — {evidence}",
            "Inference: Customer owns number inventory but is not using messaging workflows",
            "Opportunity: Messaging can add reminders, alerts, and customer notifications without a new vendor",
            "Value: Automating reminders and notifications can save ${savings}/mo in manual outreach",
            "Recommendation: Enable Messaging Profiles for existing eligible numbers",
        ],
        default_evidence_bullets=[
            "Customer has phone number inventory",
            "No messaging profile or message traffic detected",
            "Existing numbers can support customer notification workflows",
        ],
    ),
    RecommendationTemplate(
        product="Verify",
        feature="OTP / Identity Verification",
        title="Verify — Add OTP Verification to Messaging Workflows",
        demo_highlight="A login/signup OTP flow is added using the customer's existing messaging channel",
        required_signal_categories=[SignalCategory.MESSAGING_USAGE],
        required_signal_names=["has_messaging", "high_messaging_volume", "has_sms_no_verify"],
        excluded_signal_names=["has_verify"],
        reasoning_template=[
            "Signal detected: {signal_name} — {evidence}",
            "Inference: Customer already uses messaging but has no dedicated verification product",
            "Opportunity: Verify packages OTP, fraud controls, and conversion monitoring into a first-class workflow",
            "Value: Reducing custom OTP build/ops work can save ${savings}/mo and improve conversion",
            "Recommendation: Position Verify for authentication, signup, or transaction confirmation use cases",
        ],
        default_evidence_bullets=[
            "Messaging usage/configuration detected",
            "No Verify profile detected",
            "OTP and identity checks are natural adjacent workflows for messaging customers",
        ],
    ),
    RecommendationTemplate(
        product="Voice / SIP Trunking",
        feature="Programmable Voice and SIP Connectivity",
        title="Voice / SIP — Expand Messaging Customers into Voice",
        demo_highlight="Customer adds click-to-call or voice failover beside existing messaging workflows",
        required_signal_categories=[SignalCategory.MESSAGING_USAGE],
        required_signal_names=["no_voice_usage"],
        excluded_signal_names=["has_voice_product"],
        reasoning_template=[
            "Signal detected: {signal_name} — {evidence}",
            "Inference: Customer has customer-communication workflows but no voice channel on Telnyx",
            "Opportunity: Add programmable voice, SIP trunking, or failover calls for higher-touch interactions",
            "Value: Channel consolidation and voice fallback can save ${savings}/mo in vendor/tooling overhead",
            "Recommendation: Pitch Voice/SIP as the next channel adjacent to messaging",
        ],
        default_evidence_bullets=[
            "Messaging usage detected",
            "No voice/SIP usage detected",
            "Multi-channel customers retain better than single-product customers",
        ],
    ),
    RecommendationTemplate(
        product="Global Messaging / Local Presence",
        feature="Country-Specific Messaging and Local Number Presence",
        title="Global Messaging — Match International Footprint with Local Presence",
        demo_highlight="International footprint is mapped to local sender/number strategy by country",
        required_signal_categories=[SignalCategory.TECH_STACK],
        required_signal_names=["international_footprint"],
        excluded_signal_names=[],
        reasoning_template=[
            "Signal detected: {signal_name} — {evidence}",
            "Inference: Customer likely serves multiple geographies with different delivery/compliance needs",
            "Opportunity: Local presence and country-specific messaging improve trust, answer rates, and delivery",
            "Value: Better local routing/presence can save ${savings}/mo through improved conversion and fewer failed contacts",
            "Recommendation: Review global footprint and pitch local presence / global messaging package",
        ],
        default_evidence_bullets=[
            "Multiple countries detected in account footprint",
            "Local presence can improve trust and conversion",
        ],
    ),
    RecommendationTemplate(
        product="Observability / Support Intelligence",
        feature="Customer-Facing Debugging and Health Visibility",
        title="Support Intelligence — Turn Repeated Complaints into Productized Visibility",
        demo_highlight="Repeated support pain becomes an account-specific health/debug recommendation",
        required_signal_categories=[SignalCategory.SUPPORT],
        required_signal_names=["support_complaints_by_product"],
        excluded_signal_names=[],
        reasoning_template=[
            "Signal detected: {signal_name} — {evidence}",
            "Inference: Customer is repeatedly hitting support-visible friction in a product area",
            "Opportunity: Observability/debug tooling reduces support load and improves retention",
            "Value: Reducing repeat support contacts can save ${savings}/mo in operating cost and protect renewal risk",
            "Recommendation: Offer product-specific health/debug visibility or proactive support package",
        ],
        default_evidence_bullets=[
            "Repeated support complaints detected",
            "Support pain can indicate both churn risk and expansion opportunity",
        ],
    ),
])


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------


class InferenceEngine:
    """Produces cross-sell Recommendations from a set of detected Signals.

    Pipeline:
        1. Cluster related signals into OpportunityClusters
        2. Match clusters against RecommendationTemplates
        3. Instantiate Recommendations with reasoning chains
        4. Quantify value (customer savings, Telnyx revenue, retention uplift)
        5. Estimate integration effort
        6. Prioritize by composite score (confidence × value × retention_impact)
    """

    def __init__(
        self,
        agent_cost_per_minute: float = AGENT_COST_PER_MINUTE,
        monthly_call_minutes: float = 10_000.0,
        monthly_call_volume: int = 5_000,
    ) -> None:
        self.agent_cost_per_minute = agent_cost_per_minute
        self.monthly_call_minutes = monthly_call_minutes
        self.monthly_call_volume = monthly_call_volume

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_recommendations(
        self,
        signals: List[Signal],
        customer_context: Optional[Dict] = None,
    ) -> List[Recommendation]:
        """Main entry point. Returns a prioritized list of Recommendations."""
        if not signals:
            return []

        ctx = customer_context or {}
        self._apply_context(ctx)

        # Step 1 — cluster signals
        clusters = self._cluster_signals(signals)

        # Step 2 — generate recommendations per cluster
        recommendations: List[Recommendation] = []
        for cluster in clusters:
            recs = self._match_and_build(cluster, ctx)
            recommendations.extend(recs)

        # Step 3 — deduplicate (same target_product from overlapping clusters)
        recommendations = self._deduplicate(recommendations)

        # Step 4 — prioritize
        recommendations = self._prioritize(recommendations)

        return recommendations

    # ------------------------------------------------------------------
    # Step 1: Cluster related signals
    # ------------------------------------------------------------------

    def _cluster_signals(self, signals: List[Signal]) -> List[OpportunityCluster]:
        """Group signals by cross_sell_target and category affinity."""
        target_groups: Dict[str, List[Signal]] = {}

        for signal in signals:
            key = signal.cross_sell_target or signal.category.value
            target_groups.setdefault(key, []).append(signal)

        clusters: List[OpportunityCluster] = []
        for target_key, group_signals in target_groups.items():
            combined_conf = self._combine_confidences(group_signals)
            cluster = OpportunityCluster(
                name=f"Opportunity: {target_key}",
                description=f"Cluster of {len(group_signals)} signal(s) pointing to {target_key}",
                signals=group_signals,
                combined_confidence=combined_conf,
                target_product=group_signals[0].cross_sell_target or target_key,
            )
            clusters.append(cluster)

        return clusters

    @staticmethod
    def _combine_confidences(signals: List[Signal]) -> float:
        """Combine independent signal confidences using noisy-OR model.

        P(at least one is real) = 1 - ∏(1 - conf_i)
        """
        product = 1.0
        for s in signals:
            product *= (1.0 - s.confidence)
        return round(1.0 - product, 4)

    # ------------------------------------------------------------------
    # Step 2: Match clusters to templates & build recommendations
    # ------------------------------------------------------------------

    def _match_and_build(
        self,
        cluster: OpportunityCluster,
        ctx: Dict,
    ) -> List[Recommendation]:
        recommendations: List[Recommendation] = []
        signal_names = {s.name for s in cluster.signals}

        for template in TEMPLATES:
            # Check exclusion — if any excluded signal is present, skip
            if any(ex in signal_names for ex in template.excluded_signal_names):
                continue

            # Check inclusion — at least one required signal name must match,
            # or the cluster category overlaps
            category_match = any(
                s.category in template.required_signal_categories
                for s in cluster.signals
            )
            name_match = any(
                rn in signal_names for rn in template.required_signal_names
            )

            # Require either a name match OR category overlap + minimum 2 signals
            if not (name_match or (category_match and len(cluster.signals) >= 2)):
                continue

            rec = self._instantiate_recommendation(template, cluster, ctx)
            recommendations.append(rec)

        return recommendations

    def _instantiate_recommendation(
        self,
        template: RecommendationTemplate,
        cluster: OpportunityCluster,
        ctx: Dict,
    ) -> Recommendation:
        """Build a concrete Recommendation from a template + cluster."""
        # Build reasoning chain
        reasoning_chain = self._build_reasoning_chain(template, cluster)

        # Build evidence bullets
        evidence_bullets = self._build_evidence_bullets(template, cluster)

        # Quantify value
        estimated_value = self._quantify_value(template.product, cluster, ctx)

        # Integration effort & path
        effort = INTEGRATION_EFFORT_MAP.get(template.product, IntegrationEffort.MEDIUM)
        path = INTEGRATION_PATHS.get(template.product, ["Contact Telnyx for integration guidance"])

        # Confidence is the cluster's combined confidence
        confidence = cluster.combined_confidence

        # Priority will be set in _prioritize; use default
        return Recommendation(
            title=template.title,
            target_product=template.product,
            target_feature=template.feature,
            reasoning_chain=reasoning_chain,
            evidence_bullets=evidence_bullets,
            estimated_value=estimated_value,
            integration_effort=effort,
            integration_path=path,
            priority=3,
            confidence=confidence,
            demo_highlight=template.demo_highlight,
        )

    # ------------------------------------------------------------------
    # Step 2a: Reasoning chain builder
    # ------------------------------------------------------------------

    def _build_reasoning_chain(
        self,
        template: RecommendationTemplate,
        cluster: OpportunityCluster,
    ) -> List[str]:
        """Produce step-by-step reasoning: signal → inference → opportunity → recommendation."""
        chain: List[str] = []

        # Find the strongest signal for the chain header
        strongest = max(cluster.signals, key=lambda s: s.confidence)

        savings = self._estimate_customer_savings(template.product)

        for step_template in template.reasoning_template:
            step = step_template.format(
                signal_name=strongest.name,
                evidence=strongest.evidence or strongest.description,
                agent_cost=f"${self.agent_cost_per_minute:.2f}",
                savings=f"{savings:,.0f}",
            )
            chain.append(step)

        return chain

    # ------------------------------------------------------------------
    # Step 2b: Evidence bullet builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_evidence_bullets(
        template: RecommendationTemplate,
        cluster: OpportunityCluster,
    ) -> List[str]:
        bullets: List[str] = []

        # Signal-derived bullets
        for signal in cluster.signals:
            if signal.evidence:
                bullets.append(f"[{signal.name}] {signal.evidence}")
            else:
                bullets.append(f"[{signal.name}] {signal.description}")

        # Template default bullets (supplementary)
        for bullet in template.default_evidence_bullets:
            if bullet not in bullets:
                bullets.append(bullet)

        return bullets

    # ------------------------------------------------------------------
    # Step 3: Value quantification
    # ------------------------------------------------------------------

    def _quantify_value(
        self,
        product: str,
        cluster: OpportunityCluster,
        ctx: Dict,
    ) -> Dict:
        """Compute estimated monthly value for customer and Telnyx."""
        customer_savings = self._estimate_customer_savings(product)
        telnyx_revenue = self._estimate_telnyx_revenue(product, cluster)
        retention_uplift = RETENTION_UPLIFT.get(product, 5)

        return {
            "customer_savings_monthly": round(customer_savings, 2),
            "telnyx_revenue_monthly": round(telnyx_revenue, 2),
            "retention_uplift": retention_uplift,  # percentage points
        }

    def _estimate_customer_savings(self, product: str) -> float:
        """Estimate monthly $ savings for the customer."""
        minutes = self.monthly_call_minutes
        cost = self.agent_cost_per_minute

        if product == "AI Call Routing":
            return minutes * cost * AI_SAVINGS_PCT_CLASSIFICATION
        elif product == "AI Voice Assistant":
            return minutes * cost * AI_SAVINGS_PCT_ROUTING
        elif product == "AI Transcription & Analysis":
            return minutes * cost * AI_SAVINGS_PCT_TRANSCRIPTION
        elif product == "AI-Powered IVR":
            return minutes * cost * AI_SAVINGS_PCT_IVR
        elif product == "AI Verification":
            # Fraud loss reduction — assume 2 % of call volume is fraud at $5 avg loss
            return self.monthly_call_volume * 0.02 * 5.0 * VERIFICATION_FRAUD_REDUCTION
        else:
            return minutes * cost * 0.20  # default 20 % savings

    def _estimate_telnyx_revenue(
        self,
        product: str,
        cluster: OpportunityCluster,
    ) -> float:
        """Estimate monthly incremental Telnyx revenue."""
        revenue_map = {
            "AI Call Routing": TELNYX_AI_ROUTING_REVENUE,
            "AI Voice Assistant": TELNYX_AI_ASSISTANT_REVENUE,
            "AI Transcription & Analysis": TELNYX_AI_TRANSCRIPTION_REVENUE,
            "AI-Powered IVR": TELNYX_AI_IVR_REVENUE,
            "AI Verification": TELNYX_AI_VERIFICATION_REVENUE,
            "Messaging": 90.0,
            "Verify": 110.0,
            "Voice / SIP Trunking": 180.0,
            "Global Messaging / Local Presence": 140.0,
            "Observability / Support Intelligence": 80.0,
        }
        base = revenue_map.get(product, 100.0)
        # Scale by signal strength (stronger signals → higher expected adoption)
        strength_mult = self._strength_multiplier(cluster)
        return base * strength_mult

    @staticmethod
    def _strength_multiplier(cluster: OpportunityCluster) -> float:
        """Map combined confidence + signal count to a revenue multiplier."""
        conf = cluster.combined_confidence
        count = len(cluster.signals)
        # More signals and higher confidence → higher multiplier (0.5 – 1.5)
        return min(1.5, 0.5 + conf * 0.5 + count * 0.1)

    # ------------------------------------------------------------------
    # Step 4: Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(recommendations: List[Recommendation]) -> List[Recommendation]:
        """Keep only the highest-confidence recommendation per target_product."""
        seen: Dict[str, Recommendation] = {}
        for rec in recommendations:
            existing = seen.get(rec.target_product)
            if existing is None or rec.confidence > existing.confidence:
                seen[rec.target_product] = rec
        return list(seen.values())

    # ------------------------------------------------------------------
    # Step 5: Prioritization
    # ------------------------------------------------------------------

    def _prioritize(self, recommendations: List[Recommendation]) -> List[Recommendation]:
        """Assign priority scores and sort.

        Composite score = confidence × estimated_total_value × retention_impact_weight
        Priority 1 = highest score, Priority 5 = lowest.
        """
        scored: List[Tuple[Recommendation, float]] = []
        for rec in recommendations:
            value = (
                rec.estimated_value.get("customer_savings_monthly", 0)
                + rec.estimated_value.get("telnyx_revenue_monthly", 0)
            )
            retention = rec.estimated_value.get("retention_uplift", 5) / 100.0
            composite = rec.confidence * value * retention
            scored.append((rec, composite))

        # Sort descending by composite score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Assign priority 1-5
        result: List[Recommendation] = []
        for rank, (rec, score) in enumerate(scored, start=1):
            priority = min(5, max(1, rank))
            rec.priority = priority
            result.append(rec)

        return result

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------

    def _apply_context(self, ctx: Dict) -> None:
        """Override defaults with customer-specific context if provided."""
        if "agent_cost_per_minute" in ctx:
            self.agent_cost_per_minute = float(ctx["agent_cost_per_minute"])
        if "monthly_call_minutes" in ctx:
            self.monthly_call_minutes = float(ctx["monthly_call_minutes"])
        if "monthly_call_volume" in ctx:
            self.monthly_call_volume = int(ctx["monthly_call_volume"])


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def generate_recommendations(
    signals: List[Signal],
    customer_context: Optional[Dict] = None,
) -> List[Recommendation]:
    """One-call convenience wrapper around InferenceEngine."""
    engine = InferenceEngine()
    return engine.generate_recommendations(signals, customer_context)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate a set of detected signals for a customer with Call Control,
    # high call volume, no AI, manual routing, and DTMF IVR.
    demo_signals = [
        Signal(
            name="has_call_control",
            description="Customer uses Call Control API for call management",
            confidence=0.95,
            strength=SignalStrength.STRONG,
            evidence="Call Control API active with 4,200 calls/month",
            cross_sell_target="AI Call Routing",
            category=SignalCategory.TECH_STACK,
        ),
        Signal(
            name="high_call_volume",
            description="Inbound call volume exceeds automation threshold",
            confidence=0.90,
            strength=SignalStrength.STRONG,
            evidence="4,200 inbound calls/month, avg handle time 4.2 min",
            cross_sell_target="AI Call Routing",
            category=SignalCategory.VOICE_USAGE,
        ),
        Signal(
            name="no_ai_usage",
            description="No AI services enabled on this account",
            confidence=0.98,
            strength=SignalStrength.CRITICAL,
            evidence="Zero AI API calls in last 90 days",
            cross_sell_target="AI Call Routing",
            category=SignalCategory.AI_USAGE,
        ),
        Signal(
            name="manual_routing_detected",
            description="Calls routed via static rules without AI classification",
            confidence=0.85,
            strength=SignalStrength.STRONG,
            evidence="3 fixed routing rules, 0 ML-based rules in Call Control config",
            cross_sell_target="AI Call Routing",
            category=SignalCategory.TECH_STACK,
        ),
        Signal(
            name="dtmf_ivr_detected",
            description="IVR uses DTMF (touch-tone) menus only",
            confidence=0.80,
            strength=SignalStrength.MODERATE,
            evidence="IVR flow has 12 DTMF nodes, 0 NLU intents",
            cross_sell_target="AI-Powered IVR",
            category=SignalCategory.TECH_STACK,
        ),
        Signal(
            name="high_abandonment_rate",
            description="Call abandonment rate exceeds industry average",
            confidence=0.75,
            strength=SignalStrength.MODERATE,
            evidence="22 % abandonment rate (industry avg: 8 %)",
            cross_sell_target="AI-Powered IVR",
            category=SignalCategory.VOICE_USAGE,
        ),
        Signal(
            name="has_voice_product",
            description="Customer has active Telnyx voice product",
            confidence=0.99,
            strength=SignalStrength.CRITICAL,
            evidence="Voice product active since 2023-06",
            cross_sell_target="AI Voice Assistant",
            category=SignalCategory.VOICE_USAGE,
        ),
        Signal(
            name="no_ai_assistant",
            description="No AI voice assistant configured",
            confidence=0.95,
            strength=SignalStrength.STRONG,
            evidence="No assistant webhook or AI agent in call flows",
            cross_sell_target="AI Voice Assistant",
            category=SignalCategory.AI_USAGE,
        ),
        Signal(
            name="no_verify",
            description="Verify API not enabled",
            confidence=0.90,
            strength=SignalStrength.STRONG,
            evidence="Zero verification API calls in last 90 days",
            cross_sell_target="AI Verification",
            category=SignalCategory.COMPLIANCE,
        ),
        Signal(
            name="fraud_risk_indicators",
            description="Suspicious call patterns suggest potential fraud exposure",
            confidence=0.65,
            strength=SignalStrength.MODERATE,
            evidence="3 spike patterns in international inbound, 15 min avg duration",
            cross_sell_target="AI Verification",
            category=SignalCategory.COMPLIANCE,
        ),
    ]

    engine = InferenceEngine(
        agent_cost_per_minute=0.30,
        monthly_call_minutes=17_640,  # 4,200 calls × 4.2 min avg
        monthly_call_volume=4_200,
    )

    recommendations = engine.generate_recommendations(
        demo_signals,
        customer_context={"industry": "healthcare", "plan": "enterprise"},
    )

    print("=" * 80)
    print("CROSS-SELL RECOMMENDATION ENGINE — DEMO OUTPUT")
    print("=" * 80)

    for rec in recommendations:
        print(f"\n{'─' * 70}")
        print(f"  PRIORITY {rec.priority}  |  {rec.title}")
        print(f"  Product: {rec.target_product} — {rec.target_feature}")
        print(f"  Confidence: {rec.confidence:.0%}  |  Effort: {rec.integration_effort.value}")
        print(f"\n  💰 Estimated Value:")
        print(f"     Customer Savings:   ${rec.estimated_value['customer_savings_monthly']:,.2f}/mo")
        print(f"     Telnyx Revenue:     ${rec.estimated_value['telnyx_revenue_monthly']:,.2f}/mo")
        print(f"     Retention Uplift:   +{rec.estimated_value['retention_uplift']} pp")
        print(f"\n  🔗 Reasoning Chain:")
        for i, step in enumerate(rec.reasoning_chain, 1):
            print(f"     {i}. {step}")
        print(f"\n  📋 Evidence:")
        for bullet in rec.evidence_bullets:
            print(f"     • {bullet}")
        print(f"\n  🛠  Integration Path:")
        for i, step in enumerate(rec.integration_path, 1):
            print(f"     {i}. {step}")
        print(f"\n  🎯 Demo Highlight: {rec.demo_highlight}")

    print(f"\n{'=' * 80}")
    print(f"Total recommendations: {len(recommendations)}")
    print("=" * 80)
