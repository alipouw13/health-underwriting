"""
CommunicationAgent - Generate explanations for underwriters and customers

Agent Definition (from /.github/underwriting_agents.yaml):
---------------------------------------------------------
agent_id: CommunicationAgent
purpose: Generate explanations for underwriters and customers
inputs:
  decision_summary: object
outputs:
  underwriter_message: string
  customer_message: string
tools_used:
  - language-generator
evaluation_criteria:
  - clarity
  - tone
failure_modes:
  - ambiguous_language
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import Field

from data.mock.schemas import (
    UnderwritingDecision,
    RiskLevel,
    DecisionStatus,
)
from app.agents.base import (
    BaseUnderwritingAgent,
    AgentInput,
    AgentOutput,
)


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class DecisionSummary(AgentInput):
    """Summary of underwriting decision for communication generation."""
    
    decision: UnderwritingDecision = Field(..., description="The underwriting decision to explain")
    patient_name: Optional[str] = Field(None, description="Patient name for personalization (if available)")
    policy_type: str = Field(..., description="Type of policy applied for")
    coverage_amount: float = Field(..., description="Coverage amount requested")


class CommunicationInput(AgentInput):
    """Input schema for CommunicationAgent."""
    
    decision_summary: DecisionSummary = Field(..., description="Summary of decision to communicate")


class CommunicationOutput(AgentOutput):
    """Output schema for CommunicationAgent."""
    
    underwriter_message: str = Field(..., description="Technical message for underwriter review")
    customer_message: str = Field(..., description="Plain language message for customer")
    key_points: List[str] = Field(default_factory=list, description="Key points communicated")
    tone_assessment: str = Field(..., description="Assessment of message tone")
    readability_score: float = Field(..., ge=0, le=100, description="Readability score (higher = easier to read)")


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class CommunicationAgent(BaseUnderwritingAgent[CommunicationInput, CommunicationOutput]):
    """
    Generate explanations for underwriters and customers.
    
    This agent creates appropriate communications for different audiences
    based on underwriting decisions.
    
    Tools Used:
        - language-generator: Provides language generation capabilities
    
    Evaluation Criteria:
        - clarity: Messages are clear and unambiguous
        - tone: Appropriate tone for each audience
    
    Failure Modes:
        - ambiguous_language: Messages could be misinterpreted
    """
    
    agent_id = "CommunicationAgent"
    purpose = "Generate explanations for underwriters and customers"
    tools_used = ["language-generator"]
    evaluation_criteria = ["clarity", "tone"]
    failure_modes = ["ambiguous_language"]
    
    @property
    def input_type(self) -> type[CommunicationInput]:
        return CommunicationInput
    
    @property
    def output_type(self) -> type[CommunicationOutput]:
        return CommunicationOutput
    
    async def _execute(self, validated_input: CommunicationInput) -> CommunicationOutput:
        """
        Generate communications for underwriters and customers.
        
        Process:
        1. Analyze decision details
        2. Generate technical underwriter message
        3. Generate customer-friendly message
        4. Assess tone and readability
        """
        summary = validated_input.decision_summary
        decision = summary.decision
        
        # Extract key points for communication
        key_points = self._extract_key_points(decision, summary)
        
        # Generate underwriter message (technical, detailed)
        underwriter_message = self._generate_underwriter_message(decision, summary, key_points)
        
        # Generate customer message (simple, empathetic)
        customer_message = self._generate_customer_message(decision, summary, key_points)
        
        # Assess tone
        tone_assessment = self._assess_tone(decision.status, customer_message)
        
        # Calculate readability (simplified Flesch-Kincaid approximation)
        readability_score = self._calculate_readability(customer_message)
        
        return CommunicationOutput(
            agent_id=self.agent_id,
            underwriter_message=underwriter_message,
            customer_message=customer_message,
            key_points=key_points,
            tone_assessment=tone_assessment,
            readability_score=readability_score,
        )
    
    def _extract_key_points(
        self, 
        decision: UnderwritingDecision, 
        summary: DecisionSummary
    ) -> List[str]:
        """Extract key points to communicate."""
        points = []
        
        # Decision status
        status_text = {
            DecisionStatus.APPROVED: "Application approved",
            DecisionStatus.APPROVED_WITH_ADJUSTMENT: "Application approved with premium adjustment",
            DecisionStatus.REFERRED: "Manual review required",
            DecisionStatus.DECLINED: "Application requires further review",
            DecisionStatus.PENDING_INFO: "Additional information required",
        }
        points.append(status_text.get(decision.status, "Decision pending"))
        
        # Premium adjustment if applicable
        if decision.premium_adjustment and decision.status in [
            DecisionStatus.APPROVED, 
            DecisionStatus.APPROVED_WITH_ADJUSTMENT
        ]:
            if decision.premium_adjustment.adjustment_percentage == 0:
                points.append("Standard premium applies")
            elif decision.premium_adjustment.adjustment_percentage > 0:
                points.append(f"Premium adjustment of {decision.premium_adjustment.adjustment_percentage}%")
            else:
                points.append(f"Premium discount of {abs(decision.premium_adjustment.adjustment_percentage)}%")
        
        # Risk factors
        if decision.key_risk_factors:
            points.append(f"Key factors: {', '.join(decision.key_risk_factors[:3])}")
        
        # Confidence
        if decision.confidence_score >= 0.9:
            points.append("High confidence in assessment")
        elif decision.confidence_score < 0.7:
            points.append("Additional review recommended")
        
        return points
    
    def _generate_underwriter_message(
        self,
        decision: UnderwritingDecision,
        summary: DecisionSummary,
        key_points: List[str]
    ) -> str:
        """Generate technical message for underwriter."""
        lines = [
            "=" * 60,
            "UNDERWRITING DECISION SUMMARY",
            "=" * 60,
            f"Decision ID: {decision.decision_id}",
            f"Patient ID: {decision.patient_id}",
            f"Policy Type: {summary.policy_type}",
            f"Coverage Amount: ${summary.coverage_amount:,.2f}",
            "-" * 60,
            f"STATUS: {decision.status.value.upper()}",
            f"Risk Level: {decision.risk_level.value.upper()}",
            f"Confidence Score: {decision.confidence_score:.2%}",
            f"Data Quality: {decision.data_quality_level.value}",
            "-" * 60,
            "RATIONALE:",
            decision.decision_rationale,
        ]
        
        if decision.key_risk_factors:
            lines.extend([
                "-" * 60,
                "KEY RISK FACTORS:",
            ])
            for factor in decision.key_risk_factors:
                lines.append(f"  • {factor}")
        
        if decision.premium_adjustment:
            adj = decision.premium_adjustment
            lines.extend([
                "-" * 60,
                "PREMIUM CALCULATION:",
                f"  Base Premium: ${adj.base_premium_annual:,.2f}/year",
                f"  Adjustment: {adj.adjustment_percentage:+.1f}%",
                f"  Final Premium: ${adj.adjusted_premium_annual:,.2f}/year",
            ])
            
            if adj.triggered_rule_ids:
                lines.append(f"  Triggered Rules: {', '.join(adj.triggered_rule_ids)}")
        
        lines.extend([
            "-" * 60,
            "COMPLIANCE FLAGS:",
            f"  Regulatory Compliant: {'Yes' if decision.regulatory_compliant else 'NO - REVIEW REQUIRED'}",
            f"  Bias Check Passed: {'Yes' if decision.bias_check_passed else 'NO - REVIEW REQUIRED'}",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def _generate_customer_message(
        self,
        decision: UnderwritingDecision,
        summary: DecisionSummary,
        key_points: List[str]
    ) -> str:
        """Generate customer-friendly message."""
        # Personalization
        greeting = f"Dear {summary.patient_name}," if summary.patient_name else "Dear Applicant,"
        
        # Status-specific messaging
        if decision.status == DecisionStatus.APPROVED:
            status_msg = (
                f"We are pleased to inform you that your application for {summary.policy_type} "
                f"insurance with coverage of ${summary.coverage_amount:,.0f} has been approved."
            )
            closing = "Welcome to our insurance family!"
        
        elif decision.status == DecisionStatus.APPROVED_WITH_ADJUSTMENT:
            adj_pct = decision.premium_adjustment.adjustment_percentage if decision.premium_adjustment else 0
            adj_premium = decision.premium_adjustment.adjusted_premium_annual if decision.premium_adjustment else 0
            
            status_msg = (
                f"We are pleased to inform you that your application for {summary.policy_type} "
                f"insurance has been approved. Based on our review, your premium has been "
                f"adjusted to ${adj_premium:,.2f} per year."
            )
            closing = "Thank you for choosing us for your insurance needs."
        
        elif decision.status == DecisionStatus.REFERRED:
            status_msg = (
                f"Thank you for your application for {summary.policy_type} insurance. "
                f"Your application requires additional review by our underwriting team. "
                f"We will contact you within 5-7 business days with an update."
            )
            closing = "We appreciate your patience."
        
        elif decision.status == DecisionStatus.DECLINED:
            status_msg = (
                f"Thank you for your interest in {summary.policy_type} insurance. "
                f"After careful review, we are unable to offer coverage at this time. "
                f"This decision was based on the information provided in your application."
            )
            closing = (
                "You may be eligible for other products. Please contact us to discuss alternatives."
            )
        
        elif decision.status == DecisionStatus.PENDING_INFO:
            status_msg = (
                f"Thank you for your application for {summary.policy_type} insurance. "
                f"We need additional information to complete our review. "
                f"Please contact us at your earliest convenience."
            )
            closing = "We look forward to hearing from you."
        
        else:
            status_msg = "Your application is currently being processed."
            closing = "Thank you for your patience."
        
        # Build message
        message_parts = [
            greeting,
            "",
            status_msg,
            "",
        ]
        
        # Add helpful context for approved/adjusted cases
        if decision.status in [DecisionStatus.APPROVED, DecisionStatus.APPROVED_WITH_ADJUSTMENT]:
            message_parts.extend([
                "Your coverage includes:",
                f"  • Policy Type: {summary.policy_type}",
                f"  • Coverage Amount: ${summary.coverage_amount:,.0f}",
            ])
            
            if decision.premium_adjustment:
                message_parts.append(
                    f"  • Annual Premium: ${decision.premium_adjustment.adjusted_premium_annual:,.2f}"
                )
            
            message_parts.append("")
        
        message_parts.extend([
            closing,
            "",
            "If you have any questions, please don't hesitate to contact us.",
            "",
            "Sincerely,",
            "The Underwriting Team",
        ])
        
        return "\n".join(message_parts)
    
    def _assess_tone(self, status: DecisionStatus, message: str) -> str:
        """Assess the tone of the customer message."""
        positive_words = ["pleased", "welcome", "thank", "appreciate", "congratulations"]
        negative_words = ["unable", "denied", "unfortunately", "declined", "rejected"]
        neutral_words = ["review", "additional", "contact", "process"]
        
        message_lower = message.lower()
        
        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)
        
        if status in [DecisionStatus.APPROVED, DecisionStatus.APPROVED_WITH_ADJUSTMENT]:
            expected_tone = "positive"
            if positive_count > negative_count:
                return "Appropriate - positive and welcoming tone for approval"
            else:
                return "WARNING - tone may be too neutral for approval message"
        
        elif status == DecisionStatus.DECLINED:
            expected_tone = "empathetic"
            if negative_count <= 2 and "thank" in message_lower:
                return "Appropriate - empathetic and respectful tone for decline"
            else:
                return "WARNING - tone may be too harsh for decline message"
        
        else:
            return "Appropriate - neutral and informative tone for pending status"
    
    def _calculate_readability(self, text: str) -> float:
        """
        Calculate approximate readability score.
        
        Uses a simplified approach based on sentence and word length.
        Higher score = easier to read (target: 60-80 for general audience).
        """
        # Split into sentences (approximation)
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 50.0
        
        # Split into words
        words = text.split()
        
        if not words:
            return 50.0
        
        # Calculate metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simplified Flesch-like formula
        # Target: shorter sentences and words = higher score
        base_score = 100
        sentence_penalty = max(0, (avg_sentence_length - 15)) * 2  # Penalty for long sentences
        word_penalty = max(0, (avg_word_length - 5)) * 5  # Penalty for long words
        
        score = base_score - sentence_penalty - word_penalty
        
        return max(0.0, min(100.0, score))
