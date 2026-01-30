"""
BusinessRulesValidationAgent - Validate underwriting rules and regulatory constraints

Agent Definition (from /.github/underwriting_agents.yaml):
---------------------------------------------------------
agent_id: BusinessRulesValidationAgent
purpose: Validate underwriting rules and regulatory constraints
inputs:
  premium_adjustment_recommendation: object
outputs:
  approved: boolean
  rationale: string
tools_used:
  - underwriting-rules-mcp
evaluation_criteria:
  - compliance
failure_modes:
  - rule_violation
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import Field

from data.mock.schemas import PremiumAdjustment
from app.agents.base import (
    BaseUnderwritingAgent,
    AgentInput,
    AgentOutput,
)


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class BusinessRulesValidationInput(AgentInput):
    """Input schema for BusinessRulesValidationAgent."""
    
    premium_adjustment_recommendation: PremiumAdjustment = Field(
        ..., description="Premium adjustment from PolicyRiskAgent"
    )


class BusinessRulesValidationOutput(AgentOutput):
    """Output schema for BusinessRulesValidationAgent."""
    
    approved: bool = Field(..., description="Whether the recommendation is approved")
    rationale: str = Field(..., description="Explanation for approval/rejection")
    compliance_checks: List[str] = Field(default_factory=list, description="List of compliance checks performed")
    violations_found: List[str] = Field(default_factory=list, description="List of rule violations if any")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for remediation")


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class BusinessRulesValidationAgent(BaseUnderwritingAgent[BusinessRulesValidationInput, BusinessRulesValidationOutput]):
    """
    Validate underwriting rules and regulatory constraints.
    
    This agent ensures premium adjustments comply with business rules
    and regulatory requirements before final approval.
    
    Tools Used:
        - underwriting-rules-mcp: Provides regulatory and business rule checks
    
    Evaluation Criteria:
        - compliance: Decisions meet all regulatory requirements
    
    Failure Modes:
        - rule_violation: Premium adjustment violates business or regulatory rules
    """
    
    agent_id = "BusinessRulesValidationAgent"
    purpose = "Validate underwriting rules and regulatory constraints"
    tools_used = ["underwriting-rules-mcp"]
    evaluation_criteria = ["compliance"]
    failure_modes = ["rule_violation"]
    
    # Business rule thresholds
    MAX_ADJUSTMENT_PCT = 150.0  # Maximum allowed premium increase
    MIN_ADJUSTMENT_PCT = -40.0  # Maximum allowed discount
    MAX_SINGLE_FACTOR_ADJUSTMENT = 50.0  # Max adjustment from a single factor
    REQUIRED_MINIMUM_RULES = 1  # At least one rule must be evaluated
    
    @property
    def input_type(self) -> type[BusinessRulesValidationInput]:
        return BusinessRulesValidationInput
    
    @property
    def output_type(self) -> type[BusinessRulesValidationOutput]:
        return BusinessRulesValidationOutput
    
    async def _execute(self, validated_input: BusinessRulesValidationInput) -> BusinessRulesValidationOutput:
        """
        Validate premium adjustment against business and regulatory rules.
        
        Checks performed:
        1. Premium adjustment within allowed bounds
        2. Individual adjustment factors within limits
        3. Required rule evaluations completed
        4. Regulatory compliance
        5. Consistency checks
        """
        adjustment = validated_input.premium_adjustment_recommendation
        
        compliance_checks: List[str] = []
        violations: List[str] = []
        recommendations: List[str] = []
        
        # Check 1: Total adjustment bounds
        compliance_checks.append("Total adjustment bounds check")
        if adjustment.adjustment_percentage > self.MAX_ADJUSTMENT_PCT:
            violations.append(
                f"Total adjustment {adjustment.adjustment_percentage}% exceeds maximum allowed {self.MAX_ADJUSTMENT_PCT}%"
            )
            recommendations.append(f"Cap adjustment at {self.MAX_ADJUSTMENT_PCT}%")
        elif adjustment.adjustment_percentage < self.MIN_ADJUSTMENT_PCT:
            violations.append(
                f"Total adjustment {adjustment.adjustment_percentage}% exceeds maximum discount {self.MIN_ADJUSTMENT_PCT}%"
            )
            recommendations.append(f"Cap discount at {abs(self.MIN_ADJUSTMENT_PCT)}%")
        
        # Check 2: Individual factor limits
        compliance_checks.append("Individual factor limits check")
        for rule_id, factor in adjustment.adjustment_factors.items():
            if abs(factor) > self.MAX_SINGLE_FACTOR_ADJUSTMENT:
                violations.append(
                    f"Rule {rule_id} adjustment {factor}% exceeds single-factor limit of {self.MAX_SINGLE_FACTOR_ADJUSTMENT}%"
                )
                recommendations.append(f"Review rule {rule_id} adjustment methodology")
        
        # Check 3: Premium calculation consistency
        compliance_checks.append("Premium calculation consistency check")
        expected_premium = adjustment.base_premium_annual * (1 + adjustment.adjustment_percentage / 100)
        if abs(expected_premium - adjustment.adjusted_premium_annual) > 0.01:
            violations.append(
                f"Premium calculation inconsistent: expected ${expected_premium:.2f}, got ${adjustment.adjusted_premium_annual:.2f}"
            )
            recommendations.append("Recalculate adjusted premium")
        
        # Check 4: Minimum documentation
        compliance_checks.append("Rule documentation check")
        if len(adjustment.triggered_rule_ids) < self.REQUIRED_MINIMUM_RULES and adjustment.adjustment_percentage != 0:
            violations.append(
                f"Adjustment of {adjustment.adjustment_percentage}% requires at least {self.REQUIRED_MINIMUM_RULES} documented rule(s)"
            )
            recommendations.append("Ensure all adjustment factors are linked to documented rules")
        
        # Check 5: Regulatory floor/ceiling
        compliance_checks.append("Regulatory premium bounds check")
        if adjustment.adjusted_premium_annual < 0:
            violations.append("Adjusted premium cannot be negative")
            recommendations.append("Review discount calculations")
        
        # Check 6: Consistency between factors and total
        compliance_checks.append("Factor-total consistency check")
        if adjustment.adjustment_factors:
            factor_sum = sum(adjustment.adjustment_factors.values())
            if abs(factor_sum - adjustment.adjustment_percentage) > 0.01:
                # Allow this if there's a cap applied
                if adjustment.adjustment_percentage == self.MAX_ADJUSTMENT_PCT or adjustment.adjustment_percentage == self.MIN_ADJUSTMENT_PCT:
                    compliance_checks.append("Capped adjustment - factor sum difference acceptable")
                else:
                    violations.append(
                        f"Adjustment factors sum ({factor_sum}%) doesn't match total ({adjustment.adjustment_percentage}%)"
                    )
                    recommendations.append("Reconcile adjustment factors")
        
        # Determine approval
        approved = len(violations) == 0
        
        # Build rationale
        if approved:
            rationale = (
                f"Premium adjustment of {adjustment.adjustment_percentage}% approved. "
                f"All {len(compliance_checks)} compliance checks passed. "
                f"Final premium: ${adjustment.adjusted_premium_annual:.2f}/year."
            )
        else:
            rationale = (
                f"Premium adjustment of {adjustment.adjustment_percentage}% NOT approved. "
                f"Found {len(violations)} violation(s): {'; '.join(violations)}"
            )
        
        return BusinessRulesValidationOutput(
            agent_id=self.agent_id,
            approved=approved,
            rationale=rationale,
            compliance_checks=compliance_checks,
            violations_found=violations,
            recommendations=recommendations,
        )
