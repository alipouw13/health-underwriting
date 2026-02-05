"""
PolicyRiskAgent - Translate health signals into insurance risk categories

Agent Definition (from /.github/underwriting_agents.yaml):
---------------------------------------------------------
agent_id: PolicyRiskAgent
purpose: Translate health signals into insurance risk categories
inputs:
  risk_indicators: list
  policy_rules: object
outputs:
  risk_level: string
  premium_adjustment_recommendation: object
tools_used:
  - policy-rule-engine
evaluation_criteria:
  - rule_alignment
  - consistency
failure_modes:
  - conflicting_rules
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import Field

from data.mock.schemas import (
    PolicyRuleSet,
    PremiumAdjustment,
    RiskIndicator,
    RiskLevel,
)
from app.agents.base import (
    BaseUnderwritingAgent,
    AgentInput,
    AgentOutput,
)


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class PolicyRiskInput(AgentInput):
    """Input schema for PolicyRiskAgent."""
    
    risk_indicators: List[RiskIndicator] = Field(..., description="Risk indicators from HealthDataAnalysisAgent")
    policy_rules: PolicyRuleSet = Field(..., description="Policy rules from policy-rule-engine")


class PolicyRiskOutput(AgentOutput):
    """Output schema for PolicyRiskAgent.
    
    This agent produces the final underwriting decision by evaluating
    risk indicators against the JSON policy manual.
    """
    
    risk_level: RiskLevel = Field(..., description="Overall risk classification")
    premium_adjustment_recommendation: PremiumAdjustment = Field(
        ..., description="Recommended premium adjustment based on risk"
    )
    triggered_rules: List[str] = Field(default_factory=list, description="List of rule IDs that were triggered")
    rule_evaluation_log: List[str] = Field(default_factory=list, description="Detailed log of rule evaluations")
    approved: bool = Field(default=True, description="Whether the application is approved")
    decision: str = Field(default="approved", description="Final decision: approved, approved_with_adjustment, declined, or referred")
    rationale: str = Field(default="", description="Explanation for the decision")
    referral_required: bool = Field(default=False, description="Whether manual review is required")


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class PolicyRiskAgent(BaseUnderwritingAgent[PolicyRiskInput, PolicyRiskOutput]):
    """
    Translate health signals into insurance risk categories.
    
    This agent evaluates risk indicators against policy rules to determine
    the overall risk level and premium adjustments.
    
    Tools Used:
        - policy-rule-engine: Provides policy rules and thresholds
    
    Evaluation Criteria:
        - rule_alignment: Risk assessment aligns with policy rules
        - consistency: Same inputs produce consistent outputs
    
    Failure Modes:
        - conflicting_rules: Multiple rules give contradictory guidance
    """
    
    agent_id = "PolicyRiskAgent"
    purpose = "Translate health signals into insurance risk categories"
    tools_used = ["policy-rule-engine"]
    evaluation_criteria = ["rule_alignment", "consistency"]
    failure_modes = ["conflicting_rules"]
    
    # Default base premium for calculations (configurable)
    DEFAULT_BASE_PREMIUM = 1200.00
    
    # Risk level weights for aggregation
    RISK_WEIGHTS = {
        RiskLevel.LOW: 0,
        RiskLevel.MODERATE: 10,
        RiskLevel.HIGH: 25,
        RiskLevel.VERY_HIGH: 50,
        RiskLevel.DECLINE: 100,
    }
    
    @property
    def input_type(self) -> type[PolicyRiskInput]:
        return PolicyRiskInput
    
    @property
    def output_type(self) -> type[PolicyRiskOutput]:
        return PolicyRiskOutput
    
    async def _execute(self, validated_input: PolicyRiskInput) -> PolicyRiskOutput:
        """
        Evaluate risk indicators against policy rules.
        
        Process:
        1. Match risk indicators to relevant policy rules
        2. Evaluate each rule and record triggered rules
        3. Aggregate risk levels per policy's aggregation method
        4. Calculate premium adjustment
        5. Check for conflicting rules
        """
        risk_indicators = validated_input.risk_indicators
        policy_rules = validated_input.policy_rules
        
        evaluation_log: List[str] = []
        triggered_rules: List[str] = []
        adjustment_factors: Dict[str, float] = {}
        
        evaluation_log.append(f"Evaluating {len(risk_indicators)} risk indicators against {len(policy_rules.rules)} policy rules")
        evaluation_log.append(f"Policy: {policy_rules.rule_set_name} (v{policy_rules.version})")
        
        # Evaluate each rule against matching indicators
        for rule in policy_rules.rules:
            matching_indicators = self._find_matching_indicators(rule, risk_indicators)
            
            if not matching_indicators:
                evaluation_log.append(f"Rule {rule.rule_id}: No matching indicators (category: {rule.category})")
                continue
            
            # Check if rule triggers based on indicator risk levels
            is_triggered, trigger_reason = self._evaluate_rule(rule, matching_indicators)
            
            if is_triggered:
                triggered_rules.append(rule.rule_id)
                
                if rule.premium_adjustment_pct is not None:
                    adjustment_factors[rule.rule_id] = rule.premium_adjustment_pct
                
                evaluation_log.append(
                    f"Rule {rule.rule_id} TRIGGERED: {rule.rule_name} - {trigger_reason} "
                    f"(adjustment: {rule.premium_adjustment_pct or 0}%)"
                )
            else:
                evaluation_log.append(f"Rule {rule.rule_id}: Not triggered - {trigger_reason}")
        
        # Aggregate overall risk level
        overall_risk = self._aggregate_risk_level(
            risk_indicators, 
            policy_rules.risk_aggregation_method,
            evaluation_log
        )
        
        # Calculate premium adjustment
        total_adjustment = self._calculate_total_adjustment(
            adjustment_factors,
            policy_rules.max_premium_adjustment_pct,
            evaluation_log
        )
        
        # Build premium adjustment recommendation
        base_premium = self.DEFAULT_BASE_PREMIUM
        adjusted_premium = base_premium * (1 + total_adjustment / 100)
        
        premium_adjustment = PremiumAdjustment(
            base_premium_annual=base_premium,
            adjustment_percentage=total_adjustment,
            adjusted_premium_annual=round(adjusted_premium, 2),
            adjustment_factors=adjustment_factors,
            triggered_rule_ids=triggered_rules,
        )
        
        evaluation_log.append(f"Final risk level: {overall_risk.value}")
        evaluation_log.append(f"Total premium adjustment: {total_adjustment}% (${adjusted_premium:.2f}/year)")
        
        # Check for conflicting rules
        conflicts = self._detect_rule_conflicts(triggered_rules, policy_rules)
        if conflicts:
            evaluation_log.append(f"WARNING: Potential rule conflicts detected: {conflicts}")
        
        return PolicyRiskOutput(
            agent_id=self.agent_id,
            risk_level=overall_risk,
            premium_adjustment_recommendation=premium_adjustment,
            triggered_rules=triggered_rules,
            rule_evaluation_log=evaluation_log,
        )
    
    def _find_matching_indicators(
        self, 
        rule, 
        indicators: List[RiskIndicator]
    ) -> List[RiskIndicator]:
        """Find risk indicators that match a rule's category."""
        if rule.category == "combined":
            # Combined rules match all indicators
            return indicators
        
        # Match by category
        return [ind for ind in indicators if ind.category == rule.category]
    
    def _evaluate_rule(
        self, 
        rule, 
        matching_indicators: List[RiskIndicator]
    ) -> tuple[bool, str]:
        """
        Evaluate whether a rule is triggered by matching indicators.
        
        Returns:
            Tuple of (is_triggered, reason)
        """
        if not matching_indicators:
            return False, "No matching indicators"
        
        # Check if any indicator meets or exceeds the rule's risk impact
        for indicator in matching_indicators:
            indicator_weight = self.RISK_WEIGHTS.get(indicator.risk_level, 0)
            rule_weight = self.RISK_WEIGHTS.get(rule.risk_impact, 0)
            
            if indicator_weight >= rule_weight:
                return True, f"Indicator {indicator.indicator_name} at {indicator.risk_level.value} meets threshold"
        
        # Check rule thresholds if specified
        for threshold in rule.thresholds:
            for indicator in matching_indicators:
                if indicator.metric_value is not None:
                    exceeds = self._check_threshold(indicator.metric_value, threshold)
                    if exceeds:
                        return True, f"{indicator.indicator_name} exceeds {threshold.metric_name} threshold"
        
        return False, "Indicators below threshold"
    
    def _check_threshold(self, value: float, threshold) -> bool:
        """Check if a metric value exceeds threshold for risk."""
        direction = threshold.direction
        
        if direction == "lower_is_better":
            # Higher values are worse
            if threshold.high_risk_max is not None and value > threshold.high_risk_max:
                return True
            if threshold.moderate_risk_max is not None and value > threshold.moderate_risk_max:
                return True
        else:
            # Higher values are better (lower is worse)
            if threshold.low_risk_max is not None and value < threshold.low_risk_max:
                return True
        
        return False
    
    def _aggregate_risk_level(
        self, 
        indicators: List[RiskIndicator],
        method: str,
        log: List[str]
    ) -> RiskLevel:
        """Aggregate individual risk levels into overall risk."""
        if not indicators:
            log.append("No indicators to aggregate - defaulting to MODERATE")
            return RiskLevel.MODERATE
        
        risk_levels = [ind.risk_level for ind in indicators]
        
        if method == "highest":
            # Take the highest (worst) risk level
            max_weight = max(self.RISK_WEIGHTS[level] for level in risk_levels)
            for level, weight in self.RISK_WEIGHTS.items():
                if weight == max_weight:
                    log.append(f"Aggregation (highest): {level.value}")
                    return level
        
        elif method == "average":
            # Average the weights and map to nearest level
            avg_weight = sum(self.RISK_WEIGHTS[level] for level in risk_levels) / len(risk_levels)
            
            # Map to nearest risk level
            if avg_weight >= 75:
                result = RiskLevel.DECLINE
            elif avg_weight >= 40:
                result = RiskLevel.VERY_HIGH
            elif avg_weight >= 20:
                result = RiskLevel.HIGH
            elif avg_weight >= 5:
                result = RiskLevel.MODERATE
            else:
                result = RiskLevel.LOW
            
            log.append(f"Aggregation (average): {result.value} (avg weight: {avg_weight:.1f})")
            return result
        
        elif method == "weighted":
            # Weight by confidence
            total_weight = 0
            total_confidence = 0
            
            for ind in indicators:
                weight = self.RISK_WEIGHTS[ind.risk_level] * ind.confidence
                total_weight += weight
                total_confidence += ind.confidence
            
            if total_confidence > 0:
                weighted_avg = total_weight / total_confidence
            else:
                weighted_avg = 0
            
            # Map to risk level
            if weighted_avg >= 75:
                result = RiskLevel.DECLINE
            elif weighted_avg >= 40:
                result = RiskLevel.VERY_HIGH
            elif weighted_avg >= 20:
                result = RiskLevel.HIGH
            elif weighted_avg >= 5:
                result = RiskLevel.MODERATE
            else:
                result = RiskLevel.LOW
            
            log.append(f"Aggregation (weighted): {result.value} (weighted avg: {weighted_avg:.1f})")
            return result
        
        # Default to highest
        log.append(f"Unknown aggregation method '{method}', using highest")
        return max(risk_levels, key=lambda l: self.RISK_WEIGHTS[l])
    
    def _calculate_total_adjustment(
        self, 
        adjustment_factors: Dict[str, float],
        max_adjustment: float,
        log: List[str]
    ) -> float:
        """Calculate total premium adjustment from triggered rules."""
        if not adjustment_factors:
            log.append("No adjustment factors - 0% adjustment")
            return 0.0
        
        # Sum all adjustments
        total = sum(adjustment_factors.values())
        
        # Cap at maximum
        if total > max_adjustment:
            log.append(f"Total adjustment {total}% capped at maximum {max_adjustment}%")
            total = max_adjustment
        elif total < -50:  # Don't allow more than 50% discount
            log.append(f"Total adjustment {total}% capped at minimum -50%")
            total = -50.0
        
        return round(total, 2)
    
    def _detect_rule_conflicts(
        self, 
        triggered_rules: List[str],
        policy_rules: PolicyRuleSet
    ) -> List[str]:
        """Detect potentially conflicting triggered rules."""
        conflicts = []
        
        # Simple conflict detection: rules with opposite adjustments
        rule_map = {r.rule_id: r for r in policy_rules.rules}
        
        positive_rules = []
        negative_rules = []
        
        for rule_id in triggered_rules:
            rule = rule_map.get(rule_id)
            if rule and rule.premium_adjustment_pct is not None:
                if rule.premium_adjustment_pct > 0:
                    positive_rules.append(rule_id)
                elif rule.premium_adjustment_pct < 0:
                    negative_rules.append(rule_id)
        
        # Having both positive and negative adjustments in same category may indicate conflict
        if positive_rules and negative_rules:
            conflicts.append(f"Mixed adjustments: +[{','.join(positive_rules)}] vs -[{','.join(negative_rules)}]")
        
        return conflicts
