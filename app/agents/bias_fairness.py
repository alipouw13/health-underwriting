"""
BiasAndFairnessAgent - Detect bias or sensitive-attribute leakage

Agent Definition (from /.github/underwriting_agents.yaml):
---------------------------------------------------------
agent_id: BiasAndFairnessAgent
purpose: Detect bias or sensitive-attribute leakage
inputs:
  decision_context: object
outputs:
  bias_flags: list
  mitigation_notes: string
tools_used:
  - fairness-checker
evaluation_criteria:
  - fairness
failure_modes:
  - bias_detected
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import Field

from data.mock.schemas import BiasFlag, RiskLevel
from app.agents.base import (
    BaseUnderwritingAgent,
    AgentInput,
    AgentOutput,
)


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class DecisionContext(AgentInput):
    """Context for bias/fairness analysis."""
    
    patient_age: int = Field(..., ge=18, le=120, description="Patient age")
    patient_sex: str = Field(..., description="Patient biological sex")
    patient_region: Optional[str] = Field(None, description="Patient geographic region")
    risk_level: RiskLevel = Field(..., description="Assessed risk level")
    premium_adjustment_pct: float = Field(..., description="Premium adjustment percentage")
    triggered_rules: List[str] = Field(default_factory=list, description="Rules that contributed to decision")
    health_metrics_used: List[str] = Field(default_factory=list, description="Health metrics that influenced decision")


class BiasAndFairnessInput(AgentInput):
    """Input schema for BiasAndFairnessAgent."""
    
    decision_context: DecisionContext = Field(..., description="Context of the underwriting decision")


class BiasAndFairnessOutput(AgentOutput):
    """Output schema for BiasAndFairnessAgent."""
    
    bias_flags: List[BiasFlag] = Field(default_factory=list, description="Bias concerns identified")
    mitigation_notes: str = Field(..., description="Notes on bias mitigation")
    fairness_score: float = Field(..., ge=0, le=1, description="Overall fairness score (1=no concerns)")
    sensitive_attributes_check: Dict[str, str] = Field(
        default_factory=dict, description="Check results for each sensitive attribute"
    )
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for bias mitigation")


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class BiasAndFairnessAgent(BaseUnderwritingAgent[BiasAndFairnessInput, BiasAndFairnessOutput]):
    """
    Detect bias or sensitive-attribute leakage in underwriting decisions.
    
    This agent analyzes decision context to identify potential discrimination
    based on protected attributes like age, sex, or geographic location.
    
    Tools Used:
        - fairness-checker: Provides bias detection capabilities
    
    Evaluation Criteria:
        - fairness: Decisions do not discriminate based on protected attributes
    
    Failure Modes:
        - bias_detected: Decision appears to be influenced by protected attributes
    """
    
    agent_id = "BiasAndFairnessAgent"
    purpose = "Detect bias or sensitive-attribute leakage"
    tools_used = ["fairness-checker"]
    evaluation_criteria = ["fairness"]
    failure_modes = ["bias_detected"]
    
    # Age discrimination thresholds
    # Premium increases purely based on age above these thresholds may indicate bias
    AGE_PREMIUM_THRESHOLD_40 = 15.0  # Max reasonable increase at 40
    AGE_PREMIUM_THRESHOLD_50 = 25.0  # Max at 50
    AGE_PREMIUM_THRESHOLD_60 = 40.0  # Max at 60
    AGE_PREMIUM_THRESHOLD_70 = 60.0  # Max at 70
    
    # High-risk regions that should not be penalized differently
    # (geographic discrimination check)
    PROTECTED_REGIONS = {"rural", "low_income", "minority_majority"}
    
    @property
    def input_type(self) -> type[BiasAndFairnessInput]:
        return BiasAndFairnessInput
    
    @property
    def output_type(self) -> type[BiasAndFairnessOutput]:
        return BiasAndFairnessOutput
    
    async def _execute(self, validated_input: BiasAndFairnessInput) -> BiasAndFairnessOutput:
        """
        Analyze decision context for potential bias.
        
        Checks performed:
        1. Age discrimination - excessive penalties for older applicants
        2. Sex bias - differential treatment without actuarial justification
        3. Geographic bias - penalties based on region
        4. Proxy discrimination - neutral factors that correlate with protected attributes
        """
        context = validated_input.decision_context
        
        bias_flags: List[BiasFlag] = []
        sensitive_checks: Dict[str, str] = {}
        recommendations: List[str] = []
        flag_count = 0
        
        # Initialize fairness score at maximum
        fairness_score = 1.0
        
        # Check 1: Age discrimination
        age_check_result, age_flags, age_penalty = self._check_age_bias(
            context.patient_age,
            context.premium_adjustment_pct,
            context.triggered_rules,
            flag_count
        )
        sensitive_checks["age"] = age_check_result
        bias_flags.extend(age_flags)
        fairness_score -= age_penalty
        flag_count += len(age_flags)
        
        if age_flags:
            recommendations.append("Review age-related adjustment factors for actuarial justification")
        
        # Check 2: Sex bias
        sex_check_result, sex_flags, sex_penalty = self._check_sex_bias(
            context.patient_sex,
            context.premium_adjustment_pct,
            context.health_metrics_used,
            flag_count
        )
        sensitive_checks["sex"] = sex_check_result
        bias_flags.extend(sex_flags)
        fairness_score -= sex_penalty
        flag_count += len(sex_flags)
        
        if sex_flags:
            recommendations.append("Ensure sex-based factors have clear actuarial basis")
        
        # Check 3: Geographic bias
        if context.patient_region:
            geo_check_result, geo_flags, geo_penalty = self._check_geographic_bias(
                context.patient_region,
                context.premium_adjustment_pct,
                context.triggered_rules,
                flag_count
            )
            sensitive_checks["geographic"] = geo_check_result
            bias_flags.extend(geo_flags)
            fairness_score -= geo_penalty
            flag_count += len(geo_flags)
            
            if geo_flags:
                recommendations.append("Review geographic factors for disparate impact")
        else:
            sensitive_checks["geographic"] = "Not checked - no region provided"
        
        # Check 4: Proxy discrimination
        proxy_check_result, proxy_flags, proxy_penalty = self._check_proxy_discrimination(
            context.health_metrics_used,
            context.triggered_rules,
            flag_count
        )
        sensitive_checks["proxy_discrimination"] = proxy_check_result
        bias_flags.extend(proxy_flags)
        fairness_score -= proxy_penalty
        flag_count += len(proxy_flags)
        
        if proxy_flags:
            recommendations.append("Investigate potential proxy discrimination through neutral factors")
        
        # Ensure fairness score stays in bounds
        fairness_score = max(0.0, min(1.0, fairness_score))
        
        # Generate mitigation notes
        mitigation_notes = self._generate_mitigation_notes(bias_flags, fairness_score)
        
        # Add general recommendations if none specific
        if not recommendations:
            if fairness_score >= 0.9:
                recommendations.append("No significant fairness concerns identified")
            else:
                recommendations.append("Review flagged areas for compliance with fair lending laws")
        
        return BiasAndFairnessOutput(
            agent_id=self.agent_id,
            bias_flags=bias_flags,
            mitigation_notes=mitigation_notes,
            fairness_score=round(fairness_score, 3),
            sensitive_attributes_check=sensitive_checks,
            recommendations=recommendations,
        )
    
    def _check_age_bias(
        self,
        age: int,
        adjustment_pct: float,
        triggered_rules: List[str],
        start_flag_id: int
    ) -> tuple[str, List[BiasFlag], float]:
        """Check for age-based discrimination."""
        flags: List[BiasFlag] = []
        penalty = 0.0
        flag_id = start_flag_id
        
        # Determine age-appropriate threshold
        if age < 40:
            threshold = self.AGE_PREMIUM_THRESHOLD_40
        elif age < 50:
            threshold = self.AGE_PREMIUM_THRESHOLD_50
        elif age < 60:
            threshold = self.AGE_PREMIUM_THRESHOLD_60
        else:
            threshold = self.AGE_PREMIUM_THRESHOLD_70
        
        # Check if adjustment seems excessive for age bracket
        # Note: This is a simplified check - real implementation would need actuarial tables
        age_related_rules = [r for r in triggered_rules if "age" in r.lower()]
        
        if adjustment_pct > threshold and not age_related_rules:
            # High adjustment without explicit age rules might indicate implicit bias
            severity = "medium" if adjustment_pct <= threshold * 1.5 else "high"
            flags.append(BiasFlag(
                flag_id=f"BF-{flag_id + 1:03d}",
                bias_type="age_discrimination",
                severity=severity,
                description=f"High premium adjustment ({adjustment_pct}%) for age {age} without explicit age-based rules",
                mitigation_applied=False,
                mitigation_notes=None,
                blocks_decision=severity == "high",
            ))
            penalty = 0.15 if severity == "medium" else 0.25
            flag_id += 1
            check_result = f"CONCERN: Adjustment of {adjustment_pct}% may be excessive for age {age}"
        elif age >= 65 and adjustment_pct > threshold:
            # Additional scrutiny for seniors
            flags.append(BiasFlag(
                flag_id=f"BF-{flag_id + 1:03d}",
                bias_type="age_discrimination",
                severity="low",
                description=f"Senior applicant (age {age}) with {adjustment_pct}% adjustment - verify compliance with age discrimination laws",
                mitigation_applied=False,
                mitigation_notes=None,
                blocks_decision=False,
            ))
            penalty = 0.05
            flag_id += 1
            check_result = f"INFO: Senior applicant flagged for additional review"
        else:
            check_result = f"PASS: Adjustment of {adjustment_pct}% within expected range for age {age}"
        
        return check_result, flags, penalty
    
    def _check_sex_bias(
        self,
        sex: str,
        adjustment_pct: float,
        health_metrics: List[str],
        start_flag_id: int
    ) -> tuple[str, List[BiasFlag], float]:
        """Check for sex-based discrimination."""
        flags: List[BiasFlag] = []
        penalty = 0.0
        flag_id = start_flag_id
        
        # Check for sex-specific metrics that might not be justified
        sex_specific_metrics = ["pregnancy", "reproductive", "hormonal"]
        used_sex_metrics = [m for m in health_metrics if any(s in m.lower() for s in sex_specific_metrics)]
        
        if used_sex_metrics:
            flags.append(BiasFlag(
                flag_id=f"BF-{flag_id + 1:03d}",
                bias_type="gender_bias",
                severity="low",
                description=f"Sex-specific health metrics used: {used_sex_metrics}. Verify actuarial justification.",
                mitigation_applied=False,
                mitigation_notes=None,
                blocks_decision=False,
            ))
            penalty = 0.05
            flag_id += 1
            check_result = f"INFO: Sex-specific metrics in use - requires documentation"
        else:
            check_result = f"PASS: No obvious sex-based discrimination detected"
        
        return check_result, flags, penalty
    
    def _check_geographic_bias(
        self,
        region: str,
        adjustment_pct: float,
        triggered_rules: List[str],
        start_flag_id: int
    ) -> tuple[str, List[BiasFlag], float]:
        """Check for geographic discrimination."""
        flags: List[BiasFlag] = []
        penalty = 0.0
        flag_id = start_flag_id
        
        region_lower = region.lower()
        
        # Check if region matches protected categories
        is_protected = any(p in region_lower for p in self.PROTECTED_REGIONS)
        geographic_rules = [r for r in triggered_rules if "geographic" in r.lower() or "region" in r.lower()]
        
        if is_protected and geographic_rules:
            flags.append(BiasFlag(
                flag_id=f"BF-{flag_id + 1:03d}",
                bias_type="geographic_bias",
                severity="high",
                description=f"Geographic rules triggered for protected region type: {region}",
                mitigation_applied=False,
                mitigation_notes=None,
                blocks_decision=True,
            ))
            penalty = 0.3
            flag_id += 1
            check_result = f"CONCERN: Protected region with geographic adjustments"
        elif geographic_rules and adjustment_pct > 20:
            flags.append(BiasFlag(
                flag_id=f"BF-{flag_id + 1:03d}",
                bias_type="geographic_bias",
                severity="medium",
                description=f"Significant adjustment ({adjustment_pct}%) with geographic rules for region: {region}",
                mitigation_applied=False,
                mitigation_notes=None,
                blocks_decision=False,
            ))
            penalty = 0.15
            flag_id += 1
            check_result = f"WARNING: Geographic factors contributing to {adjustment_pct}% adjustment"
        else:
            check_result = f"PASS: No geographic discrimination detected for {region}"
        
        return check_result, flags, penalty
    
    def _check_proxy_discrimination(
        self,
        health_metrics: List[str],
        triggered_rules: List[str],
        start_flag_id: int
    ) -> tuple[str, List[BiasFlag], float]:
        """Check for proxy discrimination through seemingly neutral factors."""
        flags: List[BiasFlag] = []
        penalty = 0.0
        flag_id = start_flag_id
        
        # Known proxies for protected classes
        # (These neutral-seeming factors can correlate with protected attributes)
        proxy_metrics = {
            "neighborhood_walkability": "socioeconomic status",
            "gym_membership": "socioeconomic status",
            "device_type": "socioeconomic status",
            "occupation_activity": "social class",
        }
        
        detected_proxies = []
        for metric in health_metrics:
            metric_lower = metric.lower()
            for proxy, protected_class in proxy_metrics.items():
                if proxy in metric_lower:
                    detected_proxies.append((metric, protected_class))
        
        if detected_proxies:
            proxy_desc = ", ".join([f"{m} (proxy for {p})" for m, p in detected_proxies])
            flags.append(BiasFlag(
                flag_id=f"BF-{flag_id + 1:03d}",
                bias_type="proxy_discrimination",
                severity="medium",
                description=f"Potential proxy discrimination detected: {proxy_desc}",
                mitigation_applied=False,
                mitigation_notes=None,
                blocks_decision=False,
            ))
            penalty = 0.1 * len(detected_proxies)
            flag_id += 1
            check_result = f"WARNING: {len(detected_proxies)} potential proxy factors detected"
        else:
            check_result = "PASS: No obvious proxy discrimination detected"
        
        return check_result, flags, penalty
    
    def _generate_mitigation_notes(
        self, 
        flags: List[BiasFlag], 
        fairness_score: float
    ) -> str:
        """Generate mitigation notes based on detected bias flags."""
        if not flags:
            return "No bias concerns identified. Decision appears fair and compliant."
        
        blocking_flags = [f for f in flags if f.blocks_decision]
        warning_flags = [f for f in flags if not f.blocks_decision]
        
        notes_parts = []
        
        if blocking_flags:
            notes_parts.append(
                f"CRITICAL: {len(blocking_flags)} bias concern(s) require resolution before proceeding. "
                f"Types: {', '.join(set(f.bias_type for f in blocking_flags))}"
            )
        
        if warning_flags:
            notes_parts.append(
                f"Advisory: {len(warning_flags)} potential concern(s) flagged for review. "
                f"Types: {', '.join(set(f.bias_type for f in warning_flags))}"
            )
        
        notes_parts.append(f"Overall fairness score: {fairness_score:.0%}")
        
        if fairness_score < 0.7:
            notes_parts.append(
                "Recommend human review before finalizing underwriting decision."
            )
        
        return " ".join(notes_parts)
