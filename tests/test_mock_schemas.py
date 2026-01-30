"""
Unit Tests for Mock Data Schema Integrity

These tests validate that:
1. All mock schemas are properly defined and can be instantiated
2. Fixtures produce valid data matching their schemas
3. Agent input/output mappings are correct
4. Data constraints are enforced

Run with: uv run pytest tests/test_mock_schemas.py -v
"""

from __future__ import annotations

from datetime import date, datetime
from typing import get_type_hints

import pytest

# Import all schemas
from data.mock.schemas import (
    # Enums
    RiskLevel,
    DataQualityLevel,
    DecisionStatus,
    
    # Health Metrics
    ActivityMetrics,
    HeartRateMetrics,
    SleepMetrics,
    HealthTrends,
    HealthMetrics,
    
    # Patient Profiles
    PatientDemographics,
    MedicalHistory,
    PatientProfile,
    
    # Policy Rules
    RiskThreshold,
    PolicyRule,
    PolicyRuleSet,
    
    # Premium Outcomes
    PremiumAdjustment,
    UnderwritingDecision,
    PremiumOutcome,
    
    # Agent I/O
    RiskIndicator,
    QualityFlag,
    BiasFlag,
)

# Import all fixtures
from data.mock.fixtures import (
    get_healthy_patient_metrics,
    get_moderate_risk_metrics,
    get_high_risk_metrics,
    get_incomplete_metrics,
    get_sample_patient_profiles,
    get_patient_by_id,
    get_standard_policy_rules,
    get_policy_rule_by_id,
    get_expected_outcomes,
)


# =============================================================================
# ENUM VALIDATION TESTS
# =============================================================================

class TestEnumSchemas:
    """Test enum definitions are complete and valid."""
    
    def test_risk_level_values(self):
        """RiskLevel enum has all expected values."""
        expected = {"low", "moderate", "high", "very_high", "decline"}
        actual = {level.value for level in RiskLevel}
        assert actual == expected, f"RiskLevel missing values: {expected - actual}"
    
    def test_data_quality_level_values(self):
        """DataQualityLevel enum has all expected values."""
        expected = {"excellent", "good", "fair", "poor", "insufficient"}
        actual = {level.value for level in DataQualityLevel}
        assert actual == expected
    
    def test_decision_status_values(self):
        """DecisionStatus enum has all expected values."""
        expected = {"approved", "approved_with_adjustment", "referred", "declined", "pending_info"}
        actual = {status.value for status in DecisionStatus}
        assert actual == expected


# =============================================================================
# HEALTH METRICS SCHEMA TESTS
# Validates schemas consumed by: HealthDataAnalysisAgent, DataQualityConfidenceAgent
# Simulates: medical-mcp-simulator MCP server
# =============================================================================

class TestHealthMetricsSchemas:
    """Test health metrics schemas match agent input requirements."""
    
    def test_activity_metrics_schema_valid(self):
        """ActivityMetrics can be instantiated with valid data."""
        metrics = ActivityMetrics(
            daily_steps_avg=8000,
            daily_active_minutes_avg=45,
            days_with_data=85,
            measurement_period_days=90,
        )
        assert metrics.daily_steps_avg == 8000
        assert metrics.days_with_data == 85
    
    def test_activity_metrics_constraints(self):
        """ActivityMetrics enforces value constraints."""
        # Steps cannot be negative
        with pytest.raises(ValueError):
            ActivityMetrics(
                daily_steps_avg=-100,
                days_with_data=30,
                measurement_period_days=30,
            )
        
        # Active minutes cannot exceed 24 hours
        with pytest.raises(ValueError):
            ActivityMetrics(
                daily_active_minutes_avg=1500,  # > 1440 minutes
                days_with_data=30,
                measurement_period_days=30,
            )
    
    def test_heart_rate_metrics_schema_valid(self):
        """HeartRateMetrics can be instantiated with valid data."""
        metrics = HeartRateMetrics(
            resting_hr_avg=68,
            hrv_avg_ms=45.0,
            days_with_data=88,
            measurement_period_days=90,
        )
        assert metrics.resting_hr_avg == 68
        assert metrics.hrv_avg_ms == 45.0
    
    def test_heart_rate_metrics_constraints(self):
        """HeartRateMetrics enforces physiological constraints."""
        # Resting HR must be within reasonable bounds
        with pytest.raises(ValueError):
            HeartRateMetrics(
                resting_hr_avg=20,  # Too low
                days_with_data=30,
                measurement_period_days=30,
            )
        
        with pytest.raises(ValueError):
            HeartRateMetrics(
                resting_hr_avg=200,  # Too high
                days_with_data=30,
                measurement_period_days=30,
            )
    
    def test_sleep_metrics_schema_valid(self):
        """SleepMetrics can be instantiated with valid data."""
        metrics = SleepMetrics(
            avg_sleep_duration_hours=7.5,
            sleep_efficiency_pct=90.0,
            nights_with_data=82,
            measurement_period_days=90,
        )
        assert metrics.avg_sleep_duration_hours == 7.5
        assert metrics.sleep_efficiency_pct == 90.0
    
    def test_sleep_metrics_percentage_constraints(self):
        """SleepMetrics enforces percentage constraints."""
        # Sleep stages must not exceed 100%
        with pytest.raises(ValueError):
            SleepMetrics(
                deep_sleep_pct=150.0,  # Invalid
                nights_with_data=30,
                measurement_period_days=30,
            )
    
    def test_health_trends_schema_valid(self):
        """HealthTrends can be instantiated with valid data."""
        trends = HealthTrends(
            activity_trend_weekly="stable",
            overall_health_trajectory="positive",
            significant_changes=["Increased activity"],
        )
        assert trends.activity_trend_weekly == "stable"
        assert len(trends.significant_changes) == 1
    
    def test_health_metrics_bundle_schema_valid(self):
        """HealthMetrics bundle matches HealthDataAnalysisAgent input schema."""
        metrics = HealthMetrics(
            patient_id="TEST-001",
            activity=ActivityMetrics(days_with_data=30, measurement_period_days=30),
            heart_rate=HeartRateMetrics(days_with_data=30, measurement_period_days=30),
            sleep=SleepMetrics(nights_with_data=30, measurement_period_days=30),
            trends=HealthTrends(),
        )
        assert metrics.patient_id == "TEST-001"
        assert metrics.activity is not None
        assert metrics.heart_rate is not None
        assert metrics.sleep is not None
        assert metrics.trends is not None
    
    def test_health_metrics_only_allowed_categories(self):
        """HealthMetrics only contains activity, heart_rate, sleep, trends."""
        metrics = HealthMetrics(patient_id="TEST-001")
        # These are the ONLY health metric categories allowed
        allowed_fields = {"patient_id", "data_source", "collection_timestamp", 
                         "activity", "heart_rate", "sleep", "trends",
                         "consent_verified", "data_anonymized"}
        actual_fields = set(metrics.model_fields.keys())
        assert actual_fields == allowed_fields, \
            f"Unexpected health metric fields: {actual_fields - allowed_fields}"


# =============================================================================
# PATIENT PROFILE SCHEMA TESTS
# Validates schemas consumed by: HealthDataAnalysisAgent, OrchestratorAgent
# =============================================================================

class TestPatientProfileSchemas:
    """Test patient profile schemas match agent input requirements."""
    
    def test_patient_demographics_valid(self):
        """PatientDemographics can be instantiated with valid data."""
        demo = PatientDemographics(
            age=42,
            biological_sex="female",
            state_region="California",
        )
        assert demo.age == 42
        assert demo.biological_sex == "female"
    
    def test_patient_demographics_age_constraints(self):
        """PatientDemographics enforces age constraints."""
        # Age must be at least 18
        with pytest.raises(ValueError):
            PatientDemographics(age=15, biological_sex="male")
        
        # Age must be reasonable
        with pytest.raises(ValueError):
            PatientDemographics(age=150, biological_sex="male")
    
    def test_medical_history_valid(self):
        """MedicalHistory can be instantiated with valid data."""
        history = MedicalHistory(
            has_diabetes=False,
            smoker_status="never",
            bmi=24.5,
        )
        assert history.has_diabetes is False
        assert history.bmi == 24.5
    
    def test_medical_history_bmi_constraints(self):
        """MedicalHistory enforces BMI constraints."""
        # BMI must be physiologically possible
        with pytest.raises(ValueError):
            MedicalHistory(bmi=5.0)  # Too low
        
        with pytest.raises(ValueError):
            MedicalHistory(bmi=100.0)  # Too high
    
    def test_patient_profile_complete(self):
        """PatientProfile matches OrchestratorAgent patient_id lookup."""
        profile = PatientProfile(
            patient_id="TEST-001",
            demographics=PatientDemographics(age=35, biological_sex="male"),
            medical_history=MedicalHistory(),
            policy_type_requested="term_life",
            coverage_amount_requested=500000.00,
        )
        assert profile.patient_id == "TEST-001"
        assert profile.demographics.age == 35


# =============================================================================
# POLICY RULE SCHEMA TESTS
# Validates schemas consumed by: PolicyRiskAgent, BusinessRulesValidationAgent
# Simulates: policy-rule-engine, underwriting-rules-mcp MCP servers
# =============================================================================

class TestPolicyRuleSchemas:
    """Test policy rule schemas match agent input requirements."""
    
    def test_risk_threshold_valid(self):
        """RiskThreshold can be instantiated with valid data."""
        threshold = RiskThreshold(
            metric_name="resting_hr_avg",
            low_risk_max=70,
            moderate_risk_max=80,
            high_risk_max=90,
            unit="bpm",
            direction="lower_is_better",
        )
        assert threshold.metric_name == "resting_hr_avg"
        assert threshold.low_risk_max == 70
    
    def test_policy_rule_valid(self):
        """PolicyRule can be instantiated with valid data."""
        rule = PolicyRule(
            rule_id="TEST-001",
            rule_name="Test Rule",
            description="A test rule",
            category="heart_rate",
            condition_expression="resting_hr_avg > 70",
            risk_impact=RiskLevel.MODERATE,
            effective_date=date(2026, 1, 1),
        )
        assert rule.rule_id == "TEST-001"
        assert rule.risk_impact == RiskLevel.MODERATE
    
    def test_policy_rule_premium_adjustment_constraints(self):
        """PolicyRule enforces premium adjustment constraints."""
        # Premium adjustment must be within bounds
        with pytest.raises(ValueError):
            PolicyRule(
                rule_id="TEST",
                rule_name="Test",
                description="Test",
                category="test",
                condition_expression="test",
                risk_impact=RiskLevel.HIGH,
                premium_adjustment_pct=300.0,  # > 200%
                effective_date=date(2026, 1, 1),
            )
    
    def test_policy_rule_set_valid(self):
        """PolicyRuleSet matches PolicyRiskAgent.inputs.policy_rules."""
        rule_set = PolicyRuleSet(
            rule_set_id="TEST-SET",
            rule_set_name="Test Rules",
            policy_type="term_life",
            version="1.0",
            effective_date=date(2026, 1, 1),
            rules=[
                PolicyRule(
                    rule_id="R1",
                    rule_name="Rule 1",
                    description="Test",
                    category="activity",
                    condition_expression="steps < 5000",
                    risk_impact=RiskLevel.MODERATE,
                    effective_date=date(2026, 1, 1),
                ),
            ],
        )
        assert rule_set.rule_set_id == "TEST-SET"
        assert len(rule_set.rules) == 1


# =============================================================================
# PREMIUM OUTCOME SCHEMA TESTS
# Validates schemas produced by: PolicyRiskAgent, consumed by: BusinessRulesValidationAgent
# =============================================================================

class TestPremiumOutcomeSchemas:
    """Test premium outcome schemas match agent output requirements."""
    
    def test_premium_adjustment_valid(self):
        """PremiumAdjustment matches BusinessRulesValidationAgent input."""
        adjustment = PremiumAdjustment(
            base_premium_annual=1200.00,
            adjustment_percentage=15.0,
            adjusted_premium_annual=1380.00,
            adjustment_factors={"RULE-HR-001": 10.0, "RULE-SLEEP-001": 5.0},
            triggered_rule_ids=["RULE-HR-001", "RULE-SLEEP-001"],
        )
        assert adjustment.adjusted_premium_annual == 1380.00
        assert len(adjustment.triggered_rule_ids) == 2
    
    def test_premium_adjustment_constraints(self):
        """PremiumAdjustment enforces adjustment bounds."""
        # Adjustment percentage must be within -50% to +200%
        with pytest.raises(ValueError):
            PremiumAdjustment(
                base_premium_annual=1000,
                adjustment_percentage=-60.0,  # < -50%
                adjusted_premium_annual=400,
            )
    
    def test_underwriting_decision_valid(self):
        """UnderwritingDecision matches CommunicationAgent input."""
        decision = UnderwritingDecision(
            decision_id="DEC-001",
            patient_id="PAT-001",
            status=DecisionStatus.APPROVED_WITH_ADJUSTMENT,
            risk_level=RiskLevel.MODERATE,
            confidence_score=0.85,
            data_quality_level=DataQualityLevel.GOOD,
            decision_rationale="Approved with adjustment due to elevated heart rate.",
        )
        assert decision.status == DecisionStatus.APPROVED_WITH_ADJUSTMENT
        assert decision.confidence_score == 0.85
    
    def test_premium_outcome_test_case_valid(self):
        """PremiumOutcome defines valid test expectations."""
        outcome = PremiumOutcome(
            test_case_id="TC-001",
            patient_id="PAT-001",
            expected_risk_level=RiskLevel.MODERATE,
            expected_decision_status=DecisionStatus.APPROVED_WITH_ADJUSTMENT,
            expected_premium_adjustment_pct_min=10.0,
            expected_premium_adjustment_pct_max=30.0,
            must_trigger_rules=["RULE-HR-001"],
            test_description="Test case description",
        )
        assert outcome.expected_premium_adjustment_pct_min < outcome.expected_premium_adjustment_pct_max


# =============================================================================
# AGENT I/O SCHEMA TESTS
# Validates internal agent communication schemas
# =============================================================================

class TestAgentIOSchemas:
    """Test agent I/O schemas for inter-agent communication."""
    
    def test_risk_indicator_valid(self):
        """RiskIndicator matches HealthDataAnalysisAgent output."""
        indicator = RiskIndicator(
            indicator_id="IND-001",
            category="heart_rate",
            indicator_name="Elevated Resting HR",
            risk_level=RiskLevel.MODERATE,
            confidence=0.85,
            explanation="Resting HR of 78 bpm exceeds threshold.",
        )
        assert indicator.category == "heart_rate"
        assert indicator.risk_level == RiskLevel.MODERATE
    
    def test_risk_indicator_category_values(self):
        """RiskIndicator categories match health metric types."""
        # Only these categories are valid (matching health metrics)
        valid_categories = {"activity", "heart_rate", "sleep", "combined"}
        
        for category in valid_categories:
            indicator = RiskIndicator(
                indicator_id="TEST",
                category=category,
                indicator_name="Test",
                risk_level=RiskLevel.LOW,
                confidence=0.5,
                explanation="Test",
            )
            assert indicator.category == category
    
    def test_quality_flag_valid(self):
        """QualityFlag matches DataQualityConfidenceAgent output."""
        flag = QualityFlag(
            flag_id="QF-001",
            flag_type="incomplete",
            severity="warning",
            affected_metric="sleep",
            description="Sleep data only 60% complete",
            confidence_impact=-0.1,
        )
        assert flag.flag_type == "incomplete"
        assert flag.confidence_impact == -0.1
    
    def test_quality_flag_impact_constraints(self):
        """QualityFlag confidence impact must be negative."""
        # Impact must be between -1 and 0
        with pytest.raises(ValueError):
            QualityFlag(
                flag_id="TEST",
                flag_type="test",
                severity="info",
                affected_metric="test",
                description="test",
                confidence_impact=0.5,  # Must be negative
            )
    
    def test_bias_flag_valid(self):
        """BiasFlag matches BiasAndFairnessAgent output."""
        flag = BiasFlag(
            flag_id="BF-001",
            bias_type="age_discrimination",
            severity="low",
            description="Age factor may be excessive",
            mitigation_applied=True,
            mitigation_notes="Applied age-band smoothing",
            blocks_decision=False,
        )
        assert flag.bias_type == "age_discrimination"
        assert flag.mitigation_applied is True


# =============================================================================
# FIXTURE INTEGRATION TESTS
# Validates fixtures produce valid schema instances
# =============================================================================

class TestFixtureIntegrity:
    """Test fixtures produce schema-valid data."""
    
    def test_healthy_patient_metrics_valid(self):
        """Healthy patient fixture produces valid HealthMetrics."""
        metrics = get_healthy_patient_metrics()
        assert isinstance(metrics, HealthMetrics)
        assert metrics.patient_id == "PAT-HEALTHY-001"
        assert metrics.activity is not None
        assert metrics.heart_rate is not None
        assert metrics.sleep is not None
    
    def test_moderate_risk_metrics_valid(self):
        """Moderate risk fixture produces valid HealthMetrics."""
        metrics = get_moderate_risk_metrics()
        assert isinstance(metrics, HealthMetrics)
        assert metrics.patient_id == "PAT-MODERATE-001"
        # Check for moderate risk indicators
        assert metrics.heart_rate.resting_hr_avg > 70  # Elevated
    
    def test_high_risk_metrics_valid(self):
        """High risk fixture produces valid HealthMetrics."""
        metrics = get_high_risk_metrics()
        assert isinstance(metrics, HealthMetrics)
        assert metrics.patient_id == "PAT-HIGH-RISK-001"
        # Check for high risk indicators
        assert metrics.heart_rate.resting_hr_avg > 80
        assert metrics.heart_rate.irregular_rhythm_events > 0
    
    def test_incomplete_metrics_valid(self):
        """Incomplete data fixture produces valid HealthMetrics with gaps."""
        metrics = get_incomplete_metrics()
        assert isinstance(metrics, HealthMetrics)
        assert metrics.patient_id == "PAT-INCOMPLETE-001"
        # Check for data quality issues
        assert metrics.activity.days_with_data < 30  # Sparse data
        assert metrics.sleep.avg_sleep_duration_hours is None  # Missing
    
    def test_patient_profiles_all_valid(self):
        """All patient profile fixtures are valid."""
        profiles = get_sample_patient_profiles()
        assert len(profiles) >= 5  # At least 5 profiles
        
        for profile in profiles:
            assert isinstance(profile, PatientProfile)
            assert profile.patient_id is not None
            assert profile.demographics.age >= 18
    
    def test_patient_lookup_by_id(self):
        """Patient lookup returns correct profile."""
        profile = get_patient_by_id("PAT-HEALTHY-001")
        assert profile is not None
        assert profile.patient_id == "PAT-HEALTHY-001"
        
        # Non-existent patient
        missing = get_patient_by_id("PAT-NONEXISTENT")
        assert missing is None
    
    def test_policy_rules_valid(self):
        """Policy rules fixture produces valid PolicyRuleSet."""
        rules = get_standard_policy_rules()
        assert isinstance(rules, PolicyRuleSet)
        assert rules.policy_type == "term_life"
        assert len(rules.rules) > 0
        
        # Check all rules are valid
        for rule in rules.rules:
            assert isinstance(rule, PolicyRule)
            assert rule.rule_id is not None
            assert rule.risk_impact in RiskLevel
    
    def test_policy_rules_cover_all_categories(self):
        """Policy rules cover activity, heart_rate, sleep, and combined."""
        rules = get_standard_policy_rules()
        categories = {rule.category for rule in rules.rules}
        
        required = {"activity", "heart_rate", "sleep", "combined"}
        assert required.issubset(categories), \
            f"Missing rule categories: {required - categories}"
    
    def test_policy_rule_lookup_by_id(self):
        """Policy rule lookup returns correct rule."""
        rule = get_policy_rule_by_id("RULE-HR-001")
        assert rule is not None
        assert rule.rule_id == "RULE-HR-001"
        assert rule.category == "heart_rate"
        
        # Non-existent rule
        missing = get_policy_rule_by_id("RULE-NONEXISTENT")
        assert missing is None
    
    def test_expected_outcomes_valid(self):
        """Expected outcomes are valid and reference real patients."""
        outcomes = get_expected_outcomes()
        assert len(outcomes) >= 4  # At least 4 test cases
        
        for outcome in outcomes:
            assert isinstance(outcome, PremiumOutcome)
            # Verify patient exists
            patient = get_patient_by_id(outcome.patient_id)
            assert patient is not None, f"Outcome references non-existent patient: {outcome.patient_id}"
            # Verify range is valid
            assert outcome.expected_premium_adjustment_pct_min <= outcome.expected_premium_adjustment_pct_max


# =============================================================================
# AGENT INPUT MAPPING TESTS
# Validates schemas align with underwriting_agents.yaml definitions
# =============================================================================

class TestAgentInputMapping:
    """Test schemas align with agent definitions in underwriting_agents.yaml."""
    
    def test_health_data_analysis_agent_inputs(self):
        """HealthMetrics and PatientProfile match HealthDataAnalysisAgent inputs."""
        # HealthDataAnalysisAgent.inputs: health_metrics (object), patient_profile (object)
        metrics = get_healthy_patient_metrics()
        profile = get_patient_by_id(metrics.patient_id)
        
        assert profile is not None
        # Both should be valid objects
        assert isinstance(metrics, HealthMetrics)
        assert isinstance(profile, PatientProfile)
    
    def test_policy_risk_agent_inputs(self):
        """RiskIndicator[] and PolicyRuleSet match PolicyRiskAgent inputs."""
        # PolicyRiskAgent.inputs: risk_indicators (list), policy_rules (object)
        indicator = RiskIndicator(
            indicator_id="TEST",
            category="heart_rate",
            indicator_name="Test",
            risk_level=RiskLevel.MODERATE,
            confidence=0.8,
            explanation="Test",
        )
        rules = get_standard_policy_rules()
        
        # risk_indicators should be a list
        risk_indicators = [indicator]
        assert isinstance(risk_indicators, list)
        
        # policy_rules should be an object (PolicyRuleSet)
        assert isinstance(rules, PolicyRuleSet)
    
    def test_business_rules_validation_agent_inputs(self):
        """PremiumAdjustment matches BusinessRulesValidationAgent inputs."""
        # BusinessRulesValidationAgent.inputs: premium_adjustment_recommendation (object)
        adjustment = PremiumAdjustment(
            base_premium_annual=1200,
            adjustment_percentage=15.0,
            adjusted_premium_annual=1380,
        )
        
        assert isinstance(adjustment, PremiumAdjustment)
    
    def test_data_quality_confidence_agent_inputs(self):
        """HealthMetrics matches DataQualityConfidenceAgent inputs."""
        # DataQualityConfidenceAgent.inputs: health_metrics (object)
        metrics = get_incomplete_metrics()
        
        assert isinstance(metrics, HealthMetrics)
        # Should be able to detect quality issues
        assert metrics.activity.days_with_data < metrics.activity.measurement_period_days * 0.5
    
    def test_communication_agent_inputs(self):
        """UnderwritingDecision matches CommunicationAgent inputs."""
        # CommunicationAgent.inputs: decision_summary (object)
        decision = UnderwritingDecision(
            decision_id="TEST",
            patient_id="PAT-001",
            status=DecisionStatus.APPROVED,
            risk_level=RiskLevel.LOW,
            confidence_score=0.9,
            data_quality_level=DataQualityLevel.GOOD,
            decision_rationale="Approved",
        )
        
        assert isinstance(decision, UnderwritingDecision)
        # Should have all fields needed for communication
        assert decision.decision_rationale is not None
        assert decision.status is not None
    
    def test_orchestrator_agent_inputs(self):
        """patient_id lookup matches OrchestratorAgent inputs."""
        # OrchestratorAgent.inputs: patient_id (string)
        patient_id = "PAT-HEALTHY-001"
        
        # Should be able to look up patient and get full profile
        profile = get_patient_by_id(patient_id)
        assert profile is not None
        
        # Should be able to get metrics for this patient
        metrics = get_healthy_patient_metrics()
        assert metrics.patient_id == patient_id


# =============================================================================
# MCP SERVER SIMULATION MAPPING TESTS
# Validates fixtures simulate expected MCP server responses
# =============================================================================

class TestMCPServerSimulation:
    """Test fixtures correctly simulate MCP server responses."""
    
    def test_medical_mcp_simulator_response(self):
        """Health metrics fixtures simulate medical-mcp-simulator."""
        # medical-mcp-simulator returns health data
        metrics = get_healthy_patient_metrics()
        
        # Should have data_source indicating simulation
        assert "simulated" in metrics.data_source.lower()
        
        # Should have all expected health metric categories
        assert metrics.activity is not None
        assert metrics.heart_rate is not None
        assert metrics.sleep is not None
        assert metrics.trends is not None
    
    def test_policy_rule_engine_response(self):
        """Policy rules fixture simulates policy-rule-engine."""
        # policy-rule-engine returns rules for risk assessment
        rules = get_standard_policy_rules()
        
        # Should have version and effective date
        assert rules.version is not None
        assert rules.effective_date is not None
        
        # Should have rules with thresholds
        rules_with_thresholds = [r for r in rules.rules if r.thresholds]
        assert len(rules_with_thresholds) > 0
    
    def test_underwriting_rules_mcp_response(self):
        """Policy rules include regulatory/compliance rules for underwriting-rules-mcp."""
        rules = get_standard_policy_rules()
        
        # Should have mandatory/regulatory rules
        mandatory_rules = [r for r in rules.rules if r.is_mandatory]
        assert len(mandatory_rules) > 0
        
        # Should have regulatory reference on some rules
        rules_with_refs = [r for r in rules.rules if r.regulatory_reference]
        assert len(rules_with_refs) > 0
