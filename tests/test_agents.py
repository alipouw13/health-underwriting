"""
Unit Tests for Underwriting Agents

Tests validate:
1. Input schema validation
2. Output schema validation
3. Deterministic behavior with mock data
4. Agent isolation (no inter-agent calls)
5. No orchestration logic
"""

import pytest
from datetime import date, datetime, timezone
from uuid import uuid4

from data.mock.schemas import (
    ActivityMetrics,
    HealthMetrics,
    HealthTrends,
    HeartRateMetrics,
    PatientDemographics,
    PatientProfile,
    MedicalHistory,
    PolicyRule,
    PolicyRuleSet,
    PremiumAdjustment,
    RiskIndicator,
    RiskLevel,
    RiskThreshold,
    SleepMetrics,
    UnderwritingDecision,
    DecisionStatus,
    DataQualityLevel,
    QualityFlag,
    BiasFlag,
)
from data.mock.fixtures import (
    get_healthy_patient_metrics,
    get_moderate_risk_metrics,
    get_high_risk_metrics,
    get_incomplete_metrics,
    get_sample_patient_profiles,
    get_standard_policy_rules,
)

from app.agents.base import (
    AgentValidationError,
    AgentExecutionError,
)
from app.agents.health_data_analysis import (
    HealthDataAnalysisAgent,
    HealthDataAnalysisInput,
    HealthDataAnalysisOutput,
)
from app.agents.policy_risk import (
    PolicyRiskAgent,
    PolicyRiskInput,
    PolicyRiskOutput,
)
from app.agents.business_rules_validation import (
    BusinessRulesValidationAgent,
    BusinessRulesValidationInput,
    BusinessRulesValidationOutput,
)
from app.agents.data_quality_confidence import (
    DataQualityConfidenceAgent,
    DataQualityConfidenceInput,
    DataQualityConfidenceOutput,
)
from app.agents.bias_fairness import (
    BiasAndFairnessAgent,
    BiasAndFairnessInput,
    BiasAndFairnessOutput,
    DecisionContext,
)
from app.agents.communication import (
    CommunicationAgent,
    CommunicationInput,
    CommunicationOutput,
    DecisionSummary,
)
from app.agents.audit_trace import (
    AuditAndTraceAgent,
    AuditAndTraceInput,
    AuditAndTraceOutput,
    AgentOutputRecord,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def healthy_patient_profile():
    """Fixture for a healthy patient profile."""
    profiles = get_sample_patient_profiles()
    return profiles[0]  # PAT-001 - healthy


@pytest.fixture
def moderate_risk_profile():
    """Fixture for a moderate risk patient profile."""
    profiles = get_sample_patient_profiles()
    return profiles[1]  # PAT-002 - moderate risk


@pytest.fixture
def high_risk_profile():
    """Fixture for a high risk patient profile."""
    profiles = get_sample_patient_profiles()
    return profiles[2]  # PAT-003 - high risk


@pytest.fixture
def healthy_metrics():
    """Fixture for healthy health metrics."""
    return get_healthy_patient_metrics()


@pytest.fixture
def moderate_risk_metrics():
    """Fixture for moderate risk health metrics."""
    return get_moderate_risk_metrics()


@pytest.fixture
def high_risk_metrics():
    """Fixture for high risk health metrics."""
    return get_high_risk_metrics()


@pytest.fixture
def incomplete_metrics():
    """Fixture for incomplete health metrics."""
    return get_incomplete_metrics()


@pytest.fixture
def policy_rules():
    """Fixture for standard policy rules."""
    return get_standard_policy_rules()


@pytest.fixture
def health_data_analysis_agent():
    """Fixture for HealthDataAnalysisAgent."""
    return HealthDataAnalysisAgent()


@pytest.fixture
def policy_risk_agent():
    """Fixture for PolicyRiskAgent."""
    return PolicyRiskAgent()


@pytest.fixture
def business_rules_agent():
    """Fixture for BusinessRulesValidationAgent."""
    return BusinessRulesValidationAgent()


@pytest.fixture
def data_quality_agent():
    """Fixture for DataQualityConfidenceAgent."""
    return DataQualityConfidenceAgent()


@pytest.fixture
def bias_fairness_agent():
    """Fixture for BiasAndFairnessAgent."""
    return BiasAndFairnessAgent()


@pytest.fixture
def communication_agent():
    """Fixture for CommunicationAgent."""
    return CommunicationAgent()


@pytest.fixture
def audit_trace_agent():
    """Fixture for AuditAndTraceAgent."""
    return AuditAndTraceAgent()


# =============================================================================
# TEST: HEALTH DATA ANALYSIS AGENT
# =============================================================================

class TestHealthDataAnalysisAgent:
    """Tests for HealthDataAnalysisAgent."""
    
    def test_agent_attributes(self, health_data_analysis_agent):
        """Test agent has correct attributes from YAML."""
        agent = health_data_analysis_agent
        
        assert agent.agent_id == "HealthDataAnalysisAgent"
        assert "medical-mcp-simulator" in agent.tools_used
        assert "signal_accuracy" in agent.evaluation_criteria
        assert "missing_data" in agent.failure_modes
    
    @pytest.mark.asyncio
    async def test_healthy_patient_analysis(
        self, 
        health_data_analysis_agent,
        healthy_metrics,
        healthy_patient_profile
    ):
        """Test analysis of healthy patient produces low risk indicators."""
        agent = health_data_analysis_agent
        
        input_data = {
            "health_metrics": healthy_metrics.model_dump(),
            "patient_profile": healthy_patient_profile.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        # Validate output schema
        assert isinstance(output, HealthDataAnalysisOutput)
        assert output.agent_id == "HealthDataAnalysisAgent"
        assert output.success is True
        
        # Validate risk indicators
        assert isinstance(output.risk_indicators, list)
        assert len(output.risk_indicators) > 0
        
        # Healthy patient should have mostly low risk indicators
        low_risk_count = sum(
            1 for ind in output.risk_indicators 
            if ind.risk_level == RiskLevel.LOW
        )
        assert low_risk_count >= len(output.risk_indicators) // 2
        
        # Validate summary contains patient ID
        assert healthy_metrics.patient_id in output.summary
    
    @pytest.mark.asyncio
    async def test_high_risk_patient_analysis(
        self,
        health_data_analysis_agent,
        high_risk_metrics,
        high_risk_profile
    ):
        """Test analysis of high risk patient produces elevated risk indicators."""
        agent = health_data_analysis_agent
        
        input_data = {
            "health_metrics": high_risk_metrics.model_dump(),
            "patient_profile": high_risk_profile.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        # High risk patient should have elevated risk indicators
        high_risk_count = sum(
            1 for ind in output.risk_indicators
            if ind.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]
        )
        assert high_risk_count > 0
        
        # Summary should indicate elevated risk
        assert "RISK" in output.summary.upper()
    
    @pytest.mark.asyncio
    async def test_input_validation_fails_for_invalid_data(
        self,
        health_data_analysis_agent
    ):
        """Test that invalid input raises validation error."""
        agent = health_data_analysis_agent
        
        # Missing required fields
        invalid_input = {
            "health_metrics": {"patient_id": "TEST"},
            # Missing patient_profile
        }
        
        with pytest.raises(AgentValidationError) as exc_info:
            await agent.run(invalid_input)
        
        assert "input" in exc_info.value.validation_type
    
    @pytest.mark.asyncio
    async def test_deterministic_output(
        self,
        health_data_analysis_agent,
        healthy_metrics,
        healthy_patient_profile
    ):
        """Test that same input produces consistent output."""
        agent = health_data_analysis_agent
        
        input_data = {
            "health_metrics": healthy_metrics.model_dump(),
            "patient_profile": healthy_patient_profile.model_dump(),
        }
        
        output1 = await agent.run(input_data)
        output2 = await agent.run(input_data)
        
        # Risk indicators should be same count and same levels
        assert len(output1.risk_indicators) == len(output2.risk_indicators)
        
        for ind1, ind2 in zip(output1.risk_indicators, output2.risk_indicators):
            assert ind1.risk_level == ind2.risk_level
            assert ind1.category == ind2.category


# =============================================================================
# TEST: POLICY RISK AGENT
# =============================================================================

class TestPolicyRiskAgent:
    """Tests for PolicyRiskAgent."""
    
    def test_agent_attributes(self, policy_risk_agent):
        """Test agent has correct attributes from YAML."""
        agent = policy_risk_agent
        
        assert agent.agent_id == "PolicyRiskAgent"
        assert "policy-rule-engine" in agent.tools_used
        assert "rule_alignment" in agent.evaluation_criteria
        assert "conflicting_rules" in agent.failure_modes
    
    @pytest.mark.asyncio
    async def test_low_risk_indicators_produce_low_premium(
        self,
        policy_risk_agent,
        policy_rules
    ):
        """Test that low risk indicators produce no/minimal premium adjustment."""
        agent = policy_risk_agent
        
        # Create low risk indicators
        risk_indicators = [
            RiskIndicator(
                indicator_id="IND-001",
                category="activity",
                indicator_name="Daily Steps",
                risk_level=RiskLevel.LOW,
                confidence=0.9,
                metric_value=10000,
                metric_unit="steps",
                explanation="Healthy activity level",
            ),
            RiskIndicator(
                indicator_id="IND-002",
                category="heart_rate",
                indicator_name="Resting HR",
                risk_level=RiskLevel.LOW,
                confidence=0.95,
                metric_value=62,
                metric_unit="bpm",
                explanation="Excellent cardiovascular health",
            ),
        ]
        
        input_data = {
            "risk_indicators": [r.model_dump() for r in risk_indicators],
            "policy_rules": policy_rules.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        assert isinstance(output, PolicyRiskOutput)
        assert output.risk_level == RiskLevel.LOW
        # Low risk indicators should result in low or modest adjustment
        # (discounts from good metrics may offset other rule triggers)
        assert output.premium_adjustment_recommendation.adjustment_percentage <= 30
    
    @pytest.mark.asyncio
    async def test_high_risk_indicators_increase_premium(
        self,
        policy_risk_agent,
        policy_rules
    ):
        """Test that high risk indicators produce premium increase."""
        agent = policy_risk_agent
        
        # Create high risk indicators
        risk_indicators = [
            RiskIndicator(
                indicator_id="IND-001",
                category="activity",
                indicator_name="Daily Steps",
                risk_level=RiskLevel.HIGH,
                confidence=0.85,
                metric_value=2000,
                metric_unit="steps",
                explanation="Very low activity",
            ),
            RiskIndicator(
                indicator_id="IND-002",
                category="heart_rate",
                indicator_name="Resting HR",
                risk_level=RiskLevel.VERY_HIGH,
                confidence=0.9,
                metric_value=95,
                metric_unit="bpm",
                explanation="Elevated resting heart rate",
            ),
        ]
        
        input_data = {
            "risk_indicators": [r.model_dump() for r in risk_indicators],
            "policy_rules": policy_rules.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        assert output.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]
        assert output.premium_adjustment_recommendation.adjustment_percentage > 0
        assert len(output.triggered_rules) > 0
    
    @pytest.mark.asyncio
    async def test_premium_calculation_consistency(
        self,
        policy_risk_agent,
        policy_rules
    ):
        """Test that premium calculations are internally consistent."""
        agent = policy_risk_agent
        
        risk_indicators = [
            RiskIndicator(
                indicator_id="IND-001",
                category="heart_rate",
                indicator_name="Test",
                risk_level=RiskLevel.MODERATE,
                confidence=0.8,
                explanation="Test indicator",
            ),
        ]
        
        input_data = {
            "risk_indicators": [r.model_dump() for r in risk_indicators],
            "policy_rules": policy_rules.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        # Verify premium calculation
        expected_premium = (
            output.premium_adjustment_recommendation.base_premium_annual *
            (1 + output.premium_adjustment_recommendation.adjustment_percentage / 100)
        )
        
        assert abs(
            output.premium_adjustment_recommendation.adjusted_premium_annual - expected_premium
        ) < 0.01


# =============================================================================
# TEST: BUSINESS RULES VALIDATION AGENT
# =============================================================================

class TestBusinessRulesValidationAgent:
    """Tests for BusinessRulesValidationAgent."""
    
    def test_agent_attributes(self, business_rules_agent):
        """Test agent has correct attributes from YAML."""
        agent = business_rules_agent
        
        assert agent.agent_id == "BusinessRulesValidationAgent"
        assert "underwriting-rules-mcp" in agent.tools_used
        assert "compliance" in agent.evaluation_criteria
        assert "rule_violation" in agent.failure_modes
    
    @pytest.mark.asyncio
    async def test_valid_adjustment_approved(self, business_rules_agent):
        """Test that valid premium adjustment is approved."""
        agent = business_rules_agent
        
        premium_adjustment = PremiumAdjustment(
            base_premium_annual=1200.00,
            adjustment_percentage=15.0,
            adjusted_premium_annual=1380.00,
            adjustment_factors={"RULE-HR-001": 15.0},
            triggered_rule_ids=["RULE-HR-001"],
        )
        
        input_data = {
            "premium_adjustment_recommendation": premium_adjustment.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        assert isinstance(output, BusinessRulesValidationOutput)
        assert output.approved is True
        assert len(output.violations_found) == 0
    
    @pytest.mark.asyncio
    async def test_excessive_adjustment_rejected(self, business_rules_agent):
        """Test that excessive premium adjustment is rejected."""
        agent = business_rules_agent
        
        premium_adjustment = PremiumAdjustment(
            base_premium_annual=1200.00,
            adjustment_percentage=200.0,  # Exceeds 150% limit
            adjusted_premium_annual=3600.00,
            adjustment_factors={"RULE-001": 200.0},
            triggered_rule_ids=["RULE-001"],
        )
        
        input_data = {
            "premium_adjustment_recommendation": premium_adjustment.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        assert output.approved is False
        assert len(output.violations_found) > 0
        assert any("exceeds maximum" in v.lower() for v in output.violations_found)
    
    @pytest.mark.asyncio
    async def test_calculation_inconsistency_detected(self, business_rules_agent):
        """Test that calculation inconsistencies are detected."""
        agent = business_rules_agent
        
        premium_adjustment = PremiumAdjustment(
            base_premium_annual=1200.00,
            adjustment_percentage=10.0,
            adjusted_premium_annual=1500.00,  # Should be 1320, not 1500
            adjustment_factors={"RULE-001": 10.0},
            triggered_rule_ids=["RULE-001"],
        )
        
        input_data = {
            "premium_adjustment_recommendation": premium_adjustment.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        assert output.approved is False
        assert any("inconsistent" in v.lower() for v in output.violations_found)


# =============================================================================
# TEST: DATA QUALITY CONFIDENCE AGENT
# =============================================================================

class TestDataQualityConfidenceAgent:
    """Tests for DataQualityConfidenceAgent."""
    
    def test_agent_attributes(self, data_quality_agent):
        """Test agent has correct attributes from YAML."""
        agent = data_quality_agent
        
        assert agent.agent_id == "DataQualityConfidenceAgent"
        assert "data-quality-analyzer" in agent.tools_used
        assert "coverage" in agent.evaluation_criteria
        assert "insufficient_data" in agent.failure_modes
    
    @pytest.mark.asyncio
    async def test_complete_data_high_confidence(
        self,
        data_quality_agent,
        healthy_metrics
    ):
        """Test that complete data produces high confidence score."""
        agent = data_quality_agent
        
        input_data = {
            "health_metrics": healthy_metrics.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        assert isinstance(output, DataQualityConfidenceOutput)
        assert output.confidence_score >= 0.7
        assert output.data_quality_level in [DataQualityLevel.EXCELLENT, DataQualityLevel.GOOD]
    
    @pytest.mark.asyncio
    async def test_incomplete_data_low_confidence(
        self,
        data_quality_agent,
        incomplete_metrics
    ):
        """Test that incomplete data produces low confidence score."""
        agent = data_quality_agent
        
        input_data = {
            "health_metrics": incomplete_metrics.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        assert output.confidence_score < 0.8
        assert len(output.quality_flags) > 0
        assert len(output.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_missing_metrics_flagged(self, data_quality_agent):
        """Test that missing metric types are flagged."""
        agent = data_quality_agent
        
        # Metrics with no heart rate data
        metrics = HealthMetrics(
            patient_id="TEST",
            activity=ActivityMetrics(
                days_with_data=80,
                measurement_period_days=90,
                daily_steps_avg=8000,
            ),
            heart_rate=None,  # Missing
            sleep=None,  # Missing
        )
        
        input_data = {
            "health_metrics": metrics.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        # Should have flags for missing data
        missing_flags = [
            f for f in output.quality_flags 
            if f.flag_type == "missing_data"
        ]
        assert len(missing_flags) >= 2  # heart_rate and sleep missing


# =============================================================================
# TEST: BIAS AND FAIRNESS AGENT
# =============================================================================

class TestBiasAndFairnessAgent:
    """Tests for BiasAndFairnessAgent."""
    
    def test_agent_attributes(self, bias_fairness_agent):
        """Test agent has correct attributes from YAML."""
        agent = bias_fairness_agent
        
        assert agent.agent_id == "BiasAndFairnessAgent"
        assert "fairness-checker" in agent.tools_used
        assert "fairness" in agent.evaluation_criteria
        assert "bias_detected" in agent.failure_modes
    
    @pytest.mark.asyncio
    async def test_fair_decision_high_score(self, bias_fairness_agent):
        """Test that a fair decision produces high fairness score."""
        agent = bias_fairness_agent
        
        context = DecisionContext(
            patient_age=35,
            patient_sex="female",
            patient_region="California",
            risk_level=RiskLevel.LOW,
            premium_adjustment_pct=5.0,
            triggered_rules=["RULE-ACTIVITY-001"],
            health_metrics_used=["activity", "heart_rate", "sleep"],
        )
        
        input_data = {
            "decision_context": context.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        assert isinstance(output, BiasAndFairnessOutput)
        assert output.fairness_score >= 0.8
        assert len(output.bias_flags) == 0 or all(
            not f.blocks_decision for f in output.bias_flags
        )
    
    @pytest.mark.asyncio
    async def test_senior_with_high_adjustment_flagged(self, bias_fairness_agent):
        """Test that high adjustment for senior is flagged for review."""
        agent = bias_fairness_agent
        
        context = DecisionContext(
            patient_age=70,
            patient_sex="male",
            patient_region="Florida",
            risk_level=RiskLevel.HIGH,
            premium_adjustment_pct=80.0,  # High adjustment
            triggered_rules=["RULE-HR-001"],
            health_metrics_used=["heart_rate"],
        )
        
        input_data = {
            "decision_context": context.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        # Should have age-related flag
        age_flags = [f for f in output.bias_flags if f.bias_type == "age_discrimination"]
        assert len(age_flags) > 0
    
    @pytest.mark.asyncio
    async def test_protected_region_geographic_bias(self, bias_fairness_agent):
        """Test that protected region triggers geographic bias check."""
        agent = bias_fairness_agent
        
        context = DecisionContext(
            patient_age=40,
            patient_sex="female",
            patient_region="rural_low_income",  # Contains protected term
            risk_level=RiskLevel.MODERATE,
            premium_adjustment_pct=25.0,
            triggered_rules=["RULE-GEOGRAPHIC-001"],  # Geographic rule
            health_metrics_used=["activity"],
        )
        
        input_data = {
            "decision_context": context.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        # Should have geographic bias flag
        geo_flags = [f for f in output.bias_flags if f.bias_type == "geographic_bias"]
        assert len(geo_flags) > 0


# =============================================================================
# TEST: COMMUNICATION AGENT
# =============================================================================

class TestCommunicationAgent:
    """Tests for CommunicationAgent."""
    
    def test_agent_attributes(self, communication_agent):
        """Test agent has correct attributes from YAML."""
        agent = communication_agent
        
        assert agent.agent_id == "CommunicationAgent"
        assert "language-generator" in agent.tools_used
        assert "clarity" in agent.evaluation_criteria
        assert "ambiguous_language" in agent.failure_modes
    
    @pytest.mark.asyncio
    async def test_approved_decision_positive_message(self, communication_agent):
        """Test that approved decision generates positive customer message."""
        agent = communication_agent
        
        decision = UnderwritingDecision(
            decision_id="DEC-001",
            patient_id="PAT-001",
            status=DecisionStatus.APPROVED,
            risk_level=RiskLevel.LOW,
            confidence_score=0.95,
            data_quality_level=DataQualityLevel.EXCELLENT,
            decision_rationale="Excellent health profile with low risk indicators.",
            premium_adjustment=PremiumAdjustment(
                base_premium_annual=1200.00,
                adjustment_percentage=0.0,
                adjusted_premium_annual=1200.00,
            ),
        )
        
        summary = DecisionSummary(
            decision=decision,
            patient_name="John Doe",
            policy_type="term_life",
            coverage_amount=500000.00,
        )
        
        input_data = {
            "decision_summary": summary.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        assert isinstance(output, CommunicationOutput)
        assert "pleased" in output.customer_message.lower() or "approved" in output.customer_message.lower()
        assert "positive" in output.tone_assessment.lower()
        assert output.readability_score >= 50  # Should be readable
    
    @pytest.mark.asyncio
    async def test_declined_decision_empathetic_message(self, communication_agent):
        """Test that declined decision generates empathetic customer message."""
        agent = communication_agent
        
        decision = UnderwritingDecision(
            decision_id="DEC-002",
            patient_id="PAT-002",
            status=DecisionStatus.DECLINED,
            risk_level=RiskLevel.VERY_HIGH,
            confidence_score=0.9,
            data_quality_level=DataQualityLevel.GOOD,
            decision_rationale="Multiple high-risk factors present.",
        )
        
        summary = DecisionSummary(
            decision=decision,
            policy_type="term_life",
            coverage_amount=1000000.00,
        )
        
        input_data = {
            "decision_summary": summary.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        # Should not use harsh language
        assert "rejected" not in output.customer_message.lower()
        # Should offer alternatives or contact info
        assert "contact" in output.customer_message.lower() or "alternative" in output.customer_message.lower()
    
    @pytest.mark.asyncio
    async def test_underwriter_message_includes_technical_details(
        self, 
        communication_agent
    ):
        """Test that underwriter message includes technical details."""
        agent = communication_agent
        
        decision = UnderwritingDecision(
            decision_id="DEC-003",
            patient_id="PAT-003",
            status=DecisionStatus.APPROVED_WITH_ADJUSTMENT,
            risk_level=RiskLevel.MODERATE,
            confidence_score=0.85,
            data_quality_level=DataQualityLevel.GOOD,
            decision_rationale="Moderate risk due to elevated heart rate.",
            key_risk_factors=["Elevated resting HR", "Limited sleep data"],
            premium_adjustment=PremiumAdjustment(
                base_premium_annual=1200.00,
                adjustment_percentage=20.0,
                adjusted_premium_annual=1440.00,
                triggered_rule_ids=["RULE-HR-001"],
            ),
        )
        
        summary = DecisionSummary(
            decision=decision,
            policy_type="health",
            coverage_amount=250000.00,
        )
        
        input_data = {
            "decision_summary": summary.model_dump(),
        }
        
        output = await agent.run(input_data)
        
        # Underwriter message should have technical details
        assert decision.decision_id in output.underwriter_message
        assert "RULE-HR-001" in output.underwriter_message
        assert "20" in output.underwriter_message  # Adjustment percentage


# =============================================================================
# TEST: AUDIT AND TRACE AGENT
# =============================================================================

class TestAuditAndTraceAgent:
    """Tests for AuditAndTraceAgent."""
    
    def test_agent_attributes(self, audit_trace_agent):
        """Test agent has correct attributes from YAML."""
        agent = audit_trace_agent
        
        assert agent.agent_id == "AuditAndTraceAgent"
        assert "trace-logger" in agent.tools_used
        assert "completeness" in agent.evaluation_criteria
        assert "missing_steps" in agent.failure_modes
    
    @pytest.mark.asyncio
    async def test_complete_workflow_audit(self, audit_trace_agent):
        """Test audit of a complete workflow."""
        agent = audit_trace_agent
        
        # Simulate outputs from all expected agents
        now = datetime.now(timezone.utc)
        agent_outputs = [
            AgentOutputRecord(
                agent_id="HealthDataAnalysisAgent",
                execution_id=str(uuid4()),
                timestamp=now,
                execution_time_ms=150.0,
                success=True,
                input_summary="health_metrics, patient_profile",
                output_summary="5 risk indicators",
                key_decisions=["Identified elevated HR"],
            ),
            AgentOutputRecord(
                agent_id="DataQualityConfidenceAgent",
                execution_id=str(uuid4()),
                timestamp=now,
                execution_time_ms=50.0,
                success=True,
                input_summary="health_metrics",
                output_summary="confidence: 0.85",
                key_decisions=["Data quality: GOOD"],
            ),
            AgentOutputRecord(
                agent_id="PolicyRiskAgent",
                execution_id=str(uuid4()),
                timestamp=now,
                execution_time_ms=100.0,
                success=True,
                input_summary="risk_indicators, policy_rules",
                output_summary="risk: MODERATE, adjustment: 15%",
                key_decisions=["Premium increase 15%"],
            ),
            AgentOutputRecord(
                agent_id="BusinessRulesValidationAgent",
                execution_id=str(uuid4()),
                timestamp=now,
                execution_time_ms=30.0,
                success=True,
                input_summary="premium_adjustment",
                output_summary="approved: True",
                key_decisions=["Compliance verified"],
            ),
            AgentOutputRecord(
                agent_id="BiasAndFairnessAgent",
                execution_id=str(uuid4()),
                timestamp=now,
                execution_time_ms=40.0,
                success=True,
                input_summary="decision_context",
                output_summary="fairness: 0.95",
                key_decisions=["No bias detected"],
            ),
            AgentOutputRecord(
                agent_id="CommunicationAgent",
                execution_id=str(uuid4()),
                timestamp=now,
                execution_time_ms=80.0,
                success=True,
                input_summary="decision_summary",
                output_summary="messages generated",
                key_decisions=["Tone: positive"],
            ),
        ]
        
        input_data = {
            "agent_outputs": [o.model_dump() for o in agent_outputs],
            "patient_id": "PAT-001",
        }
        
        output = await agent.run(input_data)
        
        assert isinstance(output, AuditAndTraceOutput)
        assert output.audit_log.workflow_status == "COMPLETE"
        assert output.audit_log.integrity_verified is True
        assert len(output.audit_log.missing_steps) == 0
    
    @pytest.mark.asyncio
    async def test_incomplete_workflow_detected(self, audit_trace_agent):
        """Test that missing agents are detected."""
        agent = audit_trace_agent
        
        # Only include some agents
        now = datetime.now(timezone.utc)
        agent_outputs = [
            AgentOutputRecord(
                agent_id="HealthDataAnalysisAgent",
                execution_id=str(uuid4()),
                timestamp=now,
                execution_time_ms=150.0,
                success=True,
                input_summary="health_metrics",
                output_summary="indicators",
                key_decisions=[],
            ),
            # Missing: DataQualityConfidenceAgent, PolicyRiskAgent, etc.
        ]
        
        input_data = {
            "agent_outputs": [o.model_dump() for o in agent_outputs],
            "patient_id": "PAT-001",
        }
        
        output = await agent.run(input_data)
        
        assert output.audit_log.workflow_status == "INCOMPLETE"
        assert len(output.audit_log.missing_steps) > 0
        assert "DataQualityConfidenceAgent" in output.audit_log.missing_steps
    
    @pytest.mark.asyncio
    async def test_failed_agent_detected(self, audit_trace_agent):
        """Test that failed agent execution is detected."""
        agent = audit_trace_agent
        
        now = datetime.now(timezone.utc)
        agent_outputs = [
            AgentOutputRecord(
                agent_id="HealthDataAnalysisAgent",
                execution_id=str(uuid4()),
                timestamp=now,
                execution_time_ms=150.0,
                success=False,  # Failed
                input_summary="health_metrics",
                output_summary="error",
                key_decisions=[],
                errors=["Connection timeout"],
            ),
        ]
        
        input_data = {
            "agent_outputs": [o.model_dump() for o in agent_outputs],
            "patient_id": "PAT-001",
        }
        
        output = await agent.run(input_data)
        
        assert output.audit_log.workflow_status == "FAILED"
        assert output.audit_log.integrity_verified is False


# =============================================================================
# TEST: AGENT ISOLATION
# =============================================================================

class TestAgentIsolation:
    """Tests to verify agents don't contain orchestration logic."""
    
    def test_agents_have_no_agent_dependencies(self):
        """Verify no agent imports or calls other agents."""
        import inspect
        from app.agents import (
            HealthDataAnalysisAgent,
            PolicyRiskAgent,
            BusinessRulesValidationAgent,
            DataQualityConfidenceAgent,
            BiasAndFairnessAgent,
            CommunicationAgent,
            AuditAndTraceAgent,
        )
        
        agents = [
            HealthDataAnalysisAgent,
            PolicyRiskAgent,
            BusinessRulesValidationAgent,
            DataQualityConfidenceAgent,
            BiasAndFairnessAgent,
            CommunicationAgent,
            AuditAndTraceAgent,
        ]
        
        for agent_class in agents:
            source = inspect.getsourcefile(agent_class)
            with open(source, "r") as f:
                content = f.read()
            
            # Check that no agent imports other agents
            for other_agent in agents:
                if other_agent != agent_class:
                    agent_name = other_agent.__name__
                    # Allow imports in __init__.py but not in agent files
                    if "__init__" not in source:
                        assert f"from app.agents.{agent_name.lower()}" not in content, \
                            f"{agent_class.__name__} should not import {agent_name}"
    
    def test_no_orchestration_methods(self):
        """Verify agents don't have orchestration-like methods."""
        from app.agents import (
            HealthDataAnalysisAgent,
            PolicyRiskAgent,
            BusinessRulesValidationAgent,
            DataQualityConfidenceAgent,
            BiasAndFairnessAgent,
            CommunicationAgent,
            AuditAndTraceAgent,
        )
        
        orchestration_method_names = [
            "orchestrate",
            "coordinate",
            "run_workflow",
            "execute_pipeline",
            "call_next_agent",
            "invoke_agent",
        ]
        
        agents = [
            HealthDataAnalysisAgent(),
            PolicyRiskAgent(),
            BusinessRulesValidationAgent(),
            DataQualityConfidenceAgent(),
            BiasAndFairnessAgent(),
            CommunicationAgent(),
            AuditAndTraceAgent(),
        ]
        
        for agent in agents:
            for method_name in orchestration_method_names:
                assert not hasattr(agent, method_name), \
                    f"{agent.agent_id} should not have method {method_name}"
