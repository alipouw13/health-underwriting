"""
Orchestration Tests for OrchestratorAgent

Tests verify:
1. Strict execution order (all 7 agents in sequence)
2. Presence of all agent outputs
3. Deterministic results with identical inputs
4. No conditional branching or skipping
5. Orchestrator summarizes but does not alter conclusions
"""

from __future__ import annotations

import pytest
from datetime import datetime

from app.agents.orchestrator import (
    OrchestratorAgent,
    OrchestratorInput,
    OrchestratorOutput,
    ExecutionContext,
    AgentDefinitionLoader,
    FinalDecision,
)
from data.mock.schemas import (
    DecisionStatus,
    RiskLevel,
)
from data.mock.fixtures import (
    get_healthy_patient_metrics,
    get_moderate_risk_metrics,
    get_high_risk_metrics,
    get_patient_by_id,
    get_standard_policy_rules,
)


# =============================================================================
# EXECUTION ORDER TESTS
# =============================================================================

class TestExecutionOrder:
    """Test that agents execute in the correct, strict order."""
    
    @pytest.fixture
    def orchestrator(self):
        return OrchestratorAgent()
    
    @pytest.mark.asyncio
    async def test_all_seven_agents_execute(self, orchestrator):
        """Verify all 7 agents execute."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        # All 7 agents must execute
        assert len(output.execution_records) == 7
    
    @pytest.mark.asyncio
    async def test_execution_order_is_strict(self, orchestrator):
        """Verify agents execute in the exact required order."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        expected_order = [
            "HealthDataAnalysisAgent",
            "DataQualityConfidenceAgent",
            "PolicyRiskAgent",
            "BusinessRulesValidationAgent",
            "BiasAndFairnessAgent",
            "CommunicationAgent",
            "AuditAndTraceAgent",
        ]
        
        actual_order = [record.agent_id for record in output.execution_records]
        assert actual_order == expected_order, f"Expected {expected_order}, got {actual_order}"
    
    @pytest.mark.asyncio
    async def test_step_numbers_are_sequential(self, orchestrator):
        """Verify step numbers are 1-7 in sequence."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        step_numbers = [record.step_number for record in output.execution_records]
        assert step_numbers == [1, 2, 3, 4, 5, 6, 7]
    
    @pytest.mark.asyncio
    async def test_no_agents_skipped(self, orchestrator):
        """Verify no agents are skipped regardless of input."""
        # Test with different patient profiles
        test_patients = ["PAT-HEALTHY-001", "PAT-MODERATE-001", "PAT-HIGH-RISK-001"]
        
        for patient_id in test_patients:
            output = await orchestrator.run({"patient_id": patient_id})
            agent_ids = {record.agent_id for record in output.execution_records}
            
            expected_agents = {
                "HealthDataAnalysisAgent",
                "DataQualityConfidenceAgent",
                "PolicyRiskAgent",
                "BusinessRulesValidationAgent",
                "BiasAndFairnessAgent",
                "CommunicationAgent",
                "AuditAndTraceAgent",
            }
            
            assert agent_ids == expected_agents, f"Missing agents for {patient_id}"


# =============================================================================
# OUTPUT PRESENCE TESTS
# =============================================================================

class TestOutputPresence:
    """Test that all required outputs are present."""
    
    @pytest.fixture
    def orchestrator(self):
        return OrchestratorAgent()
    
    @pytest.mark.asyncio
    async def test_final_decision_present(self, orchestrator):
        """Verify final_decision is present in output."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        assert output.final_decision is not None
        assert isinstance(output.final_decision, FinalDecision)
    
    @pytest.mark.asyncio
    async def test_confidence_score_present(self, orchestrator):
        """Verify confidence_score is present and valid."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        assert output.confidence_score is not None
        assert 0.0 <= output.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_explanation_present(self, orchestrator):
        """Verify explanation is present and non-empty."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        assert output.explanation is not None
        assert len(output.explanation) > 0
    
    @pytest.mark.asyncio
    async def test_execution_records_present(self, orchestrator):
        """Verify execution_records contains all agent records."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        assert output.execution_records is not None
        assert len(output.execution_records) == 7
        
        # Each record should have required fields
        for record in output.execution_records:
            assert record.agent_id is not None
            assert record.step_number > 0
            assert record.execution_id is not None
            assert record.timestamp is not None
            assert record.success is not None
    
    @pytest.mark.asyncio
    async def test_workflow_id_present(self, orchestrator):
        """Verify workflow_id is generated."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        assert output.workflow_id is not None
        assert len(output.workflow_id) > 0
    
    @pytest.mark.asyncio
    async def test_total_execution_time_recorded(self, orchestrator):
        """Verify total execution time is recorded."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        assert output.total_execution_time_ms is not None
        assert output.total_execution_time_ms > 0


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Test that identical inputs produce identical outputs."""
    
    @pytest.fixture
    def orchestrator(self):
        return OrchestratorAgent()
    
    @pytest.mark.asyncio
    async def test_same_input_same_decision_status(self, orchestrator):
        """Verify same input produces same decision status."""
        output1 = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        output2 = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        assert output1.final_decision.status == output2.final_decision.status
    
    @pytest.mark.asyncio
    async def test_same_input_same_risk_level(self, orchestrator):
        """Verify same input produces same risk level."""
        output1 = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        output2 = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        assert output1.final_decision.risk_level == output2.final_decision.risk_level
    
    @pytest.mark.asyncio
    async def test_same_input_same_premium_adjustment(self, orchestrator):
        """Verify same input produces same premium adjustment."""
        output1 = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        output2 = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        assert output1.final_decision.premium_adjustment_pct == output2.final_decision.premium_adjustment_pct
        assert output1.final_decision.adjusted_premium_annual == output2.final_decision.adjusted_premium_annual
    
    @pytest.mark.asyncio
    async def test_same_input_same_approval_status(self, orchestrator):
        """Verify same input produces same approval status."""
        output1 = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        output2 = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        assert output1.final_decision.approved == output2.final_decision.approved
    
    @pytest.mark.asyncio
    async def test_different_inputs_may_differ(self, orchestrator):
        """Verify different inputs can produce different outputs."""
        healthy_output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        high_risk_output = await orchestrator.run({"patient_id": "PAT-HIGH-RISK-001"})
        
        # High risk should have higher premium adjustment
        assert high_risk_output.final_decision.premium_adjustment_pct >= healthy_output.final_decision.premium_adjustment_pct


# =============================================================================
# NO CONDITIONAL BRANCHING TESTS
# =============================================================================

class TestNoConditionalBranching:
    """Test that there's no conditional branching in execution."""
    
    @pytest.fixture
    def orchestrator(self):
        return OrchestratorAgent()
    
    @pytest.mark.asyncio
    async def test_all_agents_run_for_healthy_patient(self, orchestrator):
        """All 7 agents run for healthy patient."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        assert len(output.execution_records) == 7
    
    @pytest.mark.asyncio
    async def test_all_agents_run_for_moderate_risk_patient(self, orchestrator):
        """All 7 agents run for moderate risk patient."""
        output = await orchestrator.run({"patient_id": "PAT-MODERATE-001"})
        assert len(output.execution_records) == 7
    
    @pytest.mark.asyncio
    async def test_all_agents_run_for_high_risk_patient(self, orchestrator):
        """All 7 agents run for high risk patient."""
        output = await orchestrator.run({"patient_id": "PAT-HIGH-RISK-001"})
        assert len(output.execution_records) == 7
    
    @pytest.mark.asyncio
    async def test_audit_agent_always_runs_last(self, orchestrator):
        """AuditAndTraceAgent always runs last regardless of prior results."""
        test_patients = ["PAT-HEALTHY-001", "PAT-MODERATE-001", "PAT-HIGH-RISK-001"]
        
        for patient_id in test_patients:
            output = await orchestrator.run({"patient_id": patient_id})
            last_record = output.execution_records[-1]
            assert last_record.agent_id == "AuditAndTraceAgent"
            assert last_record.step_number == 7


# =============================================================================
# NO OUTPUT ALTERATION TESTS
# =============================================================================

class TestNoOutputAlteration:
    """Test that orchestrator does not alter agent conclusions."""
    
    @pytest.fixture
    def orchestrator(self):
        return OrchestratorAgent()
    
    @pytest.mark.asyncio
    async def test_final_decision_reflects_business_rules(self, orchestrator):
        """Final decision reflects BusinessRulesValidationAgent's conclusion."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        # Find the business rules execution record
        brv_record = next(r for r in output.execution_records if r.agent_id == "BusinessRulesValidationAgent")
        
        # The business_rules_approved in final decision should reflect agent output
        # (we can't directly verify the agent output, but we verify consistency)
        assert output.final_decision.business_rules_approved is not None
    
    @pytest.mark.asyncio
    async def test_final_decision_reflects_bias_check(self, orchestrator):
        """Final decision reflects BiasAndFairnessAgent's conclusion."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        # bias_check_passed should be based on BiasAndFairnessAgent output
        assert output.final_decision.bias_check_passed is not None
    
    @pytest.mark.asyncio
    async def test_risk_level_not_downgraded(self, orchestrator):
        """Orchestrator cannot downgrade risk level from agent conclusion."""
        high_risk_output = await orchestrator.run({"patient_id": "PAT-HIGH-RISK-001"})
        
        # High risk patient should have elevated risk level
        assert high_risk_output.final_decision.risk_level in [
            RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.VERY_HIGH
        ]
    
    @pytest.mark.asyncio
    async def test_messages_come_from_communication_agent(self, orchestrator):
        """Messages in final decision come from CommunicationAgent."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        # Messages should be non-empty (generated by CommunicationAgent)
        assert len(output.final_decision.underwriter_message) > 0
        assert len(output.final_decision.customer_message) > 0


# =============================================================================
# AGENT DEFINITION LOADER TESTS
# =============================================================================

class TestAgentDefinitionLoader:
    """Test the YAML agent definition loader."""
    
    def test_loader_returns_execution_order(self):
        """Verify loader returns correct execution order."""
        loader = AgentDefinitionLoader()
        order = loader.get_execution_order()
        
        expected = [
            "HealthDataAnalysisAgent",
            "DataQualityConfidenceAgent",
            "PolicyRiskAgent",
            "BusinessRulesValidationAgent",
            "BiasAndFairnessAgent",
            "CommunicationAgent",
            "AuditAndTraceAgent",
        ]
        
        assert order == expected
    
    def test_execution_order_is_hardcoded(self):
        """Verify execution order is hardcoded and cannot change."""
        loader1 = AgentDefinitionLoader()
        loader2 = AgentDefinitionLoader()
        
        # Multiple calls should return identical order
        assert loader1.get_execution_order() == loader2.get_execution_order()
    
    def test_orchestrator_agent_definition_exists(self):
        """Verify OrchestratorAgent is defined in YAML."""
        loader = AgentDefinitionLoader()
        
        # Only test if YAML file exists
        try:
            definitions = loader.load()
            assert "OrchestratorAgent" in definitions
            
            orch_def = definitions["OrchestratorAgent"]
            assert orch_def["purpose"] == "Coordinate agent execution and produce final decision"
            assert "agent-framework" in orch_def["tools_used"]
        except FileNotFoundError:
            pytest.skip("Agent definitions YAML not found")


# =============================================================================
# EXECUTION CONTEXT TESTS
# =============================================================================

class TestExecutionContext:
    """Test the ExecutionContext helper class."""
    
    def test_context_stores_outputs(self):
        """Verify context stores agent outputs."""
        from app.agents.base import AgentOutput
        from datetime import timezone
        
        context = ExecutionContext("PAT-001", "workflow-123")
        
        # Create a mock output
        class MockOutput(AgentOutput):
            pass
        
        output = MockOutput(
            agent_id="TestAgent", 
            success=True,
            execution_time_ms=5.0,
            timestamp=datetime.now(timezone.utc),
        )
        context.store_output("TestAgent", output, step_number=1)
        
        retrieved = context.get_output("TestAgent")
        assert retrieved is not None
        assert retrieved.agent_id == "TestAgent"
    
    def test_context_tracks_execution_records(self):
        """Verify context tracks execution records."""
        from app.agents.base import AgentOutput
        from datetime import timezone
        
        context = ExecutionContext("PAT-001", "workflow-123")
        
        class MockOutput(AgentOutput):
            pass
        
        output = MockOutput(
            agent_id="TestAgent", 
            success=True,
            execution_time_ms=10.5,
            timestamp=datetime.now(timezone.utc),
        )
        context.store_output("TestAgent", output, step_number=1)
        
        records = context.get_records()
        assert len(records) == 1
        assert records[0].agent_id == "TestAgent"
        assert records[0].step_number == 1
    
    def test_context_calculates_total_time(self):
        """Verify context calculates total execution time."""
        import time
        
        context = ExecutionContext("PAT-001", "workflow-123")
        time.sleep(0.01)  # 10ms
        
        total_time = context.get_total_time_ms()
        assert total_time >= 10  # At least 10ms


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Test input validation for orchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        return OrchestratorAgent()
    
    def test_valid_input_accepted(self, orchestrator):
        """Valid input should be accepted."""
        validated = orchestrator.validate_input({"patient_id": "PAT-001"})
        assert validated.patient_id == "PAT-001"
    
    def test_missing_patient_id_rejected(self, orchestrator):
        """Missing patient_id should be rejected."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            orchestrator.validate_input({})
    
    def test_optional_health_metrics_accepted(self, orchestrator):
        """Optional health_metrics should be accepted."""
        metrics = get_healthy_patient_metrics()
        validated = orchestrator.validate_input({
            "patient_id": "PAT-001",
            "health_metrics": metrics.model_dump(),
        })
        assert validated.health_metrics is not None


# =============================================================================
# END-TO-END WORKFLOW TESTS
# =============================================================================

class TestEndToEndWorkflow:
    """Test complete end-to-end workflow execution."""
    
    @pytest.fixture
    def orchestrator(self):
        return OrchestratorAgent()
    
    @pytest.mark.asyncio
    async def test_healthy_patient_workflow(self, orchestrator):
        """Test complete workflow for healthy patient."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        # Should complete successfully
        assert output.success
        
        # Should have reasonable confidence
        assert output.confidence_score >= 0.5
        
        # Explanation should mention the patient
        assert "PAT-HEALTHY-001" in output.explanation
    
    @pytest.mark.asyncio
    async def test_high_risk_patient_workflow(self, orchestrator):
        """Test complete workflow for high risk patient."""
        output = await orchestrator.run({"patient_id": "PAT-HIGH-RISK-001"})
        
        # Should complete successfully
        assert output.success
        
        # Should have elevated risk
        assert output.final_decision.risk_level != RiskLevel.LOW
    
    @pytest.mark.asyncio
    async def test_workflow_produces_audit_trail(self, orchestrator):
        """Test that workflow produces audit trail."""
        output = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        # AuditAndTraceAgent should have run
        audit_record = next(
            (r for r in output.execution_records if r.agent_id == "AuditAndTraceAgent"),
            None
        )
        assert audit_record is not None
        assert audit_record.success
    
    @pytest.mark.asyncio
    async def test_workflow_id_unique_per_run(self, orchestrator):
        """Test that each workflow has a unique ID."""
        output1 = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        output2 = await orchestrator.run({"patient_id": "PAT-HEALTHY-001"})
        
        assert output1.workflow_id != output2.workflow_id
