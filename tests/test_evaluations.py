"""
Tests for Azure AI Foundry Evaluations Integration

Tests the evaluation service, agent evaluation, and workflow evaluation
using the azure-ai-evaluation SDK.

Run with: pytest tests/test_evaluations.py -v
"""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Set test environment before importing modules
os.environ["FOUNDRY_EVALUATIONS_ENABLED"] = "false"  # Start disabled
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com"
os.environ["AZURE_OPENAI_API_KEY"] = "test-key"
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = "gpt-4"


class TestEvaluationConfiguration:
    """Tests for evaluation configuration and feature flag."""
    
    def test_evaluation_disabled_by_default(self):
        """Evaluations should be disabled when env var is not set."""
        os.environ["FOUNDRY_EVALUATIONS_ENABLED"] = "false"
        
        from app.agents.evaluations import is_evaluations_enabled
        # Force reimport to pick up new env var
        import importlib
        import app.agents.evaluations as eval_module
        importlib.reload(eval_module)
        
        assert eval_module.is_evaluations_enabled() is False
    
    def test_evaluation_enabled_with_flag(self):
        """Evaluations should be enabled when env var is 'true'."""
        os.environ["FOUNDRY_EVALUATIONS_ENABLED"] = "true"
        
        import importlib
        import app.agents.evaluations as eval_module
        importlib.reload(eval_module)
        
        assert eval_module.is_evaluations_enabled() is True
        
        # Reset
        os.environ["FOUNDRY_EVALUATIONS_ENABLED"] = "false"


class TestAgentEvaluationResult:
    """Tests for AgentEvaluationResult schema."""
    
    def test_create_evaluation_result(self):
        """Should create evaluation result with metrics."""
        from app.agents.evaluations import (
            AgentEvaluationResult,
            MetricScore,
            EvaluationStatus,
        )
        
        result = AgentEvaluationResult(
            agent_id="HealthDataAnalysisAgent",
            status=EvaluationStatus.COMPLETED,
            metrics=[
                MetricScore(
                    metric_name="groundedness",
                    score=4.0,
                    threshold=3.0,
                    passed=True,
                    reason="Response is well grounded in context"
                ),
                MetricScore(
                    metric_name="coherence",
                    score=3.5,
                    threshold=3.0,
                    passed=True,
                    reason="Logical flow is good"
                ),
            ],
            aggregate_score=3.75,
            passed=True,
        )
        
        assert result.agent_id == "HealthDataAnalysisAgent"
        assert result.status == EvaluationStatus.COMPLETED
        assert len(result.metrics) == 2
        assert result.aggregate_score == 3.75
        assert result.passed is True
    
    def test_to_cosmos_format(self):
        """Should convert to Cosmos DB format."""
        from app.agents.evaluations import (
            AgentEvaluationResult,
            MetricScore,
            EvaluationStatus,
        )
        
        result = AgentEvaluationResult(
            agent_id="TestAgent",
            status=EvaluationStatus.COMPLETED,
            metrics=[
                MetricScore(metric_name="groundedness", score=4.0, threshold=3.0, passed=True),
                MetricScore(metric_name="relevance", score=3.5, threshold=3.0, passed=True),
            ],
            aggregate_score=3.75,
            passed=True,
            duration_ms=150.0,
        )
        
        cosmos_result = result.to_cosmos_format()
        
        assert cosmos_result.groundedness == 4.0
        assert cosmos_result.relevance == 3.5
        assert cosmos_result.custom_metrics["aggregate_score"] == 3.75
        assert cosmos_result.custom_metrics["passed"] is True


class TestWorkflowEvaluationResult:
    """Tests for WorkflowEvaluationResult schema."""
    
    def test_workflow_evaluation_aggregate(self):
        """Should aggregate agent evaluations."""
        from app.agents.evaluations import (
            AgentEvaluationResult,
            WorkflowEvaluationResult,
            MetricScore,
            EvaluationStatus,
        )
        
        agent1 = AgentEvaluationResult(
            agent_id="Agent1",
            status=EvaluationStatus.COMPLETED,
            aggregate_score=4.0,
            passed=True,
        )
        
        agent2 = AgentEvaluationResult(
            agent_id="Agent2",
            status=EvaluationStatus.COMPLETED,
            aggregate_score=3.5,
            passed=True,
        )
        
        workflow = WorkflowEvaluationResult(
            workflow_id="test-workflow-123",
            status=EvaluationStatus.COMPLETED,
            agent_evaluations={"Agent1": agent1, "Agent2": agent2},
            aggregate_score=3.75,
            overall_passed=True,
        )
        
        assert workflow.workflow_id == "test-workflow-123"
        assert len(workflow.agent_evaluations) == 2
        assert workflow.aggregate_score == 3.75


class TestFoundryEvaluatorService:
    """Tests for FoundryEvaluatorService."""
    
    def test_service_disabled(self):
        """Service should skip evaluations when disabled."""
        os.environ["FOUNDRY_EVALUATIONS_ENABLED"] = "false"
        
        import importlib
        import app.agents.evaluations as eval_module
        importlib.reload(eval_module)
        
        service = eval_module.FoundryEvaluatorService(enabled=False)
        
        assert service.enabled is False
    
    @pytest.mark.asyncio
    async def test_evaluate_agent_skipped_when_disabled(self):
        """Should skip evaluation when service is disabled."""
        from app.agents.evaluations import FoundryEvaluatorService, EvaluationStatus
        
        service = FoundryEvaluatorService(enabled=False)
        
        result = await service.evaluate_agent(
            agent_id="TestAgent",
            agent_input={"query": "test"},
            agent_output={"response": "test"},
        )
        
        assert result.status == EvaluationStatus.SKIPPED
        assert "disabled" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_evaluate_agent_no_config(self):
        """Should skip evaluation when no config for agent."""
        from app.agents.evaluations import FoundryEvaluatorService, EvaluationStatus
        
        service = FoundryEvaluatorService(enabled=True)
        # Mock initialization to pass
        service._initialized = True
        
        result = await service.evaluate_agent(
            agent_id="UnknownAgent",
            agent_input={"query": "test"},
            agent_output={"response": "test"},
        )
        
        assert result.status == EvaluationStatus.SKIPPED
        assert "no evaluation config" in result.error_message.lower()


class TestEvaluationIntegration:
    """Integration tests with mocked SDK."""
    
    @pytest.mark.asyncio
    async def test_evaluation_flow_with_mock_sdk(self):
        """Should run full evaluation flow with mocked SDK."""
        from app.agents.evaluations import (
            FoundryEvaluatorService,
            EvaluatorType,
            EvaluationStatus,
        )
        
        service = FoundryEvaluatorService(enabled=True)
        
        # Mock the evaluators
        mock_groundedness = MagicMock()
        mock_groundedness.return_value = {
            "groundedness": 4.0,
            "groundedness_result": "pass",
            "groundedness_reason": "Well grounded",
        }
        
        mock_coherence = MagicMock()
        mock_coherence.return_value = {
            "coherence": 3.5,
            "coherence_result": "pass",
            "coherence_reason": "Good coherence",
        }
        
        service._evaluators = {
            EvaluatorType.GROUNDEDNESS: mock_groundedness,
            EvaluatorType.COHERENCE: mock_coherence,
        }
        service._initialized = True
        service._model_config = {"test": "config"}
        
        result = await service.evaluate_agent(
            agent_id="HealthDataAnalysisAgent",
            agent_input={
                "document_context": "Patient is 45 years old with diabetes",
                "health_metrics": {"bmi": 28.5},
            },
            agent_output={
                "health_summary": "Patient shows moderate health risk",
                "risk_indicators": [{"name": "diabetes", "level": "moderate"}],
            },
            context="Patient is 45 years old with diabetes",
        )
        
        assert result.status == EvaluationStatus.COMPLETED
        assert len(result.metrics) >= 1
        assert result.aggregate_score is not None


class TestAgentEvaluationConfigs:
    """Tests for agent evaluation configurations."""
    
    def test_health_data_analysis_config(self):
        """HealthDataAnalysisAgent should have proper config."""
        from app.agents.evaluations import (
            AGENT_EVALUATION_CONFIGS,
            EvaluatorType,
        )
        
        config = AGENT_EVALUATION_CONFIGS.get("HealthDataAnalysisAgent")
        
        assert config is not None
        assert config.agent_id == "HealthDataAnalysisAgent"
        assert EvaluatorType.GROUNDEDNESS in config.evaluators
        assert EvaluatorType.COHERENCE in config.evaluators
        assert config.query_field == "document_context"
        assert config.response_field == "health_summary"
    
    def test_business_rules_config(self):
        """BusinessRulesValidationAgent should have proper config."""
        from app.agents.evaluations import (
            AGENT_EVALUATION_CONFIGS,
            EvaluatorType,
        )
        
        config = AGENT_EVALUATION_CONFIGS.get("BusinessRulesValidationAgent")
        
        assert config is not None
        assert EvaluatorType.COHERENCE in config.evaluators
        assert EvaluatorType.RELEVANCE in config.evaluators
    
    def test_communication_config(self):
        """CommunicationAgent should have proper config."""
        from app.agents.evaluations import (
            AGENT_EVALUATION_CONFIGS,
            EvaluatorType,
        )
        
        config = AGENT_EVALUATION_CONFIGS.get("CommunicationAgent")
        
        assert config is not None
        assert EvaluatorType.FLUENCY in config.evaluators
        assert EvaluatorType.COHERENCE in config.evaluators


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
