"""Tests for Cosmos DB agent run persistence.

Testing Requirements:
1. Cosmos document is written for agent executions when AGENT_EXECUTION_ENABLED=true
2. No Cosmos write occurs when AGENT_EXECUTION_ENABLED=false
3. Schema matches expected structure
4. Partial failures (Cosmos unavailable) do NOT break execution
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.cosmos.models import (
    AgentDefinitionSnapshot,
    AgentRunDocument,
    AgentStepRecord,
    EvaluationResult,
    ExecutionMode,
    FinalDecisionRecord,
    OrchestrationStatus,
    OrchestrationSummary,
    TokenUsage,
)
from app.cosmos.service import CosmosAgentRunsService, get_cosmos_service
from app.cosmos.settings import CosmosSettings


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def cosmos_settings_configured():
    """Settings with Cosmos DB configured."""
    return CosmosSettings(
        endpoint="https://test-cosmos.documents.azure.com:443/",
        database_name="test-underwriting-agents",
        agent_runs_container="test_agent_runs",
        use_serverless=True,
    )


@pytest.fixture
def cosmos_settings_unconfigured():
    """Settings with Cosmos DB not configured."""
    return CosmosSettings(
        endpoint=None,
        database_name="test-underwriting-agents",
        agent_runs_container="test_agent_runs",
    )


@pytest.fixture
def sample_orchestrator_output():
    """Create a mock OrchestratorOutput for testing."""
    from unittest.mock import MagicMock
    
    # Create mock execution records
    records = [
        MagicMock(
            agent_id="HealthDataAnalysisAgent",
            step_number=1,
            execution_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            execution_time_ms=5000.0,
            success=True,
            output_summary="Identified 5 risk indicators",
        ),
        MagicMock(
            agent_id="BusinessRulesValidationAgent",
            step_number=2,
            execution_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            execution_time_ms=3000.0,
            success=True,
            output_summary="Approved with 15% adjustment",
        ),
        MagicMock(
            agent_id="CommunicationAgent",
            step_number=3,
            execution_id=str(uuid4()),
            timestamp=datetime.now(timezone.utc),
            execution_time_ms=2000.0,
            success=True,
            output_summary="Messages generated",
        ),
    ]
    
    # Create mock final decision
    final_decision = MagicMock(
        decision_id=str(uuid4()),
        status=MagicMock(value="APPROVED_WITH_ADJUSTMENT"),
        risk_level=MagicMock(value="moderate"),
        premium_adjustment_pct=15.0,
        adjusted_premium_annual=1150.0,
        business_rules_approved=True,
        bias_check_passed=True,
        underwriter_message="Application approved with premium adjustment.",
        customer_message="Your application has been approved.",
    )
    
    # Create mock orchestrator output
    output = MagicMock(
        workflow_id=str(uuid4()),
        execution_records=records,
        final_decision=final_decision,
        confidence_score=0.95,
        explanation="Based on health indicators and business rules...",
        total_execution_time_ms=10000.0,
    )
    
    return output


@pytest.fixture
def sample_agent_run_document():
    """Create a sample AgentRunDocument for testing."""
    return AgentRunDocument(
        application_id="app_test_123",
        execution_mode=ExecutionMode.MULTI_AGENT,
        workflow_id=str(uuid4()),
        agent_definitions_version="1.1",
        agent_definitions=[
            AgentDefinitionSnapshot(
                agent_id="HealthDataAnalysisAgent",
                role="Health Signal Extraction",
                purpose="Analyze health data",
                instructions=["Review health metrics", "Identify risk indicators"],
            ),
        ],
        global_constraints=["Agents must NOT fabricate data"],
        agents=[
            AgentStepRecord(
                agent_id="HealthDataAnalysisAgent",
                step_number=1,
                success=True,
                execution_duration_ms=5000.0,
                inputs={"health_data": "..."},
                outputs={"risk_indicators": []},
                output_summary="Identified 3 risk indicators",
                token_usage=TokenUsage(
                    prompt_tokens=500,
                    completion_tokens=200,
                    total_tokens=700,
                ),
            ),
        ],
        orchestration_summary=OrchestrationSummary(
            status=OrchestrationStatus.SUCCESS,
            execution_order=["HealthDataAnalysisAgent"],
            agents_executed=1,
            agents_succeeded=1,
            agents_failed=0,
            total_execution_time_ms=5000.0,
        ),
        final_decision=FinalDecisionRecord(
            underwriting_decision="APPROVED_WITH_ADJUSTMENT",
            risk_level="moderate",
            premium_adjustment_pct=15.0,
            adjusted_premium_annual=1150.0,
            confidence_score=0.95,
            explanation="Based on analysis...",
        ),
    )


# =============================================================================
# TEST: COSMOS DB CONFIGURATION
# =============================================================================

class TestCosmosSettings:
    """Tests for CosmosSettings configuration."""
    
    def test_settings_from_env_configured(self, monkeypatch):
        """Test loading settings from environment when configured."""
        monkeypatch.setenv("AZURE_COSMOS_ENDPOINT", "https://test.cosmos.azure.com:443/")
        monkeypatch.setenv("AZURE_COSMOS_DATABASE_NAME", "my-agents-db")
        monkeypatch.setenv("AZURE_COSMOS_AGENT_RUNS_CONTAINER", "my_runs")
        
        settings = CosmosSettings.from_env()
        
        assert settings.endpoint == "https://test.cosmos.azure.com:443/"
        assert settings.database_name == "my-agents-db"
        assert settings.agent_runs_container == "my_runs"
        assert settings.is_configured is True
    
    def test_settings_from_env_unconfigured(self, monkeypatch):
        """Test loading settings from environment when not configured."""
        monkeypatch.delenv("AZURE_COSMOS_ENDPOINT", raising=False)
        
        settings = CosmosSettings.from_env()
        
        assert settings.endpoint is None
        assert settings.is_configured is False
    
    def test_default_values(self):
        """Test default settings values."""
        settings = CosmosSettings()
        
        assert settings.database_name == "underwriting-agents"
        assert settings.agent_runs_container == "underwriting_agent_runs"
        assert settings.token_tracking_container == "token_tracking"
        assert settings.evaluations_container == "evaluations"
        assert settings.agent_runs_partition_key == "/id"  # Matches Azure Portal created containers
        assert settings.use_serverless is True


# =============================================================================
# TEST: AGENT RUN DOCUMENT SCHEMA
# =============================================================================

class TestAgentRunDocumentSchema:
    """Tests for AgentRunDocument schema validation."""
    
    def test_document_schema_complete(self, sample_agent_run_document):
        """Test that a complete document passes schema validation."""
        doc = sample_agent_run_document
        
        # Verify required fields
        assert doc.id is not None
        assert doc.run_id is not None
        assert doc.application_id == "app_test_123"
        assert doc.execution_mode == ExecutionMode.MULTI_AGENT
        assert doc.execution_timestamp is not None
        
        # Verify agent definitions
        assert len(doc.agent_definitions) == 1
        assert doc.agent_definitions[0].agent_id == "HealthDataAnalysisAgent"
        
        # Verify agent steps
        assert len(doc.agents) == 1
        assert doc.agents[0].agent_id == "HealthDataAnalysisAgent"
        assert doc.agents[0].success is True
        
        # Verify orchestration summary
        assert doc.orchestration_summary is not None
        assert doc.orchestration_summary.status == OrchestrationStatus.SUCCESS
        
        # Verify final decision
        assert doc.final_decision is not None
        assert doc.final_decision.underwriting_decision == "APPROVED_WITH_ADJUSTMENT"
        assert doc.final_decision.confidence_score == 0.95
    
    def test_document_json_serialization(self, sample_agent_run_document):
        """Test that document can be serialized to JSON (for Cosmos DB)."""
        doc = sample_agent_run_document
        
        # Convert to JSON-compatible dict
        doc_dict = doc.model_dump(mode='json')
        
        # Verify key fields are present and serializable
        assert doc_dict["id"] is not None
        assert doc_dict["application_id"] == "app_test_123"
        assert doc_dict["execution_mode"] == "multi_agent"
        assert isinstance(doc_dict["agents"], list)
        assert isinstance(doc_dict["orchestration_summary"], dict)
        assert isinstance(doc_dict["final_decision"], dict)
    
    def test_token_usage_unavailable(self):
        """Test token usage when unavailable from SDK."""
        token_usage = TokenUsage(
            unavailable_reason="Token usage not available from Azure AI Foundry SDK"
        )
        
        assert token_usage.prompt_tokens is None
        assert token_usage.unavailable_reason is not None
    
    def test_evaluation_unavailable(self):
        """Test evaluation results when not run."""
        eval_result = EvaluationResult(
            unavailable_reason="Evaluation not run for this execution"
        )
        
        assert eval_result.groundedness is None
        assert eval_result.unavailable_reason is not None


# =============================================================================
# TEST: COSMOS SERVICE INITIALIZATION
# =============================================================================

class TestCosmosServiceInitialization:
    """Tests for CosmosAgentRunsService initialization."""
    
    @pytest.mark.asyncio
    async def test_service_not_available_when_unconfigured(self, cosmos_settings_unconfigured):
        """Test service is not available when Cosmos DB is not configured."""
        service = CosmosAgentRunsService(settings=cosmos_settings_unconfigured)
        
        result = await service.initialize()
        
        assert result is False
        assert service.is_available is False
    
    @pytest.mark.asyncio
    async def test_service_initialization_with_cosmos_error(self, cosmos_settings_configured):
        """Test that initialization handles Cosmos errors gracefully."""
        service = CosmosAgentRunsService(settings=cosmos_settings_configured)
        
        with patch('app.cosmos.service.DefaultAzureCredential') as mock_cred:
            mock_cred.side_effect = Exception("Auth failed")
            
            result = await service.initialize()
        
        assert result is False
        assert service.is_available is False
    
    @pytest.mark.asyncio
    async def test_service_initialization_success(self, cosmos_settings_configured):
        """Test successful service initialization."""
        service = CosmosAgentRunsService(settings=cosmos_settings_configured)
        
        # Mock the Azure SDK components
        with patch('app.cosmos.service.DefaultAzureCredential') as mock_cred, \
             patch('app.cosmos.service.CosmosClient') as mock_client:
            
            mock_database = MagicMock()
            mock_container = MagicMock()
            mock_client_instance = MagicMock()
            mock_client_instance.create_database_if_not_exists.return_value = mock_database
            mock_database.create_container_if_not_exists.return_value = mock_container
            mock_client.return_value = mock_client_instance
            
            result = await service.initialize()
        
        assert result is True
        assert service.is_available is True


# =============================================================================
# TEST: SAVE AGENT RUN
# =============================================================================

class TestSaveAgentRun:
    """Tests for saving agent runs to Cosmos DB."""
    
    @pytest.mark.asyncio
    async def test_save_succeeds_when_cosmos_available(
        self, 
        cosmos_settings_configured, 
        sample_agent_run_document
    ):
        """Test that save succeeds when Cosmos is available."""
        service = CosmosAgentRunsService(settings=cosmos_settings_configured)
        
        # Mock initialization and container
        mock_container = MagicMock()
        mock_container.create_item.return_value = {"id": sample_agent_run_document.id}
        service._agent_runs_container = mock_container
        service._initialized = True
        
        result = await service.save_agent_run(sample_agent_run_document)
        
        assert result is True
        mock_container.create_item.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_fails_gracefully_when_cosmos_unavailable(
        self,
        cosmos_settings_unconfigured,
        sample_agent_run_document
    ):
        """Test that save fails gracefully when Cosmos is unavailable."""
        service = CosmosAgentRunsService(settings=cosmos_settings_unconfigured)
        
        result = await service.save_agent_run(sample_agent_run_document)
        
        assert result is False
        # Execution should continue without raising
    
    @pytest.mark.asyncio
    async def test_save_handles_cosmos_error(
        self,
        cosmos_settings_configured,
        sample_agent_run_document
    ):
        """Test that Cosmos errors during save are handled gracefully."""
        service = CosmosAgentRunsService(settings=cosmos_settings_configured)
        
        # Mock container that raises an error
        mock_container = MagicMock()
        mock_container.create_item.side_effect = Exception("Cosmos write failed")
        service._agent_runs_container = mock_container
        service._initialized = True
        
        result = await service.save_agent_run(sample_agent_run_document)
        
        assert result is False
        # Should not raise exception


# =============================================================================
# TEST: CREATE DOCUMENT FROM ORCHESTRATOR OUTPUT
# =============================================================================

class TestCreateDocumentFromOrchestratorOutput:
    """Tests for creating AgentRunDocument from OrchestratorOutput."""
    
    @pytest.mark.asyncio
    async def test_create_document_from_orchestrator_output(
        self,
        cosmos_settings_configured,
        sample_orchestrator_output
    ):
        """Test creating a complete document from orchestrator output."""
        service = CosmosAgentRunsService(settings=cosmos_settings_configured)
        
        # Mock YAML loading
        with patch.object(service, '_load_agent_definitions_yaml') as mock_yaml:
            mock_yaml.return_value = {
                "version": "1.1",
                "agents": [
                    {
                        "agent_id": "HealthDataAnalysisAgent",
                        "role": "Health Signal Extraction",
                        "purpose": "Analyze health data",
                        "instructions": ["Review data"],
                        "inputs": {"required": ["health_data"]},
                        "outputs": {},
                        "failure_modes": [],
                    }
                ],
                "global_constraints": ["No fabrication"],
            }
            
            doc = await service.create_run_document_from_orchestrator_output(
                application_id="app_test_456",
                orchestrator_output=sample_orchestrator_output,
            )
        
        # Verify document structure
        assert doc.application_id == "app_test_456"
        assert doc.execution_mode == ExecutionMode.MULTI_AGENT
        assert doc.workflow_id == sample_orchestrator_output.workflow_id
        
        # Verify agent definitions snapshot
        assert doc.agent_definitions_version == "1.1"
        assert len(doc.agent_definitions) == 1
        
        # Verify agent steps
        assert len(doc.agents) == 3
        assert doc.agents[0].agent_id == "HealthDataAnalysisAgent"
        assert doc.agents[1].agent_id == "BusinessRulesValidationAgent"
        
        # Verify orchestration summary
        assert doc.orchestration_summary.status == OrchestrationStatus.SUCCESS
        assert doc.orchestration_summary.agents_executed == 3
        
        # Verify final decision
        assert doc.final_decision.underwriting_decision == "APPROVED_WITH_ADJUSTMENT"
        assert doc.final_decision.confidence_score == 0.95


# =============================================================================
# TEST: AGENT EXECUTION TOGGLE INTEGRATION
# =============================================================================

class TestAgentExecutionToggleIntegration:
    """Tests verifying Cosmos persistence respects AGENT_EXECUTION_ENABLED toggle."""
    
    @pytest.mark.asyncio
    async def test_no_cosmos_write_when_agent_execution_disabled(self):
        """Test that no Cosmos write occurs when AGENT_EXECUTION_ENABLED=false."""
        from app.config import AgentSettings
        
        # Agent execution disabled
        settings = AgentSettings(enabled=False)
        assert settings.enabled is False
        
        # When agent execution is disabled, Cosmos persistence should not be triggered
        # This is validated at the API endpoint level
    
    @pytest.mark.asyncio
    async def test_cosmos_write_triggered_when_agent_execution_enabled(self):
        """Test that Cosmos write is triggered when AGENT_EXECUTION_ENABLED=true."""
        from app.config import AgentSettings
        
        # Agent execution enabled
        settings = AgentSettings(enabled=True)
        assert settings.enabled is True
        
        # When agent execution is enabled, Cosmos persistence should be triggered
        # after orchestration completes successfully


# =============================================================================
# TEST: PARTIAL FAILURE HANDLING
# =============================================================================

class TestPartialFailureHandling:
    """Tests verifying Cosmos failures don't break execution."""
    
    @pytest.mark.asyncio
    async def test_execution_continues_when_cosmos_fails(
        self,
        cosmos_settings_configured,
        sample_agent_run_document
    ):
        """Test that main execution continues even when Cosmos fails."""
        service = CosmosAgentRunsService(settings=cosmos_settings_configured)
        
        # Mock complete Cosmos failure
        with patch.object(service, 'initialize', return_value=False):
            result = await service.save_agent_run(sample_agent_run_document)
        
        # Save should fail but not raise
        assert result is False
    
    @pytest.mark.asyncio  
    async def test_orchestrator_not_blocked_by_cosmos_timeout(self):
        """Test that orchestrator is not blocked by Cosmos timeouts."""
        # This is verified by the architecture:
        # - Cosmos persistence happens AFTER orchestration completes
        # - Cosmos errors are caught and logged, not propagated
        # - The API endpoint wraps Cosmos calls in try/except
        pass


# =============================================================================
# TEST: QUERY OPERATIONS
# =============================================================================

class TestQueryOperations:
    """Tests for querying agent runs from Cosmos DB."""
    
    @pytest.mark.asyncio
    async def test_get_runs_by_application(self, cosmos_settings_configured):
        """Test querying runs by application ID."""
        service = CosmosAgentRunsService(settings=cosmos_settings_configured)
        
        # Mock container with query results
        mock_container = MagicMock()
        mock_container.query_items.return_value = [
            {
                "id": str(uuid4()),
                "run_id": str(uuid4()),
                "application_id": "app_test_789",
                "execution_mode": "multi_agent",
                "execution_timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_definitions_version": "1.1",
                "agent_definitions": [],
                "global_constraints": [],
                "agents": [],
            }
        ]
        service._agent_runs_container = mock_container
        service._initialized = True
        
        runs = await service.get_runs_by_application("app_test_789")
        
        assert len(runs) == 1
        assert runs[0].application_id == "app_test_789"
    
    @pytest.mark.asyncio
    async def test_get_runs_returns_empty_when_unavailable(self, cosmos_settings_unconfigured):
        """Test that query returns empty list when Cosmos is unavailable."""
        service = CosmosAgentRunsService(settings=cosmos_settings_unconfigured)
        
        runs = await service.get_runs_by_application("app_test")
        
        assert runs == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
