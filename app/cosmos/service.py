"""Cosmos DB service for persisting agent execution runs.

This service handles all Cosmos DB operations for the underwriting_agent_runs container.
Following Azure-native, append-only logging patterns from:
https://github.com/alipouw13/insurance-multi-agent

Key design principles:
- Writes are append-only and immutable
- Never blocks main execution path
- Fails gracefully with logging (no crash on Cosmos errors)
- Separate persistence service from agent logic
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.cosmos.container import ContainerProxy
from azure.identity import DefaultAzureCredential

from .models import (
    AgentDefinitionSnapshot,
    AgentRunDocument,
    AgentStepRecord,
    EvaluationDocument,
    EvaluationResult,
    ExecutionMode,
    FinalDecisionRecord,
    OrchestrationStatus,
    OrchestrationSummary,
    TokenTrackingDocument,
    TokenUsage,
)
from .settings import CosmosSettings

logger = logging.getLogger(__name__)


class CosmosAgentRunsService:
    """Service for persisting agent execution runs to Cosmos DB.
    
    This service:
    - Initializes Cosmos DB connection using Azure AD (DefaultAzureCredential)
    - Creates database and all containers if they don't exist
    - Provides methods to save and query agent runs, token usage, and evaluations
    - Handles all errors gracefully without propagating
    """
    
    def __init__(self, settings: Optional[CosmosSettings] = None):
        """Initialize the Cosmos DB service.
        
        Args:
            settings: Optional Cosmos DB settings. If not provided, loads from environment.
        """
        self.settings = settings or CosmosSettings.from_env()
        self.client: Optional[CosmosClient] = None
        self._agent_runs_container: Optional[ContainerProxy] = None
        self._token_tracking_container: Optional[ContainerProxy] = None
        self._evaluations_container: Optional[ContainerProxy] = None
        self._initialized = False
        self._yaml_cache: Optional[Dict[str, Any]] = None
    
    @property
    def is_available(self) -> bool:
        """Check if Cosmos DB service is available and initialized."""
        return self._initialized and self._agent_runs_container is not None
    
    async def initialize(self) -> bool:
        """Initialize Cosmos DB connection and all containers.
        
        Creates the following containers on startup if they don't exist:
        - underwriting_agent_runs: Complete execution records (partition: /id)
        - token_tracking: Token usage telemetry (partition: /id)
        - evaluations: Agent evaluation results (partition: /id)
        
        Returns:
            True if initialization successful, False otherwise.
            
        Note: This method never raises exceptions - it logs errors and returns False.
        """
        if self._initialized:
            logger.debug("Cosmos DB already initialized")
            return True
        
        if not self.settings.is_configured:
            logger.info("Cosmos DB not configured (AZURE_COSMOS_ENDPOINT not set). Agent run persistence disabled.")
            return False
        
        try:
            logger.info(f"Initializing Cosmos DB connection: endpoint={self.settings.endpoint}")
            logger.debug(f"Database: {self.settings.database_name}")
            logger.debug(f"Containers: agent_runs={self.settings.agent_runs_container}, "
                        f"token_tracking={self.settings.token_tracking_container}, "
                        f"evaluations={self.settings.evaluations_container}")
            logger.debug(f"Partition keys: {self.settings.agent_runs_partition_key}, "
                        f"{self.settings.token_tracking_partition_key}, "
                        f"{self.settings.evaluations_partition_key}")
            
            # Use Azure AD authentication (DefaultAzureCredential)
            credential = DefaultAzureCredential()
            
            # Create Cosmos client
            self.client = CosmosClient(
                url=self.settings.endpoint,
                credential=credential
            )
            
            # Create database if not exists
            database = self.client.create_database_if_not_exists(
                id=self.settings.database_name
            )
            logger.info(f"✅ Connected to Cosmos DB database: {self.settings.database_name}")
            
            # Create all containers
            # Using serverless mode (no throughput specified) for cost efficiency
            if self.settings.use_serverless:
                # Agent runs container (partition: /application_id)
                self._agent_runs_container = database.create_container_if_not_exists(
                    id=self.settings.agent_runs_container,
                    partition_key=PartitionKey(path=self.settings.agent_runs_partition_key)
                )
                logger.info(f"✅ Container ready: {self.settings.agent_runs_container} (partition: {self.settings.agent_runs_partition_key})")
                
                # Token tracking container (partition: /execution_id)
                self._token_tracking_container = database.create_container_if_not_exists(
                    id=self.settings.token_tracking_container,
                    partition_key=PartitionKey(path=self.settings.token_tracking_partition_key)
                )
                logger.info(f"✅ Container ready: {self.settings.token_tracking_container} (partition: {self.settings.token_tracking_partition_key})")
                
                # Evaluations container (partition: /evaluation_id)
                self._evaluations_container = database.create_container_if_not_exists(
                    id=self.settings.evaluations_container,
                    partition_key=PartitionKey(path=self.settings.evaluations_partition_key)
                )
                logger.info(f"✅ Container ready: {self.settings.evaluations_container} (partition: {self.settings.evaluations_partition_key})")
            else:
                # With provisioned throughput
                self._agent_runs_container = database.create_container_if_not_exists(
                    id=self.settings.agent_runs_container,
                    partition_key=PartitionKey(path=self.settings.agent_runs_partition_key),
                    offer_throughput=self.settings.provisioned_throughput
                )
                self._token_tracking_container = database.create_container_if_not_exists(
                    id=self.settings.token_tracking_container,
                    partition_key=PartitionKey(path=self.settings.token_tracking_partition_key),
                    offer_throughput=self.settings.provisioned_throughput
                )
                self._evaluations_container = database.create_container_if_not_exists(
                    id=self.settings.evaluations_container,
                    partition_key=PartitionKey(path=self.settings.evaluations_partition_key),
                    offer_throughput=self.settings.provisioned_throughput
                )
                logger.info(f"✅ All containers ready with {self.settings.provisioned_throughput} RU/s throughput")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Cosmos DB: {e}")
            self._initialized = False
            return False
    
    # Legacy property for backward compatibility
    @property
    def _container(self) -> Optional[ContainerProxy]:
        """Backward compatibility alias for _agent_runs_container."""
        return self._agent_runs_container
    
    def _load_agent_definitions_yaml(self) -> Dict[str, Any]:
        """Load agent definitions from YAML file (cached)."""
        if self._yaml_cache is not None:
            return self._yaml_cache
        
        yaml_path = Path(__file__).parent.parent.parent / ".github" / "underwriting_agents.yaml"
        
        if not yaml_path.exists():
            logger.warning(f"Agent definitions YAML not found: {yaml_path}")
            return {"version": "unknown", "agents": [], "global_constraints": []}
        
        try:
            with open(yaml_path, "r") as f:
                self._yaml_cache = yaml.safe_load(f) or {}
            return self._yaml_cache
        except Exception as e:
            logger.warning(f"Failed to load agent definitions YAML: {e}")
            return {"version": "unknown", "agents": [], "global_constraints": []}
    
    def _create_agent_definition_snapshot(self, agent_def: Dict[str, Any]) -> AgentDefinitionSnapshot:
        """Create a snapshot of an agent definition from YAML."""
        inputs = agent_def.get("inputs", {})
        return AgentDefinitionSnapshot(
            agent_id=agent_def.get("agent_id", "unknown"),
            role=agent_def.get("role"),
            purpose=agent_def.get("purpose"),
            instructions=agent_def.get("instructions", []),
            inputs_required=list(inputs.get("required", [])),
            inputs_optional=list(inputs.get("optional", [])),
            outputs=agent_def.get("outputs", {}),
            failure_modes=agent_def.get("failure_modes", []),
        )
    
    async def save_agent_run(self, run_document: AgentRunDocument) -> bool:
        """Save an agent run document to Cosmos DB.
        
        This is an append-only operation. Documents are never modified after creation.
        
        Args:
            run_document: Complete agent run document to save.
            
        Returns:
            True if save successful, False otherwise.
            
        Note: This method never raises exceptions - it logs errors and returns False.
        """
        logger.debug(f"save_agent_run called - initialized={self._initialized}, container exists={self._agent_runs_container is not None}")
        
        if not self._initialized:
            logger.debug("Service not initialized, attempting initialization...")
            if not await self.initialize():
                logger.warning("Cosmos DB not available, skipping agent run persistence")
                return False
        
        if not self._agent_runs_container:
            logger.warning("Cosmos container not available, skipping agent run persistence")
            return False
        
        try:
            # Convert to dict for Cosmos DB
            doc_dict = run_document.model_dump(mode='json')
            
            logger.debug(
                f"Preparing to save document: id={doc_dict.get('id')}, "
                f"application_id={doc_dict.get('application_id')}, "
                f"run_id={doc_dict.get('run_id')}"
            )
            
            # Ensure proper datetime serialization
            if 'execution_timestamp' in doc_dict:
                if isinstance(doc_dict['execution_timestamp'], datetime):
                    doc_dict['execution_timestamp'] = doc_dict['execution_timestamp'].isoformat()
            
            # Create document (not upsert - append-only)
            result = self._agent_runs_container.create_item(body=doc_dict)
            
            logger.info(
                f"✅ Saved agent run to Cosmos DB: run_id={run_document.run_id}, "
                f"application_id={run_document.application_id}, "
                f"mode={run_document.execution_mode.value}"
            )
            return True
            
        except exceptions.CosmosResourceExistsError:
            # Document already exists - this shouldn't happen with UUIDs but handle gracefully
            logger.warning(f"Agent run document already exists: {run_document.id}")
            return True  # Consider this a success since data is persisted
            
        except Exception as e:
            logger.error(f"❌ Failed to save agent run to Cosmos DB: {e}", exc_info=True)
            return False
    
    async def create_run_document_from_orchestrator_output(
        self,
        application_id: str,
        orchestrator_output: Any,  # OrchestratorOutput from orchestrator.py
        agent_step_details: Optional[List[Dict[str, Any]]] = None,
    ) -> AgentRunDocument:
        """Create an AgentRunDocument from orchestrator output.
        
        Args:
            application_id: The underwriting application ID.
            orchestrator_output: The OrchestratorOutput object from the orchestrator.
            agent_step_details: Optional list of detailed step information including
                               inputs, outputs, and token usage per agent.
        
        Returns:
            Complete AgentRunDocument ready for persistence.
        """
        # Load agent definitions from YAML
        yaml_data = self._load_agent_definitions_yaml()
        
        # Create agent definition snapshots
        agent_definitions = []
        for agent_def in yaml_data.get("agents", []):
            agent_definitions.append(self._create_agent_definition_snapshot(agent_def))
        
        # Build agent step records from orchestrator execution records
        agents: List[AgentStepRecord] = []
        execution_records = getattr(orchestrator_output, 'execution_records', [])
        
        for record in execution_records:
            # Try to find detailed info if provided
            step_detail = {}
            if agent_step_details:
                step_detail = next(
                    (s for s in agent_step_details if s.get('agent_id') == record.agent_id),
                    {}
                )
            
            # Build token usage if available
            token_usage = None
            raw_token = step_detail.get('token_usage')
            if raw_token:
                token_usage = TokenUsage(
                    prompt_tokens=raw_token.get('prompt_tokens'),
                    completion_tokens=raw_token.get('completion_tokens'),
                    total_tokens=raw_token.get('total_tokens'),
                    estimated_cost_usd=raw_token.get('estimated_cost_usd'),
                )
            else:
                token_usage = TokenUsage(
                    unavailable_reason="Token usage not available from Azure AI Foundry SDK"
                )
            
            # Build evaluation results if available
            evaluation_results = None
            raw_eval = step_detail.get('evaluation_results')
            if raw_eval:
                evaluation_results = EvaluationResult(
                    groundedness=raw_eval.get('groundedness'),
                    relevance=raw_eval.get('relevance'),
                    coherence=raw_eval.get('coherence'),
                    fluency=raw_eval.get('fluency'),
                    custom_metrics=raw_eval.get('custom_metrics', {}),
                )
            else:
                evaluation_results = EvaluationResult(
                    unavailable_reason="Evaluation not run for this execution"
                )
            
            agent_step = AgentStepRecord(
                agent_id=record.agent_id,
                step_number=record.step_number,
                execution_id=record.execution_id,
                started_at=record.timestamp,
                completed_at=record.timestamp,  # Same as started for now
                execution_duration_ms=record.execution_time_ms,
                success=record.success,
                inputs=step_detail.get('inputs', {}),
                outputs=step_detail.get('outputs', {}),
                output_summary=record.output_summary,
                evaluation_results=evaluation_results,
                token_usage=token_usage,
            )
            agents.append(agent_step)
        
        # Build orchestration summary
        agents_succeeded = sum(1 for a in agents if a.success)
        agents_failed = len(agents) - agents_succeeded
        
        orchestration_status = OrchestrationStatus.SUCCESS
        if agents_failed > 0:
            orchestration_status = OrchestrationStatus.PARTIAL_SUCCESS if agents_succeeded > 0 else OrchestrationStatus.FAILURE
        
        orchestration_summary = OrchestrationSummary(
            status=orchestration_status,
            execution_order=[r.agent_id for r in execution_records],
            agents_executed=len(agents),
            agents_succeeded=agents_succeeded,
            agents_failed=agents_failed,
            total_execution_time_ms=orchestrator_output.total_execution_time_ms,
            errors=[a.error_message for a in agents if a.error_message],
        )
        
        # Build final decision record
        final_dec = orchestrator_output.final_decision
        final_decision = FinalDecisionRecord(
            decision_id=final_dec.decision_id,
            underwriting_decision=final_dec.status.value if hasattr(final_dec.status, 'value') else str(final_dec.status),
            risk_level=final_dec.risk_level.value if hasattr(final_dec.risk_level, 'value') else str(final_dec.risk_level),
            premium_adjustment_pct=final_dec.premium_adjustment_pct,
            adjusted_premium_annual=final_dec.adjusted_premium_annual,
            confidence_score=orchestrator_output.confidence_score,
            explanation=orchestrator_output.explanation,
            business_rules_compliant=final_dec.business_rules_approved,
            bias_check_passed=final_dec.bias_check_passed,
            underwriter_message=final_dec.underwriter_message,
            customer_message=final_dec.customer_message,
        )
        
        # Create the complete document
        run_document = AgentRunDocument(
            application_id=application_id,
            execution_mode=ExecutionMode.MULTI_AGENT,
            workflow_id=orchestrator_output.workflow_id,
            agent_definitions_version=yaml_data.get("version", "unknown"),
            agent_definitions=agent_definitions,
            global_constraints=yaml_data.get("global_constraints", []),
            agents=agents,
            orchestration_summary=orchestration_summary,
            final_decision=final_decision,
            metadata={
                "source": "OrchestratorAgent.run_with_progress",
                "foundry_agents_used": True,
            }
        )
        
        return run_document
    
    async def get_runs_by_application(
        self,
        application_id: str,
        limit: int = 100
    ) -> List[AgentRunDocument]:
        """Get all agent runs for an application.
        
        Args:
            application_id: The application to query.
            limit: Maximum number of results.
            
        Returns:
            List of agent run documents, ordered by execution_timestamp DESC.
        """
        if not self.is_available:
            return []
        
        try:
            query = """
                SELECT * FROM c 
                WHERE c.application_id = @application_id
                ORDER BY c.execution_timestamp DESC
                OFFSET 0 LIMIT @limit
            """
            parameters = [
                {"name": "@application_id", "value": application_id},
                {"name": "@limit", "value": limit},
            ]
            
            items = list(self._agent_runs_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=False  # Using partition key
            ))
            
            return [AgentRunDocument(**item) for item in items]
            
        except Exception as e:
            logger.error(f"❌ Failed to query agent runs for {application_id}: {e}")
            return []
    
    async def get_run_by_id(self, run_id: str, application_id: str) -> Optional[AgentRunDocument]:
        """Get a specific agent run by ID.
        
        Args:
            run_id: The run ID to retrieve.
            application_id: The application ID (required for partition key).
            
        Returns:
            AgentRunDocument if found, None otherwise.
        """
        if not self.is_available:
            return None
        
        try:
            query = """
                SELECT * FROM c 
                WHERE c.run_id = @run_id AND c.application_id = @application_id
            """
            parameters = [
                {"name": "@run_id", "value": run_id},
                {"name": "@application_id", "value": application_id},
            ]
            
            items = list(self._agent_runs_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=False
            ))
            
            return AgentRunDocument(**items[0]) if items else None
            
        except Exception as e:
            logger.error(f"❌ Failed to get agent run {run_id}: {e}")
            return None

    # =========================================================================
    # TOKEN TRACKING OPERATIONS
    # =========================================================================
    
    async def save_token_usage(self, token_doc: "TokenTrackingDocument") -> bool:
        """Save a token usage record to the token_tracking container.
        
        Args:
            token_doc: Token tracking document to save.
            
        Returns:
            True if save successful, False otherwise.
        """
        if not self._initialized or not self._token_tracking_container:
            logger.debug("Token tracking container not available, skipping save")
            return False
        
        try:
            from .models import TokenTrackingDocument
            doc_dict = token_doc.model_dump(mode='json')
            self._token_tracking_container.create_item(body=doc_dict)
            logger.debug(f"✅ Saved token usage: {token_doc.agent_id} - {token_doc.total_tokens} tokens")
            return True
        except Exception as e:
            logger.warning(f"Failed to save token usage (non-fatal): {e}")
            return False
    
    async def get_token_usage_by_execution(self, execution_id: str) -> List[Dict[str, Any]]:
        """Get all token usage records for an execution.
        
        Args:
            execution_id: Workflow execution ID.
            
        Returns:
            List of token usage records.
        """
        if not self._initialized or not self._token_tracking_container:
            return []
        
        try:
            query = "SELECT * FROM c WHERE c.execution_id = @execution_id ORDER BY c.timestamp ASC"
            parameters = [{"name": "@execution_id", "value": execution_id}]
            
            items = list(self._token_tracking_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=False
            ))
            return items
        except Exception as e:
            logger.error(f"❌ Failed to get token usage for {execution_id}: {e}")
            return []
    
    async def get_token_usage_analytics(
        self,
        application_id: Optional[str] = None,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """Get aggregated token usage analytics.
        
        Args:
            application_id: Optional filter by application.
            days_back: Number of days to look back.
            
        Returns:
            Analytics summary dict.
        """
        if not self._initialized or not self._token_tracking_container:
            return {"total_tokens": 0, "total_cost_usd": 0.0, "by_agent": {}}
        
        try:
            from datetime import timedelta
            start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            query = "SELECT * FROM c WHERE c.timestamp >= @start_date"
            parameters = [{"name": "@start_date", "value": start_date.isoformat()}]
            
            if application_id:
                query += " AND c.application_id = @application_id"
                parameters.append({"name": "@application_id", "value": application_id})
            
            items = list(self._token_tracking_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            # Aggregate
            by_agent = {}
            total_tokens = 0
            total_cost = 0.0
            
            for item in items:
                agent = item.get("agent_id", "unknown")
                if agent not in by_agent:
                    by_agent[agent] = {"tokens": 0, "cost_usd": 0.0, "requests": 0}
                
                by_agent[agent]["tokens"] += item.get("total_tokens", 0)
                by_agent[agent]["cost_usd"] += item.get("total_cost_usd", 0.0)
                by_agent[agent]["requests"] += 1
                
                total_tokens += item.get("total_tokens", 0)
                total_cost += item.get("total_cost_usd", 0.0)
            
            return {
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 4),
                "by_agent": by_agent,
                "period_days": days_back,
                "total_requests": len(items)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get token analytics: {e}")
            return {"total_tokens": 0, "total_cost_usd": 0.0, "by_agent": {}}

    # =========================================================================
    # EVALUATION OPERATIONS
    # =========================================================================
    
    async def save_evaluation(self, eval_doc: "EvaluationDocument") -> bool:
        """Save an evaluation result to the evaluations container.
        
        Args:
            eval_doc: Evaluation document to save.
            
        Returns:
            True if save successful, False otherwise.
        """
        if not self._initialized or not self._evaluations_container:
            logger.debug("Evaluations container not available, skipping save")
            return False
        
        try:
            from .models import EvaluationDocument
            doc_dict = eval_doc.model_dump(mode='json')
            self._evaluations_container.create_item(body=doc_dict)
            logger.info(f"✅ Saved evaluation: {eval_doc.agent_id} - score={eval_doc.overall_score}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save evaluation (non-fatal): {e}")
            return False
    
    async def get_evaluations_by_execution(self, execution_id: str) -> List[Dict[str, Any]]:
        """Get all evaluations for an execution.
        
        Args:
            execution_id: Workflow execution ID.
            
        Returns:
            List of evaluation records.
        """
        if not self._initialized or not self._evaluations_container:
            return []
        
        try:
            query = """
                SELECT * FROM c 
                WHERE c.execution_id = @execution_id
                ORDER BY c.step_number ASC
            """
            parameters = [{"name": "@execution_id", "value": execution_id}]
            
            items = list(self._evaluations_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            return items
        except Exception as e:
            logger.error(f"❌ Failed to get evaluations for {execution_id}: {e}")
            return []
    
    async def get_evaluations_by_application(self, application_id: str) -> List[Dict[str, Any]]:
        """Get all evaluations for an application.
        
        Args:
            application_id: Application ID.
            
        Returns:
            List of evaluation records.
        """
        if not self._initialized or not self._evaluations_container:
            return []
        
        try:
            query = """
                SELECT * FROM c 
                WHERE c.application_id = @application_id
                ORDER BY c.evaluation_timestamp DESC
            """
            parameters = [{"name": "@application_id", "value": application_id}]
            
            items = list(self._evaluations_container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            return items
        except Exception as e:
            logger.error(f"❌ Failed to get evaluations for application {application_id}: {e}")
            return []


# Global singleton instance
_cosmos_service: Optional[CosmosAgentRunsService] = None


async def get_cosmos_service() -> CosmosAgentRunsService:
    """Get or create the global Cosmos DB service instance.
    
    Returns:
        Initialized Cosmos DB service.
    """
    global _cosmos_service
    if _cosmos_service is None:
        _cosmos_service = CosmosAgentRunsService()
        await _cosmos_service.initialize()
    return _cosmos_service
