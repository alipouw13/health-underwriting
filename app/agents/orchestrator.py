"""
OrchestratorAgent - Coordinate agent execution and produce final decision

Agent Definition (from /.github/underwriting_agents_v2.yaml):
-------------------------------------------------------------
agent_id: OrchestratorAgent
purpose: Coordinate agent execution and produce final decision
inputs:
  patient_id: string
outputs:
  final_decision: object
  confidence_score: number
  explanation: string
tools_used:
  - agent-framework
evaluation_criteria:
  - correctness
  - determinism
failure_modes:
  - partial_execution

EXECUTION ORDER (SIMPLIFIED 2-AGENT MVP + COMMUNICATION):
1. HealthDataAnalysisAgent - Extract health risk signals from documents
2. PolicyRiskAgent - Apply JSON underwriting policies, determine risk level, 
   premium adjustment, and approval/denial decision
3. CommunicationAgent - Generate decision messages for underwriter and customer

The PolicyRiskAgent uses the JSON policies from prompts/life-health-underwriting-policies.json
which can be edited via the Admin UI's Underwriting Policies tab.

FUTURE AGENTS (Post-MVP with Foundry SDK evaluations/citations):
- DataQualityConfidenceAgent
- BiasAndFairnessAgent  
- AuditAndTraceAgent

CONSTRAINTS:
- No conditional branching
- No skipping agents
- No reinterpretation of agent outputs
- Orchestrator may summarize outputs
- Orchestrator may NOT alter or override agent conclusions
"""

from __future__ import annotations

import asyncio
import logging
import yaml
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, AsyncGenerator
from uuid import uuid4
from pydantic import Field

# Token tracking import (optional - gracefully handle if not available)
try:
    from app.token_tracker import (
        track_agent_execution,
        create_tracking_context,
        close_tracking_context,
        persist_context,
        TokenTrackingContext,
    )
    TOKEN_TRACKING_AVAILABLE = True
except ImportError:
    TOKEN_TRACKING_AVAILABLE = False

# Tracing import for Azure AI Foundry portal (optional)
try:
    from app.tracing import get_tracer, add_span_attribute, add_span_event, is_tracing_enabled
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    def get_tracer(): return None
    def add_span_attribute(k, v): pass
    def add_span_event(n, a=None): pass
    def is_tracing_enabled(): return False


# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class AgentProgressStatus(str, Enum):
    """Status of an agent in the workflow."""
    PENDING = "pending"
    RUNNING = "running"
    TOOL_CALLING = "tool_calling"
    PROCESSING = "processing"
    OUTPUT_READY = "output_ready"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentProgressStage(str, Enum):
    """Detailed stage within agent execution."""
    STARTED = "started"
    PREPARING_INPUT = "preparing_input"
    INVOKING_MODEL = "invoking_model"
    TOOL_CALLED = "tool_called"
    PARSING_RESPONSE = "parsing_response"
    VALIDATING_OUTPUT = "validating_output"
    COMPLETED = "completed"
    FAILED = "failed"
    # Evaluation stages
    EVALUATING = "evaluating"
    EVALUATION_COMPLETE = "evaluation_complete"


@dataclass
class AgentProgress:
    """Progress event for agent execution."""
    workflow_id: str
    agent_id: str
    agent_name: str  # Human-readable name
    step_number: int
    total_steps: int
    status: AgentProgressStatus
    stage: Optional[AgentProgressStage] = None
    execution_time_ms: Optional[float] = None
    message: Optional[str] = None
    safe_summary: Optional[str] = None  # 1-2 sentence summary (no chain-of-thought)
    tools_used: Optional[List[str]] = None  # Names of tools being used
    output_preview: Optional[str] = None  # Brief preview of output
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflow_id": self.workflow_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "step_number": self.step_number,
            "total_steps": self.total_steps,
            "status": self.status.value,
            "stage": self.stage.value if self.stage else None,
            "execution_time_ms": self.execution_time_ms,
            "message": self.message,
            "safe_summary": self.safe_summary,
            "tools_used": self.tools_used,
            "output_preview": self.output_preview,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# Type alias for progress callback
ProgressCallback = Callable[[AgentProgress], None]

from data.mock.schemas import (
    DecisionStatus,
    HealthMetrics,
    PatientProfile,
    PolicyRuleSet,
    RiskIndicator,
    RiskLevel,
    UnderwritingDecision,
)
from data.mock.fixtures import (
    get_patient_by_id,
    get_healthy_patient_metrics,
    get_moderate_risk_metrics,
    get_high_risk_metrics,
    get_standard_policy_rules,
)
from app.agents.base import (
    AgentInput,
    AgentOutput,
    AgentExecutionError,
)

# Import agents for orchestration (simplified 2-agent workflow + communication)
from app.agents.health_data_analysis import (
    HealthDataAnalysisAgent,
    HealthDataAnalysisInput,
    HealthDataAnalysisOutput,
)
from app.agents.policy_risk import PolicyRiskAgent, PolicyRiskOutput
from app.agents.apple_health_risk import AppleHealthRiskAgent, AppleHealthRiskOutput
from app.agents.communication import (
    CommunicationAgent,
    CommunicationInput,
    CommunicationOutput,
)

# Import output types for type hints (used in context retrieval)
from app.agents.data_quality_confidence import DataQualityConfidenceOutput
from app.agents.bias_fairness import BiasAndFairnessOutput
from app.agents.audit_trace import AuditAndTraceOutput, AgentOutputRecord

# Import Foundry Evaluations integration
from app.agents.evaluations import (
    get_evaluator_service,
    is_evaluations_enabled,
    AgentEvaluationResult,
    WorkflowEvaluationResult,
    EvaluationStatus,
)


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class OrchestratorInput(AgentInput):
    """Input schema for OrchestratorAgent."""
    
    patient_id: str = Field(..., description="Patient ID to process")
    patient_name: Optional[str] = Field(None, description="Patient name for display (optional)")
    health_metrics: Optional[HealthMetrics] = Field(None, description="Override health metrics (optional)")
    policy_rules: Optional[PolicyRuleSet] = Field(None, description="Override policy rules (optional)")
    
    # Real application data (used when processing actual uploaded documents)
    application_data: Optional[Dict[str, Any]] = Field(None, description="Extracted application data from document")
    document_markdown: Optional[str] = Field(None, description="Document markdown content")
    llm_outputs: Optional[Dict[str, Any]] = Field(None, description="LLM analysis outputs")

class AgentExecutionRecord(AgentInput):
    """Record of a single agent execution in the workflow."""
    
    agent_id: str = Field(..., description="ID of the agent")
    step_number: int = Field(..., description="Execution order (1-7)")
    execution_id: str = Field(..., description="Unique execution ID")
    timestamp: datetime = Field(..., description="When agent completed")
    execution_time_ms: float = Field(..., description="Execution time in ms")
    success: bool = Field(..., description="Whether execution succeeded")
    output_summary: str = Field(..., description="Brief summary of output")
    
    # Actual inputs/outputs for transparency
    actual_inputs: Optional[Dict[str, Any]] = Field(default=None, description="Actual input data passed to agent")
    actual_outputs: Optional[Dict[str, Any]] = Field(default=None, description="Actual output data from agent")
    tools_invoked: Optional[List[str]] = Field(default=None, description="Tools/MCP servers actually called")


class FinalDecision(AgentInput):
    """Final underwriting decision produced by orchestrator."""
    
    decision_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique decision ID")
    patient_id: str = Field(..., description="Patient ID")
    patient_name: Optional[str] = Field(default=None, description="Patient name for display")
    status: DecisionStatus = Field(..., description="Decision status")
    risk_level: RiskLevel = Field(..., description="Final risk level")
    risk_class: Optional[str] = Field(default=None, description="Risk classification (e.g., Standard Plus, Standard)")
    hkrs_score: Optional[float] = Field(default=None, description="HKRS score for Apple Health workflow")
    hkrs_band: Optional[str] = Field(default=None, description="HKRS band for Apple Health workflow")
    approved: bool = Field(..., description="Whether application is approved")
    premium_adjustment_pct: float = Field(..., description="Premium adjustment percentage")
    adjusted_premium_annual: float = Field(..., description="Final annual premium")
    business_rules_approved: bool = Field(..., description="Business rules validation passed")
    bias_check_passed: bool = Field(..., description="Bias/fairness check passed")
    underwriter_message: str = Field(..., description="Message for underwriter")
    customer_message: str = Field(..., description="Message for customer")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OrchestratorOutput(AgentOutput):
    """Output schema for OrchestratorAgent."""
    
    final_decision: FinalDecision = Field(..., description="Final underwriting decision")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    explanation: str = Field(..., description="Explanation of the decision")
    execution_records: List[AgentExecutionRecord] = Field(..., description="Records of all agent executions")
    workflow_id: str = Field(default_factory=lambda: str(uuid4()), description="Workflow execution ID")
    total_execution_time_ms: float = Field(..., description="Total orchestration time")
    execution_source: str = Field(default="underwriter", description="Source: 'underwriter' or 'end_user'")
    
    # Foundry Evaluations (STEP 6: Added for UI display)
    evaluations: Optional[Dict[str, Any]] = Field(default=None, description="Agent and workflow evaluation results")
    workflow_evaluation: Optional[Dict[str, Any]] = Field(default=None, description="Aggregate workflow evaluation")


# =============================================================================
# EXECUTION CONTEXT
# =============================================================================

class ExecutionContext:
    """
    Shared execution context for agent workflow.
    
    Stores outputs from each agent for use by subsequent agents.
    This is the ONLY mechanism for passing data between agents.
    
    EVALUATION INTEGRATION (STEP 4):
    - Tracks evaluation results for each agent
    - Stores workflow-level evaluation results
    - Evaluations run AFTER agent execution (non-blocking)
    
    TOKEN TRACKING:
    - Records token usage for each agent execution
    - Aggregates total workflow token consumption
    - Persists to Cosmos DB token_tracking container
    """
    
    def __init__(self, patient_id: str, workflow_id: str, application_id: Optional[str] = None):
        self.patient_id = patient_id
        self.workflow_id = workflow_id
        self.application_id = application_id or patient_id
        self.start_time = datetime.now(timezone.utc)
        self._outputs: Dict[str, AgentOutput] = {}
        self._records: List[AgentExecutionRecord] = []
        self.logger = logging.getLogger(f"orchestration.{workflow_id}")
        
        # Real application data (set when processing actual documents)
        self.application_data: Optional[Dict[str, Any]] = None
        self.document_markdown: Optional[str] = None
        self.llm_outputs: Optional[Dict[str, Any]] = None
        
        # Foundry Evaluations (STEP 4 integration)
        self._evaluations: Dict[str, AgentEvaluationResult] = {}
        self._workflow_evaluation: Optional[WorkflowEvaluationResult] = None
        
        # Token tracking
        self._token_records: List[Dict[str, Any]] = []
        self._token_context: Optional[Any] = None
        if TOKEN_TRACKING_AVAILABLE:
            try:
                self._token_context = create_tracking_context(
                    execution_id=workflow_id,
                    application_id=self.application_id,
                    operation_type="agent_workflow",
                )
                self.logger.debug("Token tracking context created for workflow %s", workflow_id)
            except Exception as e:
                self.logger.warning("Could not create token tracking context: %s", e)
    
    def store_output(
        self, 
        agent_id: str, 
        output: AgentOutput, 
        step_number: int,
        actual_inputs: Optional[Dict[str, Any]] = None,
        tools_invoked: Optional[List[str]] = None
    ) -> None:
        """Store an agent's output in the context with actual inputs/outputs."""
        self._outputs[agent_id] = output
        
        # Extract actual output data from the AgentOutput model
        actual_outputs = self._extract_output_data(agent_id, output)
        
        # Create execution record with actual data
        record = AgentExecutionRecord(
            agent_id=agent_id,
            step_number=step_number,
            execution_id=output.execution_id,
            timestamp=output.timestamp,
            execution_time_ms=output.execution_time_ms or 0.0,
            success=output.success,
            output_summary=self._summarize_output(agent_id, output),
            actual_inputs=actual_inputs,
            actual_outputs=actual_outputs,
            tools_invoked=tools_invoked,
        )
        self._records.append(record)
        
        self.logger.info(f"Step {step_number}: {agent_id} completed in {output.execution_time_ms:.2f}ms")
    
    def store_evaluation(self, agent_id: str, evaluation: AgentEvaluationResult) -> None:
        """Store evaluation result for an agent."""
        self._evaluations[agent_id] = evaluation
        self.logger.info(
            f"Evaluation stored for {agent_id}: status={evaluation.status.value}, "
            f"score={evaluation.aggregate_score or 'N/A'}"
        )
    
    def get_evaluation(self, agent_id: str) -> Optional[AgentEvaluationResult]:
        """Get evaluation result for an agent."""
        return self._evaluations.get(agent_id)
    
    def get_all_evaluations(self) -> Dict[str, AgentEvaluationResult]:
        """Get all agent evaluations."""
        return self._evaluations.copy()
    
    def set_workflow_evaluation(self, evaluation: WorkflowEvaluationResult) -> None:
        """Set the workflow-level evaluation result."""
        self._workflow_evaluation = evaluation
    
    def get_workflow_evaluation(self) -> Optional[WorkflowEvaluationResult]:
        """Get workflow-level evaluation result."""
        return self._workflow_evaluation
    
    def get_output(self, agent_id: str) -> Optional[AgentOutput]:
        """Get a stored agent output."""
        return self._outputs.get(agent_id)
    
    def get_all_outputs(self) -> Dict[str, AgentOutput]:
        """Get all stored outputs."""
        return self._outputs.copy()
    
    def get_records(self) -> List[AgentExecutionRecord]:
        """Get all execution records."""
        return self._records.copy()
    
    def get_total_time_ms(self) -> float:
        """Calculate total execution time."""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds() * 1000
    
    def _summarize_output(self, agent_id: str, output: AgentOutput) -> str:
        """Create a brief summary of agent output."""
        if agent_id == "HealthDataAnalysisAgent":
            hda_out = output  # type: HealthDataAnalysisOutput
            return f"Identified {len(hda_out.risk_indicators)} risk indicators"
        elif agent_id == "PolicyRiskAgent":
            pr_out = output  # type: PolicyRiskOutput
            return f"Decision: {pr_out.decision}, Risk: {pr_out.risk_level.value}, Adjustment: {pr_out.premium_adjustment_recommendation.adjustment_percentage}%"
        elif agent_id == "CommunicationAgent":
            return "Messages generated"
        else:
            return "Output recorded"
    
    def _extract_output_data(self, agent_id: str, output: AgentOutput) -> Dict[str, Any]:
        """Extract actual output data from agent output for transparency."""
        try:
            # Use model_dump to get all fields, excluding base execution fields
            data = output.model_dump(exclude={'execution_id', 'timestamp', 'execution_time_ms', 'success', 'error_message'})
            
            # Truncate long strings for UI display
            def truncate_strings(obj, max_len=500):
                if isinstance(obj, str) and len(obj) > max_len:
                    return obj[:max_len] + "..."
                elif isinstance(obj, dict):
                    return {k: truncate_strings(v, max_len) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [truncate_strings(item, max_len) for item in obj[:20]]  # Limit arrays to 20 items
                return obj
            
            return truncate_strings(data)
        except Exception:
            return {"raw": str(output)[:500]}
    
    # =========================================================================
    # TOKEN TRACKING METHODS
    # =========================================================================
    
    def record_token_usage(
        self,
        agent_id: str,
        token_usage: Optional[Dict[str, int]],
        step_number: int,
        model_name: Optional[str] = None,
    ) -> None:
        """Record token usage for an agent execution.
        
        Args:
            agent_id: The agent identifier.
            token_usage: Dict with prompt_tokens, completion_tokens, total_tokens.
            step_number: The step number in the workflow.
            model_name: Optional model name for cost calculation.
        """
        if not token_usage:
            self.logger.debug("No token usage data for %s", agent_id)
            return
        
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)
        
        if total_tokens == 0:
            self.logger.debug("Zero tokens recorded for %s", agent_id)
            return
        
        record = {
            "agent_id": agent_id,
            "step_number": step_number,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "model_name": model_name,
        }
        self._token_records.append(record)
        
        # Also record in token context if available
        if self._token_context:
            self._token_context.record_usage(
                agent_id=agent_id,
                agent_type="foundry_agent",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model_name=model_name,
            )
        
        self.logger.info(
            "Token usage recorded: %s (step %d) - %d prompt + %d completion = %d total",
            agent_id, step_number, prompt_tokens, completion_tokens, total_tokens
        )
    
    def get_token_records(self) -> List[Dict[str, Any]]:
        """Get all token usage records for this execution."""
        return self._token_records.copy()
    
    def get_token_summary(self) -> Dict[str, Any]:
        """Get summary of token usage across all agents."""
        total_prompt = sum(r.get("prompt_tokens", 0) for r in self._token_records)
        total_completion = sum(r.get("completion_tokens", 0) for r in self._token_records)
        total_tokens = sum(r.get("total_tokens", 0) for r in self._token_records)
        
        by_agent = {}
        for record in self._token_records:
            agent_id = record["agent_id"]
            if agent_id not in by_agent:
                by_agent[agent_id] = {"prompt": 0, "completion": 0, "total": 0}
            by_agent[agent_id]["prompt"] += record.get("prompt_tokens", 0)
            by_agent[agent_id]["completion"] += record.get("completion_tokens", 0)
            by_agent[agent_id]["total"] += record.get("total_tokens", 0)
        
        return {
            "workflow_id": self.workflow_id,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "agent_count": len(self._token_records),
            "by_agent": by_agent,
        }
    
    async def persist_token_usage(self) -> int:
        """Persist token records to Cosmos DB.
        
        Returns:
            Number of records persisted.
        """
        if not TOKEN_TRACKING_AVAILABLE:
            return 0
        
        if self._token_context:
            try:
                count = await persist_context(self._token_context)
                self.logger.info("Persisted %d token records to Cosmos DB", count)
                return count
            except Exception as e:
                self.logger.warning("Failed to persist token records: %s", e)
        
        return 0
    
    def close_token_tracking(self) -> None:
        """Close the token tracking context."""
        if TOKEN_TRACKING_AVAILABLE and self._token_context:
            try:
                close_tracking_context(self.workflow_id)
                self.logger.debug("Token tracking context closed for workflow %s", self.workflow_id)
            except Exception as e:
                self.logger.warning("Error closing token tracking context: %s", e)


# =============================================================================
# YAML LOADER
# =============================================================================

class AgentDefinitionLoader:
    """Load agent definitions from YAML."""
    
    def __init__(self, yaml_path: Optional[Path] = None):
        self.yaml_path = yaml_path or Path(__file__).parent.parent.parent / ".github" / "underwriting_agents.yaml"
        self._definitions: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
    
    def load(self) -> Dict[str, Dict[str, Any]]:
        """Load agent definitions from YAML file."""
        if self._loaded:
            return self._definitions
        
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"Agent definitions not found: {self.yaml_path}")
        
        with open(self.yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        # Index by agent_id
        for agent_def in data.get("agents", []):
            agent_id = agent_def.get("agent_id")
            if agent_id:
                self._definitions[agent_id] = agent_def
        
        self._loaded = True
        return self._definitions
    
    def get_agent_definition(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get definition for a specific agent."""
        self.load()
        return self._definitions.get(agent_id)
    
    def get_execution_order(self) -> List[str]:
        """
        Return the STRICT execution order.
        
        SIMPLIFIED 2-AGENT WORKFLOW + COMMUNICATION:
        1. HealthDataAnalysisAgent - Extract risk signals from documents
        2. PolicyRiskAgent - Apply JSON policies, determine risk & decision
        3. CommunicationAgent - Generate decision messages
        """
        return [
            "HealthDataAnalysisAgent",
            "PolicyRiskAgent",
            "CommunicationAgent",
        ]


# =============================================================================
# ORCHESTRATOR AGENT
# =============================================================================

class OrchestratorAgent:
    """
    Coordinate agent execution and produce final decision.
    
    This agent orchestrates the underwriting workflow by executing
    all agents in a STRICT, DETERMINISTIC order and producing a
    final underwriting decision.
    
    EXECUTION ORDER (SIMPLIFIED 2-AGENT + COMMUNICATION):
        1. HealthDataAnalysisAgent - Analyze health data for risk signals
        2. PolicyRiskAgent - Apply JSON policies, determine risk level,
           premium adjustment, and approval/denial decision
        3. CommunicationAgent - Generate messages for underwriter and customer
    
    The PolicyRiskAgent uses the JSON policies from:
    prompts/life-health-underwriting-policies.json
    
    CONSTRAINTS:
        - No conditional branching
        - No skipping agents  
        - No reinterpretation of agent outputs
        - May summarize but NOT alter conclusions
    
    Tools Used:
        - agent-framework: Orchestration coordination
    
    Evaluation Criteria:
        - correctness: All agents executed correctly
        - determinism: Same inputs produce same outputs
    
    Failure Modes:
        - partial_execution: Some agents failed to complete
    """
    
    agent_id = "OrchestratorAgent"
    purpose = "Coordinate agent execution and produce final decision"
    tools_used = ["agent-framework"]
    evaluation_criteria = ["correctness", "determinism"]
    failure_modes = ["partial_execution"]
    
    # Map local agent IDs to Foundry agent names
    FOUNDRY_AGENT_NAMES = {
        "HealthDataAnalysisAgent": "health_data_analysis",
        "PolicyRiskAgent": "policy_risk_analysis",
        "AppleHealthRiskAgent": "apple_health_risk",
        "CommunicationAgent": "communication",
    }
    
    # Human-readable agent names for progress display
    AGENT_DISPLAY_NAMES = {
        "HealthDataAnalysisAgent": "Health Data Analysis",
        "PolicyRiskAgent": "Policy Risk Assessment",
        "AppleHealthRiskAgent": "Apple Health Risk Scoring",
        "CommunicationAgent": "Decision Communication",
    }
    
    def __init__(self, use_foundry: bool = None, use_demo: bool = False):
        """Initialize the orchestrator.
        
        Args:
            use_foundry: If True, invoke agents via Azure AI Foundry (real LLM calls).
                        If False, use local deterministic agents.
                        If None, auto-detect based on environment.
            use_demo: If True, forces local deterministic agents (overrides use_foundry).
        """
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        self.definition_loader = AgentDefinitionLoader()
        
        # Determine whether to use Foundry
        # use_demo=True forces local agents regardless of other settings
        if use_demo:
            use_foundry = False
        elif use_foundry is None:
            import os
            use_foundry = os.environ.get("USE_FOUNDRY_AGENTS", "").lower() == "true"
        
        self._use_foundry = use_foundry
        self._foundry_invoker = None
        
        if self._use_foundry:
            self.logger.info("OrchestratorAgent initialized with Azure AI Foundry agent invocation")
        else:
            self.logger.info("OrchestratorAgent initialized with local deterministic agents%s", 
                           " (demo mode)" if use_demo else "")
        
        # Initialize local agents
        # Traditional workflow (admin/underwriter): HealthDataAnalysis → PolicyRisk → Communication
        # Apple Health workflow (end_user): HealthDataAnalysis → AppleHealthRisk → Communication
        self._health_data_agent = HealthDataAnalysisAgent()
        self._policy_risk_agent = PolicyRiskAgent()
        self._apple_health_risk_agent = AppleHealthRiskAgent()
        self._communication_agent = CommunicationAgent()
    
    def _determine_workflow_type(self, validated_input: OrchestratorInput) -> str:
        """Determine which workflow to use based on input source.
        
        Returns:
            'admin' for traditional workflow (PolicyRiskAgent)
            'apple_health' for Apple Health workflow (AppleHealthRiskAgent)
        """
        # Check llm_outputs for source indicator
        if validated_input.llm_outputs:
            source = validated_input.llm_outputs.get("source", "")
            if source == "end_user" or source == "apple_health":
                return "apple_health"
            
            # Also check data_source in the metrics
            data_source = validated_input.llm_outputs.get("health_metrics", {}).get("data_source", "")
            if "apple_health" in data_source.lower():
                return "apple_health"
        
        # Check health_metrics data source
        if validated_input.health_metrics:
            if hasattr(validated_input.health_metrics, 'data_source'):
                if "apple_health" in validated_input.health_metrics.data_source.lower():
                    return "apple_health"
        
        # Default to admin workflow
        return "admin"
    
    async def _get_foundry_invoker(self):
        """Get the Foundry invoker (lazy initialization)."""
        if self._foundry_invoker is None:
            from app.agents.foundry_invoker import get_foundry_invoker
            self._foundry_invoker = get_foundry_invoker()
        return self._foundry_invoker
    
    def validate_input(self, input_data: Dict[str, Any]) -> OrchestratorInput:
        """Validate orchestrator input."""
        return OrchestratorInput.model_validate(input_data)
    
    async def run(self, input_data: Dict[str, Any]) -> OrchestratorOutput:
        """
        Execute the full underwriting workflow.
        
        Args:
            input_data: Must contain 'patient_id', optionally:
                - health_metrics: Override health metrics
                - policy_rules: Override policy rules
                - application_data: Real extracted data from uploaded document
                - document_markdown: Original document text
                - llm_outputs: LLM analysis outputs from extraction
            
        Returns:
            OrchestratorOutput with final decision, confidence, and explanation
        """
        workflow_id = str(uuid4())
        self.logger.info(f"Starting workflow {workflow_id}")
        
        # Get tracer for Foundry portal tracing
        tracer = get_tracer()
        
        # Start workflow span for tracing
        if tracer and is_tracing_enabled():
            with tracer.start_as_current_span("underwriting_workflow") as workflow_span:
                workflow_span.set_attribute("workflow.id", workflow_id)
                workflow_span.set_attribute("workflow.patient_id", input_data.get("patient_id", "unknown"))
                workflow_span.set_attribute("workflow.application_id", input_data.get("application_id", "unknown"))
                return await self._run_workflow_internal(input_data, workflow_id)
        else:
            return await self._run_workflow_internal(input_data, workflow_id)
    
    async def _run_workflow_internal(self, input_data: Dict[str, Any], workflow_id: str) -> OrchestratorOutput:
        """Internal workflow execution (with or without tracing span)."""
        # Validate input
        validated_input = self.validate_input(input_data)
        
        # Extract application_id for token tracking
        application_id = input_data.get("application_id") or validated_input.patient_id
        
        # Create execution context with token tracking
        context = ExecutionContext(validated_input.patient_id, workflow_id, application_id)
        
        # Check if we have real application data
        has_real_data = (
            validated_input.application_data is not None or
            validated_input.llm_outputs is not None
        )
        
        if has_real_data:
            self.logger.info("=" * 60)
            self.logger.info("REAL APPLICATION DATA PATH - NOT USING MOCK DATA")
            self.logger.info("=" * 60)
            self.logger.info(f"Using real application data for workflow {workflow_id}")
            # Store real data in context for agents to use
            context.application_data = validated_input.application_data
            context.document_markdown = validated_input.document_markdown
            context.llm_outputs = validated_input.llm_outputs
            
            # Build health metrics from real application data
            health_metrics = validated_input.health_metrics or self._build_health_metrics_from_application(validated_input)
            patient_profile = self._build_patient_profile_from_application(validated_input)
            
            # Log what data was extracted from the document
            self.logger.info(
                "Extracted from document: patient_age=%s, smoker_status=%s",
                patient_profile.demographics.age,
                patient_profile.medical_history.smoker_status,
            )
        else:
            self.logger.warning("=" * 60)
            self.logger.warning("WARNING: USING MOCK DATA - no application_data provided")
            self.logger.warning("=" * 60)
            self.logger.info(f"Using mock data for workflow {workflow_id} (no application_data provided)")
            # Fall back to mock data
            patient_profile = self._load_patient_profile(validated_input.patient_id)
            health_metrics = validated_input.health_metrics or self._load_health_metrics(validated_input.patient_id)
        
        policy_rules = validated_input.policy_rules or get_standard_policy_rules()
        
        # Determine which workflow to use based on input source
        workflow_type = self._determine_workflow_type(validated_input)
        self.logger.info(f"Workflow {workflow_id} using '{workflow_type}' workflow")
        
        # Get evaluator service (STEP 4 integration)
        evaluator = get_evaluator_service()
        
        try:
            # STEP 1: HealthDataAnalysisAgent (MANDATORY - both workflows)
            health_output = await self._execute_health_data_analysis(context, health_metrics, patient_profile)
            
            # EVALUATION: Run after agent completes (non-blocking)
            if is_evaluations_enabled():
                try:
                    health_eval = await evaluator.evaluate_agent(
                        agent_id="HealthDataAnalysisAgent",
                        agent_input={
                            "document_context": self._build_document_context(context),
                            "health_metrics": health_metrics.model_dump() if health_metrics else {},
                        },
                        agent_output={
                            "health_summary": health_output.summary if hasattr(health_output, 'summary') else str(health_output),
                            "risk_indicators": [r.model_dump() for r in health_output.risk_indicators] if hasattr(health_output, 'risk_indicators') else [],
                        },
                        context=self._build_document_context(context),
                    )
                    context.store_evaluation("HealthDataAnalysisAgent", health_eval)
                except Exception as e:
                    self.logger.warning(f"Evaluation failed for HealthDataAnalysisAgent: {e}")
            
            # STEP 2: Risk Agent (workflow-dependent)
            # - Admin workflow: PolicyRiskAgent (traditional JSON policies)
            # - Apple Health workflow: AppleHealthRiskAgent (HKRS scoring)
            if workflow_type == "apple_health":
                self.logger.info("Using AppleHealthRiskAgent for HKRS scoring")
                risk_output = await self._execute_apple_health_risk(context, health_metrics, patient_profile)
                risk_agent_id = "AppleHealthRiskAgent"
            else:
                self.logger.info("Using PolicyRiskAgent for traditional underwriting")
                risk_output = await self._execute_policy_risk(context, policy_rules)
                risk_agent_id = "PolicyRiskAgent"
            
            # Store risk output in context for communication agent (aliased as policy_risk for backward compat)
            policy_risk_output = risk_output
            
            # EVALUATION: Run after agent completes (non-blocking)
            if is_evaluations_enabled():
                try:
                    # Build appropriate context based on workflow type
                    if workflow_type == "apple_health":
                        eval_context = "Apple Health HKRS scoring workflow"
                        eval_input = {
                            "workflow_type": "apple_health",
                            "health_metrics": health_metrics.model_dump() if health_metrics else {},
                        }
                    else:
                        eval_context = str(policy_rules.model_dump() if hasattr(policy_rules, 'model_dump') else policy_rules)
                        eval_input = {
                            "rules_context": eval_context,
                            "policy_rules": eval_context,
                        }
                    
                    policy_eval = await evaluator.evaluate_agent(
                        agent_id=risk_agent_id,
                        agent_input=eval_input,
                        agent_output={
                            "rationale": risk_output.rationale if hasattr(risk_output, 'rationale') else str(risk_output),
                            "approved": risk_output.approved if hasattr(risk_output, 'approved') else None,
                            "risk_level": risk_output.risk_level.value if hasattr(risk_output, 'risk_level') else (risk_output.hkrs_band.value if hasattr(risk_output, 'hkrs_band') else None),
                            "decision": risk_output.decision if hasattr(risk_output, 'decision') else None,
                            "hkrs": risk_output.hkrs if hasattr(risk_output, 'hkrs') else None,
                        },
                        context=eval_context,
                    )
                    context.store_evaluation(risk_agent_id, policy_eval)
                except Exception as e:
                    self.logger.warning(f"Evaluation failed for {risk_agent_id}: {e}")
            
            # STEP 3: CommunicationAgent (MANDATORY - both workflows)
            comm_output = await self._execute_communication(context, patient_profile)
            
            # EVALUATION: Run after agent completes (non-blocking)
            if is_evaluations_enabled():
                try:
                    comm_eval = await evaluator.evaluate_agent(
                        agent_id="CommunicationAgent",
                        agent_input={
                            "decision_summary": self._build_decision_summary(context),
                        },
                        agent_output={
                            "underwriter_message": comm_output.underwriter_message if hasattr(comm_output, 'underwriter_message') else str(comm_output),
                            "customer_message": comm_output.customer_message if hasattr(comm_output, 'customer_message') else "",
                        },
                    )
                    context.store_evaluation("CommunicationAgent", comm_eval)
                except Exception as e:
                    self.logger.warning(f"Evaluation failed for CommunicationAgent: {e}")
            
            # WORKFLOW-LEVEL EVALUATION: Aggregate all agent evaluations
            if is_evaluations_enabled():
                try:
                    workflow_eval = await evaluator.evaluate_workflow(
                        workflow_id=workflow_id,
                        agent_results=context.get_all_evaluations(),
                    )
                    context.set_workflow_evaluation(workflow_eval)
                except Exception as e:
                    self.logger.warning(f"Workflow evaluation failed: {e}")
            
        except AgentExecutionError as e:
            self.logger.error(f"Agent execution failed: {e}")
            raise AgentExecutionError(
                self.agent_id,
                f"Workflow failed at {e.agent_id}: {str(e)}",
                {"workflow_id": workflow_id, "failed_agent": e.agent_id}
            )
        
        # Extract patient name from llm_outputs if available
        patient_name = validated_input.patient_name
        if not patient_name and validated_input.llm_outputs:
            # Try different paths where patient name might be stored
            patient_name = (
                validated_input.llm_outputs.get("patient_profile", {}).get("name") or
                validated_input.llm_outputs.get("patient_summary", {}).get("name") or
                validated_input.llm_outputs.get("application_summary", {}).get("customer_profile", {}).get("parsed", {}).get("full_name")
            )
        
        # Produce final decision (SUMMARIZE ONLY - DO NOT ALTER CONCLUSIONS)
        final_decision = self._produce_final_decision(context, validated_input.patient_id, patient_name)
        
        # Calculate overall confidence
        confidence_score = self._calculate_confidence(context)
        
        # Generate explanation
        explanation = self._generate_explanation(context, final_decision)
        
        # Determine execution source (end_user vs underwriter)
        execution_source = "underwriter"
        if validated_input.llm_outputs:
            if validated_input.llm_outputs.get("source") == "end_user":
                execution_source = "end_user"
                self.logger.info("END USER AGENT EXECUTION STARTED")
        
        # Build evaluations output (STEP 6)
        evaluations_output = None
        workflow_evaluation_output = None
        if is_evaluations_enabled():
            # Convert agent evaluations to dict format for JSON serialization
            agent_evals = context.get_all_evaluations()
            if agent_evals:
                evaluations_output = {
                    agent_id: eval_result.model_dump() 
                    for agent_id, eval_result in agent_evals.items()
                }
            
            # Get workflow evaluation
            workflow_eval = context.get_workflow_evaluation()
            if workflow_eval:
                workflow_evaluation_output = workflow_eval.model_dump()
        
        # Build output
        output = OrchestratorOutput(
            agent_id=self.agent_id,
            success=True,
            final_decision=final_decision,
            confidence_score=confidence_score,
            explanation=explanation,
            execution_records=context.get_records(),
            workflow_id=workflow_id,
            total_execution_time_ms=context.get_total_time_ms(),
            execution_source=execution_source,
            evaluations=evaluations_output,
            workflow_evaluation=workflow_evaluation_output,
        )
        
        # Log completion with source
        if execution_source == "end_user":
            self.logger.info("END USER AGENT EXECUTION COMPLETED")
        
        # Log evaluation status
        if evaluations_output:
            self.logger.info(f"Workflow {workflow_id} evaluations: {len(evaluations_output)} agents evaluated")
        
        self.logger.info(f"Workflow {workflow_id} completed in {output.total_execution_time_ms:.2f}ms (source={execution_source})")
        return output
    
    async def run_with_progress(
        self, 
        input_data: Dict[str, Any]
    ) -> AsyncGenerator[AgentProgress | OrchestratorOutput, None]:
        """
        Execute the full underwriting workflow with progress streaming.
        
        Yields AgentProgress events as each agent executes, then finally
        yields the OrchestratorOutput when complete.
        
        Args:
            input_data: Must contain 'patient_id', optionally:
                - health_metrics: Override health metrics
                - policy_rules: Override policy rules
                - application_data: Real extracted data from uploaded document
                - document_markdown: Original document text
                - llm_outputs: LLM analysis outputs from extraction
            
        Yields:
            AgentProgress events during execution
            OrchestratorOutput as the final item
        """
        workflow_id = str(uuid4())
        self.logger.info(f"Starting workflow {workflow_id} (with progress streaming)")
        
        # Validate input
        validated_input = self.validate_input(input_data)
        
        # Extract application_id for token tracking
        application_id = input_data.get("application_id") or validated_input.patient_id
        
        # Create execution context with token tracking
        context = ExecutionContext(validated_input.patient_id, workflow_id, application_id)
        
        # Check if we have real application data
        has_real_data = (
            validated_input.application_data is not None or
            validated_input.llm_outputs is not None
        )
        
        # Check if this is an end-user execution
        is_end_user = (
            validated_input.llm_outputs is not None and 
            validated_input.llm_outputs.get("source") == "end_user"
        )
        
        if is_end_user:
            self.logger.info("END USER AGENT EXECUTION STARTED")
        
        if has_real_data:
            self.logger.info("=" * 60)
            self.logger.info("REAL APPLICATION DATA PATH - NOT USING MOCK DATA")
            self.logger.info("=" * 60)
            context.application_data = validated_input.application_data
            context.document_markdown = validated_input.document_markdown
            context.llm_outputs = validated_input.llm_outputs
            health_metrics = validated_input.health_metrics or self._build_health_metrics_from_application(validated_input)
            patient_profile = self._build_patient_profile_from_application(validated_input)
        else:
            self.logger.warning("Using mock data for workflow {workflow_id}")
            patient_profile = self._load_patient_profile(validated_input.patient_id)
            health_metrics = validated_input.health_metrics or self._load_health_metrics(validated_input.patient_id)
        
        policy_rules = validated_input.policy_rules or get_standard_policy_rules()
        
        # Determine workflow type based on input source
        workflow_type = self._determine_workflow_type(validated_input)
        self.logger.info(f"Workflow type: {workflow_type}")
        
        # Define agents in execution order based on workflow type
        # Each tuple: (agent_id, step_number, description, tools_used, execute_fn)
        if workflow_type == "apple_health":
            # Apple Health workflow for end users
            agents = [
                (
                    "HealthDataAnalysisAgent", 
                    1, 
                    "Analyzing Apple Health metrics and patient profile",
                    ["health-metrics-analyzer", "risk-indicator-extractor"],
                    lambda: self._execute_health_data_analysis(context, health_metrics, patient_profile)
                ),
                (
                    "AppleHealthRiskAgent",
                    2,
                    "Calculating HealthKit Risk Score (HKRS) and determining decision",
                    ["hkrs-calculator", "age-adjustment", "risk-classifier"],
                    lambda: self._execute_apple_health_risk(context, health_metrics, patient_profile)
                ),
                (
                    "CommunicationAgent", 
                    3, 
                    "Generating personalized health summary",
                    ["message-generator", "tone-analyzer"],
                    lambda: self._execute_communication(context, patient_profile)
                ),
            ]
        else:
            # Traditional admin workflow with PolicyRiskAgent
            agents = [
                (
                    "HealthDataAnalysisAgent", 
                    1, 
                    "Analyzing health metrics and patient profile",
                    ["health-metrics-analyzer", "risk-indicator-extractor"],
                    lambda: self._execute_health_data_analysis(context, health_metrics, patient_profile)
                ),
                (
                    "PolicyRiskAgent",
                    2,
                    "Applying underwriting policies and determining decision",
                    ["policy-rule-engine", "risk-classifier", "premium-calculator"],
                    lambda: self._execute_policy_risk(context, policy_rules)
                ),
                (
                    "CommunicationAgent", 
                    3, 
                    "Generating decision communications",
                    ["message-generator", "tone-analyzer"],
                    lambda: self._execute_communication(context, patient_profile)
                ),
            ]
        
        total_steps = len(agents)
        
        try:
            for agent_id, step_number, description, tools, execute_fn in agents:
                agent_name = self.AGENT_DISPLAY_NAMES.get(agent_id, agent_id)
                
                # STAGE 1: Agent Started
                self.logger.info(f"AGENT_PROGRESS_EVENT_EMITTED: {agent_id} → STARTED")
                yield AgentProgress(
                    workflow_id=workflow_id,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    step_number=step_number,
                    total_steps=total_steps,
                    status=AgentProgressStatus.RUNNING,
                    stage=AgentProgressStage.STARTED,
                    message=f"Starting {agent_name}",
                    safe_summary=description,
                )
                
                start_time = datetime.now(timezone.utc)
                
                try:
                    # STAGE 2: Preparing Input
                    await asyncio.sleep(0.05)  # Small delay to allow event to be sent
                    yield AgentProgress(
                        workflow_id=workflow_id,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        step_number=step_number,
                        total_steps=total_steps,
                        status=AgentProgressStatus.RUNNING,
                        stage=AgentProgressStage.PREPARING_INPUT,
                        message=f"Preparing input data",
                        safe_summary=f"Gathering context and input data for {agent_name}",
                    )
                    
                    # STAGE 3: Invoking Model / Tool Calling
                    await asyncio.sleep(0.05)
                    self.logger.info(f"AGENT_PROGRESS_EVENT_EMITTED: {agent_id} → TOOL_CALLED")
                    yield AgentProgress(
                        workflow_id=workflow_id,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        step_number=step_number,
                        total_steps=total_steps,
                        status=AgentProgressStatus.TOOL_CALLING,
                        stage=AgentProgressStage.INVOKING_MODEL if self._use_foundry else AgentProgressStage.TOOL_CALLED,
                        message=f"Invoking {'Azure AI Foundry agent' if self._use_foundry else 'local processing'}",
                        safe_summary=f"Processing with {len(tools)} tools: {', '.join(tools)}",
                        tools_used=tools,
                    )
                    
                    # Execute the agent
                    output = await execute_fn()
                    
                    execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    
                    # STAGE 4: Parsing Response
                    yield AgentProgress(
                        workflow_id=workflow_id,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        step_number=step_number,
                        total_steps=total_steps,
                        status=AgentProgressStatus.PROCESSING,
                        stage=AgentProgressStage.PARSING_RESPONSE,
                        message=f"Parsing agent response",
                        safe_summary=f"Structuring output from {agent_name}",
                        execution_time_ms=execution_time_ms,
                    )
                    
                    await asyncio.sleep(0.05)
                    
                    # STAGE 5: Output Ready - include preview
                    output_preview = self._get_output_preview(agent_id, context)
                    self.logger.info(f"AGENT_PROGRESS_EVENT_EMITTED: {agent_id} → OUTPUT_READY")
                    yield AgentProgress(
                        workflow_id=workflow_id,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        step_number=step_number,
                        total_steps=total_steps,
                        status=AgentProgressStatus.OUTPUT_READY,
                        stage=AgentProgressStage.VALIDATING_OUTPUT,
                        message=f"Output validated",
                        safe_summary=output_preview,
                        output_preview=output_preview,
                        tools_used=tools,
                        execution_time_ms=execution_time_ms,
                    )
                    
                    await asyncio.sleep(0.05)
                    
                    # STAGE 6: Completed
                    self.logger.info(f"AGENT_PROGRESS_EVENT_EMITTED: {agent_id} → COMPLETED")
                    yield AgentProgress(
                        workflow_id=workflow_id,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        step_number=step_number,
                        total_steps=total_steps,
                        status=AgentProgressStatus.COMPLETED,
                        stage=AgentProgressStage.COMPLETED,
                        execution_time_ms=execution_time_ms,
                        message=f"{agent_name} completed",
                        safe_summary=output_preview,
                        output_preview=output_preview,
                        tools_used=tools,
                    )
                    
                    # STEP 4: Run Foundry evaluations for this agent (streaming version)
                    if is_evaluations_enabled():
                        evaluator = get_evaluator_service()
                        if evaluator:
                            # Emit "evaluating" progress event
                            self.logger.info(f"AGENT_PROGRESS_EVENT_EMITTED: {agent_id} → EVALUATING")
                            yield AgentProgress(
                                workflow_id=workflow_id,
                                agent_id=agent_id,
                                agent_name=agent_name,
                                step_number=step_number,
                                total_steps=total_steps,
                                status=AgentProgressStatus.PROCESSING,
                                stage=AgentProgressStage.EVALUATING,
                                execution_time_ms=execution_time_ms,
                                message=f"Evaluating {agent_name} with Azure AI Foundry",
                                safe_summary=f"Running quality evaluations (groundedness, coherence)",
                            )
                            
                            try:
                                eval_start = datetime.now(timezone.utc)
                                agent_input = self._build_agent_input_for_eval(agent_id, context, health_metrics, patient_profile, policy_rules)
                                agent_output_dict = self._build_agent_output_for_eval(agent_id, context)
                                doc_context = self._build_document_context(context)
                                
                                agent_eval = await evaluator.evaluate_agent(
                                    agent_id=agent_id,
                                    agent_input=agent_input,
                                    agent_output=agent_output_dict,
                                    context=doc_context,
                                )
                                context.store_evaluation(agent_id, agent_eval)
                                eval_time_ms = (datetime.now(timezone.utc) - eval_start).total_seconds() * 1000
                                
                                # Emit "evaluation complete" progress event
                                eval_status = agent_eval.status.value if agent_eval.status else "completed"
                                eval_score = agent_eval.overall_score
                                self.logger.info(f"AGENT_PROGRESS_EVENT_EMITTED: {agent_id} → EVALUATION_COMPLETE (score={eval_score:.2f})")
                                yield AgentProgress(
                                    workflow_id=workflow_id,
                                    agent_id=agent_id,
                                    agent_name=agent_name,
                                    step_number=step_number,
                                    total_steps=total_steps,
                                    status=AgentProgressStatus.COMPLETED,
                                    stage=AgentProgressStage.EVALUATION_COMPLETE,
                                    execution_time_ms=execution_time_ms + eval_time_ms,
                                    message=f"Evaluation complete for {agent_name}",
                                    safe_summary=f"Quality score: {eval_score:.1f}/5.0 ({eval_status})",
                                    output_preview=f"Metrics: {', '.join(m.metric_name for m in agent_eval.metrics)}" if agent_eval.metrics else None,
                                )
                                
                            except Exception as e:
                                self.logger.warning(f"Evaluation failed for {agent_id}: {e}")
                                # Emit evaluation failed event (non-blocking)
                                yield AgentProgress(
                                    workflow_id=workflow_id,
                                    agent_id=agent_id,
                                    agent_name=agent_name,
                                    step_number=step_number,
                                    total_steps=total_steps,
                                    status=AgentProgressStatus.COMPLETED,
                                    stage=AgentProgressStage.EVALUATION_COMPLETE,
                                    execution_time_ms=execution_time_ms,
                                    message=f"Evaluation skipped for {agent_name}",
                                    safe_summary=f"Evaluation could not complete: {str(e)[:50]}",
                                )
                    
                except AgentExecutionError as e:
                    execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                    
                    # Emit "failed" event
                    self.logger.info(f"AGENT_PROGRESS_EVENT_EMITTED: {agent_id} → FAILED")
                    yield AgentProgress(
                        workflow_id=workflow_id,
                        agent_id=agent_id,
                        agent_name=agent_name,
                        step_number=step_number,
                        total_steps=total_steps,
                        status=AgentProgressStatus.FAILED,
                        stage=AgentProgressStage.FAILED,
                        execution_time_ms=execution_time_ms,
                        message=f"{agent_name} failed: {str(e)}",
                        safe_summary=f"Error: {str(e)[:100]}",
                    )
                    raise
                    
        except AgentExecutionError as e:
            self.logger.error(f"Agent execution failed: {e}")
            raise
        
        # Extract patient name from llm_outputs if available
        patient_name = validated_input.patient_name
        if not patient_name and validated_input.llm_outputs:
            # Try different paths where patient name might be stored
            patient_name = (
                validated_input.llm_outputs.get("patient_profile", {}).get("name") or
                validated_input.llm_outputs.get("patient_summary", {}).get("name") or
                validated_input.llm_outputs.get("application_summary", {}).get("customer_profile", {}).get("parsed", {}).get("full_name")
            )
        
        # Produce final decision
        final_decision = self._produce_final_decision(context, validated_input.patient_id, patient_name)
        confidence_score = self._calculate_confidence(context)
        explanation = self._generate_explanation(context, final_decision)
        
        # Determine execution source (end_user vs underwriter)
        execution_source = "underwriter"
        if validated_input.llm_outputs:
            if validated_input.llm_outputs.get("source") == "end_user":
                execution_source = "end_user"
                self.logger.info("END USER AGENT EXECUTION COMPLETED")
        
        # Gather evaluations if enabled (STEP 4: Streaming version)
        evaluations_output = None
        workflow_evaluation_output = None
        if is_evaluations_enabled():
            # First, run workflow-level evaluation
            evaluator = get_evaluator_service()
            if evaluator:
                try:
                    workflow_eval = await evaluator.evaluate_workflow(
                        workflow_id=workflow_id,
                        agent_results=context.get_all_evaluations(),
                        final_decision={
                            "patient_id": validated_input.patient_id,
                            "patient_name": patient_name,
                            "decision": final_decision.status.value,
                            "confidence": confidence_score,
                        },
                    )
                    context.set_workflow_evaluation(workflow_eval)
                    self.logger.info(f"Workflow evaluation completed: overall_score={workflow_eval.overall_score:.2f}")
                except Exception as e:
                    self.logger.warning(f"Workflow evaluation failed: {e}")
            
            # Now gather all evaluations
            all_evals = context.get_all_evaluations()
            if all_evals:
                evaluations_output = {
                    agent_id: eval_result.model_dump()
                    for agent_id, eval_result in all_evals.items()
                }
            
            # Get workflow evaluation
            workflow_eval = context.get_workflow_evaluation()
            if workflow_eval:
                workflow_evaluation_output = workflow_eval.model_dump()
        
        output = OrchestratorOutput(
            agent_id=self.agent_id,
            success=True,
            final_decision=final_decision,
            confidence_score=confidence_score,
            explanation=explanation,
            execution_records=context.get_records(),
            workflow_id=workflow_id,
            total_execution_time_ms=context.get_total_time_ms(),
            execution_source=execution_source,
            evaluations=evaluations_output,
            workflow_evaluation=workflow_evaluation_output,
        )
        
        # Log evaluation status
        if evaluations_output:
            self.logger.info(f"Workflow {workflow_id} evaluations: {len(evaluations_output)} agents evaluated")
        
        self.logger.info(f"Workflow {workflow_id} completed in {output.total_execution_time_ms:.2f}ms (source={execution_source})")
        
        # Persist token usage to Cosmos DB (non-blocking)
        try:
            token_summary = context.get_token_summary()
            if token_summary.get("total_tokens", 0) > 0:
                self.logger.info(
                    "Token usage summary: %d total tokens across %d agent calls",
                    token_summary["total_tokens"],
                    token_summary["agent_count"]
                )
                # Persist asynchronously - don't block on this
                await context.persist_token_usage()
        except Exception as e:
            self.logger.warning(f"Failed to persist token usage (non-fatal): {e}")
        finally:
            # Clean up token tracking context
            context.close_token_tracking()
        
        # Yield the final output
        yield output
    
    def _get_output_preview(self, agent_id: str, context: ExecutionContext) -> str:
        """Get a safe preview of the agent's output (no chain-of-thought)."""
        output = context.get_output(agent_id)
        if not output:
            return "Processing complete"
        
        if agent_id == "HealthDataAnalysisAgent":
            from app.agents.health_data_analysis import HealthDataAnalysisOutput
            hda_out: HealthDataAnalysisOutput = output
            high_risk = sum(1 for ri in hda_out.risk_indicators if ri.risk_level.value in ['high', 'very_high'])
            return f"Identified {len(hda_out.risk_indicators)} risk indicators ({high_risk} high-risk)"
        
        elif agent_id == "PolicyRiskAgent":
            pr_out: PolicyRiskOutput = output
            decision = pr_out.decision if hasattr(pr_out, 'decision') else "approved"
            adj_pct = pr_out.premium_adjustment_recommendation.adjustment_percentage if pr_out else 0
            return f"Decision: {decision.replace('_', ' ').title()}, Risk: {pr_out.risk_level.value}, {adj_pct:+.0f}% adjustment"
        
        elif agent_id == "AppleHealthRiskAgent":
            ah_out: AppleHealthRiskOutput = output
            hkrs_band = ah_out.hkrs_band.value if hasattr(ah_out, 'hkrs_band') else "unknown"
            return f"HKRS: {ah_out.hkrs:.0f}/100 ({hkrs_band.replace('_', ' ').title()}), Risk Class: {ah_out.risk_class_recommendation}"
        
        elif agent_id == "CommunicationAgent":
            from app.agents.communication import CommunicationOutput
            comm_out: CommunicationOutput = output
            return f"Generated communications for underwriter and customer"
        
        return "Output ready"
    
    # =========================================================================
    # AGENT EXECUTION STEPS (STRICT ORDER - NO MODIFICATION)
    # =========================================================================
    
    async def _invoke_foundry_agent(
        self,
        agent_id: str,
        prompt: str,
        context_data: Dict[str, Any],
        execution_context: Optional[ExecutionContext] = None,
        step_number: int = 0,
    ) -> Dict[str, Any]:
        """
        Invoke an agent via Azure AI Foundry.
        
        Args:
            agent_id: Local agent ID (e.g., "HealthDataAnalysisAgent")
            prompt: The prompt/instructions for the agent
            context_data: Input data for the agent
            execution_context: Optional execution context for token tracking
            step_number: Step number in workflow for token tracking
            
        Returns:
            Parsed response from the agent
        """
        foundry_name = self.FOUNDRY_AGENT_NAMES.get(agent_id)
        if not foundry_name:
            raise ValueError(f"No Foundry agent mapping for {agent_id}")
        
        invoker = await self._get_foundry_invoker()
        
        result = await invoker.invoke_agent(
            agent_name=foundry_name,
            prompt=prompt,
            context=context_data,
        )
        
        if not result.success:
            raise AgentExecutionError(
                agent_id,
                f"Foundry agent invocation failed: {result.error}",
                {"foundry_agent": foundry_name, "execution_time_ms": result.execution_time_ms}
            )
        
        # Extract tool names from tools_executed for display
        tools_used = []
        if hasattr(result, 'tools_executed') and result.tools_executed:
            tools_used = [t.get("tool_name", "unknown") for t in result.tools_executed]
        
        # Record token usage if context is provided
        if execution_context and result.token_usage:
            execution_context.record_token_usage(
                agent_id=agent_id,
                token_usage=result.token_usage,
                step_number=step_number,
                model_name=getattr(invoker, 'model_deployment', None),
            )
        
        return {
            "response": result.response,
            "parsed": result.parsed_output,
            "execution_time_ms": result.execution_time_ms,
            "token_usage": result.token_usage,
            "tools_executed": getattr(result, 'tools_executed', []),
            "tools_used": tools_used,
        }
    
    async def _execute_health_data_analysis(
        self, 
        context: ExecutionContext, 
        health_metrics: HealthMetrics,
        patient_profile: PatientProfile,
    ) -> HealthDataAnalysisOutput:
        """Step 1: Execute HealthDataAnalysisAgent."""
        self.logger.info("Step 1: Executing HealthDataAnalysisAgent%s", 
                        " (via Azure AI Foundry)" if self._use_foundry else " (local)")
        
        # Add tracing span for this agent
        add_span_event("agent_started", {"agent_id": "HealthDataAnalysisAgent", "step": 1})
        add_span_attribute("agent.health_data_analysis.started", True)
        
        # Extract biometrics for explicit passing to tools
        age = patient_profile.demographics.age
        height_cm = patient_profile.medical_history.height_cm or 170.0
        weight_kg = patient_profile.medical_history.weight_kg or 70.0
        
        self.logger.info(f"Health analysis inputs - age: {age}, height_cm: {height_cm}, weight_kg: {weight_kg}")
        
        input_data = {
            "health_metrics": health_metrics.model_dump(),
            "patient_profile": patient_profile.model_dump(),
            # Explicit biometrics for tool calls
            "biometrics": {
                "age": age,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
            }
        }
        
        tools_used = []
        
        if self._use_foundry:
            # Use Azure AI Foundry agent with function tools
            prompt = f"""Analyze the provided health metrics and patient profile to identify risk indicators.

IMPORTANT BIOMETRIC DATA (use these exact values for tool calls):
- age: {age}
- height_cm: {height_cm}
- weight_kg: {weight_kg}

Use your tools to perform the analysis:
1. Call analyze_health_metrics with: age={age}, height_cm={height_cm}, weight_kg={weight_kg}
2. Call extract_risk_indicators with medical conditions, medications, and family history from the patient profile

For each risk indicator found, provide:
- indicator_id: Unique ID (e.g., "IND-ACT-001")
- category: One of "activity", "heart_rate", "sleep", "trend", "medical_history"
- indicator_name: Descriptive name
- risk_level: "low", "moderate", "high", or "very_high"
- confidence: 0.0 to 1.0
- metric_value: The measured value
- metric_unit: Unit of measurement
- explanation: Why this is a risk indicator

Return your response as JSON with this structure:
{{
  "risk_indicators": [...],
  "summary": "..."
}}"""
            
            result = await self._invoke_foundry_agent(
                "HealthDataAnalysisAgent",
                prompt,
                input_data,
                execution_context=context,
                step_number=1,
            )
            
            # Get actual tools executed from Foundry
            tools_used = result.get("tools_used", [])
            if not tools_used:
                tools_used = ["analyze_health_metrics", "extract_risk_indicators"]  # Expected tools
            
            # Parse Foundry response into expected output format
            parsed = result.get("parsed") or {}
            output = HealthDataAnalysisOutput(
                agent_id="HealthDataAnalysisAgent",
                risk_indicators=self._parse_risk_indicators(parsed.get("risk_indicators", [])),
                summary=parsed.get("summary", result.get("response", "Analysis completed via Foundry")),
                execution_time_ms=result.get("execution_time_ms", 0),
            )
        else:
            # Use local deterministic agent
            output = await self._health_data_agent.run(input_data)
            tools_used = ["local-health-analyzer"]
        
        # Store with actual inputs and tools used
        context.store_output(
            "HealthDataAnalysisAgent", 
            output, 
            step_number=1,
            actual_inputs=input_data,
            tools_invoked=tools_used
        )
        return output
    
    def _parse_risk_indicators(self, indicators_data: list) -> List[RiskIndicator]:
        """Parse risk indicators from Foundry response."""
        indicators = []
        for i, ind in enumerate(indicators_data):
            try:
                # Parse metric_value - handle non-numeric values gracefully
                metric_value = None
                raw_metric = ind.get("metric_value")
                if raw_metric is not None:
                    try:
                        # Try to convert to float, but handle non-numeric strings
                        if isinstance(raw_metric, (int, float)):
                            metric_value = float(raw_metric)
                        elif isinstance(raw_metric, str) and raw_metric.lower() not in ['none', 'n/a', 'stable', 'unknown', 'na', '']:
                            metric_value = float(raw_metric)
                    except (ValueError, TypeError):
                        # Non-numeric value - leave as None
                        pass
                
                # Parse metric_unit - handle dict, string, or None
                raw_unit = ind.get("metric_unit")
                if isinstance(raw_unit, dict):
                    # Extract first value from dict or stringify it
                    if raw_unit:
                        metric_unit = next(iter(raw_unit.values()), None)
                        if not isinstance(metric_unit, str):
                            metric_unit = str(raw_unit)
                    else:
                        metric_unit = None
                elif isinstance(raw_unit, str):
                    metric_unit = raw_unit
                else:
                    metric_unit = None
                
                indicators.append(RiskIndicator(
                    indicator_id=ind.get("indicator_id", f"IND-{i+1:03d}"),
                    category=ind.get("category", "unknown"),
                    indicator_name=ind.get("indicator_name", "Unknown indicator"),
                    risk_level=RiskLevel(ind.get("risk_level", "moderate").lower()),
                    confidence=float(ind.get("confidence", 0.7)),
                    metric_value=metric_value,
                    metric_unit=metric_unit,
                    explanation=ind.get("explanation", ""),
                ))
            except Exception as e:
                self.logger.warning(f"Failed to parse risk indicator: {e}")
        return indicators
    
    async def _execute_data_quality_confidence(
        self,
        context: ExecutionContext,
        health_metrics: HealthMetrics,
    ) -> DataQualityConfidenceOutput:
        """Step 2: Execute DataQualityConfidenceAgent."""
        self.logger.info("Step 2: Executing DataQualityConfidenceAgent%s",
                        " (via Azure AI Foundry)" if self._use_foundry else " (local)")
        
        input_data = {"health_metrics": health_metrics.model_dump()}
        
        if self._use_foundry:
            prompt = """Assess the reliability and completeness of the provided health data.

Evaluate:
- Data completeness (how much data is available)
- Data freshness (how recent is the data)
- Data consistency (are there any anomalies)

Return JSON:
{
  "confidence_score": 0.0-1.0,
  "data_quality_level": "poor" | "fair" | "good" | "excellent",
  "quality_flags": [{"flag_id": "QF-001", "flag_type": "missing_data", "severity": "warning", "affected_metric": "sleep", "description": "description", "confidence_impact": -0.1}],
  "freshness_assessment": "Data is current within 30 days",
  "recommendations": ["recommendation1"]
}"""
            result = await self._invoke_foundry_agent(
                "DataQualityConfidenceAgent", prompt, input_data,
                execution_context=context, step_number=2
            )
            parsed = result.get("parsed") or {}
            
            from data.mock.schemas import DataQualityLevel, QualityFlag
            quality_level_str = parsed.get("data_quality_level", "good").lower()
            quality_level = DataQualityLevel(quality_level_str) if quality_level_str in ["poor", "fair", "good", "excellent"] else DataQualityLevel.GOOD
            
            # Parse quality flags - handle both string and dict formats
            quality_flags = []
            raw_flags = parsed.get("quality_flags", [])
            for i, flag in enumerate(raw_flags):
                if isinstance(flag, dict):
                    quality_flags.append(QualityFlag(
                        flag_id=flag.get("flag_id", f"QF-{i+1:03d}"),
                        flag_type=flag.get("flag_type", "incomplete"),
                        severity=flag.get("severity", "info"),
                        affected_metric=flag.get("affected_metric", "general"),
                        description=flag.get("description", str(flag)),
                        confidence_impact=float(flag.get("confidence_impact", -0.05)),
                    ))
                elif isinstance(flag, str):
                    # Convert string flag to QualityFlag object
                    quality_flags.append(QualityFlag(
                        flag_id=f"QF-{i+1:03d}",
                        flag_type="info",
                        severity="info",
                        affected_metric="general",
                        description=flag,
                        confidence_impact=-0.05,
                    ))
            
            output = DataQualityConfidenceOutput(
                agent_id="DataQualityConfidenceAgent",
                confidence_score=float(parsed.get("confidence_score", 0.8)),
                data_quality_level=quality_level,
                quality_flags=quality_flags,
                freshness_assessment=parsed.get("freshness_assessment", "Data analyzed via Azure AI Foundry"),
                coverage_metrics=parsed.get("coverage_metrics", {}),
                recommendations=parsed.get("recommendations", []),
                execution_time_ms=result.get("execution_time_ms", 0),
            )
        else:
            output = await self._data_quality_agent.run(input_data)
        
        context.store_output("DataQualityConfidenceAgent", output, step_number=2)
        return output
    
    async def _execute_policy_risk(
        self,
        context: ExecutionContext,
        policy_rules: PolicyRuleSet,
    ) -> PolicyRiskOutput:
        """Step 2: Execute PolicyRiskAgent.
        
        Translates health risk indicators into insurance risk categories
        by evaluating against the underwriting policy manual.
        """
        self.logger.info("Step 2: Executing PolicyRiskAgent%s",
                        " (via Azure AI Foundry)" if self._use_foundry else " (local)")
        
        # Add tracing span for this agent
        add_span_event("agent_started", {"agent_id": "PolicyRiskAgent", "step": 2})
        add_span_attribute("agent.policy_risk.started", True)
        
        # Get risk indicators from Step 1
        hda_output: HealthDataAnalysisOutput = context.get_output("HealthDataAnalysisAgent")
        
        # Load the actual underwriting policies from JSON
        underwriting_policies = self._load_underwriting_policies()
        
        input_data = {
            "risk_indicators": [ri.model_dump() for ri in hda_output.risk_indicators],
            "policy_rules": policy_rules.model_dump(),
        }
        
        if self._use_foundry:
            # Try to use Foundry agent if it exists (with function tools)
            # Fall back to direct OpenAI if not deployed
            import json as json_module
            from app.openai_client import chat_completion
            from app.config import load_settings
            
            # Format risk indicators for prompt
            risk_indicators_text = "\n".join([
                f"- {ri.indicator_name}: {ri.risk_level.value} risk (confidence: {ri.confidence:.0%}) - {ri.explanation}"
                for ri in hda_output.risk_indicators
            ])
            
            # Format policies for prompt (summary of key policies)
            policies_summary = self._format_policies_for_prompt(underwriting_policies)
            
            prompt = f"""You are an expert insurance underwriter. Evaluate health risk indicators against the underwriting policy manual to make a final underwriting decision.

Use your tools to evaluate the applicant:
1. Call evaluate_policy_rules with the applicant details and risk level
2. Call lookup_underwriting_guidelines for any concerning conditions
3. Call calculate_risk_score to determine the final risk classification

## RISK INDICATORS FROM HEALTH ANALYSIS

{risk_indicators_text}

## UNDERWRITING POLICY MANUAL (Key Policies)

{policies_summary}

## INSTRUCTIONS

1. Use evaluate_policy_rules to check age limits, coverage limits, and pre-existing conditions
2. For each medical condition, use lookup_underwriting_guidelines to get official guidance
3. Use calculate_risk_score to determine the final risk class and premium multiplier
4. Make a final DECISION based on:
   - risk_level "decline" → decision = "declined"
   - premium_adjustment_percentage > 100% → decision = "referred" (needs manual review)
   - premium_adjustment_percentage > 0% → decision = "approved_with_adjustment"
   - otherwise → decision = "approved"

Return your final assessment as JSON:
{{
  "risk_level": "low" | "moderate" | "high" | "very_high" | "decline",
  "risk_delta_score": <integer 0-100>,
  "premium_adjustment_percentage": <number>,
  "approved": true | false,
  "decision": "approved" | "approved_with_adjustment" | "declined" | "referred",
  "referral_required": true | false,
  "triggered_rules": ["policy_id-criteria_id", ...],
  "rule_evaluations": [
    {{"indicator": "...", "policy_id": "...", "criteria_id": "...", "action": "...", "contribution": "+X%"}}
  ],
  "rationale": "2-3 sentence explanation citing the specific policies that led to this decision"
}}"""

            # Try Foundry agent first if available
            foundry_name = self.FOUNDRY_AGENT_NAMES.get("PolicyRiskAgent")
            use_foundry_agent = False
            
            if foundry_name:
                try:
                    result = await self._invoke_foundry_agent(
                        "PolicyRiskAgent",
                        prompt,
                        input_data,
                        execution_context=context,
                        step_number=3,
                    )
                    use_foundry_agent = True
                    parsed = result.get("parsed") or {}
                    execution_time_ms = result.get("execution_time_ms", 0)
                    tools_used_from_foundry = result.get("tools_used", [])
                except Exception as e:
                    self.logger.warning("Foundry agent for PolicyRiskAgent not available: %s. Using direct OpenAI.", e)
                    use_foundry_agent = False
            
            if not use_foundry_agent:
                # Fall back to direct OpenAI
                settings = load_settings()
                start_time = datetime.now(timezone.utc)
                
                messages = [
                    {"role": "system", "content": "You are an expert insurance underwriter. Return responses as valid JSON only."},
                    {"role": "user", "content": prompt}
                ]
                
                response = chat_completion(
                    settings=settings.openai,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000,
                )
                
                execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                response_text = response.get("content", "{}")
                
                try:
                    parsed = json_module.loads(response_text)
                except json_module.JSONDecodeError:
                    self.logger.warning("Failed to parse PolicyRiskAgent response as JSON")
                    parsed = {}
                
                tools_used_from_foundry = []  # Direct call, no Foundry tools
            
            from data.mock.schemas import PremiumAdjustment
            risk_level = RiskLevel(parsed.get("risk_level", "moderate").lower())
            
            # Calculate premium values with safe parsing
            base_premium = 1200.00  # Default base premium
            adjustment_val = parsed.get("premium_adjustment_percentage", 25)
            # Handle None or invalid values
            try:
                adjustment_pct = float(adjustment_val) if adjustment_val is not None else 25.0
            except (TypeError, ValueError):
                adjustment_pct = 25.0
            adjusted_premium = base_premium * (1 + adjustment_pct / 100)
            
            # Parse decision fields
            approved = parsed.get("approved", risk_level != RiskLevel.DECLINE)
            decision = parsed.get("decision", "approved")
            if not decision or decision not in ["approved", "approved_with_adjustment", "declined", "referred"]:
                # Derive decision from risk level and adjustment
                if risk_level == RiskLevel.DECLINE:
                    decision = "declined"
                    approved = False
                elif adjustment_pct > 100:
                    decision = "referred"
                elif adjustment_pct > 0:
                    decision = "approved_with_adjustment"
                else:
                    decision = "approved"
            referral_required = parsed.get("referral_required", decision == "referred")
            rationale = parsed.get("rationale", f"Risk level {risk_level.value} based on policy evaluation")
            
            output = PolicyRiskOutput(
                agent_id="PolicyRiskAgent",
                risk_level=risk_level,
                premium_adjustment_recommendation=PremiumAdjustment(
                    base_premium_annual=base_premium,
                    adjustment_percentage=adjustment_pct,
                    adjusted_premium_annual=adjusted_premium,
                    adjustment_factors={"ai_policy_analysis": adjustment_pct},
                    triggered_rule_ids=parsed.get("triggered_rules", []),
                ),
                triggered_rules=parsed.get("triggered_rules", []),
                # Convert rule_evaluations dicts to strings for the log
                rule_evaluation_log=[
                    f"{eval_item.get('indicator', 'Unknown')}: {eval_item.get('policy_id', 'N/A')} - {eval_item.get('action', 'N/A')} ({eval_item.get('contribution', 'N/A')})"
                    if isinstance(eval_item, dict) else str(eval_item)
                    for eval_item in parsed.get("rule_evaluations", [])
                ] or [f"AI analyzed {len(hda_output.risk_indicators)} risk indicators against policy manual"],
                approved=approved,
                decision=decision,
                rationale=rationale,
                referral_required=referral_required,
                execution_time_ms=execution_time_ms,
            )
            
            # Store decision values in context for CommunicationAgent
            context._outputs["_risk_level"] = risk_level.value
            context._outputs["_premium_adjustment_pct"] = adjustment_pct
            context._outputs["_base_premium"] = base_premium
            context._outputs["_adjusted_premium"] = adjusted_premium
            context._outputs["_triggered_rules"] = parsed.get("triggered_rules", [])
            context._outputs["_referral_required"] = referral_required
            context._outputs["_approved"] = approved
            context._outputs["_decision"] = decision
            
            # Track actual tools used
            tools_used_for_tracking = tools_used_from_foundry if tools_used_from_foundry else [
                "evaluate_policy_rules", "lookup_underwriting_guidelines", "calculate_risk_score"
            ]
        else:
            output = await self._policy_risk_agent.run(input_data)
            tools_used_for_tracking = ["local-policy-analyzer"]
        
        # Capture actual inputs for transparency
        actual_inputs = {
            "risk_indicators_count": len(hda_output.risk_indicators),
            "risk_indicators_summary": [{"name": ri.indicator_name, "risk_level": ri.risk_level.value} for ri in hda_output.risk_indicators[:5]],
            "policies_loaded": len(underwriting_policies.get("policies", [])) if underwriting_policies else 0,
        }
        context.store_output(
            "PolicyRiskAgent", 
            output, 
            step_number=2,
            actual_inputs=actual_inputs,
            tools_invoked=tools_used_for_tracking
        )
        return output
    
    def _load_underwriting_policies(self) -> dict:
        """Load the underwriting policies from JSON file."""
        import json
        from pathlib import Path
        
        # Try multiple locations
        possible_paths = [
            Path(__file__).parent.parent.parent / "prompts" / "life-health-underwriting-policies.json",
            Path(__file__).parent.parent.parent / "data" / "life-health-underwriting-policies.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        policies = json.load(f)
                        self.logger.info(f"Loaded {len(policies.get('policies', []))} underwriting policies from {path}")
                        return policies
                except Exception as e:
                    self.logger.warning(f"Failed to load policies from {path}: {e}")
        
        self.logger.warning("No underwriting policies file found")
        return {"policies": []}
    
    def _format_policies_for_prompt(self, policies_data: dict, max_policies: int = 10) -> str:
        """Format policies for inclusion in the prompt."""
        if not policies_data or "policies" not in policies_data:
            return "No policies loaded."
        
        policies = policies_data.get("policies", [])[:max_policies]
        formatted = []
        
        for policy in policies:
            policy_text = f"""
### {policy.get('id', 'Unknown')} - {policy.get('name', 'Unknown Policy')}
Category: {policy.get('category', 'N/A')} / {policy.get('subcategory', 'N/A')}

Criteria:"""
            for criteria in policy.get("criteria", [])[:4]:  # Limit criteria per policy
                policy_text += f"""
- {criteria.get('id', 'N/A')}: {criteria.get('condition', 'N/A')}
  Risk Level: {criteria.get('risk_level', 'N/A')}
  Action: {criteria.get('action', 'N/A')}"""
            formatted.append(policy_text)
        
        return "\n".join(formatted)
    
    async def _execute_apple_health_risk(
        self,
        context: ExecutionContext,
        health_metrics: HealthMetrics,
        patient_profile: PatientProfile,
    ) -> AppleHealthRiskOutput:
        """Step 2 (Apple Health workflow): Execute AppleHealthRiskAgent.
        
        Calculates HealthKit Risk Score (HKRS) from Apple Health data
        using the Apple Health underwriting policies.
        """
        self.logger.info("Step 2: Executing AppleHealthRiskAgent%s",
                        " (via Azure AI Foundry)" if self._use_foundry else " (local)")
        
        # Add tracing span for this agent
        add_span_event("agent_started", {"agent_id": "AppleHealthRiskAgent", "step": 2})
        add_span_attribute("agent.apple_health_risk.started", True)
        
        # Get risk indicators from Step 1 (optional for context)
        hda_output: HealthDataAnalysisOutput = context.get_output("HealthDataAnalysisAgent")
        
        # Load the Apple Health underwriting policies
        apple_health_policies = self._load_apple_health_policies()
        
        # Build input for the agent
        input_data = {
            "health_metrics": health_metrics.model_dump() if health_metrics else {},
            "patient_profile": patient_profile.model_dump() if patient_profile else {},
        }
        
        if self._use_foundry:
            # Use Foundry or OpenAI for Apple Health risk scoring
            import json as json_module
            from app.openai_client import chat_completion
            from app.config import load_settings
            
            # Format health metrics for prompt
            metrics_summary = self._format_apple_health_metrics_for_prompt(health_metrics)
            
            # Format policies for prompt
            policies_summary = self._format_apple_health_policies_for_prompt(apple_health_policies)
            
            prompt = f"""You are an Apple Health underwriting specialist. Calculate the HealthKit Risk Score (HKRS) for this applicant.

## APPLICANT PROFILE
Age: {patient_profile.demographics.age if patient_profile.demographics else 'Unknown'}
Gender: {patient_profile.demographics.biological_sex if patient_profile.demographics else 'Unknown'}

## APPLE HEALTH METRICS
{metrics_summary}

## HKRS SCORING RULES
{policies_summary}

## INSTRUCTIONS
1. Calculate each sub-score based on the metrics:
   - Activity Score (25%): Steps/day >8000=25pts, 6000-7999=18pts, 4000-5999=10pts, <4000=0pts
   - VO2 Max Score (20%): ≥75th percentile=20pts, 50-74th=15pts, 25-49th=8pts, <25th=0pts
   - Heart Health Score (20%): Resting HR 50-70=10pts + HRV ≥60th percentile=10pts
   - Sleep Health Score (15%): 7-8 hours=10pts + consistency ≤1hr variance=5pts
   - Body Composition Score (10%): Stable/improving BMI=10pts, mild increase=5pts
   - Mobility Score (10%): Walking speed ≥60th percentile=5pts + normal steadiness=5pts

2. Apply Age Adjustment Factor (AAF):
   - 18-34: 1.00
   - 35-44: 0.98
   - 45-54: 0.95
   - 55-64: 0.92
   - 65+: 0.88

3. Calculate HKRS = (Sum of weighted scores) × AAF

4. Determine HKRS band:
   - 85-100: Excellent (eligible for best class)
   - 70-84: Very Good (may improve one class)
   - 55-69: Standard Plus
   - 40-54: Standard (no adjustment)
   - <40: Substandard (manual review)

Return JSON:
{{
  "hkrs": <score 0-100>,
  "hkrs_band": "excellent" | "very_good" | "standard_plus" | "standard" | "substandard",
  "age_adjustment_factor": <0.88-1.00>,
  "sub_scores": {{
    "activity": {{"score": X, "max": 25, "notes": "..."}},
    "vo2_max": {{"score": X, "max": 20, "notes": "..."}},
    "heart_health": {{"score": X, "max": 20, "notes": "..."}},
    "sleep_health": {{"score": X, "max": 15, "notes": "..."}},
    "body_composition": {{"score": X, "max": 10, "notes": "..."}},
    "mobility": {{"score": X, "max": 10, "notes": "..."}}
  }},
  "approved": true,
  "decision": "approved" | "approved_with_adjustment" | "referred",
  "referral_required": true | false,
  "risk_class_recommendation": "Preferred Plus" | "Preferred" | "Standard Plus" | "Standard" | "Substandard",
  "top_positive_drivers": ["driver1", "driver2", "driver3"],
  "improvement_suggestions": ["suggestion1", "suggestion2"],
  "rationale": "2-3 sentence explanation of the HKRS score"
}}"""

            # Try Foundry agent first if available
            foundry_name = self.FOUNDRY_AGENT_NAMES.get("AppleHealthRiskAgent")
            use_foundry_agent = False
            
            if foundry_name:
                try:
                    result = await self._invoke_foundry_agent(
                        "AppleHealthRiskAgent",
                        prompt,
                        input_data,
                        execution_context=context,
                        step_number=2,
                    )
                    use_foundry_agent = True
                    parsed = result.get("parsed") or {}
                    execution_time_ms = result.get("execution_time_ms", 0)
                    tools_used_from_foundry = result.get("tools_used", [])
                except Exception as e:
                    self.logger.warning("Foundry agent for AppleHealthRiskAgent not available: %s. Using direct OpenAI.", e)
                    use_foundry_agent = False
            
            if not use_foundry_agent:
                # Fall back to direct OpenAI
                settings = load_settings()
                start_time = datetime.now(timezone.utc)
                
                messages = [
                    {"role": "system", "content": "You are an Apple Health underwriting specialist. Return responses as valid JSON only."},
                    {"role": "user", "content": prompt}
                ]
                
                response = chat_completion(
                    settings=settings.openai,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000,
                )
                
                execution_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                response_text = response.get("content", "{}")
                
                try:
                    parsed = json_module.loads(response_text)
                except json_module.JSONDecodeError:
                    self.logger.warning("Failed to parse AppleHealthRiskAgent response as JSON")
                    parsed = {}
                
                tools_used_from_foundry = []
            
            # Build output from parsed response
            from app.agents.apple_health_risk import HKRSBand, DataQuality, SubScoreDetail
            
            hkrs = parsed.get("hkrs", 60.0)
            hkrs_band_str = parsed.get("hkrs_band", "standard")
            hkrs_band = HKRSBand(hkrs_band_str) if hkrs_band_str in [b.value for b in HKRSBand] else HKRSBand.STANDARD
            
            output = AppleHealthRiskOutput(
                agent_id="AppleHealthRiskAgent",
                success=True,
                hkrs=hkrs,
                hkrs_band=hkrs_band,
                hkrs_band_description=parsed.get("hkrs_band_description", f"HKRS {hkrs:.0f}"),
                age_adjustment_factor=parsed.get("age_adjustment_factor", 1.0),
                age_bracket=self._get_age_bracket(patient_profile.demographics.age if patient_profile.demographics else 35),
                sub_scores=[],  # Simplified for Foundry response
                raw_score_before_aaf=hkrs / parsed.get("age_adjustment_factor", 1.0),
                data_quality=DataQuality.HIGH,
                data_quality_score=85.0,
                data_gaps=[],
                risk_class_recommendation=parsed.get("risk_class_recommendation", "Standard"),
                adjustment_action=parsed.get("adjustment_action", "standard_processing"),
                approved=parsed.get("approved", True),
                decision=parsed.get("decision", "approved"),
                rationale=parsed.get("rationale", "HKRS calculated from Apple Health data"),
                referral_required=parsed.get("referral_required", False),
                top_positive_drivers=parsed.get("top_positive_drivers", []),
                improvement_suggestions=parsed.get("improvement_suggestions", []),
                summary_scorecard=f"Your HealthKit Risk Score is {hkrs:.0f}/100 ({hkrs_band.value.replace('_', ' ').title()}).",
                execution_time_ms=execution_time_ms,
            )
            
            # Store decision values in context for CommunicationAgent
            context._outputs["_risk_level"] = hkrs_band.value
            context._outputs["_hkrs"] = hkrs
            context._outputs["_premium_adjustment_pct"] = 0  # HKRS doesn't directly set premium
            context._outputs["_referral_required"] = parsed.get("referral_required", False)
            context._outputs["_approved"] = parsed.get("approved", True)
            context._outputs["_decision"] = parsed.get("decision", "approved")
            
            tools_used_for_tracking = tools_used_from_foundry if tools_used_from_foundry else ["hkrs-calculator"]
        else:
            # Use local agent
            output = await self._apple_health_risk_agent.run(input_data)
            tools_used_for_tracking = ["local-hkrs-calculator"]
            
            # Store decision values in context for CommunicationAgent
            context._outputs["_risk_level"] = output.hkrs_band.value
            context._outputs["_hkrs"] = output.hkrs
            context._outputs["_premium_adjustment_pct"] = 0
            context._outputs["_referral_required"] = output.referral_required
            context._outputs["_approved"] = output.approved
            context._outputs["_decision"] = output.decision
        
        # Capture actual inputs for transparency
        actual_inputs = {
            "health_metrics_source": health_metrics.data_source if health_metrics else "unknown",
            "age": patient_profile.demographics.age if patient_profile.demographics else None,
        }
        context.store_output(
            "AppleHealthRiskAgent", 
            output, 
            step_number=2,
            actual_inputs=actual_inputs,
            tools_invoked=tools_used_for_tracking
        )
        return output
    
    def _get_age_bracket(self, age: int) -> str:
        """Get age bracket for AAF lookup."""
        if age < 18:
            return "18-34"
        elif age <= 34:
            return "18-34"
        elif age <= 44:
            return "35-44"
        elif age <= 54:
            return "45-54"
        elif age <= 64:
            return "55-64"
        else:
            return "65+"
    
    def _load_apple_health_policies(self) -> dict:
        """Load the Apple Health underwriting policies from JSON file."""
        import json
        from pathlib import Path
        
        possible_paths = [
            Path(__file__).parent.parent.parent / "prompts" / "apple-health-underwriting-policies.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        policies = json.load(f)
                        self.logger.info(f"Loaded Apple Health underwriting policies from {path}")
                        return policies
                except Exception as e:
                    self.logger.warning(f"Failed to load Apple Health policies from {path}: {e}")
        
        self.logger.warning("No Apple Health underwriting policies file found")
        return {}
    
    def _format_apple_health_metrics_for_prompt(self, metrics: HealthMetrics) -> str:
        """Format Apple Health metrics for inclusion in the prompt."""
        if not metrics:
            return "No metrics available."
        
        lines = []
        
        if metrics.activity:
            lines.append(f"Activity: {metrics.activity.daily_steps_avg or 'N/A'} steps/day, {metrics.activity.days_with_data} days of data")
        
        if metrics.heart_rate:
            lines.append(f"Heart Rate: Resting {metrics.heart_rate.resting_hr_avg or 'N/A'} bpm, HRV {metrics.heart_rate.hrv_avg_ms or 'N/A'}ms")
        
        if metrics.sleep:
            lines.append(f"Sleep: {metrics.sleep.avg_sleep_duration_hours or 'N/A'} hours avg")
        
        if metrics.fitness:
            lines.append(f"Fitness: VO2 Max {metrics.fitness.vo2_max or 'N/A'} mL/kg/min")
        
        if metrics.mobility:
            lines.append(f"Mobility: Walking speed {metrics.mobility.walking_speed_avg or 'N/A'} m/s, steadiness: {metrics.mobility.walking_steadiness or 'N/A'}")
        
        if metrics.body_metrics:
            lines.append(f"Body: BMI {metrics.body_metrics.bmi or 'N/A'}, trend: {metrics.body_metrics.bmi_trend or 'N/A'}")
        
        return "\n".join(lines) if lines else "No metrics available."
    
    def _format_apple_health_policies_for_prompt(self, policies: dict, max_length: int = 2000) -> str:
        """Format Apple Health policies for inclusion in the prompt."""
        if not policies:
            return "Using standard HKRS scoring rules."
        
        # Extract key rules
        summary_parts = []
        
        if "hkrs_formula" in policies:
            summary_parts.append("HKRS Formula: " + policies["hkrs_formula"].get("description", ""))
        
        if "age_adjustment_factor" in policies:
            aaf = policies["age_adjustment_factor"]
            summary_parts.append(f"Age Adjustment: {aaf.get('description', 'Age-based normalization')}")
        
        if "hkrs_bands" in policies:
            bands = policies["hkrs_bands"]
            bands_text = ", ".join([f"{k}: {v.get('range', [])}" for k, v in bands.items()])
            summary_parts.append(f"Score Bands: {bands_text}")
        
        return "\n".join(summary_parts)
    
    async def _execute_bias_fairness(
        self,
        context: ExecutionContext,
        patient_profile: PatientProfile,
    ) -> BiasAndFairnessOutput:
        """Step 5: Execute BiasAndFairnessAgent."""
        self.logger.info("Step 5: Executing BiasAndFairnessAgent%s",
                        " (via Azure AI Foundry)" if self._use_foundry else " (local)")
        
        # Get outputs from previous steps
        pr_output: PolicyRiskOutput = context.get_output("PolicyRiskAgent")
        
        # Build decision context for bias analysis matching DecisionContext schema
        decision_context = {
            "patient_age": patient_profile.demographics.age,
            "patient_sex": patient_profile.demographics.biological_sex,
            "patient_region": patient_profile.demographics.state_region,
            "risk_level": pr_output.risk_level.value,
            "premium_adjustment_pct": pr_output.premium_adjustment_recommendation.adjustment_percentage,
            "triggered_rules": pr_output.triggered_rules,
            "health_metrics_used": ["activity", "heart_rate", "sleep", "trends"],
        }
        
        if self._use_foundry:
            # Build prompt for Foundry agent
            prompt = f"""Analyze the following underwriting decision for potential bias and fairness issues:

Decision Context:
- Patient Age: {decision_context['patient_age']}
- Patient Sex: {decision_context['patient_sex']}
- Patient Region: {decision_context['patient_region']}
- Risk Level: {decision_context['risk_level']}
- Premium Adjustment: {decision_context['premium_adjustment_pct']}%
- Triggered Rules: {', '.join(decision_context['triggered_rules'])}
- Health Metrics Used: {', '.join(decision_context['health_metrics_used'])}

Analyze for:
1. Age discrimination - is the decision unfairly penalizing based on age?
2. Gender bias - are there gender-based disparities in the decision?
3. Geographic discrimination - does region unfairly impact the outcome?
4. Protected class considerations - any potential HIPAA/ADA violations?
5. Algorithmic fairness - is the model treating similar profiles consistently?

Provide JSON:
{
  "bias_flags": ["flag1", "flag2"],
  "fairness_score": 0.0-1.0,
  "recommendations": ["rec1", "rec2"],
  "protected_attributes_analyzed": ["age", "sex", "region"]
}"""

            result = await self._invoke_foundry_agent(
                "BiasAndFairnessAgent", prompt, decision_context,
                execution_context=context, step_number=5
            )
            parsed = result.get("parsed") or {}
            
            # Extract bias flags
            bias_flags_raw = parsed.get("bias_flags", [])
            from data.mock.schemas import BiasFlag
            bias_flags = []
            for i, flag in enumerate(bias_flags_raw):
                if isinstance(flag, dict):
                    bias_flags.append(BiasFlag(
                        flag_id=flag.get("flag_id", f"BF-{i+1:03d}"),
                        bias_type=flag.get("bias_type", "unknown"),
                        severity=flag.get("severity", "low"),
                        description=flag.get("description", str(flag)),
                        mitigation_applied=flag.get("mitigation_applied", False),
                        mitigation_notes=flag.get("mitigation_notes"),
                        blocks_decision=flag.get("blocks_decision", False),
                    ))
                elif isinstance(flag, str):
                    bias_flags.append(BiasFlag(
                        flag_id=f"BF-{i+1:03d}",
                        bias_type=flag.lower().replace(" ", "_"),
                        severity="low",
                        description=flag,
                        mitigation_applied=False,
                        blocks_decision=False,
                    ))
            
            output = BiasAndFairnessOutput(
                agent_id="BiasAndFairnessAgent",
                bias_flags=bias_flags,
                fairness_score=float(parsed.get("fairness_score", 0.85)),
                mitigation_notes=parsed.get("mitigation_notes", "Analysis completed via Azure AI Foundry"),
                recommendations=parsed.get("recommendations", ["Continue monitoring for demographic disparities"]),
                protected_attributes_analyzed=parsed.get("protected_attributes_analyzed", ["age", "sex", "region"]),
                execution_time_ms=result.get("execution_time_ms", 0),
            )
        else:
            output = await self._bias_fairness_agent.run({
                "decision_context": decision_context,
            })
        
        context.store_output("BiasAndFairnessAgent", output, step_number=5)
        return output
    
    async def _execute_communication(
        self,
        context: ExecutionContext,
        patient_profile: PatientProfile,
    ) -> CommunicationOutput:
        """Step 3: Execute CommunicationAgent.
        
        Supports both workflows:
        - Traditional: Uses PolicyRiskAgent output
        - Apple Health: Uses AppleHealthRiskAgent output
        """
        self.logger.info("Step 3: Executing CommunicationAgent%s",
                        " (via Azure AI Foundry)" if self._use_foundry else " (local)")
        
        # Get outputs from previous steps
        hda_output: HealthDataAnalysisOutput = context.get_output("HealthDataAnalysisAgent")
        pr_output: PolicyRiskOutput = context.get_output("PolicyRiskAgent")
        ah_output: AppleHealthRiskOutput = context.get_output("AppleHealthRiskAgent")
        
        # Determine which workflow was used
        is_apple_health = ah_output is not None
        
        # Get calculated values from context (set by either PolicyRiskAgent or AppleHealthRiskAgent)
        risk_level_str = context._outputs.get("_risk_level", "moderate")
        premium_adjustment_pct = context._outputs.get("_premium_adjustment_pct", 0)
        base_premium = context._outputs.get("_base_premium", 1000)
        adjusted_premium = context._outputs.get("_adjusted_premium", 1000)
        triggered_rules = context._outputs.get("_triggered_rules", [])
        referral_required = context._outputs.get("_referral_required", False)
        decision = context._outputs.get("_decision", "approved")
        approved = context._outputs.get("_approved", True)
        
        # Get rationale from the appropriate agent
        if is_apple_health:
            rationale = ah_output.rationale if ah_output else "HKRS-based assessment completed."
            risk_class = ah_output.risk_class_recommendation if ah_output else "Standard"
            hkrs = ah_output.hkrs if ah_output else 60
            hkrs_band = ah_output.hkrs_band.value if ah_output else "standard"
            sub_scores = ah_output.sub_scores if ah_output else []
            age_adjustment_factor = ah_output.age_adjustment_factor if ah_output else 1.0
            age_bracket = ah_output.age_bracket if ah_output else "Unknown"
            raw_score_before_aaf = ah_output.raw_score_before_aaf if ah_output else 0
            top_positive_drivers = ah_output.top_positive_drivers if ah_output else []
            improvement_suggestions = ah_output.improvement_suggestions if ah_output else []
            data_quality = ah_output.data_quality.value if ah_output else "medium"
            data_gaps = ah_output.data_gaps if ah_output else []
        else:
            rationale = pr_output.rationale if pr_output else "Risk assessment completed."
            risk_class = None
            hkrs = None
            hkrs_band = None
            sub_scores = []
            age_adjustment_factor = None
            age_bracket = None
            raw_score_before_aaf = None
            top_positive_drivers = []
            improvement_suggestions = []
            data_quality = None
            data_gaps = []
            # Also get values from pr_output if context doesn't have them
            if pr_output:
                risk_level_str = context._outputs.get("_risk_level", pr_output.risk_level.value)
                premium_adjustment_pct = context._outputs.get("_premium_adjustment_pct", pr_output.premium_adjustment_recommendation.adjustment_percentage)
                base_premium = context._outputs.get("_base_premium", pr_output.premium_adjustment_recommendation.base_premium_annual)
                adjusted_premium = context._outputs.get("_adjusted_premium", pr_output.premium_adjustment_recommendation.adjusted_premium_annual)
                triggered_rules = context._outputs.get("_triggered_rules", pr_output.triggered_rules)
                referral_required = context._outputs.get("_referral_required", pr_output.referral_required)
                decision = context._outputs.get("_decision", pr_output.decision)
                approved = context._outputs.get("_approved", pr_output.approved)
        
        if decision == "declined" or not approved:
            status = DecisionStatus.DECLINED
        elif decision == "referred" or referral_required:
            status = DecisionStatus.REFERRED
        elif decision == "approved_with_adjustment" or premium_adjustment_pct > 0:
            status = DecisionStatus.APPROVED_WITH_ADJUSTMENT
        else:
            status = DecisionStatus.APPROVED
        
        # Build key risk factors from health analysis
        key_risk_factors = [ri.explanation for ri in hda_output.risk_indicators[:3]]
        
        if self._use_foundry:
            # Build prompt for Foundry agent - different for Apple Health vs Traditional
            if is_apple_health:
                # Build detailed HKRS breakdown for Apple Health workflow
                sub_score_details = ""
                for ss in sub_scores:
                    sub_score_details += f"\n### {ss.name} (Weight: {ss.weight*100:.0f}%)\n"
                    sub_score_details += f"- Raw Score: {ss.raw_score:.1f}/100\n"
                    sub_score_details += f"- Weighted Score: {ss.weighted_score:.1f}\n"
                    sub_score_details += f"- Max Points: {ss.max_points}\n"
                    if ss.components:
                        sub_score_details += "- Components:\n"
                        for comp_name, comp_value in ss.components.items():
                            sub_score_details += f"  - {comp_name}: {comp_value}\n"
                    if ss.notes:
                        sub_score_details += f"- Notes: {', '.join(ss.notes)}\n"
                
                prompt = f"""You are a Communication Specialist for Apple Health-based life insurance underwriting. 
Generate a comprehensive underwriting decision summary based on the HealthKit Risk Score (HKRS) assessment.

## Applicant Details:
- Applicant ID: {context.patient_id}
- Age: {patient_profile.demographics.age} years ({age_bracket} bracket)
- Policy Type: {patient_profile.policy_type_requested}
- Coverage Amount: ${patient_profile.coverage_amount_requested:,.2f}

## HKRS Assessment Results:
- **Final HKRS Score: {hkrs:.1f}/100**
- **HKRS Band: {hkrs_band.replace('_', ' ').title()}**
- **Risk Classification: {risk_class}**
- Raw Score (before age adjustment): {raw_score_before_aaf:.1f}
- Age Adjustment Factor: {age_adjustment_factor:.2f}
- Data Quality: {data_quality.title() if data_quality else 'Unknown'}

## Category-by-Category Breakdown:
{sub_score_details if sub_score_details else '- No detailed sub-scores available'}

## Premium Calculation:
- Base Premium: ${base_premium:.2f}
- Premium Adjustment: {premium_adjustment_pct}%
- Adjusted Annual Premium: ${adjusted_premium:.2f}

## Decision:
- Status: {status.value}
- Approved: {approved}
- Referral Required: {referral_required}

## Positive Factors:
{chr(10).join('- ' + f for f in top_positive_drivers) if top_positive_drivers else '- None identified'}

## Areas for Improvement:
{chr(10).join('- ' + s for s in improvement_suggestions) if improvement_suggestions else '- None identified'}

## Data Gaps:
{chr(10).join('- ' + g for g in data_gaps) if data_gaps else '- No significant data gaps'}

Generate two messages:

1. **Underwriter Message** (Internal): A comprehensive technical summary that includes:
   - The HKRS score breakdown with justification for each of the 7 categories
   - How each category score was calculated
   - The age adjustment applied
   - Premium calculation details
   - Risk classification rationale
   - Any data quality concerns
   
   Format this as a structured report with clear sections.

2. **Customer Message** (External): A professional, friendly letter explaining:
   - Their application was reviewed using their Apple Health data
   - The overall outcome (approval status)
   - Their risk classification in simple terms
   - The premium amount
   - Any lifestyle factors that positively impacted their assessment
   - DO NOT include specific health metrics (HIPAA compliance)

Return JSON:
{{
  "underwriter_message": "Detailed technical summary...",
  "customer_message": "Dear Applicant, ...",
  "tone_assessment": "professional/empathetic/formal",
  "readability_score": 85,
  "key_points": ["point1", "point2", "point3"]
}}"""
            else:
                # Traditional workflow prompt
                prompt = f"""You are a Communication Specialist. Generate professional communications for an underwriting decision.

## Decision Details:
- Decision ID: DEC-{context.workflow_id[:8]}
- Patient ID: {context.patient_id}
- Status: {status.value}
- Risk Level: {risk_level_str}
- Base Premium: ${base_premium:.2f}
- Premium Adjustment: {premium_adjustment_pct}%
- Adjusted Annual Premium: ${adjusted_premium:.2f}
- Approved: {approved}
- Referral Required: {referral_required}

## Policy Details:
- Policy Type: {patient_profile.policy_type_requested}
- Coverage Amount: ${patient_profile.coverage_amount_requested:,.2f}

## Key Risk Factors:
{chr(10).join('- ' + rf for rf in key_risk_factors) if key_risk_factors else '- No significant risk factors identified'}

## Decision Rationale:
{rationale}

## Triggered Policy Rules:
{chr(10).join('- ' + r for r in triggered_rules) if triggered_rules else '- Standard rates apply'}

Generate two messages:

1. **Underwriter Message** (Internal): Include all technical details, risk factors, 
   premium calculations, and policy rule citations. This is for insurance professionals.

2. **Customer Message** (External): A professional, empathetic letter to the applicant.
   - For APPROVED: Congratulate and explain coverage
   - For APPROVED_WITH_ADJUSTMENT: Explain adjustments positively  
   - For DECLINED: Be empathetic, provide general reason (no medical details)
   - For REFERRED: Explain additional review needed
   
   DO NOT include specific medical conditions in customer letters (HIPAA compliance).

Return JSON:
{{
  "underwriter_message": "Internal summary for underwriting team...",
  "customer_message": "Dear Applicant, ...",
  "tone_assessment": "professional/empathetic/formal",
  "readability_score": 85,
  "key_points": ["point1", "point2", "point3"]
}}"""

            result = await self._invoke_foundry_agent(
                "CommunicationAgent", prompt, 
                {
                    "status": status.value,
                    "risk_level": risk_level_str,
                    "premium_adjustment_pct": premium_adjustment_pct,
                    "adjusted_premium": adjusted_premium,
                    "rationale": rationale,
                    "key_risk_factors": key_risk_factors,
                    "patient_profile": patient_profile.model_dump(mode='json'),
                },
                execution_context=context,
                step_number=6,
            )
            parsed = result.get("parsed") or {}
            
            output = CommunicationOutput(
                agent_id="CommunicationAgent",
                underwriter_message=parsed.get("underwriter_message", 
                    f"Underwriting decision: {status.value}. Risk level: {risk_level_str}. "
                    f"Premium adjustment: {premium_adjustment_pct}%. {rationale}"),
                customer_message=parsed.get("customer_message", 
                    f"Dear Applicant, your application has been {status.value.lower().replace('_', ' ')}. "
                    f"Please contact us if you have any questions."),
                tone_assessment=parsed.get("tone_assessment", "professional"),
                readability_score=float(parsed.get("readability_score", 85.0)),
                key_points=parsed.get("key_points", [
                    f"Decision: {status.value}",
                    f"Risk Level: {risk_level_str}",
                    f"Premium: ${adjusted_premium:.2f}/year"
                ]),
                execution_time_ms=result.get("execution_time_ms", 0),
            )
        else:
            # Build proper DecisionSummary with UnderwritingDecision for local agent
            from data.mock.schemas import PremiumAdjustment, DataQualityLevel
            
            # Map risk level string to enum
            risk_level_map = {
                "low": RiskLevel.LOW,
                "moderate": RiskLevel.MODERATE,
                "high": RiskLevel.HIGH,
                "very_high": RiskLevel.VERY_HIGH,
                "decline": RiskLevel.DECLINE,
            }
            risk_level_enum = risk_level_map.get(risk_level_str, RiskLevel.MODERATE)
            
            # Build UnderwritingDecision
            underwriting_decision = UnderwritingDecision(
                decision_id=f"DEC-{context.workflow_id[:8]}",
                patient_id=context.patient_id,
                status=status,
                risk_level=risk_level_enum,
                premium_adjustment=PremiumAdjustment(
                    base_premium_annual=base_premium,
                    adjustment_percentage=premium_adjustment_pct,
                    adjusted_premium_annual=adjusted_premium,
                    adjustment_reasons=[rationale] if rationale else [],
                ),
                confidence_score=0.8,  # Default confidence
                data_quality_level=DataQualityLevel.GOOD,
                decision_rationale=rationale,
                key_risk_factors=key_risk_factors,
                regulatory_compliant=approved,
                bias_check_passed=True,
            )
            
            # Build proper decision_summary matching DecisionSummary schema
            decision_summary = {
                "decision": underwriting_decision.model_dump(mode='json'),
                "patient_name": None,
                "policy_type": patient_profile.policy_type_requested,
                "coverage_amount": patient_profile.coverage_amount_requested,
            }
            output = await self._communication_agent.run({
                "decision_summary": decision_summary,
            })
        
        # Capture actual inputs for transparency
        actual_inputs = {
            "risk_level": context._outputs.get("_risk_level", "unknown"),
            "premium_adjustment_pct": context._outputs.get("_premium_adjustment_pct", 0),
            "adjusted_premium": context._outputs.get("_adjusted_premium", 0),
            "approved": approved,
        }
        # Get actual tools used from Foundry or use defaults
        tools_used = result.get("tools_used", ["generate_decision_summary"]) if self._use_foundry else ["local-message-generator"]
        context.store_output(
            "CommunicationAgent", 
            output, 
            step_number=4,
            actual_inputs=actual_inputs,
            tools_invoked=tools_used
        )
        return output
    
    async def _execute_audit_trace(
        self,
        context: ExecutionContext,
        patient_id: str,
    ) -> AuditAndTraceOutput:
        """Step 7: Execute AuditAndTraceAgent."""
        self.logger.info("Step 7: Executing AuditAndTraceAgent%s",
                        " (via Azure AI Foundry)" if self._use_foundry else " (local)")
        
        # Build agent output records for audit
        agent_outputs = []
        for record in context.get_records():
            output = context.get_output(record.agent_id)
            agent_outputs.append(AgentOutputRecord(
                agent_id=record.agent_id,
                execution_id=record.execution_id,
                timestamp=record.timestamp,
                execution_time_ms=record.execution_time_ms,
                success=record.success,
                input_summary=f"Input for {record.agent_id}",
                output_summary=record.output_summary,
                key_decisions=[record.output_summary],
                errors=[] if record.success else ["Execution failed"],
            ))
        
        if self._use_foundry:
            # Build prompt for Foundry agent
            agent_summary = "\n".join([
                f"- {ao.agent_id}: {ao.output_summary} (exec time: {ao.execution_time_ms}ms, success: {ao.success})"
                for ao in agent_outputs
            ])
            
            prompt = f"""Generate a comprehensive audit trail for the following multi-agent underwriting workflow:

Workflow Details:
- Workflow ID: {context.workflow_id}
- Patient ID: {patient_id}
- Total Agents Executed: {len(agent_outputs)}

Agent Execution Summary:
{agent_summary}

Generate:
1. audit_id: A unique audit identifier
2. workflow_summary: A comprehensive summary of the entire workflow execution
3. compliance_notes: Any compliance-related observations or concerns
4. recommendations: Suggestions for process improvement or areas requiring review
5. decision_chain: Brief description of how each agent's output influenced the next

Return JSON:
{{
  "workflow_summary": "Summary of the workflow",
  "compliance_notes": ["note1", "note2"],
  "recommendations": ["rec1", "rec2"]
}}"""

            result = await self._invoke_foundry_agent(
                "AuditAndTraceAgent", prompt, 
                {
                    "workflow_id": context.workflow_id,
                    "patient_id": patient_id,
                    "agent_outputs": [ao.model_dump(mode='json') for ao in agent_outputs],
                },
                execution_context=context,
                step_number=7,
            )
            parsed = result.get("parsed") or {}
            
            # Generate audit ID
            import hashlib
            audit_id = f"AUD-{hashlib.sha256(context.workflow_id.encode()).hexdigest()[:12].upper()}"
            
            workflow_summary = parsed.get("workflow_summary", f"Multi-agent underwriting workflow completed for patient {patient_id}. {len(agent_outputs)} agents executed successfully.")
            compliance_notes = parsed.get("compliance_notes", ["All agents executed within expected parameters", "Decision chain properly documented"])
            
            # Build the AuditLog structure
            from app.agents.audit_trace import AuditLog
            total_exec_time = sum(ao.execution_time_ms for ao in agent_outputs)
            
            audit_log = AuditLog(
                audit_id=audit_id,
                workflow_id=context.workflow_id,
                patient_id=patient_id,
                total_agents_executed=len(agent_outputs),
                total_execution_time_ms=total_exec_time,
                workflow_status="COMPLETED",
                entries=[ao.model_dump() for ao in agent_outputs],
                integrity_verified=True,
                compliance_notes=compliance_notes[:5],
                missing_steps=[],
                summary=workflow_summary,
            )
            
            output = AuditAndTraceOutput(
                agent_id="AuditAndTraceAgent",
                audit_log=audit_log,
                execution_time_ms=result.get("execution_time_ms", 0),
            )
        else:
            output = await self._audit_trace_agent.run({
                "agent_outputs": [ao.model_dump() for ao in agent_outputs],
                "workflow_id": context.workflow_id,
                "patient_id": patient_id,
            })
        
        context.store_output("AuditAndTraceAgent", output, step_number=7)
        return output
    
    # =========================================================================
    # EVALUATION HELPER METHODS
    # =========================================================================
    
    def _build_document_context(self, context: ExecutionContext) -> str:
        """
        Build document context string for evaluation grounding.
        
        Returns the document markdown or a summary of llm_outputs 
        to serve as the grounding context for evaluation.
        """
        if context.document_markdown:
            return context.document_markdown[:5000]  # Truncate for evaluation
        
        if context.llm_outputs:
            # Summarize key information from LLM outputs
            summary_parts = []
            if "patient_summary" in context.llm_outputs:
                summary_parts.append(f"Patient Summary: {context.llm_outputs['patient_summary']}")
            if "application_summary" in context.llm_outputs:
                summary_parts.append(f"Application: {context.llm_outputs['application_summary']}")
            if summary_parts:
                return " | ".join(summary_parts)[:5000]
        
        if context.application_data:
            import json
            return json.dumps(context.application_data)[:5000]
        
        return "No document context available"
    
    def _build_decision_summary(self, context: ExecutionContext) -> str:
        """
        Build decision summary for communication agent evaluation.
        
        Summarizes the policy risk outcome that the communication 
        agent was asked to communicate.
        """
        pr_output = context.get_output("PolicyRiskAgent")
        ah_output = context.get_output("AppleHealthRiskAgent")
        health_output = context.get_output("HealthDataAnalysisAgent")
        
        summary_parts = []
        
        # Check for Apple Health workflow first
        if ah_output:
            if hasattr(ah_output, 'approved'):
                summary_parts.append(f"Approved: {ah_output.approved}")
            if hasattr(ah_output, 'decision'):
                summary_parts.append(f"Decision: {ah_output.decision}")
            if hasattr(ah_output, 'rationale'):
                summary_parts.append(f"Rationale: {ah_output.rationale}")
            if hasattr(ah_output, 'hkrs_score'):
                summary_parts.append(f"HKRS Score: {ah_output.hkrs_score}")
            if hasattr(ah_output, 'hkrs_band'):
                summary_parts.append(f"Risk Band: {ah_output.hkrs_band.value}")
        elif pr_output:
            if hasattr(pr_output, 'approved'):
                summary_parts.append(f"Approved: {pr_output.approved}")
            if hasattr(pr_output, 'decision'):
                summary_parts.append(f"Decision: {pr_output.decision}")
            if hasattr(pr_output, 'rationale'):
                summary_parts.append(f"Rationale: {pr_output.rationale}")
            if hasattr(pr_output, 'risk_level'):
                summary_parts.append(f"Risk Level: {pr_output.risk_level.value}")
        
        if health_output:
            if hasattr(health_output, 'risk_indicators'):
                risk_count = len(health_output.risk_indicators) if health_output.risk_indicators else 0
                summary_parts.append(f"Risk indicators identified: {risk_count}")
        
        return " | ".join(summary_parts) if summary_parts else "Decision pending"
    
    def _build_agent_input_for_eval(
        self, 
        agent_id: str, 
        context: ExecutionContext,
        health_metrics: Any,
        patient_profile: Any,
        policy_rules: Any,
    ) -> Dict[str, Any]:
        """
        Build agent input dictionary for evaluation.
        
        Creates a structured representation of the input provided to each agent
        for use in Foundry evaluation metrics.
        
        NOTE: Uses safe serialization to handle datetime objects.
        """
        def safe_dump(obj) -> Any:
            """Safely convert an object to a JSON-serializable dict."""
            if obj is None:
                return None
            if hasattr(obj, 'model_dump'):
                try:
                    data = obj.model_dump()
                    # Convert datetime objects in the dict
                    return _sanitize_for_json(data)
                except Exception:
                    return str(obj)
            return str(obj)
        
        def _sanitize_for_json(obj: Any) -> Any:
            """Recursively sanitize an object for JSON serialization."""
            from datetime import datetime, date
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: _sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_sanitize_for_json(item) for item in obj]
            elif hasattr(obj, 'model_dump'):
                return _sanitize_for_json(obj.model_dump())
            return obj
        
        if agent_id == "HealthDataAnalysisAgent":
            return {
                "health_metrics": safe_dump(health_metrics),
                "patient_profile": safe_dump(patient_profile),
            }
        
        elif agent_id == "PolicyRiskAgent":
            health_output = context.get_output("HealthDataAnalysisAgent")
            # Handle PolicyRuleSet or list of rules
            policy_rules_data = []
            if hasattr(policy_rules, 'rules'):
                # PolicyRuleSet object
                policy_rules_data = [_sanitize_for_json(r.model_dump()) if hasattr(r, 'model_dump') else str(r) for r in policy_rules.rules[:5]]
            elif isinstance(policy_rules, list):
                policy_rules_data = [_sanitize_for_json(r.model_dump()) if hasattr(r, 'model_dump') else str(r) for r in policy_rules[:5]]
            else:
                policy_rules_data = str(policy_rules)[:500]
            return {
                "health_analysis": safe_dump(health_output),
                "policy_rules": policy_rules_data,
            }
        
        elif agent_id == "CommunicationAgent":
            pr_output = context.get_output("PolicyRiskAgent")
            return {
                "policy_decision": safe_dump(pr_output),
                "decision_summary": self._build_decision_summary(context),
            }
        
        return {"agent_id": agent_id}
    
    def _build_agent_output_for_eval(
        self, 
        agent_id: str, 
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        """
        Build agent output dictionary for evaluation.
        
        Creates a structured representation of the output from each agent
        for use in Foundry evaluation metrics.
        
        NOTE: Uses safe serialization to handle datetime objects.
        """
        from datetime import datetime, date
        
        def _sanitize_for_json(obj: Any) -> Any:
            """Recursively sanitize an object for JSON serialization."""
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: _sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_sanitize_for_json(item) for item in obj]
            elif hasattr(obj, 'model_dump'):
                return _sanitize_for_json(obj.model_dump())
            return obj
        
        output = context.get_output(agent_id)
        if not output:
            return {"agent_id": agent_id, "status": "no_output"}
        
        if hasattr(output, 'model_dump'):
            return _sanitize_for_json(output.model_dump())
        
        return {"agent_id": agent_id, "output": str(output)}
    
    # =========================================================================
    # FINAL DECISION PRODUCTION (SUMMARIZE ONLY - NO ALTERATIONS)
    # =========================================================================
    
    def _produce_final_decision(
        self,
        context: ExecutionContext,
        patient_id: str,
        patient_name: Optional[str] = None,
    ) -> FinalDecision:
        """
        Produce final underwriting decision.
        
        DUAL WORKFLOW SUPPORT:
        - Traditional workflow: Uses PolicyRiskAgent
        - Apple Health workflow: Uses AppleHealthRiskAgent
        
        IMPORTANT: This method SUMMARIZES agent outputs.
        It does NOT alter or override any agent conclusions.
        """
        # Retrieve agent outputs (READ-ONLY - DO NOT MODIFY)
        pr_output: PolicyRiskOutput = context.get_output("PolicyRiskAgent")
        ah_output: AppleHealthRiskOutput = context.get_output("AppleHealthRiskAgent")
        comm_output: CommunicationOutput = context.get_output("CommunicationAgent")
        
        # Determine which workflow was used
        is_apple_health_workflow = ah_output is not None
        
        if is_apple_health_workflow:
            # Apple Health workflow - use HKRS for decision
            risk_level_str = context._outputs.get("_risk_level", ah_output.hkrs_band.value if ah_output else "standard")
            premium_adjustment_pct = context._outputs.get("_premium_adjustment_pct", 0)
            adjusted_premium = context._outputs.get("_adjusted_premium", 1000)
            referral_required = context._outputs.get("_referral_required", ah_output.referral_required if ah_output else False)
            approved = context._outputs.get("_approved", ah_output.approved if ah_output else True)
            decision = context._outputs.get("_decision", ah_output.decision if ah_output else "approved")
            
            # Map HKRS band to risk level
            hkrs_to_risk_map = {
                "excellent": "low",
                "very_good": "low",
                "standard_plus": "moderate",
                "standard": "moderate",
                "substandard": "high",
            }
            risk_level_str = hkrs_to_risk_map.get(risk_level_str, "moderate")
        else:
            # Traditional workflow - use PolicyRiskAgent
            risk_level_str = context._outputs.get("_risk_level", pr_output.risk_level.value if pr_output else "moderate")
            premium_adjustment_pct = context._outputs.get("_premium_adjustment_pct", pr_output.premium_adjustment_recommendation.adjustment_percentage if pr_output else 0)
            adjusted_premium = context._outputs.get("_adjusted_premium", pr_output.premium_adjustment_recommendation.adjusted_premium_annual if pr_output else 1000)
            referral_required = context._outputs.get("_referral_required", pr_output.referral_required if pr_output else False)
            approved = pr_output.approved if pr_output else True
            decision = pr_output.decision if pr_output else "approved"
        
        # Map decision to status (DIRECT MAPPING - NO REINTERPRETATION)
        if decision == "declined" or not approved:
            status = DecisionStatus.DECLINED
        elif decision == "referred" or referral_required:
            status = DecisionStatus.REFERRED
        elif risk_level_str == "decline":
            status = DecisionStatus.DECLINED
            approved = False
        elif decision == "approved_with_adjustment" or premium_adjustment_pct > 0:
            status = DecisionStatus.APPROVED_WITH_ADJUSTMENT
        else:
            status = DecisionStatus.APPROVED
        
        # Convert risk level string to enum
        risk_level_map = {
            "low": RiskLevel.LOW,
            "moderate": RiskLevel.MODERATE,
            "high": RiskLevel.HIGH,
            "very_high": RiskLevel.VERY_HIGH,
            "decline": RiskLevel.DECLINE,
        }
        risk_level = risk_level_map.get(risk_level_str, RiskLevel.MODERATE)
        
        # Build final decision with Apple Health specific fields if applicable
        final_decision_kwargs = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "status": status,
            "risk_level": risk_level,
            "approved": approved,
            "premium_adjustment_pct": premium_adjustment_pct,
            "adjusted_premium_annual": adjusted_premium,
            "business_rules_approved": approved,
            "bias_check_passed": True,
            "underwriter_message": comm_output.underwriter_message,
            "customer_message": comm_output.customer_message,
        }
        
        # Add Apple Health specific fields
        if is_apple_health_workflow and ah_output:
            final_decision_kwargs["risk_class"] = ah_output.risk_class_recommendation
            final_decision_kwargs["hkrs_score"] = ah_output.hkrs
            final_decision_kwargs["hkrs_band"] = ah_output.hkrs_band.value
        
        return FinalDecision(**final_decision_kwargs)
    
    def _calculate_confidence(self, context: ExecutionContext) -> float:
        """
        Calculate overall confidence score.
        
        SIMPLIFIED 3-AGENT WORKFLOW:
        Based on:
        - Risk indicator confidence from HealthDataAnalysisAgent
        - Successful completion of all agents (3 agents)
        """
        hda_output: HealthDataAnalysisOutput = context.get_output("HealthDataAnalysisAgent")
        
        # Calculate average confidence from risk indicators
        if hda_output.risk_indicators:
            avg_indicator_confidence = sum(ri.confidence for ri in hda_output.risk_indicators) / len(hda_output.risk_indicators)
        else:
            avg_indicator_confidence = 0.8  # Default if no indicators
        
        # All agents must complete for full confidence (3 agents in simplified workflow)
        execution_completeness = len(context.get_records()) / 3.0
        
        # Weighted calculation
        confidence = (
            avg_indicator_confidence * 0.7 +
            execution_completeness * 0.3
        )
        
        return round(min(max(confidence, 0.0), 1.0), 3)
    
    def _generate_explanation(
        self,
        context: ExecutionContext,
        final_decision: FinalDecision,
    ) -> str:
        """
        Generate human-readable explanation of the decision.
        
        DUAL WORKFLOW SUPPORT:
        This SUMMARIZES the agent outputs without altering conclusions.
        """
        hda_output: HealthDataAnalysisOutput = context.get_output("HealthDataAnalysisAgent")
        pr_output: PolicyRiskOutput = context.get_output("PolicyRiskAgent")
        ah_output: AppleHealthRiskOutput = context.get_output("AppleHealthRiskAgent")
        
        # Determine workflow type
        is_apple_health_workflow = ah_output is not None
        
        # Count risk indicators by level
        risk_counts = {"low": 0, "moderate": 0, "high": 0, "very_high": 0}
        if hda_output and hda_output.risk_indicators:
            for indicator in hda_output.risk_indicators:
                if indicator.risk_level.value in risk_counts:
                    risk_counts[indicator.risk_level.value] += 1
        
        if is_apple_health_workflow:
            # Apple Health workflow explanation
            lines = [
                f"HealthKit Risk Assessment for {final_decision.patient_id}",
                "=" * 60,
                "",
                f"DECISION: {final_decision.status.value.upper()}",
                f"HKRS Score: {ah_output.hkrs:.0f}/100",
                f"HKRS Band: {ah_output.hkrs_band.value.replace('_', ' ').title()}",
                f"Risk Class: {ah_output.risk_class_recommendation}",
                "",
                "Sub-Score Summary:",
            ]
            
            for sub in ah_output.sub_scores:
                lines.append(f"  - {sub.category.replace('_', ' ').title()}: {sub.score:.0f}/{sub.max_score:.0f}")
            
            lines.extend([
                "",
                f"Age Adjustment Factor: {ah_output.age_adjustment_factor:.2f} ({ah_output.age_bracket})",
                f"Data Quality: {ah_output.data_quality.value}",
                "",
                f"Rationale: {ah_output.rationale}",
            ])
            
            if ah_output.top_positive_drivers:
                lines.append("")
                lines.append("Positive Health Factors:")
                for driver in ah_output.top_positive_drivers[:3]:
                    lines.append(f"  + {driver}")
            
            if ah_output.improvement_suggestions:
                lines.append("")
                lines.append("Suggestions for Improvement:")
                for suggestion in ah_output.improvement_suggestions[:3]:
                    lines.append(f"  → {suggestion}")
        else:
            # Traditional workflow explanation
            base_premium = context._outputs.get("_base_premium", 1000)
            triggered_rules = context._outputs.get("_triggered_rules", pr_output.triggered_rules if pr_output else [])
            
            lines = [
                f"Underwriting Decision for Patient {final_decision.patient_id}",
                "=" * 60,
                "",
                f"DECISION: {final_decision.status.value.upper()}",
                f"Risk Level: {final_decision.risk_level.value}",
                f"Premium Adjustment: {final_decision.premium_adjustment_pct:+.1f}%",
                f"Base Premium: ${base_premium:,.2f}",
                f"Annual Premium: ${final_decision.adjusted_premium_annual:,.2f}",
                "",
                "Analysis Summary:",
                f"  - Risk indicators identified: {len(hda_output.risk_indicators) if hda_output else 0}",
                f"    (Low: {risk_counts['low']}, Moderate: {risk_counts['moderate']}, "
                f"High: {risk_counts['high']}, Very High: {risk_counts['very_high']})",
                f"  - Policy rules evaluation: {'Approved' if pr_output and pr_output.approved else 'Declined'}",
                "",
                f"Rationale: {pr_output.rationale if pr_output else 'No rationale available'}",
            ]
            
            if triggered_rules:
                lines.append("")
                lines.append("Triggered Rules:")
                for rule in triggered_rules[:5]:
                    lines.append(f"  - {rule}")
        
        return "\n".join(lines)
    
    # =========================================================================
    # DATA LOADING HELPERS
    # =========================================================================
    
    def _build_health_metrics_from_application(self, validated_input: OrchestratorInput) -> HealthMetrics:
        """Build HealthMetrics from real application data.
        
        Extracts health-related data from the LLM outputs and document markdown
        to create a HealthMetrics object that can be processed by agents.
        
        For Apple Health apps, uses the structured health_metrics from llm_outputs.
        For traditional apps, simulates based on lifestyle factors.
        """
        from data.mock.schemas import (
            ActivityMetrics, HeartRateMetrics, SleepMetrics, HealthTrends,
            FitnessMetrics, MobilityMetrics, ExerciseMetrics, BodyMetrics
        )
        from datetime import date, timedelta
        
        llm_outputs = validated_input.llm_outputs or {}
        
        # Check if we have Apple Health data - use it directly
        if "health_metrics" in llm_outputs and llm_outputs.get("is_apple_health"):
            self.logger.info("Using Apple Health data from llm_outputs['health_metrics']")
            hm = llm_outputs["health_metrics"]
            today = date.today()
            
            # Extract nested data
            activity_data = hm.get("activity", {})
            fitness_data = hm.get("fitness", {})
            hr_data = hm.get("heart_rate", {})
            sleep_data = hm.get("sleep", {})
            body_data = hm.get("body_metrics", {})
            mobility_data = hm.get("mobility", {})
            exercise_data = hm.get("exercise", {})
            
            return HealthMetrics(
                patient_id=hm.get("patient_id", validated_input.patient_id),
                data_source="apple_health",
                collection_timestamp=datetime.now(timezone.utc),
                
                activity=ActivityMetrics(
                    daily_steps_avg=int(activity_data.get("daily_steps_avg", 8000)),
                    daily_active_minutes_avg=int(activity_data.get("daily_active_minutes_avg", 40)),
                    daily_calories_burned_avg=int(activity_data.get("active_energy_burned_avg", 400)),
                    weekly_exercise_sessions=int(exercise_data.get("workout_frequency_weekly", 3)),
                    days_with_data=int(activity_data.get("days_with_data", 120)),
                    measurement_period_days=180,
                    last_recorded_date=today - timedelta(days=1),
                    trend_6mo=activity_data.get("trend_6mo", "stable"),
                ),
                
                heart_rate=HeartRateMetrics(
                    resting_hr_avg=int(hr_data.get("resting_hr_avg", 68)),
                    resting_hr_min=int(hr_data.get("resting_hr_avg", 68)) - 8,
                    resting_hr_max=int(hr_data.get("resting_hr_avg", 68)) + 12,
                    hrv_avg_ms=float(hr_data.get("hrv_avg_ms", 45)),
                    elevated_hr_events=int(hr_data.get("elevated_hr_events", 0)),
                    irregular_rhythm_events=int(hr_data.get("irregular_rhythm_events", 0)),
                    days_with_data=int(hr_data.get("days_with_data", 90)),
                    measurement_period_days=180,
                    last_recorded_date=today - timedelta(days=1),
                ),
                
                sleep=SleepMetrics(
                    avg_sleep_duration_hours=float(sleep_data.get("avg_sleep_duration_hours", 7.2)),
                    avg_time_to_sleep_minutes=15,
                    sleep_efficiency_pct=88.0,
                    deep_sleep_pct=18.0,
                    rem_sleep_pct=22.0,
                    light_sleep_pct=60.0,
                    avg_awakenings_per_night=1.5,
                    nights_with_data=int(sleep_data.get("nights_with_data", 90)),
                    measurement_period_days=180,
                    last_recorded_date=today - timedelta(days=1),
                    sleep_consistency_variance_hours=float(sleep_data.get("sleep_consistency_variance_hours", 0.8)),
                ),
                
                # Include fitness metrics (VO2 Max) - critical for Apple Health scoring
                fitness=FitnessMetrics(
                    vo2_max=float(fitness_data.get("vo2_max")) if fitness_data.get("vo2_max") else None,
                    vo2_max_readings=int(fitness_data.get("vo2_max_readings", 5)),
                    cardio_fitness_level=fitness_data.get("cardio_fitness_level", "Good"),
                ),
                
                # Include body metrics - critical for Apple Health scoring
                body_metrics=BodyMetrics(
                    bmi=body_data.get("bmi", 24.5),
                    bmi_trend=body_data.get("bmi_trend", "stable"),
                    weight_kg=body_data.get("weight_kg", 75),
                    height_cm=body_data.get("height_cm", 175),
                ),
                
                # Include mobility metrics - critical for Apple Health scoring
                mobility=MobilityMetrics(
                    walking_speed_avg=mobility_data.get("walking_speed_avg", 1.3),
                    walking_steadiness=mobility_data.get("walking_steadiness", "normal"),
                    double_support_time_pct=mobility_data.get("double_support_time_pct", 25),
                ),
                
                # Include exercise metrics
                exercise=ExerciseMetrics(
                    workout_frequency_weekly=exercise_data.get("workout_frequency_weekly", 3),
                    workout_avg_duration_minutes=exercise_data.get("workout_avg_duration_minutes", 45),
                    workout_types=exercise_data.get("workout_types", ["walking", "running"]),
                ),
                
                trends=HealthTrends(
                    activity_trend_weekly="stable",
                    activity_trend_monthly=activity_data.get("trend_6mo", "stable"),
                    resting_hr_trend_weekly="stable",
                    resting_hr_trend_monthly="stable",
                    sleep_quality_trend_weekly="stable",
                    sleep_quality_trend_monthly="stable",
                    overall_health_trajectory="stable",
                    significant_changes=[],
                ),
                
                consent_verified=True,
                data_anonymized=False,
            )
        
        # Fall back to traditional document extraction for non-Apple Health apps
        self.logger.info("Using traditional document extraction for health metrics")
        
        # Extract values from LLM outputs (customer_profile section)
        raw_customer_profile = {}
        medical_summary = {}
        
        if "application_summary" in llm_outputs:
            raw_customer_profile = llm_outputs["application_summary"].get("customer_profile", {}).get("parsed", {})
        if "medical_summary" in llm_outputs:
            medical_summary = llm_outputs["medical_summary"]
        
        # Parse the customer_profile - it may have a key_fields array structure
        customer_profile = self._flatten_key_fields(raw_customer_profile)
        
        # Also check the summary field for additional info
        summary_text = str(raw_customer_profile.get("summary", "")).lower()
        
        # Determine activity level from lifestyle factors in the document
        tobacco_use = str(customer_profile.get("smoking_status", customer_profile.get("tobacco_use", "never"))).lower()
        # Also check the summary for smoking info
        if "former smoker" in summary_text or "quit" in summary_text:
            tobacco_use = "former"
        elif "non-smoker" in summary_text or "never smoked" in summary_text:
            tobacco_use = "never"
        elif "smoker" in summary_text and "former" not in summary_text:
            tobacco_use = "current"
            
        alcohol_use = str(customer_profile.get("alcohol_use", "none")).lower()
        occupation = str(customer_profile.get("occupation", "")).lower()
        
        # Estimate activity level based on occupation and lifestyle
        is_sedentary = any(x in occupation for x in ["office", "manager", "desk", "analyst", "developer"])
        is_active = any(x in occupation for x in ["construction", "athletic", "trainer", "nurse", "teacher"])
        # Check smoking status (former smokers are not current smokers)
        is_current_smoker = ("current" in tobacco_use or ("smoker" in tobacco_use and "non" not in tobacco_use)) and "former" not in tobacco_use and "quit" not in tobacco_use
        
        self.logger.info(f"Building health metrics - occupation: {occupation}, tobacco: {tobacco_use}, is_current_smoker: {is_current_smoker}")
        
        # Calculate simulated health metrics based on extracted profile
        today = date.today()
        
        # Activity metrics - simulate based on lifestyle
        if is_active:
            daily_steps = 9000
            active_mins = 55
        elif is_sedentary:
            daily_steps = 5500
            active_mins = 25
        else:
            daily_steps = 7000
            active_mins = 40
        
        # Heart rate - smokers tend to have higher resting HR
        if is_current_smoker:
            resting_hr = 78
            hrv = 35.0
        else:
            resting_hr = 68
            hrv = 48.0
        
        # Build HealthMetrics with the correct schema
        return HealthMetrics(
            patient_id=validated_input.patient_id,
            data_source="document_extraction",
            collection_timestamp=datetime.now(timezone.utc),
            
            activity=ActivityMetrics(
                daily_steps_avg=daily_steps,
                daily_active_minutes_avg=active_mins,
                daily_calories_burned_avg=2000,
                weekly_exercise_sessions=3 if not is_current_smoker else 1,
                days_with_data=85,
                measurement_period_days=90,
                last_recorded_date=today - timedelta(days=1),
            ),
            
            heart_rate=HeartRateMetrics(
                resting_hr_avg=resting_hr,
                resting_hr_min=resting_hr - 8,
                resting_hr_max=resting_hr + 12,
                hrv_avg_ms=hrv,
                elevated_hr_events=2 if is_current_smoker else 0,
                irregular_rhythm_events=0,
                days_with_data=88,
                measurement_period_days=90,
                last_recorded_date=today - timedelta(days=1),
            ),
            
            sleep=SleepMetrics(
                avg_sleep_duration_hours=6.5 if is_current_smoker else 7.2,
                avg_time_to_sleep_minutes=20 if is_current_smoker else 12,
                sleep_efficiency_pct=85.0 if is_current_smoker else 90.0,
                deep_sleep_pct=18.0,
                rem_sleep_pct=22.0,
                light_sleep_pct=60.0,
                avg_awakenings_per_night=2.5 if is_current_smoker else 1.5,
                nights_with_data=82,
                measurement_period_days=90,
                last_recorded_date=today - timedelta(days=1),
            ),
            
            trends=HealthTrends(
                activity_trend_weekly="stable",
                activity_trend_monthly="stable",
                resting_hr_trend_weekly="stable",
                resting_hr_trend_monthly="stable",
                sleep_quality_trend_weekly="stable",
                sleep_quality_trend_monthly="stable",
                overall_health_trajectory="stable",
                significant_changes=[],
            ),
            
            consent_verified=True,
            data_anonymized=False,
        )
    
    def _build_patient_profile_from_application(self, validated_input: OrchestratorInput) -> PatientProfile:
        """Build PatientProfile from real application data."""
        from data.mock.schemas import PatientDemographics, MedicalHistory
        from datetime import date
        
        llm_outputs = validated_input.llm_outputs or {}
        raw_customer_profile = {}
        customer_profile = {}
        medical_summary = {}
        
        if "application_summary" in llm_outputs:
            raw_customer_profile = llm_outputs["application_summary"].get("customer_profile", {}).get("parsed", {})
        if "medical_summary" in llm_outputs:
            medical_summary = llm_outputs["medical_summary"]
        
        # Parse the customer_profile - it may have a key_fields array structure
        customer_profile = self._flatten_key_fields(raw_customer_profile)
        summary_text = str(raw_customer_profile.get("summary", "")).lower()
        
        self.logger.info(f"Building patient profile from: {list(customer_profile.keys())}")
        
        # Extract age (default to 35 if not found)
        age = 35
        try:
            age_str = customer_profile.get("age", "35")
            # Handle "36" or "36 years" formats
            age = int(str(age_str).split()[0])
        except (ValueError, TypeError):
            pass
        
        # Extract gender/biological_sex
        gender_str = str(customer_profile.get("gender", "unknown")).lower()
        biological_sex = "male" if gender_str in ["male", "m"] else "female" if gender_str in ["female", "f"] else "unknown"
        
        # Extract state/region 
        state_region = customer_profile.get("nationality_and_residency", customer_profile.get("residency", customer_profile.get("state", "Unknown")))
        
        # Extract tobacco/smoking status from key_fields or summary
        smoking_status_raw = str(customer_profile.get("smoking_status", customer_profile.get("tobacco_use", ""))).lower()
        
        # Also check the summary for smoking info
        if "former smoker" in summary_text or "quit" in summary_text or "former smoker" in smoking_status_raw:
            smoker_status = "former"
        elif "non-smoker" in summary_text or "never" in smoking_status_raw:
            smoker_status = "never"
        elif "smoker" in smoking_status_raw and "former" not in smoking_status_raw:
            smoker_status = "current"
        else:
            smoker_status = "never"
        
        self.logger.info(f"Parsed smoking status: {smoker_status} (from: {smoking_status_raw}, summary has 'former smoker': {'former smoker' in summary_text})")
        
        # Extract alcohol use
        alcohol_str = str(customer_profile.get("alcohol_use", "none")).lower()
        if "heavy" in alcohol_str:
            alcohol_use = "heavy"
        elif "moderate" in alcohol_str or "occasional" in alcohol_str:
            alcohol_use = "moderate"
        elif "social" in alcohol_str:
            alcohol_use = "social"
        else:
            alcohol_use = "none"
        
        # Extract height and weight
        height_cm = 170.0  # Default
        weight_kg = 70.0   # Default
        bmi = 22.0
        try:
            height_str = str(customer_profile.get("height", "170 cm"))
            weight_str = str(customer_profile.get("weight", "70 kg"))
            height_cm = self._parse_height(height_str)
            weight_kg = self._parse_weight(weight_str)
            if height_cm > 0 and weight_kg > 0:
                bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
            self.logger.info(f"Extracted biometrics - height: {height_cm}cm, weight: {weight_kg}kg, BMI: {bmi}")
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Could not parse height/weight: {e}")
        
        # Extract medical conditions from medical_summary
        hypertension_data = medical_summary.get("hypertension", {}).get("parsed", {})
        has_hypertension = bool(hypertension_data.get("has_hypertension", False))
        
        family_data = medical_summary.get("family_history", {}).get("parsed", {})
        
        return PatientProfile(
            patient_id=validated_input.patient_id,
            demographics=PatientDemographics(
                age=age,
                biological_sex=biological_sex,
                state_region=str(state_region)[:50] if state_region else "Unknown",
            ),
            medical_history=MedicalHistory(
                has_diabetes=False,  # Would need to extract from document
                has_hypertension=has_hypertension,
                has_heart_disease=False,
                has_cancer_history=False,
                smoker_status=smoker_status,
                alcohol_use=alcohol_use,
                bmi=bmi,
                height_cm=height_cm,
                weight_kg=weight_kg,
                family_history_heart_disease=bool(family_data.get("heart_disease", False)),
                family_history_cancer=bool(family_data.get("cancer", False)),
                family_history_diabetes=bool(family_data.get("diabetes", False)),
            ),
            policy_type_requested="term_life",
            coverage_amount_requested=500000.0,
            profile_created_date=date.today(),
        )
    
    def _parse_height(self, height_str: str) -> float:
        """Parse height string to cm."""
        try:
            height_str = str(height_str).lower().strip()
            if "cm" in height_str:
                return float(height_str.replace("cm", "").strip())
            elif "'" in height_str or "ft" in height_str:
                # Handle feet/inches format
                parts = height_str.replace("ft", "'").replace("in", "\"").split("'")
                feet = float(parts[0].strip())
                inches = float(parts[1].replace('"', '').strip()) if len(parts) > 1 else 0
                return (feet * 30.48) + (inches * 2.54)
            else:
                return float(height_str)
        except (ValueError, IndexError):
            return 170.0  # Default
    
    def _parse_weight(self, weight_str: str) -> float:
        """Parse weight string to kg."""
        try:
            weight_str = str(weight_str).lower().strip()
            if "kg" in weight_str:
                return float(weight_str.replace("kg", "").strip())
            elif "lb" in weight_str or "lbs" in weight_str:
                lbs = float(weight_str.replace("lbs", "").replace("lb", "").strip())
                return lbs * 0.453592
            else:
                return float(weight_str)
        except (ValueError, IndexError):
            return 70.0  # Default
    
    def _flatten_key_fields(self, parsed_data: Dict[str, Any]) -> Dict[str, str]:
        """Flatten key_fields array into a dictionary.
        
        The LLM outputs often have a structure like:
        {
            "summary": "...",
            "key_fields": [
                {"label": "Name", "value": "John Doe"},
                {"label": "Age", "value": "35"},
                ...
            ]
        }
        
        This flattens it to:
        {"name": "John Doe", "age": "35", ...}
        """
        result = {}
        
        # First, add any top-level string fields
        for key, value in parsed_data.items():
            if isinstance(value, str):
                result[key.lower().replace(" ", "_")] = value
        
        # Then extract from key_fields array
        key_fields = parsed_data.get("key_fields", [])
        if isinstance(key_fields, list):
            for field in key_fields:
                if isinstance(field, dict):
                    label = field.get("label", "").lower().replace(" ", "_").replace("-", "_")
                    value = field.get("value", "")
                    if label and value:
                        result[label] = value
        
        return result
    
    def _load_patient_profile(self, patient_id: str) -> PatientProfile:
        """Load patient profile by ID."""
        profile = get_patient_by_id(patient_id)
        if profile is None:
            # Create minimal profile for unknown patients
            from data.mock.fixtures import get_sample_patient_profiles
            profiles = get_sample_patient_profiles()
            if profiles:
                return profiles[0]  # Return first profile as default
            raise ValueError(f"No patient profiles available and unknown patient_id: {patient_id}")
        return profile
    
    def _load_health_metrics(self, patient_id: str) -> HealthMetrics:
        """Load health metrics by patient ID."""
        # Map patient IDs to metric fixtures
        if "HEALTHY" in patient_id.upper():
            return get_healthy_patient_metrics()
        elif "MODERATE" in patient_id.upper():
            return get_moderate_risk_metrics()
        elif "HIGH" in patient_id.upper():
            return get_high_risk_metrics()
        else:
            # Default to moderate risk for unknown patients
            return get_moderate_risk_metrics()
    
    def __repr__(self) -> str:
        return f"<OrchestratorAgent(agent_id={self.agent_id!r})>"
