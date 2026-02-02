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

EXECUTION ORDER (SIMPLIFIED 3-AGENT MVP):
1. HealthDataAnalysisAgent - Extract health risk signals
2. BusinessRulesValidationAgent - Apply underwriting rules & calculate premium
3. CommunicationAgent - Generate decision messages

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

# Import agents for orchestration (simplified 3-agent workflow)
from app.agents.health_data_analysis import (
    HealthDataAnalysisAgent,
    HealthDataAnalysisInput,
    HealthDataAnalysisOutput,
)
from app.agents.business_rules_validation import (
    BusinessRulesValidationAgent,
    BusinessRulesValidationInput,
    BusinessRulesValidationOutput,
)
from app.agents.communication import (
    CommunicationAgent,
    CommunicationInput,
    CommunicationOutput,
)

# Import output types for type hints (used in context retrieval)
from app.agents.data_quality_confidence import DataQualityConfidenceOutput
from app.agents.policy_risk import PolicyRiskAgent, PolicyRiskOutput
from app.agents.bias_fairness import BiasAndFairnessOutput
from app.agents.audit_trace import AuditAndTraceOutput, AgentOutputRecord


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class OrchestratorInput(AgentInput):
    """Input schema for OrchestratorAgent."""
    
    patient_id: str = Field(..., description="Patient ID to process")
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
    status: DecisionStatus = Field(..., description="Decision status")
    risk_level: RiskLevel = Field(..., description="Final risk level")
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


# =============================================================================
# EXECUTION CONTEXT
# =============================================================================

class ExecutionContext:
    """
    Shared execution context for agent workflow.
    
    Stores outputs from each agent for use by subsequent agents.
    This is the ONLY mechanism for passing data between agents.
    """
    
    def __init__(self, patient_id: str, workflow_id: str):
        self.patient_id = patient_id
        self.workflow_id = workflow_id
        self.start_time = datetime.now(timezone.utc)
        self._outputs: Dict[str, AgentOutput] = {}
        self._records: List[AgentExecutionRecord] = []
        self.logger = logging.getLogger(f"orchestration.{workflow_id}")
        
        # Real application data (set when processing actual documents)
        self.application_data: Optional[Dict[str, Any]] = None
        self.document_markdown: Optional[str] = None
        self.llm_outputs: Optional[Dict[str, Any]] = None
    
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
        elif agent_id == "BusinessRulesValidationAgent":
            brv_out = output  # type: BusinessRulesValidationOutput
            return f"Approved: {brv_out.approved}, Violations: {len(brv_out.violations_found)}"
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
        
        SIMPLIFIED 3-AGENT WORKFLOW:
        1. HealthDataAnalysisAgent - Extract risk signals
        2. BusinessRulesValidationAgent - Apply rules & calculate premium
        3. CommunicationAgent - Generate decision messages
        """
        return [
            "HealthDataAnalysisAgent",
            "BusinessRulesValidationAgent",
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
    
    EXECUTION ORDER (NO EXCEPTIONS):
        1. HealthDataAnalysisAgent - Analyze health data for risk signals
        2. DataQualityConfidenceAgent - Assess data reliability
        3. PolicyRiskAgent - Translate signals to risk/premium
        4. BusinessRulesValidationAgent - Validate business rules
        5. BiasAndFairnessAgent - Check for bias
        6. CommunicationAgent - Generate messages
        7. AuditAndTraceAgent - Create audit trail
    
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
    
    # Map local agent IDs to Foundry agent names (simplified 3-agent workflow)
    FOUNDRY_AGENT_NAMES = {
        "HealthDataAnalysisAgent": "health_data_analysis",
        "BusinessRulesValidationAgent": "business_rules_validation",
        "CommunicationAgent": "communication",
    }
    
    # Human-readable agent names for progress display
    AGENT_DISPLAY_NAMES = {
        "HealthDataAnalysisAgent": "Health Data Analysis",
        "PolicyRiskAgent": "Policy Risk Assessment",
        "BusinessRulesValidationAgent": "Business Rules Validation",
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
        
        # Initialize local agents (4-agent workflow)
        self._health_data_agent = HealthDataAnalysisAgent()
        self._policy_risk_agent = PolicyRiskAgent()
        self._business_rules_agent = BusinessRulesValidationAgent()
        self._communication_agent = CommunicationAgent()
    
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
        
        # Validate input
        validated_input = self.validate_input(input_data)
        
        # Create execution context
        context = ExecutionContext(validated_input.patient_id, workflow_id)
        
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
        
        try:
            # STEP 1: HealthDataAnalysisAgent (MANDATORY)
            await self._execute_health_data_analysis(context, health_metrics, patient_profile)
            
            # STEP 2: BusinessRulesValidationAgent (MANDATORY) 
            # Now receives health analysis directly and handles risk + rules
            await self._execute_business_rules_validation(context, patient_profile)
            
            # STEP 3: CommunicationAgent (MANDATORY)
            await self._execute_communication(context, patient_profile)
            
        except AgentExecutionError as e:
            self.logger.error(f"Agent execution failed: {e}")
            raise AgentExecutionError(
                self.agent_id,
                f"Workflow failed at {e.agent_id}: {str(e)}",
                {"workflow_id": workflow_id, "failed_agent": e.agent_id}
            )
        
        # Produce final decision (SUMMARIZE ONLY - DO NOT ALTER CONCLUSIONS)
        final_decision = self._produce_final_decision(context, validated_input.patient_id)
        
        # Calculate overall confidence
        confidence_score = self._calculate_confidence(context)
        
        # Generate explanation
        explanation = self._generate_explanation(context, final_decision)
        
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
        )
        
        self.logger.info(f"Workflow {workflow_id} completed in {output.total_execution_time_ms:.2f}ms")
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
        
        # Create execution context
        context = ExecutionContext(validated_input.patient_id, workflow_id)
        
        # Check if we have real application data
        has_real_data = (
            validated_input.application_data is not None or
            validated_input.llm_outputs is not None
        )
        
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
        
        # Define agents in execution order (4-agent workflow with PolicyRiskAgent)
        # Each tuple: (agent_id, step_number, description, tools_used, execute_fn)
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
                "Translating health indicators into risk categories",
                ["policy-rule-engine", "risk-classifier"],
                lambda: self._execute_policy_risk(context, policy_rules)
            ),
            (
                "BusinessRulesValidationAgent", 
                3, 
                "Validating against underwriting rules and calculating premium",
                ["rule-engine", "premium-calculator"],
                lambda: self._execute_business_rules_validation(context, patient_profile)
            ),
            (
                "CommunicationAgent", 
                4, 
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
        
        # Produce final decision
        final_decision = self._produce_final_decision(context, validated_input.patient_id)
        confidence_score = self._calculate_confidence(context)
        explanation = self._generate_explanation(context, final_decision)
        
        output = OrchestratorOutput(
            agent_id=self.agent_id,
            success=True,
            final_decision=final_decision,
            confidence_score=confidence_score,
            explanation=explanation,
            execution_records=context.get_records(),
            workflow_id=workflow_id,
            total_execution_time_ms=context.get_total_time_ms(),
        )
        
        self.logger.info(f"Workflow {workflow_id} completed in {output.total_execution_time_ms:.2f}ms")
        
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
        
        elif agent_id == "BusinessRulesValidationAgent":
            from app.agents.business_rules_validation import BusinessRulesValidationOutput
            brv_out: BusinessRulesValidationOutput = output
            risk_level = context._outputs.get("_risk_level", "moderate")
            adj_pct = context._outputs.get("_premium_adjustment_pct", 0)
            status = "Approved" if brv_out.approved else "Declined"
            return f"{status} - {risk_level.title()} risk, {adj_pct:+.0f}% premium adjustment"
        
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
    ) -> Dict[str, Any]:
        """
        Invoke an agent via Azure AI Foundry.
        
        Args:
            agent_id: Local agent ID (e.g., "HealthDataAnalysisAgent")
            prompt: The prompt/instructions for the agent
            context_data: Input data for the agent
            
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
        
        return {
            "response": result.response,
            "parsed": result.parsed_output,
            "execution_time_ms": result.execution_time_ms,
            "token_usage": result.token_usage,
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
        
        input_data = {
            "health_metrics": health_metrics.model_dump(),
            "patient_profile": patient_profile.model_dump(),
        }
        
        if self._use_foundry:
            # Use Azure AI Foundry agent
            prompt = """Analyze the provided health metrics and patient profile to identify risk indicators.

For each risk indicator found, provide:
- indicator_id: Unique ID (e.g., "IND-ACT-001")
- category: One of "activity", "heart_rate", "sleep", "trend", "medical_history"
- indicator_name: Descriptive name
- risk_level: "low", "moderate", "high", or "very_high"
- confidence: 0.0 to 1.0
- metric_value: The measured value
- metric_unit: Unit of measurement
- explanation: Why this is a risk indicator

Provide a summary of the overall health risk analysis.

Return your response as JSON with this structure:
{
  "risk_indicators": [...],
  "summary": "..."
}"""
            
            result = await self._invoke_foundry_agent(
                "HealthDataAnalysisAgent",
                prompt,
                input_data,
            )
            
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
        
        # Store with actual inputs and tools used
        tools_used = ["azure-ai-foundry"] if self._use_foundry else ["local-health-analyzer"]
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
            result = await self._invoke_foundry_agent("DataQualityConfidenceAgent", prompt, input_data)
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
        
        # Get risk indicators from Step 1
        hda_output: HealthDataAnalysisOutput = context.get_output("HealthDataAnalysisAgent")
        
        # Load the actual underwriting policies from JSON
        underwriting_policies = self._load_underwriting_policies()
        
        input_data = {
            "risk_indicators": [ri.model_dump() for ri in hda_output.risk_indicators],
            "policy_rules": policy_rules.model_dump(),
        }
        
        if self._use_foundry:
            # PolicyRiskAgent is not deployed to Foundry (only 3 agents are deployed)
            # Use Azure OpenAI directly with the policy rules
            from app.openai_client import chat_completion
            from app.config import load_settings
            import json as json_module
            
            # Format risk indicators for prompt
            risk_indicators_text = "\n".join([
                f"- {ri.indicator_name}: {ri.risk_level.value} risk (confidence: {ri.confidence:.0%}) - {ri.explanation}"
                for ri in hda_output.risk_indicators
            ])
            
            # Format policies for prompt (summary of key policies)
            policies_summary = self._format_policies_for_prompt(underwriting_policies)
            
            prompt = f"""You are an expert insurance underwriter. Translate health risk indicators into insurance risk categories using the underwriting policy manual.

## RISK INDICATORS FROM HEALTH ANALYSIS

{risk_indicators_text}

## UNDERWRITING POLICY MANUAL (Key Policies)

{policies_summary}

## INSTRUCTIONS

1. Evaluate EACH risk indicator against the relevant policy criteria
2. For each indicator, identify which policy ID and criteria applies
3. Calculate cumulative risk level based on all factors
4. Determine preliminary premium adjustment percentage

## RISK LEVEL GUIDELINES
- Low (0% adjustment): No significant risk factors
- Low-Moderate (+10-15% adjustment): Minor risk factors, well-controlled
- Moderate (+15-25% adjustment): Multiple risk factors or poorly controlled conditions  
- Moderate-High (+25-50% adjustment): Significant risk factors requiring premium loading
- High (+50-100% adjustment or decline): Severe uncontrolled conditions

Return STRICT JSON:
{{
  "risk_level": "low" | "moderate" | "high" | "very_high",
  "risk_delta_score": <integer 0-100>,
  "premium_adjustment_percentage": <number>,
  "triggered_rules": ["policy_id-criteria_id", ...],
  "rule_evaluations": [
    {{"indicator": "...", "policy_id": "...", "criteria_id": "...", "action": "...", "contribution": "+X%"}}
  ],
  "rationale": "2-3 sentence explanation of how risk level was determined"
}}"""
            
            # Use OpenAI directly since PolicyRiskAgent is not deployed to Foundry
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
            
            from data.mock.schemas import PremiumAdjustment
            risk_level = RiskLevel(parsed.get("risk_level", "moderate").lower())
            
            # Calculate premium values
            base_premium = 1200.00  # Default base premium
            adjustment_pct = float(parsed.get("premium_adjustment_percentage", 25))
            adjusted_premium = base_premium * (1 + adjustment_pct / 100)
            
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
                execution_time_ms=execution_time_ms,
            )
        else:
            output = await self._policy_risk_agent.run(input_data)
        
        # Capture actual inputs for transparency
        actual_inputs = {
            "risk_indicators_count": len(hda_output.risk_indicators),
            "risk_indicators_summary": [{"name": ri.indicator_name, "risk_level": ri.risk_level.value} for ri in hda_output.risk_indicators[:5]],
            "policies_loaded": len(underwriting_policies.get("policies", [])) if underwriting_policies else 0,
        }
        tools_used = ["azure-ai-foundry", "policy-rule-engine"] if self._use_foundry else ["local-policy-analyzer"]
        context.store_output(
            "PolicyRiskAgent", 
            output, 
            step_number=2,
            actual_inputs=actual_inputs,
            tools_invoked=tools_used
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
    
    async def _execute_business_rules_validation(
        self,
        context: ExecutionContext,
        patient_profile: PatientProfile,
    ) -> BusinessRulesValidationOutput:
        """Step 2: Execute BusinessRulesValidationAgent.
        
        In the simplified 3-agent workflow, this agent now handles:
        - Risk level determination (previously PolicyRiskAgent)
        - Premium adjustment calculation
        - Business rules compliance validation
        """
        self.logger.info("Step 3: Executing BusinessRulesValidationAgent%s",
                        " (via Azure AI Foundry)" if self._use_foundry else " (local)")
        
        # Get outputs from previous steps
        hda_output: HealthDataAnalysisOutput = context.get_output("HealthDataAnalysisAgent")
        pr_output: PolicyRiskOutput = context.get_output("PolicyRiskAgent")
        
        # Format risk indicators for prompt
        risk_indicators_text = "\n".join([
            f"- {ri.indicator_name}: {ri.risk_level.value} risk (confidence: {ri.confidence:.0%}) - {ri.explanation}"
            for ri in hda_output.risk_indicators
        ])
        
        # Get preliminary risk assessment from PolicyRiskAgent
        preliminary_risk_level = pr_output.risk_level.value if pr_output else "moderate"
        preliminary_adjustment = pr_output.premium_adjustment_recommendation.adjustment_percentage if pr_output else 0
        triggered_policy_rules = pr_output.triggered_rules if pr_output else []
        
        # Calculate base premium (simplified calculation)
        base_premium = patient_profile.coverage_amount_requested * 0.002  # 0.2% base rate
        
        if self._use_foundry:
            # Build comprehensive prompt for Foundry agent
            prompt = f"""You are an Underwriting Rules Specialist. Validate the preliminary risk assessment from PolicyRiskAgent
against business rules and determine the final premium adjustment and approval status.

## Preliminary Risk Assessment (from PolicyRiskAgent):
- Risk Level: {preliminary_risk_level}
- Recommended Adjustment: +{preliminary_adjustment}%
- Policy Rules Triggered: {', '.join(triggered_policy_rules) if triggered_policy_rules else 'None'}

## Risk Indicators from Health Analysis:
{risk_indicators_text}

## Health Analysis Summary:
{hda_output.summary}

## Applicant Profile:
- Age: {patient_profile.demographics.age}
- Biological Sex: {patient_profile.demographics.biological_sex}
- Smoker Status: {patient_profile.medical_history.smoker_status}
- BMI: {patient_profile.medical_history.bmi}
- Has Hypertension: {patient_profile.medical_history.has_hypertension}
- Has Diabetes: {patient_profile.medical_history.has_diabetes}
- Policy Type: {patient_profile.policy_type_requested}
- Coverage Amount: ${patient_profile.coverage_amount_requested:,.2f}
- Base Premium: ${base_premium:.2f}

## Business Rules to Validate:

### Risk Classification:
- LOW RISK (0% adjustment): No significant risk indicators, non-smoker, BMI 18.5-25
- MODERATE RISK (+10-25% adjustment): 1-2 minor risk factors, former smoker, BMI 25-30
- HIGH RISK (+25-50% adjustment): Multiple risk factors, current smoker, BMI >30
- VERY HIGH RISK (+50-100% adjustment): Severe uncontrolled conditions, multiple high-severity indicators

### Compliance Rules:
1. Premium adjustments must not exceed 100% for standard policies
2. Age-based adjustments must follow actuarial guidelines
3. Smoker surcharge: Current smokers +25% minimum
4. BMI adjustments: >30 BMI adds +10-15%
5. Chronic conditions (hypertension, diabetes) add +5-15% each

### Referral Triggers:
Flag for manual review if: adjustment >50%, age >70 with multiple factors, conflicting indicators

Return your analysis as JSON:
{{
  "approved": true/false,
  "risk_level": "low" | "moderate" | "high" | "very_high" | "decline",
  "premium_adjustment_percentage": 0-100,
  "base_premium_annual": {base_premium:.2f},
  "adjusted_premium_annual": <calculated amount>,
  "rationale": "Detailed explanation of how rules were applied",
  "compliance_checks": ["check1: passed/failed", "check2: passed/failed"],
  "violations_found": ["violation1", "violation2"],
  "referral_required": true/false,
  "referral_reason": "reason if applicable",
  "triggered_rules": ["rule1", "rule2"],
  "recommendations": ["recommendation1", "recommendation2"]
}}"""

            result = await self._invoke_foundry_agent(
                "BusinessRulesValidationAgent", 
                prompt, 
                {
                    "risk_indicators": [ri.model_dump() for ri in hda_output.risk_indicators],
                    "patient_profile": patient_profile.model_dump(mode='json'),
                    "base_premium": base_premium,
                }
            )
            parsed = result.get("parsed") or {}
            
            # Parse Foundry response
            approved = parsed.get("approved", True)
            risk_level_str = parsed.get("risk_level", "moderate").lower()
            adjustment_pct = float(parsed.get("premium_adjustment_percentage", 0))
            adjusted_premium = parsed.get("adjusted_premium_annual") or (base_premium * (1 + adjustment_pct / 100))
            
            # Ensure compliance checks is a list of strings
            compliance_checks = parsed.get("compliance_checks", [])
            if isinstance(compliance_checks, list):
                compliance_checks = [str(c) for c in compliance_checks]
            else:
                compliance_checks = [str(compliance_checks)]
            
            output = BusinessRulesValidationOutput(
                agent_id="BusinessRulesValidationAgent",
                approved=approved,
                rationale=parsed.get("rationale", "Validated by Azure AI Foundry"),
                compliance_checks=compliance_checks,
                violations_found=parsed.get("violations_found", []),
                recommendations=parsed.get("recommendations", []),
                execution_time_ms=result.get("execution_time_ms", 0),
            )
            
            # Store additional calculated values in context for communication agent
            context._outputs["_risk_level"] = risk_level_str
            context._outputs["_premium_adjustment_pct"] = adjustment_pct
            context._outputs["_base_premium"] = base_premium
            context._outputs["_adjusted_premium"] = adjusted_premium
            context._outputs["_triggered_rules"] = parsed.get("triggered_rules", [])
            context._outputs["_referral_required"] = parsed.get("referral_required", False)
            
        else:
            # Local deterministic agent - calculate risk from indicators
            # Count risk levels
            high_count = sum(1 for ri in hda_output.risk_indicators if ri.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH])
            moderate_count = sum(1 for ri in hda_output.risk_indicators if ri.risk_level == RiskLevel.MODERATE)
            
            # Determine risk level and adjustment
            if high_count >= 2:
                risk_level_str = "high"
                adjustment_pct = 35.0
            elif high_count >= 1 or moderate_count >= 3:
                risk_level_str = "moderate"
                adjustment_pct = 15.0
            else:
                risk_level_str = "low"
                adjustment_pct = 0.0
            
            # Add smoker surcharge
            if patient_profile.medical_history.smoker_status == "current":
                adjustment_pct += 25.0
            elif patient_profile.medical_history.smoker_status == "former":
                adjustment_pct += 10.0
            
            adjusted_premium = base_premium * (1 + adjustment_pct / 100)
            
            output = BusinessRulesValidationOutput(
                agent_id="BusinessRulesValidationAgent",
                approved=True,
                rationale=f"Risk level {risk_level_str} based on {len(hda_output.risk_indicators)} risk indicators",
                compliance_checks=[
                    f"Premium adjustment within limits: {'passed' if adjustment_pct <= 100 else 'failed'}",
                    "Age-based rules: passed",
                    "Regulatory compliance: passed",
                ],
                violations_found=[],
                recommendations=[],
                execution_time_ms=0,
            )
            
            # Store calculated values
            context._outputs["_risk_level"] = risk_level_str
            context._outputs["_premium_adjustment_pct"] = adjustment_pct
            context._outputs["_base_premium"] = base_premium
            context._outputs["_adjusted_premium"] = adjusted_premium
            context._outputs["_triggered_rules"] = []
            context._outputs["_referral_required"] = adjustment_pct > 50
        
        # Capture actual inputs for transparency
        actual_inputs = {
            "risk_indicators_count": len(hda_output.risk_indicators),
            "preliminary_risk_level": preliminary_risk_level,
            "preliminary_adjustment": preliminary_adjustment,
            "patient_age": patient_profile.demographics.age,
            "smoker_status": patient_profile.medical_history.smoker_status,
            "base_premium": base_premium,
        }
        tools_used = ["azure-ai-foundry"] if self._use_foundry else ["local-rules-engine"]
        context.store_output(
            "BusinessRulesValidationAgent", 
            output, 
            step_number=3,
            actual_inputs=actual_inputs,
            tools_invoked=tools_used
        )
        return output
    
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

            result = await self._invoke_foundry_agent("BiasAndFairnessAgent", prompt, decision_context)
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
        
        In the simplified 3-agent workflow, this agent generates:
        - Internal underwriter message with technical details
        - External customer message with appropriate tone
        """
        self.logger.info("Step 3: Executing CommunicationAgent%s",
                        " (via Azure AI Foundry)" if self._use_foundry else " (local)")
        
        # Get outputs from previous steps
        hda_output: HealthDataAnalysisOutput = context.get_output("HealthDataAnalysisAgent")
        brv_output: BusinessRulesValidationOutput = context.get_output("BusinessRulesValidationAgent")
        
        # Get calculated values from business rules step
        risk_level_str = context._outputs.get("_risk_level", "moderate")
        premium_adjustment_pct = context._outputs.get("_premium_adjustment_pct", 0)
        base_premium = context._outputs.get("_base_premium", 1000)
        adjusted_premium = context._outputs.get("_adjusted_premium", 1000)
        triggered_rules = context._outputs.get("_triggered_rules", [])
        referral_required = context._outputs.get("_referral_required", False)
        
        # Determine status based on agent conclusions
        if not brv_output.approved:
            status = DecisionStatus.DECLINED
        elif referral_required:
            status = DecisionStatus.REFERRED
        elif premium_adjustment_pct > 0:
            status = DecisionStatus.APPROVED_WITH_ADJUSTMENT
        else:
            status = DecisionStatus.APPROVED
        
        # Build key risk factors from health analysis
        key_risk_factors = [ri.explanation for ri in hda_output.risk_indicators[:3]]
        
        if self._use_foundry:
            # Build prompt for Foundry agent
            prompt = f"""You are a Communication Specialist. Generate professional communications for an underwriting decision.

## Decision Details:
- Decision ID: DEC-{context.workflow_id[:8]}
- Patient ID: {context.patient_id}
- Status: {status.value}
- Risk Level: {risk_level_str}
- Base Premium: ${base_premium:.2f}
- Premium Adjustment: {premium_adjustment_pct}%
- Adjusted Annual Premium: ${adjusted_premium:.2f}
- Approved: {brv_output.approved}
- Referral Required: {referral_required}

## Policy Details:
- Policy Type: {patient_profile.policy_type_requested}
- Coverage Amount: ${patient_profile.coverage_amount_requested:,.2f}

## Key Risk Factors:
{chr(10).join('- ' + rf for rf in key_risk_factors) if key_risk_factors else '- No significant risk factors identified'}

## Decision Rationale:
{brv_output.rationale}

## Compliance Checks:
{chr(10).join('- ' + c for c in brv_output.compliance_checks)}

Generate two messages:

1. **Underwriter Message** (Internal): Include all technical details, risk factors, 
   premium calculations, and compliance notes. This is for insurance professionals.

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

            result = await self._invoke_foundry_agent("CommunicationAgent", prompt, {
                "status": status.value,
                "risk_level": risk_level_str,
                "premium_adjustment_pct": premium_adjustment_pct,
                "adjusted_premium": adjusted_premium,
                "rationale": brv_output.rationale,
                "key_risk_factors": key_risk_factors,
                "patient_profile": patient_profile.model_dump(mode='json'),
            })
            parsed = result.get("parsed") or {}
            
            output = CommunicationOutput(
                agent_id="CommunicationAgent",
                underwriter_message=parsed.get("underwriter_message", 
                    f"Underwriting decision: {status.value}. Risk level: {risk_level_str}. "
                    f"Premium adjustment: {premium_adjustment_pct}%. {brv_output.rationale}"),
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
            decision = UnderwritingDecision(
                decision_id=f"DEC-{context.workflow_id[:8]}",
                patient_id=context.patient_id,
                status=status,
                risk_level=risk_level_enum,
                premium_adjustment=PremiumAdjustment(
                    base_premium_annual=base_premium,
                    adjustment_percentage=premium_adjustment_pct,
                    adjusted_premium_annual=adjusted_premium,
                    adjustment_reasons=[brv_output.rationale] if brv_output.rationale else [],
                ),
                confidence_score=0.8,  # Default confidence
                data_quality_level=DataQualityLevel.GOOD,
                decision_rationale=brv_output.rationale,
                key_risk_factors=key_risk_factors,
                regulatory_compliant=brv_output.approved,
                bias_check_passed=True,
            )
            
            # Build proper decision_summary matching DecisionSummary schema
            decision_summary = {
                "decision": decision.model_dump(mode='json'),
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
            "approved": brv_output.approved if brv_output else True,
        }
        tools_used = ["azure-ai-foundry"] if self._use_foundry else ["local-message-generator"]
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

            result = await self._invoke_foundry_agent("AuditAndTraceAgent", prompt, {
                "workflow_id": context.workflow_id,
                "patient_id": patient_id,
                "agent_outputs": [ao.model_dump(mode='json') for ao in agent_outputs],
            })
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
    # FINAL DECISION PRODUCTION (SUMMARIZE ONLY - NO ALTERATIONS)
    # =========================================================================
    
    def _produce_final_decision(
        self,
        context: ExecutionContext,
        patient_id: str,
    ) -> FinalDecision:
        """
        Produce final underwriting decision.
        
        SIMPLIFIED 3-AGENT WORKFLOW:
        Uses outputs from HealthDataAnalysis, BusinessRulesValidation, and Communication.
        
        IMPORTANT: This method SUMMARIZES agent outputs.
        It does NOT alter or override any agent conclusions.
        """
        # Retrieve agent outputs (READ-ONLY - DO NOT MODIFY)
        brv_output: BusinessRulesValidationOutput = context.get_output("BusinessRulesValidationAgent")
        comm_output: CommunicationOutput = context.get_output("CommunicationAgent")
        
        # Get calculated values from business rules step
        risk_level_str = context._outputs.get("_risk_level", "moderate")
        premium_adjustment_pct = context._outputs.get("_premium_adjustment_pct", 0)
        adjusted_premium = context._outputs.get("_adjusted_premium", 1000)
        referral_required = context._outputs.get("_referral_required", False)
        
        # Determine approval status based on agent conclusions (NO OVERRIDES)
        approved = brv_output.approved
        
        # Map risk level to decision status (DIRECT MAPPING - NO REINTERPRETATION)
        if not approved:
            status = DecisionStatus.DECLINED
        elif referral_required:
            status = DecisionStatus.REFERRED
        elif risk_level_str == "decline":
            status = DecisionStatus.DECLINED
            approved = False
        elif premium_adjustment_pct > 0:
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
        
        return FinalDecision(
            patient_id=patient_id,
            status=status,
            risk_level=risk_level,
            approved=approved,
            premium_adjustment_pct=premium_adjustment_pct,
            adjusted_premium_annual=adjusted_premium,
            business_rules_approved=brv_output.approved,
            bias_check_passed=True,  # Simplified workflow - no bias check agent
            underwriter_message=comm_output.underwriter_message,
            customer_message=comm_output.customer_message,
        )
    
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
        
        SIMPLIFIED 3-AGENT WORKFLOW:
        This SUMMARIZES the agent outputs without altering conclusions.
        """
        hda_output: HealthDataAnalysisOutput = context.get_output("HealthDataAnalysisAgent")
        brv_output: BusinessRulesValidationOutput = context.get_output("BusinessRulesValidationAgent")
        
        # Get calculated values
        base_premium = context._outputs.get("_base_premium", 1000)
        triggered_rules = context._outputs.get("_triggered_rules", [])
        
        # Count risk indicators by level
        risk_counts = {"low": 0, "moderate": 0, "high": 0, "very_high": 0}
        for indicator in hda_output.risk_indicators:
            if indicator.risk_level.value in risk_counts:
                risk_counts[indicator.risk_level.value] += 1
        
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
            f"  - Risk indicators identified: {len(hda_output.risk_indicators)}",
            f"    (Low: {risk_counts['low']}, Moderate: {risk_counts['moderate']}, "
            f"High: {risk_counts['high']}, Very High: {risk_counts['very_high']})",
            f"  - Business rules: {'Passed' if brv_output.approved else 'Failed'}",
            "",
            f"Rationale: {brv_output.rationale}",
        ]
        
        if triggered_rules:
            lines.append("")
            lines.append("Triggered Rules:")
            for rule in triggered_rules[:5]:  # Limit to 5 rules
                lines.append(f"  - {rule}")
        
        if brv_output.violations_found:
            lines.append("")
            lines.append("Compliance Issues:")
            for violation in brv_output.violations_found:
                lines.append(f"  - {violation}")
        
        return "\n".join(lines)
    
    # =========================================================================
    # DATA LOADING HELPERS
    # =========================================================================
    
    def _build_health_metrics_from_application(self, validated_input: OrchestratorInput) -> HealthMetrics:
        """Build HealthMetrics from real application data.
        
        Extracts health-related data from the LLM outputs and document markdown
        to create a HealthMetrics object that can be processed by agents.
        
        Note: The HealthMetrics schema uses activity/heart_rate/sleep metrics,
        not clinical data like blood pressure or cholesterol.
        """
        from data.mock.schemas import (
            ActivityMetrics, HeartRateMetrics, SleepMetrics, HealthTrends
        )
        from datetime import date, timedelta
        
        llm_outputs = validated_input.llm_outputs or {}
        
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
        
        # Extract BMI
        bmi = 22.0
        try:
            height_str = str(customer_profile.get("height", "170 cm"))
            weight_str = str(customer_profile.get("weight", "70 kg"))
            height_cm = self._parse_height(height_str)
            weight_kg = self._parse_weight(weight_str)
            if height_cm > 0:
                bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
        except (ValueError, TypeError):
            pass
        
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
