"""
OrchestratorAgent - Coordinate agent execution and produce final decision

Agent Definition (from /.github/underwriting_agents.yaml):
---------------------------------------------------------
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

EXECUTION ORDER (STRICT - NO EXCEPTIONS):
1. HealthDataAnalysisAgent
2. DataQualityConfidenceAgent
3. PolicyRiskAgent
4. BusinessRulesValidationAgent
5. BiasAndFairnessAgent
6. CommunicationAgent
7. AuditAndTraceAgent

CONSTRAINTS:
- No conditional branching
- No skipping agents
- No reinterpretation of agent outputs
- Orchestrator may summarize outputs
- Orchestrator may NOT alter or override agent conclusions
"""

from __future__ import annotations

import logging
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from pydantic import Field

from data.mock.schemas import (
    DecisionStatus,
    HealthMetrics,
    PatientProfile,
    PolicyRuleSet,
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

# Import all agents for orchestration
from app.agents.health_data_analysis import (
    HealthDataAnalysisAgent,
    HealthDataAnalysisInput,
    HealthDataAnalysisOutput,
)
from app.agents.data_quality_confidence import (
    DataQualityConfidenceAgent,
    DataQualityConfidenceInput,
    DataQualityConfidenceOutput,
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
from app.agents.bias_fairness import (
    BiasAndFairnessAgent,
    BiasAndFairnessInput,
    BiasAndFairnessOutput,
)
from app.agents.communication import (
    CommunicationAgent,
    CommunicationInput,
    CommunicationOutput,
)
from app.agents.audit_trace import (
    AuditAndTraceAgent,
    AuditAndTraceInput,
    AuditAndTraceOutput,
    AgentOutputRecord,
)


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class OrchestratorInput(AgentInput):
    """Input schema for OrchestratorAgent."""
    
    patient_id: str = Field(..., description="Patient ID to process")
    health_metrics: Optional[HealthMetrics] = Field(None, description="Override health metrics (optional)")
    policy_rules: Optional[PolicyRuleSet] = Field(None, description="Override policy rules (optional)")


class AgentExecutionRecord(AgentInput):
    """Record of a single agent execution in the workflow."""
    
    agent_id: str = Field(..., description="ID of the agent")
    step_number: int = Field(..., description="Execution order (1-7)")
    execution_id: str = Field(..., description="Unique execution ID")
    timestamp: datetime = Field(..., description="When agent completed")
    execution_time_ms: float = Field(..., description="Execution time in ms")
    success: bool = Field(..., description="Whether execution succeeded")
    output_summary: str = Field(..., description="Brief summary of output")


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
    
    def store_output(self, agent_id: str, output: AgentOutput, step_number: int) -> None:
        """Store an agent's output in the context."""
        self._outputs[agent_id] = output
        
        # Create execution record
        record = AgentExecutionRecord(
            agent_id=agent_id,
            step_number=step_number,
            execution_id=output.execution_id,
            timestamp=output.timestamp,
            execution_time_ms=output.execution_time_ms or 0.0,
            success=output.success,
            output_summary=self._summarize_output(agent_id, output),
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
        elif agent_id == "DataQualityConfidenceAgent":
            dqc_out = output  # type: DataQualityConfidenceOutput
            return f"Confidence: {dqc_out.confidence_score:.2f}, Quality: {dqc_out.data_quality_level.value}"
        elif agent_id == "PolicyRiskAgent":
            pr_out = output  # type: PolicyRiskOutput
            return f"Risk: {pr_out.risk_level.value}, Adjustment: {pr_out.premium_adjustment_recommendation.adjustment_percentage:.1f}%"
        elif agent_id == "BusinessRulesValidationAgent":
            brv_out = output  # type: BusinessRulesValidationOutput
            return f"Approved: {brv_out.approved}, Violations: {len(brv_out.violations_found)}"
        elif agent_id == "BiasAndFairnessAgent":
            bf_out = output  # type: BiasAndFairnessOutput
            return f"Fairness: {bf_out.fairness_score:.2f}, Flags: {len(bf_out.bias_flags)}"
        elif agent_id == "CommunicationAgent":
            return "Messages generated"
        elif agent_id == "AuditAndTraceAgent":
            return "Audit log created"
        else:
            return "Output recorded"


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
        
        This is hardcoded per requirements - NO REORDERING ALLOWED.
        """
        return [
            "HealthDataAnalysisAgent",
            "DataQualityConfidenceAgent", 
            "PolicyRiskAgent",
            "BusinessRulesValidationAgent",
            "BiasAndFairnessAgent",
            "CommunicationAgent",
            "AuditAndTraceAgent",
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
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        self.definition_loader = AgentDefinitionLoader()
        
        # Initialize all agents (they are stateless, single instances are fine)
        self._health_data_agent = HealthDataAnalysisAgent()
        self._data_quality_agent = DataQualityConfidenceAgent()
        self._policy_risk_agent = PolicyRiskAgent()
        self._business_rules_agent = BusinessRulesValidationAgent()
        self._bias_fairness_agent = BiasAndFairnessAgent()
        self._communication_agent = CommunicationAgent()
        self._audit_trace_agent = AuditAndTraceAgent()
    
    def validate_input(self, input_data: Dict[str, Any]) -> OrchestratorInput:
        """Validate orchestrator input."""
        return OrchestratorInput.model_validate(input_data)
    
    async def run(self, input_data: Dict[str, Any]) -> OrchestratorOutput:
        """
        Execute the full underwriting workflow.
        
        Args:
            input_data: Must contain 'patient_id', optionally health_metrics and policy_rules
            
        Returns:
            OrchestratorOutput with final decision, confidence, and explanation
        """
        workflow_id = str(uuid4())
        self.logger.info(f"Starting workflow {workflow_id}")
        
        # Validate input
        validated_input = self.validate_input(input_data)
        
        # Create execution context
        context = ExecutionContext(validated_input.patient_id, workflow_id)
        
        # Load data
        patient_profile = self._load_patient_profile(validated_input.patient_id)
        health_metrics = validated_input.health_metrics or self._load_health_metrics(validated_input.patient_id)
        policy_rules = validated_input.policy_rules or get_standard_policy_rules()
        
        try:
            # STEP 1: HealthDataAnalysisAgent (MANDATORY)
            await self._execute_health_data_analysis(context, health_metrics, patient_profile)
            
            # STEP 2: DataQualityConfidenceAgent (MANDATORY)
            await self._execute_data_quality_confidence(context, health_metrics)
            
            # STEP 3: PolicyRiskAgent (MANDATORY)
            await self._execute_policy_risk(context, policy_rules)
            
            # STEP 4: BusinessRulesValidationAgent (MANDATORY)
            await self._execute_business_rules_validation(context)
            
            # STEP 5: BiasAndFairnessAgent (MANDATORY)
            await self._execute_bias_fairness(context, patient_profile)
            
            # STEP 6: CommunicationAgent (MANDATORY)
            await self._execute_communication(context, patient_profile)
            
            # STEP 7: AuditAndTraceAgent (MANDATORY)
            await self._execute_audit_trace(context, validated_input.patient_id)
            
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
    
    # =========================================================================
    # AGENT EXECUTION STEPS (STRICT ORDER - NO MODIFICATION)
    # =========================================================================
    
    async def _execute_health_data_analysis(
        self, 
        context: ExecutionContext, 
        health_metrics: HealthMetrics,
        patient_profile: PatientProfile,
    ) -> HealthDataAnalysisOutput:
        """Step 1: Execute HealthDataAnalysisAgent."""
        self.logger.info("Step 1: Executing HealthDataAnalysisAgent")
        
        output = await self._health_data_agent.run({
            "health_metrics": health_metrics.model_dump(),
            "patient_profile": patient_profile.model_dump(),
        })
        
        context.store_output("HealthDataAnalysisAgent", output, step_number=1)
        return output
    
    async def _execute_data_quality_confidence(
        self,
        context: ExecutionContext,
        health_metrics: HealthMetrics,
    ) -> DataQualityConfidenceOutput:
        """Step 2: Execute DataQualityConfidenceAgent."""
        self.logger.info("Step 2: Executing DataQualityConfidenceAgent")
        
        output = await self._data_quality_agent.run({
            "health_metrics": health_metrics.model_dump(),
        })
        
        context.store_output("DataQualityConfidenceAgent", output, step_number=2)
        return output
    
    async def _execute_policy_risk(
        self,
        context: ExecutionContext,
        policy_rules: PolicyRuleSet,
    ) -> PolicyRiskOutput:
        """Step 3: Execute PolicyRiskAgent."""
        self.logger.info("Step 3: Executing PolicyRiskAgent")
        
        # Get risk indicators from Step 1
        hda_output: HealthDataAnalysisOutput = context.get_output("HealthDataAnalysisAgent")
        
        output = await self._policy_risk_agent.run({
            "risk_indicators": [ri.model_dump() for ri in hda_output.risk_indicators],
            "policy_rules": policy_rules.model_dump(),
        })
        
        context.store_output("PolicyRiskAgent", output, step_number=3)
        return output
    
    async def _execute_business_rules_validation(
        self,
        context: ExecutionContext,
    ) -> BusinessRulesValidationOutput:
        """Step 4: Execute BusinessRulesValidationAgent."""
        self.logger.info("Step 4: Executing BusinessRulesValidationAgent")
        
        # Get premium recommendation from Step 3
        pr_output: PolicyRiskOutput = context.get_output("PolicyRiskAgent")
        
        output = await self._business_rules_agent.run({
            "premium_adjustment_recommendation": pr_output.premium_adjustment_recommendation.model_dump(),
        })
        
        context.store_output("BusinessRulesValidationAgent", output, step_number=4)
        return output
    
    async def _execute_bias_fairness(
        self,
        context: ExecutionContext,
        patient_profile: PatientProfile,
    ) -> BiasAndFairnessOutput:
        """Step 5: Execute BiasAndFairnessAgent."""
        self.logger.info("Step 5: Executing BiasAndFairnessAgent")
        
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
        """Step 6: Execute CommunicationAgent."""
        self.logger.info("Step 6: Executing CommunicationAgent")
        
        # Get outputs from previous steps
        hda_output: HealthDataAnalysisOutput = context.get_output("HealthDataAnalysisAgent")
        dqc_output: DataQualityConfidenceOutput = context.get_output("DataQualityConfidenceAgent")
        pr_output: PolicyRiskOutput = context.get_output("PolicyRiskAgent")
        brv_output: BusinessRulesValidationOutput = context.get_output("BusinessRulesValidationAgent")
        bf_output: BiasAndFairnessOutput = context.get_output("BiasAndFairnessAgent")
        
        # Determine status based on agent conclusions
        if not brv_output.approved:
            status = DecisionStatus.DECLINED
        elif bf_output.bias_flags:
            status = DecisionStatus.REFERRED
        elif pr_output.premium_adjustment_recommendation.adjustment_percentage > 0:
            status = DecisionStatus.APPROVED_WITH_ADJUSTMENT
        else:
            status = DecisionStatus.APPROVED
        
        # Build UnderwritingDecision for communication
        decision = UnderwritingDecision(
            decision_id=f"DEC-{context.workflow_id[:8]}",
            patient_id=context.patient_id,
            status=status,
            risk_level=pr_output.risk_level,
            premium_adjustment=pr_output.premium_adjustment_recommendation,
            confidence_score=dqc_output.confidence_score,
            data_quality_level=dqc_output.data_quality_level,
            decision_rationale=brv_output.rationale,
            key_risk_factors=[ri.explanation for ri in hda_output.risk_indicators[:3]],
            regulatory_compliant=brv_output.approved,
            bias_check_passed=len(bf_output.bias_flags) == 0,
        )
        
        # Build decision summary for communication (matches DecisionSummary schema)
        decision_summary = {
            "decision": decision.model_dump(),
            "patient_name": None,  # No name in demographics
            "policy_type": patient_profile.policy_type_requested,
            "coverage_amount": patient_profile.coverage_amount_requested,
        }
        
        output = await self._communication_agent.run({
            "decision_summary": decision_summary,
        })
        
        context.store_output("CommunicationAgent", output, step_number=6)
        return output
    
    async def _execute_audit_trace(
        self,
        context: ExecutionContext,
        patient_id: str,
    ) -> AuditAndTraceOutput:
        """Step 7: Execute AuditAndTraceAgent."""
        self.logger.info("Step 7: Executing AuditAndTraceAgent")
        
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
        
        IMPORTANT: This method SUMMARIZES agent outputs.
        It does NOT alter or override any agent conclusions.
        """
        # Retrieve agent outputs (READ-ONLY - DO NOT MODIFY)
        pr_output: PolicyRiskOutput = context.get_output("PolicyRiskAgent")
        brv_output: BusinessRulesValidationOutput = context.get_output("BusinessRulesValidationAgent")
        bf_output: BiasAndFairnessOutput = context.get_output("BiasAndFairnessAgent")
        comm_output: CommunicationOutput = context.get_output("CommunicationAgent")
        
        # Determine approval status based on agent conclusions (NO OVERRIDES)
        approved = brv_output.approved and len(bf_output.bias_flags) == 0
        
        # Map risk level to decision status (DIRECT MAPPING - NO REINTERPRETATION)
        if not approved:
            if not brv_output.approved:
                status = DecisionStatus.DECLINED
            else:
                status = DecisionStatus.REFERRED  # Bias flags require review
        elif pr_output.risk_level == RiskLevel.DECLINE:
            status = DecisionStatus.DECLINED
            approved = False
        elif pr_output.premium_adjustment_recommendation.adjustment_percentage > 0:
            status = DecisionStatus.APPROVED_WITH_ADJUSTMENT
        else:
            status = DecisionStatus.APPROVED
        
        return FinalDecision(
            patient_id=patient_id,
            status=status,
            risk_level=pr_output.risk_level,
            approved=approved,
            premium_adjustment_pct=pr_output.premium_adjustment_recommendation.adjustment_percentage,
            adjusted_premium_annual=pr_output.premium_adjustment_recommendation.adjusted_premium_annual,
            business_rules_approved=brv_output.approved,
            bias_check_passed=len(bf_output.bias_flags) == 0,
            underwriter_message=comm_output.underwriter_message,
            customer_message=comm_output.customer_message,
        )
    
    def _calculate_confidence(self, context: ExecutionContext) -> float:
        """
        Calculate overall confidence score.
        
        Based on:
        - Data quality confidence from DataQualityConfidenceAgent
        - Fairness score from BiasAndFairnessAgent
        - Successful completion of all agents
        """
        dqc_output: DataQualityConfidenceOutput = context.get_output("DataQualityConfidenceAgent")
        bf_output: BiasAndFairnessOutput = context.get_output("BiasAndFairnessAgent")
        
        # Weighted average (data quality is primary driver)
        data_confidence = dqc_output.confidence_score
        fairness_score = bf_output.fairness_score
        
        # All agents must complete for full confidence
        execution_completeness = len(context.get_records()) / 7.0
        
        # Weighted calculation
        confidence = (
            data_confidence * 0.5 +
            fairness_score * 0.3 +
            execution_completeness * 0.2
        )
        
        return round(min(max(confidence, 0.0), 1.0), 3)
    
    def _generate_explanation(
        self,
        context: ExecutionContext,
        final_decision: FinalDecision,
    ) -> str:
        """
        Generate human-readable explanation of the decision.
        
        This SUMMARIZES the agent outputs without altering conclusions.
        """
        hda_output: HealthDataAnalysisOutput = context.get_output("HealthDataAnalysisAgent")
        dqc_output: DataQualityConfidenceOutput = context.get_output("DataQualityConfidenceAgent")
        pr_output: PolicyRiskOutput = context.get_output("PolicyRiskAgent")
        brv_output: BusinessRulesValidationOutput = context.get_output("BusinessRulesValidationAgent")
        bf_output: BiasAndFairnessOutput = context.get_output("BiasAndFairnessAgent")
        
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
            f"Annual Premium: ${final_decision.adjusted_premium_annual:,.2f}",
            "",
            "Analysis Summary:",
            f"  - Risk indicators identified: {len(hda_output.risk_indicators)}",
            f"    (Low: {risk_counts['low']}, Moderate: {risk_counts['moderate']}, "
            f"High: {risk_counts['high']}, Very High: {risk_counts['very_high']})",
            f"  - Data quality: {dqc_output.data_quality_level.value}",
            f"  - Data confidence: {dqc_output.confidence_score:.0%}",
            f"  - Business rules: {'Passed' if brv_output.approved else 'Failed'}",
            f"  - Bias check: {'Passed' if final_decision.bias_check_passed else 'Flagged'}",
            "",
            f"Rationale: {brv_output.rationale}",
        ]
        
        if bf_output.bias_flags:
            lines.append("")
            lines.append("Bias/Fairness Notes:")
            for flag in bf_output.bias_flags:
                lines.append(f"  - {flag.bias_type}: {flag.description}")
        
        return "\n".join(lines)
    
    # =========================================================================
    # DATA LOADING HELPERS
    # =========================================================================
    
    def _load_patient_profile(self, patient_id: str) -> PatientProfile:
        """Load patient profile by ID."""
        profile = get_patient_by_id(patient_id)
        if profile is None:
            # Create minimal profile for unknown patients
            from data.mock.fixtures import get_patient_profiles
            profiles = get_patient_profiles()
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
