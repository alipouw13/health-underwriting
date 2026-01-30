"""
AuditAndTraceAgent - Produce a full decision audit trail

Agent Definition (from /.github/underwriting_agents.yaml):
---------------------------------------------------------
agent_id: AuditAndTraceAgent
purpose: Produce a full decision audit trail
inputs:
  agent_outputs: list
outputs:
  audit_log: object
tools_used:
  - trace-logger
evaluation_criteria:
  - completeness
failure_modes:
  - missing_steps
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4
from pydantic import Field

from app.agents.base import (
    BaseUnderwritingAgent,
    AgentInput,
    AgentOutput,
)


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class AgentOutputRecord(AgentInput):
    """Record of a single agent's output for audit purposes."""
    
    agent_id: str = Field(..., description="ID of the agent that produced the output")
    execution_id: str = Field(..., description="Unique execution ID")
    timestamp: datetime = Field(..., description="When the output was produced")
    execution_time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")
    success: bool = Field(..., description="Whether execution succeeded")
    input_summary: str = Field(..., description="Summary of input data")
    output_summary: str = Field(..., description="Summary of output data")
    key_decisions: List[str] = Field(default_factory=list, description="Key decisions made by this agent")
    errors: List[str] = Field(default_factory=list, description="Any errors or warnings")


class AuditAndTraceInput(AgentInput):
    """Input schema for AuditAndTraceAgent."""
    
    agent_outputs: List[AgentOutputRecord] = Field(..., description="List of agent outputs to audit")
    workflow_id: str = Field(default_factory=lambda: str(uuid4()), description="ID of the overall workflow")
    patient_id: str = Field(..., description="Patient ID for the underwriting decision")


class AuditLogEntry(AgentOutput):
    """A single entry in the audit log."""
    
    entry_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique entry ID")
    agent_id: str = Field(..., description="Agent that was audited")
    step_number: int = Field(..., description="Step number in workflow")
    input_hash: str = Field(..., description="Hash of input data for integrity")
    output_hash: str = Field(..., description="Hash of output data for integrity")
    execution_time_ms: float = Field(..., description="Execution time")
    status: str = Field(..., description="SUCCESS, FAILURE, or WARNING")
    notes: List[str] = Field(default_factory=list, description="Audit notes")


class AuditLog(AgentInput):
    """Complete audit log structure."""
    
    audit_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique audit ID")
    workflow_id: str = Field(..., description="Workflow being audited")
    patient_id: str = Field(..., description="Patient ID")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Audit creation time")
    total_agents_executed: int = Field(..., description="Number of agents executed")
    total_execution_time_ms: float = Field(..., description="Total execution time")
    workflow_status: str = Field(..., description="Overall workflow status")
    entries: List[Dict[str, Any]] = Field(default_factory=list, description="Individual audit entries")
    integrity_verified: bool = Field(..., description="Whether all steps completed with integrity")
    compliance_notes: List[str] = Field(default_factory=list, description="Compliance observations")
    missing_steps: List[str] = Field(default_factory=list, description="Any missing expected steps")
    summary: str = Field(..., description="Executive summary of audit")


class AuditAndTraceOutput(AgentOutput):
    """Output schema for AuditAndTraceAgent."""
    
    audit_log: AuditLog = Field(..., description="Complete audit trail")


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class AuditAndTraceAgent(BaseUnderwritingAgent[AuditAndTraceInput, AuditAndTraceOutput]):
    """
    Produce a full decision audit trail.
    
    This agent creates a comprehensive audit record of all agent executions
    in an underwriting workflow for compliance and traceability.
    
    Tools Used:
        - trace-logger: Provides audit logging capabilities
    
    Evaluation Criteria:
        - completeness: All workflow steps are captured
    
    Failure Modes:
        - missing_steps: Some agent executions are not recorded
    """
    
    agent_id = "AuditAndTraceAgent"
    purpose = "Produce a full decision audit trail"
    tools_used = ["trace-logger"]
    evaluation_criteria = ["completeness"]
    failure_modes = ["missing_steps"]
    
    # Expected agents in a complete workflow (in approximate order)
    EXPECTED_AGENTS = [
        "HealthDataAnalysisAgent",
        "DataQualityConfidenceAgent",
        "PolicyRiskAgent",
        "BusinessRulesValidationAgent",
        "BiasAndFairnessAgent",
        "CommunicationAgent",
    ]
    
    @property
    def input_type(self) -> type[AuditAndTraceInput]:
        return AuditAndTraceInput
    
    @property
    def output_type(self) -> type[AuditAndTraceOutput]:
        return AuditAndTraceOutput
    
    async def _execute(self, validated_input: AuditAndTraceInput) -> AuditAndTraceOutput:
        """
        Generate comprehensive audit trail.
        
        Process:
        1. Verify all expected agents executed
        2. Create audit entries for each agent output
        3. Calculate integrity hashes
        4. Generate compliance notes
        5. Produce executive summary
        """
        agent_outputs = validated_input.agent_outputs
        workflow_id = validated_input.workflow_id
        patient_id = validated_input.patient_id
        
        # Sort by timestamp
        sorted_outputs = sorted(agent_outputs, key=lambda x: x.timestamp)
        
        # Create audit entries
        entries: List[Dict[str, Any]] = []
        total_time = 0.0
        all_success = True
        
        for i, output in enumerate(sorted_outputs, 1):
            entry = {
                "entry_id": str(uuid4()),
                "agent_id": output.agent_id,
                "execution_id": output.execution_id,
                "step_number": i,
                "timestamp": output.timestamp.isoformat(),
                "input_hash": self._compute_hash(output.input_summary),
                "output_hash": self._compute_hash(output.output_summary),
                "execution_time_ms": output.execution_time_ms or 0,
                "status": "SUCCESS" if output.success else "FAILURE",
                "key_decisions": output.key_decisions,
                "errors": output.errors,
            }
            entries.append(entry)
            total_time += output.execution_time_ms or 0
            
            if not output.success:
                all_success = False
        
        # Check for missing steps
        executed_agents = {o.agent_id for o in agent_outputs}
        missing_steps = [a for a in self.EXPECTED_AGENTS if a not in executed_agents]
        
        # Generate compliance notes
        compliance_notes = self._generate_compliance_notes(
            sorted_outputs, 
            missing_steps,
            all_success
        )
        
        # Determine workflow status
        if not all_success:
            workflow_status = "FAILED"
        elif missing_steps:
            workflow_status = "INCOMPLETE"
        else:
            workflow_status = "COMPLETE"
        
        # Generate summary
        summary = self._generate_summary(
            len(agent_outputs),
            total_time,
            workflow_status,
            missing_steps,
            compliance_notes
        )
        
        # Create audit log
        audit_log = AuditLog(
            workflow_id=workflow_id,
            patient_id=patient_id,
            total_agents_executed=len(agent_outputs),
            total_execution_time_ms=round(total_time, 2),
            workflow_status=workflow_status,
            entries=entries,
            integrity_verified=all_success and not missing_steps,
            compliance_notes=compliance_notes,
            missing_steps=missing_steps,
            summary=summary,
        )
        
        return AuditAndTraceOutput(
            agent_id=self.agent_id,
            audit_log=audit_log,
        )
    
    def _compute_hash(self, data: str) -> str:
        """Compute a simple hash for integrity verification."""
        import hashlib
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _generate_compliance_notes(
        self,
        outputs: List[AgentOutputRecord],
        missing_steps: List[str],
        all_success: bool
    ) -> List[str]:
        """Generate compliance observations."""
        notes = []
        
        # Check for required agents
        if "BiasAndFairnessAgent" not in [o.agent_id for o in outputs]:
            notes.append("WARNING: Bias/fairness check was not performed")
        
        if "DataQualityConfidenceAgent" not in [o.agent_id for o in outputs]:
            notes.append("WARNING: Data quality assessment was not performed")
        
        if "BusinessRulesValidationAgent" not in [o.agent_id for o in outputs]:
            notes.append("WARNING: Business rules validation was not performed")
        
        # Check for failures
        failed_agents = [o.agent_id for o in outputs if not o.success]
        if failed_agents:
            notes.append(f"ERROR: The following agents failed: {', '.join(failed_agents)}")
        
        # Check for missing steps
        if missing_steps:
            notes.append(f"INCOMPLETE: Missing expected agents: {', '.join(missing_steps)}")
        
        # Check execution times
        slow_agents = [o.agent_id for o in outputs if (o.execution_time_ms or 0) > 5000]
        if slow_agents:
            notes.append(f"PERFORMANCE: Slow execution detected in: {', '.join(slow_agents)}")
        
        # Overall compliance assessment
        if all_success and not missing_steps:
            notes.append("COMPLIANT: All required workflow steps completed successfully")
        
        return notes
    
    def _generate_summary(
        self,
        agent_count: int,
        total_time: float,
        status: str,
        missing_steps: List[str],
        compliance_notes: List[str]
    ) -> str:
        """Generate executive summary of audit."""
        warning_count = sum(1 for n in compliance_notes if n.startswith("WARNING"))
        error_count = sum(1 for n in compliance_notes if n.startswith("ERROR"))
        
        summary_parts = [
            f"Underwriting Workflow Audit Summary",
            f"=" * 40,
            f"Status: {status}",
            f"Agents Executed: {agent_count}",
            f"Total Time: {total_time:.2f}ms",
        ]
        
        if missing_steps:
            summary_parts.append(f"Missing Steps: {len(missing_steps)}")
        
        if warning_count > 0 or error_count > 0:
            summary_parts.append(f"Issues: {error_count} errors, {warning_count} warnings")
        
        if status == "COMPLETE":
            summary_parts.append("\nAudit Result: PASSED - All compliance checks satisfied")
        elif status == "INCOMPLETE":
            summary_parts.append("\nAudit Result: REVIEW REQUIRED - Workflow incomplete")
        else:
            summary_parts.append("\nAudit Result: FAILED - Errors detected in workflow")
        
        return "\n".join(summary_parts)
