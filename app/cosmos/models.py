"""Pydantic models for agent run documents stored in Cosmos DB.

Document Schema Requirements (from specification):
- run_id (uuid)
- application_id
- execution_timestamp
- execution_mode ("legacy" | "multi_agent")
- agent_definitions: Snapshot of /.github/underwriting_agents.yaml
- agents: Array of per-agent execution records
- orchestration_summary: execution_order, success/failure, errors
- final_decision: underwriting_decision, premium_adjustment, confidence_score, explanation
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ExecutionMode(str, Enum):
    """Mode of execution for the underwriting run."""
    LEGACY = "legacy"
    MULTI_AGENT = "multi_agent"


class OrchestrationStatus(str, Enum):
    """Status of the orchestration run."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"


class TokenUsage(BaseModel):
    """Token usage metadata for an agent step."""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    unavailable_reason: Optional[str] = None  # Why tokens not available


class EvaluationResult(BaseModel):
    """Per-agent evaluation result (if available from Foundry SDK)."""
    groundedness: Optional[float] = None
    relevance: Optional[float] = None
    coherence: Optional[float] = None
    fluency: Optional[float] = None
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)
    unavailable_reason: Optional[str] = None


class AgentStepRecord(BaseModel):
    """Record of a single agent execution within the workflow.
    
    Captures inputs, outputs, evaluation results, and token usage
    for complete traceability.
    """
    agent_id: str = Field(..., description="Agent identifier (e.g., HealthDataAnalysisAgent)")
    step_number: int = Field(..., ge=1, description="Execution order (1-indexed)")
    execution_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique step execution ID")
    
    # Timing
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    execution_duration_ms: Optional[float] = None
    
    # Success/Failure
    success: bool = Field(default=True)
    error_message: Optional[str] = None
    
    # Inputs (sanitized - no PII)
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Agent input payload")
    
    # Outputs (full structured output)
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Agent output payload")
    output_summary: Optional[str] = None  # Brief human-readable summary
    
    # Evaluation results (from Foundry SDK if available)
    evaluation_results: Optional[EvaluationResult] = None
    
    # Token usage (if available)
    token_usage: Optional[TokenUsage] = None


class AgentDefinitionSnapshot(BaseModel):
    """Snapshot of an agent definition from YAML at execution time."""
    agent_id: str
    role: Optional[str] = None
    purpose: Optional[str] = None
    instructions: List[str] = Field(default_factory=list)
    inputs_required: List[str] = Field(default_factory=list)
    inputs_optional: List[str] = Field(default_factory=list)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    failure_modes: List[str] = Field(default_factory=list)


class OrchestrationSummary(BaseModel):
    """Summary of the complete orchestration run."""
    status: OrchestrationStatus = Field(..., description="Overall orchestration status")
    execution_order: List[str] = Field(..., description="Actual order agents were executed")
    agents_executed: int = Field(..., ge=0, description="Number of agents that ran")
    agents_succeeded: int = Field(..., ge=0, description="Number of agents that succeeded")
    agents_failed: int = Field(..., ge=0, description="Number of agents that failed")
    total_execution_time_ms: float = Field(..., ge=0, description="Total orchestration time")
    errors: List[str] = Field(default_factory=list, description="Error messages if any failures")


class FinalDecisionRecord(BaseModel):
    """Final underwriting decision output."""
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    underwriting_decision: str = Field(..., description="APPROVED, DENIED, APPROVED_WITH_ADJUSTMENT, MANUAL_REVIEW")
    risk_level: str = Field(..., description="low, moderate, high")
    premium_adjustment_pct: float = Field(..., description="Premium adjustment percentage")
    adjusted_premium_annual: float = Field(..., description="Final annual premium")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    explanation: str = Field(..., description="Decision explanation")
    business_rules_compliant: bool = Field(default=True)
    bias_check_passed: bool = Field(default=True)
    underwriter_message: Optional[str] = None
    customer_message: Optional[str] = None


class AgentRunDocument(BaseModel):
    """Complete document stored in Cosmos DB for each underwriting run.
    
    This is the primary document schema for the underwriting_agent_runs container.
    Partition key: application_id
    
    Each document represents ONE complete underwriting execution and is:
    - Append-only (never modified after creation)
    - Immutable (represents point-in-time execution)
    - Self-contained (can fully reconstruct what ran, why, and what decision was made)
    """
    
    # Primary identifiers
    id: str = Field(default_factory=lambda: str(uuid4()), description="Cosmos document ID")
    run_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique run identifier")
    application_id: str = Field(..., description="Underwriting application ID (partition key)")
    
    # Execution metadata
    execution_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the execution started"
    )
    execution_mode: ExecutionMode = Field(..., description="legacy or multi_agent")
    workflow_id: Optional[str] = None  # From orchestrator
    
    # Agent definitions snapshot (from YAML)
    agent_definitions_version: str = Field(default="1.1", description="Version from YAML")
    agent_definitions: List[AgentDefinitionSnapshot] = Field(
        default_factory=list,
        description="Snapshot of agent definitions used"
    )
    global_constraints: List[str] = Field(
        default_factory=list,
        description="Global constraints from YAML"
    )
    
    # Per-agent execution records
    agents: List[AgentStepRecord] = Field(
        default_factory=list,
        description="Ordered list of agent execution records"
    )
    
    # Orchestration summary
    orchestration_summary: Optional[OrchestrationSummary] = None
    
    # Final decision
    final_decision: Optional[FinalDecisionRecord] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "run_id": "run_123456",
                "application_id": "app_d69a823f",
                "execution_timestamp": "2026-02-01T15:30:00Z",
                "execution_mode": "multi_agent",
                "agent_definitions_version": "1.1",
                "orchestration_summary": {
                    "status": "success",
                    "execution_order": ["HealthDataAnalysisAgent", "BusinessRulesValidationAgent", "CommunicationAgent"],
                    "agents_executed": 3,
                    "agents_succeeded": 3,
                    "agents_failed": 0,
                    "total_execution_time_ms": 36000.0
                },
                "final_decision": {
                    "underwriting_decision": "APPROVED_WITH_ADJUSTMENT",
                    "risk_level": "moderate",
                    "premium_adjustment_pct": 15.0,
                    "confidence_score": 0.95
                }
            }
        }
    )


# =============================================================================
# TOKEN TRACKING DOCUMENT
# =============================================================================

class TokenTrackingDocument(BaseModel):
    """Document for token_tracking container.
    
    Tracks token usage per agent execution for cost analysis and optimization.
    Partition key: /execution_id
    """
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Cosmos document ID")
    record_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique record ID")
    execution_id: str = Field(..., description="Workflow execution ID (partition key)")
    application_id: str = Field(..., description="Application being processed")
    
    # Agent info
    agent_id: str = Field(..., description="Agent that consumed tokens")
    agent_type: str = Field(..., description="Type/role of agent")
    step_number: int = Field(..., ge=1, description="Step in workflow")
    
    # Token counts
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    
    # Cost estimation
    prompt_cost_usd: float = Field(default=0.0, ge=0)
    completion_cost_usd: float = Field(default=0.0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0)
    
    # Model info
    model_name: Optional[str] = None
    deployment_name: Optional[str] = None
    
    # Metadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    operation_type: str = Field(default="chat_completion", description="Type of operation")
    success: bool = Field(default=True)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# EVALUATION DOCUMENT
# =============================================================================

class EvaluationMetrics(BaseModel):
    """Standard evaluation metrics from Azure AI Foundry."""
    groundedness: Optional[float] = Field(None, ge=0, le=1)
    relevance: Optional[float] = Field(None, ge=0, le=1)
    coherence: Optional[float] = Field(None, ge=0, le=1)
    fluency: Optional[float] = Field(None, ge=0, le=1)
    similarity: Optional[float] = Field(None, ge=0, le=1)
    f1_score: Optional[float] = Field(None, ge=0, le=1)


class EvaluationDocument(BaseModel):
    """Document for evaluations container.
    
    Stores evaluation results for agent outputs.
    Partition key: /evaluation_id
    """
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Cosmos document ID")
    evaluation_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique evaluation ID (partition key)")
    
    # Execution context
    execution_id: str = Field(..., description="Workflow execution ID")
    application_id: str = Field(..., description="Application being evaluated")
    agent_id: str = Field(..., description="Agent being evaluated")
    step_number: int = Field(..., ge=1)
    
    # Evaluation details
    evaluation_type: str = Field(default="quality", description="Type of evaluation")
    evaluator_model: Optional[str] = Field(None, description="Model used for evaluation")
    
    # Metrics
    metrics: EvaluationMetrics = Field(default_factory=EvaluationMetrics)
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Overall score
    overall_score: Optional[float] = Field(None, ge=0, le=1)
    passed: bool = Field(default=True)
    
    # Input/Output being evaluated (for traceability)
    evaluated_input: Optional[str] = Field(None, description="Input that was evaluated")
    evaluated_output: Optional[str] = Field(None, description="Output that was evaluated")
    ground_truth: Optional[str] = Field(None, description="Expected output if available")
    
    # Timing
    evaluation_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evaluation_duration_ms: Optional[float] = None
    
    # Notes
    notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
