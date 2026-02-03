"""
Azure AI Foundry Evaluations Integration

This module integrates the Azure AI Evaluation SDK with the multi-agent
underwriting workflow, providing:

1. Per-agent evaluation after each agent execution
2. Workflow-level aggregate evaluation
3. Persistence of evaluation results to Cosmos DB
4. Non-blocking evaluation (does not affect agent execution)

EVALUATION POINTS (STEP 1 - Documented):
=========================================

Agent-Level Evaluations:
- HealthDataAnalysisAgent:
  - Input: patient health metrics, document context
  - Output: risk indicators, health summary
  - Evaluate: groundedness to input data, completeness of extraction

- BusinessRulesValidationAgent:
  - Input: health analysis, policy rules
  - Output: rule compliance, premium adjustment
  - Evaluate: coherence of rationale, relevance to rules

- CommunicationAgent:
  - Input: decision summary
  - Output: underwriter/customer messages
  - Evaluate: fluency, coherence of messages

Workflow-Level Evaluation (OrchestratorAgent):
- Input: all agent outputs, final decision
- Evaluate: end-to-end coherence, decision consistency

EVALUATION CRITERIA (STEP 2 - Foundry SDK Metrics Only):
=========================================================

Using ONLY official azure-ai-evaluation SDK metrics:
- GroundednessEvaluator: Response grounded in context
- CoherenceEvaluator: Logical flow and consistency
- RelevanceEvaluator: Response addresses the query
- FluencyEvaluator: Language quality
- QAEvaluator: Combined quality metrics

NO custom scoring logic. NO mock evaluations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal

from pydantic import BaseModel, Field


# =============================================================================
# JSON SERIALIZATION HELPERS
# =============================================================================

class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime, date, Decimal, and other common types."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return str(obj)


def safe_json_dumps(obj: Any, max_length: int = 2000) -> str:
    """Safely serialize an object to JSON string with truncation."""
    try:
        result = json.dumps(obj, cls=SafeJSONEncoder, default=str)
        return result[:max_length]
    except Exception:
        return str(obj)[:max_length]

# Import existing Cosmos schema for alignment
from app.cosmos.models import EvaluationResult as CosmosEvaluationResult

logger = logging.getLogger("underwriting_assistant.evaluations")


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

class EvaluatorType(str, Enum):
    """Types of evaluators available from Foundry SDK."""
    GROUNDEDNESS = "groundedness"
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    FLUENCY = "fluency"
    QA = "qa"  # Composite evaluator


class EvaluationStatus(str, Enum):
    """Status of an evaluation run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# EVALUATION RESULT SCHEMAS
# =============================================================================

class MetricScore(BaseModel):
    """Individual metric score from an evaluator."""
    metric_name: str = Field(..., description="Name of the metric")
    score: float = Field(..., ge=0.0, le=5.0, description="Score (typically 1-5)")
    threshold: float = Field(default=3.0, description="Pass/fail threshold")
    passed: bool = Field(..., description="Whether score meets threshold")
    reason: Optional[str] = Field(None, description="Explanation for the score")


class AgentEvaluationResult(BaseModel):
    """
    Evaluation result for a single agent execution.
    
    NOTE: This class is used internally for detailed evaluation tracking.
    For Cosmos DB persistence, use to_cosmos_format() to convert to the
    standard CosmosEvaluationResult schema.
    """
    agent_id: str = Field(..., description="ID of the evaluated agent")
    evaluation_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    status: EvaluationStatus = Field(default=EvaluationStatus.PENDING)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Scores
    metrics: List[MetricScore] = Field(default_factory=list)
    aggregate_score: Optional[float] = None
    passed: Optional[bool] = None
    
    # Error handling
    error_message: Optional[str] = None
    
    # Raw output from SDK
    raw_sdk_output: Optional[Dict[str, Any]] = None
    
    @property
    def overall_score(self) -> float:
        """Alias for aggregate_score for backwards compatibility."""
        return self.aggregate_score if self.aggregate_score is not None else 0.0
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert to dict with datetime handling."""
        data = super().model_dump(**kwargs)
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        # Add overall_score for frontend compatibility
        data['overall_score'] = self.overall_score
        return data
    
    def to_cosmos_format(self) -> CosmosEvaluationResult:
        """
        Convert to Cosmos DB EvaluationResult format.
        
        This aligns with the existing schema in app/cosmos/models.py
        so results can be persisted in AgentStepRecord.evaluation_results
        """
        # Extract individual scores from metrics
        scores = {m.metric_name: m.score for m in self.metrics}
        reasons = {m.metric_name: m.reason for m in self.metrics if m.reason}
        
        return CosmosEvaluationResult(
            groundedness=scores.get("groundedness"),
            relevance=scores.get("relevance"),
            coherence=scores.get("coherence"),
            fluency=scores.get("fluency"),
            custom_metrics={
                "aggregate_score": self.aggregate_score,
                "passed": self.passed,
                "evaluation_id": self.evaluation_id,
                "duration_ms": self.duration_ms,
                "reasons": reasons,
                "status": self.status.value,
            },
            unavailable_reason=self.error_message if self.status == EvaluationStatus.FAILED else None,
        )


class WorkflowEvaluationResult(BaseModel):
    """Aggregate evaluation result for the entire workflow."""
    workflow_id: str = Field(..., description="Workflow ID being evaluated")
    evaluation_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    status: EvaluationStatus = Field(default=EvaluationStatus.PENDING)
    
    # Per-agent results
    agent_evaluations: Dict[str, AgentEvaluationResult] = Field(default_factory=dict)
    
    # Aggregate metrics
    aggregate_score: Optional[float] = None
    overall_passed: Optional[bool] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_ms: Optional[float] = None
    
    # Error summary
    errors: List[str] = Field(default_factory=list)
    
    @property
    def overall_score(self) -> float:
        """Alias for aggregate_score for backwards compatibility."""
        return self.aggregate_score if self.aggregate_score is not None else 0.0
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Convert to dict with datetime handling."""
        data = super().model_dump(**kwargs)
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        # Convert agent evaluations
        data['agent_evaluations'] = {
            k: v.model_dump(**kwargs) for k, v in self.agent_evaluations.items()
        }
        # Add overall_score for frontend compatibility
        data['overall_score'] = self.overall_score
        return data


# =============================================================================
# AGENT-SPECIFIC EVALUATION CONFIGURATIONS
# =============================================================================

@dataclass
class AgentEvaluationConfig:
    """Configuration for evaluating a specific agent."""
    agent_id: str
    evaluators: List[EvaluatorType]
    query_field: str  # Field to use as 'query' input
    response_field: str  # Field to use as 'response' input
    context_field: Optional[str] = None  # Field for grounding context
    description: str = ""


# Define evaluation configurations per agent (STEP 2 implementation)
# NOTE: Field names must match the actual keys in agent input/output dictionaries
AGENT_EVALUATION_CONFIGS: Dict[str, AgentEvaluationConfig] = {
    "HealthDataAnalysisAgent": AgentEvaluationConfig(
        agent_id="HealthDataAnalysisAgent",
        evaluators=[EvaluatorType.GROUNDEDNESS, EvaluatorType.COHERENCE],
        query_field="health_metrics",  # Input: patient health metrics
        response_field="summary",  # Output: health analysis summary
        context_field="patient_profile",  # Context for grounding
        description="Evaluates health data extraction quality and grounding"
    ),
    "PolicyRiskAgent": AgentEvaluationConfig(
        agent_id="PolicyRiskAgent",
        evaluators=[EvaluatorType.COHERENCE, EvaluatorType.RELEVANCE],
        query_field="health_analysis",  # Input: health analysis from previous agent
        response_field="risk_category",  # Output: risk categorization
        context_field="policy_rules",  # Policy rules context
        description="Evaluates risk categorization quality"
    ),
    "BusinessRulesValidationAgent": AgentEvaluationConfig(
        agent_id="BusinessRulesValidationAgent",
        evaluators=[EvaluatorType.COHERENCE, EvaluatorType.RELEVANCE],
        query_field="health_analysis",  # Input: health analysis
        response_field="rationale",  # Output: decision rationale
        context_field="patient_profile",  # Patient context
        description="Evaluates business rule application coherence"
    ),
    "CommunicationAgent": AgentEvaluationConfig(
        agent_id="CommunicationAgent",
        evaluators=[EvaluatorType.FLUENCY, EvaluatorType.COHERENCE],
        query_field="decision_summary",  # Input: decision to communicate
        response_field="underwriter_message",  # Output: generated message
        description="Evaluates communication quality and clarity"
    ),
}


# =============================================================================
# FOUNDRY EVALUATOR SERVICE
# =============================================================================

class FoundryEvaluatorService:
    """
    Service for running Azure AI Foundry evaluations.
    
    STEP 3: Integrates with azure-ai-evaluation SDK.
    
    Features:
    - Instantiates evaluators per agent type
    - Runs evaluations AFTER agent execution
    - Captures metric scores with pass/fail thresholds
    - Non-blocking evaluation (does not affect agent execution)
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the evaluator service.
        
        Args:
            enabled: Whether evaluations are enabled. If False, all evaluations
                     are skipped with clear logging.
        """
        self.enabled = enabled
        self._evaluators: Dict[str, Any] = {}
        self._model_config: Optional[Dict[str, Any]] = None
        self._azure_ai_project: Optional[str] = None
        self._initialized = False
        
        if not enabled:
            logger.warning("FOUNDRY EVALUATIONS DISABLED - Set FOUNDRY_EVALUATIONS_ENABLED=true to enable")
    
    def _initialize_if_needed(self) -> bool:
        """
        Lazy initialization of evaluators.
        
        Returns True if initialization succeeded, False otherwise.
        """
        if self._initialized:
            return True
        
        if not self.enabled:
            return False
        
        try:
            # Check if SDK is available
            try:
                from azure.ai.evaluation import (
                    GroundednessEvaluator,
                    CoherenceEvaluator,
                    RelevanceEvaluator,
                    FluencyEvaluator,
                )
            except ImportError as e:
                logger.error(
                    "Azure AI Evaluation SDK not installed. "
                    "Install with: pip install azure-ai-evaluation"
                )
                self.enabled = False
                return False
            
            # Get model configuration from environment
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
            api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
            
            if not azure_endpoint or not deployment:
                logger.error(
                    "Missing Azure OpenAI configuration for evaluations. "
                    "Required: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME"
                )
                self.enabled = False
                return False
            
            # Azure AI Evaluation SDK requires API key auth
            # If no API key is provided, try to get one via Azure CLI or disable evaluations
            if not api_key:
                logger.warning(
                    "AZURE_OPENAI_API_KEY not set. Azure AI Evaluation SDK requires API key authentication. "
                    "Disabling evaluations. Set FOUNDRY_EVALUATIONS_ENABLED=false to suppress this warning."
                )
                self.enabled = False
                return False
            
            # Build model config for Azure AI Evaluation SDK
            # The SDK uses a specific format: azure_endpoint, api_key, azure_deployment
            self._model_config = {
                "azure_endpoint": azure_endpoint,
                "api_key": api_key,
                "azure_deployment": deployment,
                "api_version": api_version,
            }
            logger.info("Using API key authentication for evaluations")
            
            # Get Azure AI Project endpoint for safety evaluators (optional)
            self._azure_ai_project = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
            
            # Initialize evaluators
            logger.info("Initializing Azure AI Evaluation SDK evaluators...")
            
            self._evaluators[EvaluatorType.GROUNDEDNESS] = GroundednessEvaluator(
                model_config=self._model_config,
                threshold=3.0  # Score must be >= 3 to pass
            )
            self._evaluators[EvaluatorType.COHERENCE] = CoherenceEvaluator(
                model_config=self._model_config,
                threshold=3.0
            )
            self._evaluators[EvaluatorType.RELEVANCE] = RelevanceEvaluator(
                model_config=self._model_config,
                threshold=3.0
            )
            self._evaluators[EvaluatorType.FLUENCY] = FluencyEvaluator(
                model_config=self._model_config,
                threshold=3.0
            )
            
            self._initialized = True
            logger.info(
                "Azure AI Evaluation SDK initialized with %d evaluators",
                len(self._evaluators)
            )
            return True
            
        except Exception as e:
            logger.error("Failed to initialize evaluators: %s", e, exc_info=True)
            self.enabled = False
            return False
    
    async def evaluate_agent(
        self,
        agent_id: str,
        agent_input: Dict[str, Any],
        agent_output: Dict[str, Any],
        context: Optional[str] = None,
    ) -> AgentEvaluationResult:
        """
        Evaluate a single agent's execution.
        
        STEP 4: Called after each agent execution in orchestrator.
        
        Args:
            agent_id: ID of the agent being evaluated
            agent_input: Agent's input payload
            agent_output: Agent's output payload
            context: Optional context for grounding evaluation
            
        Returns:
            AgentEvaluationResult with scores and status
        """
        result = AgentEvaluationResult(agent_id=agent_id)
        result.started_at = datetime.now(timezone.utc)
        
        logger.info("EVALUATION STARTED: %s", agent_id)
        
        # Check if enabled
        if not self.enabled:
            result.status = EvaluationStatus.SKIPPED
            result.error_message = "Evaluations disabled"
            logger.info("EVALUATION SKIPPED: %s (disabled)", agent_id)
            return result
        
        # Initialize if needed
        if not self._initialize_if_needed():
            result.status = EvaluationStatus.FAILED
            result.error_message = "Failed to initialize evaluators"
            logger.error("EVALUATION FAILED: %s (initialization failed)", agent_id)
            return result
        
        # Get agent-specific config
        config = AGENT_EVALUATION_CONFIGS.get(agent_id)
        if not config:
            result.status = EvaluationStatus.SKIPPED
            result.error_message = f"No evaluation config for agent: {agent_id}"
            logger.warning("EVALUATION SKIPPED: %s (no config)", agent_id)
            return result
        
        result.status = EvaluationStatus.RUNNING
        
        try:
            # Extract query and response from agent I/O
            query = self._extract_field(agent_input, config.query_field)
            response = self._extract_field(agent_output, config.response_field)
            eval_context = context or self._extract_field(agent_input, config.context_field or "")
            
            if not query or not response:
                result.status = EvaluationStatus.FAILED
                result.error_message = "Missing query or response for evaluation"
                logger.warning("EVALUATION FAILED: %s (missing fields)", agent_id)
                return result
            
            # Run each configured evaluator
            metrics = []
            raw_outputs = {}
            
            for eval_type in config.evaluators:
                evaluator = self._evaluators.get(eval_type)
                if not evaluator:
                    logger.warning("Evaluator not available: %s", eval_type)
                    continue
                
                try:
                    # Run evaluation in thread pool to avoid blocking
                    eval_result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._run_single_evaluator(
                            evaluator, eval_type, query, response, eval_context
                        )
                    )
                    
                    if eval_result:
                        metrics.append(eval_result[0])
                        raw_outputs[eval_type.value] = eval_result[1]
                        
                except Exception as e:
                    logger.warning("Evaluator %s failed: %s", eval_type, e)
                    metrics.append(MetricScore(
                        metric_name=eval_type.value,
                        score=0.0,
                        threshold=3.0,
                        passed=False,
                        reason=f"Evaluation error: {str(e)}"
                    ))
            
            # Calculate aggregate score
            result.metrics = metrics
            if metrics:
                scores = [m.score for m in metrics if m.score > 0]
                if scores:
                    result.aggregate_score = sum(scores) / len(scores)
                    result.passed = all(m.passed for m in metrics)
            
            result.raw_sdk_output = raw_outputs
            result.status = EvaluationStatus.COMPLETED
            
        except Exception as e:
            result.status = EvaluationStatus.FAILED
            result.error_message = str(e)
            logger.error("EVALUATION FAILED: %s - %s", agent_id, e, exc_info=True)
        
        result.completed_at = datetime.now(timezone.utc)
        if result.started_at:
            result.duration_ms = (result.completed_at - result.started_at).total_seconds() * 1000
        
        logger.info(
            "EVALUATION COMPLETED: %s (status=%s, score=%.2f)",
            agent_id,
            result.status.value,
            result.aggregate_score or 0.0
        )
        
        return result
    
    def _run_single_evaluator(
        self,
        evaluator: Any,
        eval_type: EvaluatorType,
        query: str,
        response: str,
        context: Optional[str],
    ) -> Optional[Tuple[MetricScore, Dict[str, Any]]]:
        """
        Run a single evaluator and extract results.
        
        CRITICAL: This runs the actual Foundry SDK evaluator.
        DO NOT fabricate scores - if evaluation fails, return None.
        """
        try:
            # Build evaluation inputs based on evaluator type
            if eval_type == EvaluatorType.GROUNDEDNESS:
                if not context:
                    return None
                result = evaluator(response=response, context=context)
            elif eval_type == EvaluatorType.RELEVANCE:
                result = evaluator(query=query, response=response)
            else:
                # Coherence and Fluency just need query and response
                result = evaluator(query=query, response=response)
            
            # Extract score from result
            # SDK returns dict like {"groundedness": 4.0, "gpt_groundedness": 4.0, ...}
            score_key = eval_type.value
            gpt_score_key = f"gpt_{eval_type.value}"
            
            score = result.get(score_key) or result.get(gpt_score_key) or 0.0
            threshold = result.get(f"{eval_type.value}_threshold", 3.0)
            passed = result.get(f"{eval_type.value}_result") == "pass"
            reason = result.get(f"{eval_type.value}_reason")
            
            metric = MetricScore(
                metric_name=eval_type.value,
                score=float(score),
                threshold=float(threshold),
                passed=passed if passed is not None else score >= threshold,
                reason=reason
            )
            
            return metric, result
            
        except Exception as e:
            logger.warning("Evaluator %s error: %s", eval_type, e)
            return None
    
    def _extract_field(self, data: Dict[str, Any], field_path: str) -> Optional[str]:
        """Extract a field from nested dict using dot notation."""
        if not field_path or not data:
            return None
        
        parts = field_path.split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
            
            if current is None:
                return None
        
        # Convert to string if needed using safe serialization
        if isinstance(current, str):
            return current
        elif isinstance(current, (dict, list)):
            return safe_json_dumps(current, max_length=2000)
        else:
            return str(current)
    
    async def evaluate_workflow(
        self,
        workflow_id: str,
        agent_results: Dict[str, AgentEvaluationResult],
        final_decision: Optional[Dict[str, Any]] = None,
    ) -> WorkflowEvaluationResult:
        """
        Create aggregate workflow evaluation from agent evaluations.
        
        STEP 4: Called after all agents complete in orchestrator.
        
        Args:
            workflow_id: ID of the workflow
            agent_results: Per-agent evaluation results
            final_decision: Optional final decision for additional evaluation
            
        Returns:
            WorkflowEvaluationResult with aggregate metrics
        """
        result = WorkflowEvaluationResult(workflow_id=workflow_id)
        result.started_at = datetime.now(timezone.utc)
        result.agent_evaluations = agent_results
        
        logger.info("WORKFLOW EVALUATION STARTED: %s", workflow_id)
        
        try:
            # Calculate aggregate score from agent evaluations
            scores = []
            all_passed = True
            
            for agent_id, agent_eval in agent_results.items():
                if agent_eval.aggregate_score is not None:
                    scores.append(agent_eval.aggregate_score)
                if agent_eval.status == EvaluationStatus.FAILED:
                    result.errors.append(f"{agent_id}: {agent_eval.error_message}")
                if not agent_eval.passed:
                    all_passed = False
            
            if scores:
                result.aggregate_score = sum(scores) / len(scores)
            
            result.overall_passed = all_passed and len(result.errors) == 0
            result.status = EvaluationStatus.COMPLETED
            
        except Exception as e:
            result.status = EvaluationStatus.FAILED
            result.errors.append(str(e))
            logger.error("WORKFLOW EVALUATION FAILED: %s - %s", workflow_id, e)
        
        result.completed_at = datetime.now(timezone.utc)
        if result.started_at:
            result.total_duration_ms = (result.completed_at - result.started_at).total_seconds() * 1000
        
        logger.info(
            "WORKFLOW EVALUATION COMPLETED: %s (passed=%s, score=%.2f)",
            workflow_id,
            result.overall_passed,
            result.aggregate_score or 0.0
        )
        
        return result


# =============================================================================
# GLOBAL SERVICE INSTANCE
# =============================================================================

_evaluator_service: Optional[FoundryEvaluatorService] = None


def get_evaluator_service() -> FoundryEvaluatorService:
    """Get or create the global evaluator service instance."""
    global _evaluator_service
    
    if _evaluator_service is None:
        # Check feature flag
        enabled = os.environ.get("FOUNDRY_EVALUATIONS_ENABLED", "false").lower() == "true"
        _evaluator_service = FoundryEvaluatorService(enabled=enabled)
    
    return _evaluator_service


def is_evaluations_enabled() -> bool:
    """Check if evaluations are enabled."""
    return os.environ.get("FOUNDRY_EVALUATIONS_ENABLED", "false").lower() == "true"
