"""
Azure AI Foundry Evaluations Integration (V3 - Simplified)

This module provides simplified agent evaluation that:
1. Captures basic quality metrics without external API calls
2. Stores evaluation metadata for observability
3. Can be extended with Foundry evals when properly configured

The Foundry evals API requires specific data formats and configurations.
This simplified version provides immediate value while the full integration
is being developed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal
import re

from pydantic import BaseModel, Field

logger = logging.getLogger("underwriting_assistant.evaluations")


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


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

class EvaluatorType(str, Enum):
    """Types of evaluators."""
    COMPLETENESS = "completeness"
    STRUCTURE = "structure"  
    RELEVANCE = "relevance"
    COHERENCE = "coherence"


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
    """Evaluation result for a single agent execution."""
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
    
    # Raw output
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
        data['overall_score'] = self.overall_score
        return data


class WorkflowEvaluationResult(BaseModel):
    """Aggregate evaluation result for entire workflow."""
    workflow_id: str
    evaluation_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    status: EvaluationStatus = Field(default=EvaluationStatus.PENDING)
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_ms: Optional[float] = None
    
    # Aggregate results
    agent_evaluations: Dict[str, AgentEvaluationResult] = Field(default_factory=dict)
    aggregate_score: Optional[float] = None
    overall_passed: Optional[bool] = None
    
    # Errors
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
        data['overall_score'] = self.overall_score
        return data


# =============================================================================
# AGENT EVALUATION CONFIGURATIONS  
# =============================================================================

@dataclass
class AgentEvaluationConfig:
    """Configuration for evaluating a specific agent."""
    agent_id: str
    evaluators: List[EvaluatorType]
    expected_output_fields: List[str] = field(default_factory=list)
    min_response_length: int = 50


# Configure which evaluators run for each agent
AGENT_EVALUATION_CONFIGS: Dict[str, AgentEvaluationConfig] = {
    "HealthDataAnalysisAgent": AgentEvaluationConfig(
        agent_id="HealthDataAnalysisAgent",
        evaluators=[EvaluatorType.COMPLETENESS, EvaluatorType.STRUCTURE],
        expected_output_fields=["summary", "risk_indicators", "risk_level"],
        min_response_length=100,
    ),
    "PolicyRiskAgent": AgentEvaluationConfig(
        agent_id="PolicyRiskAgent",
        evaluators=[EvaluatorType.COMPLETENESS, EvaluatorType.RELEVANCE],
        expected_output_fields=["risk_assessment", "policy_recommendation"],
        min_response_length=50,
    ),
    "AppleHealthRiskAgent": AgentEvaluationConfig(
        agent_id="AppleHealthRiskAgent",
        evaluators=[EvaluatorType.COMPLETENESS, EvaluatorType.STRUCTURE, EvaluatorType.COHERENCE],
        expected_output_fields=["hkrs_score", "hkrs_band", "category_scores", "risk_class_recommendation", "rationale"],
        min_response_length=100,
    ),
    "BusinessRulesValidationAgent": AgentEvaluationConfig(
        agent_id="BusinessRulesValidationAgent",
        evaluators=[EvaluatorType.COMPLETENESS, EvaluatorType.COHERENCE],
        expected_output_fields=["approved", "rationale", "premium_adjustment_percentage"],
        min_response_length=50,
    ),
    "CommunicationAgent": AgentEvaluationConfig(
        agent_id="CommunicationAgent",
        evaluators=[EvaluatorType.COMPLETENESS, EvaluatorType.COHERENCE],
        expected_output_fields=["underwriter_message", "customer_message"],
        min_response_length=100,
    ),
}


# =============================================================================
# SIMPLIFIED EVALUATOR SERVICE
# =============================================================================

class FoundryEvaluatorService:
    """
    Simplified evaluator service that provides basic quality metrics.
    
    This version:
    - Evaluates output completeness (are expected fields present?)
    - Evaluates response structure (is the output well-formed?)
    - Calculates simple quality scores without external API calls
    - Stores evaluation metadata for observability
    
    For full LLM-based evaluation, Azure AI Evaluation SDK requires
    API key authentication which may not be available in all environments.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._initialized = False
        
        if not enabled:
            logger.warning("FOUNDRY EVALUATIONS DISABLED - Set FOUNDRY_EVALUATIONS_ENABLED=true to enable")
    
    def _initialize_if_needed(self) -> bool:
        """Initialize the service."""
        if self._initialized:
            return True
        
        if not self.enabled:
            return False
        
        self._initialized = True
        logger.info("Simplified evaluation service initialized")
        return True
    
    async def evaluate_agent(
        self,
        agent_id: str,
        agent_input: Dict[str, Any],
        agent_output: Dict[str, Any],
        context: Optional[str] = None,
    ) -> AgentEvaluationResult:
        """
        Evaluate a single agent's execution using simplified metrics.
        
        Evaluates:
        - Completeness: Are expected output fields present?
        - Structure: Is the output well-formed JSON/dict?
        - Response length: Is the response substantive?
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
            result.error_message = "Failed to initialize evaluator"
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
            metrics = []
            
            # Evaluate completeness - are expected fields present?
            completeness_score, completeness_reason = self._evaluate_completeness(
                agent_output, config.expected_output_fields
            )
            metrics.append(MetricScore(
                metric_name="completeness",
                score=completeness_score,
                threshold=3.0,
                passed=completeness_score >= 3.0,
                reason=completeness_reason
            ))
            
            # Evaluate structure - is output well-formed?
            structure_score, structure_reason = self._evaluate_structure(agent_output)
            metrics.append(MetricScore(
                metric_name="structure",
                score=structure_score,
                threshold=3.0,
                passed=structure_score >= 3.0,
                reason=structure_reason
            ))
            
            # Evaluate response length
            length_score, length_reason = self._evaluate_length(
                agent_output, config.min_response_length
            )
            metrics.append(MetricScore(
                metric_name="response_length",
                score=length_score,
                threshold=3.0,
                passed=length_score >= 3.0,
                reason=length_reason
            ))
            
            # Calculate aggregate score
            result.metrics = metrics
            scores = [m.score for m in metrics]
            result.aggregate_score = sum(scores) / len(scores) if scores else 0.0
            result.passed = all(m.passed for m in metrics)
            result.status = EvaluationStatus.COMPLETED
            
            # Store raw evaluation data
            result.raw_sdk_output = {
                "input_keys": list(agent_input.keys()) if isinstance(agent_input, dict) else [],
                "output_keys": list(agent_output.keys()) if isinstance(agent_output, dict) else [],
                "expected_fields": config.expected_output_fields,
            }
            
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
    
    def _evaluate_completeness(
        self, 
        output: Dict[str, Any], 
        expected_fields: List[str]
    ) -> Tuple[float, str]:
        """
        Evaluate if expected fields are present in output.
        
        Returns (score, reason) where score is 1-5.
        """
        if not expected_fields:
            return 5.0, "No expected fields configured"
        
        if not isinstance(output, dict):
            return 1.0, "Output is not a dictionary"
        
        present_fields = []
        missing_fields = []
        
        for field in expected_fields:
            # Check if field exists and has a non-empty value
            value = output.get(field)
            if value is not None and value != "" and value != []:
                present_fields.append(field)
            else:
                missing_fields.append(field)
        
        # Calculate score based on percentage of fields present
        if len(expected_fields) > 0:
            ratio = len(present_fields) / len(expected_fields)
            score = 1.0 + (ratio * 4.0)  # Scale to 1-5
        else:
            score = 5.0
        
        if missing_fields:
            reason = f"Missing fields: {', '.join(missing_fields)}"
        else:
            reason = f"All {len(expected_fields)} expected fields present"
        
        return round(score, 1), reason
    
    def _evaluate_structure(self, output: Dict[str, Any]) -> Tuple[float, str]:
        """
        Evaluate if output is well-structured.
        
        Returns (score, reason) where score is 1-5.
        """
        if not isinstance(output, dict):
            return 1.0, "Output is not a dictionary"
        
        if not output:
            return 1.0, "Output is empty"
        
        # Check for nested structure (good) vs flat (ok)
        has_nested = any(isinstance(v, (dict, list)) for v in output.values())
        
        # Check for meaningful keys (not just "output" or "result")
        meaningful_keys = [k for k in output.keys() if k not in ("output", "result", "data", "agent_id", "status")]
        
        # Calculate score
        score = 3.0  # Base score
        
        if has_nested:
            score += 1.0  # Bonus for structured data
        
        if len(meaningful_keys) >= 3:
            score += 1.0  # Bonus for multiple meaningful fields
        elif len(meaningful_keys) < 1:
            score -= 1.0  # Penalty for no meaningful fields
        
        score = max(1.0, min(5.0, score))  # Clamp to 1-5
        
        reason = f"{len(output)} fields, {'nested structure' if has_nested else 'flat structure'}"
        
        return round(score, 1), reason
    
    def _evaluate_length(
        self, 
        output: Dict[str, Any], 
        min_length: int
    ) -> Tuple[float, str]:
        """
        Evaluate if response is substantive (not too short).
        
        Returns (score, reason) where score is 1-5.
        """
        # Serialize output to measure total content length
        try:
            content = json.dumps(output, default=str)
            length = len(content)
        except Exception:
            content = str(output)
            length = len(content)
        
        if length < min_length // 2:
            score = 1.0
            reason = f"Very short response ({length} chars, expected >= {min_length})"
        elif length < min_length:
            score = 2.5
            reason = f"Short response ({length} chars, expected >= {min_length})"
        elif length < min_length * 2:
            score = 4.0
            reason = f"Adequate response length ({length} chars)"
        else:
            score = 5.0
            reason = f"Comprehensive response ({length} chars)"
        
        return round(score, 1), reason
    
    async def evaluate_workflow(
        self,
        workflow_id: str,
        agent_results: Dict[str, AgentEvaluationResult],
        final_decision: Optional[Dict[str, Any]] = None,
    ) -> WorkflowEvaluationResult:
        """Create aggregate workflow evaluation from agent evaluations."""
        result = WorkflowEvaluationResult(workflow_id=workflow_id)
        result.started_at = datetime.now(timezone.utc)
        result.agent_evaluations = agent_results
        
        logger.info("WORKFLOW EVALUATION STARTED: %s", workflow_id)
        
        try:
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
            else:
                result.aggregate_score = 0.0
            
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

# Alias for backwards compatibility  
FoundryEvaluatorServiceV2 = FoundryEvaluatorService

_evaluator_service: Optional[FoundryEvaluatorService] = None


def get_evaluator_service() -> FoundryEvaluatorService:
    """Get or create the global evaluator service instance."""
    global _evaluator_service
    
    if _evaluator_service is None:
        enabled = os.environ.get("FOUNDRY_EVALUATIONS_ENABLED", "false").lower() == "true"
        _evaluator_service = FoundryEvaluatorService(enabled=enabled)
    
    return _evaluator_service


def is_evaluations_enabled() -> bool:
    """Check if evaluations are enabled."""
    return os.environ.get("FOUNDRY_EVALUATIONS_ENABLED", "false").lower() == "true"
