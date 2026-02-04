"""Token tracking module for LLM and agent executions.

This module provides centralized token counting and cost tracking for:
- Direct LLM calls (chat completions)
- Agent executions via Azure AI Foundry
- Multi-agent workflow orchestrations

Uses tiktoken for token counting with fallback estimation.
Persists token usage to Cosmos DB token_tracking container.
"""

from __future__ import annotations

import logging
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from functools import lru_cache
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    logger.info("tiktoken loaded successfully for token counting")
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available - using estimation fallback")


# =============================================================================
# PRICING CONSTANTS (USD per 1K tokens as of 2024)
# =============================================================================

# Azure OpenAI pricing varies by region/deployment - these are estimates
MODEL_PRICING = {
    # GPT-4 models
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4.1": {"prompt": 0.002, "completion": 0.008},
    "gpt-4.1-mini": {"prompt": 0.0004, "completion": 0.0016},
    "gpt-4.1-nano": {"prompt": 0.0001, "completion": 0.0004},
    
    # GPT-3.5 models
    "gpt-35-turbo": {"prompt": 0.0015, "completion": 0.002},
    "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
    "gpt-35-turbo-16k": {"prompt": 0.003, "completion": 0.004},
    
    # Default fallback
    "default": {"prompt": 0.002, "completion": 0.008},
}


# =============================================================================
# TOKEN COUNTER CLASS
# =============================================================================

class TokenCounter:
    """Thread-safe token counter using tiktoken or estimation fallback."""
    
    _instance: Optional["TokenCounter"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "TokenCounter":
        """Singleton pattern for token counter."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._encoders: Dict[str, Any] = {}
        self._initialized = True
        logger.debug("TokenCounter initialized")
    
    @lru_cache(maxsize=10)
    def _get_encoder(self, model: str) -> Any:
        """Get or create tiktoken encoder for a model."""
        if not TIKTOKEN_AVAILABLE:
            return None
        
        try:
            # Map Azure deployment names to tiktoken model names
            model_lower = model.lower()
            
            # GPT-4 family
            if "gpt-4o" in model_lower or "gpt-4.1" in model_lower:
                return tiktoken.encoding_for_model("gpt-4o")
            elif "gpt-4" in model_lower:
                return tiktoken.encoding_for_model("gpt-4")
            # GPT-3.5 family
            elif "gpt-35" in model_lower or "gpt-3.5" in model_lower:
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Default to cl100k_base encoding (used by most modern models)
                return tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not get tiktoken encoder for {model}: {e}")
            return tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text using tiktoken or estimation.
        
        Args:
            text: The text to count tokens for.
            model: The model name (used for encoder selection).
            
        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        
        encoder = self._get_encoder(model)
        if encoder:
            try:
                return len(encoder.encode(text))
            except Exception as e:
                logger.debug(f"tiktoken encoding failed: {e}")
        
        # Fallback: estimate ~4 chars per token
        return len(text) // 4
    
    def count_messages_tokens(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "gpt-4"
    ) -> int:
        """Count tokens in a list of chat messages.
        
        Accounts for message structure overhead per OpenAI's token counting.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            model: The model name.
            
        Returns:
            Total token count including message overhead.
        """
        if not messages:
            return 0
        
        # Per-message overhead varies by model, use ~4 tokens as estimate
        tokens_per_message = 4
        total = 0
        
        for message in messages:
            total += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    total += self.count_tokens(value, model)
        
        # Add reply priming tokens
        total += 3
        
        return total


# Global token counter instance
_token_counter = TokenCounter()


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text (convenience function)."""
    return _token_counter.count_tokens(text, model)


def count_messages_tokens(messages: List[Dict[str, str]], model: str = "gpt-4") -> int:
    """Count tokens in messages (convenience function)."""
    return _token_counter.count_messages_tokens(messages, model)


# =============================================================================
# COST CALCULATOR
# =============================================================================

def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gpt-4",
) -> Tuple[float, float, float]:
    """Calculate cost in USD for token usage.
    
    Args:
        prompt_tokens: Number of input/prompt tokens.
        completion_tokens: Number of output/completion tokens.
        model: Model name for pricing lookup.
        
    Returns:
        Tuple of (prompt_cost, completion_cost, total_cost) in USD.
    """
    # Normalize model name for pricing lookup
    model_lower = model.lower()
    
    # Find matching pricing
    pricing = MODEL_PRICING.get("default")
    for model_key, model_pricing in MODEL_PRICING.items():
        if model_key in model_lower:
            pricing = model_pricing
            break
    
    prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1000) * pricing["completion"]
    total_cost = prompt_cost + completion_cost
    
    return (round(prompt_cost, 6), round(completion_cost, 6), round(total_cost, 6))


# =============================================================================
# TOKEN TRACKING CONTEXT
# =============================================================================

class TokenTrackingContext:
    """Context for tracking tokens across an execution.
    
    Collects token usage data that can be persisted to Cosmos DB.
    """
    
    def __init__(
        self,
        execution_id: str,
        application_id: str,
        operation_type: str = "workflow",
    ):
        self.execution_id = execution_id
        self.application_id = application_id
        self.operation_type = operation_type
        self.records: List[Dict[str, Any]] = []
        self._step_counter = 0
        self._lock = threading.Lock()
    
    def record_usage(
        self,
        agent_id: str,
        agent_type: str,
        prompt_tokens: int,
        completion_tokens: int,
        model_name: Optional[str] = None,
        deployment_name: Optional[str] = None,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record token usage for an operation.
        
        Args:
            agent_id: Identifier of the agent or operation.
            agent_type: Type/category of the agent.
            prompt_tokens: Input tokens consumed.
            completion_tokens: Output tokens generated.
            model_name: Name of the model used.
            deployment_name: Azure deployment name.
            success: Whether the operation succeeded.
            metadata: Additional metadata to store.
            
        Returns:
            The token tracking record.
        """
        with self._lock:
            self._step_counter += 1
            step_number = self._step_counter
        
        total_tokens = prompt_tokens + completion_tokens
        prompt_cost, completion_cost, total_cost = calculate_cost(
            prompt_tokens, completion_tokens, model_name or "gpt-4"
        )
        
        record = {
            "id": str(uuid4()),
            "record_id": str(uuid4()),
            "execution_id": self.execution_id,
            "application_id": self.application_id,
            "agent_id": agent_id,
            "agent_type": agent_type,
            "step_number": step_number,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_cost_usd": prompt_cost,
            "completion_cost_usd": completion_cost,
            "total_cost_usd": total_cost,
            "model_name": model_name,
            "deployment_name": deployment_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation_type": self.operation_type,
            "success": success,
            "metadata": metadata or {},
        }
        
        with self._lock:
            self.records.append(record)
        
        logger.debug(
            "Token usage recorded: %s - %d tokens ($%.4f)",
            agent_id, total_tokens, total_cost
        )
        
        return record
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all token usage in this context.
        
        Returns:
            Summary dict with totals and breakdown by agent.
        """
        total_prompt = 0
        total_completion = 0
        total_cost = 0.0
        by_agent: Dict[str, Dict[str, Any]] = {}
        
        for record in self.records:
            agent_id = record["agent_id"]
            total_prompt += record["prompt_tokens"]
            total_completion += record["completion_tokens"]
            total_cost += record["total_cost_usd"]
            
            if agent_id not in by_agent:
                by_agent[agent_id] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0,
                    "calls": 0,
                }
            
            by_agent[agent_id]["prompt_tokens"] += record["prompt_tokens"]
            by_agent[agent_id]["completion_tokens"] += record["completion_tokens"]
            by_agent[agent_id]["total_tokens"] += record["total_tokens"]
            by_agent[agent_id]["total_cost_usd"] += record["total_cost_usd"]
            by_agent[agent_id]["calls"] += 1
        
        return {
            "execution_id": self.execution_id,
            "application_id": self.application_id,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost_usd": round(total_cost, 6),
            "total_calls": len(self.records),
            "by_agent": by_agent,
        }


# =============================================================================
# GLOBAL TRACKING STATE
# =============================================================================

# Thread-local storage for current tracking context
_tracking_contexts: Dict[str, TokenTrackingContext] = {}
_contexts_lock = threading.Lock()


def create_tracking_context(
    execution_id: str,
    application_id: str,
    operation_type: str = "workflow",
) -> TokenTrackingContext:
    """Create a new token tracking context.
    
    Args:
        execution_id: Unique execution/workflow ID.
        application_id: Application being processed.
        operation_type: Type of operation (workflow, chat, analysis).
        
    Returns:
        New TokenTrackingContext instance.
    """
    context = TokenTrackingContext(execution_id, application_id, operation_type)
    with _contexts_lock:
        _tracking_contexts[execution_id] = context
    logger.debug("Created tracking context for execution %s", execution_id)
    return context


def get_tracking_context(execution_id: str) -> Optional[TokenTrackingContext]:
    """Get tracking context by execution ID."""
    with _contexts_lock:
        return _tracking_contexts.get(execution_id)


def close_tracking_context(execution_id: str) -> Optional[TokenTrackingContext]:
    """Close and remove a tracking context.
    
    Returns the context for final persistence before removal.
    """
    with _contexts_lock:
        return _tracking_contexts.pop(execution_id, None)


# =============================================================================
# PERSISTENCE TO COSMOS DB
# =============================================================================

async def persist_token_records(
    records: List[Dict[str, Any]],
    cosmos_service: Optional[Any] = None,
) -> int:
    """Persist token tracking records to Cosmos DB.
    
    Args:
        records: List of token tracking records.
        cosmos_service: Optional Cosmos service instance.
        
    Returns:
        Number of records successfully persisted.
    """
    if not records:
        return 0
    
    # Get cosmos service if not provided
    if cosmos_service is None:
        try:
            from app.cosmos.service import get_cosmos_service
            cosmos_service = await get_cosmos_service()
        except Exception as e:
            logger.warning(f"Could not get Cosmos service for token persistence: {e}")
            return 0
    
    if not cosmos_service.is_available:
        logger.debug("Cosmos DB not available, skipping token persistence")
        return 0
    
    saved = 0
    for record in records:
        try:
            from app.cosmos.models import TokenTrackingDocument
            token_doc = TokenTrackingDocument(**record)
            success = await cosmos_service.save_token_usage(token_doc)
            if success:
                saved += 1
        except Exception as e:
            logger.warning(f"Failed to persist token record: {e}")
    
    logger.info(f"Persisted {saved}/{len(records)} token tracking records to Cosmos DB")
    return saved


async def persist_context(context: TokenTrackingContext) -> int:
    """Persist all records from a tracking context.
    
    Args:
        context: The tracking context to persist.
        
    Returns:
        Number of records persisted.
    """
    return await persist_token_records(context.records)


# =============================================================================
# TRACKING DECORATOR FOR LLM CALLS
# =============================================================================

def track_llm_call(
    agent_id: str = "direct_llm",
    agent_type: str = "chat_completion",
    execution_id: Optional[str] = None,
    application_id: Optional[str] = None,
):
    """Decorator to track token usage for LLM calls.
    
    Usage:
        @track_llm_call(agent_id="chat", execution_id="exec123")
        def my_llm_call(messages):
            return chat_completion(settings, messages)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Extract usage from result
            usage = result.get("usage", {}) if isinstance(result, dict) else {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            
            # Get or create context
            exec_id = execution_id or str(uuid4())
            app_id = application_id or "unknown"
            
            context = get_tracking_context(exec_id)
            if context is None:
                context = create_tracking_context(exec_id, app_id, "llm_call")
            
            # Record usage
            context.record_usage(
                agent_id=agent_id,
                agent_type=agent_type,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                metadata={"function": func.__name__},
            )
            
            return result
        return wrapper
    return decorator


# =============================================================================
# STANDALONE TRACKING FUNCTIONS
# =============================================================================

async def track_chat_completion(
    result: Dict[str, Any],
    messages: List[Dict[str, str]],
    agent_id: str = "chat_completion",
    execution_id: Optional[str] = None,
    application_id: Optional[str] = None,
    model_name: Optional[str] = None,
    deployment_name: Optional[str] = None,
    operation_type: str = "chat",
    persist: bool = True,
) -> Dict[str, Any]:
    """Track token usage from a chat completion result.
    
    Can be called after any chat_completion call to record usage.
    
    Args:
        result: The result dict from chat_completion.
        messages: The messages sent to the LLM.
        agent_id: Identifier for this operation.
        execution_id: Execution context ID.
        application_id: Application being processed.
        model_name: Model name for cost calculation.
        deployment_name: Azure deployment name.
        operation_type: Type of operation.
        persist: Whether to persist to Cosmos immediately.
        
    Returns:
        Token tracking record.
    """
    usage = result.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    
    # If no usage data, estimate from messages and response
    if prompt_tokens == 0 and messages:
        prompt_tokens = count_messages_tokens(messages, model_name or "gpt-4")
    if completion_tokens == 0 and result.get("content"):
        completion_tokens = count_tokens(result["content"], model_name or "gpt-4")
    
    exec_id = execution_id or str(uuid4())
    app_id = application_id or "unknown"
    
    # Create record
    total_tokens = prompt_tokens + completion_tokens
    prompt_cost, completion_cost, total_cost = calculate_cost(
        prompt_tokens, completion_tokens, model_name or "gpt-4"
    )
    
    record = {
        "id": str(uuid4()),
        "record_id": str(uuid4()),
        "execution_id": exec_id,
        "application_id": app_id,
        "agent_id": agent_id,
        "agent_type": "llm_direct",
        "step_number": 1,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_cost_usd": prompt_cost,
        "completion_cost_usd": completion_cost,
        "total_cost_usd": total_cost,
        "model_name": model_name,
        "deployment_name": deployment_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation_type": operation_type,
        "success": True,
        "metadata": {},
    }
    
    logger.info(
        "Token usage: %s - %d prompt + %d completion = %d total ($%.4f)",
        agent_id, prompt_tokens, completion_tokens, total_tokens, total_cost
    )
    
    # Persist if requested
    if persist:
        await persist_token_records([record])
    
    return record


async def track_agent_execution(
    agent_id: str,
    result: Any,
    execution_id: str,
    application_id: str,
    step_number: int = 1,
    model_name: Optional[str] = None,
    persist: bool = True,
) -> Optional[Dict[str, Any]]:
    """Track token usage from an agent execution result.
    
    Works with both Foundry agent results and local agent outputs.
    
    Args:
        agent_id: Agent identifier (e.g., "HealthDataAnalysisAgent").
        result: Result from agent invocation (dict or AgentInvocationResult).
        execution_id: Workflow execution ID.
        application_id: Application being processed.
        step_number: Step in the workflow.
        model_name: Model used by the agent.
        persist: Whether to persist to Cosmos immediately.
        
    Returns:
        Token tracking record if usage data available, None otherwise.
    """
    # Extract token usage from result
    token_usage = None
    
    if isinstance(result, dict):
        token_usage = result.get("token_usage")
    elif hasattr(result, "token_usage"):
        token_usage = result.token_usage
    
    if not token_usage:
        logger.debug("No token usage data available for agent %s", agent_id)
        return None
    
    # Handle different token_usage formats
    if isinstance(token_usage, dict):
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
    else:
        prompt_tokens = getattr(token_usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(token_usage, "completion_tokens", 0) or 0
    
    if prompt_tokens == 0 and completion_tokens == 0:
        logger.debug("Zero tokens recorded for agent %s", agent_id)
        return None
    
    total_tokens = prompt_tokens + completion_tokens
    prompt_cost, completion_cost, total_cost = calculate_cost(
        prompt_tokens, completion_tokens, model_name or "gpt-4"
    )
    
    record = {
        "id": str(uuid4()),
        "record_id": str(uuid4()),
        "execution_id": execution_id,
        "application_id": application_id,
        "agent_id": agent_id,
        "agent_type": "foundry_agent",
        "step_number": step_number,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "prompt_cost_usd": prompt_cost,
        "completion_cost_usd": completion_cost,
        "total_cost_usd": total_cost,
        "model_name": model_name,
        "deployment_name": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation_type": "agent_execution",
        "success": True,
        "metadata": {
            "execution_time_ms": result.get("execution_time_ms") if isinstance(result, dict) else getattr(result, "execution_time_ms", None),
        },
    }
    
    logger.info(
        "Agent token usage: %s (step %d) - %d prompt + %d completion = %d total ($%.4f)",
        agent_id, step_number, prompt_tokens, completion_tokens, total_tokens, total_cost
    )
    
    if persist:
        await persist_token_records([record])
    
    return record
