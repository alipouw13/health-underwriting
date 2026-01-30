"""
Base Agent Class for Azure AI Foundry Underwriting Agents

This module provides the abstract base class and common types for all
underwriting agents. Follows patterns from the Microsoft Multi-Agent
Custom Automation Engine Solution Accelerator.

Design Principles:
-----------------
1. Azure AI Foundry agent abstractions only (no LangChain, AutoGen, CrewAI)
2. Structured input/output validation at runtime
3. Isolated agents - no inter-agent calls
4. No orchestration logic
5. No UI state dependencies
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class AgentExecutionError(Exception):
    """Raised when an agent fails during execution."""
    
    def __init__(self, agent_id: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.details = details or {}
        super().__init__(f"[{agent_id}] {message}")


class AgentValidationError(Exception):
    """Raised when agent input or output validation fails."""
    
    def __init__(self, agent_id: str, validation_type: str, errors: List[str]):
        self.agent_id = agent_id
        self.validation_type = validation_type  # 'input' or 'output'
        self.errors = errors
        super().__init__(
            f"[{agent_id}] {validation_type.title()} validation failed: {'; '.join(errors)}"
        )


# =============================================================================
# BASE INPUT/OUTPUT MODELS
# =============================================================================

class AgentInput(BaseModel):
    """Base class for all agent inputs. Subclass per agent."""
    model_config = ConfigDict(extra="forbid")  # Strict validation


class AgentOutput(BaseModel):
    """Base class for all agent outputs. Subclass per agent."""
    model_config = ConfigDict(extra="forbid")
    
    execution_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique execution ID")
    agent_id: str = Field(..., description="ID of the agent that produced this output")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When output was produced")
    execution_time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")
    success: bool = Field(default=True, description="Whether execution succeeded")
    error_message: Optional[str] = Field(None, description="Error message if success=False")


# Type variables for generic agent typing
InputT = TypeVar("InputT", bound=AgentInput)
OutputT = TypeVar("OutputT", bound=AgentOutput)


# =============================================================================
# MCP TOOL INTERFACE (STUB)
# =============================================================================

class MCPToolStub:
    """
    Stubbed MCP tool interface for agent-tool interactions.
    
    In production, this would connect to actual MCP servers:
    - medical-mcp-simulator
    - policy-rule-engine
    - underwriting-rules-mcp
    - data-quality-analyzer
    - fairness-checker
    - language-generator
    - trace-logger
    
    For development/testing, this provides a stub implementation.
    """
    
    def __init__(self, tool_name: str, server_url: Optional[str] = None):
        self.tool_name = tool_name
        self.server_url = server_url
        self.logger = logging.getLogger(f"mcp.{tool_name}")
    
    async def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stub method - in production this would make actual MCP calls.
        
        Args:
            method: MCP method name
            params: Method parameters
            
        Returns:
            Stubbed response dict
        """
        self.logger.info(f"[STUB] MCP call: {self.tool_name}.{method}({params})")
        return {"stub": True, "tool": self.tool_name, "method": method}


# =============================================================================
# BASE AGENT CLASS
# =============================================================================

class BaseUnderwritingAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for all underwriting agents.
    
    Follows Azure AI Foundry agent patterns from the Microsoft Multi-Agent
    Custom Automation Engine Solution Accelerator.
    
    Each agent:
    - Has a unique agent_id matching the YAML definition
    - Has a defined purpose (from YAML)
    - Accepts typed input (InputT)
    - Returns typed output (OutputT)
    - May use MCP tools (stubbed for now)
    - Validates input/output at runtime
    - Is completely isolated from other agents
    
    Subclasses must implement:
    - input_type: ClassVar specifying the Pydantic input model
    - output_type: ClassVar specifying the Pydantic output model
    - _execute(): Core agent logic
    """
    
    # Class-level attributes to be defined by subclasses
    agent_id: str
    purpose: str
    tools_used: List[str]
    evaluation_criteria: List[str]
    failure_modes: List[str]
    
    def __init__(self, mcp_tools: Optional[Dict[str, MCPToolStub]] = None):
        """
        Initialize the agent.
        
        Args:
            mcp_tools: Optional dict of MCP tool stubs keyed by tool name
        """
        self.mcp_tools = mcp_tools or {}
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        self._execution_count = 0
    
    @property
    @abstractmethod
    def input_type(self) -> type[InputT]:
        """Return the Pydantic model class for input validation."""
        ...
    
    @property
    @abstractmethod
    def output_type(self) -> type[OutputT]:
        """Return the Pydantic model class for output validation."""
        ...
    
    @abstractmethod
    async def _execute(self, validated_input: InputT) -> OutputT:
        """
        Core agent execution logic. Override in subclasses.
        
        Args:
            validated_input: Already validated input data
            
        Returns:
            Agent output (will be validated before return)
        """
        ...
    
    def validate_input(self, input_data: Dict[str, Any]) -> InputT:
        """
        Validate input data against the agent's input schema.
        
        Args:
            input_data: Raw input dict
            
        Returns:
            Validated input model instance
            
        Raises:
            AgentValidationError: If validation fails
        """
        try:
            return self.input_type.model_validate(input_data)
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            raise AgentValidationError(self.agent_id, "input", errors) from e
    
    def validate_output(self, output_data: OutputT) -> OutputT:
        """
        Validate output data against the agent's output schema.
        
        Args:
            output_data: Output model instance
            
        Returns:
            Same output (validated)
            
        Raises:
            AgentValidationError: If validation fails
        """
        try:
            # Re-validate to ensure all fields are correct
            return self.output_type.model_validate(output_data.model_dump())
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            raise AgentValidationError(self.agent_id, "output", errors) from e
    
    async def run(self, input_data: Dict[str, Any]) -> OutputT:
        """
        Main execution entry point. Validates input, executes, validates output.
        
        Args:
            input_data: Raw input dict matching the agent's input schema
            
        Returns:
            Validated output model instance
            
        Raises:
            AgentValidationError: If input or output validation fails
            AgentExecutionError: If execution fails
        """
        execution_id = str(uuid4())
        start_time = datetime.now(timezone.utc)
        
        self.logger.info(f"Starting execution {execution_id}")
        self._execution_count += 1
        
        try:
            # Validate input
            validated_input = self.validate_input(input_data)
            self.logger.debug(f"Input validated: {type(validated_input).__name__}")
            
            # Execute agent logic
            output = await self._execute(validated_input)
            
            # Set execution metadata
            end_time = datetime.now(timezone.utc)
            output.execution_id = execution_id
            output.agent_id = self.agent_id
            output.timestamp = end_time
            output.execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Validate output
            validated_output = self.validate_output(output)
            self.logger.info(f"Execution {execution_id} completed in {output.execution_time_ms:.2f}ms")
            
            return validated_output
            
        except AgentValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Execution {execution_id} failed: {e}", exc_info=True)
            raise AgentExecutionError(
                self.agent_id, 
                str(e),
                {"execution_id": execution_id}
            ) from e
    
    def get_mcp_tool(self, tool_name: str) -> MCPToolStub:
        """
        Get an MCP tool by name, creating a stub if not provided.
        
        Args:
            tool_name: Name of the MCP tool
            
        Returns:
            MCPToolStub instance
        """
        if tool_name not in self.mcp_tools:
            self.mcp_tools[tool_name] = MCPToolStub(tool_name)
        return self.mcp_tools[tool_name]
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(agent_id={self.agent_id!r})>"
