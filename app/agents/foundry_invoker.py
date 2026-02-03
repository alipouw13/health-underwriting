"""
Azure AI Foundry Agent Invoker

This module provides the ability to invoke deployed agents in Azure AI Foundry,
making real LLM calls with function tool support.

Implements the Azure AI Foundry function calling pattern:
1. Create agent with tool definitions
2. Create thread and add message
3. Run agent and poll for status
4. Handle requires_action status by executing tools locally
5. Submit tool outputs back to agent
6. Continue until completed

Reference: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/how-to/tools/function-calling
"""

import logging
import os
import json
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone, date

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime and date objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)


@dataclass
class AgentInvocationResult:
    """Result from invoking an Azure AI Foundry agent."""
    agent_id: str
    foundry_agent_id: str
    success: bool
    response: Optional[str] = None
    parsed_output: Optional[Dict[str, Any]] = None
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    tools_executed: List[Dict[str, Any]] = field(default_factory=list)  # Track which tools were called


class FoundryAgentInvoker:
    """
    Invokes deployed agents in Azure AI Foundry via the Agents API.
    
    Implements the full function calling loop:
    1. Create thread and add message
    2. Start run with agent
    3. Poll for status
    4. If requires_action: execute tools locally, submit outputs
    5. Continue polling until completed
    6. Retrieve response messages
    
    API Pattern (Azure AI Agents SDK 1.x):
    - client.threads.create() - create thread
    - client.messages.create() - add message to thread  
    - client.runs.create() - start run (NOT create_and_process for tool handling)
    - client.runs.get() - poll run status
    - client.runs.submit_tool_outputs() - submit tool results
    - client.messages.list() - get response messages
    - client.threads.delete() - cleanup
    """
    
    # Maximum time to wait for a run to complete (10 minutes per Foundry docs)
    MAX_RUN_TIME_SECONDS = 600
    POLL_INTERVAL_SECONDS = 0.5
    MAX_POLL_INTERVAL_SECONDS = 5.0
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._client = None
        self._credential = None
        
        self.project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
        self.model_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
        
        # Cache of agent IDs
        self._agent_cache: Dict[str, str] = {}
        
        # Import tool execution function
        try:
            from app.agents.agent_tools import execute_tool, get_tools_for_agent
            self._execute_tool = execute_tool
            self._get_tools_for_agent = get_tools_for_agent
            self.logger.info("Agent tools module loaded successfully")
        except ImportError as e:
            self.logger.warning("Agent tools module not available: %s", e)
            self._execute_tool = None
            self._get_tools_for_agent = None
    
    def _get_client(self):
        """Get or create the Azure AI Agents client (SYNC version).
        
        Note: As of azure-ai-projects 2.0.0, agent functionality moved to azure-ai-agents package.
        """
        if self._client is not None:
            return self._client
        
        if not self.project_endpoint:
            raise RuntimeError("AZURE_AI_PROJECT_ENDPOINT not configured")
        
        # Use SYNC client from azure-ai-agents (moved from azure-ai-projects in SDK 2.0)
        from azure.ai.agents import AgentsClient
        from azure.identity import DefaultAzureCredential
        
        self._credential = DefaultAzureCredential()
        self._client = AgentsClient(
            endpoint=self.project_endpoint,
            credential=self._credential,
        )
        
        self.logger.info("Initialized AgentsClient for endpoint: %s", self.project_endpoint)
        return self._client
    
    def _find_agent_by_name(self, agent_name: str) -> Optional[str]:
        """Find an agent's Foundry ID by name."""
        if agent_name in self._agent_cache:
            return self._agent_cache[agent_name]
        
        client = self._get_client()
        
        # Search for the agent (sync iteration)
        # Note: azure-ai-agents 1.x uses direct methods on AgentsClient
        for agent in client.list_agents():
            if agent.name == agent_name:
                self._agent_cache[agent_name] = agent.id
                return agent.id
        
        return None
    
    async def invoke_agent(
        self,
        agent_name: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentInvocationResult:
        """
        Invoke a deployed Azure AI Foundry agent with full function calling support.
        
        Implements the Azure AI Foundry function calling pattern:
        1. Create thread and add message
        2. Start run with agent
        3. Poll for status with exponential backoff
        4. If requires_action: execute tools locally and submit outputs
        5. Continue polling until completed or failed
        6. Retrieve response messages
        
        Args:
            agent_name: The name of the agent in Foundry (e.g., "health_data_analysis")
            prompt: The prompt/instructions for the agent
            context: Optional context data to include
            
        Returns:
            AgentInvocationResult with the agent's response and tool execution details
        """
        start_time = datetime.now(timezone.utc)
        tools_executed = []
        
        self.logger.info("FOUNDRY AGENT INVOCATION START: %s", agent_name)
        
        try:
            client = self._get_client()
            
            # Validate client
            if not client or not hasattr(client, 'threads'):
                raise RuntimeError("Invalid AgentsClient - threads attribute missing")
            
            # Find the agent (sync)
            agent_id = self._find_agent_by_name(agent_name)
            if not agent_id:
                self.logger.error("FOUNDRY AGENT INVOCATION FAILURE: Agent '%s' not found", agent_name)
                return AgentInvocationResult(
                    agent_id=agent_name,
                    foundry_agent_id="",
                    success=False,
                    error=f"Agent '{agent_name}' not found in Azure AI Foundry",
                    execution_time_ms=self._elapsed_ms(start_time),
                    tools_executed=[],
                )
            
            self.logger.info("Found agent %s with ID: %s", agent_name, agent_id)
            
            # Build the full prompt with context
            full_prompt = prompt
            if context:
                full_prompt = f"{prompt}\n\nContext:\n```json\n{json.dumps(context, indent=2, cls=DateTimeEncoder)}\n```"
            
            self.logger.info("Prompt length: %d chars", len(full_prompt))
            
            # Step 1: Create thread
            self.logger.info("Creating thread...")
            thread = client.threads.create()
            self.logger.info("Created thread: %s", thread.id)
            
            # Step 2: Add message to thread
            self.logger.info("Creating message...")
            client.messages.create(
                thread_id=thread.id,
                role="user",
                content=full_prompt,
            )
            
            # Step 3: Start run (use create() not create_and_process() to handle tools)
            self.logger.info("Starting run for agent %s on thread %s...", agent_id, thread.id)
            run = client.runs.create(
                thread_id=thread.id,
                agent_id=agent_id,
            )
            self.logger.info("Created run: %s with status: %s", run.id, run.status)
            
            # Step 4: Poll with exponential backoff, handle requires_action
            poll_interval = self.POLL_INTERVAL_SECONDS
            total_wait_time = 0
            
            while run.status in ["queued", "in_progress", "requires_action"]:
                # Check timeout
                if total_wait_time > self.MAX_RUN_TIME_SECONDS:
                    self.logger.error("Run timed out after %d seconds", total_wait_time)
                    break
                
                time.sleep(poll_interval)
                total_wait_time += poll_interval
                
                # Get updated run status
                run = client.runs.get(thread_id=thread.id, run_id=run.id)
                self.logger.debug("Run status: %s (waited %.1fs)", run.status, total_wait_time)
                
                # Handle requires_action - agent wants to call function tools
                if run.status == "requires_action":
                    self.logger.info("Run requires action - processing tool calls")
                    
                    # Execute tools and collect outputs
                    tool_outputs = self._handle_tool_calls(run, tools_executed)
                    
                    if tool_outputs:
                        # Submit tool outputs back to the run
                        self.logger.info("Submitting %d tool outputs", len(tool_outputs))
                        run = client.runs.submit_tool_outputs(
                            thread_id=thread.id,
                            run_id=run.id,
                            tool_outputs=tool_outputs,
                        )
                        self.logger.info("Tool outputs submitted, run status: %s", run.status)
                        # Reset poll interval after action
                        poll_interval = self.POLL_INTERVAL_SECONDS
                    else:
                        self.logger.warning("No tool outputs to submit - this may cause the run to fail")
                        break
                else:
                    # Exponential backoff for normal polling
                    poll_interval = min(poll_interval * 1.5, self.MAX_POLL_INTERVAL_SECONDS)
            
            self.logger.info("Run completed with status: %s", run.status)
            
            if run.status != "completed":
                error_msg = f"Agent run failed with status: {run.status}"
                if hasattr(run, 'last_error') and run.last_error:
                    error_msg += f" - {run.last_error}"
                self.logger.error("FOUNDRY AGENT INVOCATION FAILURE: %s", error_msg)
                
                # Cleanup thread
                self._cleanup_thread(client, thread.id)
                
                return AgentInvocationResult(
                    agent_id=agent_name,
                    foundry_agent_id=agent_id,
                    success=False,
                    error=error_msg,
                    execution_time_ms=self._elapsed_ms(start_time),
                    tools_executed=tools_executed,
                )
            
            # Step 5: Get response messages
            messages = client.messages.list(thread_id=thread.id)
            
            response_text = ""
            for msg in messages:
                if msg.role == "assistant":
                    # Get text from message content
                    if hasattr(msg, 'text_messages') and msg.text_messages:
                        response_text = msg.text_messages[-1].text.value
                        break
                    elif hasattr(msg, 'content'):
                        for content in msg.content:
                            if hasattr(content, "text") and hasattr(content.text, "value"):
                                response_text = content.text.value
                                break
                        if response_text:
                            break
            
            if not response_text:
                self.logger.warning("No response text found from agent %s", agent_name)
            else:
                self.logger.info("Got response from agent %s: %d chars", agent_name, len(response_text))
            
            # Try to parse as JSON if possible
            parsed_output = None
            try:
                # Try to extract JSON from the response
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_str = response_text[json_start:json_end].strip()
                    parsed_output = json.loads(json_str)
                elif response_text.strip().startswith("{"):
                    parsed_output = json.loads(response_text)
            except json.JSONDecodeError:
                pass
            
            # Clean up thread
            self._cleanup_thread(client, thread.id)
            
            execution_time = self._elapsed_ms(start_time)
            
            # Validate execution time - if < 100ms and no tools, may be cached
            if execution_time < 100 and not tools_executed:
                self.logger.warning(
                    "FOUNDRY AGENT INVOCATION WARNING: Agent %s completed in %.2fms - suspiciously fast",
                    agent_name, execution_time
                )
            
            self.logger.info(
                "FOUNDRY AGENT INVOCATION SUCCESS: Agent %s completed in %.2fms with %d tool calls",
                agent_name, execution_time, len(tools_executed)
            )
            
            # Extract token usage safely
            token_usage = self._extract_token_usage(run)
            
            return AgentInvocationResult(
                agent_id=agent_name,
                foundry_agent_id=agent_id,
                success=True,
                response=response_text,
                parsed_output=parsed_output,
                execution_time_ms=execution_time,
                token_usage=token_usage,
                tools_executed=tools_executed,
            )
            
        except Exception as e:
            self.logger.error("FOUNDRY AGENT INVOCATION FAILURE: %s - %s", agent_name, e)
            import traceback
            self.logger.error("Traceback: %s", traceback.format_exc())
            return AgentInvocationResult(
                agent_id=agent_name,
                foundry_agent_id="",
                success=False,
                error=str(e),
                execution_time_ms=self._elapsed_ms(start_time),
                tools_executed=tools_executed,
            )
    
    def _handle_tool_calls(self, run, tools_executed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle tool calls from a run that requires_action.
        
        Args:
            run: The run object with required_action
            tools_executed: List to append executed tool info to
            
        Returns:
            List of tool outputs to submit back to the run
        """
        tool_outputs = []
        
        if not hasattr(run, 'required_action') or not run.required_action:
            self.logger.warning("Run has no required_action")
            return tool_outputs
        
        # Get tool calls from the run
        submit_tool_outputs = getattr(run.required_action, 'submit_tool_outputs', None)
        if not submit_tool_outputs:
            self.logger.warning("No submit_tool_outputs in required_action")
            return tool_outputs
        
        tool_calls = getattr(submit_tool_outputs, 'tool_calls', [])
        if not tool_calls:
            self.logger.warning("No tool_calls found")
            return tool_outputs
        
        for tool_call in tool_calls:
            tool_call_id = tool_call.id
            
            # Check if this is a function tool call
            if hasattr(tool_call, 'function'):
                function_name = tool_call.function.name
                function_args_str = tool_call.function.arguments
                
                self.logger.info("Processing tool call: %s (id: %s)", function_name, tool_call_id)
                
                try:
                    # Parse arguments
                    function_args = json.loads(function_args_str) if function_args_str else {}
                    
                    # Execute the tool
                    if self._execute_tool:
                        tool_result = self._execute_tool(function_name, function_args)
                    else:
                        tool_result = json.dumps({"error": "Tool execution not available"})
                    
                    # Track execution
                    tools_executed.append({
                        "tool_name": function_name,
                        "tool_call_id": tool_call_id,
                        "arguments": function_args,
                        "result_preview": tool_result[:200] if len(tool_result) > 200 else tool_result,
                        "executed_at": datetime.now(timezone.utc).isoformat(),
                    })
                    
                    # Add to outputs
                    tool_outputs.append({
                        "tool_call_id": tool_call_id,
                        "output": tool_result,
                    })
                    
                    self.logger.info("Tool %s executed successfully", function_name)
                    
                except Exception as e:
                    self.logger.error("Tool %s execution failed: %s", function_name, e)
                    # Return error as tool output so agent can handle it
                    error_output = json.dumps({"error": str(e), "tool": function_name})
                    tool_outputs.append({
                        "tool_call_id": tool_call_id,
                        "output": error_output,
                    })
                    tools_executed.append({
                        "tool_name": function_name,
                        "tool_call_id": tool_call_id,
                        "error": str(e),
                        "executed_at": datetime.now(timezone.utc).isoformat(),
                    })
            else:
                self.logger.warning("Unknown tool call type: %s", type(tool_call))
        
        return tool_outputs
    
    def _cleanup_thread(self, client, thread_id: str):
        """Clean up a thread after use."""
        try:
            client.threads.delete(thread_id=thread_id)
        except Exception as del_err:
            self.logger.warning("Failed to delete thread %s: %s", thread_id, del_err)
    
    def _extract_token_usage(self, run) -> Optional[Dict[str, int]]:
        """Extract token usage from a run."""
        if not hasattr(run, "usage") or not run.usage:
            return None
        
        try:
            if hasattr(run.usage, "model_dump"):
                return run.usage.model_dump()
            elif hasattr(run.usage, "as_dict"):
                return run.usage.as_dict()
            elif hasattr(run.usage, "__dict__"):
                return {
                    "prompt_tokens": getattr(run.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(run.usage, "completion_tokens", 0),
                    "total_tokens": getattr(run.usage, "total_tokens", 0),
                }
        except Exception as usage_err:
            self.logger.warning("Could not extract token usage: %s", usage_err)
        
        return None
    
    def _elapsed_ms(self, start_time: datetime) -> float:
        """Calculate elapsed milliseconds since start_time."""
        return (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    
    def close(self):
        """Close the client connection (sync)."""
        if self._client:
            self._client.close()
            self._client = None
        self._credential = None


# Singleton instance
_invoker: Optional[FoundryAgentInvoker] = None


def get_foundry_invoker() -> FoundryAgentInvoker:
    """Get the singleton FoundryAgentInvoker instance."""
    global _invoker
    if _invoker is None:
        _invoker = FoundryAgentInvoker()
    return _invoker