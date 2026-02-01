"""
Azure AI Foundry Agent Invoker

This module provides the ability to invoke deployed agents in Azure AI Foundry,
making real LLM calls instead of running local deterministic logic.

Follows patterns from Microsoft Multi-Agent Custom Automation Engine Solution Accelerator.
"""

import logging
import os
import json
from typing import Any, Dict, Optional
from dataclasses import dataclass
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


class FoundryAgentInvoker:
    """
    Invokes deployed agents in Azure AI Foundry via the Agents API.
    
    Uses the SYNCHRONOUS Azure AI Projects SDK (azure.ai.projects.AIProjectClient).
    
    API Pattern (verified against installed SDK):
    - client.agents.threads.create() - create thread
    - client.agents.messages.create() - add message to thread  
    - client.agents.runs.create_and_process() - run agent and wait
    - client.agents.messages.list() - get response messages
    - client.agents.threads.delete() - cleanup
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._client = None
        self._credential = None
        
        self.project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
        self.model_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
        
        # Cache of agent IDs
        self._agent_cache: Dict[str, str] = {}
    
    def _get_client(self):
        """Get or create the Azure AI Projects client (SYNC version)."""
        if self._client is not None:
            return self._client
        
        if not self.project_endpoint:
            raise RuntimeError("AZURE_AI_PROJECT_ENDPOINT not configured")
        
        # Use SYNC client - the SDK methods are synchronous
        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential
        
        self._credential = DefaultAzureCredential()
        self._client = AIProjectClient(
            endpoint=self.project_endpoint,
            credential=self._credential,
        )
        
        self.logger.info("Initialized AIProjectClient for endpoint: %s", self.project_endpoint)
        return self._client
    
    def _find_agent_by_name(self, agent_name: str) -> Optional[str]:
        """Find an agent's Foundry ID by name."""
        if agent_name in self._agent_cache:
            return self._agent_cache[agent_name]
        
        client = self._get_client()
        
        # Search for the agent (sync iteration)
        for agent in client.agents.list_agents():
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
        Invoke a deployed Azure AI Foundry agent.
        
        Uses the CORRECT Azure AI Projects SDK pattern:
        - client.agents.threads.create() - create thread
        - client.agents.messages.create() - add message to thread
        - client.agents.runs.create_and_process() - run agent and wait
        - client.agents.messages.list() - get response messages
        
        Args:
            agent_name: The name of the agent in Foundry (e.g., "health_data_analysis")
            prompt: The prompt/instructions for the agent
            context: Optional context data to include
            
        Returns:
            AgentInvocationResult with the agent's response
        """
        start_time = datetime.now(timezone.utc)
        
        self.logger.info("FOUNDRY AGENT INVOCATION START: %s", agent_name)
        
        try:
            client = self._get_client()
            
            # Validate client
            if not client or not hasattr(client, 'agents'):
                raise RuntimeError("Invalid AIProjectClient - agents attribute missing")
            
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
                )
            
            self.logger.info("Found agent %s with ID: %s", agent_name, agent_id)
            
            # Build the full prompt with context
            full_prompt = prompt
            if context:
                full_prompt = f"{prompt}\n\nContext:\n```json\n{json.dumps(context, indent=2, cls=DateTimeEncoder)}\n```"
            
            self.logger.info("Prompt length: %d chars", len(full_prompt))
            
            # CORRECT API: Use client.agents.threads.create()
            self.logger.info("Creating thread...")
            thread = client.agents.threads.create()
            self.logger.info("Created thread: %s", thread.id)
            
            # CORRECT API: Use client.agents.messages.create()
            self.logger.info("Creating message...")
            client.agents.messages.create(
                thread_id=thread.id,
                role="user",
                content=full_prompt,
            )
            
            # CORRECT API: Use client.agents.runs.create_and_process() for synchronous completion
            self.logger.info("Running agent %s on thread %s...", agent_id, thread.id)
            run = client.agents.runs.create_and_process(
                thread_id=thread.id,
                agent_id=agent_id,
            )
            
            self.logger.info("Run completed with status: %s", run.status)
            
            if run.status != "completed":
                error_msg = f"Agent run failed with status: {run.status}"
                if hasattr(run, 'last_error') and run.last_error:
                    error_msg += f" - {run.last_error}"
                self.logger.error("FOUNDRY AGENT INVOCATION FAILURE: %s", error_msg)
                return AgentInvocationResult(
                    agent_id=agent_name,
                    foundry_agent_id=agent_id,
                    success=False,
                    error=error_msg,
                    execution_time_ms=self._elapsed_ms(start_time),
                )
            
            # CORRECT API: Use client.agents.messages.list() to get response
            messages = client.agents.messages.list(thread_id=thread.id)
            
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
            try:
                client.agents.threads.delete(thread_id=thread.id)
            except Exception as del_err:
                self.logger.warning("Failed to delete thread %s: %s", thread.id, del_err)
            
            execution_time = self._elapsed_ms(start_time)
            
            # Validate execution time - if < 100ms, likely not a real LLM call
            if execution_time < 100:
                self.logger.warning(
                    "FOUNDRY AGENT INVOCATION WARNING: Agent %s completed in %.2fms - suspiciously fast, may be cached or mock",
                    agent_name, execution_time
                )
            
            self.logger.info(
                "FOUNDRY AGENT INVOCATION SUCCESS: Agent %s completed in %.2fms",
                agent_name, execution_time
            )
            
            # Extract token usage safely
            token_usage = None
            if hasattr(run, "usage") and run.usage:
                try:
                    # Try different approaches to serialize usage
                    if hasattr(run.usage, "model_dump"):
                        token_usage = run.usage.model_dump()
                    elif hasattr(run.usage, "as_dict"):
                        token_usage = run.usage.as_dict()
                    elif hasattr(run.usage, "__dict__"):
                        token_usage = {
                            "prompt_tokens": getattr(run.usage, "prompt_tokens", 0),
                            "completion_tokens": getattr(run.usage, "completion_tokens", 0),
                            "total_tokens": getattr(run.usage, "total_tokens", 0),
                        }
                except Exception as usage_err:
                    self.logger.warning("Could not extract token usage: %s", usage_err)
            
            return AgentInvocationResult(
                agent_id=agent_name,
                foundry_agent_id=agent_id,
                success=True,
                response=response_text,
                parsed_output=parsed_output,
                execution_time_ms=execution_time,
                token_usage=token_usage,
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
            )
    
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