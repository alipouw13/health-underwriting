"""
Azure AI Foundry Service for Agent Deployment

This service handles:
1. Checking if underwriting agents are deployed in Azure AI Foundry
2. Deploying agents if they don't exist
3. Managing agent lifecycle

Follows patterns from Microsoft Multi-Agent Custom Automation Engine Solution Accelerator.
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentDeploymentStatus:
    """Status of an agent deployment in Azure AI Foundry."""
    agent_id: str
    deployed: bool
    foundry_id: Optional[str] = None
    error: Optional[str] = None


class FoundryService:
    """
    Service for managing agents in Azure AI Foundry.
    
    This service:
    1. Loads agent definitions from YAML
    2. Checks if agents are deployed in Azure AI Foundry
    3. Creates/deploys agents that don't exist
    4. Returns agent references for invocation
    """
    
    # Map local agent IDs to Foundry-compatible names
    # SIMPLIFIED 3-AGENT WORKFLOW (MVP)
    AGENT_NAME_MAP = {
        "HealthDataAnalysisAgent": "health_data_analysis",
        "BusinessRulesValidationAgent": "business_rules_validation",
        "CommunicationAgent": "communication",
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._client = None
        self._agents_yaml_path = Path(__file__).parent.parent.parent / ".github" / "underwriting_agents_v2.yaml"
        self._agent_definitions: Dict[str, Dict[str, Any]] = {}
        self._deployed_agents: Dict[str, str] = {}  # agent_id -> foundry_id
        
        # Azure AI Foundry configuration
        self.project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
        self.subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
        self.project_name = os.environ.get("AZURE_AI_PROJECT_NAME")
        
        # Load agent definitions
        self._load_agent_definitions()
    
    def _load_agent_definitions(self) -> None:
        """Load agent definitions from YAML file."""
        if not self._agents_yaml_path.exists():
            self.logger.warning(f"Agent definitions not found: {self._agents_yaml_path}")
            return
        
        try:
            with open(self._agents_yaml_path, "r") as f:
                data = yaml.safe_load(f)
            
            for agent_def in data.get("agents", []):
                agent_id = agent_def.get("agent_id")
                if agent_id:
                    self._agent_definitions[agent_id] = agent_def
            
            self.logger.info(f"Loaded {len(self._agent_definitions)} agent definitions")
        except Exception as e:
            self.logger.error(f"Failed to load agent definitions: {e}")
    
    async def get_client(self):
        """Get or create the Azure AI Agents client.
        
        Note: As of azure-ai-projects 2.0.0, agent functionality moved to azure-ai-agents package.
        """
        if self._client is not None:
            return self._client
        
        if not self.project_endpoint:
            self.logger.warning("AZURE_AI_PROJECT_ENDPOINT not set - Foundry integration disabled")
            return None
        
        try:
            # Agent APIs moved from azure.ai.projects to azure.ai.agents in SDK 2.0
            from azure.ai.agents.aio import AgentsClient
            from azure.identity.aio import DefaultAzureCredential
            
            credential = DefaultAzureCredential()
            self._client = AgentsClient(
                endpoint=self.project_endpoint,
                credential=credential,
            )
            self.logger.info(f"Connected to Azure AI Foundry Agents at {self.project_endpoint}")
            return self._client
        except ImportError:
            self.logger.warning("azure-ai-agents not installed - Foundry integration disabled")
            return None
        except Exception as e:
            self.logger.error(f"Failed to create Foundry Agents client: {e}")
            return None
    
    async def check_agents_deployed(self) -> Dict[str, AgentDeploymentStatus]:
        """
        Check which agents are deployed in Azure AI Foundry.
        
        Returns a dict mapping agent_id to deployment status.
        """
        status: Dict[str, AgentDeploymentStatus] = {}
        
        client = await self.get_client()
        if client is None:
            # No Foundry connection - all agents marked as not deployed
            for agent_id in self._agent_definitions:
                status[agent_id] = AgentDeploymentStatus(
                    agent_id=agent_id,
                    deployed=False,
                    error="Azure AI Foundry not configured"
                )
            return status
        
        try:
            # List existing agents in Foundry
            # Note: azure-ai-agents 1.x uses direct methods on AgentsClient
            existing_agents = {}
            async for agent in client.list_agents():
                existing_agents[agent.name] = agent.id
            
            self.logger.info(f"Found {len(existing_agents)} agents in Azure AI Foundry")
            
            # Check each required agent
            for agent_id, agent_def in self._agent_definitions.items():
                foundry_name = self.AGENT_NAME_MAP.get(agent_id, agent_id.lower())
                
                if foundry_name in existing_agents:
                    foundry_id = existing_agents[foundry_name]
                    self._deployed_agents[agent_id] = foundry_id
                    status[agent_id] = AgentDeploymentStatus(
                        agent_id=agent_id,
                        deployed=True,
                        foundry_id=foundry_id,
                    )
                    self.logger.info(f"Agent {agent_id} found in Foundry as {foundry_name}")
                else:
                    status[agent_id] = AgentDeploymentStatus(
                        agent_id=agent_id,
                        deployed=False,
                    )
                    self.logger.info(f"Agent {agent_id} NOT found in Foundry")
                    
        except Exception as e:
            self.logger.error(f"Failed to check agent deployments: {e}")
            for agent_id in self._agent_definitions:
                status[agent_id] = AgentDeploymentStatus(
                    agent_id=agent_id,
                    deployed=False,
                    error=str(e)
                )
        
        return status
    
    async def deploy_agent(self, agent_id: str) -> AgentDeploymentStatus:
        """
        Deploy a single agent to Azure AI Foundry.
        
        Args:
            agent_id: The agent ID from the YAML definition
            
        Returns:
            AgentDeploymentStatus with deployment result
        """
        if agent_id not in self._agent_definitions:
            return AgentDeploymentStatus(
                agent_id=agent_id,
                deployed=False,
                error=f"Unknown agent: {agent_id}"
            )
        
        client = await self.get_client()
        if client is None:
            return AgentDeploymentStatus(
                agent_id=agent_id,
                deployed=False,
                error="Azure AI Foundry not configured"
            )
        
        agent_def = self._agent_definitions[agent_id]
        foundry_name = self.AGENT_NAME_MAP.get(agent_id, agent_id.lower())
        
        try:
            # Get the model deployment name from environment
            model_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
            
            # Create agent instructions from the YAML definition
            instructions = self._build_agent_instructions(agent_def)
            
            # Create the agent in Foundry
            # Note: azure-ai-agents 1.x uses direct methods on AgentsClient
            agent = await client.create_agent(
                model=model_deployment,
                name=foundry_name,
                instructions=instructions,
            )
            
            self._deployed_agents[agent_id] = agent.id
            self.logger.info(f"Deployed agent {agent_id} to Foundry as {foundry_name} (id={agent.id})")
            
            return AgentDeploymentStatus(
                agent_id=agent_id,
                deployed=True,
                foundry_id=agent.id,
            )
            
        except Exception as e:
            self.logger.error(f"Failed to deploy agent {agent_id}: {e}")
            return AgentDeploymentStatus(
                agent_id=agent_id,
                deployed=False,
                error=str(e)
            )
    
    async def deploy_all_agents(self) -> Dict[str, AgentDeploymentStatus]:
        """
        Deploy all agents that are not already in Azure AI Foundry.
        
        Returns deployment status for all agents.
        """
        # First check what's already deployed
        status = await self.check_agents_deployed()
        
        # Deploy missing agents
        for agent_id, agent_status in status.items():
            if not agent_status.deployed and agent_status.error != "Azure AI Foundry not configured":
                deploy_result = await self.deploy_agent(agent_id)
                status[agent_id] = deploy_result
        
        return status
    
    def _build_agent_instructions(self, agent_def: Dict[str, Any]) -> str:
        """Build agent instructions from YAML definition."""
        parts = [
            f"You are the {agent_def.get('agent_id', 'Agent')}.",
            f"Purpose: {agent_def.get('purpose', 'Perform assigned task')}",
            "",
            "## Inputs",
        ]
        
        inputs = agent_def.get("inputs", {})
        for key, value_type in inputs.items():
            parts.append(f"- {key}: {value_type}")
        
        parts.append("")
        parts.append("## Outputs")
        
        outputs = agent_def.get("outputs", {})
        for key, value_type in outputs.items():
            parts.append(f"- {key}: {value_type}")
        
        parts.append("")
        parts.append("## Evaluation Criteria")
        for criterion in agent_def.get("evaluation_criteria", []):
            parts.append(f"- {criterion}")
        
        parts.append("")
        parts.append("## Failure Modes to Handle")
        for mode in agent_def.get("failure_modes", []):
            parts.append(f"- {mode}")
        
        return "\n".join(parts)
    
    def get_deployed_agent_id(self, agent_id: str) -> Optional[str]:
        """Get the Foundry agent ID for a local agent ID."""
        return self._deployed_agents.get(agent_id)
    
    def is_foundry_enabled(self) -> bool:
        """Check if Azure AI Foundry integration is enabled."""
        return self.project_endpoint is not None


# Singleton instance
_foundry_service: Optional[FoundryService] = None


def get_foundry_service() -> FoundryService:
    """Get the singleton FoundryService instance."""
    global _foundry_service
    if _foundry_service is None:
        _foundry_service = FoundryService()
    return _foundry_service


async def ensure_agents_deployed() -> Dict[str, AgentDeploymentStatus]:
    """
    Ensure all underwriting agents are deployed to Azure AI Foundry.
    
    This should be called during application startup.
    
    Returns deployment status for all agents.
    """
    service = get_foundry_service()
    return await service.deploy_all_agents()
