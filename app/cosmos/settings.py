"""Cosmos DB settings for agent run persistence."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class CosmosSettings:
    """Configuration for Azure Cosmos DB connection.
    
    Following patterns from reference implementation:
    https://github.com/alipouw13/insurance-multi-agent
    
    Containers:
    - underwriting_agent_runs: Complete agent execution records (partition: /application_id)
    - token_tracking: Token usage telemetry per agent (partition: /execution_id)
    - evaluations: Agent evaluation results (partition: /evaluation_id)
    """
    
    # Connection
    endpoint: Optional[str] = None
    database_name: str = "underwriting-agents"
    
    # Containers
    agent_runs_container: str = "underwriting_agent_runs"
    token_tracking_container: str = "token_tracking"
    evaluations_container: str = "evaluations"
    
    # Partition key paths per container
    # These should match the containers created in Azure Portal (default: /id)
    agent_runs_partition_key: str = "/id"
    token_tracking_partition_key: str = "/id"
    evaluations_partition_key: str = "/id"
    
    # Throughput settings (serverless uses automatic scaling)
    use_serverless: bool = True
    provisioned_throughput: int = 400  # Only used if not serverless
    
    @classmethod
    def from_env(cls) -> "CosmosSettings":
        """Load Cosmos DB settings from environment variables."""
        return cls(
            endpoint=os.getenv("AZURE_COSMOS_ENDPOINT"),
            database_name=os.getenv("AZURE_COSMOS_DATABASE_NAME", "underwriting-agents"),
            agent_runs_container=os.getenv("AZURE_COSMOS_AGENT_RUNS_CONTAINER", "underwriting_agent_runs"),
            token_tracking_container=os.getenv("AZURE_COSMOS_TOKEN_TRACKING_CONTAINER", "token_tracking"),
            evaluations_container=os.getenv("AZURE_COSMOS_EVALUATIONS_CONTAINER", "evaluations"),
            use_serverless=os.getenv("AZURE_COSMOS_USE_SERVERLESS", "true").lower() == "true",
            provisioned_throughput=int(os.getenv("AZURE_COSMOS_THROUGHPUT", "400")),
        )
    
    @property
    def is_configured(self) -> bool:
        """Check if Cosmos DB is configured."""
        return bool(self.endpoint)
