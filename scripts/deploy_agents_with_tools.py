"""
Deploy Agents with Function Tools to Azure AI Foundry

This script deploys agents with function tool definitions to Azure AI Foundry.
Agents need to be created WITH their tool definitions - they can't be added later.

Usage:
    python scripts/deploy_agents_with_tools.py [--delete-existing]
    
Options:
    --delete-existing    Delete existing agents before creating new ones

Requirements:
    - AZURE_AI_PROJECT_ENDPOINT environment variable
    - AZURE_OPENAI_DEPLOYMENT_NAME environment variable  
    - Azure credentials (DefaultAzureCredential)
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.agent_tools import (
    AGENT_TOOLS,
    TOOL_DEFINITIONS,
    get_tool_schemas_for_agent,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

AGENT_CONFIGS = {
    "health_data_analysis": {
        "name": "health_data_analysis",
        "instructions": """You are a Health Data Analysis Agent specializing in medical underwriting.

Your role is to analyze health information extracted from documents and provide a structured assessment.

You have access to the following tools:
- analyze_health_metrics: Analyzes BMI, blood pressure, cholesterol, glucose levels
- extract_risk_indicators: Categorizes medical conditions, medications, and risk factors

IMPORTANT: Use your tools to perform analysis. Do NOT make up numbers or assessments.
Always call the appropriate tool for each analysis task.

When analyzing an applicant:
1. First call analyze_health_metrics with the biometric data (age, height, weight, etc.)
2. Then call extract_risk_indicators with conditions, medications, and family history
3. Synthesize the tool results into your final assessment

Output your final assessment as JSON with:
- health_score (0-100)
- risk_factors (array of identified risks)
- bmi_assessment (from analyze_health_metrics)
- condition_summary (from extract_risk_indicators)
- recommendations (array of recommendations)
""",
        "tools": ["analyze_health_metrics", "extract_risk_indicators"],
    },
    
    "policy_risk_analysis": {
        "name": "policy_risk_analysis",
        "instructions": """You are a Policy Risk Analysis Agent specializing in insurance underwriting rules.

Your role is to evaluate applicants against underwriting policy rules and calculate risk scores.

You have access to the following tools:
- evaluate_policy_rules: Checks eligibility based on age, coverage amount, health risk, conditions
- lookup_underwriting_guidelines: Retrieves official guidelines for specific conditions
- calculate_risk_score: Computes numerical risk score and premium multiplier

IMPORTANT: Use your tools for all rule evaluations. Do NOT guess at policy rules.
Always call the appropriate tool for each evaluation task.

When evaluating an applicant:
1. First call evaluate_policy_rules with the applicant's details
2. For any concerning conditions, call lookup_underwriting_guidelines to get official guidance
3. Call calculate_risk_score to determine the final risk classification
4. Synthesize the tool results into your decision

Output your decision as JSON with:
- decision (approve/decline/review)
- risk_class (from calculate_risk_score)
- premium_multiplier (from calculate_risk_score)
- rule_evaluations (from evaluate_policy_rules)
- guidelines_referenced (from lookup_underwriting_guidelines)
- conditions (any conditions on approval)
- exclusions (any exclusions)
""",
        "tools": ["evaluate_policy_rules", "lookup_underwriting_guidelines", "calculate_risk_score"],
    },
    
    "business_rules_validation": {
        "name": "business_rules_validation",
        "instructions": """You are a Business Rules Validation Agent ensuring compliance with company policies.

Your role is to validate that the proposed coverage meets all business requirements.

You have access to the following tools:
- validate_coverage_eligibility: Checks state availability, coverage limits, employment status
- evaluate_policy_rules: Validates against underwriting policy rules

IMPORTANT: Use your tools to validate all business rules. Do NOT assume compliance.
Always call the appropriate tool for each validation.

When validating an application:
1. Call validate_coverage_eligibility with the applicant and coverage details
2. Call evaluate_policy_rules to confirm underwriting requirements
3. Synthesize the results into a validation report

Output your validation as JSON with:
- validated (true/false)
- eligibility_checks (from validate_coverage_eligibility)
- policy_rule_checks (from evaluate_policy_rules)
- issues (array of any issues found)
- required_actions (what needs to be done to resolve issues)
""",
        "tools": ["validate_coverage_eligibility", "evaluate_policy_rules"],
    },
    
    "communication": {
        "name": "communication",
        "instructions": """You are a Communication Agent responsible for generating clear, professional communications.

Your role is to create decision summaries and correspondence for applicants.

You have access to the following tools:
- generate_decision_summary: Creates formatted decision letters with all required information

IMPORTANT: Use the generate_decision_summary tool to ensure consistent, professional output.
Do NOT write decision letters from scratch - use the tool.

When creating communications:
1. Gather all decision information (decision, risk class, coverage, conditions, exclusions)
2. Call generate_decision_summary with all the details
3. Review and enhance the generated summary if needed

Output your communication as JSON with:
- summary_text (from generate_decision_summary)
- applicant_name
- decision
- formatted_date
- next_steps (array of action items for the applicant)
""",
        "tools": ["generate_decision_summary"],
    },
}


def get_project_client():
    """Create Azure AI Projects client."""
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential
    
    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    if not endpoint:
        raise ValueError("AZURE_AI_PROJECT_ENDPOINT environment variable not set")
    
    credential = DefaultAzureCredential()
    client = AIProjectClient(endpoint=endpoint, credential=credential)
    
    logger.info("Connected to Azure AI Foundry: %s", endpoint)
    return client


def list_existing_agents(client) -> Dict[str, str]:
    """List all existing agents and return name->id mapping."""
    agents = {}
    for agent in client.agents.list_agents():
        agents[agent.name] = agent.id
        logger.info("Found existing agent: %s (ID: %s)", agent.name, agent.id)
    return agents


def delete_agent(client, agent_id: str, agent_name: str):
    """Delete an agent by ID."""
    try:
        client.agents.delete_agent(agent_id)
        logger.info("Deleted agent: %s (ID: %s)", agent_name, agent_id)
    except Exception as e:
        logger.warning("Failed to delete agent %s: %s", agent_name, e)


def create_agent_with_tools(
    client,
    name: str,
    instructions: str,
    tool_names: List[str],
    model: str,
) -> str:
    """
    Create an agent with function tools in Azure AI Foundry.
    
    Args:
        client: AIProjectClient
        name: Agent name
        instructions: Agent instructions
        tool_names: List of tool names from TOOL_DEFINITIONS
        model: Model deployment name
        
    Returns:
        Agent ID
    """
    # Build tool definitions
    tools = []
    for tool_name in tool_names:
        if tool_name in TOOL_DEFINITIONS:
            tool_def = TOOL_DEFINITIONS[tool_name]
            tools.append(tool_def.to_foundry_schema())
            logger.info("  Adding tool: %s", tool_name)
        else:
            logger.warning("  Unknown tool: %s", tool_name)
    
    logger.info("Creating agent '%s' with %d tools...", name, len(tools))
    
    # Create the agent
    agent = client.agents.create_agent(
        model=model,
        name=name,
        instructions=instructions,
        tools=tools,
    )
    
    logger.info("Created agent: %s (ID: %s)", name, agent.id)
    return agent.id


def deploy_agents(delete_existing: bool = False):
    """Deploy all agents with their function tools."""
    
    # Get configuration
    model = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
    logger.info("Using model deployment: %s", model)
    
    # Connect to Foundry
    client = get_project_client()
    
    # List existing agents
    existing_agents = list_existing_agents(client)
    
    # Track deployment results
    deployed = []
    failed = []
    
    for agent_name, config in AGENT_CONFIGS.items():
        try:
            # Check if agent exists
            if agent_name in existing_agents:
                if delete_existing:
                    logger.info("Deleting existing agent: %s", agent_name)
                    delete_agent(client, existing_agents[agent_name], agent_name)
                else:
                    logger.info("Agent '%s' already exists (ID: %s) - skipping", 
                               agent_name, existing_agents[agent_name])
                    deployed.append((agent_name, existing_agents[agent_name], "existing"))
                    continue
            
            # Create agent with tools
            agent_id = create_agent_with_tools(
                client=client,
                name=config["name"],
                instructions=config["instructions"],
                tool_names=config["tools"],
                model=model,
            )
            deployed.append((agent_name, agent_id, "created"))
            
        except Exception as e:
            logger.error("Failed to deploy agent '%s': %s", agent_name, e)
            failed.append((agent_name, str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("DEPLOYMENT SUMMARY")
    print("="*60)
    
    print(f"\nSuccessfully deployed: {len(deployed)}")
    for name, agent_id, status in deployed:
        print(f"  ✓ {name} ({status}): {agent_id}")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for name, error in failed:
            print(f"  ✗ {name}: {error}")
    
    print("\n" + "="*60)
    
    # Close client
    client.close()
    
    return len(failed) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Deploy agents with function tools to Azure AI Foundry"
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing agents before creating new ones"
    )
    
    args = parser.parse_args()
    
    # Load environment from .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
        logger.info("Loaded environment from .env")
    except ImportError:
        logger.info("python-dotenv not installed, using system environment")
    
    # Validate environment
    required_vars = ["AZURE_AI_PROJECT_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT_NAME"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        logger.error("Missing required environment variables: %s", missing)
        sys.exit(1)
    
    # Deploy
    success = deploy_agents(delete_existing=args.delete_existing)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
