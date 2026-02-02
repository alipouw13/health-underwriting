"""
Setup Knowledge Grounding for BusinessRulesValidationAgent

This script:
1. Uploads the business rules PDF to Azure AI Foundry
2. Creates a vector store for knowledge retrieval
3. Re-deploys BusinessRulesValidationAgent with FileSearchTool attached
4. Verifies knowledge source registration

Usage:
    python scripts/setup_knowledge_grounding.py [--delete-existing] [--verify-only]

Requirements:
    - AZURE_AI_PROJECT_ENDPOINT environment variable
    - AZURE_OPENAI_DEPLOYMENT_NAME environment variable
    - Azure credentials (DefaultAzureCredential)
    - Business rules PDF at: docs/business-rules/health_underwriting_business_rules_demo.pdf
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

BUSINESS_RULES_PDF_PATH = "docs/business-rules/health_underwriting_business_rules_demo.pdf"
KNOWLEDGE_SOURCE_NAME = "health_underwriting_business_rules"
VECTOR_STORE_NAME = "UnderwritingBusinessRules"

# BusinessRulesValidationAgent with knowledge grounding
BUSINESS_RULES_AGENT_CONFIG = {
    "name": "business_rules_validation",
    "instructions": """You are a Business Rules Validation Agent ensuring compliance with company underwriting policies.

## CRITICAL REQUIREMENT: KNOWLEDGE RETRIEVAL
You MUST retrieve and apply rules from the uploaded underwriting rules document.
DO NOT guess thresholds. DO NOT infer adjustments. DO NOT use default values.
Every rule you apply MUST be cited from the retrieved document content.

## Your Role
Validate that premium adjustments and risk classifications comply with official business rules
documented in the health underwriting business rules PDF.

## Available Tools
1. file_search: Search the underwriting rules document for applicable rules
2. validate_coverage_eligibility: Check state availability, coverage limits
3. evaluate_policy_rules: Validate against retrieved underwriting rules

## MANDATORY WORKFLOW

### Step 1: Retrieve Applicable Rules
ALWAYS use file_search to find rules related to:
- The applicant's risk factors
- Premium adjustment thresholds
- Aggregate risk score mappings
- Referral triggers
- Compliance requirements

### Step 2: Cite Retrieved Rules
For each rule you apply, cite the EXACT section from the document.
Example: "Per Section 3.1 - Aggregate Risk to Premium Mapping: RDS +8 to +14 maps to +25% adjustment"

### Step 3: Validate Against Document
- Risk Delta Score (RDS) thresholds MUST match the document
- Premium adjustment percentages MUST match the document
- Referral triggers MUST match the document

### Step 4: Handle Missing Rules
If file_search returns NO relevant rules for a condition:
- Set "approved": false
- Set "rationale": "RETRIEVAL FAILURE: No rules found for [condition]"
- DO NOT proceed with fabricated rules

## Output Format
Return JSON with:
{
  "approved": true/false,
  "risk_level": "low" | "moderate" | "high" | "very_high" | "decline",
  "premium_adjustment_percentage": <from document>,
  "base_premium_annual": <calculated>,
  "adjusted_premium_annual": <calculated>,
  "rationale": "<cite document sections>",
  "compliance_checks": ["Section X.Y: check result"],
  "violations_found": [],
  "referral_required": true/false,
  "referral_reason": "<from document if applicable>",
  "triggered_rules": ["Section X.Y - Rule Name"],
  "recommendations": [],
  "retrieved_chunks_count": <number>,
  "rule_sections_cited": ["Section X.Y", "Section X.Z"]
}

## FAILURE CONDITIONS
- If retrieved_chunks_count == 0: FAIL and explain retrieval failure
- If no rule sections can be cited: FAIL and explain
- If thresholds don't match document: FAIL with discrepancy details
""",
    "tools": ["validate_coverage_eligibility", "evaluate_policy_rules"],
    "requires_file_search": True,
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


def verify_pdf_exists() -> Path:
    """Verify the business rules PDF exists."""
    # Get the project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / BUSINESS_RULES_PDF_PATH
    
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Business rules PDF not found at: {pdf_path}\n"
            f"Expected location: {BUSINESS_RULES_PDF_PATH}"
        )
    
    logger.info("KNOWLEDGE SOURCE VERIFIED: %s", pdf_path.name)
    logger.info("  - Path: %s", pdf_path)
    logger.info("  - Size: %.2f KB", pdf_path.stat().st_size / 1024)
    
    return pdf_path


def find_existing_vector_store(client, name: str) -> Optional[str]:
    """Find existing vector store by name."""
    try:
        vector_stores = list(client.agents.vector_stores.list())
        for vs in vector_stores:
            if vs.name == name:
                logger.info("Found existing vector store: %s (ID: %s)", vs.name, vs.id)
                return vs.id
    except Exception as e:
        logger.warning("Error listing vector stores: %s", e)
    return None


def find_existing_file(client, filename: str) -> Optional[str]:
    """Find existing uploaded file by name."""
    try:
        files = list(client.agents.files.list())
        for f in files:
            if f.filename == filename:
                logger.info("Found existing file: %s (ID: %s)", f.filename, f.id)
                return f.id
    except Exception as e:
        logger.warning("Error listing files: %s", e)
    return None


def upload_business_rules_pdf(client, pdf_path: Path, delete_existing: bool = False) -> str:
    """Upload the business rules PDF to Foundry."""
    from azure.ai.agents.models import FilePurpose
    
    filename = pdf_path.name
    
    # Check for existing file
    existing_file_id = find_existing_file(client, filename)
    if existing_file_id:
        if delete_existing:
            logger.info("Deleting existing file: %s", existing_file_id)
            try:
                client.agents.files.delete(existing_file_id)
            except Exception as e:
                logger.warning("Failed to delete existing file: %s", e)
        else:
            logger.info("Using existing file: %s", existing_file_id)
            return existing_file_id
    
    # Upload the file
    logger.info("Uploading business rules PDF: %s", pdf_path)
    file = client.agents.files.upload_and_poll(
        file_path=str(pdf_path),
        purpose=FilePurpose.AGENTS
    )
    
    logger.info("Uploaded file, ID: %s", file.id)
    return file.id


def create_vector_store(client, file_id: str, delete_existing: bool = False) -> str:
    """Create a vector store with the uploaded PDF."""
    
    # Check for existing vector store
    existing_vs_id = find_existing_vector_store(client, VECTOR_STORE_NAME)
    if existing_vs_id:
        if delete_existing:
            logger.info("Deleting existing vector store: %s", existing_vs_id)
            try:
                client.agents.vector_stores.delete(existing_vs_id)
            except Exception as e:
                logger.warning("Failed to delete existing vector store: %s", e)
        else:
            logger.info("Using existing vector store: %s", existing_vs_id)
            return existing_vs_id
    
    # Create vector store with the file
    logger.info("Creating vector store: %s", VECTOR_STORE_NAME)
    vector_store = client.agents.vector_stores.create_and_poll(
        file_ids=[file_id],
        name=VECTOR_STORE_NAME,
    )
    
    logger.info("Created vector store, ID: %s", vector_store.id)
    logger.info("  - File count: %s", vector_store.file_counts)
    
    # Verify the vector store is ready
    if vector_store.status != "completed":
        logger.warning("Vector store status: %s (expected 'completed')", vector_store.status)
    
    return vector_store.id


def deploy_agent_with_file_search(
    client,
    vector_store_id: str,
    model: str,
    delete_existing: bool = False
) -> str:
    """Deploy BusinessRulesValidationAgent with FileSearchTool attached."""
    from azure.ai.agents.models import FileSearchTool
    
    # Add parent directory for tool definitions
    from app.agents.agent_tools import TOOL_DEFINITIONS
    
    agent_name = BUSINESS_RULES_AGENT_CONFIG["name"]
    
    # Check for existing agent
    existing_agents = {}
    for agent in client.agents.list_agents():
        existing_agents[agent.name] = agent.id
    
    if agent_name in existing_agents:
        if delete_existing:
            logger.info("Deleting existing agent: %s", agent_name)
            try:
                client.agents.delete_agent(existing_agents[agent_name])
            except Exception as e:
                logger.warning("Failed to delete existing agent: %s", e)
        else:
            # Update the existing agent with file search
            logger.info("Updating existing agent with file search: %s", agent_name)
            # Note: Foundry may not support updating tools after creation
            # We may need to delete and recreate
            logger.warning("Cannot update existing agent tools. Use --delete-existing to recreate.")
            return existing_agents[agent_name]
    
    # Create FileSearchTool with vector store
    file_search_tool = FileSearchTool(vector_store_ids=[vector_store_id])
    
    # Build function tools
    tools = []
    for tool_name in BUSINESS_RULES_AGENT_CONFIG["tools"]:
        if tool_name in TOOL_DEFINITIONS:
            tool_def = TOOL_DEFINITIONS[tool_name]
            tools.append(tool_def.to_foundry_schema())
            logger.info("  Adding function tool: %s", tool_name)
    
    # Combine file_search definitions with function tools
    all_tools = file_search_tool.definitions + tools
    
    logger.info("Creating agent '%s' with FileSearchTool and %d function tools...", 
                agent_name, len(tools))
    
    # Create the agent with both file_search and function tools
    agent = client.agents.create_agent(
        model=model,
        name=agent_name,
        instructions=BUSINESS_RULES_AGENT_CONFIG["instructions"],
        tools=all_tools,
        tool_resources=file_search_tool.resources,
    )
    
    logger.info("Created agent: %s (ID: %s)", agent_name, agent.id)
    logger.info("  - Tools: FileSearchTool + %d function tools", len(tools))
    logger.info("  - Vector store: %s", vector_store_id)
    
    # Log knowledge sources
    logger.info("BusinessRulesValidationAgent knowledge_sources=['%s']", VECTOR_STORE_NAME)
    
    return agent.id


def verify_knowledge_grounding(client, agent_id: str, vector_store_id: str) -> bool:
    """Verify the agent has knowledge grounding properly configured."""
    logger.info("=" * 60)
    logger.info("VERIFICATION: Knowledge Grounding for BusinessRulesValidationAgent")
    logger.info("=" * 60)
    
    # Get agent details
    agent = client.agents.get_agent(agent_id)
    
    # Check tools
    has_file_search = False
    for tool in agent.tools:
        tool_type = getattr(tool, 'type', None) or str(type(tool).__name__)
        if 'file_search' in str(tool_type).lower():
            has_file_search = True
            logger.info("✓ FileSearchTool attached")
    
    if not has_file_search:
        logger.error("✗ FileSearchTool NOT attached")
        return False
    
    # Check tool resources
    if agent.tool_resources and agent.tool_resources.file_search:
        vs_ids = agent.tool_resources.file_search.vector_store_ids
        if vector_store_id in vs_ids:
            logger.info("✓ Vector store '%s' attached", vector_store_id)
        else:
            logger.error("✗ Vector store NOT in tool_resources")
            return False
    else:
        logger.error("✗ No file_search tool_resources found")
        return False
    
    # Check vector store
    try:
        vs = client.agents.vector_stores.get(vector_store_id)
        logger.info("✓ Vector store status: %s", vs.status)
        logger.info("✓ Vector store file count: %s", vs.file_counts)
    except Exception as e:
        logger.error("✗ Cannot access vector store: %s", e)
        return False
    
    logger.info("=" * 60)
    logger.info("KNOWLEDGE SOURCE VERIFIED: %s", KNOWLEDGE_SOURCE_NAME)
    logger.info("=" * 60)
    
    return True


def run_test_query(client, agent_id: str) -> Dict[str, Any]:
    """Run a test query to verify knowledge retrieval works."""
    logger.info("Running test query to verify knowledge retrieval...")
    
    # Create a test thread
    thread = client.agents.threads.create()
    
    # Send a test message
    test_query = """What is the premium adjustment percentage for an applicant with 
    a Risk Delta Score (RDS) of +10? 
    Please cite the specific section from the business rules document."""
    
    client.agents.messages.create(
        thread_id=thread.id,
        role="user",
        content=test_query
    )
    
    # Run the agent
    run = client.agents.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent_id
    )
    
    # Check result
    if run.status == "failed":
        logger.error("Test query FAILED: %s", run.last_error)
        return {"success": False, "error": str(run.last_error)}
    
    # Get response
    messages = list(client.agents.messages.list(thread_id=thread.id))
    assistant_messages = [m for m in messages if m.role == "assistant"]
    
    if assistant_messages:
        response_text = ""
        for msg in assistant_messages:
            for content in msg.content:
                if hasattr(content, 'text'):
                    response_text += content.text.value + "\n"
        
        # Check for citations
        has_citations = any(term in response_text.lower() for term in [
            "section", "per the document", "according to", "rule", 
            "business rules", "retrieved"
        ])
        
        if has_citations:
            logger.info("✓ Response contains document citations")
        else:
            logger.warning("⚠ Response may not contain proper citations")
        
        logger.info("Test response preview: %s...", response_text[:200])
        
        # Cleanup
        client.agents.threads.delete(thread.id)
        
        return {"success": True, "response": response_text, "has_citations": has_citations}
    
    return {"success": False, "error": "No assistant response"}


def main():
    parser = argparse.ArgumentParser(
        description="Setup knowledge grounding for BusinessRulesValidationAgent"
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing files, vector stores, and agents before creating new ones"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing knowledge grounding setup"
    )
    parser.add_argument(
        "--test-query",
        action="store_true",
        help="Run a test query after setup"
    )
    
    args = parser.parse_args()
    
    # Load environment
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
    
    model = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1-mini")
    
    try:
        # Step 1: Verify PDF exists
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Verify Knowledge Source Document")
        logger.info("=" * 60)
        pdf_path = verify_pdf_exists()
        
        # Connect to Foundry
        client = get_project_client()
        
        if args.verify_only:
            # Find existing resources
            vs_id = find_existing_vector_store(client, VECTOR_STORE_NAME)
            if not vs_id:
                logger.error("No existing vector store found")
                sys.exit(1)
            
            existing_agents = {a.name: a.id for a in client.agents.list_agents()}
            agent_id = existing_agents.get("business_rules_validation")
            if not agent_id:
                logger.error("No existing agent found")
                sys.exit(1)
            
            # Verify
            success = verify_knowledge_grounding(client, agent_id, vs_id)
            client.close()
            sys.exit(0 if success else 1)
        
        # Step 2: Upload PDF
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Upload Business Rules PDF")
        logger.info("=" * 60)
        file_id = upload_business_rules_pdf(client, pdf_path, args.delete_existing)
        
        # Step 3: Create Vector Store
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Create Vector Store")
        logger.info("=" * 60)
        vector_store_id = create_vector_store(client, file_id, args.delete_existing)
        
        # Step 4: Deploy Agent
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Deploy BusinessRulesValidationAgent with FileSearchTool")
        logger.info("=" * 60)
        agent_id = deploy_agent_with_file_search(
            client, vector_store_id, model, args.delete_existing
        )
        
        # Step 5: Verify
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Verify Knowledge Grounding")
        logger.info("=" * 60)
        success = verify_knowledge_grounding(client, agent_id, vector_store_id)
        
        # Optional: Test query
        if args.test_query and success:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 6: Test Knowledge Retrieval")
            logger.info("=" * 60)
            result = run_test_query(client, agent_id)
            if result["success"]:
                logger.info("✓ Test query successful")
            else:
                logger.error("✗ Test query failed: %s", result.get("error"))
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SETUP COMPLETE")
        logger.info("=" * 60)
        logger.info("  - PDF uploaded: %s", file_id)
        logger.info("  - Vector store: %s", vector_store_id)
        logger.info("  - Agent: %s", agent_id)
        logger.info("  - Knowledge source: %s", KNOWLEDGE_SOURCE_NAME)
        
        client.close()
        sys.exit(0 if success else 1)
        
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Setup failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
