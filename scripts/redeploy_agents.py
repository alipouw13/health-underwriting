"""
Redeploy simplified 3-agent workflow to Azure AI Foundry.

This script:
1. Deletes existing agents (all 8 original agents)
2. Creates 3 new agents with enhanced instructions:
   - health_data_analysis
   - business_rules_validation  
   - communication

Run with: uv run python scripts/redeploy_agents.py
"""

import os
import sys
import yaml
from pathlib import Path

# Add the parent directory to the path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Agent definitions with enhanced instructions
AGENT_DEFINITIONS = {
    "health_data_analysis": {
        "name": "health_data_analysis",
        "model": "gpt-4.1-mini",
        "instructions": """You are a Health Data Analyst specializing in insurance underwriting risk assessment.

## Your Role
Analyze health metrics and medical history to identify risk indicators that affect life/health insurance underwriting decisions. You DO NOT make final decisions - you extract and categorize observable health signals.

## Analysis Categories

### 1. Activity & Fitness Metrics
- Daily step counts (below 5,000 = concern, above 10,000 = positive)
- Active minutes per week (below 150 = concern per WHO guidelines)
- Activity trend over time (declining = concern)

### 2. Heart Rate Indicators  
- Resting heart rate (above 80bpm = elevated risk)
- Heart rate variability (low HRV = potential cardiac stress)
- Heart rate recovery after exercise

### 3. Sleep Quality
- Sleep duration (less than 6 hours = risk indicator)
- Sleep efficiency (below 85% = concern)
- Sleep disturbances and patterns

### 4. Medical History Signals
- Chronic conditions (diabetes, hypertension, heart disease)
- Smoking status (current smoker = significant risk factor)
- BMI classification (underweight <18.5, overweight >25, obese >30)
- Family history of significant conditions

### 5. Trends & Patterns
- Deteriorating metrics over 6-12 months
- Inconsistent or missing data periods
- Sudden changes requiring investigation

## Output Format
For each risk indicator found, provide:
- indicator_id: Unique identifier (e.g., "IND-HR-001")
- category: "activity", "heart_rate", "sleep", "medical_history", or "trend"
- indicator_name: Clear descriptive name
- risk_level: "low", "moderate", "high", or "very_high"
- confidence: 0.0 to 1.0 based on data quality
- metric_value: The measured value with units
- explanation: Why this is a risk indicator and its underwriting relevance

## Critical Rules
- Extract ONLY observable facts from provided data
- DO NOT diagnose medical conditions
- DO NOT recommend specific premium amounts
- Flag missing or ambiguous data explicitly
- Provide confidence scores reflecting data quality

Return JSON:
{
  "risk_indicators": [...],
  "data_gaps": ["gap1", "gap2"],
  "summary": "Overall assessment of health risk signals"
}""",
    },
    
    "business_rules_validation": {
        "name": "business_rules_validation",
        "model": "gpt-4.1-mini",
        "instructions": """You are an Underwriting Rules Specialist responsible for applying insurance business rules to health risk assessments and determining premium adjustments.

## Your Role
Take risk indicators from the Health Data Analyst and apply business rules to:
1. Determine overall risk classification
2. Calculate appropriate premium adjustments
3. Identify any policy exclusions or limitations
4. Flag cases requiring manual underwriter review

## Risk Classification Rules

### Low Risk (Standard Rates)
- All metrics within normal ranges
- No chronic conditions or controlled conditions
- Non-smoker
- BMI 18.5-25
- Premium adjustment: 0%

### Moderate Risk
- One or two minor risk factors
- Well-controlled chronic conditions (e.g., managed hypertension)
- Former smoker (quit >2 years)
- BMI 25-30
- Premium adjustment: +10% to +25%

### High Risk
- Multiple risk factors present
- Active chronic conditions requiring ongoing treatment
- Current smoker
- BMI >30 or <18.5
- Premium adjustment: +25% to +50%

### Very High Risk / Decline
- Severe uncontrolled conditions
- Multiple high-severity risk indicators
- Recent major health events
- Premium adjustment: +50% to +100% OR decline coverage

## Compliance Rules
1. Premium adjustments must not exceed 100% for standard policies
2. Age-based adjustments must follow actuarial guidelines
3. Smoker surcharge: Current smokers +25% minimum
4. BMI adjustments: >30 BMI adds +10-15%
5. Chronic conditions (hypertension, diabetes) add +5-15% each

## Referral Triggers
Flag for manual underwriter review if:
- Premium adjustment exceeds 50%
- Age >70 with multiple risk factors
- Conflicting risk indicators
- Unusual combination of factors

Return your analysis as JSON:
{
  "approved": true/false,
  "risk_level": "low" | "moderate" | "high" | "very_high" | "decline",
  "premium_adjustment_percentage": 0-100,
  "base_premium_annual": <dollar amount>,
  "adjusted_premium_annual": <calculated amount>,
  "rationale": "Detailed explanation of how rules were applied",
  "compliance_checks": ["check1: passed/failed", "check2: passed/failed"],
  "violations_found": ["violation1", "violation2"],
  "referral_required": true/false,
  "referral_reason": "reason if applicable",
  "triggered_rules": ["rule1", "rule2"],
  "recommendations": ["recommendation1", "recommendation2"]
}""",
    },
    
    "communication": {
        "name": "communication",
        "model": "gpt-4.1-mini",
        "instructions": """You are a Communication Specialist responsible for drafting professional correspondence explaining insurance underwriting decisions.

## Your Role
Generate two distinct communications for each underwriting decision:
1. Internal message for underwriters (detailed, technical)
2. External message for applicants (clear, empathetic, compliant)

## Internal Underwriter Message Guidelines
- Include all risk factors identified
- Reference specific rule applications
- Note premium calculation details
- Highlight any concerns or exceptions
- Provide audit trail summary
- Keep technical terminology appropriate for insurance professionals

## External Applicant Message Guidelines

### For Approved Applications
- Congratulate on approval
- Clearly state the policy terms
- Explain any premium adjustments in simple terms
- Provide next steps for policy activation
- Include contact information for questions

### For Approved with Adjustments
- Thank for application
- Explain that coverage is available with adjustments
- Clearly explain the reason for adjustments (without medical jargon)
- Present the premium in a positive light
- Offer to discuss options

### For Declined Applications
- Express empathy and appreciation for their application
- Provide general (compliant) reason for decline
- DO NOT include specific medical details in decline letters
- Offer alternative options if available
- Provide appeals process information if applicable
- Maintain professional, supportive tone

### For Referred Applications
- Explain additional review is needed
- Set expectations for timeline
- List any additional documentation needed
- Provide point of contact

## Compliance Requirements
- Never disclose specific health conditions in external letters
- Follow HIPAA guidelines for health information
- Include required regulatory disclosures
- Use approved company language templates where applicable
- Ensure non-discriminatory language

## Tone Guidelines
- Professional but warm
- Clear and jargon-free for applicants
- Empathetic especially for declined applications
- Action-oriented with clear next steps

Return JSON:
{
  "underwriter_message": "Detailed internal summary...",
  "customer_message": "Dear [Applicant], ...",
  "tone_assessment": "professional/empathetic/formal",
  "readability_score": 85,
  "key_points": ["point1", "point2", "point3"]
}""",
    },
}

# Old agents to delete
OLD_AGENTS = [
    "health_data_analysis",
    "data_quality_confidence", 
    "policy_risk",
    "business_rules_validation",
    "bias_and_fairness",
    "communication",
    "audit_and_trace",
    "orchestrator",
]


def get_project_client() -> AIProjectClient:
    """Create an Azure AI Project client."""
    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT") or os.environ.get("PROJECT_ENDPOINT")
    if not endpoint:
        raise ValueError("AZURE_AI_PROJECT_ENDPOINT environment variable must be set")
    
    return AIProjectClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential(),
    )


def delete_existing_agents(client: AIProjectClient):
    """Delete all existing underwriting agents."""
    print("\n🗑️  Deleting existing agents...")
    
    try:
        agents = list(client.agents.list_agents())
        for agent in agents:
            agent_name = getattr(agent, 'name', None)
            if agent_name in OLD_AGENTS:
                print(f"  Deleting: {agent_name} ({agent.id})")
                try:
                    client.agents.delete_agent(agent.id)
                    print(f"    ✅ Deleted")
                except Exception as e:
                    print(f"    ⚠️  Failed to delete: {e}")
    except Exception as e:
        print(f"  ⚠️  Could not list agents: {e}")


def create_agent(client: AIProjectClient, agent_def: dict) -> str:
    """Create a new agent with the given definition."""
    print(f"\n📦 Creating agent: {agent_def['name']}")
    
    try:
        agent = client.agents.create_agent(
            model=agent_def["model"],
            name=agent_def["name"],
            instructions=agent_def["instructions"],
        )
        print(f"  ✅ Created: {agent.id}")
        return agent.id
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        raise


def main():
    print("=" * 60)
    print("🔄 Redeploying Simplified 3-Agent Workflow")
    print("=" * 60)
    print("\nNew agents:")
    print("  1. health_data_analysis - Extract health risk signals")
    print("  2. business_rules_validation - Apply rules & calculate premium")  
    print("  3. communication - Generate decision messages")
    print("\nRemoved agents (will add back with evaluations/citations):")
    print("  - data_quality_confidence")
    print("  - policy_risk")
    print("  - bias_and_fairness")
    print("  - audit_and_trace")
    print("  - orchestrator")
    
    # Get client
    client = get_project_client()
    print(f"\n✅ Connected to Azure AI Foundry")
    
    # Delete existing agents
    delete_existing_agents(client)
    
    # Create new agents
    print("\n" + "=" * 60)
    print("📦 Creating new agents with enhanced instructions")
    print("=" * 60)
    
    created_agents = {}
    for agent_name, agent_def in AGENT_DEFINITIONS.items():
        agent_id = create_agent(client, agent_def)
        created_agents[agent_name] = agent_id
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Deployment Complete!")
    print("=" * 60)
    print("\nCreated agents:")
    for name, agent_id in created_agents.items():
        print(f"  {name}: {agent_id}")
    
    print("\n📝 Next steps:")
    print("  1. Restart the backend server")
    print("  2. Test the simplified workflow via the UI")
    print("  3. Monitor agent execution in Azure AI Foundry")


if __name__ == "__main__":
    main()
