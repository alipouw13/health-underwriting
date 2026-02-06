"""
Underwriting Agents Module

Azure AI Foundry Agent implementations for health underwriting.

This module provides isolated, single-responsibility agents aligned with
the agent definitions in /.github/underwriting_agents.yaml.

Agent Architecture:
------------------
- Each agent is a single class in a single file
- Agents accept structured input per YAML spec
- Agents return structured output per YAML spec
- Agents validate input/output at runtime
- Agents do NOT call other agents
- Agents do NOT contain orchestration logic
- Agents do NOT depend on UI state

Available Agents:
----------------
- HealthDataAnalysisAgent: Analyze health metrics for risk signals
- PolicyRiskAgent: Translate health signals to insurance risk categories
- AppleHealthRiskAgent: Calculate HKRS for Apple Health workflow
- DataQualityConfidenceAgent: Assess data reliability and completeness
- BiasAndFairnessAgent: Detect bias in decision context
- CommunicationAgent: Generate explanations for stakeholders
- AuditAndTraceAgent: Produce decision audit trails
- OrchestratorAgent: Coordinate agent execution and produce final decision

Workflows:
---------
ADMIN WORKFLOW (Traditional - document upload):
  HealthDataAnalysisAgent → PolicyRiskAgent → CommunicationAgent

APPLE HEALTH WORKFLOW (End User - HealthKit data):
  HealthDataAnalysisAgent → AppleHealthRiskAgent → CommunicationAgent

The orchestrator selects workflow based on persona (admin vs end_user).
"""

from app.agents.base import (
    BaseUnderwritingAgent,
    AgentInput,
    AgentOutput,
    AgentExecutionError,
    AgentValidationError,
)
from app.agents.health_data_analysis import HealthDataAnalysisAgent
from app.agents.policy_risk import PolicyRiskAgent
from app.agents.apple_health_risk import AppleHealthRiskAgent
from app.agents.data_quality_confidence import DataQualityConfidenceAgent
from app.agents.bias_fairness import BiasAndFairnessAgent
from app.agents.communication import CommunicationAgent
from app.agents.audit_trace import AuditAndTraceAgent
from app.agents.orchestrator import OrchestratorAgent, AgentProgress, AgentProgressStatus, AgentProgressStage

__all__ = [
    # Base classes
    "BaseUnderwritingAgent",
    "AgentInput",
    "AgentOutput",
    "AgentExecutionError",
    "AgentValidationError",
    # Agent implementations
    "HealthDataAnalysisAgent",
    "PolicyRiskAgent",
    "AppleHealthRiskAgent",  # Apple Health workflow
    "DataQualityConfidenceAgent",
    "BiasAndFairnessAgent",
    "CommunicationAgent",
    "AuditAndTraceAgent",
    # Orchestrator
    "OrchestratorAgent",
    # Progress tracking
    "AgentProgress",
    "AgentProgressStatus",
    "AgentProgressStage",
]
