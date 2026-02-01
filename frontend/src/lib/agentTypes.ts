/**
 * Type definitions for Agent Transparency UI
 * 
 * These types match the OrchestratorOutput from the backend.
 * UI must consume orchestrator output only - no direct agent calls.
 */

// =============================================================================
// EXECUTION RECORD TYPES
// =============================================================================

export interface AgentExecutionRecord {
  agent_id: string;
  step_number: number;
  execution_id: string;
  timestamp: string;
  execution_time_ms: number;
  success: boolean;
  output_summary: string;
}

// =============================================================================
// AGENT DEFINITION TYPES (from YAML)
// =============================================================================

export interface AgentDefinition {
  agent_id: string;
  purpose: string;
  inputs: Record<string, string>;
  outputs: Record<string, string>;
  tools_used: string[];
  evaluation_criteria: string[];
  failure_modes: string[];
}

// Static agent definitions from /.github/underwriting_agents.yaml
export const AGENT_DEFINITIONS: Record<string, AgentDefinition> = {
  HealthDataAnalysisAgent: {
    agent_id: 'HealthDataAnalysisAgent',
    purpose: 'Analyze simulated Apple Health data to extract health risk signals',
    inputs: { health_metrics: 'object', patient_profile: 'object' },
    outputs: { risk_indicators: 'list', summary: 'string' },
    tools_used: ['medical-mcp-simulator'],
    evaluation_criteria: ['signal_accuracy', 'explainability'],
    failure_modes: ['missing_data', 'inconsistent_metrics'],
  },
  DataQualityConfidenceAgent: {
    agent_id: 'DataQualityConfidenceAgent',
    purpose: 'Assess reliability and completeness of health data',
    inputs: { health_metrics: 'object' },
    outputs: { confidence_score: 'number', quality_flags: 'list' },
    tools_used: ['data-quality-analyzer'],
    evaluation_criteria: ['coverage', 'freshness'],
    failure_modes: ['insufficient_data'],
  },
  PolicyRiskAgent: {
    agent_id: 'PolicyRiskAgent',
    purpose: 'Translate health signals into insurance risk categories',
    inputs: { risk_indicators: 'list', policy_rules: 'object' },
    outputs: { risk_level: 'string', premium_adjustment_recommendation: 'object' },
    tools_used: ['policy-rule-engine'],
    evaluation_criteria: ['rule_alignment', 'consistency'],
    failure_modes: ['conflicting_rules'],
  },
  BusinessRulesValidationAgent: {
    agent_id: 'BusinessRulesValidationAgent',
    purpose: 'Validate underwriting rules and regulatory constraints',
    inputs: { premium_adjustment_recommendation: 'object' },
    outputs: { approved: 'boolean', rationale: 'string' },
    tools_used: ['underwriting-rules-mcp'],
    evaluation_criteria: ['compliance'],
    failure_modes: ['rule_violation'],
  },
  BiasAndFairnessAgent: {
    agent_id: 'BiasAndFairnessAgent',
    purpose: 'Detect bias or sensitive-attribute leakage',
    inputs: { decision_context: 'object' },
    outputs: { bias_flags: 'list', mitigation_notes: 'string' },
    tools_used: ['fairness-checker'],
    evaluation_criteria: ['fairness'],
    failure_modes: ['bias_detected'],
  },
  CommunicationAgent: {
    agent_id: 'CommunicationAgent',
    purpose: 'Generate explanations for underwriters and customers',
    inputs: { decision_summary: 'object' },
    outputs: { underwriter_message: 'string', customer_message: 'string' },
    tools_used: ['language-generator'],
    evaluation_criteria: ['clarity', 'tone'],
    failure_modes: ['ambiguous_language'],
  },
  AuditAndTraceAgent: {
    agent_id: 'AuditAndTraceAgent',
    purpose: 'Produce a full decision audit trail',
    inputs: { agent_outputs: 'list' },
    outputs: { audit_log: 'object' },
    tools_used: ['trace-logger'],
    evaluation_criteria: ['completeness'],
    failure_modes: ['missing_steps'],
  },
};

// =============================================================================
// FINAL DECISION TYPES
// =============================================================================

export type DecisionStatus = 
  | 'approved'
  | 'approved_with_adjustment'
  | 'referred'
  | 'declined'
  | 'pending_info';

export type RiskLevel = 
  | 'low'
  | 'moderate'
  | 'high'
  | 'very_high'
  | 'decline';

export interface FinalDecision {
  decision_id: string;
  patient_id: string;
  status: DecisionStatus;
  risk_level: RiskLevel;
  approved: boolean;
  premium_adjustment_pct: number;
  adjusted_premium_annual: number;
  business_rules_approved: boolean;
  bias_check_passed: boolean;
  underwriter_message: string;
  customer_message: string;
  timestamp: string;
}

// =============================================================================
// ORCHESTRATOR OUTPUT (main type consumed by UI)
// =============================================================================

export interface OrchestratorOutput {
  agent_id: string;
  success: boolean;
  execution_id: string;
  timestamp: string;
  execution_time_ms: number;
  final_decision: FinalDecision;
  confidence_score: number;
  explanation: string;
  execution_records: AgentExecutionRecord[];
  workflow_id: string;
  total_execution_time_ms: number;
}

// =============================================================================
// UI HELPER TYPES
// =============================================================================

export interface AgentStatus {
  agent_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  step_number: number;
}

export function getAgentDisplayName(agentId: string): string {
  const names: Record<string, string> = {
    HealthDataAnalysisAgent: 'Health Data Analysis',
    DataQualityConfidenceAgent: 'Data Quality & Confidence',
    PolicyRiskAgent: 'Policy Risk Assessment',
    BusinessRulesValidationAgent: 'Business Rules Validation',
    BiasAndFairnessAgent: 'Bias & Fairness Check',
    CommunicationAgent: 'Communication Generation',
    AuditAndTraceAgent: 'Audit & Trace',
  };
  return names[agentId] || agentId;
}

export function getAgentIcon(agentId: string): string {
  const icons: Record<string, string> = {
    HealthDataAnalysisAgent: '🩺',
    DataQualityConfidenceAgent: '📊',
    PolicyRiskAgent: '⚖️',
    BusinessRulesValidationAgent: '✅',
    BiasAndFairnessAgent: '🛡️',
    CommunicationAgent: '💬',
    AuditAndTraceAgent: '📝',
  };
  return icons[agentId] || '🤖';
}

export function getRiskLevelColor(level: RiskLevel): string {
  const colors: Record<RiskLevel, string> = {
    low: 'text-emerald-600 bg-emerald-50 border-emerald-200',
    moderate: 'text-amber-600 bg-amber-50 border-amber-200',
    high: 'text-orange-600 bg-orange-50 border-orange-200',
    very_high: 'text-red-600 bg-red-50 border-red-200',
    decline: 'text-red-800 bg-red-100 border-red-300',
  };
  return colors[level] || 'text-slate-600 bg-slate-50 border-slate-200';
}

export function getStatusColor(status: DecisionStatus): string {
  const colors: Record<DecisionStatus, string> = {
    approved: 'text-emerald-600 bg-emerald-50 border-emerald-200',
    approved_with_adjustment: 'text-blue-600 bg-blue-50 border-blue-200',
    referred: 'text-amber-600 bg-amber-50 border-amber-200',
    declined: 'text-red-600 bg-red-50 border-red-200',
    pending_info: 'text-slate-600 bg-slate-50 border-slate-200',
  };
  return colors[status] || 'text-slate-600 bg-slate-50 border-slate-200';
}
