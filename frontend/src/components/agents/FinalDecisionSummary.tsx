'use client';

import { cn } from '@/lib/utils';
import { 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  Clock, 
  ShieldCheck, 
  ShieldAlert,
  TrendingUp,
  DollarSign,
  FileText,
  AlertCircle,
} from 'lucide-react';
import type { 
  FinalDecision, 
  OrchestratorOutput,
  DecisionStatus,
  RiskLevel,
} from '@/lib/agentTypes';
import { 
  getRiskLevelColor, 
  getStatusColor,
} from '@/lib/agentTypes';

interface FinalDecisionSummaryProps {
  orchestratorOutput: OrchestratorOutput;
  className?: string;
}

/**
 * FinalDecisionSummary - Displays the final underwriting decision
 * 
 * Shows:
 * - Decision status (approved, declined, referred, etc.)
 * - Confidence score
 * - Key contributing factors
 * - Bias or compliance flags
 * - Premium adjustment
 */
export default function FinalDecisionSummary({
  orchestratorOutput,
  className,
}: FinalDecisionSummaryProps) {
  const { final_decision, confidence_score, explanation } = orchestratorOutput;

  return (
    <div className={cn("bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm", className)}>
      {/* Decision Header */}
      <div className={cn(
        "px-6 py-4 border-b",
        final_decision.approved 
          ? "bg-gradient-to-r from-emerald-50 to-green-50 border-emerald-100"
          : final_decision.status === 'referred'
          ? "bg-gradient-to-r from-amber-50 to-yellow-50 border-amber-100"
          : "bg-gradient-to-r from-red-50 to-rose-50 border-red-100"
      )}>
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            {/* Status Icon */}
            {final_decision.approved ? (
              <div className="w-12 h-12 rounded-full bg-emerald-100 flex items-center justify-center">
                <CheckCircle className="w-7 h-7 text-emerald-600" />
              </div>
            ) : final_decision.status === 'referred' ? (
              <div className="w-12 h-12 rounded-full bg-amber-100 flex items-center justify-center">
                <AlertTriangle className="w-7 h-7 text-amber-600" />
              </div>
            ) : (
              <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center">
                <XCircle className="w-7 h-7 text-red-600" />
              </div>
            )}
            
            <div>
              <h2 className="text-xl font-bold text-slate-900">
                {formatDecisionStatus(final_decision.status)}
              </h2>
              <p className="text-sm text-slate-600">
                Patient: <span className={final_decision.patient_name ? "" : "font-mono"}>
                  {final_decision.patient_name || final_decision.patient_id}
                </span>
              </p>
            </div>
          </div>

          {/* Confidence Score */}
          <div className="text-right">
            <ConfidenceGauge score={confidence_score} />
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-4 divide-x divide-slate-100 border-b border-slate-100">
        {/* Risk Level */}
        <div className="p-4 text-center">
          <TrendingUp className="w-5 h-5 text-slate-400 mx-auto mb-1" />
          <div className={cn(
            "inline-block px-2 py-1 rounded-full text-xs font-semibold border",
            getRiskLevelColor(final_decision.risk_level)
          )}>
            {final_decision.risk_level.replace('_', ' ').toUpperCase()}
          </div>
          <p className="text-[10px] text-slate-400 mt-1">Risk Level</p>
        </div>

        {/* Premium Adjustment */}
        <div className="p-4 text-center">
          <DollarSign className="w-5 h-5 text-slate-400 mx-auto mb-1" />
          <div className={cn(
            "text-lg font-bold",
            final_decision.premium_adjustment_pct > 0 
              ? "text-amber-600" 
              : final_decision.premium_adjustment_pct < 0 
              ? "text-emerald-600"
              : "text-slate-600"
          )}>
            {final_decision.premium_adjustment_pct > 0 ? '+' : ''}{final_decision.premium_adjustment_pct.toFixed(1)}%
          </div>
          <p className="text-[10px] text-slate-400 mt-1">Premium Adj.</p>
        </div>

        {/* Annual Premium */}
        <div className="p-4 text-center">
          <FileText className="w-5 h-5 text-slate-400 mx-auto mb-1" />
          <div className="text-lg font-bold text-slate-700">
            ${final_decision.adjusted_premium_annual.toLocaleString()}
          </div>
          <p className="text-[10px] text-slate-400 mt-1">Annual Premium</p>
        </div>

        {/* Decision Time */}
        <div className="p-4 text-center">
          <Clock className="w-5 h-5 text-slate-400 mx-auto mb-1" />
          <div className="text-lg font-bold text-slate-700">
            {(orchestratorOutput.total_execution_time_ms / 1000).toFixed(1)}s
          </div>
          <p className="text-[10px] text-slate-400 mt-1">Decision Time</p>
        </div>
      </div>

      {/* Compliance Flags */}
      <div className="p-4 border-b border-slate-100">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-3">
          Compliance & Fairness Checks
        </h4>
        <div className="flex flex-wrap gap-3">
          {/* Business Rules */}
          <ComplianceFlag
            label="Business Rules"
            passed={final_decision.business_rules_approved}
            passedText="Compliant"
            failedText="Violations Found"
          />
          
          {/* Bias Check */}
          <ComplianceFlag
            label="Bias & Fairness"
            passed={final_decision.bias_check_passed}
            passedText="No Issues"
            failedText="Flags Detected"
          />
        </div>
      </div>

      {/* Explanation */}
      <div className="p-4">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
          Decision Explanation
        </h4>
        <div className="bg-slate-50 rounded-lg p-3 border border-slate-100">
          <pre className="text-xs text-slate-700 whitespace-pre-wrap font-sans leading-relaxed">
            {explanation}
          </pre>
        </div>
      </div>

      {/* Footer */}
      <div className="px-4 py-2 bg-slate-50 border-t border-slate-100 flex justify-between items-center">
        <span className="text-[10px] text-slate-400">
          Decision ID: <code className="font-mono">{final_decision.decision_id}</code>
        </span>
        <span className="text-[10px] text-slate-400">
          {new Date(final_decision.timestamp).toLocaleString()}
        </span>
      </div>
    </div>
  );
}

/**
 * Confidence Gauge Component
 */
function ConfidenceGauge({ score }: { score: number }) {
  const percentage = Math.round(score * 100);
  const getConfidenceLevel = () => {
    if (percentage >= 90) return { label: 'Very High', color: 'text-emerald-600', bg: 'bg-emerald-500' };
    if (percentage >= 75) return { label: 'High', color: 'text-emerald-600', bg: 'bg-emerald-500' };
    if (percentage >= 60) return { label: 'Moderate', color: 'text-amber-600', bg: 'bg-amber-500' };
    if (percentage >= 40) return { label: 'Low', color: 'text-orange-600', bg: 'bg-orange-500' };
    return { label: 'Very Low', color: 'text-red-600', bg: 'bg-red-500' };
  };

  const level = getConfidenceLevel();

  return (
    <div className="text-center">
      <div className="relative w-16 h-16">
        {/* Background Circle */}
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="32"
            cy="32"
            r="28"
            fill="none"
            stroke="#e2e8f0"
            strokeWidth="6"
          />
          <circle
            cx="32"
            cy="32"
            r="28"
            fill="none"
            stroke={percentage >= 75 ? '#10b981' : percentage >= 50 ? '#f59e0b' : '#ef4444'}
            strokeWidth="6"
            strokeDasharray={`${percentage * 1.76} 176`}
            strokeLinecap="round"
          />
        </svg>
        {/* Percentage Text */}
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={cn("text-lg font-bold", level.color)}>
            {percentage}%
          </span>
        </div>
      </div>
      <p className={cn("text-xs font-medium mt-1", level.color)}>
        {level.label} Confidence
      </p>
    </div>
  );
}

/**
 * Compliance Flag Component
 */
function ComplianceFlag({
  label,
  passed,
  passedText,
  failedText,
}: {
  label: string;
  passed: boolean;
  passedText: string;
  failedText: string;
}) {
  return (
    <div className={cn(
      "flex items-center gap-2 px-3 py-2 rounded-lg border",
      passed 
        ? "bg-emerald-50 border-emerald-200"
        : "bg-red-50 border-red-200"
    )}>
      {passed ? (
        <ShieldCheck className="w-5 h-5 text-emerald-600" />
      ) : (
        <ShieldAlert className="w-5 h-5 text-red-600" />
      )}
      <div>
        <p className="text-xs font-medium text-slate-700">{label}</p>
        <p className={cn(
          "text-xs",
          passed ? "text-emerald-600" : "text-red-600"
        )}>
          {passed ? passedText : failedText}
        </p>
      </div>
    </div>
  );
}

/**
 * Format decision status for display
 */
function formatDecisionStatus(status: DecisionStatus): string {
  const labels: Record<DecisionStatus, string> = {
    approved: 'Application Approved',
    approved_with_adjustment: 'Approved with Adjustment',
    referred: 'Referred for Review',
    declined: 'Application Declined',
    pending_info: 'Pending Information',
  };
  return labels[status] || status;
}

/**
 * Compact version for dashboard cards
 */
export function FinalDecisionCompact({
  orchestratorOutput,
}: {
  orchestratorOutput: OrchestratorOutput;
}) {
  const { final_decision, confidence_score } = orchestratorOutput;

  return (
    <div className="flex items-center justify-between p-3 bg-white rounded-lg border border-slate-200">
      <div className="flex items-center gap-3">
        {final_decision.approved ? (
          <CheckCircle className="w-6 h-6 text-emerald-500" />
        ) : final_decision.status === 'referred' ? (
          <AlertTriangle className="w-6 h-6 text-amber-500" />
        ) : (
          <XCircle className="w-6 h-6 text-red-500" />
        )}
        <div>
          <p className="font-medium text-sm text-slate-900">
            {formatDecisionStatus(final_decision.status)}
          </p>
          <p className="text-xs text-slate-500">
            {final_decision.risk_level.replace('_', ' ')} risk
          </p>
        </div>
      </div>
      <div className="text-right">
        <p className={cn(
          "text-sm font-bold",
          confidence_score >= 0.75 ? "text-emerald-600" : 
          confidence_score >= 0.5 ? "text-amber-600" : "text-red-600"
        )}>
          {(confidence_score * 100).toFixed(0)}%
        </p>
        <p className="text-[10px] text-slate-400">confidence</p>
      </div>
    </div>
  );
}
