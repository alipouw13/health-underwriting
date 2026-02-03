'use client';

import { cn } from '@/lib/utils';
import { 
  CheckCircle2, 
  XCircle, 
  AlertCircle, 
  Clock, 
  Star,
  BarChart3,
} from 'lucide-react';
import type { AgentEvaluationResult, MetricScore, WorkflowEvaluationResult } from '@/lib/agentTypes';
import { getAgentDisplayName } from '@/lib/agentTypes';

// =============================================================================
// METRIC SCORE DISPLAY
// =============================================================================

interface MetricScoreBarProps {
  metric: MetricScore;
}

function MetricScoreBar({ metric }: MetricScoreBarProps) {
  const percentage = (metric.score / 5) * 100; // Assuming 5 is max score
  const passColor = metric.passed ? 'bg-emerald-500' : 'bg-amber-500';
  
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="font-medium text-slate-700 capitalize">
          {metric.metric_name.replace(/_/g, ' ')}
        </span>
        <div className="flex items-center gap-2">
          <span className={cn(
            "font-semibold",
            metric.passed ? "text-emerald-600" : "text-amber-600"
          )}>
            {metric.score.toFixed(2)}
          </span>
          {metric.passed ? (
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
          ) : (
            <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
          )}
        </div>
      </div>
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div 
          className={cn("h-full rounded-full transition-all", passColor)}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {metric.reason && (
        <p className="text-[10px] text-slate-500 italic">{metric.reason}</p>
      )}
    </div>
  );
}

// =============================================================================
// AGENT EVALUATION CARD
// =============================================================================

interface AgentEvaluationCardProps {
  evaluation: AgentEvaluationResult;
  agentId: string;
  compact?: boolean;
}

export function AgentEvaluationCard({ evaluation, agentId, compact = false }: AgentEvaluationCardProps) {
  const displayName = getAgentDisplayName(agentId);
  
  const statusConfig = {
    pending: { color: 'text-slate-500', bg: 'bg-slate-100', icon: Clock },
    running: { color: 'text-blue-600', bg: 'bg-blue-100', icon: Clock },
    completed: { color: 'text-emerald-600', bg: 'bg-emerald-100', icon: CheckCircle2 },
    failed: { color: 'text-red-600', bg: 'bg-red-100', icon: XCircle },
    skipped: { color: 'text-slate-400', bg: 'bg-slate-50', icon: AlertCircle },
  };
  
  const config = statusConfig[evaluation.status] || statusConfig.pending;
  const StatusIcon = config.icon;
  
  // Use overall_score if available, fall back to aggregate_score
  const score = evaluation.overall_score ?? evaluation.aggregate_score;
  
  if (compact) {
    return (
      <div className="flex items-center gap-2 text-xs">
        <StatusIcon className={cn("w-3.5 h-3.5", config.color)} />
        <span className="font-medium">{displayName}</span>
        {score !== undefined && (
          <span className={cn(
            "px-1.5 py-0.5 rounded-full font-semibold",
            config.bg, config.color
          )}>
            {score.toFixed(2)}
          </span>
        )}
      </div>
    );
  }
  
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={cn("p-1.5 rounded-lg", config.bg)}>
            <BarChart3 className={cn("w-4 h-4", config.color)} />
          </div>
          <h4 className="font-medium text-slate-900">{displayName}</h4>
        </div>
        
        <div className="flex items-center gap-2">
          {evaluation.passed !== undefined && (
            <span className={cn(
              "px-2 py-1 rounded-full text-xs font-medium",
              evaluation.passed 
                ? "bg-emerald-100 text-emerald-700" 
                : "bg-amber-100 text-amber-700"
            )}>
              {evaluation.passed ? 'PASSED' : 'NEEDS REVIEW'}
            </span>
          )}
          {score !== undefined && (
            <div className="flex items-center gap-1 text-sm font-semibold text-slate-700">
              <Star className="w-4 h-4 text-amber-500" />
              {score.toFixed(2)}/5
            </div>
          )}
        </div>
      </div>
      
      {/* Metrics */}
      {evaluation.metrics.length > 0 && (
        <div className="space-y-2">
          {evaluation.metrics.map((metric, idx) => (
            <MetricScoreBar key={idx} metric={metric} />
          ))}
        </div>
      )}
      
      {/* Duration */}
      {evaluation.duration_ms && (
        <div className="flex items-center gap-1 text-xs text-slate-400">
          <Clock className="w-3 h-3" />
          <span>Evaluated in {(evaluation.duration_ms / 1000).toFixed(2)}s</span>
        </div>
      )}
      
      {/* Error */}
      {evaluation.error_message && (
        <div className="p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
          {evaluation.error_message}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// WORKFLOW EVALUATION SUMMARY
// =============================================================================

interface WorkflowEvaluationSummaryProps {
  evaluation: WorkflowEvaluationResult;
}

export function WorkflowEvaluationSummary({ evaluation }: WorkflowEvaluationSummaryProps) {
  const agentCount = Object.keys(evaluation.agent_evaluations).length;
  const passedCount = Object.values(evaluation.agent_evaluations)
    .filter(e => e.passed).length;
  
  // Use overall_score if available, fall back to aggregate_score
  const score = evaluation.overall_score ?? evaluation.aggregate_score;
  
  return (
    <div className="bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-200 rounded-xl p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-100 rounded-lg">
            <BarChart3 className="w-5 h-5 text-indigo-600" />
          </div>
          <div>
            <h3 className="font-semibold text-slate-900">Workflow Evaluation</h3>
            <p className="text-xs text-slate-500">
              {agentCount} agents evaluated • {passedCount} passed
            </p>
          </div>
        </div>
        
        {/* Overall Status */}
        <div className="flex items-center gap-3">
          {score !== undefined && (
            <div className="text-right">
              <div className="flex items-center gap-1 text-lg font-bold text-indigo-700">
                <Star className="w-5 h-5 text-amber-500" />
                {score.toFixed(2)}/5
              </div>
              <span className="text-xs text-slate-500">Aggregate Score</span>
            </div>
          )}
          
          {evaluation.overall_passed !== undefined && (
            <div className={cn(
              "px-3 py-1.5 rounded-full font-semibold",
              evaluation.overall_passed 
                ? "bg-emerald-100 text-emerald-700" 
                : "bg-amber-100 text-amber-700"
            )}>
              {evaluation.overall_passed ? 'ALL PASSED' : 'REVIEW NEEDED'}
            </div>
          )}
        </div>
      </div>
      
      {/* Agent Evaluations Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {Object.entries(evaluation.agent_evaluations).map(([agentId, agentEval]) => (
          <AgentEvaluationCard 
            key={agentId} 
            evaluation={agentEval} 
            agentId={agentId}
            compact 
          />
        ))}
      </div>
      
      {/* Errors */}
      {evaluation.errors.length > 0 && (
        <div className="p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
          <strong>Errors:</strong>
          <ul className="list-disc pl-4 mt-1">
            {evaluation.errors.map((err, idx) => (
              <li key={idx}>{err}</li>
            ))}
          </ul>
        </div>
      )}
      
      {/* Duration */}
      {evaluation.total_duration_ms && (
        <div className="flex items-center gap-1 text-xs text-slate-400">
          <Clock className="w-3 h-3" />
          <span>Total evaluation time: {(evaluation.total_duration_ms / 1000).toFixed(2)}s</span>
        </div>
      )}
    </div>
  );
}

// =============================================================================
// EVALUATIONS PANEL (combines all)
// =============================================================================

interface EvaluationsPanelProps {
  evaluations?: Record<string, AgentEvaluationResult>;
  workflowEvaluation?: WorkflowEvaluationResult;
  className?: string;
}

export default function EvaluationsPanel({ 
  evaluations, 
  workflowEvaluation,
  className 
}: EvaluationsPanelProps) {
  if (!evaluations && !workflowEvaluation) {
    return (
      <div className={cn("text-center py-6 text-slate-500", className)}>
        <BarChart3 className="w-8 h-8 mx-auto mb-2 text-slate-300" />
        <p className="text-sm">No evaluation results available</p>
        <p className="text-xs text-slate-400 mt-1">
          Enable FOUNDRY_EVALUATIONS_ENABLED=true to run evaluations
        </p>
      </div>
    );
  }
  
  return (
    <div className={cn("space-y-4", className)}>
      {/* Workflow Summary */}
      {workflowEvaluation && (
        <WorkflowEvaluationSummary evaluation={workflowEvaluation} />
      )}
      
      {/* Individual Agent Evaluations */}
      {evaluations && Object.keys(evaluations).length > 0 && (
        <div className="space-y-3">
          <h4 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <BarChart3 className="w-4 h-4" />
            Agent Evaluation Details
          </h4>
          <div className="grid grid-cols-1 gap-3">
            {Object.entries(evaluations).map(([agentId, evaluation]) => (
              <AgentEvaluationCard 
                key={agentId} 
                evaluation={evaluation} 
                agentId={agentId} 
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
