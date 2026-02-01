'use client';

import { cn } from '@/lib/utils';
import { CheckCircle, Clock, ArrowRight } from 'lucide-react';
import type { AgentExecutionRecord } from '@/lib/agentTypes';
import { getAgentDisplayName, getAgentIcon } from '@/lib/agentTypes';

interface OrchestrationTimelineProps {
  executionRecords: AgentExecutionRecord[];
  workflowId: string;
  totalExecutionTimeMs: number;
  className?: string;
}

/**
 * OrchestrationTimeline - Shows sequential agent execution order
 * 
 * Displays:
 * - Visual timeline of all 7 agents
 * - Execution order (1-7)
 * - Status and timing for each step
 * - Total workflow duration
 */
export default function OrchestrationTimeline({
  executionRecords,
  workflowId,
  totalExecutionTimeMs,
  className,
}: OrchestrationTimelineProps) {
  // Sort by step number to ensure correct order
  const sortedRecords = [...executionRecords].sort((a, b) => a.step_number - b.step_number);

  return (
    <div className={cn("bg-white border border-slate-200 rounded-xl overflow-hidden", className)}>
      {/* Header */}
      <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-sm text-slate-900">Orchestration Timeline</h3>
            <p className="text-xs text-slate-500">
              Sequential execution • {sortedRecords.length} agents
            </p>
          </div>
          <div className="text-right">
            <div className="flex items-center gap-1 text-emerald-600 text-sm font-medium">
              <Clock className="w-4 h-4" />
              {(totalExecutionTimeMs / 1000).toFixed(1)}s
            </div>
            <p className="text-xs text-slate-400 font-mono">{workflowId.slice(0, 8)}</p>
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="p-4">
        <div className="relative">
          {sortedRecords.map((record, index) => {
            const isLast = index === sortedRecords.length - 1;
            const icon = getAgentIcon(record.agent_id);
            const displayName = getAgentDisplayName(record.agent_id);

            return (
              <div key={record.execution_id} className="relative">
                {/* Connection Line */}
                {!isLast && (
                  <div className="absolute left-[18px] top-10 w-0.5 h-8 bg-gradient-to-b from-emerald-300 to-emerald-200" />
                )}

                <div className="flex items-start gap-3 pb-4">
                  {/* Step Indicator */}
                  <div className="relative flex-shrink-0">
                    <div className={cn(
                      "w-9 h-9 rounded-full flex items-center justify-center",
                      record.success 
                        ? "bg-emerald-100 text-emerald-700 ring-2 ring-emerald-200"
                        : "bg-red-100 text-red-700 ring-2 ring-red-200"
                    )}>
                      {record.success ? (
                        <CheckCircle className="w-5 h-5" />
                      ) : (
                        <span className="text-sm font-bold">{record.step_number}</span>
                      )}
                    </div>
                    {/* Step Number Badge */}
                    <span className="absolute -top-1 -right-1 w-4 h-4 bg-slate-700 text-white text-[10px] font-bold rounded-full flex items-center justify-center">
                      {record.step_number}
                    </span>
                  </div>

                  {/* Agent Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-base">{icon}</span>
                      <h4 className="font-medium text-sm text-slate-900 truncate">
                        {displayName}
                      </h4>
                    </div>
                    <p className="text-xs text-slate-500 mt-0.5 truncate">
                      {record.output_summary}
                    </p>
                  </div>

                  {/* Timing */}
                  <div className="flex-shrink-0 text-right">
                    <div className="text-xs font-medium text-slate-600">
                      {(record.execution_time_ms / 1000).toFixed(1)}s
                    </div>
                    <div className="text-[10px] text-slate-400">
                      {new Date(record.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>

                {/* Arrow to next */}
                {!isLast && (
                  <div className="absolute left-[14px] top-10 transform translate-y-2">
                    <ArrowRight className="w-3 h-3 text-emerald-400 rotate-90" />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer - Execution Order Constraint */}
      <div className="px-4 py-2 bg-slate-50 border-t border-slate-100">
        <p className="text-[10px] text-slate-400 text-center">
          Strict execution order enforced • No conditional branching • No agent skipping
        </p>
      </div>
    </div>
  );
}

/**
 * Compact version for sidebar or smaller spaces
 */
export function OrchestrationTimelineCompact({
  executionRecords,
  totalExecutionTimeMs,
}: {
  executionRecords: AgentExecutionRecord[];
  totalExecutionTimeMs: number;
}) {
  const sortedRecords = [...executionRecords].sort((a, b) => a.step_number - b.step_number);

  return (
    <div className="flex items-center gap-1">
      {sortedRecords.map((record, index) => (
        <div key={record.execution_id} className="flex items-center">
          <div
            className={cn(
              "w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold",
              record.success 
                ? "bg-emerald-100 text-emerald-700" 
                : "bg-red-100 text-red-700"
            )}
            title={`${getAgentDisplayName(record.agent_id)} - ${(record.execution_time_ms / 1000).toFixed(1)}s`}
          >
            {getAgentIcon(record.agent_id)}
          </div>
          {index < sortedRecords.length - 1 && (
            <ArrowRight className="w-3 h-3 text-slate-300 mx-0.5" />
          )}
        </div>
      ))}
      <span className="ml-2 text-xs text-slate-500">
        {(totalExecutionTimeMs / 1000).toFixed(1)}s
      </span>
    </div>
  );
}
