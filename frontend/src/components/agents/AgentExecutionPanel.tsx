'use client';

import { useState } from 'react';
import { ChevronDown, ChevronRight, CheckCircle, Clock, AlertCircle, Wrench } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { 
  AgentExecutionRecord, 
  AgentDefinition,
} from '@/lib/agentTypes';
import { 
  AGENT_DEFINITIONS, 
  getAgentDisplayName, 
  getAgentIcon 
} from '@/lib/agentTypes';

interface AgentExecutionPanelProps {
  record: AgentExecutionRecord;
  isExpanded?: boolean;
  onToggle?: () => void;
}

/**
 * AgentExecutionPanel - Displays a single agent's execution details
 * 
 * Shows:
 * - Agent name and purpose
 * - Inputs consumed
 * - Tools used
 * - Outputs produced
 * - Execution time and status
 */
export default function AgentExecutionPanel({
  record,
  isExpanded = false,
  onToggle,
}: AgentExecutionPanelProps) {
  const [localExpanded, setLocalExpanded] = useState(isExpanded);
  const expanded = onToggle ? isExpanded : localExpanded;
  const handleToggle = onToggle || (() => setLocalExpanded(!localExpanded));

  const definition = AGENT_DEFINITIONS[record.agent_id];
  const displayName = getAgentDisplayName(record.agent_id);
  const icon = getAgentIcon(record.agent_id);

  return (
    <div className={cn(
      "border rounded-lg overflow-hidden transition-all duration-200",
      record.success 
        ? "border-slate-200 hover:border-slate-300" 
        : "border-red-200 hover:border-red-300"
    )}>
      {/* Header - Always visible */}
      <button
        onClick={handleToggle}
        className={cn(
          "w-full flex items-center justify-between px-4 py-3 transition-colors",
          record.success ? "bg-slate-50 hover:bg-slate-100" : "bg-red-50 hover:bg-red-100"
        )}
      >
        <div className="flex items-center gap-3">
          {/* Step Number */}
          <div className={cn(
            "w-7 h-7 rounded-full flex items-center justify-center text-sm font-bold",
            record.success 
              ? "bg-emerald-100 text-emerald-700" 
              : "bg-red-100 text-red-700"
          )}>
            {record.step_number}
          </div>
          
          {/* Agent Icon & Name */}
          <span className="text-xl">{icon}</span>
          <div className="text-left">
            <h4 className="font-semibold text-sm text-slate-900">{displayName}</h4>
            <p className="text-xs text-slate-500 truncate max-w-md">
              {definition?.purpose || 'Processing...'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Status Icon */}
          {record.success ? (
            <CheckCircle className="w-5 h-5 text-emerald-500" />
          ) : (
            <AlertCircle className="w-5 h-5 text-red-500" />
          )}
          
          {/* Execution Time */}
          <div className="flex items-center gap-1 text-xs text-slate-500">
            <Clock className="w-3 h-3" />
            <span>{(record.execution_time_ms / 1000).toFixed(1)}s</span>
          </div>

          {/* Expand/Collapse */}
          {expanded ? (
            <ChevronDown className="w-5 h-5 text-slate-400" />
          ) : (
            <ChevronRight className="w-5 h-5 text-slate-400" />
          )}
        </div>
      </button>

      {/* Expanded Content */}
      <div
        className={cn(
          "transition-all duration-300 ease-in-out overflow-hidden",
          expanded ? "max-h-[1000px] opacity-100" : "max-h-0 opacity-0"
        )}
      >
        <div className="p-4 bg-white space-y-4 border-t border-slate-100">
          {/* Output Summary */}
          <div>
            <h5 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
              Output Summary
            </h5>
            <div className="bg-slate-50 rounded-lg p-3 border border-slate-100">
              <p className="text-sm text-slate-700">{record.output_summary}</p>
            </div>
          </div>

          {definition && (
            <>
              {/* Inputs */}
              <div>
                <h5 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
                  Inputs
                </h5>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(definition.inputs).map(([key, type]) => (
                    <span
                      key={key}
                      className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-blue-50 text-blue-700 border border-blue-100"
                    >
                      {key}: <span className="text-blue-500 ml-1">{type}</span>
                    </span>
                  ))}
                </div>
              </div>

              {/* Tools Used */}
              <div>
                <h5 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2 flex items-center gap-1">
                  <Wrench className="w-3 h-3" />
                  Tools Used
                </h5>
                <div className="flex flex-wrap gap-2">
                  {definition.tools_used.map((tool) => (
                    <span
                      key={tool}
                      className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-purple-50 text-purple-700 border border-purple-100"
                    >
                      {tool}
                    </span>
                  ))}
                </div>
              </div>

              {/* Outputs */}
              <div>
                <h5 className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">
                  Outputs
                </h5>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(definition.outputs).map(([key, type]) => (
                    <span
                      key={key}
                      className="inline-flex items-center px-2.5 py-1 rounded-md text-xs font-medium bg-emerald-50 text-emerald-700 border border-emerald-100"
                    >
                      {key}: <span className="text-emerald-500 ml-1">{type}</span>
                    </span>
                  ))}
                </div>
              </div>

              {/* Evaluation Criteria & Failure Modes */}
              <div className="grid grid-cols-2 gap-4 pt-2 border-t border-slate-100">
                <div>
                  <h5 className="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-1">
                    Evaluation Criteria
                  </h5>
                  <div className="flex flex-wrap gap-1">
                    {definition.evaluation_criteria.map((criterion) => (
                      <span
                        key={criterion}
                        className="text-xs text-slate-600 bg-slate-100 px-2 py-0.5 rounded"
                      >
                        {criterion}
                      </span>
                    ))}
                  </div>
                </div>
                <div>
                  <h5 className="text-xs font-semibold uppercase tracking-wide text-slate-400 mb-1">
                    Failure Modes
                  </h5>
                  <div className="flex flex-wrap gap-1">
                    {definition.failure_modes.map((mode) => (
                      <span
                        key={mode}
                        className="text-xs text-amber-700 bg-amber-50 px-2 py-0.5 rounded"
                      >
                        {mode}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Execution Metadata */}
          <div className="text-xs text-slate-400 pt-2 border-t border-slate-100 flex gap-4">
            <span>Execution ID: <code className="font-mono">{record.execution_id.slice(0, 8)}</code></span>
            <span>Timestamp: {new Date(record.timestamp).toLocaleTimeString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
