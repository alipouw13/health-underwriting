'use client';

import { useState } from 'react';
import { cn } from '@/lib/utils';
import { 
  Activity, 
  ChevronDown, 
  ChevronUp, 
  Cpu, 
  Eye,
  Layers,
} from 'lucide-react';
import AgentExecutionPanel from './AgentExecutionPanel';
import OrchestrationTimeline, { OrchestrationTimelineCompact } from './OrchestrationTimeline';
import FinalDecisionSummary, { FinalDecisionCompact } from './FinalDecisionSummary';
import CollapsibleSection from '@/components/CollapsibleSection';
import type { OrchestratorOutput } from '@/lib/agentTypes';

interface AgentTransparencyViewProps {
  orchestratorOutput: OrchestratorOutput;
  className?: string;
  defaultExpanded?: boolean;
}

/**
 * AgentTransparencyView - Main component for agent transparency and explainability
 * 
 * Combines:
 * 1. Final Decision Summary - Top-level decision with confidence
 * 2. Orchestration Timeline - Sequential execution visualization
 * 3. Agent Execution Panels - Detailed view of each agent's work
 * 
 * CONSTRAINT: UI consumes orchestrator output only - no direct agent calls
 */
export default function AgentTransparencyView({
  orchestratorOutput,
  className,
  defaultExpanded = true,
}: AgentTransparencyViewProps) {
  const [expandedAgentId, setExpandedAgentId] = useState<string | null>(null);
  const [showAllAgents, setShowAllAgents] = useState(false);

  // Sort execution records by step number
  const sortedRecords = [...orchestratorOutput.execution_records].sort(
    (a, b) => a.step_number - b.step_number
  );

  const handleAgentToggle = (agentId: string) => {
    setExpandedAgentId(prev => prev === agentId ? null : agentId);
  };

  return (
    <div className={cn("space-y-6", className)}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-indigo-100 flex items-center justify-center">
            <Cpu className="w-6 h-6 text-indigo-600" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-900">Agent Orchestration</h2>
            <p className="text-sm text-slate-500">
              Multi-agent underwriting decision • {sortedRecords.length} agents executed
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <OrchestrationTimelineCompact 
            executionRecords={sortedRecords}
            totalExecutionTimeMs={orchestratorOutput.total_execution_time_ms}
          />
        </div>
      </div>

      {/* Final Decision Summary */}
      <CollapsibleSection
        title="Final Decision"
        icon={<Eye className="w-4 h-4" />}
        defaultOpen={defaultExpanded}
        accentColor="emerald"
      >
        <FinalDecisionSummary orchestratorOutput={orchestratorOutput} />
      </CollapsibleSection>

      {/* Orchestration Timeline */}
      <CollapsibleSection
        title="Execution Timeline"
        icon={<Activity className="w-4 h-4" />}
        defaultOpen={defaultExpanded}
        accentColor="blue"
      >
        <OrchestrationTimeline
          executionRecords={sortedRecords}
          workflowId={orchestratorOutput.workflow_id}
          totalExecutionTimeMs={orchestratorOutput.total_execution_time_ms}
        />
      </CollapsibleSection>

      {/* Agent Execution Details */}
      <CollapsibleSection
        title="Agent Execution Details"
        icon={<Layers className="w-4 h-4" />}
        defaultOpen={defaultExpanded}
        accentColor="indigo"
      >
        <div className="space-y-3">
          {/* Toggle All Button */}
          <div className="flex justify-end">
            <button
              onClick={() => setShowAllAgents(!showAllAgents)}
              className="flex items-center gap-1 text-xs text-indigo-600 hover:text-indigo-800 transition-colors"
            >
              {showAllAgents ? (
                <>
                  <ChevronUp className="w-4 h-4" />
                  Collapse All
                </>
              ) : (
                <>
                  <ChevronDown className="w-4 h-4" />
                  Expand All
                </>
              )}
            </button>
          </div>

          {/* Agent Panels */}
          <div className="space-y-2">
            {sortedRecords.map((record) => (
              <AgentExecutionPanel
                key={record.execution_id}
                record={record}
                isExpanded={showAllAgents || expandedAgentId === record.agent_id}
                onToggle={() => handleAgentToggle(record.agent_id)}
              />
            ))}
          </div>
        </div>
      </CollapsibleSection>

      {/* Workflow Metadata */}
      <div className="text-xs text-slate-400 text-center py-2 border-t border-slate-100">
        Workflow ID: <code className="font-mono">{orchestratorOutput.workflow_id}</code>
        {' • '}
        Completed: {new Date(orchestratorOutput.timestamp).toLocaleString()}
        {' • '}
        Total Duration: {(orchestratorOutput.total_execution_time_ms / 1000).toFixed(1)}s
      </div>
    </div>
  );
}

/**
 * Compact version for embedding in existing panels
 */
export function AgentTransparencyCompact({
  orchestratorOutput,
  onViewDetails,
}: {
  orchestratorOutput: OrchestratorOutput;
  onViewDetails?: () => void;
}) {
  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-indigo-600" />
          <span className="font-medium text-sm text-slate-900">Agent Decision</span>
        </div>
        {onViewDetails && (
          <button
            onClick={onViewDetails}
            className="text-xs text-indigo-600 hover:text-indigo-800"
          >
            View Details →
          </button>
        )}
      </div>
      
      <FinalDecisionCompact orchestratorOutput={orchestratorOutput} />
      
      <OrchestrationTimelineCompact 
        executionRecords={orchestratorOutput.execution_records}
        totalExecutionTimeMs={orchestratorOutput.total_execution_time_ms}
      />
    </div>
  );
}
