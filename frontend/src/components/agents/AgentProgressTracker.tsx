'use client';

import { CheckCircle2, Clock, Loader2, XCircle, Wrench } from 'lucide-react';
import { cn } from '@/lib/utils';

// Granular progress stages within each agent (lowercase to match backend)
export type AgentProgressStage = 
  | 'started'           // Agent execution initiated
  | 'preparing_input'   // Building input payload
  | 'invoking_model'    // Calling Azure AI Foundry
  | 'tool_called'       // Agent is using a tool
  | 'parsing_response'  // Processing agent response
  | 'validating_output' // Validating output schema
  | 'completed'         // Successfully finished
  | 'failed';           // Error occurred

export interface AgentProgressEvent {
  workflow_id: string;
  agent_id: string;
  agent_name: string;
  step_number: number;
  total_steps: number;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'tool_calling' | 'processing' | 'output_ready';
  stage?: AgentProgressStage;
  execution_time_ms?: number;
  message?: string;
  safe_summary?: string;
  tools_used?: string[];
  output_preview?: string;
  timestamp?: string;
}

// Agent definitions for the 4-agent workflow (with PolicyRiskAgent)
const ADMIN_AGENT_DEFINITIONS = [
  { 
    id: 'HealthDataAnalysisAgent', 
    name: 'Health Data Analysis', 
    description: 'Extracting risk indicators from medical data',
    icon: '🏥'
  },
  {
    id: 'PolicyRiskAgent',
    name: 'Policy Risk Assessment',
    description: 'Translating health signals into risk categories',
    icon: '📊'
  },
  { 
    id: 'BusinessRulesValidationAgent', 
    name: 'Business Rules Validation', 
    description: 'Validating against underwriting rules',
    icon: '📋'
  },
  { 
    id: 'CommunicationAgent', 
    name: 'Decision Communication', 
    description: 'Generating underwriter and customer messages',
    icon: '💬'
  },
];

// Agent definitions for Apple Health 3-agent workflow
const APPLE_HEALTH_AGENT_DEFINITIONS = [
  { 
    id: 'HealthDataAnalysisAgent', 
    name: 'Health Data Analysis', 
    description: 'Analyzing Apple Health metrics from HealthKit',
    icon: '🏥'
  },
  {
    id: 'AppleHealthRiskAgent',
    name: 'Apple Health Risk Assessment',
    description: 'Calculating HKRS score from wellness data',
    icon: '🍎'
  },
  { 
    id: 'CommunicationAgent', 
    name: 'Decision Communication', 
    description: 'Generating underwriter and customer messages',
    icon: '💬'
  },
];

interface AgentProgressTrackerProps {
  progress: AgentProgressEvent[];
  className?: string;
  compact?: boolean;
  showTitle?: boolean;
  isAppleHealth?: boolean;
}

/**
 * AgentProgressTracker - Displays real-time agent execution progress
 * 
 * Reusable component showing:
 * - Visual progress for each agent in the workflow
 * - Current execution stage with status
 * - Execution time when completed
 * - Tools being used
 * - Output previews
 */
export default function AgentProgressTracker({
  progress,
  className,
  compact = false,
  showTitle = true,
  isAppleHealth = false,
}: AgentProgressTrackerProps) {
  // Select the appropriate agent definitions based on workflow type
  const AGENT_DEFINITIONS = isAppleHealth ? APPLE_HEALTH_AGENT_DEFINITIONS : ADMIN_AGENT_DEFINITIONS;

  const getAgentStatus = (agentId: string): AgentProgressEvent | undefined => {
    // Get the latest progress event for this agent
    const events = progress.filter(p => p.agent_id === agentId);
    return events.length > 0 ? events[events.length - 1] : undefined;
  };

  // Get user-friendly stage description
  const getStageDescription = (stage?: AgentProgressStage, safeSummary?: string): string => {
    if (safeSummary) return safeSummary;
    switch (stage) {
      case 'started': return 'Starting agent...';
      case 'preparing_input': return 'Preparing input data...';
      case 'invoking_model': return 'Calling AI model...';
      case 'tool_called': return 'Using tools...';
      case 'parsing_response': return 'Processing response...';
      case 'validating_output': return 'Validating results...';
      case 'completed': return 'Completed';
      case 'failed': return 'Failed';
      default: return 'Processing...';
    }
  };

  // Determine if status is "active" (running, tool_calling, processing, output_ready)
  const isActiveStatus = (status: string): boolean => {
    return ['running', 'tool_calling', 'processing', 'output_ready'].includes(status);
  };

  // Get the overall workflow status
  const getOverallStatus = () => {
    const statuses = AGENT_DEFINITIONS.map(agent => getAgentStatus(agent.id)?.status || 'pending');
    if (statuses.some(s => s === 'failed')) return 'failed';
    if (statuses.every(s => s === 'completed')) return 'completed';
    if (statuses.some(s => isActiveStatus(s))) return 'running';
    return 'pending';
  };

  const overallStatus = getOverallStatus();

  // Compact mode - horizontal progress bar
  if (compact) {
    return (
      <div className={cn("flex items-center gap-2", className)}>
        {AGENT_DEFINITIONS.map((agent, index) => {
          const agentProgress = getAgentStatus(agent.id);
          const status = agentProgress?.status || 'pending';
          const isActive = isActiveStatus(status);
          
          return (
            <div key={agent.id} className="flex items-center">
              <div
                className={cn(
                  "w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-all",
                  isActive && "bg-blue-500 text-white ring-2 ring-blue-200 animate-pulse",
                  status === 'completed' && "bg-emerald-500 text-white",
                  status === 'failed' && "bg-red-500 text-white",
                  status === 'pending' && "bg-slate-200 text-slate-500"
                )}
                title={`${agent.name}: ${status}`}
              >
                {isActive ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : status === 'completed' ? (
                  <CheckCircle2 className="w-4 h-4" />
                ) : status === 'failed' ? (
                  <XCircle className="w-4 h-4" />
                ) : (
                  <span>{agent.icon}</span>
                )}
              </div>
              {index < AGENT_DEFINITIONS.length - 1 && (
                <div className={cn(
                  "w-6 h-0.5 mx-1 transition-colors",
                  status === 'completed' ? "bg-emerald-400" : "bg-slate-200"
                )} />
              )}
            </div>
          );
        })}
      </div>
    );
  }

  // Full mode - vertical list with details
  return (
    <div className={cn("bg-white rounded-xl border border-slate-200 overflow-hidden", className)}>
      {showTitle && (
        <div className="px-4 py-3 bg-slate-50 border-b border-slate-200">
          <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              {overallStatus === 'running' && (
                <>
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
                </>
              )}
              {overallStatus === 'completed' && (
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              )}
              {overallStatus === 'failed' && (
                <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
              )}
              {overallStatus === 'pending' && (
                <span className="relative inline-flex rounded-full h-2 w-2 bg-slate-400"></span>
              )}
            </span>
            Agent Execution Progress
          </h3>
        </div>
      )}
      
      <div className="p-4 space-y-3">
        {AGENT_DEFINITIONS.map((agent, index) => {
          const agentProgress = getAgentStatus(agent.id);
          const status = agentProgress?.status || 'pending';
          const executionTime = agentProgress?.execution_time_ms;
          const isActive = isActiveStatus(status);

          return (
            <div 
              key={agent.id}
              className={cn(
                "flex flex-col p-3 rounded-lg transition-all border",
                isActive && "bg-blue-50 border-blue-200 shadow-sm",
                status === 'completed' && "bg-emerald-50 border-emerald-200",
                status === 'failed' && "bg-red-50 border-red-200",
                status === 'pending' && "bg-slate-50 border-slate-200"
              )}
            >
              <div className="flex items-center gap-3">
                {/* Step Number */}
                <div className={cn(
                  "w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-all",
                  isActive && "bg-blue-500 text-white",
                  status === 'completed' && "bg-emerald-500 text-white",
                  status === 'failed' && "bg-red-500 text-white",
                  status === 'pending' && "bg-slate-300 text-slate-600"
                )}>
                  {index + 1}
                </div>

                {/* Status Icon */}
                <div className="w-5 flex-shrink-0">
                  {isActive && (
                    <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
                  )}
                  {status === 'completed' && (
                    <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                  )}
                  {status === 'failed' && (
                    <XCircle className="w-5 h-5 text-red-500" />
                  )}
                  {status === 'pending' && (
                    <Clock className="w-5 h-5 text-slate-400" />
                  )}
                </div>

                {/* Agent Name & Description */}
                <div className="flex-1 min-w-0">
                  <div className={cn(
                    "font-medium text-sm",
                    isActive && "text-blue-700",
                    status === 'completed' && "text-emerald-700",
                    status === 'failed' && "text-red-700",
                    status === 'pending' && "text-slate-500"
                  )}>
                    {agent.name}
                  </div>
                  {/* Stage description */}
                  {isActive && (
                    <div className="text-xs text-blue-600 mt-0.5">
                      {getStageDescription(agentProgress?.stage, agentProgress?.safe_summary)}
                    </div>
                  )}
                  {status === 'pending' && (
                    <div className="text-xs text-slate-400 mt-0.5">
                      {agent.description}
                    </div>
                  )}
                  {status === 'failed' && agentProgress?.message && (
                    <div className="text-xs text-red-600 mt-0.5">{agentProgress.message}</div>
                  )}
                </div>

                {/* Execution Time */}
                {executionTime !== undefined && status === 'completed' && (
                  <div className="text-xs text-emerald-600 font-mono">
                    {(executionTime / 1000).toFixed(1)}s
                  </div>
                )}
              </div>

              {/* Tools being used */}
              {isActive && agentProgress?.tools_used && agentProgress.tools_used.length > 0 && (
                <div className="mt-2 ml-11 pl-2 border-l-2 border-blue-200">
                  <div className="flex items-center gap-1 text-xs text-blue-500 font-medium">
                    <Wrench className="w-3 h-3" />
                    Tools:
                  </div>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {agentProgress.tools_used.map((tool, i) => (
                      <span key={i} className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                        {tool}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Output preview */}
              {agentProgress?.output_preview && status === 'completed' && (
                <div className="mt-2 ml-11 pl-2 border-l-2 border-emerald-200">
                  <div className="text-xs text-emerald-600 font-medium">Output:</div>
                  <div className="text-xs text-emerald-700 mt-0.5 bg-emerald-50 p-1.5 rounded line-clamp-2">
                    {agentProgress.output_preview}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Mini version for inline display
 */
export function AgentProgressMini({ progress }: { progress: AgentProgressEvent[] }) {
  const getAgentStatus = (agentId: string) => {
    const events = progress.filter(p => p.agent_id === agentId);
    return events.length > 0 ? events[events.length - 1]?.status || 'pending' : 'pending';
  };

  const isActiveStatus = (status: string): boolean => {
    return ['running', 'tool_calling', 'processing', 'output_ready'].includes(status);
  };

  return (
    <div className="flex items-center gap-1">
      {AGENT_DEFINITIONS.map((agent, index) => {
        const status = getAgentStatus(agent.id);
        const isActive = isActiveStatus(status);
        
        return (
          <div
            key={agent.id}
            className={cn(
              "w-2 h-2 rounded-full transition-all",
              isActive && "bg-blue-500 animate-pulse",
              status === 'completed' && "bg-emerald-500",
              status === 'failed' && "bg-red-500",
              status === 'pending' && "bg-slate-300"
            )}
            title={agent.name}
          />
        );
      })}
    </div>
  );
}
