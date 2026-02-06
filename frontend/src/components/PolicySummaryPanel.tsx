'use client';

import { useState, useRef, useEffect } from 'react';
import { Shield, FileText, AlertTriangle, CheckCircle, Clock, Play, Loader2, Sparkles, Users, Bot } from 'lucide-react';
import type { ApplicationMetadata, RiskFinding } from '@/lib/types';
import AgentProgressTracker, { type AgentProgressEvent } from './agents/AgentProgressTracker';
import AgentInsightsModal from './agents/AgentInsightsModal';

interface PolicySummaryPanelProps {
  application: ApplicationMetadata;
  onViewFullReport: () => void;
  onRiskAnalysisComplete?: () => void;
}

function getRiskLevelInfo(rating: string): { icon: React.ReactNode; bgColor: string; textColor: string; borderColor: string; label: string } {
  const lowerRating = (rating || '').toLowerCase();
  if (lowerRating.includes('high')) {
    return {
      icon: <AlertTriangle className="w-5 h-5" />,
      bgColor: 'bg-rose-50',
      textColor: 'text-rose-700',
      borderColor: 'border-rose-200',
      label: 'High Risk',
    };
  }
  if (lowerRating.includes('moderate')) {
    return {
      icon: <Clock className="w-5 h-5" />,
      bgColor: 'bg-amber-50',
      textColor: 'text-amber-700',
      borderColor: 'border-amber-200',
      label: 'Moderate Risk',
    };
  }
  if (lowerRating.includes('low')) {
    return {
      icon: <CheckCircle className="w-5 h-5" />,
      bgColor: 'bg-emerald-50',
      textColor: 'text-emerald-700',
      borderColor: 'border-emerald-200',
      label: 'Low Risk',
    };
  }
  return {
    icon: <Shield className="w-5 h-5" />,
    bgColor: 'bg-slate-50',
    textColor: 'text-slate-600',
    borderColor: 'border-slate-200',
    label: 'Not Assessed',
  };
}

export default function PolicySummaryPanel({
  application,
  onViewFullReport,
  onRiskAnalysisComplete,
}: PolicySummaryPanelProps) {
  const [isRunningAnalysis, setIsRunningAnalysis] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [agentProgress, setAgentProgress] = useState<AgentProgressEvent[]>([]);
  const [showAgentInsights, setShowAgentInsights] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const riskAnalysis = application.risk_analysis?.parsed;
  const hasRiskAnalysis = !!riskAnalysis;

  // Check if agent execution is available
  const hasAgentExecution = !!application.agent_execution;

  // Detect Apple Health workflow - check multiple indicators
  const isAppleHealth = 
    application.llm_outputs?.is_apple_health === true ||
    application.llm_outputs?.workflow_type === 'apple_health' ||
    application.llm_outputs?.source === 'end_user' ||
    application.persona === 'end_user';

  // Cleanup EventSource on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const handleRunRiskAnalysis = async () => {
    // Close any existing EventSource
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setIsRunningAnalysis(true);
    setError(null);
    setAgentProgress([]);

    // Use the streaming endpoint with SSE for real-time progress
    // Connect directly to backend (port 8000) to avoid Next.js proxy buffering SSE events
    const url = `http://localhost:8000/api/applications/${application.id}/risk-analysis-stream`;
    
    console.log('[SSE] Connecting to:', url);
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    // Handle progress events - these come from each agent step
    eventSource.addEventListener('progress', (event) => {
      try {
        const data: AgentProgressEvent = JSON.parse(event.data);
        console.log('[AgentProgress] Received:', data.agent_id, data.status, data);
        setAgentProgress(prev => [...prev, data]);
      } catch (err) {
        console.error('[AgentProgress] Failed to parse progress event:', err);
      }
    });

    // Handle result event - orchestration completed
    eventSource.addEventListener('result', async () => {
      eventSource.close();
      setIsRunningAnalysis(false);
      setAgentProgress([]);
      
      // Trigger reload of application data
      if (onRiskAnalysisComplete) {
        onRiskAnalysisComplete();
      }
    });

    // Handle server-sent error events
    eventSource.addEventListener('error', (event: Event) => {
      if (event instanceof MessageEvent && event.data) {
        try {
          const data = JSON.parse(event.data);
          setError(data.error || 'Unknown error occurred');
          eventSource.close();
          setIsRunningAnalysis(false);
          setAgentProgress([]);
        } catch {
          // Not JSON, ignore
        }
      }
    });

    // Handle connection errors
    eventSource.onerror = (event) => {
      // Only treat as error if we haven't received a result
      if (eventSource.readyState === EventSource.CLOSED) {
        // Normal close after completion, ignore
        return;
      }
      console.error('SSE connection error:', event);
      setError('Connection lost. Please try again.');
      eventSource.close();
      setIsRunningAnalysis(false);
      setAgentProgress([]);
    };
  };

  // If no risk analysis, show the "Run Risk Analysis" prompt
  if (!hasRiskAnalysis) {
    return (
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="px-6 py-4 bg-slate-50 border-b border-slate-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-indigo-100 flex items-center justify-center text-indigo-600">
              <Shield className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-slate-900">
                Policy Risk Analysis
              </h2>
              <p className="text-sm text-slate-500">
                {isRunningAnalysis ? 'Multi-agent underwriting in progress' : 'Run multi-agent risk assessment'}
              </p>
            </div>
          </div>
        </div>

        <div className="px-6 py-6">
          {/* Show agent progress when running */}
          {isRunningAnalysis ? (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-sm text-blue-600 font-medium">
                <Users className="w-4 h-4" />
                <span>Multi-Agent Orchestration Running</span>
              </div>
              {/* Full detailed agent progress tracker */}
              <AgentProgressTracker 
                progress={agentProgress} 
                compact={false} 
                showTitle={false} 
                isAppleHealth={isAppleHealth}
              />
              {agentProgress.length > 0 ? (
                <div className="text-xs text-slate-500">
                  Step {Math.max(...agentProgress.map(p => p.step_number), 1)} of {isAppleHealth ? 3 : 4} agents processing...
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
                  <span className="text-xs text-slate-500">Initializing agents...</span>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center">
              <div className="w-16 h-16 rounded-full bg-indigo-100 flex items-center justify-center mx-auto mb-4">
                <FileText className="w-8 h-8 text-indigo-600" />
              </div>
              <h3 className="text-lg font-medium text-slate-900 mb-2">
                Risk Analysis Not Run
              </h3>
              <p className="text-sm text-slate-600 mb-6 max-w-sm mx-auto">
                Run a comprehensive multi-agent risk analysis to evaluate this application against underwriting guidelines.
              </p>
              
              {error && (
                <div className="mb-4 p-3 bg-rose-50 border border-rose-200 rounded-lg text-sm text-rose-700">
                  {error}
                </div>
              )}

              <button
                onClick={handleRunRiskAnalysis}
                disabled={isRunningAnalysis || application.status !== 'completed'}
                className="inline-flex items-center gap-2 px-6 py-3 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isRunningAnalysis ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Running Analysis...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" />
                    Run Risk Analysis
                  </>
                )}
              </button>

              {application.status !== 'completed' && (
                <p className="text-xs text-slate-500 mt-3">
                  Standard analysis must be completed first
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Risk analysis is available - show results
  const riskInfo = getRiskLevelInfo(riskAnalysis.overall_risk_level);
  const topFindings = (riskAnalysis.findings || []).slice(0, 3);
  const premium = riskAnalysis.premium_recommendation;
  
  // Get agent execution data if available
  const agentExecution = application.agent_execution;
  const orchestratorOutput = agentExecution?.orchestrator_output;

  return (
    <>
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
      {/* Header with Risk Rating */}
      <div className={`px-6 py-4 ${riskInfo.bgColor}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg ${riskInfo.bgColor} flex items-center justify-center ${riskInfo.textColor}`}>
              {riskInfo.icon}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-semibold text-slate-900">
                  Policy Risk Analysis
                </h2>
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-indigo-50 text-indigo-600 border border-indigo-100">
                  <Sparkles className="w-3 h-3" />
                  AI Analysis
                </span>
              </div>
              <div className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium ${riskInfo.bgColor} ${riskInfo.textColor} border ${riskInfo.borderColor}`}>
                {riskAnalysis.overall_risk_level || 'Unknown'}
              </div>
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-2xl font-bold text-slate-900">
              {(riskAnalysis.findings || []).length}
            </div>
            <div className="text-xs text-slate-500">Policy Findings</div>
          </div>
        </div>
      </div>

      {/* Show streaming progress when re-running analysis */}
      {isRunningAnalysis ? (
        <div className="px-6 py-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm text-blue-600 font-medium">
                <Users className="w-4 h-4" />
                <span>Multi-Agent Orchestration Running</span>
              </div>
            </div>
            {/* Full-width agent progress tracker */}
            <AgentProgressTracker 
              progress={agentProgress} 
              compact={false} 
              showTitle={false} 
              isAppleHealth={isAppleHealth}
            />
            {agentProgress.length > 0 ? (
              <div className="text-xs text-slate-500">
                Step {Math.max(...agentProgress.map(p => p.step_number), 1)} of {isAppleHealth ? 3 : 4} agents processing...
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
                <span className="text-xs text-slate-500">Initializing agents...</span>
              </div>
            )}
          </div>
        </div>
      ) : (
        <>
          {/* Overall Rationale */}
          {riskAnalysis.overall_rationale && (
            <div className="px-6 py-4 border-b border-slate-100">
              <div className="flex items-start gap-3">
                <Shield className="w-5 h-5 text-indigo-500 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-sm text-slate-700">
                    {riskAnalysis.overall_rationale}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Premium Recommendation */}
          {premium && (
        <div className="px-6 py-4 border-b border-slate-100 bg-slate-50">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">
            Premium Recommendation
          </h3>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-600">Decision:</span>
              <span className={`font-medium ${
                premium.base_decision === 'Standard' ? 'text-emerald-600' :
                premium.base_decision === 'Rated' ? 'text-amber-600' :
                premium.base_decision === 'Decline' ? 'text-rose-600' :
                'text-slate-700'
              }`}>
                {premium.base_decision}
              </span>
            </div>
            {premium.loading_percentage && premium.loading_percentage !== '0%' && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-slate-600">Loading:</span>
                <span className="font-medium text-amber-600">{premium.loading_percentage}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Top Findings */}
      <div className="px-6 py-4">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">
          Key Policy Findings
        </h3>
        
        {topFindings.length === 0 ? (
          <p className="text-sm text-slate-500 italic">
            No specific policy findings identified.
          </p>
        ) : (
          <div className="space-y-3">
            {topFindings.map((finding: RiskFinding, idx: number) => (
              <div
                key={idx}
                className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg"
              >
                <FileText className="w-4 h-4 text-indigo-500 mt-0.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-mono text-xs bg-indigo-100 text-indigo-700 px-1.5 py-0.5 rounded">
                      {finding.policy_id}
                    </span>
                    <span className="text-sm font-medium text-slate-700">
                      {finding.policy_name}
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 mt-1 line-clamp-2">
                    {finding.finding}
                  </p>
                </div>
                <span className={`text-xs font-medium flex-shrink-0 px-2 py-0.5 rounded ${
                  finding.risk_level?.toLowerCase().includes('high')
                    ? 'bg-rose-100 text-rose-700'
                    : finding.risk_level?.toLowerCase().includes('moderate')
                    ? 'bg-amber-100 text-amber-700'
                    : 'bg-emerald-100 text-emerald-700'
                }`}>
                  {finding.risk_level}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
        </>
      )}

      {/* Action Buttons */}
      <div className="px-6 py-4 bg-slate-50 border-t border-slate-100 flex items-center gap-3">
        <button
          onClick={onViewFullReport}
          className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-white bg-indigo-600 rounded-lg hover:bg-indigo-700 transition-colors"
        >
          <FileText className="w-4 h-4" />
          View Full Report
        </button>
        
        {/* Agent Insights Button - only shown after analysis is complete */}
        {orchestratorOutput && (
          <button
            onClick={() => setShowAgentInsights(true)}
            className="inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-indigo-700 bg-indigo-50 border border-indigo-200 rounded-lg hover:bg-indigo-100 transition-colors"
            title="View agent execution details"
          >
            <Bot className="w-4 h-4" />
            Agent Insights
          </button>
        )}
        
        <button
          onClick={handleRunRiskAnalysis}
          disabled={isRunningAnalysis}
          className="inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-slate-700 bg-white border border-slate-300 rounded-lg hover:bg-slate-50 transition-colors disabled:opacity-50"
          title="Re-run risk analysis"
        >
          {isRunningAnalysis ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
        </button>
      </div>
    </div>
    
    {/* Agent Insights Modal */}
    {orchestratorOutput && (
      <AgentInsightsModal
        isOpen={showAgentInsights}
        onClose={() => setShowAgentInsights(false)}
        orchestratorOutput={orchestratorOutput}
        applicationId={application.id}
      />
    )}
    </>
  );
}
