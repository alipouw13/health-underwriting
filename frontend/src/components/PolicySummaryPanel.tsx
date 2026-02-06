'use client';

import { useState, useRef, useEffect } from 'react';
import { Shield, FileText, AlertTriangle, CheckCircle, Clock, Play, Loader2, Sparkles, Users, Bot, Activity, Heart, Moon, PersonStanding, Footprints, Dumbbell } from 'lucide-react';
import type { ApplicationMetadata, RiskFinding } from '@/lib/types';
import AgentProgressTracker, { type AgentProgressEvent } from './agents/AgentProgressTracker';
import AgentInsightsModal from './agents/AgentInsightsModal';

// Apple Health category icons mapping
const CATEGORY_ICONS: Record<string, React.ReactNode> = {
  activity: <Footprints className="w-4 h-4" />,
  vo2_max: <Dumbbell className="w-4 h-4" />,
  heart_health: <Heart className="w-4 h-4" />,
  sleep_health: <Moon className="w-4 h-4" />,
  body_composition: <PersonStanding className="w-4 h-4" />,
  mobility: <Activity className="w-4 h-4" />,
};

// Human-readable category names
const CATEGORY_NAMES: Record<string, string> = {
  activity: 'Daily Activity',
  vo2_max: 'Cardio Fitness (VO₂ Max)',
  heart_health: 'Heart Health',
  sleep_health: 'Sleep Quality',
  body_composition: 'Body Composition',
  mobility: 'Mobility & Balance',
};

interface SubScore {
  name?: string;           // Backend uses 'name' for category
  category?: string;       // Also accept 'category' for compatibility
  raw_score: number;
  max_points?: number;     // Backend uses 'max_points'
  max_score?: number;      // Also accept 'max_score' for compatibility
  weight?: number;         // Backend uses 'weight' (0-1)
  weight_pct?: number;     // Also accept 'weight_pct' for compatibility
  weighted_score: number;
  notes?: string | string[]; // Backend uses array, UI may use string
  components?: Record<string, unknown>;
}

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
  
  // Extract Apple Health sub_scores from execution records
  const appleHealthSubScores: SubScore[] = (() => {
    if (!isAppleHealth || !orchestratorOutput?.execution_records) return [];
    
    const ahRecord = orchestratorOutput.execution_records.find(
      (r: { agent_id: string }) => r.agent_id === 'AppleHealthRiskAgent'
    );
    
    // Check both 'output' and 'actual_outputs' since the API may use either
    const recordOutput = ahRecord?.output || ahRecord?.actual_outputs;
    if (!recordOutput?.sub_scores) return [];
    return recordOutput.sub_scores;
  })();

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
              {isAppleHealth ? appleHealthSubScores.length || 6 : (riskAnalysis.findings || []).length}
            </div>
            <div className="text-xs text-slate-500">
              {isAppleHealth ? 'Health Categories' : 'Policy Findings'}
            </div>
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

      {/* Top Findings - Apple Health categories or standard findings */}
      <div className="px-6 py-4">
        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-3">
          {isAppleHealth ? 'Apple Health Assessment' : 'Key Policy Findings'}
        </h3>
        
        {isAppleHealth && appleHealthSubScores.length > 0 ? (
          // Apple Health Category Scores
          <div className="space-y-2">
            {appleHealthSubScores.map((score: SubScore, idx: number) => {
              // Handle both backend formats: 'name' or 'category', 'max_points' or 'max_score'
              const categoryKey = score.name?.replace('_score', '') || score.category || 'unknown';
              const maxPts = score.max_points || score.max_score || 25;
              const pct = maxPts > 0 ? (score.raw_score / maxPts) * 100 : 0;
              const weightPct = (score.weight_pct || (score.weight ? score.weight * 100 : 0));
              const isGood = pct >= 70;
              const isModerate = pct >= 40 && pct < 70;
              
              // Handle notes as array or string
              const notesText = Array.isArray(score.notes) 
                ? score.notes[0] || '' 
                : (score.notes || `${weightPct.toFixed(0)}% weight in HKRS calculation`);
              
              return (
                <div
                  key={idx}
                  className="flex items-start gap-3 p-3 bg-slate-50 rounded-lg"
                >
                  <div className={`mt-0.5 flex-shrink-0 ${
                    isGood ? 'text-emerald-500' : isModerate ? 'text-amber-500' : 'text-rose-500'
                  }`}>
                    {CATEGORY_ICONS[categoryKey] || <Activity className="w-4 h-4" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-sm font-medium text-slate-700">
                        {CATEGORY_NAMES[categoryKey] || categoryKey.replace(/_/g, ' ')}
                      </span>
                      <span className={`text-xs font-medium px-2 py-0.5 rounded ${
                        isGood ? 'bg-emerald-100 text-emerald-700' :
                        isModerate ? 'bg-amber-100 text-amber-700' :
                        'bg-rose-100 text-rose-700'
                      }`}>
                        {score.raw_score}/{maxPts} pts
                      </span>
                    </div>
                    <p className="text-xs text-slate-600 mt-1">
                      {notesText}
                    </p>
                    {/* Progress bar */}
                    <div className="mt-2 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all ${
                          isGood ? 'bg-emerald-500' : isModerate ? 'bg-amber-500' : 'bg-rose-500'
                        }`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : topFindings.length === 0 ? (
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
