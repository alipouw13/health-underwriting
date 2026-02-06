'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import { 
  ArrowLeft, 
  Play, 
  Loader2, 
  Users,
  AlertCircle,
  FileText,
  RefreshCw,
} from 'lucide-react';
import AgentTransparencyView from '@/components/agents/AgentTransparencyView';
import AgentProgressTracker, { type AgentProgressEvent } from '@/components/agents/AgentProgressTracker';
import type { OrchestratorOutput } from '@/lib/agentTypes';

interface Application {
  id: string;
  created_at: string;
  persona: string;
  status: string;
  llm_outputs: {
    is_apple_health?: boolean;
    workflow_type?: string;
    [key: string]: unknown;
  } | null;
  agent_execution: {
    workflow_id: string;
    orchestrator_output: OrchestratorOutput;
  } | null;
}

export default function AgentsPage() {
  const [applications, setApplications] = useState<Application[]>([]);
  const [selectedAppId, setSelectedAppId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMode, setLoadingMode] = useState<'demo' | 'foundry' | null>(null);
  const [loadingApps, setLoadingApps] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [orchestratorOutput, setOrchestratorOutput] = useState<OrchestratorOutput | null>(null);
  const [executionMode, setExecutionMode] = useState<'foundry' | 'demo' | null>(null);
  const [agentProgress, setAgentProgress] = useState<AgentProgressEvent[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Load applications with agent execution data
  useEffect(() => {
    const fetchApplications = async () => {
      try {
        const response = await fetch('/api/applications?persona=underwriting');
        if (!response.ok) throw new Error('Failed to load applications');
        const data = await response.json();
        
        // API returns array directly (not wrapped in { applications: [...] })
        // Filter to completed underwriting applications (they have been analyzed)
        const apps = Array.isArray(data) ? data : (data.applications || []);
        const underwritingApps = apps.filter((app: Application) => 
          app.persona === 'underwriting' && app.status === 'completed'
        );
        
        setApplications(underwritingApps);
        
        // If we have apps, fetch details for the first one to check for agent_execution
        if (underwritingApps.length > 0) {
          // Fetch full details for apps to check agent_execution
          const detailsPromises = underwritingApps.slice(0, 5).map(async (app: Application) => {
            try {
              const detailRes = await fetch(`/api/applications/${app.id}`);
              if (detailRes.ok) {
                return await detailRes.json();
              }
            } catch {
              // Ignore errors for individual apps
            }
            return null;
          });
          
          const details = await Promise.all(detailsPromises);
          const appsWithDetails = underwritingApps.map((app: Application, idx: number) => {
            if (idx < 5 && details[idx]) {
              return { ...app, ...details[idx] };
            }
            return app;
          });
          
          setApplications(appsWithDetails);
          
          // Auto-select first app with agent_execution, or first app
          const appWithExecution = appsWithDetails.find((app: Application) => app.agent_execution);
          if (appWithExecution) {
            setSelectedAppId(appWithExecution.id);
            setOrchestratorOutput(appWithExecution.agent_execution?.orchestrator_output || null);
          } else {
            setSelectedAppId(underwritingApps[0].id);
          }
        }
      } catch (err) {
        console.error('Failed to load applications:', err);
      } finally {
        setLoadingApps(false);
      }
    };

    fetchApplications();
  }, []);

  // Cleanup EventSource on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const runOrchestration = async (useDemo: boolean = false) => {
    if (!selectedAppId) {
      setError('Please select an application first');
      return;
    }

    // Close any existing EventSource
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setLoading(true);
    setLoadingMode(useDemo ? 'demo' : 'foundry');
    setError(null);
    setOrchestratorOutput(null);
    setExecutionMode(useDemo ? 'demo' : 'foundry');
    setAgentProgress([]);
    setIsStreaming(true);

    // Use the streaming endpoint with SSE
    // Connect directly to backend (port 8000) to avoid Next.js proxy buffering SSE events
    const backendUrl = `http://localhost:8000/api/applications/${selectedAppId}/risk-analysis-stream?use_demo=${useDemo}`;
    
    console.log('[SSE] Connecting to:', backendUrl);
    const eventSource = new EventSource(backendUrl);
    eventSourceRef.current = eventSource;

    // Debug: log when connection opens
    eventSource.onopen = () => {
      console.log('[SSE] Connection opened');
    };

    // Handle progress events
    eventSource.addEventListener('progress', (event) => {
      console.log('Progress event received:', event.data);
      try {
        const data: AgentProgressEvent = JSON.parse(event.data);
        setAgentProgress(prev => [...prev, data]);
      } catch (err) {
        console.error('Failed to parse progress event:', err);
      }
    });

    // Handle result events
    eventSource.addEventListener('result', async (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Fetch the full orchestrator output
        const appResponse = await fetch(`/api/applications/${selectedAppId}`);
        if (appResponse.ok) {
          const appData = await appResponse.json();
          if (appData.agent_execution?.orchestrator_output) {
            setOrchestratorOutput(appData.agent_execution.orchestrator_output);
          }
        }

        // Update the applications list
        setApplications(prev => 
          prev.map(app => 
            app.id === selectedAppId 
              ? { ...app, agent_execution: { workflow_id: data.workflow_id, orchestrator_output: data } }
              : app
          )
        );
      } catch (err) {
        console.error('Failed to parse result event:', err);
      } finally {
        eventSource.close();
        setLoading(false);
        setLoadingMode(null);
        setIsStreaming(false);
      }
    });

    // Handle server-sent error events (custom error event from our API)
    eventSource.addEventListener('error', (event: Event) => {
      console.log('SSE error event received:', event);
      // Only handle MessageEvent (server-sent data), not connection errors
      if (event instanceof MessageEvent && event.data) {
        try {
          const data = JSON.parse(event.data);
          setError(data.error || 'Unknown error occurred');
          eventSource.close();
          setLoading(false);
          setLoadingMode(null);
          setIsStreaming(false);
        } catch {
          // Not JSON, ignore
        }
      }
    });

    // Track if we got a result
    let gotResult = false;
    
    // Handle result event to set flag
    eventSource.addEventListener('result', () => {
      gotResult = true;
    });

    eventSource.onerror = (event) => {
      console.log('SSE onerror fired, readyState:', eventSource.readyState);
      // ReadyState 2 = CLOSED - this happens after the stream ends normally
      // Only error if we didn't get a result and the connection failed
      if (eventSource.readyState === EventSource.CLOSED) {
        // Give time for result event to be processed
        setTimeout(() => {
          if (!gotResult && loading) {
            console.log('No result received, setting error');
            setError('Connection closed without result');
            setLoading(false);
            setLoadingMode(null);
            setIsStreaming(false);
          }
        }, 500);
      }
      eventSource.close();
    };
  };

  // Load existing agent execution when selection changes
  const handleSelectApp = async (appId: string) => {
    setSelectedAppId(appId);
    setOrchestratorOutput(null);
    setAgentProgress([]);
    
    // Check if this app already has agent execution data
    const app = applications.find(a => a.id === appId);
    if (app?.agent_execution?.orchestrator_output) {
      setOrchestratorOutput(app.agent_execution.orchestrator_output);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      {/* Header */}
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link 
                href="/"
                className="flex items-center gap-2 text-slate-600 hover:text-slate-900 transition-colors"
              >
                <ArrowLeft className="w-4 h-4" />
                <span className="text-sm">Back to Dashboard</span>
              </Link>
              <div className="h-6 w-px bg-slate-200" />
              <div className="flex items-center gap-2">
                <Users className="w-5 h-5 text-indigo-600" />
                <h1 className="text-lg font-bold text-slate-900">Agent Insights</h1>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Control Panel */}
        <div className="bg-white rounded-xl border border-slate-200 p-6 mb-8 shadow-sm">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500 mb-4">
            Run Multi-Agent Orchestration on Real Application Data
          </h2>
          
          <div className="flex flex-col sm:flex-row gap-4">
            {/* Application Selection */}
            <div className="flex-1">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Select Application
              </label>
              
              {loadingApps ? (
                <div className="flex items-center gap-2 text-slate-500">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Loading applications...
                </div>
              ) : applications.length === 0 ? (
                <div className="text-sm text-slate-500 bg-slate-50 p-3 rounded-lg">
                  No underwriting applications found. Upload and analyze a document first.
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {applications.map((app) => (
                    <button
                      key={app.id}
                      onClick={() => handleSelectApp(app.id)}
                      className={`p-3 rounded-lg border text-left transition-all ${
                        selectedAppId === app.id
                          ? 'border-indigo-500 bg-indigo-50 ring-2 ring-indigo-200'
                          : 'border-slate-200 hover:border-slate-300 bg-white'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4 text-slate-400" />
                        <span className="font-medium text-sm text-slate-900 truncate">
                          {app.id.slice(0, 8)}
                        </span>
                        {app.agent_execution && (
                          <span className="ml-auto text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">
                            Analyzed
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-slate-500 mt-1">
                        {new Date(app.created_at).toLocaleDateString()}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Run Buttons */}
            <div className="flex items-end gap-3">
              {/* Demo Mode Button */}
              <button
                onClick={() => runOrchestration(true)}
                disabled={loading || !selectedAppId}
                className={`
                  flex items-center gap-2 px-5 py-3 rounded-lg font-medium transition-all
                  ${loading || !selectedAppId
                    ? 'bg-slate-100 text-slate-400 cursor-not-allowed' 
                    : 'bg-amber-500 text-white hover:bg-amber-600 shadow-lg shadow-amber-200'
                  }
                `}
                title="Run with local deterministic agents (fast, mock data)"
              >
                {loadingMode === 'demo' ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Demo Mode
                  </>
                )}
              </button>
              
              {/* Foundry Mode Button */}
              <button
                onClick={() => runOrchestration(false)}
                disabled={loading || !selectedAppId}
                className={`
                  flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-all
                  ${loading || !selectedAppId
                    ? 'bg-slate-100 text-slate-400 cursor-not-allowed' 
                    : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg shadow-indigo-200'
                  }
                `}
                title="Run with Azure AI Foundry agents (real LLM calls)"
              >
                {loadingMode === 'foundry' ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Running...
                  </>
                ) : orchestratorOutput ? (
                  <>
                    <RefreshCw className="w-5 h-5" />
                    Re-Run Orchestration
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" />
                    Run Orchestration
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-8 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="font-medium text-red-800">Orchestration Error</h3>
              <p className="text-sm text-red-600 mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Live Progress Tracker - show during loading even if no progress yet */}
        {(isStreaming || loading) && (
          <AgentProgressTracker 
            progress={agentProgress} 
            isAppleHealth={applications.find(a => a.id === selectedAppId)?.llm_outputs?.is_apple_health === true}
          />
        )}

        {/* Results */}
        {orchestratorOutput ? (
          <div>
            {/* Execution Mode Badge */}
            {executionMode && (
              <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium mb-4 ${
                executionMode === 'demo' 
                  ? 'bg-amber-100 text-amber-800' 
                  : 'bg-indigo-100 text-indigo-800'
              }`}>
                {executionMode === 'demo' ? (
                  <>
                    <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                    Demo Mode - Local Deterministic Agents
                  </>
                ) : (
                  <>
                    <span className="w-2 h-2 rounded-full bg-indigo-500"></span>
                    Azure AI Foundry - Real LLM Agents
                  </>
                )}
              </div>
            )}
            <AgentTransparencyView 
              orchestratorOutput={orchestratorOutput}
              defaultExpanded={true}
            />
          </div>
        ) : !loading && !isStreaming && !error && (
          <div className="text-center py-16 bg-slate-50 rounded-xl border border-slate-200">
            <Users className="w-12 h-12 text-slate-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-slate-600">No Orchestration Results</h3>
            <p className="text-sm text-slate-500 mt-2">
              {selectedAppId 
                ? 'Click "Run Orchestration" to analyze this application with multi-agent workflow'
                : 'Select an application to run multi-agent analysis'
              }
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
