'use client';

import { useState, useEffect } from 'react';
import { 
  User, 
  Heart, 
  Activity, 
  Moon, 
  Scale,
  FileText,
  PlayCircle,
  CheckCircle,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
  Shield,
  DollarSign,
  LogOut,
  Loader2,
  ArrowRight,
  AlertCircle,
  Clock
} from 'lucide-react';

interface EndUserDashboardProps {
  session: any;
  connectionResult?: any;
  onLogout: () => void;
}

interface RiskAnalysisResult {
  risk_level?: string;
  premium_adjustment?: {
    base_premium_annual?: number;
    adjustment_percentage?: number;
    adjusted_premium_annual?: number;
  };
  confidence_score?: number;
  key_risk_factors?: string[];
  explanation?: string;
}

export default function EndUserDashboard({ session, connectionResult, onLogout }: EndUserDashboardProps) {
  const [application, setApplication] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [analysisRunning, setAnalysisRunning] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<RiskAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progressMessages, setProgressMessages] = useState<string[]>([]);

  // Load application data
  useEffect(() => {
    const fetchApplication = async () => {
      try {
        const response = await fetch(`/api/end-user/application/${session.session_id}`);
        const data = await response.json();
        
        if (response.ok) {
          setApplication(data.application);
          
          // Check if risk analysis already exists
          if (data.application?.risk_analysis) {
            setAnalysisResult(data.application.risk_analysis);
          }
        }
      } catch (err) {
        console.error('Failed to load application:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchApplication();
  }, [session.session_id]);

  const runRiskAnalysis = async () => {
    setAnalysisRunning(true);
    setError(null);
    setProgressMessages([]);

    try {
      // Use SSE for real-time progress - connect directly to backend to avoid Next.js proxy buffering
      const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const eventSource = new EventSource(
        `${backendUrl}/api/end-user/run-risk-analysis-stream/${session.session_id}`
      );

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'progress') {
            setProgressMessages(prev => [...prev, data.data.message || `${data.data.agent_name}: ${data.data.status}`]);
          } else if (data.type === 'result') {
            setAnalysisResult(data.data.risk_analysis || data.data);
            eventSource.close();
            setAnalysisRunning(false);
          } else if (data.type === 'error') {
            setError(data.data.error || 'Analysis failed');
            eventSource.close();
            setAnalysisRunning(false);
          }
        } catch (parseErr) {
          console.error('Failed to parse SSE data:', parseErr);
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        // Fall back to non-streaming endpoint
        runRiskAnalysisNonStreaming();
      };

    } catch (err) {
      // Fall back to non-streaming
      runRiskAnalysisNonStreaming();
    }
  };

  const runRiskAnalysisNonStreaming = async () => {
    try {
      const response = await fetch(`/api/end-user/run-risk-analysis/${session.session_id}`, {
        method: 'POST',
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Analysis failed');
      }

      setAnalysisResult(data.risk_analysis);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setAnalysisRunning(false);
    }
  };

  const healthSummary = connectionResult?.health_summary || {
    bmi: application?.llm_outputs?.health_metrics?.activity?.bmi || 24.5,
    daily_steps_avg: 8000,
    resting_hr_avg: 68,
    sleep_hours_avg: 7.2,
  };

  const getRiskLevelColor = (level?: string) => {
    switch (level?.toLowerCase()) {
      case 'low': return 'text-green-600 bg-green-100';
      case 'moderate': return 'text-yellow-600 bg-yellow-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'very_high': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const formatCurrency = (amount?: number) => {
    if (!amount) return '$0';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="flex items-center gap-3">
          <Loader2 className="w-6 h-6 animate-spin text-indigo-600" />
          <span className="text-gray-600">Loading your dashboard...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top Navigation */}
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center gap-3">
              <Shield className="w-8 h-8 text-indigo-600" />
              <span className="text-xl font-semibold text-gray-900">Insurance Quote</span>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-right">
                <p className="text-sm font-medium text-gray-900">{session.profile.full_name}</p>
                <p className="text-xs text-gray-500">Age: {session.profile.age}</p>
              </div>
              <button
                onClick={onLogout}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                title="Logout"
              >
                <LogOut className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Demo Notice */}
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-6">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm">
              <p className="font-medium text-amber-800">Demo Mode — Synthetic Data</p>
              <p className="text-amber-700 mt-1">
                This is a demonstration using synthetic Apple Health data. Real insurance quotes require actual medical underwriting.
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Health Summary */}
          <div className="lg:col-span-2 space-y-6">
            {/* Application Info */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <FileText className="w-5 h-5 text-indigo-600" />
                <h2 className="text-lg font-semibold text-gray-900">Your Application</h2>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-gray-500">Application ID</p>
                  <p className="font-mono text-sm text-gray-900">{session.application_id || 'N/A'}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Policy Type</p>
                  <p className="font-medium text-gray-900 capitalize">
                    {(application?.llm_outputs?.patient_profile?.policy_type_requested || 'term_life').replace('_', ' ')}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Coverage Amount</p>
                  <p className="font-medium text-gray-900">
                    {formatCurrency(application?.llm_outputs?.patient_profile?.coverage_amount_requested)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Status</p>
                  <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                    analysisResult ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'
                  }`}>
                    {analysisResult ? 'Analyzed' : 'Ready for Analysis'}
                  </span>
                </div>
              </div>
            </div>

            {/* Health Metrics */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <Heart className="w-5 h-5 text-red-500" />
                <h2 className="text-lg font-semibold text-gray-900">Health Metrics (from Apple Health)</h2>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard
                  icon={Scale}
                  label="BMI"
                  value={healthSummary.bmi?.toFixed(1)}
                  status={healthSummary.bmi < 25 ? 'good' : healthSummary.bmi < 30 ? 'warning' : 'alert'}
                />
                <MetricCard
                  icon={Activity}
                  label="Daily Steps"
                  value={healthSummary.daily_steps_avg?.toLocaleString()}
                  status={healthSummary.daily_steps_avg >= 8000 ? 'good' : healthSummary.daily_steps_avg >= 5000 ? 'warning' : 'alert'}
                />
                <MetricCard
                  icon={Heart}
                  label="Resting HR"
                  value={`${healthSummary.resting_hr_avg} bpm`}
                  status={healthSummary.resting_hr_avg < 70 ? 'good' : healthSummary.resting_hr_avg < 85 ? 'warning' : 'alert'}
                />
                <MetricCard
                  icon={Moon}
                  label="Sleep"
                  value={`${healthSummary.sleep_hours_avg?.toFixed(1)}h`}
                  status={healthSummary.sleep_hours_avg >= 7 ? 'good' : healthSummary.sleep_hours_avg >= 6 ? 'warning' : 'alert'}
                />
              </div>
            </div>

            {/* Analysis Progress */}
            {analysisRunning && (
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Loader2 className="w-5 h-5 animate-spin text-indigo-600" />
                  <h2 className="text-lg font-semibold text-gray-900">Running Risk Analysis...</h2>
                </div>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {progressMessages.map((msg, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-sm">
                      <Clock className="w-4 h-4 text-gray-400" />
                      <span className="text-gray-600">{msg}</span>
                    </div>
                  ))}
                  {progressMessages.length === 0 && (
                    <p className="text-gray-500 text-sm">Initializing agent pipeline...</p>
                  )}
                </div>
              </div>
            )}

            {/* Analysis Results */}
            {analysisResult && !analysisRunning && (
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <div className="flex items-center gap-3 mb-4">
                  <CheckCircle className="w-5 h-5 text-green-600" />
                  <h2 className="text-lg font-semibold text-gray-900">Risk Analysis Complete</h2>
                </div>

                {/* Risk Level & Premium */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-500 mb-1">Risk Classification</p>
                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getRiskLevelColor(analysisResult.risk_level)}`}>
                      {analysisResult.risk_level?.replace('_', ' ').toUpperCase() || 'PENDING'}
                    </span>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-500 mb-1">Confidence Score</p>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-indigo-600 rounded-full"
                          style={{ width: `${(analysisResult.confidence_score || 0) * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900">
                        {((analysisResult.confidence_score || 0) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Key Risk Factors */}
                {analysisResult.key_risk_factors && analysisResult.key_risk_factors.length > 0 && (
                  <div className="mb-6">
                    <h3 className="text-sm font-medium text-gray-900 mb-2">Key Findings</h3>
                    <ul className="space-y-2">
                      {analysisResult.key_risk_factors.map((factor, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm text-gray-600">
                          <AlertTriangle className="w-4 h-4 text-yellow-500 flex-shrink-0 mt-0.5" />
                          {factor}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Explanation */}
                {analysisResult.explanation && (
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <p className="text-sm text-blue-800">{analysisResult.explanation}</p>
                  </div>
                )}

                {/* View Full Report Button */}
                {session.application_id && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <a
                      href={`/?app=${session.application_id}`}
                      className="w-full inline-flex items-center justify-center gap-2 py-2.5 px-4 bg-indigo-600 text-white font-medium rounded-lg hover:bg-indigo-700 transition-colors"
                    >
                      <FileText className="w-4 h-4" />
                      View Full Report in Admin Portal
                      <ArrowRight className="w-4 h-4" />
                    </a>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Right Column - Premium Estimate */}
          <div className="space-y-6">
            {/* Run Analysis Button */}
            {!analysisResult && !analysisRunning && (
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <div className="text-center">
                  <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <PlayCircle className="w-8 h-8 text-indigo-600" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Ready for Analysis</h3>
                  <p className="text-sm text-gray-500 mb-4">
                    Run our AI-powered risk analysis to get your personalized premium estimate.
                  </p>
                  <button
                    onClick={runRiskAnalysis}
                    className="w-full py-3 px-4 bg-indigo-600 text-white font-medium rounded-lg hover:bg-indigo-700 flex items-center justify-center gap-2 transition-colors"
                  >
                    <span>Run Risk Analysis</span>
                    <ArrowRight className="w-5 h-5" />
                  </button>
                </div>
              </div>
            )}

            {/* Premium Estimate */}
            {analysisResult?.premium_adjustment && (
              <div className="bg-gradient-to-br from-indigo-600 to-indigo-800 rounded-xl shadow-lg p-6 text-white">
                <div className="flex items-center gap-2 mb-4">
                  <DollarSign className="w-5 h-5" />
                  <h3 className="text-lg font-semibold">Premium Estimate</h3>
                </div>
                
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-indigo-200">Base Premium</span>
                    <span className="font-medium">
                      {formatCurrency(analysisResult.premium_adjustment.base_premium_annual)}/year
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-indigo-200">Adjustment</span>
                    <span className={`font-medium ${
                      (analysisResult.premium_adjustment.adjustment_percentage || 0) > 0 
                        ? 'text-yellow-300' 
                        : 'text-green-300'
                    }`}>
                      {(analysisResult.premium_adjustment.adjustment_percentage || 0) > 0 ? '+' : ''}
                      {analysisResult.premium_adjustment.adjustment_percentage}%
                    </span>
                  </div>
                  <div className="border-t border-indigo-500 pt-3">
                    <div className="flex justify-between items-center">
                      <span className="text-indigo-100 font-medium">Estimated Premium</span>
                      <span className="text-2xl font-bold">
                        {formatCurrency(analysisResult.premium_adjustment.adjusted_premium_annual)}/year
                      </span>
                    </div>
                  </div>
                </div>

                <p className="text-xs text-indigo-200 mt-4">
                  * Estimated premium impact (demo). Actual rates subject to full underwriting.
                </p>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-medium text-red-800">Analysis Failed</p>
                    <p className="text-sm text-red-700 mt-1">{error}</p>
                  </div>
                </div>
                <button
                  onClick={runRiskAnalysis}
                  className="mt-3 w-full py-2 px-4 bg-red-100 text-red-700 font-medium rounded-lg hover:bg-red-200 transition-colors"
                >
                  Try Again
                </button>
              </div>
            )}

            {/* Info Card */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="font-medium text-gray-900 mb-3">How it works</h3>
              <ul className="space-y-3 text-sm text-gray-600">
                <li className="flex items-start gap-2">
                  <span className="w-5 h-5 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0">1</span>
                  <span>Your health data is analyzed by our AI agents</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="w-5 h-5 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0">2</span>
                  <span>Multiple risk factors are evaluated against underwriting policies</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="w-5 h-5 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0">3</span>
                  <span>A personalized premium estimate is generated</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper component for metric cards
function MetricCard({ 
  icon: Icon, 
  label, 
  value, 
  status 
}: { 
  icon: any; 
  label: string; 
  value: string; 
  status: 'good' | 'warning' | 'alert';
}) {
  const statusColors = {
    good: 'text-green-600 bg-green-100',
    warning: 'text-yellow-600 bg-yellow-100',
    alert: 'text-red-600 bg-red-100',
  };

  const StatusIcon = status === 'good' ? TrendingUp : status === 'alert' ? TrendingDown : Minus;

  return (
    <div className="p-4 bg-gray-50 rounded-lg">
      <div className="flex items-center justify-between mb-2">
        <Icon className="w-5 h-5 text-gray-400" />
        <div className={`p-1 rounded ${statusColors[status]}`}>
          <StatusIcon className="w-3 h-3" />
        </div>
      </div>
      <p className="text-2xl font-semibold text-gray-900">{value}</p>
      <p className="text-sm text-gray-500">{label}</p>
    </div>
  );
}
