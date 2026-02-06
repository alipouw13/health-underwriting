'use client';

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { Sparkles, FileText, User, Shield, LogOut, Check, Smartphone } from 'lucide-react';
import TopNav from '@/components/TopNav';
import PatientHeader from '@/components/PatientHeader';
import PatientSummary from '@/components/PatientSummary';
import LabResultsPanel from '@/components/LabResultsPanel';
import SubstanceUsePanel from '@/components/SubstanceUsePanel';
import FamilyHistoryPanel from '@/components/FamilyHistoryPanel';
import AllergiesPanel from '@/components/AllergiesPanel';
import OccupationPanel from '@/components/OccupationPanel';
import ChronologicalOverview from '@/components/ChronologicalOverview';
import DocumentsPanel from '@/components/DocumentsPanel';
import SourcePagesPanel from '@/components/SourcePagesPanel';
import LoadingSpinner from '@/components/LoadingSpinner';
import PolicySummaryPanel from '@/components/PolicySummaryPanel';
import PolicyReportModal from '@/components/PolicyReportModal';
import ChatDrawer from '@/components/ChatDrawer';
import AppleHealthDataPanel from '@/components/AppleHealthDataPanel';
import RiskClassificationsPanel from '@/components/RiskClassificationsPanel';
import { ClaimsSummary, MedicalRecordsPanel, EligibilityPanel } from '@/components/claims';
import LifeHealthClaimsOverview from '@/components/claims/LifeHealthClaimsOverview';
import PropertyCasualtyClaimsOverview from '@/components/claims/PropertyCasualtyClaimsOverview';
import AutomotiveClaimsOverview from '@/components/claims/AutomotiveClaimsOverview';
import { usePersona } from '@/lib/PersonaContext';
import type { ApplicationMetadata, ApplicationListItem } from '@/lib/types';

type ViewType = 'overview' | 'timeline' | 'documents' | 'source' | 'risk-classes';

// Applicant session from sessionStorage
interface ApplicantSession {
  session_id: string;
  user_id: string;
  profile: {
    first_name: string;
    last_name: string;
    full_name: string;
    age: number;
  };
  application_id: string;
  apple_health_connected: boolean;
}

export default function Home() {
  const searchParams = useSearchParams();
  const [applications, setApplications] = useState<ApplicationListItem[]>([]);
  const [selectedApp, setSelectedApp] = useState<ApplicationMetadata | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeView, setActiveView] = useState<ViewType>('overview');
  const [isPolicyReportOpen, setIsPolicyReportOpen] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const { currentPersona, personaConfig } = usePersona();
  
  // Applicant mode state
  const [isApplicantMode, setIsApplicantMode] = useState(false);
  const [applicantSession, setApplicantSession] = useState<ApplicantSession | null>(null);

  // Check for applicant mode on mount
  useEffect(() => {
    const applicantParam = searchParams.get('applicant');
    const appParam = searchParams.get('app');
    
    if (applicantParam === 'true' && appParam) {
      setIsApplicantMode(true);
      // Try to load session from sessionStorage
      const sessionData = sessionStorage.getItem('applicantSession');
      if (sessionData) {
        try {
          setApplicantSession(JSON.parse(sessionData));
        } catch (e) {
          console.error('Failed to parse applicant session:', e);
        }
      }
    }
  }, [searchParams]);

  // Load applications list - reload when persona changes
  useEffect(() => {
    async function fetchApplications() {
      // If in applicant mode with specific app, just load that app
      const appParam = searchParams.get('app');
      if (isApplicantMode && appParam) {
        loadApplication(appParam);
        return;
      }
      
      try {
        setLoading(true);
        setError(null);
        setSelectedApp(null); // Clear selection when switching personas
        setApplications([]); // Clear applications immediately to prevent stale data
        console.log('Loading applications for persona:', currentPersona);
        const response = await fetch(`/api/applications?persona=${currentPersona}`, {
          cache: 'no-store',
          headers: {
            'Cache-Control': 'no-cache',
          },
        });
        if (response.ok) {
          const apps = await response.json();
          console.log('Loaded applications:', apps.length, 'for persona:', currentPersona, apps.map((a: any) => ({ id: a.id, persona: a.persona })));
          setApplications(apps);
          // Select the first completed app if available
          if (apps.length > 0) {
            const completedApp = apps.find((a: ApplicationListItem) => a.status === 'completed') || apps[0];
            loadApplication(completedApp.id);
          } else {
            setLoading(false);
          }
        } else {
          setError('Failed to load applications');
          setApplications([]);
          setLoading(false);
        }
      } catch (err) {
        console.error('Failed to load applications:', err);
        setError('Failed to connect to API server');
        setApplications([]);
        setLoading(false);
      }
    }
    
    fetchApplications();
  }, [currentPersona, isApplicantMode, searchParams]);

  async function loadApplications() {
    // This function is now just for manual refresh
    try {
      setLoading(true);
      setError(null);
      setSelectedApp(null);
      setApplications([]);
      console.log('Loading applications for persona:', currentPersona);
      const response = await fetch(`/api/applications?persona=${currentPersona}`, {
        cache: 'no-store',
        headers: {
          'Cache-Control': 'no-cache',
        },
      });
      if (response.ok) {
        const apps = await response.json();
        console.log('Loaded applications:', apps.length, 'for persona:', currentPersona, apps.map((a: any) => ({ id: a.id, persona: a.persona })));
        setApplications(apps);
        if (apps.length > 0) {
          const completedApp = apps.find((a: ApplicationListItem) => a.status === 'completed') || apps[0];
          loadApplication(completedApp.id);
        } else {
          setLoading(false);
        }
      } else {
        setError('Failed to load applications');
        setApplications([]);
        setLoading(false);
      }
    } catch (err) {
      console.error('Failed to load applications:', err);
      setError('Failed to connect to API server');
      setApplications([]);
      setLoading(false);
    }
  }

  async function loadApplication(appId: string) {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`/api/applications/${appId}`);
      if (response.ok) {
        const app = await response.json();
        setSelectedApp(app);
        // Reset to overview when selecting a new application
        setActiveView('overview');
      } else {
        setError('Failed to load application details');
        setSelectedApp(null);
      }
    } catch (err) {
      console.error('Failed to load application:', err);
      setError('Failed to load application details');
      setSelectedApp(null);
    } finally {
      setLoading(false);
    }
  }

  const renderMainContent = () => {
    if (!selectedApp) return null;

    switch (activeView) {
      case 'risk-classes':
        return (
          <div className="flex-1 overflow-auto p-6">
            <div className="max-w-7xl mx-auto">
              <RiskClassificationsPanel />
            </div>
          </div>
        );
      case 'timeline':
        return (
          <div className="flex-1 overflow-auto p-6">
            <ChronologicalOverview application={selectedApp} fullWidth />
          </div>
        );
      case 'documents':
        return (
          <div className="flex-1 overflow-auto p-6">
            <DocumentsPanel files={selectedApp.files || []} />
          </div>
        );
      case 'source':
        return (
          <div className="flex-1 overflow-auto p-6 h-full">
            <SourcePagesPanel pages={selectedApp.markdown_pages || []} />
          </div>
        );
      case 'overview':
      default:
        // Render persona-specific overview
        if (currentPersona === 'automotive_claims') {
          return renderAutomotiveClaimsOverview();
        }
        if (currentPersona === 'life_health_claims') {
          return renderLifeHealthClaimsOverview();
        }
        if (currentPersona === 'property_casualty_claims') {
          return renderPropertyCasualtyClaimsOverview();
        }
        if (currentPersona === 'mortgage') {
          return renderMortgageOverview();
        }
        // Default: Underwriting overview
        return renderUnderwritingOverview();
    }
  };

  const renderUnderwritingOverview = () => {
    if (!selectedApp) return null;
    
    // Check if this is an Apple Health application
    const isAppleHealthApp = selectedApp?.llm_outputs?.is_apple_health === true || 
                              selectedApp?.llm_outputs?.workflow_type === 'apple_health' ||
                              selectedApp?.llm_outputs?.source === 'end_user';
    
    const handleRerunAnalysis = async () => {
      if (!selectedApp) return;
      try {
        // Re-run risk analysis (separate from extraction)
        const response = await fetch(`/api/applications/${selectedApp.id}/risk-analysis`, {
          method: 'POST',
        });
        if (response.ok) {
          // Reload application to get updated analysis
          loadApplication(selectedApp.id);
        }
      } catch (err) {
        console.error('Failed to re-run risk analysis:', err);
      }
    };
    
    // Apple Health Application Layout - NO lab results, family history, substance use
    if (isAppleHealthApp) {
      return (
        <div className="flex-1 overflow-auto p-6">
          <div className="max-w-7xl mx-auto space-y-6">
            {/* Demo Notice for Apple Health */}
            <div className="bg-gradient-to-r from-red-50 to-pink-50 border border-red-200 rounded-xl p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-red-100 rounded-lg">
                  <Smartphone className="w-5 h-5 text-red-500" />
                </div>
                <div>
                  <p className="font-medium text-red-900">Apple Health Connected Application</p>
                  <p className="text-sm text-red-700">
                    This application uses HealthKit data for HKRS-based underwriting. No traditional medical records.
                  </p>
                </div>
              </div>
            </div>

            {/* Top Section: Patient Summary + Policy Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Patient Summary */}
              <PatientSummary 
                application={selectedApp} 
                onPolicyClick={(policyId) => {
                  setIsPolicyReportOpen(true);
                }}
              />
              
              {/* Policy Summary Panel (HKRS-based risk analysis) */}
              <PolicySummaryPanel
                application={selectedApp}
                onViewFullReport={() => setIsPolicyReportOpen(true)}
                onRiskAnalysisComplete={() => loadApplication(selectedApp.id)}
              />
            </div>

            {/* Section Divider */}
            <div className="flex items-center gap-4 py-2">
              <div className="flex-1 border-t border-slate-200" />
              <div className="flex items-center gap-2 text-xs font-medium text-slate-400 uppercase tracking-wider">
                <Smartphone className="w-4 h-4" />
                <span>Apple Health Data</span>
              </div>
              <div className="flex-1 border-t border-slate-200" />
            </div>

            {/* Apple Health Data Panel - All 7 Categories */}
            <AppleHealthDataPanel application={selectedApp} />
          </div>
          
          {/* Policy Report Modal */}
          <PolicyReportModal
            isOpen={isPolicyReportOpen}
            onClose={() => setIsPolicyReportOpen(false)}
            application={selectedApp}
            onRerunAnalysis={handleRerunAnalysis}
          />
        </div>
      );
    }
    
    // Traditional Admin Application Layout
    return (
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-7xl mx-auto space-y-6">
          {/* Top Section: AI Analysis + Chronological Overview side by side */}
          <div className="flex gap-6 items-stretch">
            {/* Left Column - AI Analysis */}
            <div className="flex-1 flex flex-col gap-6">
              {/* Patient Summary */}
              <PatientSummary 
                application={selectedApp} 
                onPolicyClick={(policyId) => {
                  setIsPolicyReportOpen(true);
                }}
              />
              
              {/* Policy Summary Panel (risk analysis) */}
              <PolicySummaryPanel
                application={selectedApp}
                onViewFullReport={() => setIsPolicyReportOpen(true)}
                onRiskAnalysisComplete={() => loadApplication(selectedApp.id)}
              />
            </div>

            {/* Right Column - Chronological Overview (matches height of left column) */}
            <div className="w-80 flex-shrink-0 flex flex-col">
              <div className="flex-1 overflow-y-auto">
                <ChronologicalOverview application={selectedApp} />
              </div>
            </div>
          </div>

          {/* Section Divider */}
          <div className="flex items-center gap-4 py-2">
            <div className="flex-1 border-t border-slate-200" />
            <div className="flex items-center gap-2 text-xs font-medium text-slate-400 uppercase tracking-wider">
              <FileText className="w-4 h-4" />
              <span>Evidence from Documents</span>
            </div>
            <div className="flex-1 border-t border-slate-200" />
          </div>

          {/* Evidence Section - Full Width */}
          <div className="grid grid-cols-3 gap-6">
            <LabResultsPanel application={selectedApp} />
            <SubstanceUsePanel application={selectedApp} />
            <FamilyHistoryPanel application={selectedApp} />
          </div>

          <div className="grid grid-cols-2 gap-6">
            <AllergiesPanel application={selectedApp} />
            <OccupationPanel application={selectedApp} />
          </div>
        </div>
        
        {/* Policy Report Modal */}
        <PolicyReportModal
          isOpen={isPolicyReportOpen}
          onClose={() => setIsPolicyReportOpen(false)}
          application={selectedApp}
          onRerunAnalysis={handleRerunAnalysis}
        />
      </div>
    );
  };

  const renderLifeHealthClaimsOverview = () => {
    if (!selectedApp) {
      return (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center text-slate-500">
            <p className="text-lg font-medium">No claims application selected</p>
            <p className="text-sm mt-2">Select a claim from the dropdown to view details</p>
          </div>
        </div>
      );
    }
    return (
      <LifeHealthClaimsOverview application={selectedApp} />
    );
  };

  const renderPropertyCasualtyClaimsOverview = () => {
    if (!selectedApp) {
      return (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center text-slate-500">
            <p className="text-lg font-medium">No claims application selected</p>
            <p className="text-sm mt-2">Select a claim from the dropdown to view details</p>
          </div>
        </div>
      );
    }
    return (
      <PropertyCasualtyClaimsOverview application={selectedApp} />
    );
  };

  const renderAutomotiveClaimsOverview = () => {
    if (!selectedApp) {
      return (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center text-slate-500">
            <p className="text-lg font-medium">No automotive claim selected</p>
            <p className="text-sm mt-2">Select a claim from the dropdown to view details</p>
          </div>
        </div>
      );
    }
    return (
      <AutomotiveClaimsOverview 
        applicationId={selectedApp.id}
      />
    );
  };

  const renderMortgageOverview = () => {
    return (
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-3xl mx-auto">
          <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 text-center">
            <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="text-3xl">🏠</span>
            </div>
            <h2 className="text-2xl font-semibold text-slate-900 mb-2">
              Mortgage Workbench
            </h2>
            <p className="text-slate-600 mb-6">
              The Mortgage underwriting workbench is coming soon. This workspace will help you 
              process loan applications, property documents, and borrower verification.
            </p>
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-indigo-100 text-indigo-700 rounded-lg">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-indigo-500"></span>
              </span>
              Coming Soon
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Handle applicant logout
  const handleApplicantLogout = () => {
    sessionStorage.removeItem('applicantSession');
    localStorage.removeItem('endUserSessionId');
    sessionStorage.removeItem('endUserFlowInProgress');
    window.location.href = '/user';
  };

  // Render applicant header with progress indicator
  const renderApplicantHeader = () => {
    if (!isApplicantMode || !applicantSession) return null;
    
    const profile = applicantSession.profile;
    const hasAnalysis = selectedApp?.risk_analysis?.parsed;
    
    return (
      <div className="bg-white border-b border-gray-200">
        {/* Top bar with logo and user info */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">InsureAI</h1>
                <p className="text-sm text-gray-500">Instant Life Insurance Quotes</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-600">
                Welcome, {profile.first_name}
              </span>
              <a
                href="/"
                className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors"
              >
                Admin Portal
              </a>
              <button
                onClick={handleApplicantLogout}
                className="inline-flex items-center gap-2 px-3 py-1.5 text-sm text-gray-500 hover:text-gray-700"
              >
                <LogOut className="w-4 h-4" />
                Sign out
              </button>
            </div>
          </div>
        </div>
        
        {/* Progress indicator */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 border-t border-gray-100">
          <div className="flex items-center justify-center gap-4">
            <div className="flex items-center gap-2 text-gray-400">
              <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium bg-green-500 text-white">
                <Check className="w-4 h-4" />
              </div>
              <span className="text-sm font-medium">Your Info</span>
            </div>
            
            <div className="w-12 h-px bg-gray-300"></div>
            
            <div className="flex items-center gap-2 text-gray-400">
              <div className="w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium bg-green-500 text-white">
                <Check className="w-4 h-4" />
              </div>
              <span className="text-sm font-medium">Health Data</span>
            </div>
            
            <div className="w-12 h-px bg-gray-300"></div>
            
            <div className={`flex items-center gap-2 ${hasAnalysis ? 'text-gray-400' : 'text-blue-600'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                hasAnalysis ? 'bg-green-500 text-white' : 'bg-blue-600 text-white'
              }`}>
                {hasAnalysis ? <Check className="w-4 h-4" /> : '3'}
              </div>
              <span className="text-sm font-medium">Your Quote</span>
            </div>
          </div>
        </div>
        
        {/* Page title with user name and age */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 border-t border-gray-100 bg-slate-50">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="w-6 h-6 text-indigo-500" />
              <h2 className="text-lg font-semibold text-gray-900">Insurance Quote</h2>
            </div>
            <div className="flex items-center gap-2 text-right">
              <span className="text-sm font-medium text-gray-900">{profile.full_name}</span>
              <span className="text-sm text-gray-500">Age: {profile.age}</span>
              <LogOut className="w-4 h-4 text-gray-400 cursor-pointer" onClick={handleApplicantLogout} />
            </div>
          </div>
        </div>
        
        {/* Demo notice */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3">
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-3">
            <div className="flex items-start gap-2">
              <span className="text-amber-600 mt-0.5">ℹ️</span>
              <div className="text-sm">
                <p className="font-medium text-amber-800">Demo Mode — Synthetic Data</p>
                <p className="text-amber-700">
                  This is a demonstration using synthetic Apple Health data. Real insurance quotes require actual medical underwriting.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Show TopNav only when NOT in applicant mode */}
      {!isApplicantMode && (
        <TopNav
          applications={applications}
          selectedAppId={selectedApp?.id}
          selectedApp={selectedApp || undefined}
          activeView={activeView}
          onSelectApp={loadApplication}
          onChangeView={setActiveView}
          isApplicantMode={isApplicantMode}
        />
      )}
      
      {/* Show Applicant Header when in applicant mode */}
      {isApplicantMode && renderApplicantHeader()}

      {/* Main Content */}
      <main className="flex flex-col" style={{ minHeight: 'calc(100vh - 120px)' }}>
        {loading ? (
          <div className="flex-1 flex items-center justify-center">
            <LoadingSpinner />
          </div>
        ) : error ? (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <p className="text-rose-500 mb-2">{error}</p>
              <p className="text-slate-500 text-sm">
                Make sure the API server is running on port 8000
              </p>
              <button
                onClick={() => loadApplications()}
                className="mt-4 px-4 py-2 bg-indigo-500 text-white rounded-lg hover:bg-indigo-600 transition-colors"
              >
                Retry
              </button>
            </div>
          </div>
        ) : selectedApp ? (
          <>
            {/* Patient Header - only for underwriting */}
            {currentPersona === 'underwriting' && <PatientHeader application={selectedApp} />}

            {/* Main Content Area based on active view */}
            {renderMainContent()}
          </>
        ) : currentPersona === 'mortgage' ? (
          // Mortgage "Coming Soon" view
          renderMortgageOverview()
        ) : applications.length === 0 ? (
          <div className="flex-1 flex items-center justify-center text-slate-500">
            <div className="text-center">
              <div 
                className="w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4"
                style={{ backgroundColor: `${personaConfig.color}15` }}
              >
                <personaConfig.icon className="w-8 h-8" style={{ color: personaConfig.color }} />
              </div>
              <p className="text-lg mb-2 text-slate-700">No {personaConfig.name.toLowerCase()} applications found</p>
              <p className="text-sm text-slate-500 mb-6">Go to the Admin page to upload and process documents</p>
              <div className="flex items-center justify-center gap-4">
                <a
                  href="/admin"
                  className="px-4 py-2 text-white rounded-lg transition-colors"
                  style={{ backgroundColor: personaConfig.color }}
                >
                  Go to Admin
                </a>
                <Link
                  href="/user"
                  className="flex items-center gap-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
                >
                  <User className="w-4 h-4" />
                  Try Applicant Portal
                </Link>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-500">
            <p>Select an application from the dropdown to view details</p>
          </div>
        )}
      </main>

      {/* Chat Drawer - Always mounted at root for smooth animations */}
      {selectedApp && (
        <ChatDrawer
          isOpen={isChatOpen}
          onClose={() => setIsChatOpen(false)}
          onOpen={() => setIsChatOpen(true)}
          applicationId={selectedApp.id}
          persona={currentPersona}
        />
      )}
    </div>
  );
}
