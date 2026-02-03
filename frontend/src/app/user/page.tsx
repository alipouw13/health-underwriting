'use client';

import { useState, useEffect } from 'react';
import { EndUserLogin, AppleHealthConnect, EndUserDashboard } from '@/components/end-user';

/**
 * End-User Application Flow
 * 
 * This page provides a consumer-facing interface for life insurance applicants
 * to submit their health data (via mock Apple Health) and receive an instant
 * underwriting decision using the SAME agent pipeline as underwriters.
 * 
 * Flow:
 * 1. Login (demo auth with basic info)
 * 2. Connect Apple Health (simulated consent + mock data generation)
 * 3. Dashboard - view health data and run real risk analysis
 */

type FlowState = 'login' | 'connect' | 'dashboard';

interface EndUserProfile {
  first_name: string;
  last_name: string;
  full_name: string;
  date_of_birth: string;
  age: number;
  biological_sex: string;
}

interface EndUserSession {
  session_id: string;
  user_id: string;
  profile: EndUserProfile;
  flow_state: string;
  application_id?: string;
  apple_health_connected: boolean;
  risk_analysis_completed: boolean;
  created_at: string;
}

export default function EndUserPage() {
  const [flowState, setFlowState] = useState<FlowState>('login');
  const [session, setSession] = useState<EndUserSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Clear any existing session on mount to always start fresh with login
  // This ensures users see the login page first when launching the app
  useEffect(() => {
    const initSession = async () => {
      // Check if we should restore session (e.g., after page refresh during flow)
      const shouldRestore = sessionStorage.getItem('endUserFlowInProgress');
      const storedSessionId = localStorage.getItem('endUserSessionId');
      
      if (shouldRestore && storedSessionId) {
        try {
          const response = await fetch(`/api/end-user/session/${storedSessionId}`);
          if (response.ok) {
            const data = await response.json();
            const sessionData = data.session || data;
            setSession(sessionData);
            
            // Determine flow state from session
            if (sessionData.risk_analysis_completed || sessionData.analysis_complete) {
              setFlowState('dashboard');
            } else if (sessionData.apple_health_connected || sessionData.health_data_connected) {
              setFlowState('dashboard');
            } else if (sessionData.flow_state === 'health_connected' || sessionData.flow_state === 'application_created') {
              setFlowState('dashboard');
            } else if (sessionData.flow_state === 'logged_in') {
              setFlowState('connect');
            } else {
              setFlowState('login');
            }
          } else {
            // Session expired or invalid - clear and start fresh
            localStorage.removeItem('endUserSessionId');
            sessionStorage.removeItem('endUserFlowInProgress');
            setFlowState('login');
          }
        } catch (err) {
          console.error('Failed to restore session:', err);
          localStorage.removeItem('endUserSessionId');
          sessionStorage.removeItem('endUserFlowInProgress');
          setFlowState('login');
        }
      } else {
        // Always start fresh with login page
        localStorage.removeItem('endUserSessionId');
        sessionStorage.removeItem('endUserFlowInProgress');
        setFlowState('login');
      }
      
      setLoading(false);
    };

    initSession();
  }, []);

  const handleLogout = async () => {
    if (session) {
      try {
        await fetch(`/api/end-user/session/${session.session_id}`, {
          method: 'DELETE',
        });
      } catch (err) {
        console.error('Logout error:', err);
      }
    }
    
    // Clear all session data
    localStorage.removeItem('endUserSessionId');
    sessionStorage.removeItem('endUserFlowInProgress');
    setSession(null);
    setFlowState('login');
  };

  const handleBack = () => {
    if (flowState === 'connect') {
      setFlowState('login');
    } else if (flowState === 'dashboard' && !session?.risk_analysis_completed) {
      setFlowState('connect');
    }
  };

  // Loading state
  if (loading && flowState === 'login' && !session) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-semibold text-gray-900">InsureAI</h1>
                <p className="text-sm text-gray-500">Instant Life Insurance Quotes</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {session && (
                <span className="text-sm text-gray-600">
                  Welcome, {session.profile?.first_name || 'User'}
                </span>
              )}
              <a
                href="/"
                className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.343 3.94c.09-.542.56-.94 1.11-.94h1.093c.55 0 1.02.398 1.11.94l.149.894c.07.424.384.764.78.93.398.164.855.142 1.205-.108l.737-.527a1.125 1.125 0 011.45.12l.773.774c.39.389.44 1.002.12 1.45l-.527.737c-.25.35-.272.806-.107 1.204.165.397.505.71.93.78l.893.15c.543.09.94.56.94 1.109v1.094c0 .55-.397 1.02-.94 1.11l-.893.149c-.425.07-.765.383-.93.78-.165.398-.143.854.107 1.204l.527.738c.32.447.269 1.06-.12 1.45l-.774.773a1.125 1.125 0 01-1.449.12l-.738-.527c-.35-.25-.806-.272-1.203-.107-.397.165-.71.505-.781.929l-.149.894c-.09.542-.56.94-1.11.94h-1.094c-.55 0-1.019-.398-1.11-.94l-.148-.894c-.071-.424-.384-.764-.781-.93-.398-.164-.854-.142-1.204.108l-.738.527c-.447.32-1.06.269-1.45-.12l-.773-.774a1.125 1.125 0 01-.12-1.45l.527-.737c.25-.35.273-.806.108-1.204-.165-.397-.505-.71-.93-.78l-.894-.15c-.542-.09-.94-.56-.94-1.109v-1.094c0-.55.398-1.02.94-1.11l.894-.149c.424-.07.765-.383.93-.78.165-.398.143-.854-.107-1.204l-.527-.738a1.125 1.125 0 01.12-1.45l.773-.773a1.125 1.125 0 011.45-.12l.737.527c.35.25.807.272 1.204.107.397-.165.71-.505.78-.929l.15-.894z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Admin Portal
              </a>
              {session && (
                <button
                  onClick={handleLogout}
                  className="text-sm text-gray-500 hover:text-gray-700"
                >
                  Sign out
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Progress indicator */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex items-center justify-center gap-4">
          <div className={`flex items-center gap-2 ${flowState === 'login' ? 'text-blue-600' : 'text-gray-400'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              flowState === 'login' ? 'bg-blue-600 text-white' : 
              session ? 'bg-green-500 text-white' : 'bg-gray-200 text-gray-500'
            }`}>
              {session ? '✓' : '1'}
            </div>
            <span className="text-sm font-medium">Your Info</span>
          </div>
          
          <div className="w-12 h-px bg-gray-300"></div>
          
          <div className={`flex items-center gap-2 ${flowState === 'connect' ? 'text-blue-600' : 'text-gray-400'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              flowState === 'connect' ? 'bg-blue-600 text-white' : 
              session?.apple_health_connected ? 'bg-green-500 text-white' : 'bg-gray-200 text-gray-500'
            }`}>
              {session?.apple_health_connected ? '✓' : '2'}
            </div>
            <span className="text-sm font-medium">Health Data</span>
          </div>
          
          <div className="w-12 h-px bg-gray-300"></div>
          
          <div className={`flex items-center gap-2 ${flowState === 'dashboard' ? 'text-blue-600' : 'text-gray-400'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              flowState === 'dashboard' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-500'
            }`}>
              3
            </div>
            <span className="text-sm font-medium">Your Quote</span>
          </div>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="flex items-center gap-2 text-red-700">
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>{error}</span>
            </div>
          </div>
        </div>
      )}

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-12">
        {flowState === 'login' && (
          <EndUserLogin 
            onLoginSuccess={(sessionData) => {
              setSession(sessionData);
              localStorage.setItem('endUserSessionId', sessionData.session_id);
              sessionStorage.setItem('endUserFlowInProgress', 'true');
              setFlowState('connect');
            }}
          />
        )}
        
        {flowState === 'connect' && session && (
          <AppleHealthConnect
            session={session}
            onConnected={(result) => {
              // Redirect to the main admin view with the created application
              // The applicant=true query param shows the applicant header/progress
              const appId = result.application_id;
              if (appId) {
                // Store session info for the admin view
                sessionStorage.setItem('applicantSession', JSON.stringify({
                  ...session,
                  application_id: appId,
                  apple_health_connected: true,
                }));
                // Redirect to main page with the application and applicant mode
                window.location.href = `/?app=${appId}&applicant=true`;
              } else {
                // Fallback to dashboard if no app ID
                setSession({
                  ...session,
                  apple_health_connected: true,
                  application_id: result.application_id,
                  flow_state: 'health_connected',
                });
                setFlowState('dashboard');
              }
            }}
            onBack={handleBack}
          />
        )}
        
        {flowState === 'dashboard' && session && (
          <EndUserDashboard
            session={session}
            onLogout={handleLogout}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-sm text-gray-500">
            <p className="mb-2">
              <strong>Demo Application</strong> - This is a demonstration of AI-powered insurance underwriting.
            </p>
            <p>
              No real personal information is collected or stored. Apple Health data is simulated.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
