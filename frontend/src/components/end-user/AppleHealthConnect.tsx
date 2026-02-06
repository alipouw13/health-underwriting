'use client';

import { useState } from 'react';
import { 
  Heart, 
  Activity, 
  Moon, 
  Scale, 
  Check, 
  Shield, 
  Smartphone,
  ArrowRight,
  AlertCircle,
  Loader2,
  Wind,
  Footprints,
  Dumbbell
} from 'lucide-react';

interface AppleHealthConnectProps {
  session: any;
  onConnected: (result: any) => void;
  onBack: () => void;
}

export default function AppleHealthConnect({ session, onConnected, onBack }: AppleHealthConnectProps) {
  const [step, setStep] = useState<'intro' | 'consent' | 'connecting' | 'connected'>('intro');
  const [policyType, setPolicyType] = useState('term_life');
  const [coverageAmount, setCoverageAmount] = useState(500000);
  const [error, setError] = useState<string | null>(null);

  // All 7 Apple Health categories from the underwriting policy
  const dataCategories = [
    { icon: Activity, name: 'Activity', description: 'Daily steps, active energy', weight: '25%' },
    { icon: Wind, name: 'Fitness', description: 'VO2 Max, cardio fitness', weight: '20%' },
    { icon: Heart, name: 'Vitals', description: 'Resting HR, HRV, rhythm events', weight: '20%' },
    { icon: Moon, name: 'Sleep', description: 'Duration, consistency', weight: '15%' },
    { icon: Scale, name: 'Body Metrics', description: 'BMI, weight trend', weight: '10%' },
    { icon: Footprints, name: 'Mobility', description: 'Walking speed, steadiness', weight: '10%' },
    { icon: Dumbbell, name: 'Exercise', description: 'Workout frequency, types', weight: '—' },
  ];

  const handleConsent = async () => {
    setStep('connecting');
    setError(null);

    try {
      const response = await fetch('/api/end-user/connect-apple-health', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: session.session_id,
          consent_granted: true,
          policy_type: policyType,
          coverage_amount: coverageAmount,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to connect Apple Health');
      }

      setStep('connected');
      
      // Short delay to show success state
      setTimeout(() => {
        onConnected(data);
      }, 1500);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
      setStep('consent');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-xl max-w-lg w-full p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-red-100 rounded-full mb-4">
            <Heart className="w-8 h-8 text-red-500" />
          </div>
          <h1 className="text-2xl font-bold text-gray-900">
            {step === 'intro' && 'Connect Your Health Data'}
            {step === 'consent' && 'Review Data Access'}
            {step === 'connecting' && 'Syncing Health Data'}
            {step === 'connected' && 'Successfully Connected!'}
          </h1>
          <p className="text-gray-500 mt-2">
            {step === 'intro' && `Welcome, ${session.profile.first_name}! Connect your health data for a personalized quote.`}
            {step === 'consent' && 'Review what data will be accessed for your insurance assessment.'}
            {step === 'connecting' && 'Please wait while we sync your health data...'}
            {step === 'connected' && 'Your health data has been analyzed and your application is ready.'}
          </p>
        </div>

        {/* Demo Disclaimer */}
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 mb-6">
          <div className="flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-amber-700">
              <span className="font-medium">Demo Mode:</span> No real Apple Health connection. Synthetic data will be generated.
            </p>
          </div>
        </div>

        {/* Step: Intro */}
        {step === 'intro' && (
          <>
            <div className="space-y-4 mb-6">
              <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg">
                <Smartphone className="w-10 h-10 text-gray-400" />
                <div>
                  <p className="font-medium text-gray-900">Apple Health Integration</p>
                  <p className="text-sm text-gray-500">7 health categories for risk assessment</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {dataCategories.map((cat, index) => (
                  <div key={index} className="p-3 border border-gray-200 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <cat.icon className="w-5 h-5 text-indigo-500" />
                      {cat.weight !== '—' && (
                        <span className="text-xs text-gray-400 font-medium">{cat.weight}</span>
                      )}
                    </div>
                    <p className="font-medium text-sm text-gray-900">{cat.name}</p>
                    <p className="text-xs text-gray-500">{cat.description}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Policy Selection */}
            <div className="space-y-4 mb-6 p-4 bg-indigo-50 rounded-lg">
              <h3 className="font-medium text-gray-900">Policy Preferences</h3>
              
              <div>
                <label className="block text-sm text-gray-600 mb-1">Policy Type</label>
                <select
                  value={policyType}
                  onChange={(e) => setPolicyType(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="term_life">Term Life Insurance</option>
                  <option value="whole_life">Whole Life Insurance</option>
                  <option value="health">Health Insurance</option>
                </select>
              </div>

              <div>
                <label className="block text-sm text-gray-600 mb-1">Coverage Amount</label>
                <select
                  value={coverageAmount}
                  onChange={(e) => setCoverageAmount(Number(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                >
                  <option value={100000}>$100,000</option>
                  <option value={250000}>$250,000</option>
                  <option value={500000}>$500,000</option>
                  <option value={1000000}>$1,000,000</option>
                  <option value={2000000}>$2,000,000</option>
                </select>
              </div>
            </div>

            <button
              onClick={() => setStep('consent')}
              className="w-full py-3 px-4 bg-indigo-600 text-white font-medium rounded-lg hover:bg-indigo-700 flex items-center justify-center gap-2 transition-colors"
            >
              <span>Review Data Access</span>
              <ArrowRight className="w-5 h-5" />
            </button>
          </>
        )}

        {/* Step: Consent */}
        {step === 'consent' && (
          <>
            <div className="space-y-2 mb-6 max-h-64 overflow-y-auto">
              {dataCategories.map((cat, index) => (
                <div key={index} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                  <div className="w-8 h-8 bg-green-100 rounded-full flex items-center justify-center">
                    <Check className="w-4 h-4 text-green-600" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <p className="font-medium text-gray-900">{cat.name}</p>
                      {cat.weight !== '—' && (
                        <span className="text-xs text-indigo-600 font-medium">{cat.weight} weight</span>
                      )}
                    </div>
                    <p className="text-sm text-gray-500">{cat.description}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
              <div className="flex items-start gap-3">
                <Shield className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium text-blue-900">Data Privacy</p>
                  <p className="text-blue-700 mt-1">
                    Your health data is only used for insurance assessment purposes and is handled securely.
                    You can revoke access at any time.
                  </p>
                </div>
              </div>
            </div>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4 text-sm text-red-700">
                {error}
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={onBack}
                className="flex-1 py-3 px-4 border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-50 transition-colors"
              >
                Back
              </button>
              <button
                onClick={handleConsent}
                className="flex-1 py-3 px-4 bg-green-600 text-white font-medium rounded-lg hover:bg-green-700 flex items-center justify-center gap-2 transition-colors"
              >
                <Check className="w-5 h-5" />
                <span>Allow Access</span>
              </button>
            </div>
          </>
        )}

        {/* Step: Connecting */}
        {step === 'connecting' && (
          <div className="text-center py-8">
            <div className="relative inline-flex mb-6">
              <div className="w-20 h-20 bg-indigo-100 rounded-full flex items-center justify-center">
                <Heart className="w-10 h-10 text-indigo-500" />
              </div>
              <div className="absolute inset-0 rounded-full border-4 border-indigo-500 border-t-transparent animate-spin" />
            </div>
            <p className="text-gray-600 mb-2">Syncing health data...</p>
            <p className="text-sm text-gray-400">This may take a few moments</p>
          </div>
        )}

        {/* Step: Connected */}
        {step === 'connected' && (
          <div className="text-center py-8">
            <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Check className="w-10 h-10 text-green-600" />
            </div>
            <p className="text-gray-600 mb-4">Your application is ready for assessment!</p>
            <div className="flex items-center justify-center gap-2 text-indigo-600">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Redirecting...</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
