'use client';

import { User, Sparkles, Smartphone } from 'lucide-react';
import type { ApplicationMetadata, PolicyCitation } from '@/lib/types';
import RiskRatingPopover from './RiskRatingPopover';

interface PatientSummaryProps {
  application: ApplicationMetadata;
  onPolicyClick?: (policyId: string) => void;
}

export default function PatientSummary({ application, onPolicyClick }: PatientSummaryProps) {
  // Check if this is an Apple Health application
  const isAppleHealth = application.llm_outputs?.is_apple_health === true || 
                        application.llm_outputs?.workflow_type === 'apple_health' ||
                        application.llm_outputs?.source === 'end_user' ||
                        application.persona === 'end_user';

  // Get summary from LLM outputs
  const customerProfile = application.llm_outputs?.application_summary?.customer_profile?.parsed as any;
  let summary = customerProfile?.summary || customerProfile?.medical_summary || null;
  const riskAssessment = customerProfile?.risk_assessment || null;
  const policyCitations: PolicyCitation[] = customerProfile?.policy_citations || [];
  const underwritingAction = customerProfile?.underwriting_action || null;

  // For Apple Health apps, generate a clean summary from health metrics if the existing one mentions labs
  if (isAppleHealth && summary && (summary.toLowerCase().includes('lab result') || summary.toLowerCase().includes('cholesterol') || summary.toLowerCase().includes('glucose'))) {
    const healthMetrics = application.llm_outputs?.health_metrics as any;
    const name = customerProfile?.full_name || 'The applicant';
    const age = customerProfile?.age || 'unknown';
    const gender = customerProfile?.gender || '';
    
    if (healthMetrics) {
      const steps = healthMetrics?.activity?.daily_steps_avg || 8000;
      const restingHr = healthMetrics?.heart_rate?.resting_hr_avg || 70;
      const sleepHours = healthMetrics?.sleep?.avg_sleep_duration_hours || 7;
      const vo2Max = healthMetrics?.fitness?.vo2_max || 35;
      const bmi = healthMetrics?.body_metrics?.bmi || 25;
      
      summary = `${name} is a ${age}-year-old ${gender} with Apple Health data showing an average of ${steps.toLocaleString()} daily steps. ` +
                `Their resting heart rate averages ${restingHr} bpm with a VO2 Max of ${vo2Max.toFixed(1)} mL/kg/min. ` +
                `Sleep tracking shows an average of ${sleepHours.toFixed(1)} hours per night. ` +
                `BMI is ${typeof bmi === 'number' ? bmi.toFixed(1) : bmi}.`;
    }
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${isAppleHealth ? 'bg-green-100' : 'bg-indigo-100'}`}>
            {isAppleHealth ? (
              <Smartphone className="w-5 h-5 text-green-600" />
            ) : (
              <User className="w-5 h-5 text-indigo-600" />
            )}
          </div>
          <h2 className="text-lg font-semibold text-slate-900">
            {isAppleHealth ? 'Health Summary' : 'Patient Summary'}
          </h2>
          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${
            isAppleHealth 
              ? 'bg-green-50 text-green-600 border-green-100' 
              : 'bg-indigo-50 text-indigo-600 border-indigo-100'
          }`}>
            <Sparkles className="w-3 h-3" />
            {isAppleHealth ? 'Apple Health' : 'AI Analysis'}
          </span>
        </div>

        {/* Risk Badge with Popover */}
        {riskAssessment && (
          <RiskRatingPopover
            rating={riskAssessment}
            rationale={summary}
            citations={policyCitations}
            onPolicyClick={onPolicyClick}
          />
        )}
      </div>

      {/* Summary Text */}
      {summary ? (
        <div className="space-y-4">
          <p className="text-sm text-slate-700 leading-relaxed">
            {summary}
          </p>
        </div>
      ) : (
        <p className="text-sm text-slate-500 italic">
          No summary available. Run analysis to generate health summary.
        </p>
      )}
    </div>
  );
}
