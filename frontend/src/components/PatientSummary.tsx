'use client';

import { User } from 'lucide-react';
import type { ApplicationMetadata } from '@/lib/types';

interface PatientSummaryProps {
  application: ApplicationMetadata;
}

export default function PatientSummary({ application }: PatientSummaryProps) {
  // Get summary from LLM outputs
  const customerProfile = application.llm_outputs?.application_summary?.customer_profile?.parsed;
  const summary = customerProfile?.summary || null;
  const riskAssessment = customerProfile?.risk_assessment || null;
  const underwritingAction = customerProfile?.underwriting_action || null;

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-indigo-100 rounded-lg flex items-center justify-center">
            <User className="w-5 h-5 text-indigo-600" />
          </div>
          <h2 className="text-lg font-semibold text-slate-900">Patient Summary</h2>
        </div>

        {/* Risk Badge */}
        {riskAssessment && (
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${
            riskAssessment.toLowerCase().includes('high') 
              ? 'bg-rose-100 text-rose-700'
              : riskAssessment.toLowerCase().includes('moderate')
              ? 'bg-amber-100 text-amber-700'
              : 'bg-emerald-100 text-emerald-700'
          }`}>
            {riskAssessment}
          </span>
        )}
      </div>

      {/* Summary Text */}
      {summary ? (
        <div className="space-y-4">
          <p className="text-sm text-slate-700 leading-relaxed">
            {summary}
          </p>
          
          {underwritingAction && (
            <div className="pt-3 border-t border-slate-100">
              <h4 className="text-xs font-medium text-slate-500 uppercase mb-1">
                Recommended Action
              </h4>
              <p className="text-sm text-slate-700">{underwritingAction}</p>
            </div>
          )}
        </div>
      ) : (
        <p className="text-sm text-slate-500 italic">
          No summary available. Run analysis to generate patient summary.
        </p>
      )}
    </div>
  );
}
