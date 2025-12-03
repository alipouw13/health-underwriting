'use client';

import { useState } from 'react';
import {
  Stethoscope,
  FileText,
  ExternalLink,
  ListChecks,
  Activity,
  Shield,
  ClipboardList,
  CheckCircle,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import type { ApplicationMetadata } from '@/lib/types';
import clsx from 'clsx';

interface LifeHealthClaimsOverviewProps {
  application: ApplicationMetadata | null;
}

function getFieldValue(field: unknown, defaultValue: string = ''): string {
  if (field && typeof field === 'object' && 'value' in field) {
    const val = (field as { value: unknown }).value;
    return val != null && val !== '' ? String(val) : defaultValue;
  }
  return defaultValue;
}

// Compact Header Strip
function HeaderStrip({ application }: { application: ApplicationMetadata | null }) {
  const extractedFields = application?.extracted_fields || {};
  
  // Try different possible field name variations and only show defaults if truly no data
  const planName = getFieldValue(extractedFields.PlanName) || 
                   getFieldValue(extractedFields.plan_name) || 
                   getFieldValue(extractedFields.policy_number) || 'N/A';
  const coverageDates = getFieldValue(extractedFields.CoverageDates) || 
                        getFieldValue(extractedFields.coverage_dates) || 
                        getFieldValue(extractedFields.date_of_service) || 'N/A';
  const providerName = getFieldValue(extractedFields.ProviderName) || 
                       getFieldValue(extractedFields.provider_name) || 'N/A';
  const billedAmount = getFieldValue(extractedFields.BilledAmount) || 
                       getFieldValue(extractedFields.total_charges) || 
                       getFieldValue(extractedFields.billed_amount) || 'N/A';
  const allowedAmount = getFieldValue(extractedFields.AllowedAmount) || 
                        getFieldValue(extractedFields.allowed_amount) || 'N/A';
  const planLiability = getFieldValue(extractedFields.PlanLiability) || 
                        getFieldValue(extractedFields.plan_liability) || 
                        getFieldValue(extractedFields.plan_pays) || 'N/A';
  const memberOOP = getFieldValue(extractedFields.MemberOOP) || 
                    getFieldValue(extractedFields.member_oop) || 
                    getFieldValue(extractedFields.member_responsibility) || 'N/A';

  return (
    <div className="bg-gradient-to-r from-cyan-600 to-cyan-700 text-white px-5 py-2.5 flex-shrink-0">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-6 text-sm">
          <div><span className="text-cyan-200 text-xs">Policy:</span> <span className="font-medium">{planName}</span></div>
          <div><span className="text-cyan-200 text-xs">Coverage:</span> <span className="font-medium">{coverageDates}</span></div>
          <div><span className="text-cyan-200 text-xs">Provider:</span> <span className="font-medium">{providerName}</span></div>
        </div>
        <div className="flex items-center gap-5">
          <div className="text-center"><div className="text-xs text-cyan-200">Billed</div><div className="font-semibold">{billedAmount}</div></div>
          <div className="text-center"><div className="text-xs text-cyan-200">Allowed</div><div className="font-semibold">{allowedAmount}</div></div>
          <div className="text-center"><div className="text-xs text-cyan-200">Plan Pays</div><div className="font-semibold text-emerald-300">{planLiability}</div></div>
          <div className="text-center"><div className="text-xs text-cyan-200">Member OOP</div><div className="font-semibold">{memberOOP}</div></div>
          <span className="px-2.5 py-1 bg-rose-500 rounded-full text-xs font-medium">High</span>
        </div>
      </div>
    </div>
  );
}

// Main Component with Horizontal 2-Row Layout
export default function LifeHealthClaimsOverview({ application }: LifeHealthClaimsOverviewProps) {
  const [checkedTasks, setCheckedTasks] = useState<number[]>([]);
  const [expandedSection, setExpandedSection] = useState<string>('reason');
  
  const llmOutputs = (application?.llm_outputs || {}) as Record<string, any>;
  const clinicalNotes = (llmOutputs.clinical_case_notes || {}) as Record<string, any>;
  const reasonForVisit = (clinicalNotes.reason_for_visit || {}) as Record<string, any>;
  const keyDiagnoses = (clinicalNotes.key_diagnoses || {}) as Record<string, any>;
  
  const files = application?.files || [];
  const documents = files.length > 0 ? files.map(f => ({ name: f.filename, type: 'Document' })) : [];

  // Extract timeline from actual data if available
  const extractedFields = application?.extracted_fields || {};
  const timelineEvents = [];
  const serviceDate = getFieldValue(extractedFields.date_of_service) || getFieldValue(extractedFields.DateOfService);
  if (serviceDate) {
    const diagnosis = getFieldValue(extractedFields.primary_diagnosis) || getFieldValue(extractedFields.PrimaryDiagnosis);
    const procedure = getFieldValue(extractedFields.procedure_name) || getFieldValue(extractedFields.ProcedureName);
    if (diagnosis) timelineEvents.push({ date: serviceDate, type: 'Diagnosis', desc: diagnosis, isPrimary: true });
    if (procedure) timelineEvents.push({ date: serviceDate, type: 'Procedure', desc: procedure, isPrimary: false });
  }

  // Extract benefit info from LLM outputs if available
  const eligibility = (llmOutputs.eligibility || {}) as Record<string, any>;
  const coverageVerification = (eligibility.coverage_verification?.parsed || {}) as Record<string, any>;
  const benefitItems = [];
  if (coverageVerification.eligibility_status) {
    benefitItems.push({ label: 'Eligibility', value: coverageVerification.eligibility_status, ok: coverageVerification.eligibility_status === 'Eligible' });
  }
  if (coverageVerification.network_status) {
    benefitItems.push({ label: 'Network', value: coverageVerification.network_status, ok: coverageVerification.network_status === 'In-Network' });
  }
  if (coverageVerification.deductible_status?.remaining) {
    const remaining = coverageVerification.deductible_status.remaining;
    benefitItems.push({ label: 'Deductible', value: remaining === '$0' ? 'Met' : `${remaining} remaining`, ok: remaining === '$0' });
  }

  // Extract claim line items from payment calculation if available
  const claimAssessment = (llmOutputs.claim_assessment || llmOutputs.payment_calculation || {}) as Record<string, any>;
  const paymentDetermination = (claimAssessment.payment_determination?.parsed || claimAssessment.benefit_determination?.parsed || {}) as Record<string, any>;
  const paymentCalc = paymentDetermination.payment_calculation || {};
  const lineItems = paymentCalc.line_items || paymentDetermination.charge_breakdown || [];
  
  const claimLines = lineItems.map((item: any) => ({
    line: item.line || 0,
    code: item.cpt || item.code || 'N/A',
    desc: item.description || item.service || item.desc || 'N/A',
    billed: item.billed || 'N/A',
    allowed: item.allowed || 'N/A',
    decision: item.status || (paymentDetermination.claim_decision === 'Approve' ? 'Approve' : 'Pending')
  }));

  // Extract tasks from multiple possible sources
  const tasks = [];
  const medicalReview = (llmOutputs.medical_review || llmOutputs.claim_assessment || {}) as Record<string, any>;
  const clinicalAssessment = (medicalReview.clinical_assessment?.parsed || medicalReview.medical_necessity_review?.parsed || {}) as Record<string, any>;
  
  if (clinicalAssessment.processing_action) {
    tasks.push({ task: clinicalAssessment.processing_action, due: 'Pending' });
  }
  
  // Check for service reviews that might need attention
  const serviceReviews = clinicalAssessment.service_reviews || [];
  serviceReviews.forEach((review: any) => {
    if (review.medical_necessity === 'Review Required' || review.medical_necessity === 'Pending') {
      tasks.push({ task: `Review: ${review.service}`, due: 'Today' });
    }
  });

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-slate-100">
      <HeaderStrip application={application} />
      
      {/* Main Content - 3 Horizontal Rows */}
      <div className="flex-1 overflow-auto p-4">
        <div className="h-full flex flex-col gap-4">
          
          {/* ROW 1: Clinical Notes | Benefits | Tasks */}
          <div className="grid grid-cols-12 gap-4 min-h-0" style={{ flex: '1 1 35%' }}>
            
            {/* Clinical Notes - 4 cols */}
            <div className="col-span-4 bg-white rounded-lg shadow-sm border border-slate-200 flex flex-col overflow-hidden">
              <div className="px-3 py-2 border-b border-slate-100 flex items-center gap-2 bg-slate-50 flex-shrink-0">
                <Stethoscope className="w-4 h-4 text-cyan-600" />
                <span className="font-semibold text-slate-900 text-sm">Clinical Notes</span>
                <span className="ml-auto text-xs text-cyan-600 bg-cyan-50 px-1.5 py-0.5 rounded">AI</span>
              </div>
              <div className="flex-1 overflow-auto p-3 text-sm">
                <div className="space-y-3">
                  <div 
                    className="border border-slate-200 rounded cursor-pointer"
                    onClick={() => setExpandedSection(expandedSection === 'reason' ? '' : 'reason')}
                  >
                    <div className="px-3 py-2 flex items-center justify-between hover:bg-slate-50">
                      <span className="font-medium text-slate-900">Reason for Visit</span>
                      {expandedSection === 'reason' ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />}
                    </div>
                    {expandedSection === 'reason' && (
                      <div className="px-3 pb-2 text-slate-600 text-xs">
                        {reasonForVisit.summary || 'Emergency room visit for acute chest pain with shortness of breath.'}
                      </div>
                    )}
                  </div>
                  <div 
                    className="border border-slate-200 rounded cursor-pointer"
                    onClick={() => setExpandedSection(expandedSection === 'diagnoses' ? '' : 'diagnoses')}
                  >
                    <div className="px-3 py-2 flex items-center justify-between hover:bg-slate-50">
                      <span className="font-medium text-slate-900">Key Diagnoses</span>
                      {expandedSection === 'diagnoses' ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />}
                    </div>
                    {expandedSection === 'diagnoses' && (
                      <div className="px-3 pb-2 space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="px-1.5 py-0.5 bg-cyan-100 text-cyan-700 text-xs rounded font-mono">R07.9</span>
                          <span className="text-slate-600 text-xs">Chest pain, unspecified</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="px-1.5 py-0.5 bg-slate-100 text-slate-600 text-xs rounded font-mono">R06.02</span>
                          <span className="text-slate-600 text-xs">Shortness of breath</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
            
            {/* Benefits & Policy - 4 cols */}
            <div className="col-span-4 bg-white rounded-lg shadow-sm border border-slate-200 flex flex-col overflow-hidden">
              <div className="px-3 py-2 border-b border-slate-100 flex items-center gap-2 bg-slate-50 flex-shrink-0">
                <FileText className="w-4 h-4 text-cyan-600" />
                <span className="font-semibold text-slate-900 text-sm">Benefits & Policy</span>
              </div>
              <div className="flex-1 overflow-auto p-3 text-xs">
                <div className="space-y-2">
                  <div className="flex justify-between"><span className="text-slate-600">Eligibility</span><span className="font-medium text-emerald-600">Active</span></div>
                  <div className="flex justify-between"><span className="text-slate-600">Network</span><span className="font-medium">In-Network</span></div>
                  <div className="flex justify-between"><span className="text-slate-600">Deductible</span><span className="font-medium">$500 met</span></div>
                  <div className="flex justify-between border-t pt-2 mt-2"><span className="text-slate-600">OOP Max</span><span className="font-medium">$2,100 / $6,500</span></div>
                  <div className="flex justify-between"><span className="text-slate-600">Benefit Limits</span><span className="font-medium text-emerald-600">Within Limits</span></div>
                </div>
              </div>
            </div>
            
            {/* Tasks - 4 cols */}
            <div className="col-span-4 bg-white rounded-lg shadow-sm border border-slate-200 flex flex-col overflow-hidden">
              <div className="px-3 py-2 border-b border-slate-100 flex items-center gap-2 bg-slate-50 flex-shrink-0">
                <ListChecks className="w-4 h-4 text-cyan-600" />
                <span className="font-semibold text-slate-900 text-sm">Tasks</span>
                <span className="ml-auto text-xs text-amber-600 bg-amber-50 px-1.5 py-0.5 rounded">{tasks.length - checkedTasks.length} pending</span>
              </div>
              <div className="flex-1 overflow-auto p-3">
                <div className="space-y-2">
                  {tasks.map((t, i) => (
                    <label key={i} className="flex items-center gap-2 cursor-pointer text-sm">
                      <input 
                        type="checkbox" 
                        checked={checkedTasks.includes(i)} 
                        onChange={() => setCheckedTasks(prev => prev.includes(i) ? prev.filter(x => x !== i) : [...prev, i])}
                        className="w-3.5 h-3.5 rounded border-slate-300 text-cyan-600" 
                      />
                      <span className={clsx('flex-1', checkedTasks.includes(i) ? 'text-slate-400 line-through' : 'text-slate-900')}>{t.task}</span>
                      <span className="text-xs text-slate-400">{t.due}</span>
                    </label>
                  ))}
                </div>
                <button className="mt-3 w-full py-2 bg-cyan-600 hover:bg-cyan-700 text-white text-sm font-medium rounded-lg transition-colors">
                  Propose Final Decision
                </button>
              </div>
            </div>
          </div>
          
          {/* ROW 2: Timeline | Documents */}
          <div className="grid grid-cols-12 gap-4 min-h-0" style={{ flex: '1 1 30%' }}>
            
            {/* Timeline - 6 cols */}
            <div className="col-span-6 bg-white rounded-lg shadow-sm border border-slate-200 flex flex-col overflow-hidden">
              <div className="px-3 py-2 border-b border-slate-100 flex items-center gap-2 bg-slate-50 flex-shrink-0">
                <Activity className="w-4 h-4 text-cyan-600" />
                <span className="font-semibold text-slate-900 text-sm">Timeline</span>
              </div>
              <div className="flex-1 overflow-auto p-3">
                <div className="space-y-2">
                  {timelineEvents.map((e, i) => (
                    <div key={i} className="flex items-start gap-2 text-xs">
                      <div className="w-10 text-slate-400 flex-shrink-0">{e.date}</div>
                      <div className={clsx('w-1.5 h-1.5 rounded-full mt-1 flex-shrink-0',
                        e.isPrimary ? 'bg-cyan-500' : 'bg-slate-300'
                      )} />
                      <div className="min-w-0">
                        <div className="font-medium text-slate-900">{e.type}</div>
                        <div className="text-slate-500">{e.desc}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            {/* Documents - 6 cols */}
            <div className="col-span-6 bg-white rounded-lg shadow-sm border border-slate-200 flex flex-col overflow-hidden">
              <div className="px-3 py-2 border-b border-slate-100 flex items-center gap-2 bg-slate-50 flex-shrink-0">
                <FileText className="w-4 h-4 text-cyan-600" />
                <span className="font-semibold text-slate-900 text-sm">Documents</span>
                <span className="ml-auto text-xs text-slate-500">{documents.length} files</span>
              </div>
              <div className="flex-1 overflow-auto p-3">
                <div className="space-y-1.5">
                  {documents.map((doc, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs p-1.5 hover:bg-slate-50 rounded cursor-pointer group">
                      <FileText className="w-3.5 h-3.5 text-slate-400 flex-shrink-0" />
                      <span className="flex-1 truncate text-slate-700">{doc.name}</span>
                      <span className="px-1.5 py-0.5 bg-cyan-100 text-cyan-700 rounded text-xs flex-shrink-0">{doc.type}</span>
                      <ExternalLink className="w-3 h-3 text-slate-400 opacity-0 group-hover:opacity-100 flex-shrink-0" />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          
          {/* ROW 3: Claim Line Evaluation - Full Width */}
          <div className="bg-white rounded-lg shadow-sm border border-slate-200 flex flex-col overflow-hidden" style={{ flex: '1 1 45%' }}>
            <div className="px-4 py-2.5 border-b border-slate-100 flex items-center gap-2 bg-slate-50 flex-shrink-0">
              <ListChecks className="w-4 h-4 text-cyan-600" />
              <span className="font-semibold text-slate-900 text-sm">Claim Line Evaluation</span>
              <span className="ml-auto text-xs text-slate-500">{claimLines.length} lines • Total: $1,755</span>
            </div>
            <div className="flex-1 overflow-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 sticky top-0">
                  <tr className="text-xs text-slate-500">
                    <th className="px-4 py-2 text-left w-12">#</th>
                    <th className="px-4 py-2 text-left w-20">Code</th>
                    <th className="px-4 py-2 text-left">Description</th>
                    <th className="px-4 py-2 text-right w-24">Billed</th>
                    <th className="px-4 py-2 text-right w-24">Allowed</th>
                    <th className="px-4 py-2 text-center w-24">AI Decision</th>
                    <th className="px-4 py-2 text-center w-24">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {claimLines.map((l: any) => (
                    <tr key={l.line} className="hover:bg-slate-50">
                      <td className="px-4 py-2.5 text-slate-600">{l.line}</td>
                      <td className="px-4 py-2.5 font-mono text-slate-700">{l.code}</td>
                      <td className="px-4 py-2.5 text-slate-900">{l.desc}</td>
                      <td className="px-4 py-2.5 text-right text-slate-600">{l.billed}</td>
                      <td className="px-4 py-2.5 text-right text-slate-900 font-medium">{l.allowed}</td>
                      <td className="px-4 py-2.5 text-center">
                        <span className={clsx('px-2 py-0.5 rounded text-xs font-medium',
                          l.decision === 'Approve' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'
                        )}>{l.decision}</span>
                      </td>
                      <td className="px-4 py-2.5 text-center">
                        <button className="px-2 py-1 text-xs text-cyan-600 hover:bg-cyan-50 rounded">Override</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
        </div>
      </div>
    </div>
  );
}
