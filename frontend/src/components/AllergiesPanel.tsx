'use client';

import { AlertTriangle } from 'lucide-react';
import type { ApplicationMetadata } from '@/lib/types';
import ConfidenceIndicator from './ConfidenceIndicator';

interface AllergiesPanelProps {
  application: ApplicationMetadata;
}

interface AllergiesData {
  allergies: string[];
  confidence?: number;
}

function parseAllergies(application: ApplicationMetadata): AllergiesData {
  const fields = application.extracted_fields || {};
  
  // Check medical conditions for allergies
  const medicalField = Object.values(fields).find(f => f.field_name === 'MedicalConditionsSummary');
  if (!medicalField?.value) {
    return { allergies: [] };
  }

  const value = String(medicalField.value).toLowerCase();
  const allergies: string[] = [];

  // Look for allergy-related keywords
  const allergyPatterns = [
    /allerg\w*\s+to\s+([^;,]+)/gi,
    /([^;,]+)\s+allergy/gi,
    /allergic\s+to\s+([^;,]+)/gi,
  ];

  for (const pattern of allergyPatterns) {
    const matches = value.matchAll(pattern);
    for (const match of matches) {
      if (match[1]) {
        allergies.push(match[1].trim());
      }
    }
  }

  return { allergies, confidence: medicalField.confidence };
}

export default function AllergiesPanel({ application }: AllergiesPanelProps) {
  const { allergies, confidence } = parseAllergies(application);

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <div className="w-8 h-8 bg-amber-100 rounded-lg flex items-center justify-center">
          <AlertTriangle className="w-5 h-5 text-amber-600" />
        </div>
        <h2 className="text-base font-semibold text-slate-900">Allergies</h2>
        {confidence !== undefined && (
          <ConfidenceIndicator confidence={confidence} fieldName="Allergies" />
        )}
      </div>

      {/* Content */}
      {allergies.length > 0 ? (
        <ul className="space-y-2">
          {allergies.map((allergy, idx) => (
            <li key={idx} className="flex items-start gap-2 text-sm text-slate-700">
              <span className="text-slate-400 mt-1">•</span>
              <span>{allergy}</span>
            </li>
          ))}
        </ul>
      ) : (
        <div className="flex flex-col items-center justify-center py-4 text-center">
          <p className="text-sm text-slate-500 italic">No allergies extracted</p>
        </div>
      )}
    </div>
  );
}
