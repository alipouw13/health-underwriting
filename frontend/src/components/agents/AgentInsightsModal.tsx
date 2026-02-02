'use client';

import { useEffect } from 'react';
import { X, Users, Clock, CheckCircle } from 'lucide-react';
import AgentTransparencyView from './AgentTransparencyView';
import type { OrchestratorOutput } from '@/lib/agentTypes';

interface AgentInsightsModalProps {
  isOpen: boolean;
  onClose: () => void;
  orchestratorOutput: OrchestratorOutput;
  applicationId: string;
}

/**
 * Modal that displays the full Agent Insights view for a specific application.
 * This is the same content as the /agents page but in a modal context.
 */
export default function AgentInsightsModal({
  isOpen,
  onClose,
  orchestratorOutput,
  applicationId,
}: AgentInsightsModalProps) {
  // Handle escape key to close
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const executionTimeSeconds = (orchestratorOutput.total_execution_time_ms / 1000).toFixed(2);

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto">
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal Content */}
      <div className="relative bg-white rounded-xl shadow-2xl w-full max-w-5xl mx-4 my-8 max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 bg-gradient-to-r from-indigo-50 to-white">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-indigo-100 flex items-center justify-center">
              <Users className="w-5 h-5 text-indigo-600" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-900">Agent Insights</h2>
              <p className="text-sm text-slate-500">
                Multi-agent underwriting analysis for application {applicationId.slice(0, 8)}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {/* Execution Stats */}
            <div className="flex items-center gap-3 text-sm">
              <div className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-50 text-emerald-700 rounded-full">
                <CheckCircle className="w-4 h-4" />
                <span className="font-medium">
                  {orchestratorOutput.success ? 'Completed' : 'Failed'}
                </span>
              </div>
              <div className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-100 text-slate-600 rounded-full">
                <Clock className="w-4 h-4" />
                <span className="font-medium">{executionTimeSeconds}s</span>
              </div>
            </div>
            
            {/* Close Button */}
            <button
              onClick={onClose}
              className="p-2 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-6 bg-slate-50">
          <AgentTransparencyView 
            orchestratorOutput={orchestratorOutput}
            defaultExpanded={true}
          />
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-200 bg-white flex items-center justify-between">
          <div className="text-sm text-slate-500">
            Workflow ID: <span className="font-mono text-slate-700">{orchestratorOutput.workflow_id}</span>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-slate-700 bg-slate-100 rounded-lg hover:bg-slate-200 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
