'use client';

import Link from 'next/link';
import {
  LayoutDashboard,
  Clock,
  FileText,
  FileStack,
  Plus,
  Settings,
  Bot,
} from 'lucide-react';
import type { ApplicationListItem, ApplicationMetadata } from '@/lib/types';
import clsx from 'clsx';
import { useFeatureFlags } from '@/lib/useFeatureFlags';

interface SidebarProps {
  applications: ApplicationListItem[];
  selectedAppId?: string;
  selectedApp?: ApplicationMetadata;
  activeView: 'overview' | 'timeline' | 'documents' | 'source';
  onSelectApp: (appId: string) => void;
  onChangeView: (view: 'overview' | 'timeline' | 'documents' | 'source') => void;
}

export default function Sidebar({
  applications,
  selectedAppId,
  selectedApp,
  activeView,
  onSelectApp,
  onChangeView,
}: SidebarProps) {
  const { flags } = useFeatureFlags();
  // Check if there are documents available
  const hasDocuments = selectedApp?.files && selectedApp.files.length > 0;
  const hasSourcePages = selectedApp?.markdown_pages && selectedApp.markdown_pages.length > 0;

  return (
    <aside className="w-56 bg-sidebar text-white flex flex-col">
      {/* Logo */}
      <div className="p-4 border-b border-slate-700">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">UA</span>
          </div>
          <span className="font-semibold text-lg">Underwriting Assistant</span>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-auto sidebar-scroll py-4">
        <ul className="space-y-1">
          <li>
            <button
              onClick={() => onChangeView('overview')}
              className={clsx(
                'w-full flex items-center gap-3 px-4 py-2.5 text-sm transition-colors',
                activeView === 'overview'
                  ? 'bg-sidebar-active text-white'
                  : 'text-slate-400 hover:bg-sidebar-hover hover:text-white'
              )}
            >
              <LayoutDashboard className="w-5 h-5" />
              <span>Overview</span>
            </button>
          </li>
          <li>
            <button
              onClick={() => onChangeView('timeline')}
              className={clsx(
                'w-full flex items-center gap-3 px-4 py-2.5 text-sm transition-colors',
                activeView === 'timeline'
                  ? 'bg-sidebar-active text-white'
                  : 'text-slate-400 hover:bg-sidebar-hover hover:text-white'
              )}
            >
              <Clock className="w-5 h-5" />
              <span>Timeline</span>
            </button>
          </li>
          <li>
            <button
              onClick={() => onChangeView('documents')}
              disabled={!hasDocuments}
              className={clsx(
                'w-full flex items-center gap-3 px-4 py-2.5 text-sm transition-colors',
                activeView === 'documents'
                  ? 'bg-sidebar-active text-white'
                  : hasDocuments
                  ? 'text-slate-400 hover:bg-sidebar-hover hover:text-white'
                  : 'text-slate-600 cursor-not-allowed'
              )}
            >
              <FileText className="w-5 h-5" />
              <span>Documents</span>
              {hasDocuments && (
                <span className="ml-auto text-xs bg-slate-700 px-1.5 py-0.5 rounded">
                  {selectedApp.files.length}
                </span>
              )}
            </button>
          </li>
          <li>
            <button
              onClick={() => onChangeView('source')}
              disabled={!hasSourcePages}
              className={clsx(
                'w-full flex items-center gap-3 px-4 py-2.5 text-sm transition-colors',
                activeView === 'source'
                  ? 'bg-sidebar-active text-white'
                  : hasSourcePages
                  ? 'text-slate-400 hover:bg-sidebar-hover hover:text-white'
                  : 'text-slate-600 cursor-not-allowed'
              )}
            >
              <FileStack className="w-5 h-5" />
              <span>Source Pages</span>
              {hasSourcePages && selectedApp?.markdown_pages && (
                <span className="ml-auto text-xs bg-slate-700 px-1.5 py-0.5 rounded">
                  {selectedApp.markdown_pages.length}
                </span>
              )}
            </button>
          </li>
        </ul>

        {/* Applications Section */}
        {applications.length > 0 && (
          <div className="mt-6 px-4">
            <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">
              Applications
            </h3>
            <ul className="space-y-1">
              {applications.map((app) => (
                <li key={app.id}>
                  <button
                    onClick={() => onSelectApp(app.id)}
                    className={clsx(
                      'w-full text-left px-3 py-2 text-sm rounded-lg transition-colors',
                      selectedAppId === app.id
                        ? 'bg-blue-600 text-white'
                        : 'text-slate-400 hover:bg-sidebar-hover hover:text-white'
                    )}
                  >
                    <div className="font-medium truncate">
                      {app.summary_title || app.id.substring(0, 8)}
                    </div>
                    <div className="text-xs text-slate-500">
                      {app.external_reference || 'No reference'}
                    </div>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </nav>

      {/* Footer Actions */}
      <div className="p-4 border-t border-slate-700 space-y-2">
        {flags.agent_execution_enabled && (
          <Link
            href="/agents"
            className="w-full flex items-center justify-center gap-2 px-4 py-2 text-slate-400 hover:text-white hover:bg-sidebar-hover rounded-lg transition-colors"
          >
            <Bot className="w-4 h-4" />
            <span className="text-sm">Agent Insights</span>
          </Link>
        )}
        <Link
          href="/admin"
          className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
        >
          <Plus className="w-4 h-4" />
          <span className="text-sm font-medium">New Application</span>
        </Link>
        <Link
          href="/admin"
          className="w-full flex items-center justify-center gap-2 px-4 py-2 text-slate-400 hover:text-white hover:bg-sidebar-hover rounded-lg transition-colors"
        >
          <Settings className="w-4 h-4" />
          <span className="text-sm">Admin</span>
        </Link>
      </div>
    </aside>
  );
}
