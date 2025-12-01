'use client';

import { FileText, Download, ExternalLink } from 'lucide-react';
import type { StoredFile } from '@/lib/types';

interface DocumentsPanelProps {
  files: StoredFile[];
}

export default function DocumentsPanel({ files }: DocumentsPanelProps) {
  if (!files || files.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Documents</h2>
        <div className="text-center py-8 text-gray-500">
          <FileText className="w-12 h-12 mx-auto mb-2 text-gray-400" />
          <p>No documents uploaded for this application.</p>
        </div>
      </div>
    );
  }

  const getFileSize = (file: StoredFile): string => {
    // In a real implementation, file size would come from the backend
    return 'PDF';
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-gray-900">
          Documents ({files.length})
        </h2>
      </div>

      <div className="space-y-4">
        {files.map((file, index) => (
          <div
            key={index}
            className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
                <FileText className="w-6 h-6 text-red-600" />
              </div>
              <div>
                <h3 className="font-medium text-gray-900">{file.filename}</h3>
                <p className="text-sm text-gray-500">
                  {getFileSize(file)} • Uploaded
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {file.url && (
                <>
                  <a
                    href={file.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                    title="Open in new tab"
                  >
                    <ExternalLink className="w-5 h-5" />
                  </a>
                  <a
                    href={file.url}
                    download={file.filename}
                    className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                    title="Download"
                  >
                    <Download className="w-5 h-5" />
                  </a>
                </>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 p-4 bg-blue-50 rounded-lg">
        <p className="text-sm text-blue-700">
          <strong>Tip:</strong> These are the original PDF documents uploaded for
          this application. The extracted text and analysis are available in the
          Source Pages view.
        </p>
      </div>
    </div>
  );
}
