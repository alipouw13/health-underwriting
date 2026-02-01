'use client';

import { useState, useEffect } from 'react';

export interface FeatureFlags {
  agent_execution_enabled: boolean;
  rag_enabled: boolean;
  automotive_claims_enabled: boolean;
}

const DEFAULT_FLAGS: FeatureFlags = {
  agent_execution_enabled: false,
  rag_enabled: false,
  automotive_claims_enabled: false,
};

export function useFeatureFlags(): {
  flags: FeatureFlags;
  loading: boolean;
  error: Error | null;
} {
  const [flags, setFlags] = useState<FeatureFlags>(DEFAULT_FLAGS);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    async function fetchFlags() {
      try {
        const response = await fetch('/api/config/features');
        if (!response.ok) {
          throw new Error(`Failed to fetch feature flags: ${response.status}`);
        }
        const data = await response.json();
        setFlags(data);
      } catch (err) {
        console.error('Error fetching feature flags:', err);
        setError(err instanceof Error ? err : new Error(String(err)));
        // Keep default flags on error
      } finally {
        setLoading(false);
      }
    }

    fetchFlags();
  }, []);

  return { flags, loading, error };
}
