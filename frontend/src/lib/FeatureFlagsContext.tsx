'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

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

interface FeatureFlagsContextValue {
  flags: FeatureFlags;
  loading: boolean;
  error: Error | null;
}

const FeatureFlagsContext = createContext<FeatureFlagsContextValue>({
  flags: DEFAULT_FLAGS,
  loading: true,
  error: null,
});

export function FeatureFlagsProvider({ children }: { children: ReactNode }) {
  const [flags, setFlags] = useState<FeatureFlags>(DEFAULT_FLAGS);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchFlags() {
      try {
        const response = await fetch('/api/config/features');
        if (!response.ok) {
          throw new Error(`Failed to fetch feature flags: ${response.status}`);
        }
        const data = await response.json();
        if (!cancelled) setFlags(data);
      } catch (err) {
        console.error('Error fetching feature flags:', err);
        if (!cancelled) setError(err instanceof Error ? err : new Error(String(err)));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchFlags();
    return () => { cancelled = true; };
  }, []);

  return (
    <FeatureFlagsContext.Provider value={{ flags, loading, error }}>
      {children}
    </FeatureFlagsContext.Provider>
  );
}

/**
 * Hook to read feature flags from the shared context.
 * Drop-in replacement for the old per-component useFeatureFlags hook.
 */
export function useFeatureFlagsContext(): FeatureFlagsContextValue {
  return useContext(FeatureFlagsContext);
}
