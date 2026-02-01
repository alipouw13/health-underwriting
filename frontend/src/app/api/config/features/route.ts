import { NextResponse } from 'next/server';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface FeatureFlags {
  agent_execution_enabled: boolean;
  rag_enabled: boolean;
  automotive_claims_enabled: boolean;
}

export async function GET() {
  try {
    const response = await fetch(`${API_BASE}/api/config/features`, {
      headers: {
        'Content-Type': 'application/json',
      },
      // Don't cache feature flags - always fetch fresh
      cache: 'no-store',
    });

    if (!response.ok) {
      console.error('Failed to fetch feature flags:', response.status);
      // Return default flags on error
      return NextResponse.json({
        agent_execution_enabled: false,
        rag_enabled: false,
        automotive_claims_enabled: false,
      });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching feature flags:', error);
    // Return default flags on error
    return NextResponse.json({
      agent_execution_enabled: false,
      rag_enabled: false,
      automotive_claims_enabled: false,
    });
  }
}
