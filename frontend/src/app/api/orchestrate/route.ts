import { NextRequest, NextResponse } from 'next/server';

/**
 * API Route: POST /api/orchestrate
 * 
 * Trigger the OrchestratorAgent for a given patient ID.
 * Returns the full orchestrator output for UI consumption.
 * 
 * CONSTRAINT: This is the ONLY way UI should interact with agents.
 *             All agent execution happens through the orchestrator.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { patient_id } = body;

    if (!patient_id) {
      return NextResponse.json(
        { error: 'patient_id is required' },
        { status: 400 }
      );
    }

    // Call the backend orchestrator API
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
    const response = await fetch(`${backendUrl}/api/orchestrate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ patient_id }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json(
        { error: `Backend error: ${errorText}` },
        { status: response.status }
      );
    }

    const result = await response.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error('Orchestration error:', error);
    return NextResponse.json(
      { error: 'Failed to execute orchestration' },
      { status: 500 }
    );
  }
}

/**
 * API Route: GET /api/orchestrate
 * 
 * Get a list of available patient IDs for orchestration demo.
 */
export async function GET() {
  // Return demo patient IDs
  const demoPatients = [
    { id: 'PAT-HEALTHY-001', label: 'Healthy Patient', description: 'Low risk, healthy metrics' },
    { id: 'PAT-MODERATE-001', label: 'Moderate Risk Patient', description: 'Some health concerns' },
    { id: 'PAT-HIGH-RISK-001', label: 'High Risk Patient', description: 'Multiple risk factors' },
  ];

  return NextResponse.json({ patients: demoPatients });
}
