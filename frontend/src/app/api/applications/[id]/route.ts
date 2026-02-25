import { NextRequest, NextResponse } from 'next/server';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

const API_BASE_URL = process.env.API_URL || 'http://localhost:8000';

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const appId = params.id;
  
  try {
    // Forward query parameters (e.g. ?exclude=markdown_pages,agent_execution)
    const searchParams = request.nextUrl.searchParams;
    const queryString = searchParams.toString();
    const url = queryString
      ? `${API_BASE_URL}/api/applications/${appId}?${queryString}`
      : `${API_BASE_URL}/api/applications/${appId}`;

    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
      cache: 'no-store',
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error(`Failed to fetch application ${appId}:`, error);
    
    // Try to load from local data directory for demo purposes
    try {
      const metadataPath = join(
        process.cwd(),
        '..',
        'data',
        'applications',
        appId,
        'metadata.json'
      );
      
      if (existsSync(metadataPath)) {
        const metadata = JSON.parse(readFileSync(metadataPath, 'utf-8'));
        return NextResponse.json(metadata);
      }
    } catch (localError) {
      console.error('Local data load failed:', localError);
    }

    return NextResponse.json(
      { error: 'Application not found' },
      { status: 404 }
    );
  }
}
