# Underwriting Assistant - Next.js Frontend

A modern Next.js + Tailwind CSS frontend for the Underwriting Assistant application, providing a rich UI for insurance underwriting workflows.

## Features

- **Patient Overview Dashboard**: View patient information, medical history, and risk assessment at a glance
- **Chronological Timeline**: Interactive timeline of medical conditions and events
- **Lab Results Panel**: Display recent lab results and measurements
- **Substance Use Tracking**: Tobacco, alcohol, and drug usage history
- **Family History**: Hereditary conditions and family medical history
- **Responsive Design**: Modern UI matching the provided mockup

## Architecture

```
frontend/
├── src/
│   ├── app/                 # Next.js App Router pages
│   │   ├── globals.css      # Global styles with Tailwind
│   │   ├── layout.tsx       # Root layout
│   │   └── page.tsx         # Main dashboard page
│   ├── components/          # React components
│   │   ├── Sidebar.tsx
│   │   ├── PatientHeader.tsx
│   │   ├── PatientSummary.tsx
│   │   ├── LabResultsPanel.tsx
│   │   ├── SubstanceUsePanel.tsx
│   │   ├── FamilyHistoryPanel.tsx
│   │   ├── AllergiesPanel.tsx
│   │   ├── OccupationPanel.tsx
│   │   ├── ChronologicalOverview.tsx
│   │   └── LoadingSpinner.tsx
│   └── lib/                 # Utilities and API
│       ├── api.ts           # Backend API client
│       └── types.ts         # TypeScript type definitions
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── next.config.js
```

## Prerequisites

- Node.js 18+ 
- npm or yarn
- Python backend running (see parent README)

## Installation

```bash
cd frontend
npm install
```

## Development

Start the development server:

```bash
npm run dev
```

The frontend will be available at http://localhost:3000

## Backend API

The frontend connects to the Python backend API running on http://localhost:8000. Make sure to start the backend first:

```bash
# From the project root
uv run python -m uvicorn api_server:app --reload --port 8000
```

## API Endpoints

The frontend communicates with these backend endpoints:

- `GET /api/applications` - List all applications
- `GET /api/applications/{id}` - Get application details
- `POST /api/applications` - Create new application (multipart form with PDF files)
- `POST /api/applications/{id}/extract` - Run content understanding extraction
- `POST /api/applications/{id}/analyze` - Run underwriting analysis

## Building for Production

```bash
npm run build
npm start
```

## Environment Variables

Create a `.env.local` file for local development:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## UI Components

### Sidebar
Navigation panel with application list and quick actions.

### PatientHeader
Displays patient demographics, BMI, and key metrics.

### PatientSummary
AI-generated summary with highlighted medical terms and conditions.

### LabResultsPanel
Recent laboratory results including cholesterol, blood pressure, and HbA1c.

### SubstanceUsePanel
Tabbed view of tobacco, alcohol, marijuana, and substance abuse history.

### ChronologicalOverview
Timeline view of medical conditions, treatments, and significant events.

## Styling

The UI uses:
- **Tailwind CSS** for utility-first styling
- **Lucide React** for icons
- Custom color palette matching the DigitalOwl mockup
- Responsive design for various screen sizes
