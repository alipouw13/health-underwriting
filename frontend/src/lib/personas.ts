/**
 * Persona definitions and types for InsureAI.
 * This module defines the available personas and their UI configurations.
 *
 * Status tiers:
 *   - "active"      → fully functional, production-quality
 *   - "preview"     → UI exists, backend limited — shown with a Preview badge
 *   - "coming_soon" → placeholder only — disabled in the selector
 */

import { ClipboardList, Home, Car, Stethoscope } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

export type PersonaId = 'underwriting' | 'life_health_claims' | 'automotive_claims' | 'mortgage';

export type PersonaStatus = 'active' | 'preview' | 'coming_soon';

export interface Persona {
  id: PersonaId;
  name: string;
  description: string;
  icon: LucideIcon;
  color: string;
  enabled: boolean;
  status: PersonaStatus;
}

export interface PersonaConfig extends Persona {
  // UI-specific settings
  primaryColor: string;
  secondaryColor: string;
  accentColor: string;
}

/**
 * Persona registry with UI configurations
 */
export const PERSONAS: Record<PersonaId, PersonaConfig> = {
  underwriting: {
    id: 'underwriting',
    name: 'Underwriting',
    description: 'Life insurance underwriting wor...',
    icon: ClipboardList,
    color: '#6366f1',
    enabled: true,
    status: 'active',
    primaryColor: '#6366f1', // Indigo
    secondaryColor: '#818cf8',
    accentColor: '#4f46e5',
  },
  automotive_claims: {
    id: 'automotive_claims',
    name: 'Automotive Claims',
    description: 'Multimodal automotive claims w...',
    icon: Car,
    color: '#dc2626',
    enabled: true,
    status: 'active',
    primaryColor: '#dc2626', // Red
    secondaryColor: '#ef4444',
    accentColor: '#b91c1c',
  },
  life_health_claims: {
    id: 'life_health_claims',
    name: 'Life & Health Claims',
    description: 'Health insurance claims proces...',
    icon: Stethoscope,
    color: '#6366f1',
    enabled: true,
    status: 'preview',
    primaryColor: '#6366f1', // Indigo
    secondaryColor: '#818cf8',
    accentColor: '#4f46e5',
  },
  mortgage: {
    id: 'mortgage',
    name: 'Mortgage',
    description: 'Mortgage underwriting workben...',
    icon: Home,
    color: '#6366f1',
    enabled: false,
    status: 'coming_soon',
    primaryColor: '#6366f1', // Indigo
    secondaryColor: '#818cf8',
    accentColor: '#4f46e5',
  },
};

/** Ordered groups for the persona selector dropdown */
export const PERSONA_GROUPS: { label: string; statuses: PersonaStatus[] }[] = [
  { label: 'Active', statuses: ['active'] },
  { label: 'Preview', statuses: ['preview'] },
  { label: 'Coming Soon', statuses: ['coming_soon'] },
];

/**
 * Get persona configuration by ID
 */
export function getPersona(id: PersonaId): PersonaConfig {
  return PERSONAS[id];
}

/**
 * Get all available personas
 */
export function getAllPersonas(): PersonaConfig[] {
  return Object.values(PERSONAS);
}

/**
 * Get only enabled personas
 */
export function getEnabledPersonas(): PersonaConfig[] {
  return Object.values(PERSONAS).filter(p => p.enabled);
}

/**
 * Default persona
 */
export const DEFAULT_PERSONA: PersonaId = 'underwriting';

/**
 * Local storage key for persisting selected persona
 */
export const PERSONA_STORAGE_KEY = 'insureai-persona';
