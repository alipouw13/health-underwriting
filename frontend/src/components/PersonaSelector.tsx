'use client';

import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Check } from 'lucide-react';
import { usePersona } from '@/lib/PersonaContext';
import { getAllPersonas, PersonaConfig } from '@/lib/personas';
import clsx from 'clsx';

export default function PersonaSelector() {
  const { currentPersona, personaConfig, setPersona } = usePersona();
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  
  const allPersonas = getAllPersonas();

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelectPersona = (persona: PersonaConfig) => {
    if (persona.enabled) {
      setPersona(persona.id);
      setIsOpen(false);
    }
  };

  const IconComponent = personaConfig.icon;

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Selector Button - matches Application Selector styling */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 bg-slate-50 hover:bg-slate-100 rounded-lg border border-slate-200 transition-colors"
      >
        <IconComponent className="w-4 h-4 text-slate-600" />
        <span className="text-sm font-medium text-slate-700">{personaConfig.name}</span>
        <ChevronDown 
          className={clsx(
            'w-4 h-4 text-slate-500 transition-transform',
            isOpen && 'rotate-180'
          )} 
        />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setIsOpen(false)} />
          <div className="absolute top-full left-0 mt-1 w-64 bg-white rounded-lg shadow-lg border border-slate-200 py-1 z-20">
            {allPersonas.map((persona) => {
              const PersonaIcon = persona.icon;
              return (
                <button
                  key={persona.id}
                  onClick={() => handleSelectPersona(persona)}
                  disabled={!persona.enabled}
                  className={clsx(
                    'w-full text-left px-4 py-2 text-sm transition-colors flex items-center gap-3',
                    persona.enabled 
                      ? 'hover:bg-slate-50 cursor-pointer' 
                      : 'opacity-50 cursor-not-allowed',
                    currentPersona === persona.id && 'bg-indigo-50 text-indigo-700'
                  )}
                >
                  <PersonaIcon className="w-4 h-4 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium truncate">{persona.name}</span>
                      {currentPersona === persona.id && (
                        <Check className="w-3 h-3" />
                      )}
                      {!persona.enabled && (
                        <span className="text-[10px] px-1.5 py-0.5 bg-slate-100 text-slate-500 rounded font-medium">
                          Coming Soon
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-500 truncate">
                      {persona.description}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
