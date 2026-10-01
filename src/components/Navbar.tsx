import React from 'react';
import { Home } from 'lucide-react';
import { ChickenLogo } from './ChickenLogo';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isTraining: boolean;
}

/** The app's path, in order. Start is the intro; the rest are numbered steps. */
export const TABS = [
  { id: 'start', label: 'Start here', step: null },
  { id: 'setup', label: 'Set up', step: 1 },
  { id: 'training', label: 'Train', step: 2 },
  { id: 'pipeline', label: 'Look inside', step: 3 },
] as const;

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  isTraining,
}) => {
  const tabs = TABS;

  return (
    <header className="glass-panel" style={{ borderRadius: '0 0 10px 10px', borderTop: 'none', padding: '12px 24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        {/* Brand Logo & Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            background: 'var(--primary)',
            padding: '8px',
            borderRadius: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <ChickenLogo size={22} style={{ color: '#ffffff' }} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.2rem', fontWeight: 700, letterSpacing: '-0.01em', color: 'var(--text-main)' }}>
              LLMStead
            </h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Raise your own models
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav style={{ display: 'flex', gap: '6px', background: 'var(--surface-inset)', padding: '4px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 14px',
                  borderRadius: '7px',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  background: isActive ? 'var(--bg-card-hover)' : 'transparent',
                  color: isActive ? 'var(--text-main)' : 'var(--text-muted)',
                  boxShadow: 'none'
                }}
              >
                {tab.step === null ? (
                  <Home size={15} />
                ) : (
                  <span
                    className="font-mono"
                    style={{
                      width: 18,
                      height: 18,
                      borderRadius: 9,
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      background: isActive ? 'var(--primary)' : 'var(--bg-card-hover)',
                      color: isActive ? '#ffffff' : 'var(--text-muted)',
                    }}
                  >
                    {tab.step}
                  </span>
                )}
                <span>{tab.label}</span>
                {tab.id === 'training' && isTraining && (
                  <span
                    title="Training is running"
                    style={{ width: 7, height: 7, borderRadius: 4, background: 'var(--accent-emerald)', animation: 'pulse 1.2s ease-in-out infinite' }}
                  />
                )}
              </button>
            );
          })}
        </nav>
      </div>
    </header>
  );
};
