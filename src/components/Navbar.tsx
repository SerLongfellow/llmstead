import React from 'react';
import { Cpu, Activity, Workflow } from 'lucide-react';
import { ChickenLogo } from './ChickenLogo';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  paramCount: number;
  datasetName: string;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  paramCount,
  datasetName,
}) => {
  const tabs = [
    { id: 'setup', label: 'Setup', icon: Cpu },
    { id: 'pipeline', label: 'Pipeline', icon: Workflow },
    { id: 'training', label: 'Training Dashboard', icon: Activity },
  ];

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
            const Icon = tab.icon;
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
                <Icon size={16} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Status Badges */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div className="badge badge-purple" style={{ fontFamily: 'var(--font-mono)' }}>
            Dataset: {datasetName}
          </div>
          <div className="badge badge-primary" style={{ fontFamily: 'var(--font-mono)' }}>
            {paramCount.toLocaleString()} Params
          </div>
        </div>
      </div>
    </header>
  );
};
