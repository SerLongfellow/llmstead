import React from 'react';
import { Cpu, Activity, Sparkles, Workflow } from 'lucide-react';

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
    <header className="glass-panel" style={{ borderRadius: '0 0 16px 16px', padding: '14px 24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        {/* Brand Logo & Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            background: 'linear-gradient(135deg, #6366f1 0%, #a855f7 100%)',
            padding: '10px',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 16px rgba(99, 102, 241, 0.4)'
          }}>
            <Sparkles size={22} color="#ffffff" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em', background: 'linear-gradient(to right, #ffffff, #c7d2fe)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              LLMStead
            </h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Raise your own models
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav style={{ display: 'flex', gap: '6px', background: 'rgba(15, 23, 42, 0.6)', padding: '4px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
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
                  background: isActive ? 'var(--primary)' : 'transparent',
                  color: isActive ? '#ffffff' : 'var(--text-muted)',
                  boxShadow: isActive ? '0 2px 10px rgba(99, 102, 241, 0.4)' : 'none'
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
