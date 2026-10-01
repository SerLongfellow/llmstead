import React, { useState } from 'react';
import { HelpCircle } from 'lucide-react';

interface InfoTooltipProps {
  title: string;
  description: string;
  impact: string;
}

export const InfoTooltip: React.FC<InfoTooltipProps> = ({ title, description, impact }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div style={{ position: 'relative', display: 'inline-flex', alignItems: 'center' }}>
      <button
        type="button"
        onMouseEnter={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
        onClick={() => setIsOpen(!isOpen)}
        style={{
          background: 'none',
          border: 'none',
          padding: '2px',
          cursor: 'pointer',
          color: 'var(--text-dim)',
          display: 'flex',
          alignItems: 'center',
          transition: 'color 0.2s ease',
        }}
        aria-label="Info"
      >
        <HelpCircle size={15} color={isOpen ? 'var(--primary)' : 'var(--text-muted)'} />
      </button>

      {isOpen && (
        <div
          style={{
            position: 'absolute',
            bottom: '100%',
            left: '50%',
            transform: 'translateX(-50%) translateY(-8px)',
            width: '260px',
            padding: '12px 14px',
            background: 'var(--surface-inset)',
            backdropFilter: 'blur(12px)',
            border: '1px solid var(--primary)',
            borderRadius: '8px',
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
            zIndex: 100,
            fontSize: '0.78rem',
            lineHeight: 1.4,
            pointerEvents: 'none',
          }}
        >
          <p style={{ fontWeight: 700, color: 'var(--accent-cyan)', marginBottom: '4px' }}>
            {title}
          </p>
          <p style={{ color: 'var(--text-main)', marginBottom: '6px' }}>
            {description}
          </p>
          <p style={{ color: 'var(--accent-emerald)', fontSize: '0.73rem', fontWeight: 600 }}>
            ⚡ <strong>Impact:</strong> {impact}
          </p>

          {/* Triangle Pointer */}
          <div
            style={{
              position: 'absolute',
              top: '100%',
              left: '50%',
              transform: 'translateX(-50%)',
              width: 0,
              height: 0,
              borderLeft: '6px solid transparent',
              borderRight: '6px solid transparent',
              borderTop: '6px solid var(--primary)',
            }}
          />
        </div>
      )}
    </div>
  );
};
