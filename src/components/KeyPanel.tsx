import React from 'react';

/**
 * A small two-column key: a symbol or term on the left, a plain-language meaning on the right.
 * Used for "Reading the formula" in Look inside and "Terms used here" in What's next.
 */
export const KeyPanel: React.FC<{ title: string; items: [term: string, meaning: React.ReactNode][] }> = ({ title, items }) => (
  <div style={{ margin: '0 0 12px', padding: '10px 14px', borderRadius: 8, background: 'var(--surface-inset)', border: '1px solid var(--border-color)' }}>
    <div style={{ fontSize: '0.7rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 6 }}>
      {title}
    </div>
    <div className="key-panel-grid" style={{ fontSize: '0.8rem', lineHeight: 1.5 }}>
      {items.map(([term, meaning]) => (
        <React.Fragment key={term}>
          <code className="font-mono" style={{ color: 'var(--primary)', whiteSpace: 'nowrap' }}>{term}</code>
          <span style={{ color: 'var(--text-muted)' }}>{meaning}</span>
        </React.Fragment>
      ))}
    </div>
  </div>
);
