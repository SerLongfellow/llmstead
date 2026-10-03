import React from 'react';

export interface ExternalLink {
  label: string;
  url: string;
}

/** A short row of outside links for readers who want the full story */
export const GoDeeper: React.FC<{ links: ExternalLink[] }> = ({ links }) => (
  <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)', margin: '0 0 12px', display: 'flex', flexWrap: 'wrap', gap: '4px 12px' }}>
    <span>Go deeper:</span>
    {links.map(l => (
      <a key={l.url} href={l.url} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)' }}>
        {l.label}
      </a>
    ))}
  </p>
);
