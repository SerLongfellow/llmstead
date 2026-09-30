import React from 'react';

/** Make whitespace inside tokens visible */
export const showTok = (s: string) => s.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '→');

export const cosine = (a: number[], b: number[]) => {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
};

export const TokenChip: React.FC<{ text: string; active?: boolean; onClick?: () => void; bg?: string; title?: string }> = ({
  text,
  active,
  onClick,
  bg,
  title,
}) => (
  <span
    onClick={onClick}
    title={title}
    className="font-mono"
    style={{
      display: 'inline-block',
      padding: '2px 7px',
      borderRadius: 4,
      fontSize: '0.8rem',
      fontWeight: 600,
      cursor: onClick ? 'pointer' : 'default',
      whiteSpace: 'pre',
      color: 'var(--text-main)',
      background: bg ?? (active ? 'rgba(245, 158, 11, 0.25)' : 'rgba(16, 185, 129, 0.12)'),
      border: `1px solid ${active ? 'var(--accent-amber)' : 'transparent'}`,
    }}
  >
    {showTok(text)}
  </span>
);
