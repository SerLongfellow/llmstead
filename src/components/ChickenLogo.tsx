import React from 'react';

// Rooster, chest up, facing right, on a 24 × 24 grid. Shared with public/favicon.svg.
export const ROOSTER_PATHS = {
  // comb (four rounded lobes)
  comb: 'M9.3 6.5 C7.8 5.7 8 3.5 9.7 4 C9.6 2 11.7 1.6 12.2 3.4 C12.7 1.4 14.9 1.5 14.8 3.6 C16.1 3.1 17.2 4.7 15.6 6.7 Z',
  // head, neck hackles (the pointed feathers down the back) and puffed chest
  body:
    'M12.2 5.5 C13.9 5.5 15.2 6.3 15.5 7.5 L15.4 9.8 L13.3 12.2 L13.6 15.3 ' +
    'C16.8 15.5 18.7 17.4 18.4 19.8 L18 22 L4.2 22 ' +
    'L4.8 18.8 L3.4 17.8 L5.4 16.8 L4.2 15.2 L6.4 14.6 L5.6 12.8 L7.6 12.5 L7.3 10.6 L9.1 10.3 ' +
    'C8.9 7.4 10.2 5.5 12.2 5.5 Z',
  beak: 'M15.2 7.3 L19 8.5 L15.2 9.7 Z',
  wattle: 'M14.6 10 C16.2 10.2 16.4 12.4 15.5 13.3 C14.8 14 14 13.3 14.1 12.3 C14.2 11.4 14.2 10.6 14.6 10 Z',
  eye: { cx: 13.6, cy: 7.9, r: 0.95 },
};

/**
 * LLMStead's logo: a rooster silhouette in `currentColor`. `bgColor` should match whatever
 * the logo sits on; it draws the eye.
 */
export const ChickenLogo: React.FC<{ size?: number; bgColor?: string; style?: React.CSSProperties }> = ({
  size = 20,
  bgColor = 'var(--primary)',
  style,
}) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" style={style}>
    <path d={ROOSTER_PATHS.comb} fill="currentColor" stroke="currentColor" strokeWidth={0.7} strokeLinejoin="round" />
    <path d={ROOSTER_PATHS.body} fill="currentColor" strokeLinejoin="round" />
    <path d={ROOSTER_PATHS.beak} fill="currentColor" />
    <path d={ROOSTER_PATHS.wattle} fill="currentColor" />
    <circle {...ROOSTER_PATHS.eye} fill={bgColor} />
  </svg>
);
