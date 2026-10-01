// Theme colors for code that draws on <canvas> or needs raw color values (Chart.js,
// heatmaps, the embedding map), since those can't read CSS variables.
// Mirrors the :root variables in main.css. Keep the two in sync.

export const THEME = {
  bg: '#0d1117',
  surface: '#161b22',
  border: '#30363d',
  textMain: '#e6edf3',
  textMuted: '#9198a1',
  textDim: '#6e7681',
  primary: '#2f81f7',
  cyan: '#39c5cf',
  emerald: '#3fb950',
  amber: '#d29922',
  purple: '#a371f7',
  rose: '#f85149',
} as const;

/** '#rrggbb' → [r, g, b] */
export const rgb = (hex: string): [number, number, number] => [
  parseInt(hex.slice(1, 3), 16),
  parseInt(hex.slice(3, 5), 16),
  parseInt(hex.slice(5, 7), 16),
];

/** '#rrggbb' + alpha (0–1) → 'rgba(r, g, b, a)' */
export const withAlpha = (hex: string, alpha: number) => {
  const [r, g, b] = rgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};
