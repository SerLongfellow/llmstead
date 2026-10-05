import React, { useEffect, useRef } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { THEME, withAlpha } from '../../styles/theme';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

export interface Series {
  label: string;
  color: string;
  data: (number | null)[];
  dashed?: boolean;
}

/** A small line chart of metrics over post-training steps (gaps where a series has no value) */
export const MetricChart: React.FC<{
  labels: string[];
  series: Series[];
  yTitle: string;
  /** Fixed y range, e.g. [0, 1] for rates */
  yRange?: [number, number];
  /** Re-measure the canvas when this turns true (the tab is kept mounted while hidden) */
  visible: boolean;
  height?: number;
}> = ({ labels, series, yTitle, yRange, visible, height = 220 }) => {
  const boxRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!visible) return;
    const timer = window.setTimeout(() => {
      const canvas = boxRef.current?.querySelector('canvas');
      if (canvas) ChartJS.getChart(canvas)?.resize();
    });
    return () => clearTimeout(timer);
  }, [visible]);

  const data = {
    labels,
    datasets: series.map(s => ({
      label: s.label,
      data: s.data,
      borderColor: s.color,
      backgroundColor: withAlpha(s.color, 0.12),
      borderDash: s.dashed ? [6, 4] : undefined,
      tension: 0.2,
      spanGaps: true,
      pointRadius: labels.length > 40 ? 0 : 2,
      pointHoverRadius: 4,
    })),
  };
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false as const,
    plugins: {
      legend: { labels: { color: THEME.textMuted, font: { family: 'Inter', size: 11 }, boxWidth: 14 } },
      tooltip: { mode: 'index' as const, intersect: false },
    },
    scales: {
      x: { ticks: { color: THEME.textDim, maxTicksLimit: 8 }, grid: { color: withAlpha(THEME.border, 0.6) } },
      y: {
        min: yRange?.[0],
        max: yRange?.[1],
        title: { display: true, text: yTitle, color: THEME.textMuted, font: { family: 'Inter', size: 11 } },
        ticks: { color: THEME.textDim },
        grid: { color: withAlpha(THEME.border, 0.6) },
      },
    },
  };
  return (
    <div ref={boxRef} style={{ height, position: 'relative', width: '100%' }}>
      <Line data={data} options={options} />
    </div>
  );
};
