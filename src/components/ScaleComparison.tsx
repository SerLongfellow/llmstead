import React, { useLayoutEffect, useRef, useState } from 'react';
import { TransformerConfig } from '../types';
import { REFERENCE_MODELS, FRONTIER_NOTE, compact } from '../engine/referenceModels';

interface ScaleComparisonProps {
  config: TransformerConfig; // effective config (real vocab size)
  paramCount: number;
}

const th: React.CSSProperties = { padding: '8px 10px', fontWeight: 600, textAlign: 'right', whiteSpace: 'nowrap' };
const td: React.CSSProperties = { padding: '8px 10px', textAlign: 'right', whiteSpace: 'nowrap' };

/** Your model next to real LLMs: a log-scale parameter chart and a spec table */
export const ScaleComparison: React.FC<ScaleComparisonProps> = ({ config, paramCount }) => {
  const rows = [
    {
      name: 'Your model',
      year: null as number | null,
      params: paramCount,
      dModel: config.dModel,
      layers: config.numLayers,
      heads: config.numHeads,
      contextWindow: config.contextWindow,
      vocabSize: config.vocabSize,
      trainingTokens: undefined as number | undefined,
      mine: true,
    },
    ...REFERENCE_MODELS.map(m => ({ ...m, mine: false })),
  ];

  // Log scale from 10K to 1T parameters: each step of the axis is 10× more
  const LOG_MIN = 4;
  const LOG_MAX = 12;
  const pct = (n: number) => ((Math.log10(n) - LOG_MIN) / (LOG_MAX - LOG_MIN)) * 100;
  // Label every power of ten if the axis is wide enough, else every 2nd or 4th (phones)
  const axisRef = useRef<HTMLDivElement>(null);
  const [axisWidth, setAxisWidth] = useState(400);
  useLayoutEffect(() => {
    const el = axisRef.current;
    if (!el) return;
    setAxisWidth(el.getBoundingClientRect().width);
    if (typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(([e]) => setAxisWidth(e.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const labelStep = axisWidth >= 300 ? 1 : axisWidth >= 160 ? 2 : 4;
  const gpt3 = REFERENCE_MODELS.find(m => m.name === 'GPT-3')!;
  const r = gpt3.params / paramCount;
  const ratio = r >= 1e6 ? `${(r / 1e6).toFixed(1).replace(/\.0$/, '')} million` : r >= 1e3 ? `${Math.round(r / 1e3).toLocaleString()} thousand` : Math.round(r).toLocaleString();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
      <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
        Your model has <b style={{ color: 'var(--text-main)' }}>{paramCount.toLocaleString()}</b> parameters. GPT-3 has about{' '}
        <b style={{ color: 'var(--text-main)' }}>{ratio} times</b> as many. The architecture is the same kind
        of transformer; real models are just vastly wider, deeper and trained on far more text.
      </p>

      {/* Parameter count on a log scale */}
      <div>
        <div style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: 8 }}>Parameters (log scale: each gridline is 10× more)</div>
        <div style={{ position: 'relative', display: 'flex', flexDirection: 'column', gap: 6 }}>
          {/* gridlines */}
          <div style={{ position: 'absolute', left: 130, right: 60, top: 0, bottom: 0, pointerEvents: 'none' }}>
            {Array.from({ length: LOG_MAX - LOG_MIN + 1 }, (_, i) => (
              <div key={i} style={{ position: 'absolute', left: `${(i / (LOG_MAX - LOG_MIN)) * 100}%`, top: 0, bottom: 0, borderLeft: '1px dashed var(--border-color)' }} />
            ))}
          </div>
          {[...rows].sort((a, b) => a.params - b.params).map(r => (
            <div key={r.name} style={{ display: 'grid', gridTemplateColumns: '130px minmax(0, 1fr) 60px', alignItems: 'center', fontSize: '0.78rem' }}>
              <span style={{ color: r.mine ? 'var(--accent-amber)' : 'var(--text-muted)', fontWeight: r.mine ? 700 : 500 }}>{r.name}</span>
              <div style={{ height: 12, position: 'relative' }}>
                <div
                  style={{
                    width: `${Math.max(1, pct(r.params))}%`,
                    height: '100%',
                    borderRadius: 3,
                    background: r.mine ? 'var(--accent-amber)' : 'var(--primary)',
                  }}
                />
              </div>
              <span className="font-mono" style={{ textAlign: 'right', color: r.mine ? 'var(--accent-amber)' : 'var(--text-main)' }}>{compact(r.params)}</span>
            </div>
          ))}
          <div style={{ display: 'grid', gridTemplateColumns: '130px minmax(0, 1fr) 60px', fontSize: '0.65rem', color: 'var(--text-dim)' }}>
            <span />
            <div ref={axisRef} style={{ display: 'flex', justifyContent: 'space-between' }}>
              {Array.from({ length: Math.floor((LOG_MAX - LOG_MIN) / labelStep) + 1 }, (_, i) => (
                <span key={i} className="font-mono">{compact(10 ** (LOG_MIN + labelStep * i))}</span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Spec table */}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border-color)', color: 'var(--text-muted)' }}>
              <th style={{ ...th, textAlign: 'left' }}>Model</th>
              <th style={th}>Params</th>
              <th style={th}>d_model</th>
              <th style={th}>Layers</th>
              <th style={th}>Heads</th>
              <th style={th}>Context</th>
              <th style={th}>Vocab</th>
              <th style={th}>Trained on</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(r => (
              <tr
                key={r.name}
                style={{
                  borderBottom: '1px solid var(--border-color)',
                  background: r.mine ? 'var(--amber-soft)' : undefined,
                  color: r.mine ? 'var(--accent-amber)' : 'var(--text-main)',
                }}
              >
                <td style={{ ...td, textAlign: 'left', fontWeight: r.mine ? 700 : 500 }}>
                  {r.name}
                  {r.year && <span style={{ color: 'var(--text-dim)', fontWeight: 400 }}> ({r.year})</span>}
                </td>
                <td style={td} className="font-mono">{compact(r.params)}</td>
                <td style={td} className="font-mono">{r.dModel.toLocaleString()}</td>
                <td style={td} className="font-mono">{r.layers}</td>
                <td style={td} className="font-mono">{r.heads}</td>
                <td style={td} className="font-mono">{r.contextWindow.toLocaleString()}</td>
                <td style={td} className="font-mono">{r.vocabSize.toLocaleString()}</td>
                <td style={td} className="font-mono">{r.trainingTokens ? `${compact(r.trainingTokens)} tokens` : '—'}</td>
              </tr>
            ))}
            <tr style={{ color: 'var(--text-muted)' }}>
              <td style={{ ...td, textAlign: 'left' }}>{FRONTIER_NOTE.name}</td>
              <td style={{ ...td, textAlign: 'left' }} colSpan={7}>
                Size and architecture not published. Context windows: {FRONTIER_NOTE.contextRange}.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};
