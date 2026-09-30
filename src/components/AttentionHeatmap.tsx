import React, { useState } from 'react';
import { StepInspectionData, TransformerConfig } from '../types';
import { Grid, Eye, Lock } from 'lucide-react';

interface AttentionHeatmapProps {
  inspectionData: StepInspectionData;
  config: TransformerConfig;
}

export const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({
  inspectionData,
  config,
}) => {
  const [selectedLayerIndex, setSelectedLayerIndex] = useState<number>(0);
  const [selectedHeadIndex, setSelectedHeadIndex] = useState<number>(0);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; val: number } | null>(null);

  const layerData = inspectionData.layerInspections[selectedLayerIndex] || inspectionData.layerInspections[0];
  const attentionMap = layerData?.attentionWeights[selectedHeadIndex] || [];
  const tokenStrings = inspectionData.tokenStrings;

  // Heatmap color generator (0.0 -> dark, 1.0 -> vivid magenta/cyan)
  const getCellBg = (val: number, isMasked: boolean) => {
    if (isMasked) return 'rgba(15, 23, 42, 0.9)';
    
    // Smooth HSL color scale from dark indigo (240deg) to glowing cyan/pink (320deg)
    const hue = 240 + val * 80;
    const lightness = 15 + val * 55;
    const alpha = 0.3 + val * 0.7;
    return `hsla(${hue}, 90%, ${lightness}%, ${alpha})`;
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '20px' }}>
      {/* Controls & Heatmap Grid */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px', marginBottom: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Grid size={20} color="var(--accent-cyan)" />
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Multi-Head Attention Map</h2>
          </div>

          {/* Layer & Head Selectors */}
          <div style={{ display: 'flex', gap: '12px' }}>
            {/* Layer selector */}
            <select
              value={selectedLayerIndex}
              onChange={(e) => setSelectedLayerIndex(Number(e.target.value))}
              style={{ fontSize: '0.85rem' }}
            >
              {inspectionData.layerInspections.map((_, idx) => (
                <option key={idx} value={idx}>Layer {idx + 1}</option>
              ))}
            </select>

            {/* Head selector */}
            <select
              value={selectedHeadIndex}
              onChange={(e) => setSelectedHeadIndex(Number(e.target.value))}
              style={{ fontSize: '0.85rem' }}
            >
              {Array.from({ length: config.numHeads }).map((_, idx) => (
                <option key={idx} value={idx}>Head {idx + 1}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Matrix Grid */}
        <div style={{ overflowX: 'auto', padding: '10px 0' }}>
          <table style={{ borderCollapse: 'separate', borderSpacing: '4px', margin: '0 auto' }}>
            <thead>
              <tr>
                <th style={{ padding: '6px', fontSize: '0.75rem', color: 'var(--text-dim)' }}>Q \ K</th>
                {tokenStrings.map((t, idx) => (
                  <th key={idx} style={{ padding: '6px', fontSize: '0.8rem', color: 'var(--accent-emerald)' }} className="font-mono">
                    {t}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tokenStrings.map((queryToken, rowIdx) => (
                <tr key={rowIdx}>
                  {/* Row Label (Query token) */}
                  <td style={{ padding: '6px', fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent-cyan)', textAlign: 'right' }} className="font-mono">
                    {queryToken}
                  </td>

                  {/* Columns (Key tokens) */}
                  {tokenStrings.map((keyToken, colIdx) => {
                    const isMasked = colIdx > rowIdx; // Causal Masking (cannot look ahead)
                    const val = attentionMap[rowIdx]?.[colIdx] ?? 0;

                    return (
                      <td
                        key={colIdx}
                        className="heatmap-cell"
                        onMouseEnter={() => setHoveredCell({ row: rowIdx, col: colIdx, val })}
                        onMouseLeave={() => setHoveredCell(null)}
                        style={{
                          width: '44px',
                          height: '44px',
                          textAlign: 'center',
                          verticalAlign: 'middle',
                          background: getCellBg(val, isMasked),
                          border: '1px solid rgba(255, 255, 255, 0.08)',
                          borderRadius: '6px',
                          color: '#ffffff',
                          fontSize: '0.75rem',
                          fontFamily: 'var(--font-mono)'
                        }}
                      >
                        {isMasked ? (
                          <Lock size={12} color="var(--text-dim)" style={{ margin: '0 auto' }} />
                        ) : (
                          val.toFixed(2)
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Right Column: Attention Inspection Panel */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            <Eye size={20} color="var(--accent-purple)" />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Attention Cell Inspection</h3>
          </div>

          {hoveredCell ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '16px', borderRadius: '10px', border: '1px solid var(--border-glow)' }}>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Query Token (Target Position {hoveredCell.row})</p>
                <p className="font-mono" style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--accent-cyan)' }}>
                  "{tokenStrings[hoveredCell.row]}"
                </p>
              </div>

              <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '16px', borderRadius: '10px', border: '1px solid var(--border-glow)' }}>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Key Token (Attended Position {hoveredCell.col})</p>
                <p className="font-mono" style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--accent-emerald)' }}>
                  "{tokenStrings[hoveredCell.col]}"
                </p>
              </div>

              <div style={{
                background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(168, 85, 247, 0.25) 100%)',
                padding: '20px',
                borderRadius: '12px',
                border: '1px solid var(--primary)',
                textAlign: 'center'
              }}>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Attention Score Weight (Softmax)</p>
                <p className="font-mono" style={{ fontSize: '2rem', fontWeight: 800, color: '#ffffff', margin: '4px 0' }}>
                  {(hoveredCell.val * 100).toFixed(1)}%
                </p>
                <p style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)' }}>
                  {hoveredCell.col > hoveredCell.row ? '🔒 Masked (Future token)' : 'Allowed Causal Attention'}
                </p>
              </div>
            </div>
          ) : (
            <div style={{ padding: '30px 20px', textAlign: 'center', color: 'var(--text-muted)', background: 'rgba(15, 23, 42, 0.4)', borderRadius: '10px', border: '1px dashed var(--border-color)' }}>
              <p style={{ fontSize: '0.85rem' }}>Hover over any cell in the matrix grid to inspect token-to-token attention weight scores.</p>
            </div>
          )}
        </div>

        <div style={{ padding: '12px', background: 'rgba(99, 102, 241, 0.08)', border: '1px solid rgba(99, 102, 241, 0.2)', borderRadius: '8px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <p>💡 <strong>Causal Masking</strong> ensures position $i$ can only attend to previous positions $j \le i$. This enforces autoregressive left-to-right text generation!</p>
        </div>
      </div>
    </div>
  );
};
