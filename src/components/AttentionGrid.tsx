import React, { useState } from 'react';
import { Eye, Lock } from 'lucide-react';

// Heatmap color: blends from the panel color (weight 0) to the primary blue (weight 1)
const getCellBg = (val: number, isMasked: boolean) => {
  if (isMasked) return 'var(--surface-inset)';
  return `color-mix(in srgb, var(--primary) ${Math.round(8 + val * 92)}%, var(--bg-card-hover))`;
};

interface AttentionGridProps {
  attentionMap: number[][]; // [query][key] softmax weights for one head
  tokenStrings: string[];
  highlightRow?: number;    // query row to outline (e.g. the token being followed)
  onSelectRow?: (row: number) => void;
}

/** One head's attention matrix plus a hover inspector. Shown in the Pipeline's block stages. */
export const AttentionGrid: React.FC<AttentionGridProps> = ({ attentionMap, tokenStrings, highlightRow, onSelectRow }) => {
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; val: number } | null>(null);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
      {/* Matrix Grid */}
      <div style={{ overflowX: 'auto', padding: '10px 0' }}>
        <table style={{ borderCollapse: 'separate', borderSpacing: '4px', margin: '0 auto' }}>
          <thead>
            <tr>
              <th style={{ padding: '6px', fontSize: '0.75rem', color: 'var(--text-dim)', whiteSpace: 'nowrap' }}>Q \ K</th>
              {tokenStrings.map((t, idx) => (
                <th key={idx} style={{ padding: '6px', fontSize: '0.8rem', color: 'var(--accent-emerald)', whiteSpace: 'pre' }} className="font-mono">
                  {t}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tokenStrings.map((queryToken, rowIdx) => {
              const isHighlighted = rowIdx === highlightRow;
              return (
                <tr
                  key={rowIdx}
                  onClick={onSelectRow ? () => onSelectRow(rowIdx) : undefined}
                  style={{ cursor: onSelectRow ? 'pointer' : undefined }}
                >
                  {/* Row Label (Query token) */}
                  <td
                    style={{
                      padding: '6px',
                      fontSize: '0.8rem',
                      fontWeight: 700,
                      color: isHighlighted ? 'var(--accent-amber)' : 'var(--accent-cyan)',
                      textAlign: 'right',
                      whiteSpace: 'pre',
                    }}
                    className="font-mono"
                  >
                    {queryToken}
                  </td>

                  {/* Columns (Key tokens) */}
                  {tokenStrings.map((_, colIdx) => {
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
                          border: isHighlighted ? '1px solid var(--accent-amber)' : '1px solid var(--border-color)',
                          borderRadius: '6px',
                          color: 'var(--text-main)',
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
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Attention Inspection Panel */}
      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            <Eye size={20} color="var(--accent-purple)" />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Attention Cell Inspection</h3>
          </div>

          {hoveredCell ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div style={{ background: 'var(--surface-inset)', padding: '16px', borderRadius: '10px', border: '1px solid var(--border-glow)' }}>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Query Token (Target Position {hoveredCell.row})</p>
                <p className="font-mono" style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--accent-cyan)', whiteSpace: 'pre' }}>
                  "{tokenStrings[hoveredCell.row]}"
                </p>
              </div>

              <div style={{ background: 'var(--surface-inset)', padding: '16px', borderRadius: '10px', border: '1px solid var(--border-glow)' }}>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Key Token (Attended Position {hoveredCell.col})</p>
                <p className="font-mono" style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--accent-emerald)', whiteSpace: 'pre' }}>
                  "{tokenStrings[hoveredCell.col]}"
                </p>
              </div>

              <div style={{
                background: 'var(--primary-soft)',
                padding: '20px',
                borderRadius: '12px',
                border: '1px solid var(--primary)',
                textAlign: 'center'
              }}>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Attention Score Weight (Softmax)</p>
                <p className="font-mono" style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--text-main)', margin: '4px 0' }}>
                  {(hoveredCell.val * 100).toFixed(1)}%
                </p>
                <p style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)' }}>
                  {hoveredCell.col > hoveredCell.row ? '🔒 Masked (Future token)' : 'Allowed Causal Attention'}
                </p>
              </div>
            </div>
          ) : (
            <div style={{ padding: '30px 20px', textAlign: 'center', color: 'var(--text-muted)', background: 'var(--surface-inset)', borderRadius: '10px', border: '1px dashed var(--border-color)' }}>
              <p style={{ fontSize: '0.85rem' }}>Hover over any cell in the matrix grid to inspect token-to-token attention weight scores.</p>
            </div>
          )}
        </div>

        <div style={{ padding: '12px', background: 'var(--primary-soft)', border: '1px solid var(--primary-soft)', borderRadius: '8px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <p>💡 <strong>Causal Masking</strong> ensures position i can only attend to previous positions j ≤ i. This enforces autoregressive left-to-right text generation!</p>
        </div>
      </div>
    </div>
  );
};
