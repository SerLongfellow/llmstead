import React from 'react';
import { BPETokenizer } from '../../engine/bpeTokenizer';
import { TokenizedExample } from '../../engine/posttrain/examples';
import { PairScore, PreparedPair } from '../../engine/posttrain/dpo';
import { RlGroup } from '../../engine/posttrain/grpo';
import { ExampleTokens } from './ExampleTokens';
import { showTok } from '../tokenUi';

const caption: React.CSSProperties = { fontSize: '0.75rem', color: 'var(--text-dim)', lineHeight: 1.5 };
const show = (s: string) => showTok(s) || '∅';

/** SFT: one training example, each response token coloured by how likely the model now finds it */
export const SftInspector: React.FC<{
  example: TokenizedExample;
  probs: number[];
  tokenizer: BPETokenizer;
  maskPrompt: boolean;
}> = ({ example, probs, tokenizer, maskPrompt }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
    <ExampleTokens example={example} tokenizer={tokenizer} probs={probs} maskPrompt={maskPrompt} />
    <p style={caption}>
      {maskPrompt
        ? 'Grey tokens are the prompt: the model reads them but is never graded on them. Coloured tokens are the response, '
        : 'The prompt is not masked, so every token is graded, including the question itself. '}
      from red (the model finds the next token unlikely) to green (it would write it). Hover a token for its probability.
    </p>
  </div>
);

/** A signed bar for a log-probability change */
const ShiftBar: React.FC<{ value: number; max: number; color: string }> = ({ value, max, color }) => {
  const w = Math.min(50, (Math.abs(value) / Math.max(max, 1e-9)) * 50);
  return (
    <div style={{ position: 'relative', height: 8, background: 'var(--bg-card-hover)', borderRadius: 4, minWidth: 80 }}>
      <div style={{ position: 'absolute', left: '50%', top: -2, bottom: -2, width: 1, background: 'var(--border-strong)' }} />
      <div
        style={{
          position: 'absolute', top: 0, bottom: 0, borderRadius: 4, background: color,
          left: value >= 0 ? '50%' : `${50 - w}%`, width: `${w}%`,
        }}
      />
    </div>
  );
};

/** DPO: every pair's log-probability change since the start, chosen vs rejected */
export const DpoInspector: React.FC<{ pairs: PreparedPair[]; scores: PairScore[] }> = ({ pairs, scores }) => {
  const max = Math.max(1, ...scores.flatMap(s => [Math.abs(s.chosenShift), Math.abs(s.rejectedShift)]));
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.78rem' }}>
          <thead>
            <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
              <th style={{ padding: '4px 6px' }}>Prompt</th>
              <th style={{ padding: '4px 6px', color: 'var(--accent-emerald)' }}>Chosen: Δ log P</th>
              <th style={{ padding: '4px 6px', color: 'var(--accent-rose)' }}>Rejected: Δ log P</th>
              <th style={{ padding: '4px 6px' }}>Margin</th>
            </tr>
          </thead>
          <tbody>
            {pairs.map((p, i) => {
              const s = scores[i];
              if (!s) return null;
              return (
                <tr key={i} style={{ borderTop: '1px solid var(--border-color)' }}>
                  <td className="font-mono" style={{ padding: '4px 6px', color: 'var(--text-muted)', maxWidth: 160, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {show(p.pair.prompt)}
                  </td>
                  <td style={{ padding: '4px 6px' }}>
                    <div className="font-mono" style={{ fontSize: '0.7rem', marginBottom: 2 }}>{show(p.pair.chosen)} <span style={{ color: 'var(--text-dim)' }}>{s.chosenShift >= 0 ? '+' : ''}{s.chosenShift.toFixed(2)}</span></div>
                    <ShiftBar value={s.chosenShift} max={max} color="var(--accent-emerald)" />
                  </td>
                  <td style={{ padding: '4px 6px' }}>
                    <div className="font-mono" style={{ fontSize: '0.7rem', marginBottom: 2 }}>{show(p.pair.rejected)} <span style={{ color: 'var(--text-dim)' }}>{s.rejectedShift >= 0 ? '+' : ''}{s.rejectedShift.toFixed(2)}</span></div>
                    <ShiftBar value={s.rejectedShift} max={max} color="var(--accent-rose)" />
                  </td>
                  <td className="font-mono" style={{ padding: '4px 6px', fontWeight: 700, color: s.margin > 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                    {s.margin.toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p style={caption}>
        Δ log P is how much more (or less) likely the model is to write each response than the frozen starting model was. The
        margin is β × (chosen Δ − rejected Δ); DPO only needs it to be positive. If both go down, the model is learning to
        avoid the rejected reply faster than it learns the chosen one.
      </p>
    </div>
  );
};

/** RL: the latest step's groups — every sampled answer, its reward, and its advantage */
export const RlInspector: React.FC<{ groups: RlGroup[] }> = ({ groups }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
    {groups.map((g, gi) => (
      <div key={gi}>
        <p className="font-mono" style={{ fontSize: '0.8rem', marginBottom: 4 }}>
          {show(g.task.prompt)}
          {g.noSignal && (
            <span style={{ marginLeft: 8, fontFamily: 'var(--font-sans)', fontSize: '0.7rem', color: 'var(--accent-amber)' }}>
              all {g.samples[0]?.reward ? 'right' : 'wrong'}: no signal, nothing learned from this group
            </span>
          )}
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {g.samples.map((s, si) => (
            <span
              key={si}
              className="font-mono"
              title={`reward ${s.reward}, advantage ${s.advantage.toFixed(2)}`}
              style={{
                fontSize: '0.75rem', padding: '2px 7px', borderRadius: 4, whiteSpace: 'pre',
                border: `1px solid ${s.reward ? 'var(--accent-emerald)' : 'var(--border-color)'}`,
                background: s.advantage > 0
                  ? `color-mix(in srgb, var(--accent-emerald) ${Math.min(60, 20 + s.advantage * 20)}%, transparent)`
                  : s.advantage < 0
                    ? `color-mix(in srgb, var(--accent-rose) ${Math.min(60, 20 - s.advantage * 20)}%, transparent)`
                    : 'var(--surface-inset)',
              }}
            >
              {s.reward ? '✓ ' : '✗ '}
              {show(s.text.split('\n')[0])}
            </span>
          ))}
        </div>
      </div>
    ))}
    <p style={caption}>
      Each question gets a group of sampled answers. A checker marks them right (✓) or wrong (✗), and each answer's
      advantage is its reward minus the group average, divided by the group's spread: green answers are made more likely,
      red ones less. A group that is all right or all wrong teaches nothing.
    </p>
  </div>
);
