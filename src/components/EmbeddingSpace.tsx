import React, { useMemo, useState } from 'react';
import { BPETokenizer } from '../engine/bpeTokenizer';
import { showTok, TokenChip } from './tokenUi';
import { THEME } from '../styles/theme';

interface EmbeddingSpaceProps {
  embeddings: number[][];       // snapshot of the token embedding table [vocab × dModel]
  idToToken: Map<number, string>;
  tokenizer: BPETokenizer;
  focusTokenId: number;         // the token being followed in the Pipeline
}

type Vec = number[];

const SPECIAL = /^<(PAD|UNK|BOS|EOS)>$/;

const EXAMPLES = ['4 - 2 + 7', 'A - a + b', '"the " - t + a', 'M + a'];

const add = (a: Vec, b: Vec, s = 1) => a.map((v, i) => v + s * b[i]);
const dot = (a: Vec, b: Vec) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const norm = (a: Vec) => Math.sqrt(dot(a, a));

/** Rough token categories, used to color the map */
const category = (t: string) => {
  if (/^\d$/.test(t)) return { label: 'digit', color: THEME.cyan };
  if (/^[A-Z]$/.test(t)) return { label: 'uppercase letter', color: THEME.amber };
  if (/^[a-z]$/.test(t)) return { label: 'lowercase letter', color: THEME.emerald };
  if (t.length === 1) return { label: 'space / punctuation', color: THEME.textMuted };
  return { label: 'merged token', color: THEME.purple };
};
const LEGEND = ([['digit', '1'], ['uppercase letter', 'A'], ['lowercase letter', 'a'], ['space / punctuation', ','], ['merged token', 'ab']] as const).map(
  ([label, sample]) => ({ label, color: category(sample).color })
);

/**
 * Top-2 principal components of the (centered) embeddings via power iteration on the
 * d × d covariance matrix. Returns the mean, the two directions and the share of total
 * variance each one explains.
 */
function pca2(rows: Vec[]) {
  const d = rows[0].length;
  const mean = new Array(d).fill(0);
  for (const r of rows) for (let j = 0; j < d; j++) mean[j] += r[j] / rows.length;
  const X = rows.map(r => r.map((v, j) => v - mean[j]));
  const C: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const x of X) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += x[i] * x[j];
  const trace = C.reduce((s, row, i) => s + row[i], 0) || 1;

  const components: { dir: Vec; share: number }[] = [];
  for (let k = 0; k < 2; k++) {
    let v: Vec = Array.from({ length: d }, (_, i) => Math.sin(i + 1 + k)); // deterministic start
    for (let it = 0; it < 200; it++) {
      let w = C.map(row => dot(row, v));
      for (const { dir } of components) w = add(w, dir, -dot(w, dir)); // stay orthogonal to earlier PCs
      const n = norm(w) || 1;
      v = w.map(x => x / n);
    }
    const lambda = dot(v, C.map(row => dot(row, v)));
    components.push({ dir: v, share: lambda / trace });
  }
  return { mean, components };
}

export const EmbeddingSpace: React.FC<EmbeddingSpaceProps> = ({ embeddings, idToToken, tokenizer, focusTokenId }) => {
  const [expr, setExpr] = useState<string>(EXAMPLES[0]);
  const [excludeInputs, setExcludeInputs] = useState<boolean>(true);
  const [hovered, setHovered] = useState<number | null>(null);

  // Every real (non-special) token and its vector
  const ids = useMemo(
    () => embeddings.map((_, id) => id).filter(id => idToToken.has(id) && !SPECIAL.test(idToToken.get(id)!)),
    [embeddings, idToToken]
  );
  const norms = useMemo(() => embeddings.map(norm), [embeddings]);

  const nearest = (v: Vec, k: number, exclude: Set<number>) => {
    const nv = norm(v) || 1;
    return ids
      .filter(id => !exclude.has(id))
      .map(id => ({ id, sim: dot(v, embeddings[id]) / (nv * (norms[id] || 1)) }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, k);
  };

  // ── Token math ──
  // Terms are separated by " + " or " - " (spaces required, so "-" can still be a token).
  // Quote a term to keep leading/trailing spaces, e.g. "the ".
  const parsed = useMemo(() => {
    const parts = expr.trim().split(/\s+([+-])\s+/);
    const terms: { sign: number; text: string; tokenIds: number[]; tokenStrs: string[] }[] = [];
    for (let i = 0; i < parts.length; i += 2) {
      const raw = parts[i].trim();
      const quoted = raw.match(/^"(.*)"$/);
      const text = quoted ? quoted[1] : raw;
      if (!text) return { error: 'Each term needs some text, e.g. 4 - 2 + 7' };
      const enc = tokenizer.encode(text);
      terms.push({ sign: i === 0 || parts[i - 1] === '+' ? 1 : -1, text, tokenIds: enc.tokens, tokenStrs: enc.tokenStrings });
    }
    return { terms };
  }, [expr, tokenizer]);

  const mathResult = useMemo(() => {
    if (!('terms' in parsed) || !parsed.terms) return null;
    const d = embeddings[0].length;
    let v: Vec = new Array(d).fill(0);
    for (const t of parsed.terms) {
      // A multi-token term is represented by the average of its tokens' vectors
      let tv: Vec = new Array(d).fill(0);
      for (const id of t.tokenIds) tv = add(tv, embeddings[id], 1 / t.tokenIds.length);
      v = add(v, tv, t.sign);
    }
    const inputIds = new Set(parsed.terms.flatMap(t => t.tokenIds));
    return { v, top: nearest(v, 8, excludeInputs ? inputIds : new Set()), inputIds };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [parsed, embeddings, excludeInputs, ids, norms]);

  // ── Neighbors of the followed token ──
  const focusNeighbors = useMemo(
    () => (embeddings[focusTokenId] ? nearest(embeddings[focusTokenId], 8, new Set([focusTokenId])) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [embeddings, focusTokenId, ids, norms]
  );

  // ── 2D map ──
  const map = useMemo(() => {
    const { mean, components } = pca2(ids.map(id => embeddings[id]));
    const project = (v: Vec) => components.map(({ dir }) => dot(add(v, mean, -1), dir));
    const pts = ids.map(id => ({ id, xy: project(embeddings[id]) }));
    // Scale to the middle 96% of points so a few outliers don't squash everything else;
    // outliers are pinned to the edge.
    const range = (vals: number[]) => {
      const s = vals.slice().sort((a, b) => a - b);
      return [s[Math.floor(s.length * 0.02)], s[Math.ceil(s.length * 0.98) - 1]];
    };
    const [minX, maxX] = range(pts.map(p => p.xy[0]));
    const [minY, maxY] = range(pts.map(p => p.xy[1]));
    const clamp01 = (t: number) => Math.min(1, Math.max(0, t));
    const W = 900, H = 380, pad = 28;
    const sx = (x: number) => pad + clamp01((x - minX) / (maxX - minX || 1)) * (W - 2 * pad);
    const sy = (y: number) => H - pad - clamp01((y - minY) / (maxY - minY || 1)) * (H - 2 * pad);
    return { W, H, pts, sx, sy, project, shares: components.map(c => c.share) };
  }, [embeddings, ids]);

  const tokStr = (id: number) => idToToken.get(id) ?? '?';
  const resultXY = mathResult ? map.project(mathResult.v) : null;
  const topId = mathResult?.top[0]?.id;
  const highlight = new Set<number>([focusTokenId, ...(mathResult ? [...mathResult.inputIds] : []), ...(topId !== undefined ? [topId] : [])]);

  const SimBars: React.FC<{ rows: { id: number; sim: number }[] }> = ({ rows }) => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      {rows.map(({ id, sim }) => (
        <div key={id} style={{ display: 'grid', gridTemplateColumns: '80px 1fr 48px', gap: 8, alignItems: 'center' }}>
          <TokenChip text={tokStr(id)} />
          <div style={{ height: 10, background: 'var(--surface-inset)', borderRadius: 3 }}>
            <div style={{ width: `${Math.max(0, sim) * 100}%`, height: '100%', background: 'var(--accent-purple)', borderRadius: 3 }} />
          </div>
          <span className="font-mono" style={{ fontSize: '0.72rem', textAlign: 'right' }}>{sim.toFixed(2)}</span>
        </div>
      ))}
    </div>
  );

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 20 }}>
        {/* Token math */}
        <div>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Token math</div>
          <input type="text" value={expr} onChange={e => setExpr(e.target.value)} className="font-mono" style={{ width: '100%', fontSize: '0.9rem' }} />
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', margin: '8px 0' }}>
            {EXAMPLES.map(ex => (
              <button key={ex} className="btn-secondary" onClick={() => setExpr(ex)} style={{ padding: '2px 8px', fontSize: '0.72rem' }}>
                {ex}
              </button>
            ))}
          </div>
          {'error' in parsed && parsed.error ? (
            <p style={{ fontSize: '0.8rem', color: 'var(--accent-rose)' }}>{parsed.error}</p>
          ) : (
            'terms' in parsed && parsed.terms && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', marginBottom: 10, fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                {parsed.terms.map((t, i) => (
                  <React.Fragment key={i}>
                    <span className="font-mono" style={{ color: 'var(--text-muted)' }}>{i === 0 ? (t.sign < 0 ? '−' : '') : t.sign > 0 ? '+' : '−'}</span>
                    {t.tokenStrs.length === 1 ? (
                      <TokenChip text={t.tokenStrs[0]} />
                    ) : (
                      <span title="Not a single token, so its tokens' vectors are averaged">
                        avg(
                        {t.tokenStrs.map((s, j) => <TokenChip key={j} text={s} />)}
                        )
                      </span>
                    )}
                  </React.Fragment>
                ))}
              </div>
            )
          )}
          {mathResult && <SimBars rows={mathResult.top} />}
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 8 }}>
            <input type="checkbox" checked={excludeInputs} onChange={e => setExcludeInputs(e.target.checked)} />
            Hide the input tokens from the results (the usual convention)
          </label>
        </div>

        {/* Neighbors of the followed token */}
        <div>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>
            Closest tokens to <TokenChip text={tokStr(focusTokenId)} active />
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: 10 }}>
            Cosine similarity: 1 means the vectors point the same way, 0 means unrelated.
          </p>
          <SimBars rows={focusNeighbors} />
        </div>
      </div>

      {/* 2D map */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 10, flexWrap: 'wrap', marginBottom: 6 }}>
          <span style={{ fontWeight: 700 }}>Every token, flattened to 2D</span>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>
            PCA: these two directions keep {Math.round((map.shares[0] + map.shares[1]) * 100)}% of the spread in the {embeddings[0].length}-D vectors
          </span>
        </div>
        <svg viewBox={`0 0 ${map.W} ${map.H}`} style={{ width: '100%', background: 'var(--bg)', border: '1px solid var(--border-color)', borderRadius: 10 }}>
          {map.pts.map(({ id, xy }) => {
            const t = tokStr(id);
            const hl = highlight.has(id) || hovered === id;
            return (
              <text
                key={id}
                x={map.sx(xy[0])}
                y={map.sy(xy[1])}
                textAnchor="middle"
                dominantBaseline="middle"
                fontFamily="JetBrains Mono, monospace"
                fontSize={hl ? 13 : 9}
                fontWeight={hl ? 800 : 500}
                fill={id === focusTokenId ? THEME.amber : id === topId ? THEME.textMain : category(t).color}
                opacity={hl ? 1 : 0.75}
                style={{ cursor: 'default' }}
                onMouseEnter={() => setHovered(id)}
                onMouseLeave={() => setHovered(null)}
              >
                {showTok(t)}
                <title>{`"${t}" (id ${id})`}</title>
              </text>
            );
          })}
          {resultXY && (
            <g>
              <circle cx={map.sx(resultXY[0])} cy={map.sy(resultXY[1])} r={7} fill="none" stroke={THEME.rose} strokeWidth={2} />
              <text
                x={map.sx(resultXY[0]) + (map.sx(resultXY[0]) > map.W * 0.8 ? -10 : 10)}
                y={map.sy(resultXY[1]) - 8}
                textAnchor={map.sx(resultXY[0]) > map.W * 0.8 ? 'end' : 'start'}
                fontSize={10}
                fill={THEME.rose}
                fontFamily="Inter, sans-serif"
              >
                math result
              </text>
            </g>
          )}
        </svg>
        <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginTop: 6, fontSize: '0.72rem', color: 'var(--text-muted)' }}>
          {LEGEND.map(l => (
            <span key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <span style={{ width: 8, height: 8, borderRadius: 4, background: l.color }} /> {l.label}
            </span>
          ))}
          <span style={{ color: 'var(--accent-amber)' }}>■ followed token</span>
          <span style={{ color: 'var(--accent-rose)' }}>○ math result</span>
        </div>
      </div>
    </div>
  );
};
