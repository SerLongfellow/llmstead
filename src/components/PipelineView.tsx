import React, { useEffect, useMemo, useRef, useState } from 'react';
import { BPETokenizerState, StepInspectionData, TransformerConfig } from '../types';
import { BPETokenizer } from '../engine/bpeTokenizer';
import { Sparkles, ChevronRight, Workflow, Repeat } from 'lucide-react';
import { AttentionGrid } from './AttentionGrid';
import { MergeHistory } from './MergeHistory';
import { showTok, cosine, TokenChip } from './tokenUi';
import { EmbeddingSpace } from './EmbeddingSpace';
import { MicroTransformer } from '../engine/transformer';

interface PipelineViewProps {
  inspectionData: StepInspectionData;
  config: TransformerConfig;
  tokenizer: BPETokenizer;
  testInput: string;
  setTestInput: (str: string) => void;
  onRunInspect: () => void;
  // How the tokenizer was built (for the Tokens stage)
  tokenizerState: BPETokenizerState;
  model: MicroTransformer; // read only: for the embedding table
}

type Matrix = number[][];

const argmax = (row: number[]) => row.reduce((best, v, i) => (v > row[best] ? i : best), 0);

// ── Small visual building blocks ──────────────────────────────────────────────

/**
 * A matrix drawn one pixel per value and scaled up. Cyan = positive, rose = negative,
 * brightness = magnitude (relative to the largest value in the matrix). An optional
 * highlighted row marks the followed token.
 */
const MatrixHeatmap: React.FC<{ matrix: Matrix; highlightRow?: number; height?: number }> = ({
  matrix,
  highlightRow,
  height = 72,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rows === 0 || cols === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    let maxAbs = 1e-9;
    for (const row of matrix) for (const v of row) maxAbs = Math.max(maxAbs, Math.abs(v));

    const img = ctx.createImageData(cols, rows);
    const bg = [11, 18, 32];
    const pos = [6, 182, 212];
    const neg = [244, 63, 94];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const t = matrix[r][c] / maxAbs;
        const col = t >= 0 ? pos : neg;
        const a = Math.min(1, Math.abs(t));
        const idx = (r * cols + c) * 4;
        img.data[idx] = bg[0] + (col[0] - bg[0]) * a;
        img.data[idx + 1] = bg[1] + (col[1] - bg[1]) * a;
        img.data[idx + 2] = bg[2] + (col[2] - bg[2]) * a;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [matrix, rows, cols]);

  return (
    <div style={{ position: 'relative', width: '100%', height, borderRadius: 4, overflow: 'hidden', border: '1px solid var(--border-color)' }}>
      <canvas
        ref={canvasRef}
        width={Math.max(cols, 1)}
        height={Math.max(rows, 1)}
        style={{ width: '100%', height: '100%', imageRendering: 'pixelated', display: 'block' }}
      />
      {highlightRow !== undefined && rows > 0 && (
        <div
          style={{
            position: 'absolute',
            left: 0,
            right: 0,
            top: `${(highlightRow / rows) * 100}%`,
            height: `${100 / rows}%`,
            minHeight: 2,
            outline: '2px solid var(--accent-amber)',
            pointerEvents: 'none',
          }}
        />
      )}
    </div>
  );
};

/** One vector as a single strip of colored cells */
const VectorStrip: React.FC<{ label: React.ReactNode; vector: number[]; note?: string }> = ({ label, vector, note }) => (
  <div style={{ display: 'grid', gridTemplateColumns: '220px 1fr', gap: 12, alignItems: 'center' }}>
    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
      {label}
      {note && <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{note}</div>}
    </div>
    <MatrixHeatmap matrix={[vector]} height={18} />
  </div>
);


const SectionTitle: React.FC<{ step: string; title: string; formula?: string }> = ({ step, title, formula }) => (
  <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap', marginBottom: 8 }}>
    <span className="font-mono" style={{ fontSize: '0.75rem', color: 'var(--accent-purple)' }}>{step}</span>
    <span style={{ fontWeight: 700 }}>{title}</span>
    {formula && <code className="font-mono" style={{ fontSize: '0.8rem', color: 'var(--primary)' }}>{formula}</code>}
  </div>
);

const Explain: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6, margin: '4px 0 12px' }}>{children}</p>
);

const SubHeading: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <h4 style={{ fontSize: '0.95rem', fontWeight: 700, margin: '28px 0 10px', paddingTop: 20, borderTop: '1px solid var(--border-color)' }}>
    {children}
  </h4>
);

const vec4 = (v: number[] | undefined) => `[${(v ?? []).slice(0, 4).map(x => x.toFixed(3)).join(', ')}…]`;

const th: React.CSSProperties = { padding: '8px 10px', fontWeight: 600 };
const td: React.CSSProperties = { padding: '8px 10px' };

/** Table row that highlights the followed token and selects on click */
const FocusRow: React.FC<{ active: boolean; onClick: () => void; children: React.ReactNode }> = ({ active, onClick, children }) => (
  <tr
    onClick={onClick}
    style={{
      cursor: 'pointer',
      borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
      background: active ? 'rgba(245, 158, 11, 0.12)' : undefined,
      outline: active ? '1px solid var(--accent-amber)' : undefined,
    }}
  >
    {children}
  </tr>
);

const DataTable: React.FC<{ headers: string[]; children: React.ReactNode }> = ({ headers, children }) => (
  <div style={{ overflowX: 'auto' }}>
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
      <thead>
        <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
          {headers.map(h => <th key={h} style={th}>{h}</th>)}
        </tr>
      </thead>
      <tbody>{children}</tbody>
    </table>
  </div>
);

// ── Main view ─────────────────────────────────────────────────────────────────

export const PipelineView: React.FC<PipelineViewProps> = ({
  inspectionData: data,
  config,
  tokenizer,
  testInput,
  setTestInput,
  onRunInspect,
  tokenizerState,
  model,
}) => {
  const seqLen = data.tokens.length;
  const [focus, setFocus] = useState<number>(Math.max(0, seqLen - 1));
  const [stage, setStage] = useState<string>('output');
  const [headSel, setHeadSel] = useState<number>(0);
  // Copy of the embedding table, refreshed on every run. Training updates the live table in
  // place, so a snapshot gives the embedding-space views a new reference to recompute from.
  const embeddingSnapshot = useMemo(
    () => model.getParameters().wTokenEmbed.map(row => row.slice()),
    [model, data]
  );
  const head = Math.min(headSel, config.numHeads - 1);
  // Weights may have changed (e.g. training) since the last pass, so re-run it on this
  // tab's input whenever the tab opens
  // the pass on this tab's own input (and the latest weights) whenever the tab opens
  useEffect(() => {
    onRunInspect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep the followed token valid when the input changes; default to the last token,
  // since that's the position whose prediction becomes the model's answer
  useEffect(() => {
    setFocus(Math.max(0, seqLen - 1));
  }, [data.inputString, seqLen]);

  if (seqLen === 0) {
    return (
      <div className="glass-panel" style={{ padding: 24 }}>
        Type some text to run it through the model.
      </div>
    );
  }

  const f = Math.min(focus, seqLen - 1);
  const focusTok = data.tokenStrings[f];
  const probs = data.probabilities[f];
  const predId = argmax(probs);
  const predTok = tokenizer.decode([predId]);
  const truncated = seqLen >= config.contextWindow && tokenizer.encode(testInput).tokens.length > seqLen;

  const stages: { id: string; title: string; shape: string; caption: string; visual: React.ReactNode }[] = [
    {
      id: 'text',
      title: 'Text',
      shape: `${data.inputString.length} chars`,
      caption: 'Raw input string',
      visual: (
        <div className="font-mono" style={{ fontSize: '0.75rem', color: 'var(--text-main)', height: 72, overflow: 'hidden', wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}>
          {data.inputString}
        </div>
      ),
    },
    {
      id: 'tokens',
      title: 'Tokens',
      shape: `[${seqLen}]`,
      caption: 'Split into vocab IDs',
      visual: (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, height: 72, overflow: 'hidden', alignContent: 'flex-start' }}>
          {data.tokenStrings.map((t, i) => (
            <TokenChip key={i} text={t} active={i === f} />
          ))}
        </div>
      ),
    },
    {
      id: 'embed',
      title: 'Embeddings',
      shape: `[${seqLen} × ${config.dModel}]`,
      caption: 'Each token → a vector',
      visual: <MatrixHeatmap matrix={data.combinedEmbeddings} highlightRow={f} />,
    },
    ...data.layerInspections.map((layer, l) => ({
      id: `block-${l}`,
      title: `Block ${l + 1}`,
      shape: `[${seqLen} × ${config.dModel}]`,
      caption: 'Attention → MLP',
      visual: <MatrixHeatmap matrix={layer.norm2Output} highlightRow={f} />,
    })),
    {
      id: 'probs',
      title: 'Probabilities',
      shape: `[${seqLen} × ${config.vocabSize}]`,
      caption: 'Score every vocab token',
      visual: <MatrixHeatmap matrix={data.probabilities} highlightRow={f} />,
    },
    {
      id: 'output',
      title: 'Next token',
      shape: `${(probs[predId] * 100).toFixed(0)}% sure`,
      caption: 'Most likely pick',
      visual: (
        <div style={{ height: 72, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <TokenChip text={predTok} bg="rgba(99, 102, 241, 0.3)" />
        </div>
      ),
    },
  ];

  const renderDetail = () => {
    if (stage === 'text') {
      return (
        <>
          <SectionTitle step="Stage 1" title="Text in" />
          <div className="font-mono" style={{ padding: 12, background: 'rgba(15, 23, 42, 0.6)', borderRadius: 8, whiteSpace: 'pre-wrap' }}>
            {data.inputString}
          </div>
          <Explain>
            The model never sees characters directly. The first thing that happens to any prompt is that it gets
            split into tokens from a fixed vocabulary.
            {truncated && ` Only the first ${config.contextWindow} tokens fit in the context window, so the rest of your input was cut off.`}
          </Explain>
        </>
      );
    }

    if (stage === 'tokens') {
      return (
        <>
          <SectionTitle step="Stage 2" title="Tokenize" formula="text → [token IDs]" />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {data.tokenStrings.map((t, i) => (
              <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                <TokenChip text={t} active={i === f} onClick={() => setFocus(i)} />
                <span className="font-mono" style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{data.tokens[i]}</span>
              </div>
            ))}
          </div>
          <Explain>
            The BPE tokenizer breaks the text into {seqLen} tokens (common character runs become single tokens). From here on,
            the model only works with the numbers underneath: each token's ID in a vocabulary of {config.vocabSize}.
          </Explain>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 12 }}>
            <div style={{ background: 'rgba(15, 23, 42, 0.4)', padding: 14, borderRadius: 8, border: '1px solid var(--border-color)' }}>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 4 }}>What the model receives: token IDs [1 × {seqLen}]</p>
              <code className="font-mono" style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)' }}>[{data.tokens.join(', ')}]</code>
            </div>
            <div style={{ background: 'rgba(15, 23, 42, 0.4)', padding: 14, borderRadius: 8, border: '1px solid var(--border-color)' }}>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 4 }}>Decoded back to text (tokenization is lossless)</p>
              <p className="font-mono" style={{ fontSize: '0.85rem', color: 'var(--accent-emerald)', whiteSpace: 'pre-wrap' }}>"{tokenizer.decode(data.tokens)}"</p>
            </div>
          </div>

          <SubHeading>Where these tokens come from</SubHeading>
          <Explain>
            The vocabulary was built from the training dataset before the model was ever created. BPE starts from single
            characters and repeatedly merges the most frequent adjacent pair into a new token until the vocabulary reaches its
            target size. Encoding any text means replaying these merges in order.
          </Explain>
          <Explain>
            The vocabulary stays fixed for the model's whole life. The model never sees text, only IDs, so everything it learns
            about token {data.tokens[f]} depends on the tokenizer always meaning the same thing by it. Change the vocabulary and
            the learned weights would point at the wrong tokens, which is why the vocab size lives in Setup, before the model:
            changing it builds a fresh model.
          </Explain>
          <MergeHistory tokenizerState={tokenizerState} />
        </>
      );
    }

    if (stage === 'embed') {
      return (
        <>
          <SectionTitle step="Stage 3" title="Embed" formula="x = E_token[id] + E_pos[position]" />
          <Explain>
            Each token ID picks out one row of a learned table, turning the token into a list of {config.dModel} numbers. A
            second table adds where the token sits in the sequence, since attention on its own doesn't know word order.
            Following <TokenChip text={focusTok} active /> (position {f}):
          </Explain>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <VectorStrip label={<>Token embedding</>} note={`row ${data.tokens[f]} of the token table`} vector={data.tokenEmbeddings[f]} />
            <VectorStrip label={<>+ Position embedding</>} note={`row ${f} of the position table`} vector={data.positionEmbeddings[f]} />
            <VectorStrip label={<>= Input vector</>} note="what enters Block 1" vector={data.combinedEmbeddings[f]} />
          </div>
          <div style={{ marginTop: 16 }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 6 }}>
              Every token at once: one row per token, {config.dModel} columns
            </div>
            <MatrixHeatmap matrix={data.combinedEmbeddings} highlightRow={f} height={Math.min(160, seqLen * 12)} />
          </div>

          <SubHeading>The numbers</SubHeading>
          <DataTable headers={['Pos', 'Token', 'ID', 'Token embedding', 'Position embedding', 'Input vector (sum)']}>
            {data.tokens.map((id, i) => (
              <FocusRow key={i} active={i === f} onClick={() => setFocus(i)}>
                <td style={td} className="font-mono">{i}</td>
                <td style={td}><TokenChip text={data.tokenStrings[i]} active={i === f} /></td>
                <td style={{ ...td, color: 'var(--text-muted)' }} className="font-mono">{id}</td>
                <td style={{ ...td, color: 'var(--accent-cyan)' }} className="font-mono">{vec4(data.tokenEmbeddings[i])}</td>
                <td style={{ ...td, color: 'var(--accent-amber)' }} className="font-mono">{vec4(data.positionEmbeddings[i])}</td>
                <td style={{ ...td, fontWeight: 600 }} className="font-mono">{vec4(data.combinedEmbeddings[i])}</td>
              </FocusRow>
            ))}
          </DataTable>
          <Explain>First 4 of {config.dModel} values per vector. Click a row to follow that token.</Explain>

          <SubHeading>Embedding space</SubHeading>
          <Explain>
            Every token in the vocabulary has a vector in the same table, so tokens can be compared by the direction their
            vectors point. Training nudges tokens that are used in similar ways toward similar directions. Token math adds and
            subtracts vectors, then shows which tokens land closest (the famous example is king − man + woman ≈ queen in large
            models). With a small vocabulary most tokens are single characters, so try digits on the math dataset, letters, or a
            larger vocab size in Setup. Results are only meaningful after training; with random weights they're noise.
          </Explain>
          <EmbeddingSpace
            embeddings={embeddingSnapshot}
            idToToken={tokenizerState.idToToken}
            tokenizer={tokenizer}
            focusTokenId={data.tokens[f]}
          />
        </>
      );
    }

    if (stage.startsWith('block-')) {
      const l = Number(stage.slice(6));
      const layer = data.layerInspections[l];
      const blockInput = l === 0 ? data.combinedEmbeddings[f] : data.layerInspections[l - 1].norm2Output[f];
      const similarity = cosine(blockInput, layer.norm2Output[f]);
      return (
        <>
          <SectionTitle step={`Stage ${4 + l}`} title={`Transformer block ${l + 1}`} />
          <Explain>
            A block has two halves. <b>Attention</b> is the only place tokens exchange information: each token pulls in
            information from earlier tokens. The <b>MLP</b> then processes each token on its own. Following{' '}
            <TokenChip text={focusTok} active /> through this block:
          </Explain>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            <div>
              <SectionTitle step="a" title="Attention: which earlier tokens does it look at?" formula="softmax(Q·Kᵀ / √d) · V" />
              {layer.attentionWeights.map((head, h) => {
                const row = head[f].slice(0, f + 1);
                const maxW = Math.max(...row, 1e-9);
                return (
                  <div key={h} style={{ display: 'flex', alignItems: 'flex-start', gap: 6, marginBottom: 8, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', width: 52, paddingTop: 3 }}>Head {h + 1}</span>
                    {row.map((w, j) => (
                      <div key={j} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                        <TokenChip
                          text={data.tokenStrings[j]}
                          active={j === f}
                          bg={`rgba(168, 85, 247, ${0.06 + 0.8 * (w / maxW)})`}
                          onClick={() => setFocus(j)}
                        />
                        <span className="font-mono" style={{ fontSize: '0.6rem', color: 'var(--text-dim)' }}>{Math.round(w * 100)}%</span>
                      </div>
                    ))}
                  </div>
                );
              })}
              <Explain>
                Each head spreads 100% of its attention over the tokens up to and including this one; brightness is relative to
                that head's strongest pick. Later tokens are masked out, so the model can't peek at the future. An untrained
                model spreads attention almost evenly; after training, clear favorites show up.
              </Explain>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6, margin: '18px 0 6px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginRight: 4 }}>Inspect head:</span>
                {layer.attentionWeights.map((_, h) => (
                  <button
                    key={h}
                    onClick={() => setHeadSel(h)}
                    style={{
                      padding: '4px 12px',
                      borderRadius: 6,
                      border: '1px solid var(--border-color)',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                      background: head === h ? 'var(--primary)' : 'rgba(15, 23, 42, 0.6)',
                      color: head === h ? '#ffffff' : 'var(--text-muted)',
                    }}
                  >
                    Head {h + 1}
                  </button>
                ))}
              </div>

              <div style={{ height: 16 }} />
              <SectionTitle step="a.1" title="Where the attention weights come from" />
              <Explain>
                The token turns its vector into a <b>query</b> (what it's looking for), and every token turns its vector into a{' '}
                <b>key</b> (what it offers). The dot product of the query with each key is a raw score; softmax turns the scores into
                weights. Then the head passes along a weighted mix of the tokens' <b>value</b> vectors.
              </Explain>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
                <VectorStrip label={<>Query of <TokenChip text={focusTok} active /></>} note={`Q = x · W_Q, head ${head + 1}`} vector={layer.queries[head][f]} />
              </div>
              <DataTable headers={['Key token', 'Key vector', 'Raw score (q·k / √d)', 'Weight (softmax)']}>
                {data.tokenStrings.slice(0, f + 1).map((t, j) => (
                  <FocusRow key={j} active={j === f} onClick={() => setFocus(j)}>
                    <td style={td}><TokenChip text={t} active={j === f} /></td>
                    <td style={{ ...td, minWidth: 160 }}><MatrixHeatmap matrix={[layer.keys[head][j]]} height={14} /></td>
                    <td style={td} className="font-mono">{layer.rawAttentionScores[head][f][j].toFixed(3)}</td>
                    <td style={{ ...td, fontWeight: 600 }} className="font-mono">{(layer.attentionWeights[head][f][j] * 100).toFixed(1)}%</td>
                  </FocusRow>
                ))}
              </DataTable>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 12 }}>
                <VectorStrip label="Head output" note="Σ weight · value, one slice of the attention output" vector={layer.headOutputs[head][f]} />
              </div>

              <div style={{ height: 16 }} />
              <SectionTitle step="a.2" title={`Every token at once (head ${head + 1})`} />
              <Explain>
                The same weights for every token: each row is one token's attention over the tokens before it. Your token's row is
                outlined. Click a row to follow that token.
              </Explain>
              <AttentionGrid
                attentionMap={layer.attentionWeights[head]}
                tokenStrings={data.tokenStrings}
                highlightRow={f}
                onSelectRow={setFocus}
              />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <SectionTitle step="b" title="Add & normalize" formula="LN(x + attention)" />
              <VectorStrip label="After attention" note="added to the input, then normalized" vector={layer.norm1Output[f]} />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <SectionTitle step="c" title="MLP" formula="GELU(x · W₁) · W₂" />
              <VectorStrip
                label="Hidden layer"
                note={`${layer.mlpHidden[f].length} neurons (${config.mlpRatio}× wider)`}
                vector={layer.mlpHidden[f]}
              />
              <VectorStrip label="MLP output" note={`back down to ${config.dModel}`} vector={layer.mlpOutput[f]} />
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <SectionTitle step="d" title="Add & normalize → block output" formula="LN(x + MLP)" />
              <VectorStrip label="Block output" note={l + 1 < config.numLayers ? `enters Block ${l + 2}` : 'goes to the output head'} vector={layer.norm2Output[f]} />
              <Explain>
                Similarity between this token's vector going in and coming out: <b className="font-mono">{similarity.toFixed(2)}</b>{' '}
                (1.00 = unchanged direction). A low number means this block rewrote a lot about the token.
              </Explain>
            </div>
          </div>
        </>
      );
    }

    if (stage === 'probs') {
      const ranked = probs.map((p, id) => ({ id, p })).sort((a, b) => b.p - a.p);
      const top = ranked.slice(0, 8);
      const actualNext = f + 1 < seqLen ? data.tokens[f + 1] : null;
      const actualRank = actualNext === null ? -1 : ranked.findIndex(r => r.id === actualNext);
      return (
        <>
          <SectionTitle step={`Stage ${4 + config.numLayers}`} title="Score every token in the vocabulary" formula="softmax(LN(x) · W_head)" />
          <Explain>
            The final vector for <TokenChip text={focusTok} active /> is compared against every one of the {config.vocabSize} vocab
            tokens, giving one score each. Softmax turns the scores into probabilities that add up to 100%. Top guesses for
            what comes next:
          </Explain>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5, maxWidth: 560 }}>
            {top.map(({ id, p }) => (
              <div key={id} style={{ display: 'grid', gridTemplateColumns: '90px 1fr 56px', gap: 10, alignItems: 'center' }}>
                <TokenChip text={tokenizer.decode([id])} bg={id === actualNext ? 'rgba(16, 185, 129, 0.3)' : undefined} />
                <div style={{ height: 12, background: 'rgba(15, 23, 42, 0.6)', borderRadius: 3 }}>
                  <div style={{ width: `${p * 100}%`, height: '100%', background: 'var(--primary)', borderRadius: 3 }} />
                </div>
                <span className="font-mono" style={{ fontSize: '0.75rem', textAlign: 'right' }}>{(p * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
          {actualNext !== null && (
            <Explain>
              In your input, the token that actually comes next is <TokenChip text={data.tokenStrings[f + 1]} bg="rgba(16, 185, 129, 0.3)" />.
              The model ranks it <b>#{actualRank + 1}</b> at {(probs[actualNext] * 100).toFixed(1)}%. Training pushes this
              number up: the loss is −log of it.
            </Explain>
          )}

          <SubHeading>Every position predicts</SubHeading>
          <Explain>
            All {seqLen} positions go through the same steps in parallel, so one pass yields {seqLen} next-token predictions.
            During training each one is scored against the token that actually follows. Click a row to follow that token.
          </Explain>
          <DataTable headers={['Pos', 'After token', 'Top prediction', 'Confidence', 'Actually next', 'P(actual)', 'Logits (first 4)']}>
            {data.tokens.map((_, i) => {
              const p = data.probabilities[i];
              const top = argmax(p);
              const next = i + 1 < seqLen ? data.tokens[i + 1] : null;
              const hit = next !== null && next === top;
              return (
                <FocusRow key={i} active={i === f} onClick={() => setFocus(i)}>
                  <td style={td} className="font-mono">{i}</td>
                  <td style={td}><TokenChip text={data.tokenStrings[i]} active={i === f} /></td>
                  <td style={td}><TokenChip text={tokenizer.decode([top])} bg={hit ? 'rgba(16, 185, 129, 0.3)' : 'rgba(99, 102, 241, 0.25)'} /></td>
                  <td style={{ ...td, fontWeight: 600 }} className="font-mono">{(p[top] * 100).toFixed(1)}%</td>
                  <td style={td}>{next !== null ? <TokenChip text={data.tokenStrings[i + 1]} /> : <span style={{ color: 'var(--text-dim)' }}>(end of input)</span>}</td>
                  <td style={td} className="font-mono">{next !== null ? `${(p[next] * 100).toFixed(1)}%` : '—'}</td>
                  <td style={{ ...td, color: 'var(--text-muted)' }} className="font-mono">{vec4(data.logits[i]).replace(/(\.\d\d)\d/g, '$1')}</td>
                </FocusRow>
              );
            })}
          </DataTable>
        </>
      );
    }

    // output
    const isLast = f === seqLen - 1;
    return (
      <>
        <SectionTitle step={`Stage ${5 + config.numLayers}`} title="Pick the next token" />
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', padding: 12, background: 'rgba(15, 23, 42, 0.6)', borderRadius: 8 }}>
          {data.tokenStrings.slice(0, f + 1).map((t, i) => (
            <TokenChip key={i} text={t} active={i === f} />
          ))}
          <ChevronRight size={16} color="var(--text-dim)" />
          <TokenChip text={predTok} bg="rgba(99, 102, 241, 0.35)" />
          <span className="font-mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{(probs[predId] * 100).toFixed(1)}%</span>
        </div>
        <Explain>
          Greedy decoding just takes the most likely token. Sampling with a temperature picks randomly in proportion to the
          probabilities instead, which is what the text generator in the Training Dashboard does.
          {!isLast && ' Every position makes a prediction (that is how training gets many examples from one sequence), but when generating text only the last position\'s prediction is used.'}
        </Explain>
        <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', padding: 12, borderRadius: 8, border: '1px dashed var(--border-color)' }}>
          <Repeat size={18} color="var(--accent-cyan)" style={{ flexShrink: 0, marginTop: 2 }} />
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
            <b style={{ color: 'var(--text-main)' }}>That's one token.</b> To write a longer reply, an LLM appends the chosen token to
            the input and runs this entire pipeline again, once per token.
            <div style={{ marginTop: 8 }}>
              <button
                className="btn-secondary"
                onClick={() => setTestInput(data.inputString + predTok)}
                disabled={seqLen >= config.contextWindow}
                title={seqLen >= config.contextWindow ? 'The context window is full' : undefined}
              >
                <Repeat size={14} /> Append “{showTok(predTok)}” and run again
              </button>
            </div>
          </div>
        </div>
      </>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Input */}
      <div className="glass-panel" style={{ padding: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
          <Workflow size={20} color="var(--accent-cyan)" />
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>One Forward Pass, Start to Finish</h2>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <input type="text" value={testInput} onChange={e => setTestInput(e.target.value)} style={{ flex: 1, fontSize: '0.95rem' }} />
          <button className="btn-primary" onClick={onRunInspect}>
            <Sparkles size={16} /> Run
          </button>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', marginTop: 12 }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginRight: 4 }}>Follow a token:</span>
          {data.tokenStrings.map((t, i) => (
            <TokenChip key={i} text={t} active={i === f} onClick={() => setFocus(i)} />
          ))}
        </div>
      </div>

      {/* Pipeline strip */}
      <div className="glass-panel" style={{ padding: 20, overflowX: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'stretch', gap: 4, minWidth: 'min-content' }}>
          {stages.map((s, i) => {
            const active = stage === s.id;
            return (
              <React.Fragment key={s.id}>
                {i > 0 && (
                  <div style={{ display: 'flex', alignItems: 'center', color: 'var(--text-dim)' }}>
                    <ChevronRight size={18} />
                  </div>
                )}
                <button
                  onClick={() => setStage(s.id)}
                  style={{
                    flex: '1 0 130px',
                    textAlign: 'left',
                    padding: 10,
                    borderRadius: 10,
                    cursor: 'pointer',
                    background: active ? 'rgba(99, 102, 241, 0.15)' : 'rgba(15, 23, 42, 0.6)',
                    border: `1px solid ${active ? 'var(--primary)' : 'var(--border-color)'}`,
                    color: 'var(--text-main)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 6,
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '0.85rem' }}>{s.title}</div>
                    <div className="font-mono" style={{ fontSize: '0.65rem', color: 'var(--text-dim)' }}>{s.shape}</div>
                  </div>
                  {s.visual}
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{s.caption}</span>
                </button>
              </React.Fragment>
            );
          })}
        </div>
        <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: 10 }}>
          Click a stage for details. Heatmaps: one row per token, <span style={{ color: 'var(--accent-cyan)' }}>cyan</span> positive,{' '}
          <span style={{ color: 'var(--accent-rose)' }}>rose</span> negative, <span style={{ color: 'var(--accent-amber)' }}>outlined row</span> = the
          token you're following.
        </div>
      </div>

      {/* Stage detail */}
      <div className="glass-panel" style={{ padding: 24 }}>{renderDetail()}</div>
    </div>
  );
};
