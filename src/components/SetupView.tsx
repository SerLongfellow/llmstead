import React, { useEffect, useMemo, useRef, useState } from 'react';
import { DatasetOption, TransformerConfig } from '../types';
import { BPETokenizer } from '../engine/bpeTokenizer';
import { VALIDATION_FRACTION } from '../engine/datasets';
import { MODEL_PROFILES, ProfileShape, matchingProfile } from '../engine/modelProfiles';
import { FromSpeedWorker, ToSpeedWorker } from '../engine/speedWorker';
import { BookOpen, Binary, Sliders, Info, Plus, ChevronRight, Ruler, RotateCcw, Download, Gauge, LayoutGrid } from 'lucide-react';
import { InfoTooltip } from './InfoTooltip';
import { GoDeeper } from './GoDeeper';
import { ScaleComparison } from './ScaleComparison';
import { referenceHint, FRONTIER_NOTE } from '../engine/referenceModels';

interface SetupViewProps {
  // 1. Data
  datasets: DatasetOption[];
  selectedDataset: DatasetOption;
  onSelectDataset: (ds: DatasetOption) => void;
  onAddDataset: (ds: DatasetOption) => void;
  /** Set when a dataset download failed */
  datasetError: string | null;
  // 2. Tokenizer
  tokenizer: BPETokenizer;
  targetVocabSize: number;
  onChangeVocabSize: (vocabSize: number) => void;
  // 3. Architecture
  config: TransformerConfig; // vocabSize here is the tokenizer's actual vocab
  onChangeConfig: (newConfig: TransformerConfig) => void;
  /** Sets the vocabulary and architecture together (one rebuild) */
  onApplyProfile: (shape: ProfileShape) => void;
  /** The last settings that matched no profile (what the Custom card restores), if any */
  lastCustomShape: ProfileShape | null;
  paramCount: number;
}

/** The text preview shows at most this much (a big dataset would make the page sluggish) */
const PREVIEW_CHARS = 20_000;
/** Chinchilla's rule of thumb: compute-optimal training sees ~20 tokens per parameter */
const TOKENS_PER_PARAM = 20;
/** Below this many parameters, a dataset's suggested (bigger) profile is pointed out */
const SMALL_FOR_STORIES = 300_000;
/** Weights, gradients and AdamW's two moment estimates, as 8-byte numbers */
const BYTES_PER_PARAM = 4 * 8;

/** "40 s", "12 min", "3.5 h", "2 days" */
function formatDuration(seconds: number): string {
  if (seconds < 90) return `${Math.max(1, Math.round(seconds))} s`;
  if (seconds < 90 * 60) return `${Math.round(seconds / 60)} min`;
  if (seconds < 48 * 3600) return `${(seconds / 3600).toFixed(seconds < 10 * 3600 ? 1 : 0)} h`;
  return `${Math.round(seconds / 86400)} days`;
}

/**
 * Times one training step for this model shape in a background thread (engine/speedWorker.ts).
 * Returns ms per step, or null while measuring (or if this browser can't start the worker).
 */
function useMeasuredStepMs(config: TransformerConfig): number | null {
  const worker = useRef<Worker | null>(null);
  const requestId = useRef(0);
  const [result, setResult] = useState<{ id: number; ms: number } | null>(null);
  useEffect(() => {
    try {
      worker.current = new Worker(new URL('../engine/speedWorker.ts', import.meta.url), { type: 'module' });
      worker.current.onmessage = (e: MessageEvent<FromSpeedWorker>) => setResult({ id: e.data.id, ms: e.data.msPerStep });
    } catch {
      worker.current = null;
    }
    return () => worker.current?.terminate();
  }, []);
  const { vocabSize, contextWindow, dModel, numHeads, numLayers, mlpRatio, learningRate, optimizer } = config;
  useEffect(() => {
    const id = ++requestId.current;
    // Wait for the sliders to settle before measuring
    const timer = window.setTimeout(() => {
      const message: ToSpeedWorker = { id, config: { vocabSize, contextWindow, dModel, numHeads, numLayers, mlpRatio, learningRate, optimizer } };
      worker.current?.postMessage(message);
    }, 300);
    return () => window.clearTimeout(timer);
  }, [vocabSize, contextWindow, dModel, numHeads, numLayers, mlpRatio, learningRate, optimizer]);
  return result && result.id === requestId.current ? result.ms : null;
}

// ── Layout helpers ────────────────────────────────────────────────────────────

const Section: React.FC<{
  step: number;
  icon: React.ReactNode;
  title: string;
  id?: string;
  children: React.ReactNode;
}> = ({ step, icon, title, id, children }) => (
  <div id={id} className="glass-panel" style={{ padding: '24px', scrollMarginTop: 16 }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 18 }}>
      <span
        className="font-mono"
        style={{ width: 24, height: 24, borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', fontWeight: 700, background: 'var(--primary)', color: '#ffffff' }}
      >
        {step}
      </span>
      {icon}
      <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>{title}</h2>
    </div>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>{children}</div>
  </div>
);

const SliderRow: React.FC<{
  label: string;
  tooltip: { title: string; description: string; impact: string };
  valueLabel: React.ReactNode;
  color: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  hint?: string;
  reference?: string; // how real models set this, e.g. "GPT-2 small: 768 · ..."
}> = ({ label, tooltip, valueLabel, color, min, max, step, value, onChange, hint, reference }) => (
  <div>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>{label}</label>
        <InfoTooltip {...tooltip} />
      </div>
      <span className="font-mono" style={{ fontSize: '0.85rem', color }}>{valueLabel}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(Number(e.target.value))} />
    {hint && <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{hint}</span>}
    {reference && <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>For scale: {reference}</div>}
  </div>
);

const Stat: React.FC<{ label: string; value: string; color: string; hint?: string }> = ({ label, value, color, hint }) => (
  <div style={{ background: 'var(--surface-inset)', padding: 12, borderRadius: 8, border: '1px solid var(--border-color)' }}>
    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{label}</p>
    <p className="font-mono" style={{ fontSize: '1.3rem', fontWeight: 700, color }}>{value}</p>
    {hint && <p style={{ fontSize: '0.68rem', color: 'var(--text-dim)' }}>{hint}</p>}
  </div>
);

const BreakdownRow: React.FC<{ label: string; value: number }> = ({ label, value }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: 'var(--surface-inset)', borderRadius: 6, fontSize: '0.85rem' }}>
    <span style={{ color: 'var(--text-muted)' }}>{label}</span>
    <span className="font-mono" style={{ fontWeight: 600 }}>{value.toLocaleString()}</span>
  </div>
);

/** One card in the Model profile picker (a preset, or Custom) */
const ProfileCard: React.FC<{ name: string; tagline: string; shape: ProfileShape | null; current: boolean; onClick: () => void }> = ({
  name,
  tagline,
  shape,
  current,
  onClick,
}) => (
  <button
    onClick={() => !current && onClick()}
    aria-pressed={current}
    style={{
      textAlign: 'left', padding: 12, borderRadius: 10, cursor: current ? 'default' : 'pointer', font: 'inherit', color: 'inherit',
      border: '1px solid ' + (current ? 'var(--primary)' : 'var(--border-color)'),
      background: current ? 'var(--primary-soft)' : 'var(--surface-inset)',
    }}
  >
    <div style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: 2 }}>{name}</div>
    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.4 }}>{tagline}</div>
    {shape && (
      <div className="font-mono" style={{ fontSize: '0.68rem', color: 'var(--text-dim)', marginTop: 6 }}>
        d{shape.dModel} · {shape.numLayers} {shape.numLayers === 1 ? 'block' : 'blocks'} · ctx {shape.contextWindow} · V {shape.vocabSize}
      </div>
    )}
  </button>
);

const FitRow: React.FC<{ label: string; value: string; hint?: string }> = ({ label, value, hint }) => (
  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10, padding: '8px 12px', background: 'var(--surface-inset)', borderRadius: 6 }}>
    <span style={{ color: 'var(--text-muted)' }}>
      {label}
      {hint && <span style={{ display: 'block', fontSize: '0.68rem', color: 'var(--text-dim)' }}>{hint}</span>}
    </span>
    <span className="font-mono" style={{ fontWeight: 600, whiteSpace: 'nowrap' }}>{value}</span>
  </div>
);

/**
 * Collapsible explainer on spending parameters on width (d_model) or depth (layers). The numbers
 * quoted are from the runs described in modelProfiles.ts.
 */
const WidthVsDepth: React.FC<{ dModel: number; numLayers: number }> = ({ dModel, numLayers }) => {
  const para = { fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.6, margin: '0 0 10px' };
  const strong = { color: 'var(--text-main)' };
  return (
    <details>
      <summary style={{ cursor: 'pointer', fontSize: '0.85rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
        <ChevronRight size={14} /> Width or depth? How the shape changes what a model learns
      </summary>
      <div style={{ marginTop: 12, padding: 16, background: 'var(--surface-inset)', border: '1px solid var(--border-color)', borderRadius: 10 }}>
        <p style={para}>
          Think of each token's vector as a scratchpad that flows up through the blocks: every block reads it, works something out,
          and adds its result back.
        </p>
        <p style={para}>
          <b style={strong}>Width (d_model) is the size of the scratchpad.</b> A wider vector holds more facts about a token at
          once (which word it is, where it sits, whether it's inside a quote), and since the MLP is a few times wider still, it also
          sets how many patterns the model can memorize, like "after <i>once upon a</i> comes <i>time</i>".
        </p>
        <p style={para}>
          <b style={strong}>Depth (layers) is the number of rounds.</b> A later block can build on what an earlier one found. The
          classic example needs two blocks: the first marks every token with the token before it, and the second uses that to
          find where the current word appeared earlier and copy what followed. That's how a model repeats a name from earlier in
          the context. With one block, attention can only look things up directly, so a shallow model behaves like a very good
          phrase predictor: fluent locally, weak at keeping track of anything across the context.
        </p>
        <p style={para}>
          <b style={strong}>Which wins depends on the data and on training time.</b> On these small datasets most of what there is to
          learn is local word patterns, which width captures directly. In our runs (4,000 steps), Wide &amp; shallow beat Deep &amp;
          narrow on Shakespeare and Q&amp;A and tied on math, where Deep was still improving when we stopped. Depth pays off when the
          text rewards multi-step patterns and there's enough training to learn them.
        </p>
        <p style={para}>
          <b style={strong}>In practice:</b> blocks run one after another while width splits into parallel work, so deep models
          take longer per token. At a fixed size, the exact shape matters much less than the total number of parameters, as long as
          it isn't extreme. Real models sit around 64–128 of d_model per block (GPT-2 small: 768 over 12 blocks; Llama 3 8B: 4,096
          over 32). Yours: {dModel} over {numLayers} {numLayers === 1 ? 'block' : 'blocks'}, {Math.round(dModel / numLayers)} per
          block. Shallow models are also easier to read in Look inside: each attention pattern feeds the prediction directly.
        </p>
        <GoDeeper
          links={[
            { label: 'A Mathematical Framework for Transformer Circuits (Anthropic, 2021)', url: 'https://transformer-circuits.pub/2021/framework/index.html' },
            { label: 'Induction heads (Anthropic, 2022)', url: 'https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html' },
            { label: 'Scaling Laws for Neural Language Models (Kaplan et al., 2020)', url: 'https://arxiv.org/abs/2001.08361' },
          ]}
        />
      </div>
    </details>
  );
};

// ── Main view ─────────────────────────────────────────────────────────────────

export const SetupView: React.FC<SetupViewProps> = ({
  datasets,
  selectedDataset,
  onSelectDataset,
  onAddDataset,
  datasetError,
  tokenizer,
  targetVocabSize,
  onChangeVocabSize,
  config,
  onChangeConfig,
  onApplyProfile,
  lastCustomShape,
  paramCount,
}) => {
  // Custom dataset form
  const [isCreatingCustom, setIsCreatingCustom] = useState<boolean>(false);
  // The vocab slider shows its value immediately but applies it a moment after you stop
  // dragging: each new value retrains the tokenizer and rebuilds the model.
  const [vocabDraft, setVocabDraft] = useState<number>(targetVocabSize);
  const vocabTimer = useRef<number>();
  useEffect(() => setVocabDraft(targetVocabSize), [targetVocabSize]);
  useEffect(() => () => window.clearTimeout(vocabTimer.current), []);
  const changeVocabDraft = (v: number) => {
    setVocabDraft(v);
    window.clearTimeout(vocabTimer.current);
    vocabTimer.current = window.setTimeout(() => onChangeVocabSize(v), 250);
  };

  const [customTitle, setCustomTitle] = useState<string>('My Custom Dataset');
  const [customText, setCustomText] = useState<string>(
    'Artificial Intelligence and Machine Learning models predict tokens based on probability distributions learned during pre-training.'
  );

  const handleSaveCustomDataset = () => {
    if (!customTitle.trim() || !customText.trim()) return;
    onAddDataset({
      id: 'custom-' + Date.now(),
      name: customTitle.trim(),
      category: 'synthetic',
      description: 'User-provided custom text dataset (' + customText.length + ' chars)',
      text: customText,
    });
    setIsCreatingCustom(false);
  };

  const updateField = (key: keyof TransformerConfig, value: number | string) => {
    onChangeConfig({ ...config, [key]: value });
  };

  const MAX_HEADS = 8;
  // Only head counts that divide d_model evenly — otherwise dimensions would be dropped
  const headOptions = (dModel: number) =>
    Array.from({ length: MAX_HEADS }, (_, i) => i + 1).filter(h => dModel % h === 0);

  const updateDModel = (dModel: number) => {
    let numHeads = config.numHeads;
    if (dModel % numHeads !== 0) {
      numHeads = Math.max(...headOptions(dModel).filter(h => h <= numHeads));
    }
    onChangeConfig({ ...config, dModel, numHeads });
  };

  // Dataset + tokenizer stats
  const text = selectedDataset.text;
  const charCount = text.length;
  const wordCount = text.trim().split(/\s+/).length;
  const tokenCount = useMemo(() => tokenizer.encode(text).tokens.length, [tokenizer, text]);
  const compressionRatio = (charCount / Math.max(tokenCount, 1)).toFixed(2);

  const dMlp = config.dModel * config.mlpRatio;
  const headDim = config.dModel / config.numHeads;

  // Profiles compare against the *target* vocab (what the profile sets), not what BPE reached
  const activeProfile = matchingProfile({ ...config, vocabSize: targetVocabSize });
  const suggestedProfile = MODEL_PROFILES.find(p => p.id === selectedDataset.suggestedProfile);

  // Fit check: how long training takes on this device
  const msPerStep = useMeasuredStepMs(config);
  const trainTokens = Math.round(tokenCount * (1 - VALIDATION_FRACTION));
  const tokensPerSecond = msPerStep ? (1000 / msPerStep) * config.contextWindow : null;
  const onePass = tokensPerSecond ? trainTokens / tokensPerSecond : null;
  const optimalBudget = tokensPerSecond ? (TOKENS_PER_PARAM * paramCount) / tokensPerSecond : null;
  const memoryMb = (paramCount * BYTES_PER_PARAM) / 1e6;

  const tokenEmbedParams = config.vocabSize * config.dModel;
  const posEmbedParams = config.contextWindow * config.dModel;
  const qkvParams = 3 * config.dModel * config.dModel * config.numLayers;
  const outProjParams = config.dModel * config.dModel * config.numLayers;
  const mlpParams = 2 * config.dModel * dMlp * config.numLayers;
  const headParams = config.dModel * config.vocabSize;

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 20, alignItems: 'flex-start' }}>
      {/* Left: the four setup steps, in the order you'd build a model */}
      <div style={{ flex: '999 1 560px', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 20 }}>
        <div className="glass-panel" style={{ padding: '16px 24px', display: 'flex', alignItems: 'center', gap: 10 }}>
          <RotateCcw size={20} color="var(--accent-amber)" />
          <div>
            <p style={{ fontSize: '0.95rem', fontWeight: 700 }}>Changing any setting resets the model</p>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              New data, vocabulary or shape means fresh weights, so training starts over.
            </p>
          </div>
        </div>

        {/* Shortcut: set the vocabulary and architecture (steps 2 and 3) in one go */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
            <LayoutGrid size={20} color="var(--accent-cyan)" />
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Model profile</h2>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: 14 }}>
            Ready-made settings for steps 2 and 3. Pick one, then fine-tune the sliders below if you like: any change makes it
            Custom.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
            {MODEL_PROFILES.map(p => (
              <ProfileCard
                key={p.id}
                name={p.name}
                tagline={p.tagline}
                shape={p.shape}
                current={activeProfile?.id === p.id}
                onClick={() => onApplyProfile(p.shape)}
              />
            ))}
            {/* Selected whenever the settings match no profile; from a profile, it brings the last custom settings back */}
            <ProfileCard
              name="Custom"
              tagline={activeProfile ? (lastCustomShape ? 'Back to your own settings.' : 'Move any slider below.') : 'Your own settings.'}
              shape={activeProfile ? lastCustomShape : { ...config, vocabSize: targetVocabSize }}
              current={!activeProfile}
              onClick={() =>
                lastCustomShape ? onApplyProfile(lastCustomShape) : document.getElementById('setup-tokenizer')?.scrollIntoView({ behavior: 'smooth' })
              }
            />
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 12, lineHeight: 1.5 }}>
            {activeProfile ? (
              <>
                <b style={{ color: 'var(--text-main)' }}>{activeProfile.name}:</b> {activeProfile.description}
              </>
            ) : (
              <>
                <b style={{ color: 'var(--text-main)' }}>Custom:</b> your own mix of settings, which no profile matches. Pick a profile
                to go back to a preset; the Custom card remembers these settings, so you can come back to them.
              </>
            )}
          </p>
        </div>
        {/* 1. Data */}
        <Section step={1} icon={<BookOpen size={20} color="var(--accent-purple)" />} title="Data">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
              The text the model learns from. The tokenizer is also built from it, and the last 15% is held out to measure
              generalization.
            </p>
            <button className="btn-secondary" onClick={() => setIsCreatingCustom(!isCreatingCustom)} style={{ padding: '6px 12px', fontSize: '0.8rem', flexShrink: 0 }}>
              <Plus size={14} /> Add Custom Text
            </button>
          </div>

          {isCreatingCustom && (
            <div style={{ background: 'var(--surface-inset)', padding: 16, borderRadius: 10, border: '1px solid var(--primary)', display: 'flex', flexDirection: 'column', gap: 10 }}>
              <div>
                <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Dataset Title</label>
                <input type="text" value={customTitle} onChange={e => setCustomTitle(e.target.value)} style={{ width: '100%', marginTop: 4, fontSize: '0.85rem' }} />
              </div>
              <div>
                <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Paste Text Content</label>
                <textarea rows={4} value={customText} onChange={e => setCustomText(e.target.value)} style={{ width: '100%', marginTop: 4, fontSize: '0.82rem' }} />
              </div>
              <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                <button className="btn-secondary" onClick={() => setIsCreatingCustom(false)} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>Cancel</button>
                <button className="btn-primary" onClick={handleSaveCustomDataset} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>Save Dataset</button>
              </div>
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 10 }}>
            {datasets.map(ds => {
              const isCurrent = selectedDataset.id === ds.id;
              return (
                <div
                  key={ds.id}
                  onClick={() => onSelectDataset(ds)}
                  style={{
                    padding: 14,
                    borderRadius: 10,
                    border: '1px solid ' + (isCurrent ? 'var(--primary)' : 'var(--border-color)'),
                    background: isCurrent ? 'var(--primary-soft)' : 'var(--surface-inset)',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    boxShadow: 'none',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                    <h3 style={{ fontSize: '0.9rem', fontWeight: 700 }}>{ds.name}</h3>
                    <span className="badge badge-purple" style={{ fontSize: '0.65rem' }}>{ds.category}</span>
                  </div>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{ds.description}</p>
                  {ds.download && (
                    <p style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: 6, display: 'flex', alignItems: 'center', gap: 4 }}>
                      <Download size={12} /> {(ds.download.bytes / 1e6).toFixed(1)} MB{ds.text ? ', downloaded' : ' download'}
                    </p>
                  )}
                </div>
              );
            })}
          </div>

          {datasetError && <p style={{ fontSize: '0.85rem', color: 'var(--accent-rose)' }}>{datasetError}</p>}

          {selectedDataset.source && (
            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              Source:{' '}
              <a href={selectedDataset.source.url} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--text-muted)' }}>
                {selectedDataset.source.label}
              </a>
              , shared under the{' '}
              <a href={selectedDataset.source.licenseUrl} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--text-muted)' }}>
                {selectedDataset.source.license}
              </a>{' '}
              license.
            </p>
          )}

          {suggestedProfile && activeProfile?.id !== suggestedProfile.id && paramCount < SMALL_FOR_STORIES && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', padding: '10px 14px', borderRadius: 10, background: 'var(--surface-inset)', border: '1px solid var(--accent-amber)' }}>
              <span style={{ flex: 1, minWidth: 200, fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                {selectedDataset.name} is meant for the <b style={{ color: 'var(--text-main)' }}>{suggestedProfile.name}</b> profile. This model
                will pick up common words, but it's too small to string sentences together.
              </span>
              <button className="btn-secondary" onClick={() => onApplyProfile(suggestedProfile.shape)} style={{ padding: '4px 10px', fontSize: '0.8rem' }}>
                Use {suggestedProfile.name}
              </button>
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <Stat label="Characters" value={charCount.toLocaleString()} color="var(--accent-cyan)" />
            <Stat label="Words" value={wordCount.toLocaleString()} color="var(--accent-emerald)" />
          </div>

          <details>
            <summary style={{ cursor: 'pointer', fontSize: '0.85rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 6 }}>
              <ChevronRight size={14} /> Preview the text: {selectedDataset.name}
            </summary>
            <pre
              className="font-mono"
              style={{
                margin: '10px 0 0',
                maxHeight: 360,
                overflowY: 'auto',
                padding: 16,
                background: 'var(--bg)',
                border: '1px solid var(--border-color)',
                borderRadius: 10,
                fontSize: '0.82rem',
                color: 'var(--text-main)',
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                lineHeight: 1.5,
              }}
            >
              {text.length > PREVIEW_CHARS ? text.slice(0, PREVIEW_CHARS) + '\n…' : text}
            </pre>
            {text.length > PREVIEW_CHARS && (
              <p style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: 6 }}>
                Showing the first {PREVIEW_CHARS.toLocaleString()} of {charCount.toLocaleString()} characters.
              </p>
            )}
          </details>
        </Section>

        {/* 2. Tokenizer */}
        <Section step={2} id="setup-tokenizer" icon={<Binary size={20} color="var(--accent-rose)" />} title="Tokenizer">
          <SliderRow
            label="Vocabulary Size (V)"
            reference={referenceHint('vocabSize')}
            tooltip={{
              title: 'Vocabulary Size (V)',
              description: 'How many distinct tokens the BPE tokenizer learns from the dataset: every single character, plus merged character runs until it reaches this size.',
              impact: 'Sets the rows of the embedding table and the columns of the output head. Bigger vocabularies mean fewer tokens per sentence but more parameters. The tokenizer is fixed before training, so changing this builds a new tokenizer and a fresh model.',
            }}
            valueLabel={vocabDraft !== targetVocabSize ? <>{vocabDraft}…</> : <>{config.vocabSize}{config.vocabSize < targetVocabSize ? ` / ${targetVocabSize} target` : ''}</>}
            color="var(--accent-rose)"
            min={50}
            max={500}
            step={10}
            value={vocabDraft}
            onChange={changeVocabDraft}
            hint={config.vocabSize < targetVocabSize ? 'BPE ran out of repeated pairs before reaching the target' : 'Single characters plus the merges BPE learned from the dataset'}
          />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <Stat label="Dataset as tokens" value={tokenCount.toLocaleString()} color="var(--accent-amber)" hint={`from ${charCount.toLocaleString()} characters`} />
            <Stat label="Compression" value={`${compressionRatio} chars/token`} color="var(--accent-purple)" hint="a bigger vocabulary packs more text into each token" />
          </div>
        </Section>

        {/* 3. Architecture */}
        <Section step={3} icon={<Sliders size={20} color="var(--primary)" />} title="Architecture">
          <SliderRow
            label="Embedding Dimension (d_model)"
            reference={referenceHint('dModel')}
            tooltip={{
              title: 'Embedding Dimension (d_model)',
              description: 'The length of the dense numerical vector assigned to each token. Higher dimensions capture richer semantic relationships.',
              impact: 'Quadratically increases QKV projection matrix sizes and total parameter count.',
            }}
            valueLabel={config.dModel}
            color="var(--accent-cyan)"
            min={8}
            max={128}
            step={8}
            value={config.dModel}
            onChange={updateDModel}
            hint="Width of vector representation per token"
          />

          <SliderRow
            label="Context Window Length (Seq Len)"
            reference={`${referenceHint('contextWindow')} · frontier models: ${FRONTIER_NOTE.contextRange}`}
            tooltip={{
              title: 'Context Window Length (Seq Len)',
              description: 'The maximum number of sequence tokens the model can process, attend to, and remember at one time.',
              impact: 'Quadratically increases attention matrix calculation memory size (N x N).',
            }}
            valueLabel={config.contextWindow}
            color="var(--accent-emerald)"
            min={8}
            max={64}
            step={8}
            value={config.contextWindow}
            onChange={v => updateField('contextWindow', v)}
            hint="Maximum tokens processed simultaneously"
          />

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Attention Heads (H)</label>
                <InfoTooltip
                  title="Multi-Head Attention (H)"
                  description="Splits d_model into H parallel heads. Head 1 might focus on grammar while Head 2 focuses on subject-verb relationships."
                  impact="Splits head dimension (d_k = d_model / H) without increasing total parameter count!"
                />
              </div>
              <span className="font-mono" style={{ fontSize: '0.85rem', color: 'var(--accent-amber)' }}>
                {config.numHeads} heads ({headDim}d each)
              </span>
            </div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {headOptions(config.dModel).map(h => (
                <button
                  key={h}
                  className={h === config.numHeads ? 'btn-primary' : 'btn-secondary'}
                  style={{ padding: '4px 12px', minWidth: 40 }}
                  onClick={() => updateField('numHeads', h)}
                >
                  {h}
                </button>
              ))}
            </div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
              Only divisors of d_model ({config.dModel}) are allowed, so every head gets an equal slice
            </span>
            <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>For scale: {referenceHint('heads')}</div>
          </div>

          <SliderRow
            label="Transformer Blocks (Layers L)"
            reference={referenceHint('layers')}
            tooltip={{
              title: 'Transformer Stack Depth (L)',
              description: 'The number of stacked attention + MLP blocks. Deeper layers form abstract reasoning and higher-level concepts.',
              impact: 'Linearly multiplies total parameter count and computation time per pass.',
            }}
            valueLabel={config.numLayers}
            color="var(--accent-purple)"
            min={1}
            max={4}
            step={1}
            value={config.numLayers}
            onChange={v => updateField('numLayers', v)}
            hint="Stack depth of attention + MLP blocks"
          />

          <SliderRow
            label="MLP Expansion Ratio"
            tooltip={{
              title: 'MLP Expansion Ratio',
              description: 'The width multiplier for the Feed-Forward Neural Network layer inside each transformer block (standard is 4x).',
              impact: 'Provides non-linear capacity (GELU) for storing factual memories and associations.',
            }}
            valueLabel={`${config.mlpRatio}x (${dMlp} hidden)`}
            color="var(--primary)"
            min={2}
            max={4}
            step={1}
            value={config.mlpRatio}
            onChange={v => updateField('mlpRatio', v)}
          />

          <WidthVsDepth dModel={config.dModel} numLayers={config.numLayers} />
        </Section>

        {/* Not a step: real models for comparison */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
            <Ruler size={20} color="var(--accent-cyan)" />
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>For scale: real LLMs</h2>
          </div>
          <ScaleComparison config={config} paramCount={paramCount} />
        </div>
      </div>

      {/* Right: parameter breakdown, kept in view while scrolling the steps */}
      <div className="glass-panel" style={{ flex: '1 1 300px', padding: 24, position: 'sticky', top: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
          <Info size={20} color="var(--accent-cyan)" />
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Parameter Breakdown</h2>
        </div>

        <div
          style={{
            background: 'var(--surface-inset)',
            border: '1px solid var(--border-color)',
            borderRadius: 12,
            padding: 20,
            textAlign: 'center',
            marginBottom: 20,
          }}
        >
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Total Trainable Weights</p>
          <p className="font-mono" style={{ fontSize: '2.2rem', fontWeight: 800, color: 'var(--text-main)', margin: '4px 0' }}>
            {paramCount.toLocaleString()}
          </p>
          <p style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)' }}>Micro-Transformer Architecture (Nano Scale)</p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <BreakdownRow label="Token Embeddings (V × d_model)" value={tokenEmbedParams} />
          <BreakdownRow label="Positional Embeddings (N × d_model)" value={posEmbedParams} />
          <BreakdownRow label="Q, K, V Projections (3 × d_model² × L)" value={qkvParams} />
          <BreakdownRow label="Output Attention Proj (d_model² × L)" value={outProjParams} />
          <BreakdownRow label="Feed-Forward MLP (2 × d_model × d_mlp × L)" value={mlpParams} />
          <BreakdownRow label="Output Unembedding Head (d_model × V)" value={headParams} />
        </div>

        {/* Fit check: what this model and dataset cost to train here, measured on this device */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '24px 0 12px' }}>
          <Gauge size={18} color="var(--accent-emerald)" />
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Fit check</h3>
          <InfoTooltip
            title="Fit check"
            description="Times a few training steps of this model shape on your device (in the background, on a throwaway copy), then estimates how long training takes. Training runs on one CPU core, so a faster computer helps but a GPU doesn't."
            impact="Estimates assume the tab stays open and the computer stays awake. Background tabs keep training, and progress is autosaved every 10 seconds."
          />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: '0.8rem' }}>
          <FitRow label="Speed on this device" value={msPerStep ? `${(1000 / msPerStep).toFixed(1)} steps/s` : 'measuring…'} hint={tokensPerSecond ? `${Math.round(tokensPerSecond).toLocaleString()} tokens/s` : undefined} />
          <FitRow label="One pass over the training text" value={onePass ? formatDuration(onePass) : '…'} hint={`${trainTokens.toLocaleString()} tokens`} />
          <FitRow
            label="Rule-of-thumb training budget"
            value={optimalBudget ? formatDuration(optimalBudget) : '…'}
            hint={`${TOKENS_PER_PARAM} tokens per parameter (Chinchilla)`}
          />
          <FitRow label="Memory for training" value={`${memoryMb < 10 ? memoryMb.toFixed(1) : Math.round(memoryMb)} MB`} hint="weights, gradients, optimizer state" />
        </div>
        <p style={{ fontSize: '0.72rem', color: 'var(--text-dim)', marginTop: 10, lineHeight: 1.5 }}>
          {trainTokens < paramCount
            ? `The dataset has fewer tokens than the model has parameters, so it has room to memorize the text. Watch the validation loss on the Train tab.`
            : `The dataset has more tokens than the model has parameters (${(trainTokens / paramCount).toFixed(1)} per parameter), so memorizing it all is much harder.`}
        </p>
        {optimalBudget !== null && optimalBudget > 3600 && (
          <p style={{ fontSize: '0.75rem', color: 'var(--accent-amber)', marginTop: 8, lineHeight: 1.5 }}>
            A long run: expect hours, not minutes, before the loss levels off. Keep the tab open and the computer awake; it's
            fine to switch tabs or come back later.
          </p>
        )}
      </div>
    </div>
  );
};
