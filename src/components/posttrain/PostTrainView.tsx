import React, { useEffect, useMemo, useRef, useState } from 'react';
import { FastForward, Pause, Play, RotateCcw, Activity, TrendingDown, Microscope, Database, SlidersHorizontal } from 'lucide-react';
import { DatasetOption, TransformerConfig } from '../../types';
import { MicroTransformer, ModelState } from '../../engine/transformer';
import { BPETokenizer } from '../../engine/bpeTokenizer';
import { samplePromptFor, splitDataset } from '../../engine/datasets';
import { validationLoss } from '../../engine/training';
import { Trainer, TokenizedExample, makeTrainer } from '../../engine/posttrain/examples';
import { LoraAdapters } from '../../engine/posttrain/lora';
import { SftExample, prepareSft, sftStep } from '../../engine/posttrain/sft';
import { PairScore, PreferencePair, PreparedPair, dpoStep, preparePairs, scorePair, summarize } from '../../engine/posttrain/dpo';
import { RlGroup, greedyAccuracy, rlStep } from '../../engine/posttrain/grpo';
import { DPO_PRESETS, RL_PRESETS, SFT_PRESETS } from '../../engine/posttrain/presets';
import { InfoTooltip } from '../InfoTooltip';
import { GoDeeper } from '../GoDeeper';
import { THEME } from '../../styles/theme';
import { MetricChart, Series } from './MetricChart';
import { DpoEditor, SftEditor } from './DataEditors';
import { DpoInspector, RlInspector, SftInspector } from './Inspectors';
import { BeforeAfter } from './BeforeAfter';

export type PostTrainMethod = 'sft' | 'dpo' | 'rl';

/** The pre-trained model as it was when post-training started (App keeps it, and autosaves it) */
export interface PostTrainSession {
  base: ModelState;
  /** A frozen copy of the base: DPO's and RL's reference, and the "before" in comparisons */
  reference: MicroTransformer;
}

/** Each tick trains for about this long, then yields so the page can repaint */
const TICK_BUDGET_MS = 60;
const TICK_GAP_MS = 16;
/** Steps between measurements of validation loss, DPO scores and RL accuracy */
const EVAL_EVERY = 10;

/** Learning rates offered (a log scale: post-training wants far smaller steps than pre-training) */
const LR_CHOICES = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003];

const METHODS: { id: PostTrainMethod; name: string; short: string }[] = [
  { id: 'sft', name: 'Supervised fine-tuning', short: 'SFT' },
  { id: 'dpo', name: 'Preference tuning', short: 'DPO' },
  { id: 'rl', name: 'Reinforcement learning', short: 'RL' },
];

const ABOUT: Record<PostTrainMethod, { what: string; real: string; links: { label: string; url: string }[] }> = {
  sft: {
    what:
      'Keep training on next-token prediction, but only on hand-picked prompt → response examples, and only grade the ' +
      'response. This is how a base model that merely continues text becomes one that answers in a set format and stops.',
    real:
      'Labs use tens of thousands to millions of written conversations, wrapped in a chat template with special tokens ' +
      'marking each turn. Here, <EOS> plays the "end of turn" role.',
    links: [
      { label: 'InstructGPT (Ouyang et al., 2022)', url: 'https://arxiv.org/abs/2203.02155' },
      { label: 'Hugging Face TRL: SFT', url: 'https://huggingface.co/docs/trl/sft_trainer' },
    ],
  },
  dpo: {
    what:
      'Show the model two responses to the same prompt and say which is better. DPO raises the preferred one and lowers ' +
      'the other, measured against a frozen copy of the starting model so it can\'t drift arbitrarily far.',
    real:
      'Preference pairs come from people (or a stronger model) ranking the model\'s own replies. The older recipe, RLHF, ' +
      'trains a separate reward model on those rankings and then runs RL against it; DPO gets a similar result in one step.',
    links: [
      { label: 'DPO (Rafailov et al., 2023)', url: 'https://arxiv.org/abs/2305.18290' },
      { label: 'Hugging Face TRL: DPO', url: 'https://huggingface.co/docs/trl/dpo_trainer' },
    ],
  },
  rl: {
    what:
      'No example answers at all: the model answers each question several times, a program checks which answers are ' +
      'right, and answers that beat their group\'s average are made more likely (GRPO). A KL penalty keeps it close to the ' +
      'starting model.',
    real:
      'Reasoning models such as DeepSeek-R1 were trained this way on maths and code, where answers can be checked. It ' +
      'only works if the model already gets some answers right: RL sharpens a skill, it rarely creates one.',
    links: [
      { label: 'GRPO in DeepSeekMath (Shao et al., 2024)', url: 'https://arxiv.org/abs/2402.03300' },
      { label: 'DeepSeek-R1 (2025)', url: 'https://arxiv.org/abs/2501.12948' },
    ],
  },
};

/** One chart point: whatever was measured during one tick */
interface Point {
  step: number;
  sftLoss?: number;
  dpoLoss?: number;
  dpoAcc?: number;
  rlReward?: number;
  rlNoSignal?: number;
  rlTrainAcc?: number;
  rlHeldAcc?: number;
  kl?: number;
  valLoss?: number;
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : undefined);

interface PostTrainViewProps {
  model: MicroTransformer;
  tokenizer: BPETokenizer;
  config: TransformerConfig;
  selectedDataset: DatasetOption;
  session: PostTrainSession | null;
  /** Freeze a copy of the current model as the base and reference; returns the new session */
  onBeginSession: () => PostTrainSession;
  /** Put the pre-trained weights back and end the session */
  onRestoreBase: () => void;
  /** Pre-training is running (or handing back its weights); the two never run at once */
  pretrainBusy: boolean;
  visible: boolean;
  onRunningChange: (running: boolean) => void;
  onNavigateToTrain: () => void;
}

export const PostTrainView: React.FC<PostTrainViewProps> = ({
  model,
  tokenizer,
  config,
  selectedDataset,
  session,
  onBeginSession,
  onRestoreBase,
  pretrainBusy,
  visible,
  onRunningChange,
  onNavigateToTrain,
}) => {
  const dsId = selectedDataset.id;
  const rlPresets = RL_PRESETS[dsId] ?? [];

  // ── Data ────────────────────────────────────────────────────────────────
  const starterSft = (): SftExample[] => SFT_PRESETS[dsId]?.examples ?? [{ prompt: samplePromptFor(selectedDataset), response: '' }];
  const starterDpo = (): PreferencePair[] => DPO_PRESETS[dsId]?.pairs ?? [{ prompt: samplePromptFor(selectedDataset), chosen: '', rejected: '' }];
  const [method, setMethod] = useState<PostTrainMethod>('sft');
  const [sftExamples, setSftExamples] = useState<SftExample[]>(starterSft);
  const [dpoPairs, setDpoPairs] = useState<PreferencePair[]>(starterDpo);
  const [rlPresetId, setRlPresetId] = useState<string>(() => rlPresets[0]?.id ?? '');
  useEffect(() => {
    setSftExamples(starterSft());
    setDpoPairs(starterDpo());
    setRlPresetId(rlPresets[0]?.id ?? '');
    if (method === 'rl' && !rlPresets.length) setMethod('sft');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dsId]);
  const rlPreset = rlPresets.find(p => p.id === rlPresetId) ?? rlPresets[0];

  // ── Settings ────────────────────────────────────────────────────────────
  const [lr, setLr] = useState<Record<PostTrainMethod, number>>({ sft: 0.0003, dpo: 0.0001, rl: 0.001 });
  const [maskPrompt, setMaskPrompt] = useState(true);
  const [teachEos, setTeachEos] = useState(true);
  const [beta, setBeta] = useState(0.1);
  const [klBeta, setKlBeta] = useState(0.04);
  const [groupSize, setGroupSize] = useState(8);
  const [useLora, setUseLora] = useState(false);
  const [loraRank, setLoraRank] = useState(4);

  const [running, setRunning] = useState(false);

  const sftData = useMemo(
    () => prepareSft(tokenizer, sftExamples.filter(e => e.prompt || e.response), { contextWindow: config.contextWindow, eos: teachEos }),
    [tokenizer, sftExamples, config.contextWindow, teachEos]
  );
  // DPO compares against the reference, which is the current model until a session freezes one.
  // Scoring every pair is two forward passes each, and every Setup change builds a new model, so
  // it's only done when DPO is the chosen method and this tab is in use.
  const dpoInUse = method === 'dpo' && (visible || running);
  const dpoData = useMemo(
    () => (dpoInUse ? preparePairs(tokenizer, session?.reference ?? model, dpoPairs.filter(p => p.chosen && p.rejected), config.contextWindow) : []),
    [dpoInUse, tokenizer, session, model, dpoPairs, config.contextWindow]
  );

  const valTokens = useMemo(() => tokenizer.encode(splitDataset(selectedDataset.text).valText).tokens, [tokenizer, selectedDataset]);
  const trainText = useMemo(() => splitDataset(selectedDataset.text).trainText, [selectedDataset]);

  const loraCount = useMemo(() => new LoraAdapters(model, { rank: loraRank, targets: 'all' }).trainableCount(), [model, loraRank]);
  const fullCount = model.getParameterCount();

  // ── Run state ───────────────────────────────────────────────────────────
  const [stepCount, setStepCount] = useState(0);
  const [points, setPoints] = useState<Point[]>([]);
  const [sftView, setSftView] = useState<{ example: TokenizedExample; probs: number[] } | null>(null);
  const [dpoScores, setDpoScores] = useState<PairScore[]>([]);
  const [rlGroups, setRlGroups] = useState<RlGroup[]>([]);
  const [compareKey, setCompareKey] = useState(0);
  const trainer = useRef<Trainer | null>(null);
  const stepRef = useRef(0);
  const lastEval = useRef(-Infinity);

  useEffect(() => onRunningChange(running), [running, onRunningChange]);

  // The session ended (base restored, weights kept, or a new model): start the story over
  useEffect(() => {
    if (session) return;
    setRunning(false);
    trainer.current = null;
    stepRef.current = 0;
    lastEval.current = -Infinity;
    setStepCount(0);
    setPoints([]);
    setSftView(null);
    setDpoScores([]);
    setRlGroups([]);
    setCompareKey(k => k + 1);
  }, [session]);

  // Pre-training started (from the Train tab): stop here
  useEffect(() => {
    if (pretrainBusy) setRunning(false);
  }, [pretrainBusy]);

  // Compare before/after whenever a run pauses
  const wasRunning = useRef(false);
  useEffect(() => {
    if (wasRunning.current && !running) setCompareKey(k => k + 1);
    wasRunning.current = running;
  }, [running]);

  const latest = useRef({ method, sftData, dpoData, rlPreset, lr, maskPrompt, beta, klBeta, groupSize, session, model, tokenizer, valTokens, config });
  latest.current = { method, sftData, dpoData, rlPreset, lr, maskPrompt, beta, klBeta, groupSize, session, model, tokenizer, valTokens, config };

  /** Make sure there's a session and a trainer (the first step of a session creates both) */
  const ensureStarted = (): PostTrainSession => {
    const s = latest.current.session ?? onBeginSession();
    if (!trainer.current) {
      trainer.current = makeTrainer(useLora ? new LoraAdapters(model, { rank: loraRank, targets: 'all' }) : null);
    }
    return s;
  };

  const canTrain = (m: PostTrainMethod) =>
    m === 'sft' ? sftData.length > 0 : m === 'dpo' ? dpoData.length > 0 : !!rlPreset;

  /**
   * Train for up to `budgetMs` (at least one step), then record one chart point: the mean of
   * the steps' metrics, plus validation loss and the method's evaluation every EVAL_EVERY steps.
   */
  const runSteps = (budgetMs: number, reference: MicroTransformer) => {
    const L = latest.current;
    const t = trainer.current;
    if (!t || !canTrain(L.method)) return;
    const t0 = performance.now();
    const sft: number[] = [], dpoLoss: number[] = [], rew: number[] = [], noSig: number[] = [], kl: number[] = [];
    let lastGroups: RlGroup[] | null = null;
    do {
      if (L.method === 'sft') {
        sft.push(sftStep(L.model, t, L.sftData, { lr: L.lr.sft, batchSize: 4, maskPrompt: L.maskPrompt }).loss);
      } else if (L.method === 'dpo') {
        dpoLoss.push(dpoStep(L.model, t, L.dpoData, { beta: L.beta, lr: L.lr.dpo, batchSize: 4 }).loss);
      } else if (L.rlPreset) {
        const r = rlStep(L.model, reference, L.tokenizer, t, L.rlPreset.train, {
          groupSize: L.groupSize, batchSize: 2, temperature: 1, klBeta: L.klBeta, lr: L.lr.rl,
        });
        rew.push(r.reward);
        noSig.push(r.noSignalShare);
        kl.push(r.kl);
        lastGroups = r.groups;
      }
      stepRef.current++;
    } while (performance.now() - t0 < budgetMs);

    const point: Point = { step: stepRef.current, sftLoss: mean(sft), rlReward: mean(rew), rlNoSignal: mean(noSig), kl: mean(kl) };
    if (L.method === 'dpo') {
      // Score every pair (not just the batch), so the curve is smooth
      const scores = L.dpoData.map(p => scorePair(L.model, p, L.beta));
      const m = summarize(scores);
      point.dpoLoss = m.loss;
      point.dpoAcc = m.accuracy;
      setDpoScores(scores);
    }
    if (stepRef.current - lastEval.current >= EVAL_EVERY) {
      lastEval.current = stepRef.current;
      point.valLoss = validationLoss(L.model, L.valTokens, L.config.contextWindow) ?? undefined;
      if (L.method === 'rl' && L.rlPreset) {
        point.rlTrainAcc = greedyAccuracy(L.model, L.tokenizer, L.rlPreset.train);
        point.rlHeldAcc = greedyAccuracy(L.model, L.tokenizer, L.rlPreset.heldOut);
      }
    }
    if (L.method === 'sft' && L.sftData.length) {
      const ex = L.sftData[Math.floor(Math.random() * L.sftData.length)];
      setSftView({ example: ex, probs: L.model.tokenLogProbs(ex.tokens).map(Math.exp) });
    }
    if (lastGroups) setRlGroups(lastGroups);
    setStepCount(stepRef.current);
    setPoints(p => [...p, point]);
  };

  // The run loop: one time-boxed tick at a time, with a gap so the page stays responsive
  useEffect(() => {
    if (!running) return;
    const s = ensureStarted();
    let timer: number;
    const tick = () => {
      runSteps(TICK_BUDGET_MS, s.reference);
      timer = window.setTimeout(tick, TICK_GAP_MS);
    };
    timer = window.setTimeout(tick, 0);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  const stepOnce = () => {
    const s = ensureStarted();
    runSteps(0, s.reference);
    setCompareKey(k => k + 1);
  };

  // ── Charts ──────────────────────────────────────────────────────────────
  const labels = points.map(p => `#${p.step}`);
  const col = (key: keyof Point) => points.map(p => (p[key] === undefined ? null : (p[key] as number)));
  const signal: Record<PostTrainMethod, { title: string; yTitle: string; yRange?: [number, number]; series: Series[] }> = {
    sft: {
      title: 'Response loss',
      yTitle: 'Loss on response tokens',
      series: [{ label: 'Response loss (batch)', color: THEME.primary, data: col('sftLoss') }],
    },
    dpo: {
      title: 'Preference fit',
      yTitle: 'Loss / share of pairs',
      series: [
        { label: 'DPO loss', color: THEME.primary, data: col('dpoLoss') },
        { label: 'Pairs where chosen wins', color: THEME.emerald, data: col('dpoAcc') },
      ],
    },
    rl: {
      title: 'Reward',
      yTitle: 'Share (0 to 1)',
      yRange: [0, 1],
      series: [
        { label: 'Mean reward (sampled)', color: THEME.primary, data: col('rlReward') },
        { label: 'Greedy: training questions', color: THEME.emerald, data: col('rlTrainAcc') },
        { label: 'Greedy: held-out questions', color: THEME.amber, data: col('rlHeldAcc'), dashed: true },
        { label: 'Groups with no signal', color: THEME.textDim, data: col('rlNoSignal'), dashed: true },
      ],
    },
  };
  const cost: Series[] = [
    { label: 'Validation loss on pre-training text', color: THEME.amber, data: col('valLoss') },
    ...(method === 'rl' ? [{ label: 'KL from the pre-trained model', color: THEME.purple, data: col('kl'), dashed: true }] : []),
  ];

  // ── Comparison prompts ──────────────────────────────────────────────────
  const comparePrompts = useMemo(() => {
    if (method === 'rl' && rlPreset) return [...rlPreset.heldOut.slice(0, 4), ...rlPreset.train.slice(0, 3)].map(t => t.prompt);
    const preset = method === 'sft' ? SFT_PRESETS[dsId] : DPO_PRESETS[dsId];
    const own = method === 'sft' ? SFT_PRESETS[dsId]?.examples.map(e => e.prompt) : DPO_PRESETS[dsId]?.pairs.map(p => p.prompt);
    return preset ? [...new Set([...(own ?? []).slice(0, 2), ...preset.tryPrompts])] : [samplePromptFor(selectedDataset)];
  }, [method, rlPreset, dsId, selectedDataset]);

  const started = !!session && stepCount > 0;
  const about = ABOUT[method];
  const preset = method === 'sft' ? SFT_PRESETS[dsId] : method === 'dpo' ? DPO_PRESETS[dsId] : rlPreset;
  const lrIndex = Math.max(0, LR_CHOICES.indexOf(lr[method]));

  const label: React.CSSProperties = { fontSize: '0.8rem', fontWeight: 600 };
  const value: React.CSSProperties = { fontSize: '0.8rem', color: 'var(--accent-rose)' };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 480px), 1fr))', gap: 20 }}>
      {/* Method picker + controls */}
      <div className="glass-panel" style={{ padding: '16px 24px', gridColumn: '1 / -1', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {METHODS.map(m => {
            const disabled = m.id === 'rl' && !rlPresets.length;
            return (
              <button
                key={m.id}
                className={method === m.id ? 'btn-primary' : 'btn-secondary'}
                onClick={() => setMethod(m.id)}
                disabled={running || disabled}
                title={disabled ? 'RL needs answers a program can check: switch to the Synthetic Math & Logic dataset in Setup' : undefined}
                style={{ padding: '8px 14px', fontSize: '0.85rem' }}
              >
                <span className="font-mono" style={{ fontWeight: 800 }}>{m.short}</span> {m.name}
              </button>
            );
          })}
          <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="badge badge-primary font-mono">Post-training step #{stepCount}</span>
          </span>
        </div>

        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          <p><b style={{ color: 'var(--text-main)' }}>What it does.</b> {about.what}</p>
          <p style={{ marginTop: 6 }}><b style={{ color: 'var(--text-main)' }}>At real scale.</b> {about.real}</p>
          <div style={{ marginTop: 8 }}><GoDeeper links={about.links} /></div>
        </div>

        {pretrainBusy && (
          <p style={{ fontSize: '0.8rem', color: 'var(--accent-amber)' }}>
            Pre-training is running. Pause it on the{' '}
            <button onClick={onNavigateToTrain} style={{ background: 'none', border: 'none', padding: 0, color: 'var(--primary)', cursor: 'pointer', font: 'inherit' }}>
              Train tab
            </button>{' '}
            first: both would change the same weights.
          </p>
        )}

        <div style={{ display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap', paddingTop: 14, borderTop: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <button className={running ? 'btn-secondary' : 'btn-primary'} onClick={() => setRunning(!running)} disabled={!running && (pretrainBusy || !canTrain(method))}>
              {running ? <Pause size={16} /> : <Play size={16} />}
              {running ? 'Pause' : started ? 'Continue post-training' : 'Start post-training'}
            </button>
            <button className="btn-secondary" onClick={stepOnce} disabled={running || pretrainBusy || !canTrain(method)}>
              <FastForward size={16} /> Step
            </button>
            <button
              className="btn-secondary"
              onClick={onRestoreBase}
              disabled={!session || running}
              title="Put the pre-trained weights back and clear these charts"
            >
              <RotateCcw size={16} /> Restore the pre-trained model
            </button>
          </div>

          <div style={{ flex: '1 1 200px', minWidth: 180 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={label}>Learning rate</span>
                <InfoTooltip
                  title="Post-training learning rate"
                  description="Post-training nudges a model that already works, so it uses much smaller steps than pre-training. Each method keeps its own setting."
                  impact="Too high and the model forgets what it knew, or (DPO, RL) collapses into repeating itself. Real models use rates around 1e-5 to 1e-7."
                />
              </span>
              <span className="font-mono" style={value}>{lr[method]}</span>
            </div>
            <input type="range" min={0} max={LR_CHOICES.length - 1} step={1} value={lrIndex}
              onChange={e => setLr(prev => ({ ...prev, [method]: LR_CHOICES[Number(e.target.value)] }))} />
          </div>

          <label
            style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.8rem', cursor: started ? 'not-allowed' : 'pointer' }}
            title={started ? 'Restore the pre-trained model to switch between LoRA and full fine-tuning' : undefined}
          >
            <input type="checkbox" checked={useLora} disabled={started} onChange={e => setUseLora(e.target.checked)} />
            <span style={label}>LoRA</span>
            <InfoTooltip
              title="LoRA (low-rank adaptation)"
              description="Freeze every weight and learn a small correction for each block matrix instead: W + A·B, where A and B are thin (rank r). Only A and B are trained."
              impact="Far fewer numbers to train and store, the original model stays intact underneath, and it usually forgets less. It learns more slowly, so it is normally given a larger learning rate. In a model this small the adapters are a bigger share than in a real one (where they are well under 1%)."
            />
          </label>
          {useLora && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: '0.8rem' }}>rank</span>
              <select value={loraRank} disabled={started} onChange={e => setLoraRank(Number(e.target.value))}>
                {[1, 2, 4, 8].map(r => <option key={r} value={r}>{r}</option>)}
              </select>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                trains {loraCount.toLocaleString()} of {fullCount.toLocaleString()} numbers ({((loraCount / fullCount) * 100).toFixed(1)}%)
              </span>
            </div>
          )}
        </div>

        {/* Method-specific settings */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap' }}>
          <SlidersHorizontal size={16} color="var(--text-dim)" />
          {method === 'sft' && (
            <>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem' }}>
                <input type="checkbox" checked={maskPrompt} onChange={e => setMaskPrompt(e.target.checked)} />
                Grade the response only (mask the prompt)
                <InfoTooltip
                  title="Loss masking"
                  description="With masking, the loss only counts the response tokens: the model learns to answer, not to write questions. Without it, every token counts, prompt included."
                  impact="Turn it off and watch the model start inventing prompts of its own."
                />
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem' }}>
                <input type="checkbox" checked={teachEos} disabled={running} onChange={e => setTeachEos(e.target.checked)} />
                Teach it to stop (end each response with &lt;EOS&gt;)
              </label>
            </>
          )}
          {method === 'dpo' && (
            <div style={{ flex: '0 1 260px', minWidth: 180 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={label}>β (beta)</span>
                  <InfoTooltip
                    title="DPO's β"
                    description="How sharply the loss reacts to the gap between chosen and rejected, measured against the frozen starting model. A pair stops pulling once β × (gap) is comfortably positive."
                    impact="Large β: the model is satisfied after a small change and stays close to where it started. Small β: it keeps pushing, further from the starting model."
                  />
                </span>
                <span className="font-mono" style={value}>{beta}</span>
              </div>
              <input type="range" min={0.05} max={1} step={0.05} value={beta} onChange={e => setBeta(Number(e.target.value))} />
            </div>
          )}
          {method === 'rl' && (
            <>
              <select value={rlPresetId} disabled={running} onChange={e => setRlPresetId(e.target.value)} style={{ fontSize: '0.8rem' }}>
                {rlPresets.map(p => <option key={p.id} value={p.id}>{p.title}</option>)}
              </select>
              <div style={{ flex: '0 1 200px', minWidth: 160 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={label}>KL penalty</span>
                    <InfoTooltip
                      title="KL penalty (β)"
                      description="A cost for every token the model now writes with a very different probability than the pre-trained model would. It keeps RL from wandering far from what the model knew."
                      impact="Set it to 0 and train for a while: the reward may hold up while the text around the answer turns to nonsense."
                    />
                  </span>
                  <span className="font-mono" style={value}>{klBeta}</span>
                </div>
                <input type="range" min={0} max={0.2} step={0.01} value={klBeta} onChange={e => setKlBeta(Number(e.target.value))} />
              </div>
              <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem' }}>
                Answers per question
                <select value={groupSize} onChange={e => setGroupSize(Number(e.target.value))}>
                  {[4, 8, 16].map(n => <option key={n} value={n}>{n}</option>)}
                </select>
              </label>
            </>
          )}
        </div>
      </div>

      {/* Data */}
      <div className="glass-panel" style={{ padding: 24, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Database size={20} color="var(--accent-purple)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>
            {method === 'sft' ? 'Examples' : method === 'dpo' ? 'Preference pairs' : 'Questions'}
            {preset && <span style={{ fontWeight: 400, color: 'var(--text-muted)' }}>: {preset.title}</span>}
          </h3>
        </div>
        {preset && <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{preset.goal}</p>}
        {method === 'sft' && (
          <SftEditor
            examples={sftExamples}
            onChange={setSftExamples}
            onReset={() => setSftExamples(starterSft())}
            tokenizer={tokenizer}
            contextWindow={config.contextWindow}
            eos={teachEos}
            maskPrompt={maskPrompt}
            disabled={running}
          />
        )}
        {method === 'dpo' && (
          <DpoEditor
            pairs={dpoPairs}
            onChange={setDpoPairs}
            onReset={() => setDpoPairs(starterDpo())}
            tokenizer={tokenizer}
            contextWindow={config.contextWindow}
            disabled={running}
          />
        )}
        {method === 'rl' && rlPreset && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: '0.8rem' }}>
            {(['train', 'heldOut'] as const).map(split => (
              <div key={split}>
                <p style={{ fontWeight: 600, marginBottom: 4, color: split === 'train' ? 'var(--text-main)' : 'var(--accent-amber)' }}>
                  {split === 'train' ? `Trained on (${rlPreset.train.length})` : `Held out, never trained on (${rlPreset.heldOut.length})`}
                </p>
                <div className="font-mono" style={{ display: 'flex', flexWrap: 'wrap', gap: 4, maxHeight: 120, overflowY: 'auto' }}>
                  {rlPreset[split].map(t => (
                    <span key={t.prompt} title={`accepted: ${t.answers.length > 6 ? `${t.answers[0]} … ${t.answers[t.answers.length - 1]}` : t.answers.join(', ')}`}
                      style={{ padding: '1px 6px', borderRadius: 4, background: 'var(--surface-inset)', border: '1px solid var(--border-color)', whiteSpace: 'pre' }}>
                      {t.prompt.trimEnd()} …
                    </span>
                  ))}
                </div>
              </div>
            ))}
            <p style={{ color: 'var(--text-dim)' }}>
              The checker accepts any answer on a question's list (hover a question to see it), and only the first word counts.
            </p>
          </div>
        )}
      </div>

      {/* Step inspector */}
      <div className="glass-panel" style={{ padding: 24, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Microscope size={20} color="var(--accent-emerald)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Inside the latest step</h3>
        </div>
        {method === 'sft' && (sftView
          ? <SftInspector example={sftView.example} probs={sftView.probs} tokenizer={tokenizer} maskPrompt={maskPrompt} />
          : <Empty />)}
        {method === 'dpo' && (dpoScores.length === dpoData.length && dpoScores.length
          ? <DpoInspector pairs={dpoData} scores={dpoScores} />
          : <Empty />)}
        {method === 'rl' && (rlGroups.length ? <RlInspector groups={rlGroups} /> : <Empty />)}
      </div>

      {/* Charts */}
      <div className="glass-panel" style={{ padding: 24, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
          <Activity size={20} color="var(--primary)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>{signal[method].title}</h3>
        </div>
        <MetricChart labels={labels} series={signal[method].series} yTitle={signal[method].yTitle} yRange={signal[method].yRange} visible={visible} />
      </div>
      <div className="glass-panel" style={{ padding: 24, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
          <TrendingDown size={20} color="var(--accent-amber)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>The cost: drifting from pre-training</h3>
          <InfoTooltip
            title="Forgetting"
            description="Validation loss on the held-out part of the pre-training text. Post-training only shows the model its own small dataset, so this usually rises: the model trades some of what it knew for the new behaviour."
            impact="Labs fight this with small learning rates, few steps, LoRA, a KL penalty, or by mixing some pre-training data back in."
          />
        </div>
        <MetricChart labels={labels} series={cost} yTitle="Loss / KL" visible={visible} />
      </div>

      <div style={{ gridColumn: '1 / -1', minWidth: 0 }}>
        <BeforeAfter
          base={session?.reference ?? model}
          current={model}
          tokenizer={tokenizer}
          datasetId={dsId}
          trainText={trainText}
          defaultPrompts={comparePrompts}
          refreshKey={compareKey}
          untouched={!started}
          visible={visible}
        />
      </div>
    </div>
  );
};

const Empty: React.FC = () => (
  <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>Press Start or Step to see what one post-training step does.</p>
);
