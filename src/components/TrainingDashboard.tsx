import React, { useState, useEffect, useRef, useMemo } from 'react';
import { DatasetOption, TransformerConfig } from '../types';
import { MicroTransformer } from '../engine/transformer';
import { BPETokenizer } from '../engine/bpeTokenizer';
import { splitDataset, samplePromptFor, VALIDATION_FRACTION } from '../engine/datasets';
import { generateContinuation } from '../engine/generate';
import { Play, Pause, RotateCcw, FastForward, Activity, Sparkles, BookOpen, Turtle } from 'lucide-react';
import { InfoTooltip } from './InfoTooltip';
import { BenchmarkPanel } from './BenchmarkPanel';
import { SlowMotionPanel, SlowStep, STAGES } from './SlowMotionPanel';
import { THEME, withAlpha } from '../styles/theme';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

/** Measure validation loss every N training steps (it runs the model over the whole held-out split) */
const VAL_EVERY = 50;
const MAX_VAL_WINDOWS = 32;

/** Slow motion: how long each stage (Sample, Forward, Compare, Backward, Update) stays on screen */
const STAGE_MS = [1600, 1600, 2600, 1600, 3000];

/**
 * While training runs, each tick does as many steps as fit in this many milliseconds, then
 * yields so the page can repaint. The chart and validation update once per tick, not once
 * per step, so almost all of the time goes to actual training.
 */
const TICK_BUDGET_MS = 100;
const TICK_GAP_MS = 16;

interface TrainingDashboardProps {
  model: MicroTransformer;
  tokenizer: BPETokenizer;
  config: TransformerConfig;
  selectedDataset: DatasetOption;
  onNavigateToSetup: () => void;
  onTrainingChange?: (isTraining: boolean) => void;
  onChangeLearningRate: (lr: number) => void;
  onChangeOptimizer: (opt: TransformerConfig['optimizer']) => void;
  visible: boolean; // whether the Train tab is showing (it stays mounted while hidden)
}

export const TrainingDashboard: React.FC<TrainingDashboardProps> = ({
  model,
  tokenizer,
  config,
  selectedDataset,
  onNavigateToSetup,
  onTrainingChange,
  onChangeLearningRate,
  onChangeOptimizer,
  visible,
}) => {
  const [isTraining, setIsTraining] = useState<boolean>(false);
  useEffect(() => {
    onTrainingChange?.(isTraining);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTraining]);
  const [stepCount, setStepCount] = useState<number>(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [valLossHistory, setValLossHistory] = useState<(number | null)[]>([]);
  const [stepLabels, setStepLabels] = useState<string[]>([]);
  
  // Sampling controls
  const [seedPrompt, setSeedPrompt] = useState<string>(() => samplePromptFor(selectedDataset));
  const [temperature, setTemperature] = useState<number>(0.7);
  const [maxGenTokens, setMaxGenTokens] = useState<number>(40);
  const [generatedText, setGeneratedText] = useState<string>('');
  // New dataset, new example prompt
  useEffect(() => {
    setSeedPrompt(samplePromptFor(selectedDataset));
    setGeneratedText('');
  }, [selectedDataset]);

  // Train on the first part of the dataset; hold out the tail to measure generalization
  const split = useMemo(() => splitDataset(selectedDataset.text), [selectedDataset]);
  const trainTokens = useMemo(() => tokenizer.encode(split.trainText).tokens, [tokenizer, split]);
  const valTokens = useMemo(() => tokenizer.encode(split.valText).tokens, [tokenizer, split]);

  // Refs so the timer callback always sees the latest values (no stale closures)
  const stepRef = useRef<number>(0);
  const latest = useRef({ model, config, trainTokens, valTokens });
  latest.current = { model, config, trainTokens, valTokens };

  const resetHistory = () => {
    stepRef.current = 0;
    setStepCount(0);
    setLossHistory([]);
    setValLossHistory([]);
    setStepLabels([]);
  };

  // A new model instance means fresh random weights — the old curve no longer applies
  useEffect(() => {
    setIsTraining(false);
    resetHistory();
    setSlowStep(null);
    setStage(0);
  }, [model]);

  /**
   * Average loss over windows of the held-out tail (forward pass only). Large splits are
   * capped at MAX_VAL_WINDOWS evenly spaced windows — the same ones every time, so the
   * curve stays smooth — to keep each measurement from freezing the page.
   */
  const computeValLoss = (): number | null => {
    const { model, config, valTokens } = latest.current;
    if (valTokens.length < 2) return null;
    const W = config.contextWindow;
    const numWindows = Math.ceil((valTokens.length - 1) / W);
    const stride = W * Math.max(1, Math.ceil(numWindows / MAX_VAL_WINDOWS));
    let total = 0;
    let count = 0;
    for (let start = 0; start < valTokens.length - 1; start += stride) {
      const input = valTokens.slice(start, start + W);
      const target = valTokens.slice(start + 1, start + W + 1);
      const n = Math.min(input.length, target.length);
      if (n === 0) continue;
      total += model.evaluateLoss(input.slice(0, n), target.slice(0, n)).loss;
      count++;
    }
    return count > 0 ? total / count : null;
  };

  /**
   * Run training steps until `maxSteps` are done or `budgetMs` has elapsed, then record
   * ONE chart point (the mean loss over those steps).
   */
  const runTrainingSteps = (maxSteps: number, budgetMs: number) => {
    const { model, config, trainTokens } = latest.current;
    if (trainTokens.length < 2) return;

    const startStep = stepRef.current;
    const t0 = performance.now();
    let lossSum = 0;
    let steps = 0;
    do {
      const { inputSeq, targetSeq } = pickWindow();
      lossSum += model.trainStep(inputSeq, targetSeq, config.learningRate).loss;
      steps++;
    } while (steps < maxSteps && performance.now() - t0 < budgetMs);

    recordSteps(startStep, lossSum, steps);
  };

  /**
   * Pick a random context-sized window from the TRAINING split only. Random sampling
   * (rather than sliding one token per step) means every part of the text gets seen early.
   */
  const pickWindow = () => {
    const { config, trainTokens } = latest.current;
    const maxStart = Math.max(1, trainTokens.length - config.contextWindow - 1);
    const startIdx = Math.floor(Math.random() * maxStart);
    return {
      startIdx,
      inputSeq: trainTokens.slice(startIdx, startIdx + config.contextWindow),
      targetSeq: trainTokens.slice(startIdx + 1, startIdx + config.contextWindow + 1),
    };
  };

  /** Advance the step counter and add ONE chart point (mean loss over `steps` steps) */
  const recordSteps = (startStep: number, lossSum: number, steps: number) => {
    stepRef.current += steps;
    const nextStep = stepRef.current;
    // Validate on the first step and whenever this batch crossed a multiple of VAL_EVERY
    const crossedVal = Math.floor(nextStep / VAL_EVERY) > Math.floor(startStep / VAL_EVERY);
    const valLoss = startStep === 0 || crossedVal ? computeValLoss() : null;

    setStepCount(nextStep);
    setStepLabels(l => [...l, `#${nextStep}`]);
    setLossHistory(h => [...h, Number((lossSum / steps).toFixed(4))]);
    setValLossHistory(h => [...h, valLoss === null ? null : Number(valLoss.toFixed(4))]);
  };

  /**
   * One training step that also captures everything the slow-motion panel replays:
   * per-position predictions and errors, per-layer gradient sizes, and the same window
   * re-checked after the update. Costs a few extra forward passes, fine at slow-motion speed.
   */
  const runDetailedStep = (): SlowStep | null => {
    const { model, config, trainTokens } = latest.current;
    if (trainTokens.length < 2) return null;
    const startStep = stepRef.current;
    const { startIdx, inputSeq, targetSeq } = pickWindow();
    const n = Math.min(inputSeq.length, targetSeq.length);
    const strs = (ids: number[]) => ids.map(id => tokenizer.decode([id]));

    const before = model.inspectForwardPass(inputSeq.slice(0, n), strs(inputSeq.slice(0, n)));
    const probsBefore = before.probabilities.map((row, i) => row[targetSeq[i]] ?? 0);
    const losses = probsBefore.map(p => -Math.log(Math.max(p, 1e-10)));
    const preds = before.probabilities.map(row => row.reduce((best, v, k) => (v > row[best] ? k : best), 0));

    // Gradient size per part of the model (computed before the update, from the same window)
    const { grads } = model.computeGradients(inputSeq.slice(0, n), targetSeq.slice(0, n));
    const norm = (names: string[]) =>
      Math.sqrt(names.reduce((s, name) => s + grads[name].reduce((t, row) => t + row.reduce((u, v) => u + v * v, 0), 0), 0));
    const names = Object.keys(grads);
    const gradGroups = [
      { name: 'Embedding', norm: norm(['wTokenEmbed', 'wPosEmbed']) },
      ...Array.from({ length: config.numLayers }, (_, l) => ({
        name: `Block ${l + 1}`,
        norm: norm(names.filter(k => k.endsWith(`.${l}`))),
      })),
      { name: 'Output head', norm: norm(['wHead']) },
    ];

    const result = model.trainStep(inputSeq.slice(0, n), targetSeq.slice(0, n), config.learningRate);

    const after = model.inspectForwardPass(inputSeq.slice(0, n), strs(inputSeq.slice(0, n)));
    const probsAfter = after.probabilities.map((row, i) => row[targetSeq[i]] ?? 0);
    const lossAfter = probsAfter.reduce((s, p) => s - Math.log(Math.max(p, 1e-10)), 0) / n;

    recordSteps(startStep, result.loss, 1);
    return {
      stepNumber: stepRef.current,
      windowStart: startIdx,
      totalTokens: trainTokens.length,
      inputStrs: strs(inputSeq.slice(0, n)),
      targetStrs: strs(targetSeq.slice(0, n)),
      predStrs: strs(preds),
      predProbs: preds.map((k, i) => before.probabilities[i][k]),
      probsBefore,
      probsAfter,
      losses,
      lossBefore: result.loss,
      lossAfter,
      gradGroups,
      learningRate: config.learningRate,
    };
  };

  // ── Slow motion: one detailed step at a time, replayed stage by stage ──
  const [slowMo, setSlowMo] = useState<boolean>(false);
  const [slowStep, setSlowStep] = useState<SlowStep | null>(null);
  const [stage, setStage] = useState<number>(0);
  const [spotlight, setSpotlight] = useState<number>(Infinity); // Infinity = last position

  const startSlowStep = () => {
    const s = runDetailedStep();
    if (!s) return;
    setSlowStep(s);
    setStage(1);
  };

  // While playing, advance one stage at a time; after the last stage, run the next step
  useEffect(() => {
    if (!slowMo || !isTraining) return;
    const timer = window.setTimeout(
      () => (!slowStep || stage >= STAGES.length ? startSlowStep() : setStage(st => st + 1)),
      slowStep ? STAGE_MS[stage - 1] ?? 0 : 0
    );
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slowMo, isTraining, stage, slowStep]);

  // Paused: the panel's button steps through stages by hand
  const nextSlowStage = () => (!slowStep || stage >= STAGES.length ? startSlowStep() : setStage(st => st + 1));

  // "Step" button: exactly one training step (replayed stage by stage in slow motion)
  const performTrainStep = () => (slowMo ? startSlowStep() : runTrainingSteps(1, Infinity));

  // Continuous loop: a time-boxed batch of steps per tick, with a short gap so React can repaint
  useEffect(() => {
    if (!isTraining || slowMo) return;
    let timer: number;
    const tick = () => {
      runTrainingSteps(Infinity, TICK_BUDGET_MS);
      timer = window.setTimeout(tick, TICK_GAP_MS);
    };
    timer = window.setTimeout(tick, 0);
    return () => clearTimeout(timer);
  }, [isTraining, slowMo]);

  // Handle Text Generation Sampling
  const handleGenerate = () => {
    const continuation = generateContinuation(model, tokenizer, seedPrompt, {
      maxTokens: maxGenTokens,
      temperature,
    });
    setGeneratedText(seedPrompt + continuation);
  };

  // The chart is created while the tab may be hidden (0×0) and doesn't always notice when it
  // becomes visible, so resize it explicitly whenever the Train tab is shown
  const chartBoxRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!visible) return;
    const timer = window.setTimeout(() => {
      const canvas = chartBoxRef.current?.querySelector('canvas');
      if (canvas) ChartJS.getChart(canvas)?.resize();
    });
    return () => clearTimeout(timer);
  }, [visible]);

  const chartData = {
    labels: stepLabels,
    datasets: [
      {
        label: 'Training Loss',
        data: lossHistory,
        borderColor: THEME.primary,
        backgroundColor: withAlpha(THEME.primary, 0.12),
        tension: 0.2,
        fill: true,
        pointRadius: lossHistory.length > 60 ? 0 : 3,
        pointHoverRadius: 5,
      },
      {
        label: `Validation Loss (held-out ${Math.round(VALIDATION_FRACTION * 100)}%)`,
        data: valLossHistory,
        borderColor: THEME.amber,
        backgroundColor: withAlpha(THEME.amber, 0.12),
        borderDash: [6, 4],
        tension: 0.2,
        fill: false,
        spanGaps: true,
        pointRadius: 2,
        pointHoverRadius: 5,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false as const,
    plugins: {
      legend: {
        labels: { color: THEME.textMuted, font: { family: 'Inter' } }
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
      }
    },
    scales: {
      x: {
        ticks: { color: THEME.textDim, maxTicksLimit: 12 },
        grid: { color: withAlpha(THEME.border, 0.6) }
      },
      y: {
        title: { display: true, text: 'Loss (lower is better)', color: THEME.textMuted, font: { family: 'Inter', size: 11 } },
        ticks: { color: THEME.textDim },
        grid: { color: withAlpha(THEME.border, 0.6) }
      }
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 480px), 1fr))', gap: '20px' }}>
      {/* Training controls: full width, so the slow-motion panel can sit directly underneath */}
      <div className="glass-panel" style={{ padding: '16px 24px', gridColumn: '1 / -1', display: 'flex', flexDirection: 'column', gap: '14px' }}>
        {/* What's being trained on (chosen in Setup, since changing it resets the model) */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <BookOpen size={20} color="var(--accent-purple)" />
            <div>
              <p style={{ fontSize: '0.95rem', fontWeight: 700 }}>Training on: {selectedDataset.name}</p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                First {trainTokens.length} tokens for training · last {valTokens.length} held out for validation
              </p>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <span className="badge badge-primary font-mono">Step: #{stepCount}</span>
            <button className="btn-secondary" onClick={onNavigateToSetup} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
              Change in Setup
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '18px', flexWrap: 'wrap', paddingTop: '14px', borderTop: '1px solid var(--border-color)' }}>
          {/* Actions */}
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
            <button
              className={isTraining ? 'btn-secondary' : 'btn-primary'}
              onClick={() => setIsTraining(!isTraining)}
            >
              {isTraining ? <Pause size={16} /> : <Play size={16} />}
              {isTraining ? 'Pause Training' : slowMo ? 'Play Slow Motion' : 'Start Continuous Train'}
            </button>

            <button className="btn-secondary" onClick={performTrainStep}>
              <FastForward size={16} /> Step
            </button>

            <button
              className="btn-secondary"
              onClick={() => {
                setIsTraining(false);
                resetHistory();
              }}
              title="Clears the chart. Weights are only re-initialized when the architecture, tokenizer or dataset changes."
            >
              <RotateCcw size={16} /> Reset
            </button>
          </div>

          {/* Settings that are safe to change while training runs */}
          <div style={{ flex: '1 1 200px', minWidth: '180px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <label style={{ fontSize: '0.8rem', fontWeight: 600 }}>Learning rate</label>
                <InfoTooltip
                  title="Learning Rate"
                  description="How big a step each weight takes on every update. Safe to change while training: the weights are kept."
                  impact="Too high and the loss jumps around or explodes; too low and learning crawls. 0.001 works well with AdamW here."
                />
              </div>
              <span className="font-mono" style={{ fontSize: '0.8rem', color: 'var(--accent-rose)' }}>{config.learningRate}</span>
            </div>
            <input
              type="range"
              min={0.001}
              max={0.05}
              step={0.001}
              value={config.learningRate}
              onChange={(e) => onChangeLearningRate(Number(e.target.value))}
            />
          </div>

          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
              <label style={{ fontSize: '0.8rem', fontWeight: 600 }}>Optimizer</label>
              <InfoTooltip
                title="Optimizer"
                description="SGD steps every weight by lr × gradient. AdamW keeps running averages of each weight's gradient (momentum) and squared gradient, giving every weight its own adaptive step size, and applies weight decay separately from the gradient."
                impact="AdamW usually converges much faster on transformers. Switching resets AdamW's running averages but keeps the learned weights."
              />
            </div>
            <div style={{ display: 'flex', gap: '6px' }}>
              {(['adamw', 'sgd'] as const).map(opt => (
                <button
                  key={opt}
                  className={config.optimizer === opt ? 'btn-primary' : 'btn-secondary'}
                  style={{ padding: '4px 12px', fontSize: '0.8rem' }}
                  onClick={() => onChangeOptimizer(opt)}
                >
                  {opt === 'adamw' ? 'AdamW' : 'SGD'}
                </button>
              ))}
            </div>
          </div>

          {/* Slow motion is a viewing mode, not an action: a switch, set apart at the end */}
          <button
            role="switch"
            aria-checked={slowMo}
            onClick={() => setSlowMo(!slowMo)}
            title="Replay training one step at a time: sample, forward pass, error, backpropagation, update"
            style={{
              marginLeft: 'auto',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
              padding: '8px 14px',
              borderRadius: '10px',
              cursor: 'pointer',
              border: `1px solid ${slowMo ? 'var(--accent-emerald)' : 'var(--border-color)'}`,
              background: slowMo ? 'var(--emerald-soft)' : 'var(--surface-inset)',
              color: 'var(--text-main)',
            }}
          >
            <Turtle size={18} color={slowMo ? 'var(--accent-emerald)' : 'var(--text-muted)'} />
            <span style={{ textAlign: 'left', lineHeight: 1.25 }}>
              <span style={{ display: 'block', fontSize: '0.85rem', fontWeight: 700 }}>Slow motion</span>
              <span style={{ display: 'block', fontSize: '0.7rem', color: 'var(--text-dim)' }}>See inside each step</span>
            </span>
            {/* the switch itself */}
            <span
              style={{
                width: 34,
                height: 20,
                borderRadius: 10,
                position: 'relative',
                flexShrink: 0,
                background: slowMo ? 'var(--accent-emerald)' : 'var(--border-strong)',
                transition: 'background 0.15s',
              }}
            >
              <span
                style={{
                  position: 'absolute',
                  top: 2,
                  left: slowMo ? 16 : 2,
                  width: 16,
                  height: 16,
                  borderRadius: 8,
                  background: '#ffffff',
                  transition: 'left 0.15s',
                }}
              />
            </span>
          </button>
        </div>
      </div>

      {slowMo && (
        <SlowMotionPanel
          step={slowStep}
          stage={stage}
          playing={isTraining}
          spotlight={spotlight}
          onSpotlight={setSpotlight}
          onNextStage={nextSlowStage}
        />
      )}

      {/* Left Column: Loss Chart */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
          <Activity size={20} color="var(--primary)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Real-Time Training Loss Curve</h3>
          <InfoTooltip
            title="Cross-Entropy Loss Curve"
            description="Measures model prediction uncertainty. Training loss is measured on the window just trained on; validation loss is measured on the held-out tail of the dataset the model never trains on."
            impact="If training loss keeps falling while validation loss rises, the model is memorizing (overfitting) rather than generalizing."
          />
        </div>
        <div ref={chartBoxRef} style={{ flex: 1, minHeight: '260px', width: '100%', position: 'relative' }}>
          <Line data={chartData} options={chartOptions} />
        </div>
      </div>

      {/* Right Column: Generation Playground */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Sparkles size={20} color="var(--accent-emerald)" />
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Generation Playground & Sampling</h2>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
              <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>Seed Prompt String</label>
              <InfoTooltip
                title="Seed Prompt String"
                description="The initial text context provided to the model. The model autoregressively appends predicted next tokens onto this prompt."
                impact="Sets the starting state for context window embedding and attention lookup."
              />
            </div>
            <input
              type="text"
              value={seedPrompt}
              onChange={(e) => setSeedPrompt(e.target.value)}
              style={{ width: '100%', fontSize: '0.9rem' }}
            />
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>Temperature (T)</label>
                <InfoTooltip
                  title="Sampling Temperature (T)"
                  description="Scales output logits before softmax. T < 0.2 turns into greedy argmax; T > 1.0 flattens probability distribution."
                  impact="Low temperature = highly deterministic & focused; High temperature = creative & random."
                />
              </div>
              <span className="font-mono" style={{ fontSize: '0.8rem', color: 'var(--accent-cyan)' }}>{temperature}</span>
            </div>
            <input
              type="range"
              min={0.1}
              max={2.0}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
            />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
              Lower T = more deterministic/greedy; Higher T = more creative/diverse.
            </span>
          </div>

          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>Max Generated Tokens</label>
                <InfoTooltip
                  title="Max Generated Tokens"
                  description="The maximum number of sequential token generation loops executed per click."
                  impact="Determines output text length before stopping."
                />
              </div>
              <span className="font-mono" style={{ fontSize: '0.8rem', color: 'var(--accent-emerald)' }}>{maxGenTokens}</span>
            </div>
            <input
              type="range"
              min={10}
              max={150}
              step={5}
              value={maxGenTokens}
              onChange={(e) => setMaxGenTokens(Number(e.target.value))}
            />
          </div>

          <button className="btn-primary" onClick={handleGenerate}>
            <Sparkles size={16} /> Generate Tokens
          </button>
        </div>

        {/* Output Text Window */}
        <div style={{ flex: 1, background: 'var(--surface-inset)', padding: '16px', borderRadius: '10px', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column' }}>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '8px' }}>Model Output Generation:</p>
          <div className="font-mono" style={{ flex: 1, minHeight: '120px', fontSize: '0.9rem', color: 'var(--text-main)', whiteSpace: 'pre-wrap' }}>
            {generatedText || <span style={{ color: 'var(--text-dim)' }}>Click "Generate Tokens" to sample text output from the model...</span>}
          </div>
        </div>
      </div>

      {/* Full-width: Benchmarks (seen vs held-out) */}
      <div style={{ gridColumn: '1 / -1' }}>
        <BenchmarkPanel
          model={model}
          tokenizer={tokenizer}
          selectedDataset={selectedDataset}
          trainText={split.trainText}
          stepCount={stepCount}
        />
      </div>
    </div>
  );
};
