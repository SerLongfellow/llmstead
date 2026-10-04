import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { Activity, FastForward, Grid3x3, Images, Pause, Play, RotateCcw, Target } from 'lucide-react';
import { CAPTION_DETAILS, comboName } from '../../engine/vl/captions';
import { MicroClip } from '../../engine/vl/clip';
import { ClipData, Pair, chanceExact, gridPairs as makeGridPairs } from '../../engine/vl/clipData';
import { EvalResult, evalSets, isEvalDue, runEval, trainClipChunk } from '../../engine/vl/clipTraining';
import type { FromClipWorker, ToClipWorker } from '../../engine/vl/clipWorker';
import { THEME, withAlpha } from '../../styles/theme';
import { InfoTooltip } from '../InfoTooltip';
import { ImageCanvas } from './ImageCanvas';
import { BATCH_SIZES, VisionSettings } from './visionSettings';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

/** On-page fallback (no Web Workers): steps per tick are time-boxed, with a gap for repainting */
const TICK_BUDGET_MS = 100;
const TICK_GAP_MS = 16;

function createWorker(): Worker | null {
  try {
    return new Worker(new URL('../../engine/vl/clipWorker.ts', import.meta.url), { type: 'module' });
  } catch {
    return null;
  }
}

interface VisionTrainViewProps {
  model: MicroClip;
  data: ClipData;
  settings: VisionSettings;
  onChange: (s: VisionSettings) => void;
  visible: boolean;
  isTraining: boolean;
  setIsTraining: (t: boolean) => void;
  weightsVersion: number;
  onWeightsChanged: () => void;
  onStartOver: () => void;
  onNavigateToSetup: () => void;
}

const pct = (v: number | null | undefined) => (v === null || v === undefined ? '–' : `${Math.round(v * 100)}%`);

export const VisionTrainView: React.FC<VisionTrainViewProps> = ({
  model, data, settings, onChange, visible, isTraining, setIsTraining,
  weightsVersion, onWeightsChanged, onStartOver, onNavigateToSetup,
}) => {
  const [workerBusy, setWorkerBusy] = useState(false);
  const [stepCount, setStepCount] = useState(0);
  const [labels, setLabels] = useState<string[]>([]);
  const [losses, setLosses] = useState<number[]>([]);
  const [scales, setScales] = useState<number[]>([]);
  const [evals, setEvals] = useState<EvalResult[]>([]);
  const [speed, setSpeed] = useState<number | null>(null);

  const stepRef = useRef(0);
  const latest = useRef({ model, data, settings });
  latest.current = { model, data, settings };
  const sets = useMemo(() => evalSets(data), [data]);

  const resetHistory = () => {
    stepRef.current = 0;
    setStepCount(0);
    setLabels([]);
    setLosses([]);
    setScales([]);
    setEvals([]);
    setSpeed(null);
  };

  const recordChunk = (startStep: number, steps: number, lossSum: number, scale: number, ms: number | null) => {
    stepRef.current = startStep + steps;
    setStepCount(stepRef.current);
    setLabels(l => [...l, `#${stepRef.current}`]);
    setLosses(h => [...h, Number((lossSum / steps).toFixed(4))]);
    setScales(h => [...h, Number(scale.toFixed(2))]);
    if (ms) setSpeed((steps * 1000) / ms);
  };

  // A new model (Setup change or Start over) means the old curves no longer apply
  const lastModel = useRef(model);
  useEffect(() => {
    if (model === lastModel.current) return;
    lastModel.current = model;
    runRef.current++; // late messages from the old run are ignored
    setWorkerBusy(false);
    setIsTraining(false);
    resetHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  /** Train on the page (Step button, or continuous training when there are no workers) */
  const trainOnPage = (maxSteps: number, budgetMs: number) => {
    const { model, data, settings } = latest.current;
    const startStep = stepRef.current;
    if (startStep === 0) setEvals([runEval(model, sets, data, 0)]);
    const t0 = performance.now();
    const r = trainClipChunk(model, data, { batchSize: settings.batchSize, learningRate: settings.learningRate, budgetMs, maxSteps });
    recordChunk(startStep, r.steps, r.lossSum, model.logitScale(), maxSteps === Infinity ? performance.now() - t0 : null);
    if (startStep > 0 && isEvalDue(startStep, stepRef.current)) {
      const result = runEval(model, sets, data, stepRef.current);
      setEvals(e => [...e, result]);
    }
    onWeightsChanged();
  };

  // ── Continuous training in a Web Worker (same protocol as the GPT's trainWorker) ──
  const workerRef = useRef<Worker | null | undefined>(undefined);
  const runRef = useRef(0);
  const runModel = useRef<MicroClip | null>(null);
  const lastProgress = useRef(0);
  const acceptProgress = useRef(true);
  const onWorkerMessage = useRef<(msg: FromClipWorker) => void>(() => {});
  onWorkerMessage.current = msg => {
    if (msg.run !== runRef.current) return;
    if (msg.type === 'progress') {
      if (!acceptProgress.current) return;
      const now = performance.now();
      recordChunk(msg.startStep, msg.steps, msg.lossSum, msg.scale, lastProgress.current ? now - lastProgress.current : null);
      lastProgress.current = now;
    } else if (msg.type === 'eval') {
      if (acceptProgress.current) setEvals(e => [...e.filter(x => x.step < msg.result.step), msg.result]);
    } else {
      runModel.current?.importState(msg.state);
      if (runModel.current) runModel.current.steps = msg.step;
      onWeightsChanged();
      if (msg.final) setWorkerBusy(false);
    }
  };
  const getWorker = () => {
    if (workerRef.current === undefined) {
      const w = createWorker();
      if (w) w.onmessage = (e: MessageEvent<FromClipWorker>) => onWorkerMessage.current(e.data);
      workerRef.current = w;
    }
    return workerRef.current;
  };
  const post = (msg: ToClipWorker, transfer: Transferable[] = []) => workerRef.current?.postMessage(msg, transfer);
  useEffect(
    () => () => {
      workerRef.current?.terminate();
      workerRef.current = undefined; // a remount (or hot reload) starts a fresh worker
    },
    []
  );

  useEffect(() => {
    if (!isTraining) return;
    const worker = getWorker();
    if (!worker) {
      let timer: number;
      const tick = () => {
        trainOnPage(Infinity, TICK_BUDGET_MS);
        timer = window.setTimeout(tick, TICK_GAP_MS);
      };
      timer = window.setTimeout(tick, 0);
      return () => clearTimeout(timer);
    }
    const { model, data, settings } = latest.current;
    const run = ++runRef.current;
    runModel.current = model;
    acceptProgress.current = true;
    lastProgress.current = 0;
    setWorkerBusy(true);
    const state = model.exportState();
    const buffers = [...Object.values(state.weights), ...Object.values(state.optimizer.m), ...Object.values(state.optimizer.v)].map(a => a.buffer);
    post(
      { type: 'start', run, config: model.config, state, data, batchSize: settings.batchSize, learningRate: settings.learningRate, step: stepRef.current },
      buffers
    );
    return () => post({ type: 'stop' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTraining]);

  // Mid-training settings go straight to the running worker
  useEffect(() => {
    if (workerBusy) post({ type: 'set', learningRate: settings.learningRate, batchSize: settings.batchSize, falseNegatives: settings.falseNegatives });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.learningRate, settings.batchSize, settings.falseNegatives]);

  // ── The live similarity grid: a fixed handful of pairs, re-scored whenever the weights change ──
  const gridPairs = useMemo(() => makeGridPairs(data), [data]);
  const grid = useMemo(
    () => (visible ? model.evaluateBatch(gridPairs.map(p => p.example)) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [visible, model, gridPairs, weightsVersion]
  );

  // Charts are created while the tab may be hidden (0×0), so resize them when it's shown
  const chartsRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!visible) return;
    const timer = window.setTimeout(() => {
      chartsRef.current?.querySelectorAll('canvas').forEach(c => ChartJS.getChart(c)?.resize());
    });
    return () => clearTimeout(timer);
  }, [visible]);

  const lossChart = {
    labels,
    datasets: [
      {
        label: 'Training loss',
        data: losses,
        borderColor: THEME.primary,
        backgroundColor: withAlpha(THEME.primary, 0.12),
        fill: true,
        tension: 0.2,
        pointRadius: 0,
        yAxisID: 'y',
      },
      {
        label: 'Temperature scale',
        data: scales,
        borderColor: THEME.purple,
        borderDash: [5, 4],
        tension: 0.2,
        pointRadius: 0,
        yAxisID: 'y1',
      },
    ],
  };
  const axis = { ticks: { color: THEME.textDim }, grid: { color: withAlpha(THEME.border, 0.6) } };
  const axisTitle = (text: string) => ({ display: true, text, color: THEME.textMuted, font: { family: 'Inter', size: 11 } });
  const legend = { labels: { color: THEME.textMuted, font: { family: 'Inter' }, boxWidth: 14 } };
  const lossOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false as const,
    plugins: { legend, tooltip: { mode: 'index' as const, intersect: false } },
    scales: {
      x: { ...axis, ticks: { ...axis.ticks, maxTicksLimit: 10 } },
      y: { ...axis, title: axisTitle('Loss (lower is better)') },
      y1: { position: 'right' as const, ticks: { color: THEME.purple }, grid: { display: false }, title: axisTitle('Scale') },
    },
  };

  const hasWhere = data.detail !== 'shape';
  const hasSize = data.detail === 'full';
  const evalLine = (label: string, values: (number | null)[], color: string, dashed = false) => ({
    label, data: values.map(v => (v === null ? null : Math.round(v * 100))), borderColor: color, backgroundColor: color,
    borderDash: dashed ? [6, 4] : undefined, tension: 0.2, pointRadius: 2, spanGaps: true,
  });
  const accChart = {
    labels: evals.map(e => `#${e.step}`),
    datasets: [
      evalLine('Colour', evals.map(e => e.seen.colour), THEME.rose),
      evalLine('Shape', evals.map(e => e.seen.shape), THEME.emerald),
      ...(hasWhere ? [evalLine('Where', evals.map(e => e.seen.where), THEME.cyan)] : []),
      ...(hasSize ? [evalLine('Size', evals.map(e => e.seen.size), THEME.purple)] : []),
      evalLine('Whole caption right', evals.map(e => e.seen.exact), THEME.textMain),
      ...(data.heldOut.length ? [evalLine('Whole caption, never-seen combos', evals.map(e => e.heldOut?.exact ?? null), THEME.amber, true)] : []),
    ],
  };
  const accOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false as const,
    plugins: { legend, tooltip: { mode: 'index' as const, intersect: false } },
    scales: {
      x: { ...axis, ticks: { ...axis.ticks, maxTicksLimit: 10 } },
      y: { ...axis, min: 0, max: 100, title: axisTitle('% of test pictures') },
    },
  };

  const lastEval = evals.length ? evals[evals.length - 1] : null;
  const firstEval = evals.length ? evals[0] : null;
  const detailLabel = CAPTION_DETAILS.find(d => d.id === data.detail)!;
  const set = (patch: Partial<VisionSettings>) => onChange({ ...settings, ...patch });

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 480px), 1fr))', gap: 20 }} ref={chartsRef}>
      {/* Controls */}
      <div className="glass-panel" style={{ padding: '16px 24px', gridColumn: '1 / -1', display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Images size={20} color="var(--accent-purple)" />
            <div>
              <p style={{ fontSize: '0.95rem', fontWeight: 700 }}>Training on: endless random pictures, captioned "{detailLabel.example}"</p>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                {data.heldOut.length ? <>Never shown: {data.heldOut.map(comboName).join(' and ')} · </> : null}
                {model.getParameterCount().toLocaleString()} parameters · not saved when you leave the page
              </p>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            {speed !== null && isTraining && <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>{speed.toFixed(0)} steps/s</span>}
            <span className="badge badge-primary font-mono">Step: #{stepCount}</span>
            <button className="btn-secondary" onClick={onNavigateToSetup} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
              Change in Setup
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap', paddingTop: 14, borderTop: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <button className={isTraining ? 'btn-secondary' : 'btn-primary'} onClick={() => setIsTraining(!isTraining)} disabled={!isTraining && workerBusy}>
              {isTraining ? <Pause size={16} /> : <Play size={16} />}
              {isTraining ? 'Pause Training' : workerBusy ? 'Pausing…' : stepCount ? 'Resume Training' : 'Start Training'}
            </button>
            <button className="btn-secondary" onClick={() => trainOnPage(1, Infinity)} disabled={workerBusy} title="Exactly one training step (one batch)">
              <FastForward size={16} /> Step
            </button>
            <button
              className="btn-secondary"
              onClick={() => {
                acceptProgress.current = false;
                setIsTraining(false);
                onStartOver();
              }}
              title="Fresh random weights with the same settings, and clear the charts"
            >
              <RotateCcw size={16} /> Start over
            </button>
          </div>

          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              <label style={{ fontSize: '0.8rem', fontWeight: 600 }}>Batch size</label>
              <InfoTooltip
                title="Batch size"
                description="How many picture + caption pairs are compared with each other in one step. Every picture is scored against every caption in the batch, so the batch is also the set of wrong answers the model learns to reject."
                impact="Bigger batches give each pair more competition, so the model must notice finer differences; each step also takes longer (the cost grows with the batch)."
              />
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {BATCH_SIZES.map(b => (
                <button key={b} className={settings.batchSize === b ? 'btn-primary' : 'btn-secondary'} style={{ padding: '4px 10px', fontSize: '0.8rem' }} onClick={() => set({ batchSize: b })}>
                  {b}
                </button>
              ))}
            </div>
          </div>

          <div style={{ flex: '1 1 180px', minWidth: 160 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <label style={{ fontSize: '0.8rem', fontWeight: 600 }}>Learning rate</label>
                <InfoTooltip
                  title="Learning rate"
                  description="How big a step each weight takes on every update (AdamW). Safe to change while training: the weights are kept."
                  impact="0.001 works well here. Much higher and the loss jumps around; lower and learning crawls."
                />
              </div>
              <span className="font-mono" style={{ fontSize: '0.8rem', color: 'var(--accent-rose)' }}>{settings.learningRate}</span>
            </div>
            <input type="range" min={0.0005} max={0.01} step={0.0005} value={settings.learningRate} onChange={e => set({ learningRate: Number(e.target.value) })} />
          </div>

          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              <label style={{ fontSize: '0.8rem', fontWeight: 600 }}>Two pairs, same caption</label>
              <InfoTooltip
                title="False negatives"
                description="With only 16 colour + shape pairs, a batch often holds two pictures with the same caption. Plain CLIP still insists that each picture matches only its own caption, so it is penalized for matching the other one, which is just as right."
                impact="'Both count' treats every caption that truly describes the picture as a right answer. It matters most with short captions (colour + shape) and big batches, where repeats are common."
              />
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {([['ignore', 'Only its own (plain CLIP)'], ['multi-positive', 'Both count']] as const).map(([id, label]) => (
                <button key={id} className={settings.falseNegatives === id ? 'btn-primary' : 'btn-secondary'} style={{ padding: '4px 10px', fontSize: '0.8rem' }} onClick={() => set({ falseNegatives: id })}>
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Similarity grid */}
      <div className="glass-panel" style={{ padding: 24, gridColumn: '1 / -1' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
          <Grid3x3 size={20} color="var(--accent-cyan)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Every caption against every picture</h3>
          <InfoTooltip
            title="The similarity grid"
            description="Each cell is the cosine similarity between one caption's vector (from the text tower) and one picture's vector (from the image tower), from −1 to 1. Training pushes the right matches (outlined) up and every other cell down."
            impact="This is exactly what one training step looks at, for 8 fixed pairs. Brighter = more similar; the ring marks the picture each caption currently likes best."
          />
        </div>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 14 }}>
          Before training the grid is noise. As the model learns, the right matches (dashed outline) light up along the diagonal.
          Neighbouring pictures differ in one thing (shape, colour or place), so a caption that still picks the wrong one shows
          which fact the model hasn't learned yet.
          {grid && <> Right now, <b style={{ color: 'var(--text-main)' }}>{pct(grid.textToImage)}</b> of these captions pick a right picture.</>}
        </p>
        {grid && <SimilarityGrid pairs={gridPairs} similarity={grid.similarity} />}
      </div>

      {/* Loss chart */}
      <div className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
          <Activity size={20} color="var(--primary)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Contrastive loss</h3>
          <InfoTooltip
            title="Contrastive loss"
            description="For each picture: how surprised the model is by its own caption among all the captions in the batch (cross-entropy), and the same for each caption among the pictures; the loss is the average of the two. The dashed line is the learned temperature: how sharply similarities are turned into probabilities."
            impact={`A model that guesses at random scores ln(batch size) ≈ ${Math.log(settings.batchSize).toFixed(2)}. Low loss only means it can tell this batch apart, not that it knows every fact in the captions (see the test scores).`}
          />
        </div>
        <div style={{ flex: 1, minHeight: 260, position: 'relative' }}>
          <Line data={lossChart} options={lossOptions} />
        </div>
      </div>

      {/* Zero-shot scores */}
      <div className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
          <Target size={20} color="var(--accent-emerald)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Test: pick the right caption</h3>
          <InfoTooltip
            title="Zero-shot classification"
            description={`Every 250 steps, each of ${sets.seen.length + sets.heldOut.length} fixed test pictures is compared with every caption it could possibly have, and the best-scoring caption is its answer. Nothing extra is trained for this: it's the same similarity score used in training. That's what "zero-shot" means.`}
            impact="Each line is how often the chosen caption gets that fact right. The dashed line uses only the colour + shape pairs training never showed."
          />
        </div>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 12 }}>
          Guessing would get the whole caption right {pct(chanceExact(data.detail))} of the time.
          {lastEval && firstEval && lastEval.step > 0 && (
            <>
              {' '}At step #{lastEval.step}: whole caption {pct(lastEval.seen.exact)}
              {lastEval.heldOut && <>, never-seen combos {pct(lastEval.heldOut.exact)}</>} (untrained: {pct(firstEval.seen.exact)}).
            </>
          )}
        </p>
        <div style={{ flex: 1, minHeight: 240, position: 'relative' }}>
          {evals.length ? (
            <Line data={accChart} options={accOptions} />
          ) : (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', color: 'var(--text-dim)', fontSize: '0.85rem', border: '1px dashed var(--border-color)', borderRadius: 10, padding: 20 }}>
              The first scores (the untrained baseline) appear as soon as training starts.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

/** Captions down the side, pictures along the top, cosine similarity in each cell */
const SimilarityGrid: React.FC<{ pairs: Pair[]; similarity: number[][] }> = ({ pairs, similarity }) => {
  // similarity[i][j] = picture i · caption j; rows here are captions, so read it transposed
  const n = pairs.length;
  const cell = 46;
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'separate', borderSpacing: 3, margin: '0 auto' }}>
        <thead>
          <tr>
            <th />
            {pairs.map((p, i) => (
              <th key={i} style={{ padding: 0 }}>
                <div style={{ border: '1px solid var(--border-strong)', borderRadius: 7, width: cell, margin: '0 auto' }}>
                  <ImageCanvas image={p.image} size={cell - 2} />
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {pairs.map((p, j) => {
            let best = 0;
            for (let i = 1; i < n; i++) if (similarity[i][j] > similarity[best][j]) best = i;
            return (
              <tr key={j}>
                <td className="font-mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)', paddingRight: 8, textAlign: 'right', whiteSpace: 'nowrap' }}>
                  {p.caption}
                </td>
                {pairs.map((q, i) => {
                  const v = similarity[i][j];
                  const right = q.example.key === p.example.key;
                  const t = Math.max(0, Math.min(1, (v + 1) / 2));
                  return (
                    <td
                      key={i}
                      title={`"${p.caption}" vs picture ${i + 1}: ${v.toFixed(2)}${right ? ' (a right match)' : ''}`}
                      style={{
                        width: cell, height: 30, textAlign: 'center', fontSize: '0.7rem', fontFamily: 'var(--font-mono)', borderRadius: 5,
                        color: t > 0.6 ? '#ffffff' : 'var(--text-muted)',
                        background: `color-mix(in srgb, var(--primary) ${Math.round(t * t * 100)}%, var(--surface-inset))`,
                        border: right ? '2px dashed var(--accent-emerald)' : '2px solid transparent',
                        boxShadow: i === best ? 'inset 0 0 0 2px var(--accent-amber)' : undefined,
                      }}
                    >
                      {v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textAlign: 'center', marginTop: 8 }}>
        <span style={{ border: '2px dashed var(--accent-emerald)', borderRadius: 3, padding: '0 4px' }}>dashed</span> = a right match ·{' '}
        <span style={{ boxShadow: 'inset 0 0 0 2px var(--accent-amber)', padding: '0 4px', borderRadius: 3 }}>ring</span> = the picture this caption scores highest
      </p>
    </div>
  );
};
