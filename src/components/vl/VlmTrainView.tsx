import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { Activity, Eye, FastForward, Info, Pause, Play, RotateCcw } from 'lucide-react';
import { MicroVlm, VlmPhase } from '../../engine/vl/vlm';
import { VQA_TASKS, VqaTask } from '../../engine/vl/vlmData';
import { EVAL_EVERY, VlmDataSettings, VlmEvalResult, isEvalDue, runVlmEval, tasksForPhase, trainVlmChunk, vlmEvalSet } from '../../engine/vl/vlmTraining';
import type { FromVlmWorker, ToVlmWorker } from '../../engine/vl/vlmWorker';
import { THEME, withAlpha } from '../../styles/theme';
import { InfoTooltip } from '../InfoTooltip';
import { AlignExplainer } from './AlignExplainer';
import { VlmPipeline } from './VlmPipeline';
import { PHASE_LR, VlmSettings } from './visionSettings';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const TICK_BUDGET_MS = 100;
const TICK_GAP_MS = 16;
const BATCH_SIZES = [4, 8, 16];

function createWorker(): Worker | null {
  try {
    return new Worker(new URL('../../engine/vl/vlmWorker.ts', import.meta.url), { type: 'module' });
  } catch {
    return null;
  }
}

const PHASES: { phase: VlmPhase; title: string; what: string; why: string }[] = [
  {
    phase: 0,
    title: 'Text only',
    what: 'The image slots are left empty. Only the language model learns, from the questions and answers alone.',
    why: 'It learns the words, the question formats and what an answer looks like, but it can only guess the facts. LLaVA starts from a language model already trained on huge amounts of text; this phase is our tiny stand-in.',
  },
  {
    phase: 1,
    title: 'Align the projector',
    what: 'The language model is frozen. Only the projector learns, and only from "describe the picture".',
    why: 'The projector has to find image tokens that the frozen language model already reads as "red", "circle" or "top left". It learns to describe, but not yet to answer questions.',
  },
  {
    phase: 2,
    title: 'Instruction-tune',
    what: 'The projector and the language model learn together, on every kind of question you chose.',
    why: 'Now the language model also learns to use the image tokens to answer each kind of question, and the projector keeps adjusting with it.',
  },
];

interface VlmTrainViewProps {
  vlm: MicroVlm;
  data: VlmDataSettings;
  settings: VlmSettings;
  onChange: (s: VlmSettings) => void;
  visible: boolean;
  isTraining: boolean;
  setIsTraining: (t: boolean) => void;
  onWeightsChanged: () => void;
  onStartOver: () => void;
  onNavigateToSetup: () => void;
}

const pct = (v: number | null | undefined) => (v === null || v === undefined ? '–' : `${Math.round(v * 100)}%`);
const PHASE_COLORS = [THEME.cyan, THEME.purple, THEME.primary];

export const VlmTrainView: React.FC<VlmTrainViewProps> = ({
  vlm, data, settings, onChange, visible, isTraining, setIsTraining, onWeightsChanged, onStartOver, onNavigateToSetup,
}) => {
  const [workerBusy, setWorkerBusy] = useState(false);
  const [stepCount, setStepCount] = useState(0);
  const [labels, setLabels] = useState<string[]>([]);
  const [losses, setLosses] = useState<{ loss: number; phase: VlmPhase }[]>([]);
  const [evals, setEvals] = useState<VlmEvalResult[]>([]);
  const [phaseSteps, setPhaseSteps] = useState<Record<VlmPhase, number>>({ 0: 0, 1: 0, 2: 0 });
  const [speed, setSpeed] = useState<number | null>(null);
  const [chartTask, setChartTask] = useState<VqaTask>('colour');

  const stepRef = useRef(0);
  const latest = useRef({ vlm, data, settings });
  latest.current = { vlm, data, settings };
  const set = useMemo(() => vlmEvalSet(data.patchSize, data.heldOut), [data.patchSize, data.heldOut]);
  const busy = isTraining || workerBusy;

  const resetHistory = () => {
    stepRef.current = 0;
    setStepCount(0);
    setLabels([]);
    setLosses([]);
    setEvals([]);
    setPhaseSteps({ 0: 0, 1: 0, 2: 0 });
    setSpeed(null);
  };

  const recordChunk = (startStep: number, steps: number, lossSum: number, phase: VlmPhase, ms: number | null) => {
    stepRef.current = startStep + steps;
    setStepCount(stepRef.current);
    setLabels(l => [...l, `#${stepRef.current}`]);
    setLosses(h => [...h, { loss: Number((lossSum / steps).toFixed(4)), phase }]);
    setPhaseSteps(p => ({ ...p, [phase]: p[phase] + steps }));
    if (ms) setSpeed((steps * 1000) / ms);
  };
  const addEval = (r: VlmEvalResult) => setEvals(e => [...e.filter(x => x.step < r.step || x.phase !== r.phase), r]);

  // A new VLM (Setup change, a new CLIP copy, or Start over): the old curves no longer apply
  const lastModel = useRef(vlm);
  useEffect(() => {
    if (vlm === lastModel.current) return;
    lastModel.current = vlm;
    runRef.current++;
    setWorkerBusy(false);
    setIsTraining(false);
    resetHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vlm]);

  const phaseRef = useRef(phaseSteps);
  phaseRef.current = phaseSteps;

  /** Train on the page (Step button, or continuous training when there are no workers) */
  const trainOnPage = (maxSteps: number, budgetMs: number) => {
    const { vlm, data, settings } = latest.current;
    const phase = settings.phase;
    const startStep = stepRef.current;
    if (phaseRef.current[phase] === 0) addEval(runVlmEval(vlm, set, startStep, phase));
    const t0 = performance.now();
    const r = trainVlmChunk(vlm, data, { phase, batchSize: settings.batchSize, learningRate: settings.learningRate, budgetMs, maxSteps });
    recordChunk(startStep, r.steps, r.lossSum, phase, maxSteps === Infinity ? performance.now() - t0 : null);
    if (isEvalDue(startStep, stepRef.current) && startStep > 0) addEval(runVlmEval(vlm, set, stepRef.current, phase));
    onWeightsChanged();
  };

  // ── Continuous training in a Web Worker ──
  const workerRef = useRef<Worker | null | undefined>(undefined);
  const runRef = useRef(0);
  const runModel = useRef<MicroVlm | null>(null);
  const lastProgress = useRef(0);
  const runPhase = useRef<VlmPhase>(0);
  const onWorkerMessage = useRef<(msg: FromVlmWorker) => void>(() => {});
  onWorkerMessage.current = msg => {
    if (msg.run !== runRef.current) return;
    if (msg.type === 'progress') {
      const now = performance.now();
      recordChunk(msg.startStep, msg.steps, msg.lossSum, runPhase.current, lastProgress.current ? now - lastProgress.current : null);
      lastProgress.current = now;
    } else if (msg.type === 'eval') {
      addEval(msg.result);
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
      if (w) w.onmessage = (e: MessageEvent<FromVlmWorker>) => onWorkerMessage.current(e.data);
      workerRef.current = w;
    }
    return workerRef.current;
  };
  const post = (msg: ToVlmWorker, transfer: Transferable[] = []) => workerRef.current?.postMessage(msg, transfer);
  useEffect(
    () => () => {
      workerRef.current?.terminate();
      workerRef.current = undefined;
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
    const { vlm, data, settings } = latest.current;
    const run = ++runRef.current;
    runModel.current = vlm;
    runPhase.current = settings.phase;
    lastProgress.current = 0;
    setWorkerBusy(true);
    const state = vlm.exportState();
    const visionState = vlm.vision.exportState();
    const buffers = [
      ...Object.values(state.weights), ...Object.values(state.optimizer.m), ...Object.values(state.optimizer.v),
      ...Object.values(visionState.weights), ...Object.values(visionState.optimizer.m), ...Object.values(visionState.optimizer.v),
    ].map(a => a.buffer);
    post(
      {
        type: 'start', run, visionConfig: vlm.vision.config, visionState, config: vlm.config, state, data,
        phase: settings.phase, batchSize: settings.batchSize, learningRate: settings.learningRate,
        step: stepRef.current, phaseStep: phaseRef.current[settings.phase],
      },
      buffers
    );
    return () => post({ type: 'stop' });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTraining]);

  useEffect(() => {
    if (workerBusy) post({ type: 'set', learningRate: settings.learningRate, batchSize: settings.batchSize });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.learningRate, settings.batchSize]);

  const choosePhase = (phase: VlmPhase) => onChange({ ...settings, phase, learningRate: PHASE_LR[phase] });

  // Charts are created while the tab may be hidden (0×0), so resize them when it's shown
  const chartsRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!visible) return;
    const timer = window.setTimeout(() => chartsRef.current?.querySelectorAll('canvas').forEach(c => ChartJS.getChart(c)?.resize()));
    return () => clearTimeout(timer);
  }, [visible]);

  const axis = { ticks: { color: THEME.textDim }, grid: { color: withAlpha(THEME.border, 0.6) } };
  const axisTitle = (text: string) => ({ display: true, text, color: THEME.textMuted, font: { family: 'Inter', size: 11 } });
  const legend = { labels: { color: THEME.textMuted, font: { family: 'Inter' }, boxWidth: 14 } };
  const baseOptions = {
    responsive: true, maintainAspectRatio: false, animation: false as const,
    plugins: { legend, tooltip: { mode: 'index' as const, intersect: false } },
  };

  const lossChart = {
    labels,
    datasets: PHASES.map(p => ({
      label: `Phase ${p.phase}: ${p.title}`,
      data: losses.map(l => (l.phase === p.phase ? l.loss : null)),
      borderColor: PHASE_COLORS[p.phase],
      backgroundColor: PHASE_COLORS[p.phase],
      tension: 0.2, pointRadius: 0, spanGaps: false,
    })),
  };
  const lossOptions = {
    ...baseOptions,
    scales: { x: { ...axis, ticks: { ...axis.ticks, maxTicksLimit: 10 } }, y: { ...axis, title: axisTitle('Loss on the answer words') } },
  };

  const line = (label: string, values: (number | null)[], color: string, dashed = false) => ({
    label, data: values.map(v => (v === null ? null : Math.round(v * 100))), borderColor: color, backgroundColor: color,
    borderDash: dashed ? [6, 4] : undefined, tension: 0.2, pointRadius: 2, spanGaps: true,
  });
  const lookChart = {
    labels: evals.map(e => `#${e.step} (phase ${e.phase})`),
    datasets: [
      line('With the picture', evals.map(e => e.tasks[chartTask].seen), THEME.emerald),
      line('With a blank picture', evals.map(e => e.tasks[chartTask].seenBlank), THEME.textMuted, true),
      ...(data.heldOut.length ? [line('With the picture, never-seen combos', evals.map(e => e.tasks[chartTask].heldOut), THEME.amber)] : []),
    ],
  };
  const lookOptions = {
    ...baseOptions,
    scales: { x: { ...axis, ticks: { ...axis.ticks, maxTicksLimit: 8 } }, y: { ...axis, min: 0, max: 100, title: axisTitle('% answered exactly right') } },
  };
  const lastEval = evals.length ? evals[evals.length - 1] : null;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 480px), 1fr))', gap: 20 }} ref={chartsRef}>
      {/* Phases */}
      <div className="glass-panel" style={{ padding: '18px 24px', gridColumn: '1 / -1', display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Training in phases, like LLaVA</h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            {speed !== null && isTraining && <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>{speed.toFixed(0)} steps/s</span>}
            <span className="badge badge-primary font-mono">Step: #{stepCount}</span>
            <button className="btn-secondary" onClick={onNavigateToSetup} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
              Change in Setup
            </button>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12, alignItems: 'start' }}>
          {PHASES.map(p => {
            const current = settings.phase === p.phase;
            return (
              <div
                key={p.phase}
                style={{
                  padding: 14, borderRadius: 10, display: 'flex', flexDirection: 'column', gap: 8,
                  background: current ? 'var(--primary-soft)' : 'var(--surface-inset)',
                  border: `1px solid ${current ? 'var(--primary)' : 'var(--border-color)'}`,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span className="font-mono" style={{ width: 22, height: 22, borderRadius: 11, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', fontWeight: 700, background: PHASE_COLORS[p.phase], color: '#0d1117' }}>
                    {p.phase}
                  </span>
                  <span style={{ fontWeight: 700 }}>{p.title}</span>
                  <span style={{ marginLeft: 'auto', fontSize: '0.7rem', color: 'var(--text-dim)' }}>{phaseSteps[p.phase].toLocaleString()} steps</span>
                </div>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-main)', lineHeight: 1.5 }}>{p.what}</p>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{p.why}</p>
                {p.phase === 1 && <AlignExplainer phase1Evals={evals.filter(e => e.phase === 1)} />}
                <p style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
                  Questions: {tasksForPhase(p.phase, settings.tasks).map(t => VQA_TASKS.find(x => x.id === t)!.label.toLowerCase()).join(', ')}
                </p>
                <div style={{ marginTop: 'auto' }}>
                  {current ? (
                    <button className={isTraining ? 'btn-secondary' : 'btn-primary'} onClick={() => setIsTraining(!isTraining)} disabled={!isTraining && workerBusy} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
                      {isTraining ? <Pause size={14} /> : <Play size={14} />}
                      {isTraining ? 'Pause' : workerBusy ? 'Pausing…' : phaseSteps[p.phase] ? 'Resume this phase' : 'Start this phase'}
                    </button>
                  ) : (
                    <button
                      className="btn-secondary"
                      onClick={() => choosePhase(p.phase)}
                      disabled={busy}
                      title={busy ? 'Pause training first' : undefined}
                      style={{ padding: '6px 12px', fontSize: '0.8rem' }}
                    >
                      Switch to phase {p.phase}
                    </button>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        <VlmPipeline phase={settings.phase} />


        <div style={{ display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap', paddingTop: 12, borderTop: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <button className="btn-secondary" onClick={() => trainOnPage(1, Infinity)} disabled={busy} title="Exactly one training step (one batch) of the current phase">
              <FastForward size={16} /> Step
            </button>
            <button className="btn-secondary" onClick={() => { setIsTraining(false); onStartOver(); }} title="Fresh random projector and language model (same eyes), and clear the charts">
              <RotateCcw size={16} /> Start over
            </button>
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              <label style={{ fontSize: '0.8rem', fontWeight: 600 }}>Batch size</label>
              <InfoTooltip
                title="Batch size"
                description="How many picture + question examples each step averages over."
                impact="Bigger batches give smoother steps, and each step takes longer."
              />
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {BATCH_SIZES.map(b => (
                <button key={b} className={settings.batchSize === b ? 'btn-primary' : 'btn-secondary'} style={{ padding: '4px 10px', fontSize: '0.8rem' }} onClick={() => onChange({ ...settings, batchSize: b })}>
                  {b}
                </button>
              ))}
            </div>
          </div>
          <div style={{ flex: '1 1 200px', minWidth: 160 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <label style={{ fontSize: '0.8rem', fontWeight: 600 }}>Learning rate</label>
                <InfoTooltip
                  title="Learning rate"
                  description="How big a step the learning parts take on every update (AdamW). Switching phase resets it to that phase's default."
                  impact="Phase 1 starts higher (0.003): the projector is small and learns slowly at 0.001."
                />
              </div>
              <span className="font-mono" style={{ fontSize: '0.8rem', color: 'var(--accent-rose)' }}>{settings.learningRate}</span>
            </div>
            <input type="range" min={0.0005} max={0.01} step={0.0005} value={settings.learningRate} onChange={e => onChange({ ...settings, learningRate: Number(e.target.value) })} />
          </div>
        </div>

        <div style={{ display: 'flex', gap: 10, padding: '10px 14px', borderRadius: 8, background: 'var(--surface-inset)', border: '1px solid var(--border-color)' }}>
          <Info size={16} color="var(--primary)" style={{ flexShrink: 0, marginTop: 2 }} />
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>
            <b style={{ color: 'var(--text-main)' }}>Honest note:</b> at this size, skipping straight to phase 2 works about as well
            (we measured it: try Start over, then switch straight to phase 2). This tiny language model knows nothing worth
            protecting. LLaVA's phases exist for a large language model trained on vast amounts of text: aligning the projector first
            keeps random image tokens from damaging what it already knows.
          </p>
        </div>
      </div>

      {/* Loss */}
      <div className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
          <Activity size={20} color="var(--primary)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Loss, by phase</h3>
          <InfoTooltip
            title="Loss on the answer words"
            description="Cross-entropy of each next word of the answer, given the image tokens, the question and the answer so far. The question's own words don't count."
            impact="Phases train on different questions (phase 1 only describes), so the levels aren't directly comparable across phases."
          />
        </div>
        <div style={{ flex: 1, minHeight: 260, position: 'relative' }}>
          <Line data={lossChart} options={lossOptions} />
        </div>
      </div>

      {/* Does it look? */}
      <div className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Eye size={20} color="var(--accent-emerald)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Does it look at the picture?</h3>
          <InfoTooltip
            title="Picture vs blank picture"
            description={`Every ${EVAL_EVERY} steps (and at the start of each phase), fixed test pictures are each asked every kind of question: once with the picture, once with a blank picture (just the background). An answer only counts if every word is right.`}
            impact="If the two lines match, the model answers from habit (whatever is usually true), not from the picture. The gap between them is how much it actually looks."
          />
        </div>
        <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
          {VQA_TASKS.map(t => (
            <button key={t.id} className={chartTask === t.id ? 'btn-primary' : 'btn-secondary'} style={{ padding: '3px 10px', fontSize: '0.75rem' }} onClick={() => setChartTask(t.id)}>
              {t.label}
            </button>
          ))}
        </div>
        <div style={{ flex: 1, minHeight: 220, position: 'relative' }}>
          {evals.length ? (
            <Line data={lookChart} options={lookOptions} />
          ) : (
            <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', textAlign: 'center', color: 'var(--text-dim)', fontSize: '0.85rem', border: '1px dashed var(--border-color)', borderRadius: 10, padding: 20 }}>
              The first scores (before any training) appear as soon as a phase starts.
            </div>
          )}
        </div>
      </div>

      {/* Latest scores */}
      {lastEval && (
        <div className="glass-panel" style={{ padding: 24, gridColumn: '1 / -1' }}>
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: 12 }}>
            Latest test, step #{lastEval.step} (phase {lastEval.phase})
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', fontSize: '0.8rem', minWidth: 520 }}>
              <thead>
                <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
                  <th style={{ padding: '6px 10px' }}>Question</th>
                  <th style={{ padding: '6px 10px' }}>With the picture</th>
                  <th style={{ padding: '6px 10px' }}>Blank picture</th>
                  {data.heldOut.length > 0 && <th style={{ padding: '6px 10px' }}>Never-seen combos</th>}
                </tr>
              </thead>
              <tbody>
                {VQA_TASKS.map(t => {
                  const s = lastEval.tasks[t.id];
                  const bar = (v: number | null, color: string) => (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ width: 90, height: 8, borderRadius: 4, background: 'var(--surface-inset)', overflow: 'hidden' }}>
                        <div style={{ width: `${(v ?? 0) * 100}%`, height: '100%', background: color }} />
                      </div>
                      <span className="font-mono" style={{ fontSize: '0.75rem' }}>{pct(v)}</span>
                    </div>
                  );
                  return (
                    <tr key={t.id} style={{ borderTop: '1px solid var(--border-color)', opacity: settings.tasks.includes(t.id) ? 1 : 0.55 }}>
                      <td className="font-mono" style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>{t.example.split(/ \? | : /)[0]}</td>
                      <td style={{ padding: '6px 10px' }}>{bar(s.seen, 'var(--accent-emerald)')}</td>
                      <td style={{ padding: '6px 10px' }}>{bar(s.seenBlank, 'var(--text-dim)')}</td>
                      {data.heldOut.length > 0 && <td style={{ padding: '6px 10px' }}>{bar(s.heldOut, 'var(--accent-amber)')}</td>}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: 10 }}>
            Faded rows are question kinds not chosen for training. Yes/no questions score about 50% by guessing; a description only
            counts if every word is right.
          </p>
        </div>
      )}
    </div>
  );
};
