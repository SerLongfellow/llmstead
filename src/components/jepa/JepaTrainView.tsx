import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Activity, FastForward, Microscope, Pause, Play, RotateCcw, Shield } from 'lucide-react';
import {
  CategoryScale, Chart as ChartJS, Legend, LinearScale, LineElement, LogarithmicScale, PointElement, Tooltip,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { MicroJepa, sampleMask } from '../../engine/jepa/jepa';
import {
  CollapseStats, collapseStats, imageFeatures, pixelFeatures, runFewLabelProbes, runProbes,
} from '../../engine/jepa/probes';
import { ShapeImage, patchify, randomShapeImage } from '../../engine/jepa/shapes';
import { Matrix } from '../../engine/tensor';
import { THEME, withAlpha } from '../../styles/theme';
import { InfoTooltip } from '../InfoTooltip';
import { ANTI_COLLAPSE, GRID, IMAGE_SIZE, JepaSettings, PATCH_SIZE, RECIPES, shapeSizes } from './jepaSettings';

ChartJS.register(CategoryScale, LinearScale, LogarithmicScale, PointElement, LineElement, Tooltip, Legend);

/** Each tick trains for up to this long, then yields so the page can repaint (as on the text side) */
const TICK_BUDGET_MS = 100;
const TICK_GAP_MS = 16;
/** One loss-chart point per this many steps (the mean over them) */
const POINT_EVERY = 10;
/** Collapse metrics are measured every this many steps, on COLLAPSE_IMAGES fixed images */
const COLLAPSE_EVERY = 100;
const COLLAPSE_IMAGES = 128;

type Sets = { train: ShapeImage[]; test: ShapeImage[] };

interface ProbeRow {
  shape5: number;
  shape20: number;
  shapeAll: number;
  colour5: number;
  pos20: number;
  posAll: number;
}

type Pool = 'mean' | 'concat';

interface CollapsePoint extends CollapseStats {
  step: number;
}

interface JepaTrainViewProps {
  model: MicroJepa;
  untrained: MicroJepa;
  settings: JepaSettings;
  onChange: (next: JepaSettings) => void;
  probeSets: Sets;
  visible: boolean;
  isTraining: boolean;
  setIsTraining: (t: boolean) => void;
  onStartOver: () => void;
  onNavigateToSetup: () => void;
}

const nextFrame = () => new Promise(resolve => setTimeout(resolve, 0));

/** Few-label (5 and 20 per shape class) and full probes on one set of features */
function probeRow(trainX: Matrix, testX: Matrix, sets: Sets): ProbeRow {
  const few5 = runFewLabelProbes(trainX, sets.train, testX, sets.test, 5);
  const few20 = runFewLabelProbes(trainX, sets.train, testX, sets.test, 20);
  const all = runProbes(trainX, sets.train, testX, sets.test);
  return {
    shape5: few5.shape, shape20: few20.shape, shapeAll: all.shape,
    colour5: few5.colour, pos20: few20.position, posAll: all.position,
  };
}

/** Below this spread the embeddings count as collapsed */
const COLLAPSED_SPREAD = 0.02;

/**
 * Plain-language reading of the collapse metrics. The rank threshold comes from the experiments:
 * the plain recipe settles around 2.5 directions, the anti-collapse term keeps 13+, and an
 * untrained encoder sits around 4–7.
 */
function collapseVerdict(s: CollapseStats, dModel: number): { label: string; color: string; text: string } {
  if (s.spread < COLLAPSED_SPREAD) {
    return { label: 'Collapsed', color: 'var(--accent-rose)', text: 'Every image gets almost the same embedding. The loss can be tiny and mean nothing.' };
  }
  if (s.effectiveRank < 3) {
    return {
      label: 'Partly collapsed',
      color: 'var(--accent-amber)',
      text: `The embeddings differ between images, but use only about ${s.effectiveRank.toFixed(1)} of ${dModel} possible directions.`,
    };
  }
  return { label: 'Spread out', color: 'var(--accent-emerald)', text: `The embeddings use about ${s.effectiveRank.toFixed(1)} of ${dModel} directions.` };
}

const pct = (v: number) => `${Math.round(v * 100)}%`;
const r2 = (v: number) => v.toFixed(2);

export const JepaTrainView: React.FC<JepaTrainViewProps> = ({
  model, untrained, settings, onChange, probeSets, visible, isTraining, setIsTraining, onStartOver, onNavigateToSetup,
}) => {
  const [stepCount, setStepCount] = useState(0);
  const [lossPoints, setLossPoints] = useState<{ step: number; loss: number; reg: number }[]>([]);
  const [collapse, setCollapse] = useState<CollapsePoint[]>([]);
  const [msPerStep, setMsPerStep] = useState<number | null>(null);
  const [events, setEvents] = useState<{ step: number; text: string }[]>([]);

  const collapseImages = useMemo(() => probeSets.train.slice(0, COLLAPSE_IMAGES), [probeSets]);
  const untrainedStats = useMemo(() => collapseStats(imageFeatures(untrained, collapseImages)), [untrained, collapseImages]);

  // Refs so the timer callback always sees the latest values
  const latest = useRef({ model, settings, collapseImages });
  latest.current = { model, settings, collapseImages };
  const pending = useRef({ lossSum: 0, regSum: 0, count: 0, ms: 0 });

  const measureCollapse = () => {
    const { model, collapseImages } = latest.current;
    const stats = collapseStats(imageFeatures(model, collapseImages));
    setCollapse(c => [...c, { step: model.steps, ...stats }]);
  };

  // A new model (new architecture, or Start over) starts every chart from scratch
  useEffect(() => {
    setIsTraining(false);
    setStepCount(0);
    setLossPoints([]);
    setEvents([]);
    setProbes({});
    pending.current = { lossSum: 0, regSum: 0, count: 0, ms: 0 };
    setCollapse([{ step: 0, ...collapseStats(imageFeatures(model, collapseImages)) }]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  // Note recipe changes on the timeline, since they explain sudden turns in the charts
  const lastRecipe = useRef(settings.recipe);
  useEffect(() => {
    if (settings.recipe === lastRecipe.current) return;
    lastRecipe.current = settings.recipe;
    const r = settings.recipe;
    const name = RECIPES.find(x => x.id === r.ablation)?.label ?? r.ablation;
    const text = `Recipe: ${name}${r.varWeight > 0 ? ' + anti-collapse term' : ''}`;
    setEvents(e => [...e, { step: model.steps, text }]);
  }, [settings.recipe, model]);

  const flushPoint = () => {
    const p = pending.current;
    if (p.count === 0) return;
    setLossPoints(pts => [...pts, { step: latest.current.model.steps, loss: p.lossSum / p.count, reg: p.regSum / p.count }]);
    setMsPerStep(p.ms / p.count);
    pending.current = { lossSum: 0, regSum: 0, count: 0, ms: 0 };
  };

  /** Train until `maxSteps` are done or `budgetMs` has passed */
  const runSteps = (maxSteps: number, budgetMs: number) => {
    const { model, settings } = latest.current;
    const sizes = shapeSizes(settings);
    const t0 = performance.now();
    let steps = 0;
    do {
      const tStep = performance.now();
      const batch = Array.from({ length: settings.batchSize }, () => patchify(randomShapeImage(Math.random, IMAGE_SIZE, sizes), PATCH_SIZE));
      const masks = batch.map(() => sampleMask(Math.random, GRID, settings.numTargets));
      const { predLoss, regLoss } = model.trainStep(batch, masks, settings.learningRate);
      const p = pending.current;
      p.lossSum += predLoss;
      p.regSum += regLoss;
      p.count++;
      p.ms += performance.now() - tStep;
      steps++;
      if (p.count >= POINT_EVERY) flushPoint();
      if (model.steps % COLLAPSE_EVERY === 0) measureCollapse();
    } while (steps < maxSteps && performance.now() - t0 < budgetMs);
    setStepCount(model.steps);
  };

  useEffect(() => {
    if (!isTraining) {
      flushPoint();
      return;
    }
    let timer: number;
    const tick = () => {
      runSteps(Infinity, TICK_BUDGET_MS);
      timer = window.setTimeout(tick, TICK_GAP_MS);
    };
    timer = window.setTimeout(tick, 0);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTraining]);

  const stepOnce = () => {
    runSteps(1, Infinity);
    flushPoint();
  };

  // ── Probes ────────────────────────────────────────────────────────────────
  const [pool, setPool] = useState<Pool>('mean');
  const [probes, setProbes] = useState<{
    pixels?: ProbeRow;
    untrained?: ProbeRow;
    trained?: ProbeRow & { step: number; spread: number };
  }>({});
  const [probeStatus, setProbeStatus] = useState<string | null>(null);
  // Raw pixels and the untrained encoder never change for a given image set / model, so compute them once
  const cache = useRef(new Map<string, ProbeRow>());
  const cacheKey = (what: string) => `${what}|${pool}|${settings.shapes}`;
  useEffect(() => {
    cache.current.clear();
  }, [untrained]);
  useEffect(() => {
    setProbes({});
  }, [pool, settings.shapes]);

  const runAllProbes = async () => {
    const sets = probeSets;
    const features = (m: MicroJepa) => [imageFeatures(m, sets.train, pool), imageFeatures(m, sets.test, pool)] as const;
    const cached = async (what: string, label: string, compute: () => ProbeRow) => {
      const key = cacheKey(what);
      if (!cache.current.has(key)) {
        setProbeStatus(`Probing ${label}…`);
        await nextFrame();
        cache.current.set(key, compute());
      }
      return cache.current.get(key)!;
    };
    const pixels = await cached('pixels', 'raw pixels', () => probeRow(pixelFeatures(sets.train), pixelFeatures(sets.test), sets));
    setProbes(p => ({ ...p, pixels }));
    const base = await cached('untrained', 'the untrained encoder', () => probeRow(...features(untrained), sets));
    setProbes(p => ({ ...p, untrained: base }));
    setProbeStatus('Probing the trained encoder…');
    await nextFrame();
    const step = model.steps;
    const [trainX, testX] = features(model);
    const { spread } = collapseStats(pool === 'mean' ? trainX.slice(0, COLLAPSE_IMAGES) : imageFeatures(model, sets.train.slice(0, COLLAPSE_IMAGES)));
    const trained = probeRow(trainX, testX, sets);
    setProbes(p => ({ ...p, trained: { ...trained, step, spread } }));
    setProbeStatus(null);
  };

  // ── Charts ────────────────────────────────────────────────────────────────
  // Charts created while hidden (0×0) don't always notice becoming visible, so resize on show
  const chartsRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!visible) return;
    const timer = window.setTimeout(() => {
      chartsRef.current?.querySelectorAll('canvas').forEach(c => ChartJS.getChart(c)?.resize());
    });
    return () => clearTimeout(timer);
  }, [visible]);

  const axis = (title: string, extra: object = {}) => ({
    title: { display: true, text: title, color: THEME.textMuted, font: { family: 'Inter', size: 11 } },
    ticks: { color: THEME.textDim },
    grid: { color: withAlpha(THEME.border, 0.6) },
    ...extra,
  });
  const baseOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false as const,
    plugins: {
      legend: { labels: { color: THEME.textMuted, font: { family: 'Inter' } } },
      tooltip: { mode: 'index' as const, intersect: false },
    },
  };

  const showReg = lossPoints.some(p => p.reg > 0);
  const lossData = {
    labels: lossPoints.map(p => `#${p.step}`),
    datasets: [
      {
        label: 'Prediction loss',
        data: lossPoints.map(p => p.loss),
        borderColor: THEME.primary,
        backgroundColor: withAlpha(THEME.primary, 0.12),
        tension: 0.2,
        pointRadius: 0,
        borderWidth: 2,
      },
      ...(showReg
        ? [{
            label: 'Anti-collapse term',
            data: lossPoints.map(p => (p.reg > 0 ? p.reg : null)),
            borderColor: THEME.purple,
            backgroundColor: withAlpha(THEME.purple, 0.12),
            borderDash: [6, 4],
            tension: 0.2,
            pointRadius: 0,
            borderWidth: 2,
            spanGaps: false,
          }]
        : []),
    ],
  };
  const lossOptions = {
    ...baseOptions,
    scales: {
      x: axis('Training step', { ticks: { color: THEME.textDim, maxTicksLimit: 10 } }),
      y: axis('Loss (log scale)', { type: 'logarithmic' as const }),
    },
  };

  const collapseData = {
    labels: collapse.map(c => `#${c.step}`),
    datasets: [
      {
        label: 'Effective rank (directions used)',
        data: collapse.map(c => (c.spread < 0.01 ? null : c.effectiveRank)),
        borderColor: THEME.emerald,
        backgroundColor: withAlpha(THEME.emerald, 0.12),
        tension: 0.2,
        pointRadius: collapse.length > 40 ? 0 : 2,
        borderWidth: 2,
        yAxisID: 'rank',
      },
      {
        label: 'Spread across images',
        data: collapse.map(c => c.spread),
        borderColor: THEME.amber,
        backgroundColor: withAlpha(THEME.amber, 0.12),
        borderDash: [6, 4],
        tension: 0.2,
        pointRadius: collapse.length > 40 ? 0 : 2,
        borderWidth: 2,
        yAxisID: 'spread',
      },
    ],
  };
  const collapseOptions = {
    ...baseOptions,
    scales: {
      x: axis('Training step', { ticks: { color: THEME.textDim, maxTicksLimit: 10 } }),
      rank: axis('Effective rank', { position: 'left' as const, min: 0, max: settings.dModel }),
      spread: axis('Spread', { position: 'right' as const, min: 0, grid: { drawOnChartArea: false } }),
    },
  };

  const current = collapse[collapse.length - 1];
  const verdict = current ? collapseVerdict(current, settings.dModel) : null;
  const recipe = settings.recipe;
  const antiCollapseOn = recipe.varWeight > 0;
  const setRecipe = (patch: Partial<JepaSettings['recipe']>) => onChange({ ...settings, recipe: { ...recipe, ...patch } });

  return (
    <div ref={chartsRef} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 480px), 1fr))', gap: 20 }}>
      {/* Controls */}
      <div className="glass-panel" style={{ padding: '16px 24px', gridColumn: '1 / -1', display: 'flex', flexDirection: 'column', gap: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
          <div>
            <p style={{ fontSize: '0.95rem', fontWeight: 700 }}>Training on: endless random {settings.shapes} shapes, {settings.batchSize} per step</p>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              {settings.numTargets} hidden block{settings.numTargets > 1 ? 's' : ''} per image · {model.getParameterCount().toLocaleString()} parameters
              · EMA momentum {model.momentum().toFixed(4)}
              {msPerStep !== null && <> · {msPerStep.toFixed(0)} ms per step</>}
              {' '}· this model isn't saved: a reload starts it over
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="badge badge-primary font-mono">Step: #{stepCount}</span>
            <button className="btn-secondary" onClick={onNavigateToSetup} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
              Change in Setup
            </button>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 18, flexWrap: 'wrap', paddingTop: 14, borderTop: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <button className={isTraining ? 'btn-secondary' : 'btn-primary'} onClick={() => setIsTraining(!isTraining)}>
              {isTraining ? <Pause size={16} /> : <Play size={16} />}
              {isTraining ? 'Pause Training' : 'Start Training'}
            </button>
            <button className="btn-secondary" onClick={stepOnce} disabled={isTraining}>
              <FastForward size={16} /> Step
            </button>
            <button className="btn-secondary" onClick={onStartOver} title="New random weights with the same settings; clears the charts">
              <RotateCcw size={16} /> Start over
            </button>
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
              <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>Learning rate</span>
              <InfoTooltip
                title="Learning rate"
                description="How big a step each weight takes per update (AdamW). Safe to change while training."
                impact="In our tests 0.0003 kept the representation healthiest; 0.001 learns faster but collapses partially sooner."
              />
            </div>
            <div style={{ display: 'flex', gap: 6 }}>
              {[0.0001, 0.0003, 0.001, 0.003].map(lr => (
                <button
                  key={lr}
                  className={settings.learningRate === lr ? 'btn-primary' : 'btn-secondary'}
                  onClick={() => onChange({ ...settings, learningRate: lr })}
                  style={{ padding: '4px 10px', fontSize: '0.75rem' }}
                >
                  {lr}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Recipe: the anti-collapse machinery, switchable mid-training */}
        <div style={{ paddingTop: 14, borderTop: '1px solid var(--border-color)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
            <Shield size={15} color="var(--accent-emerald)" />
            <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>Training recipe</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>safe to switch while training: watch the collapse chart react</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
            {RECIPES.map(r => {
              const on = recipe.ablation === r.id;
              return (
                <button
                  key={r.id}
                  onClick={() => setRecipe({ ablation: r.id })}
                  aria-pressed={on}
                  style={{
                    textAlign: 'left', padding: '10px 12px', borderRadius: 8, cursor: 'pointer', font: 'inherit', color: 'inherit',
                    background: on ? 'var(--primary-soft)' : 'var(--surface-inset)',
                    border: `1px solid ${on ? 'var(--primary)' : 'var(--border-color)'}`,
                  }}
                >
                  <span style={{ fontWeight: 700, fontSize: '0.85rem' }}>{r.label}</span>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}> · {r.short}</span>
                  <span style={{ display: 'block', fontSize: '0.775rem', color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.45 }}>{r.description}</span>
                </button>
              );
            })}
            <label
              style={{
                display: 'flex', gap: 10, alignItems: 'flex-start', padding: '10px 12px', borderRadius: 8, cursor: 'pointer',
                background: antiCollapseOn ? 'color-mix(in srgb, var(--accent-purple) 14%, transparent)' : 'var(--surface-inset)',
                border: `1px solid ${antiCollapseOn ? 'var(--accent-purple)' : 'var(--border-color)'}`,
              }}
            >
              <input
                type="checkbox"
                checked={antiCollapseOn}
                onChange={e => setRecipe(e.target.checked ? ANTI_COLLAPSE : { varWeight: 0, covWeight: 0 })}
                style={{ marginTop: 3, accentColor: 'var(--accent-purple)' }}
              />
              <span>
                <span style={{ fontWeight: 700, fontSize: '0.85rem' }}>+ Anti-collapse term</span>
                <span style={{ display: 'block', fontSize: '0.775rem', color: 'var(--text-muted)', marginTop: 4, lineHeight: 1.45 }}>
                  Adds a penalty (from VICReg) when an embedding dimension barely varies across the batch, or when dimensions
                  repeat each other. Fixes the partial collapse.
                </span>
              </span>
            </label>
          </div>
        </div>
      </div>

      {/* Loss */}
      <div className="glass-panel" style={{ padding: '18px 22px' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Activity size={17} color="var(--primary)" /> Prediction loss
        </h3>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 10, lineHeight: 1.5 }}>
          How far the predictor's guesses are from the target encoder's embeddings of the hidden patches. Unlike the text model,
          a low loss here doesn't prove much: it also falls if the targets themselves become easy to guess.
        </p>
        <div style={{ height: 260 }}>
          {lossPoints.length > 0 ? (
            <Line data={lossData} options={lossOptions} />
          ) : (
            <p style={{ color: 'var(--text-dim)', fontSize: '0.85rem', paddingTop: 100, textAlign: 'center' }}>Press Start to train.</p>
          )}
        </div>
      </div>

      {/* Collapse */}
      <div className="glass-panel" style={{ padding: '18px 22px' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Shield size={17} color="var(--accent-emerald)" /> Collapse check
          <InfoTooltip
            title="Collapse check"
            description={`Every ${COLLAPSE_EVERY} steps, ${COLLAPSE_IMAGES} fixed images are embedded (each image's patch embeddings averaged into one vector). Spread is how much each dimension varies across those images. Effective rank is how many independent directions they spread along, from 1 to the embedding size.`}
            impact="Spread near 0 means full collapse (every image looks the same). Low effective rank with normal spread is partial collapse: images differ only in a couple of ways."
          />
        </h3>
        {verdict && current && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, margin: '6px 0 10px', flexWrap: 'wrap' }}>
            <span className="badge" style={{ color: verdict.color, border: `1px solid ${verdict.color}`, background: 'transparent' }}>{verdict.label}</span>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              {verdict.text} (Untrained: rank {untrainedStats.effectiveRank.toFixed(1)}, spread {untrainedStats.spread.toFixed(2)}.)
            </span>
          </div>
        )}
        <div style={{ height: 230 }}>
          <Line data={collapseData} options={collapseOptions} />
        </div>
        {events.length > 0 && (
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: 8 }}>
            {events.slice(-4).map(e => `#${e.step}: ${e.text}`).join(' · ')}
          </p>
        )}
      </div>

      {/* Probes */}
      <div className="glass-panel" style={{ padding: '18px 22px', gridColumn: '1 / -1' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', marginBottom: 8 }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 8 }}>
            <Microscope size={17} color="var(--accent-purple)" /> What did it learn? Linear probes
          </h3>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Read the embeddings</span>
            {([['mean', 'averaged into one vector'], ['concat', 'patch by patch']] as const).map(([p, label]) => (
              <button
                key={p}
                className={pool === p ? 'btn-primary' : 'btn-secondary'}
                onClick={() => setPool(p)}
                disabled={!!probeStatus}
                style={{ padding: '4px 10px', fontSize: '0.75rem' }}
              >
                {label}
              </button>
            ))}
            <button className="btn-primary" onClick={runAllProbes} disabled={!!probeStatus}>
              <Microscope size={15} /> {probeStatus ?? 'Run probes'}
            </button>
          </div>
        </div>
        <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: 12 }}>
          The JEPA never sees a label. To test what its embeddings capture, freeze the encoder, embed {probeSets.train.length} labelled
          images, and fit the simplest possible reader, one linear map, to recover each image's shape, colour and position. Then
          score it on {probeSets.test.length} images it hasn't seen. The real promise of self-supervised learning is needing{' '}
          <i>few</i> labels, so the probe also gets to learn from only 5 or 20 images per shape. Compare against raw pixels and
          against the same network before training.
        </p>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', minWidth: 640 }}>
            <thead>
              <tr style={{ color: 'var(--text-dim)', textAlign: 'right' }}>
                <th style={{ textAlign: 'left', padding: '6px 8px', fontWeight: 600 }}>Features</th>
                <th style={{ padding: '6px 8px', fontWeight: 600 }}>Shape · 5 labels/class</th>
                <th style={{ padding: '6px 8px', fontWeight: 600 }}>Shape · 20</th>
                <th style={{ padding: '6px 8px', fontWeight: 600 }}>Shape · all</th>
                <th style={{ padding: '6px 8px', fontWeight: 600 }}>Colour · 5</th>
                <th style={{ padding: '6px 8px', fontWeight: 600 }}>Position R² · 20</th>
                <th style={{ padding: '6px 8px', fontWeight: 600 }}>Position R² · all</th>
              </tr>
            </thead>
            <tbody>
              {([
                ['Raw pixels', probes.pixels],
                ['Untrained encoder', probes.untrained],
                [probes.trained ? `Trained encoder (step #${probes.trained.step})` : 'Trained encoder', probes.trained],
              ] as const).map(([label, row]) => {
                const base = probes.untrained;
                const isTrained = row !== undefined && row === probes.trained;
                const cell = (key: keyof ProbeRow, fmt: (v: number) => string) => {
                  if (!row) return <td style={{ padding: '7px 8px', textAlign: 'right', color: 'var(--text-dim)' }}>-</td>;
                  const delta = isTrained && base ? row[key] - base[key] : 0;
                  const color = !isTrained || Math.abs(delta) < 0.03 ? 'var(--text-main)' : delta > 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)';
                  return <td className="font-mono" style={{ padding: '7px 8px', textAlign: 'right', color }}>{fmt(row[key])}</td>;
                };
                return (
                  <tr key={label} style={{ borderTop: '1px solid var(--border-color)' }}>
                    <td style={{ padding: '7px 8px', fontWeight: isTrained ? 700 : 500 }}>{label}</td>
                    {cell('shape5', pct)}
                    {cell('shape20', pct)}
                    {cell('shapeAll', pct)}
                    {cell('colour5', pct)}
                    {cell('pos20', r2)}
                    {cell('posAll', r2)}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {probes.trained && probes.trained.spread < COLLAPSED_SPREAD && (
          <p style={{ fontSize: '0.8rem', color: 'var(--accent-amber)', marginTop: 10, lineHeight: 1.5 }}>
            Careful: this encoder had collapsed when it was probed (spread {probes.trained.spread.toExponential(0)}). Its embeddings
            still differ by tiny amounts, and the probe rescales every number to the same size before reading it, so it can still
            pick up a signal. Scores here overstate what a collapsed model is good for. Collapse metrics and probes measure
            different things.
          </p>
        )}
        <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: 10, lineHeight: 1.5 }}>
          Chance is 25% for shape and colour; R² is 0 for guessing the average position and 1 for perfect. Trained scores turn green
          or red when they beat or trail the untrained encoder by 3 points or more. Few-label scores average 5 random picks of the
          labelled images. Colour is easy for everything, so it's only shown with 5 labels.
        </p>
      </div>
    </div>
  );
};
