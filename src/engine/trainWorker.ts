// Continuous training in a background thread (a Web Worker).
//
// Why: on the page, training has to pause every ~100 ms so the screen can repaint, and browsers
// slow page timers to a crawl in background tabs. A worker trains nonstop on its own thread and
// isn't throttled that way. It runs the very same engine code (trainChunk → trainStep).
//
// The page keeps its own copy of the model (Look inside, generation and benchmarks read it).
// The worker sends its weights back about once a second, and once more when it stops.
import { MicroTransformer, ModelState } from './transformer';
import { OptimizerType } from './optimizer';
import { trainChunk } from './training';
import { TransformerConfig } from '../types';

export type ToWorker =
  | {
      type: 'start';
      run: number;
      config: TransformerConfig;
      state: ModelState;
      trainTokens: number[];
      valTokens: number[];
      step: number;
    }
  | { type: 'set'; learningRate?: number; optimizer?: OptimizerType }
  | { type: 'stop' };

export type FromWorker =
  | { type: 'progress'; run: number; startStep: number; steps: number; lossSum: number; valLoss: number | null }
  | { type: 'weights'; run: number; step: number; state: ModelState; final: boolean };

/** One chunk of training per chart point, like the on-page loop */
const CHUNK_MS = 100;
/** How often the page's copy of the model is refreshed while training */
const SYNC_MS = 1000;

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null;
  postMessage(message: FromWorker, transfer?: Transferable[]): void;
};

let model: MicroTransformer | null = null;
let job: Extract<ToWorker, { type: 'start' }> | null = null;
let running = false;
let step = 0;
let lastSync = 0;

/** Send the weights home. Their buffers are transferred (moved, not copied); exportState makes fresh ones. */
function sendWeights(final: boolean) {
  if (!model || !job) return;
  const state = model.exportState();
  const buffers = [
    ...Object.values(state.weights),
    ...Object.values(state.optimizer.m),
    ...Object.values(state.optimizer.v),
  ].map(a => a.buffer);
  scope.postMessage({ type: 'weights', run: job.run, step, state, final }, buffers);
  lastSync = performance.now();
}

function loop() {
  if (!running || !model || !job) return;
  const startStep = step;
  const r = trainChunk(model, job.trainTokens, job.valTokens, {
    contextWindow: job.config.contextWindow,
    learningRate: job.config.learningRate,
    startStep,
    budgetMs: CHUNK_MS,
    maxSteps: Infinity,
  });
  step += r.steps;
  scope.postMessage({ type: 'progress', run: job.run, startStep, steps: r.steps, lossSum: r.lossSum, valLoss: r.valLoss });
  if (performance.now() - lastSync >= SYNC_MS) sendWeights(false);
  // Yield between chunks so 'set' and 'stop' messages get handled
  setTimeout(loop, 0);
}

scope.onmessage = e => {
  const msg = e.data;
  if (msg.type === 'start') {
    job = msg;
    model = new MicroTransformer(msg.config);
    model.importState(msg.state);
    step = msg.step;
    lastSync = performance.now();
    running = true;
    setTimeout(loop, 0);
  } else if (msg.type === 'set' && job && model) {
    if (msg.learningRate !== undefined) job.config = { ...job.config, learningRate: msg.learningRate };
    if (msg.optimizer !== undefined) model.setOptimizer(msg.optimizer);
  } else if (msg.type === 'stop') {
    if (running) {
      running = false;
      sendWeights(true);
    }
  }
};
