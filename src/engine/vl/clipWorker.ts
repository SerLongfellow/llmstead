// Continuous training of the vision-language model in a background thread (a Web Worker), the
// same way trainWorker.ts trains the GPT: nonstop, unthrottled in background tabs, never blocking
// the page. It also runs the zero-shot tests every EVAL_EVERY steps, so the page doesn't have to.
//
// The page keeps its own copy of the model (Look inside and the similarity grid read it). The
// worker sends its weights back about once a second, and once more when it stops.
import { ClipConfig, ClipState, FalseNegatives, MicroClip } from './clip';
import { ClipData } from './clipData';
import { EvalResult, EvalSets, evalSets, isEvalDue, runEval, trainClipChunk } from './clipTraining';

export type ToClipWorker =
  | {
      type: 'start';
      run: number;
      config: ClipConfig;
      state: ClipState;
      data: ClipData;
      batchSize: number;
      learningRate: number;
      step: number;
    }
  | { type: 'set'; learningRate?: number; batchSize?: number; falseNegatives?: FalseNegatives }
  | { type: 'stop' };

export type FromClipWorker =
  | { type: 'progress'; run: number; startStep: number; steps: number; lossSum: number; accSum: number; scale: number }
  | { type: 'eval'; run: number; result: EvalResult }
  | { type: 'weights'; run: number; step: number; state: ClipState; final: boolean };

/** One chunk of training per chart point */
const CHUNK_MS = 100;
/** How often the page's copy of the model is refreshed while training */
const SYNC_MS = 1000;

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<ToClipWorker>) => void) | null;
  postMessage(message: FromClipWorker, transfer?: Transferable[]): void;
};

let model: MicroClip | null = null;
let job: Extract<ToClipWorker, { type: 'start' }> | null = null;
let sets: EvalSets | null = null;
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
  if (!running || !model || !job || !sets) return;
  const startStep = step;
  // The untrained model's scores first: the baseline every later score is compared with
  if (startStep === 0) scope.postMessage({ type: 'eval', run: job.run, result: runEval(model, sets, job.data, 0) });
  const r = trainClipChunk(model, job.data, { batchSize: job.batchSize, learningRate: job.learningRate, budgetMs: CHUNK_MS, maxSteps: Infinity });
  step += r.steps;
  scope.postMessage({ type: 'progress', run: job.run, startStep, steps: r.steps, lossSum: r.lossSum, accSum: r.accSum, scale: model.logitScale() });
  if (startStep > 0 && isEvalDue(startStep, step)) {
    scope.postMessage({ type: 'eval', run: job.run, result: runEval(model, sets, job.data, step) });
  }
  if (performance.now() - lastSync >= SYNC_MS) sendWeights(false);
  // Yield between chunks so 'set' and 'stop' messages get handled
  setTimeout(loop, 0);
}

scope.onmessage = e => {
  const msg = e.data;
  if (msg.type === 'start') {
    job = msg;
    model = new MicroClip(msg.config);
    model.importState(msg.state);
    model.steps = msg.step;
    sets = evalSets(msg.data);
    step = msg.step;
    lastSync = performance.now();
    running = true;
    setTimeout(loop, 0);
  } else if (msg.type === 'set' && job && model) {
    if (msg.learningRate !== undefined) job.learningRate = msg.learningRate;
    if (msg.batchSize !== undefined) job.batchSize = msg.batchSize;
    if (msg.falseNegatives !== undefined) model.setFalseNegatives(msg.falseNegatives);
  } else if (msg.type === 'stop') {
    if (running) {
      running = false;
      sendWeights(true);
    }
  }
};
