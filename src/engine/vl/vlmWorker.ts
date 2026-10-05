// Continuous training of the vision-language model in a background thread (a Web Worker), with
// the same protocol as clipWorker.ts: one progress point per ~100 ms chunk, the question test
// every EVAL_EVERY steps, and the weights sent back about once a second and when stopped.
// The frozen vision encoder is sent once at the start (it never changes during a run).
import { ClipConfig, ClipState, MicroClip } from './clip';
import { MicroVlm, VlmConfig, VlmPhase, VlmState } from './vlm';
import { EVAL_EVERY, VlmDataSettings, VlmEvalResult, VlmEvalSet, runVlmEval, trainVlmChunk, vlmEvalSet } from './vlmTraining';

export type ToVlmWorker =
  | {
      type: 'start';
      run: number;
      visionConfig: ClipConfig;
      visionState: ClipState;
      config: VlmConfig;
      state: VlmState;
      data: VlmDataSettings;
      phase: VlmPhase;
      batchSize: number;
      learningRate: number;
      step: number;
      /** Steps since the phase began: the first chunk of a phase is evaluated as its baseline */
      phaseStep: number;
    }
  | { type: 'set'; learningRate?: number; batchSize?: number }
  | { type: 'stop' };

export type FromVlmWorker =
  | { type: 'progress'; run: number; startStep: number; steps: number; lossSum: number }
  | { type: 'eval'; run: number; result: VlmEvalResult }
  | { type: 'weights'; run: number; step: number; state: VlmState; final: boolean };

const CHUNK_MS = 100;
const SYNC_MS = 1000;

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<ToVlmWorker>) => void) | null;
  postMessage(message: FromVlmWorker, transfer?: Transferable[]): void;
};

let model: MicroVlm | null = null;
let job: Extract<ToVlmWorker, { type: 'start' }> | null = null;
let set: VlmEvalSet | null = null;
let running = false;
let step = 0;
let lastSync = 0;

function sendWeights(final: boolean) {
  if (!model || !job) return;
  const state = model.exportState();
  const buffers = [...Object.values(state.weights), ...Object.values(state.optimizer.m), ...Object.values(state.optimizer.v)].map(a => a.buffer);
  scope.postMessage({ type: 'weights', run: job.run, step, state, final }, buffers);
  lastSync = performance.now();
}

function loop() {
  if (!running || !model || !job || !set) return;
  const startStep = step;
  const r = trainVlmChunk(model, job.data, { phase: job.phase, batchSize: job.batchSize, learningRate: job.learningRate, budgetMs: CHUNK_MS, maxSteps: Infinity });
  step += r.steps;
  scope.postMessage({ type: 'progress', run: job.run, startStep, steps: r.steps, lossSum: r.lossSum });
  if (Math.floor(step / EVAL_EVERY) > Math.floor(startStep / EVAL_EVERY)) {
    scope.postMessage({ type: 'eval', run: job.run, result: runVlmEval(model, set, step, job.phase) });
  }
  if (performance.now() - lastSync >= SYNC_MS) sendWeights(false);
  setTimeout(loop, 0);
}

scope.onmessage = e => {
  const msg = e.data;
  if (msg.type === 'start') {
    job = msg;
    const vision = new MicroClip(msg.visionConfig);
    vision.importState(msg.visionState);
    model = new MicroVlm(vision, msg.config);
    model.importState(msg.state);
    model.steps = msg.step;
    set = vlmEvalSet(msg.data.patchSize, msg.data.heldOut);
    step = msg.step;
    lastSync = performance.now();
    running = true;
    // A phase's starting point, before any of its training: the baseline for that phase
    if (msg.phaseStep === 0) scope.postMessage({ type: 'eval', run: msg.run, result: runVlmEval(model, set, step, msg.phase) });
    setTimeout(loop, 0);
  } else if (msg.type === 'set' && job) {
    if (msg.learningRate !== undefined) job.learningRate = msg.learningRate;
    if (msg.batchSize !== undefined) job.batchSize = msg.batchSize;
  } else if (msg.type === 'stop') {
    if (running) {
      running = false;
      sendWeights(true);
    }
  }
};
