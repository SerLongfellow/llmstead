// The vision-language model's training loop building blocks, shared by the background worker
// (clipWorker.ts) and the Train tab's on-page fallback, so both train exactly the same way.
import { seededRandom } from '../datasets';
import { MicroClip } from './clip';
import { ClipData, Pair, ZeroShotScore, captionBank, sampleBatch, testPairs, zeroShot } from './clipData';

/** Run the zero-shot tests every this many steps (and at step 0, the untrained baseline) */
export const EVAL_EVERY = 250;
const SEEN_TESTS = 128;
const HELD_OUT_TESTS = 64;

/** The fixed test pictures: the same ones every time, so scores are comparable across a run */
export interface EvalSets {
  seen: Pair[];
  heldOut: Pair[];
}

export function evalSets(data: ClipData): EvalSets {
  return {
    seen: testPairs(seededRandom(1001), data, SEEN_TESTS, 'seen'),
    heldOut: testPairs(seededRandom(1002), data, HELD_OUT_TESTS, 'held-out'),
  };
}

export interface EvalResult {
  step: number;
  seen: ZeroShotScore;
  heldOut: ZeroShotScore | null; // null when nothing is held out
}

export function runEval(model: MicroClip, sets: EvalSets, data: ClipData, step: number): EvalResult {
  const bank = captionBank(model, data.detail);
  return {
    step,
    seen: zeroShot(model, sets.seen, data.detail, bank),
    heldOut: sets.heldOut.length ? zeroShot(model, sets.heldOut, data.detail, bank) : null,
  };
}

export interface ClipChunkResult {
  steps: number;
  lossSum: number;
  accSum: number; // sum over steps of the batch's picture → caption accuracy
}

/** Train until `maxSteps` are done or `budgetMs` has passed. One chunk becomes one chart point. */
export function trainClipChunk(
  model: MicroClip,
  data: ClipData,
  opts: { batchSize: number; learningRate: number; budgetMs: number; maxSteps: number; rand?: () => number }
): ClipChunkResult {
  const rand = opts.rand ?? Math.random;
  const t0 = performance.now();
  let steps = 0;
  let lossSum = 0;
  let accSum = 0;
  do {
    const r = model.trainStep(sampleBatch(rand, data, opts.batchSize), opts.learningRate);
    lossSum += r.loss;
    accSum += r.imageToText;
    steps++;
  } while (steps < opts.maxSteps && performance.now() - t0 < opts.budgetMs);
  return { steps, lossSum, accSum };
}

/** Evaluate on the first chunk, and whenever a chunk crosses a multiple of EVAL_EVERY */
export function isEvalDue(startStep: number, endStep: number): boolean {
  return startStep === 0 || Math.floor(endStep / EVAL_EVERY) > Math.floor(startStep / EVAL_EVERY);
}
