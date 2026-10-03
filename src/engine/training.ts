// The continuous training loop's building blocks, shared by the background worker
// (trainWorker.ts) and the Train tab's on-page fallback, so both train exactly the same way.
import { MicroTransformer } from './transformer';

/**
 * Measure validation loss every N training steps (and on the very first chunk). Each measurement
 * is up to MAX_VAL_WINDOWS forward passes, so measuring too often eats into training time
 * (every 50 steps cost ~15%).
 */
export const VAL_EVERY = 200;
/** Each validation measurement uses at most this many evenly spaced held-out windows */
export const MAX_VAL_WINDOWS = 32;

/**
 * A random context-sized window from the training tokens. Random sampling (rather than sliding
 * one token per step) means every part of the text gets seen early.
 */
export function sampleWindow(tokens: number[], contextWindow: number) {
  const maxStart = Math.max(1, tokens.length - contextWindow - 1);
  const startIdx = Math.floor(Math.random() * maxStart);
  return {
    startIdx,
    inputSeq: tokens.slice(startIdx, startIdx + contextWindow),
    targetSeq: tokens.slice(startIdx + 1, startIdx + contextWindow + 1),
  };
}

/**
 * Average loss over windows of the held-out tail (forward pass only). Large splits are capped at
 * MAX_VAL_WINDOWS evenly spaced windows, the same ones every time, so the curve stays smooth.
 */
export function validationLoss(model: MicroTransformer, valTokens: number[], contextWindow: number): number | null {
  if (valTokens.length < 2) return null;
  const W = contextWindow;
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
}

export interface ChunkResult {
  steps: number;
  lossSum: number;
  /** Measured if this chunk started at step 0 or crossed a multiple of VAL_EVERY, else null */
  valLoss: number | null;
}

/**
 * Train until `maxSteps` are done or `budgetMs` has passed, starting at step `startStep`.
 * One chunk becomes one point on the loss chart (its mean loss).
 */
export function trainChunk(
  model: MicroTransformer,
  trainTokens: number[],
  valTokens: number[],
  opts: { contextWindow: number; learningRate: number; startStep: number; budgetMs: number; maxSteps: number }
): ChunkResult {
  const t0 = performance.now();
  let lossSum = 0;
  let steps = 0;
  do {
    const { inputSeq, targetSeq } = sampleWindow(trainTokens, opts.contextWindow);
    lossSum += model.trainStep(inputSeq, targetSeq, opts.learningRate).loss;
    steps++;
  } while (steps < opts.maxSteps && performance.now() - t0 < opts.budgetMs);

  const valLoss = isValidationDue(opts.startStep, opts.startStep + steps) ? validationLoss(model, valTokens, opts.contextWindow) : null;
  return { steps, lossSum, valLoss };
}

/** Validate on the first chunk, and whenever a chunk crosses a multiple of VAL_EVERY */
export function isValidationDue(startStep: number, endStep: number): boolean {
  return startStep === 0 || Math.floor(endStep / VAL_EVERY) > Math.floor(startStep / VAL_EVERY);
}
