// The vision-language model's training loop building blocks, shared by the background worker
// (vlmWorker.ts) and the on-page fallback, so both train exactly the same way.
import { seededRandom } from '../datasets';
import { Matrix } from '../tensor';
import { Combo, isHeldOut } from './captions';
import { IMAGE_SIZE } from './clipData';
import { drawBlankImage, drawShapeImage, patchify, randomLabels } from './shapes';
import { MicroVlm, VlmPhase } from './vlm';
import { VQA_TASKS, VlmExample, VqaTask, makeExample, sampleVlmBatch } from './vlmData';

export { EVAL_EVERY, isEvalDue } from './clipTraining';

/** Phase 1 (align) trains on descriptions only, as LLaVA's first stage trains on captions */
export const tasksForPhase = (phase: VlmPhase, chosen: VqaTask[]): VqaTask[] => (phase === 1 ? ['describe'] : chosen);

export interface VlmDataSettings {
  patchSize: number;
  heldOut: Combo[];
  tasks: VqaTask[];   // the question kinds trained on in phases 0 and 2
}

// ── The test ─────────────────────────────────────────────────────────────────

const SEEN_PICTURES = 32;
const HELD_OUT_PICTURES = 16;

/** Fixed test pictures, each asked every kind of question */
export interface VlmEvalSet {
  pictures: { patches: Matrix; heldOut: boolean; examples: VlmExample[] }[];
  blankPatches: Matrix;
}

export function vlmEvalSet(patchSize: number, heldOut: Combo[]): VlmEvalSet {
  const make = (seed: number, n: number, wantHeldOut: boolean) => {
    if (wantHeldOut && heldOut.length === 0) return [];
    const rand = seededRandom(seed);
    return Array.from({ length: n }, () => {
      const labels = randomLabels(rand, l => isHeldOut(l, heldOut) === wantHeldOut);
      const image = drawShapeImage(labels, rand, IMAGE_SIZE);
      return {
        patches: patchify(image, patchSize),
        heldOut: wantHeldOut,
        examples: VQA_TASKS.map(t => makeExample(image, patchSize, t.id, rand, heldOut)),
      };
    });
  };
  return {
    pictures: [...make(4001, SEEN_PICTURES, false), ...make(4002, HELD_OUT_PICTURES, true)],
    blankPatches: patchify(drawBlankImage(seededRandom(4003), IMAGE_SIZE), patchSize),
  };
}

/** Fraction answered exactly right, with the real picture and with a blank one instead */
export interface VqaScore {
  seen: number;
  seenBlank: number;
  heldOut: number | null;      // null when nothing is held out
  heldOutBlank: number | null;
}

export interface VlmEvalResult {
  step: number;
  phase: VlmPhase;
  tasks: Record<VqaTask, VqaScore>;
}

/**
 * Ask every test picture every kind of question, once with the picture and once with a blank
 * picture (just the background). If the answers are as good without the picture, the model
 * isn't looking: it's answering from what's usually true.
 */
export function runVlmEval(model: MicroVlm, set: VlmEvalSet, step: number, phase: VlmPhase): VlmEvalResult {
  const blankTokens = model.imageTokens(model.features(set.blankPatches));
  const tally = new Map<VqaTask, { s: number; sb: number; sn: number; h: number; hb: number; hn: number }>();
  for (const t of VQA_TASKS) tally.set(t.id, { s: 0, sb: 0, sn: 0, h: 0, hb: 0, hn: 0 });
  for (const pic of set.pictures) {
    const tokens = model.imageTokens(model.features(pic.patches));
    for (const ex of pic.examples) {
      const t = tally.get(ex.task)!;
      const right = model.answersCorrectly(ex, tokens) ? 1 : 0;
      const rightBlank = model.answersCorrectly(ex, blankTokens) ? 1 : 0;
      if (pic.heldOut) {
        t.h += right;
        t.hb += rightBlank;
        t.hn++;
      } else {
        t.s += right;
        t.sb += rightBlank;
        t.sn++;
      }
    }
  }
  const tasks = {} as Record<VqaTask, VqaScore>;
  for (const [task, t] of tally) {
    tasks[task] = {
      seen: t.s / Math.max(1, t.sn),
      seenBlank: t.sb / Math.max(1, t.sn),
      heldOut: t.hn ? t.h / t.hn : null,
      heldOutBlank: t.hn ? t.hb / t.hn : null,
    };
  }
  return { step, phase, tasks };
}

// ── Training ─────────────────────────────────────────────────────────────────

export interface VlmChunkResult {
  steps: number;
  lossSum: number;
}

/** Train until `maxSteps` are done or `budgetMs` has passed. One chunk becomes one chart point. */
export function trainVlmChunk(
  model: MicroVlm,
  data: VlmDataSettings,
  opts: { phase: VlmPhase; batchSize: number; learningRate: number; budgetMs: number; maxSteps: number; rand?: () => number }
): VlmChunkResult {
  const rand = opts.rand ?? Math.random;
  const tasks = tasksForPhase(opts.phase, data.tasks);
  const t0 = performance.now();
  let steps = 0;
  let lossSum = 0;
  do {
    const batch = sampleVlmBatch(rand, data.patchSize, tasks, data.heldOut, opts.batchSize);
    // The vision encoder is frozen, so its features are just computed (no gradient); phase 0 doesn't use them
    const features = opts.phase === 0 ? [] : batch.map(ex => model.features(ex.patches));
    lossSum += model.trainStep(batch, features, opts.phase, opts.learningRate).loss;
    steps++;
  } while (steps < opts.maxSteps && performance.now() - t0 < opts.budgetMs);
  return { steps, lossSum };
}
