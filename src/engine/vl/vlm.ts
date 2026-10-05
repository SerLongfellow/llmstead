import { StepInspectionData, TransformerConfig } from '../../types';
import { Optimizer, OptimizerState } from '../optimizer';
import { Matrix, MatrixMath } from '../tensor';
import { MicroTransformer } from '../transformer';
import { MicroClip } from './clip';
import { END_ID, MAX_QA_WORDS, VLM_VOCAB, VlmExample } from './vlmData';

/**
 * MicroVlm: a tiny LLaVA-style vision-language model. Three parts:
 *
 *   picture → patches → vision encoder (the CLIP's image tower, frozen) → one vector per patch
 *          → projector (a small MLP) → one "image token" per patch, the size of a word embedding
 *   [image tokens] + question words → language model (a GPT, the same MicroTransformer as the
 *          text side) → the answer, one word at a time
 *
 * The language model never sees pixels. It reads the projected patches exactly where it would
 * read token embeddings, and learns to use them because answering needs them.
 *
 * Training follows LLaVA's recipe in phases, each choosing what learns (and what stays frozen):
 *   0  text only: the image slots are empty (zeros); only the language model learns. It picks
 *      up the language and the answer format, and can only guess the facts.
 *   1  align: only the projector learns, on picture descriptions, with the language model
 *      frozen: the projector learns to say "red circle" in a form the LM already reads.
 *   2  instruction-tune: projector and language model learn together, on every kind of question.
 * The vision encoder stays frozen throughout, as in LLaVA.
 */

export type VlmPhase = 0 | 1 | 2;

export interface VlmConfig {
  dModel: number;    // language model width (and the projector's output size)
  numHeads: number;
  numLayers: number;
  mlpRatio: number;
}

export const DEFAULT_VLM_CONFIG: VlmConfig = { dModel: 32, numHeads: 2, numLayers: 2, mlpRatio: 4 };

/** The learned numbers (projector + language model), as plain typed arrays */
export interface VlmState {
  weights: Record<string, Float64Array>;
  optimizer: OptimizerState;
}

/** What one step returns */
export interface VlmStepResult {
  loss: number;
  gradNorm: number;
}

/** One answer, word by word, with what the model saw at each step (for Look inside) */
export interface VlmAnswer {
  words: string[];
  ids: number[];
  /** One forward pass per answer word: the sequence so far, ending where that word was chosen */
  steps: StepInspectionData[];
}

/** Which parameter groups each phase trains */
export const PHASE_TRAINS: Record<VlmPhase, { projector: boolean; lm: boolean }> = {
  0: { projector: false, lm: true },
  1: { projector: true, lm: false },
  2: { projector: true, lm: true },
};

const argmax = (row: number[]) => row.reduce((best, v, k) => (v > row[best] ? k : best), 0);

export class MicroVlm {
  public readonly config: VlmConfig;
  public readonly vision: MicroClip;     // frozen: never updated here
  public readonly lm: MicroTransformer;
  public readonly numPatches: number;
  public steps = 0;

  private wProj1: Matrix;   // [dVision x dModel]
  private wProj2: Matrix;   // [dModel x dModel]
  private optimizer = new Optimizer('adamw');

  /** `vision` is used as is and never trained: pass a copy (clone) if the original keeps training */
  constructor(vision: MicroClip, config: VlmConfig = DEFAULT_VLM_CONFIG) {
    this.config = config;
    this.vision = vision;
    this.numPatches = vision.numPatches;
    const dVision = vision.config.dModel;
    const lmConfig: TransformerConfig = {
      vocabSize: VLM_VOCAB.length,
      contextWindow: this.numPatches + MAX_QA_WORDS,
      dModel: config.dModel,
      numHeads: config.numHeads,
      numLayers: config.numLayers,
      mlpRatio: config.mlpRatio,
      learningRate: 0.001,
      optimizer: 'adamw',
    };
    this.lm = new MicroTransformer(lmConfig);
    this.wProj1 = MatrixMath.random(dVision, config.dModel, Math.sqrt(2.0 / dVision));
    this.wProj2 = MatrixMath.random(config.dModel, config.dModel, Math.sqrt(1.0 / config.dModel));
  }

  /** Every trainable matrix: 'proj.*' (the projector) and 'lm.*' (the language model); live arrays */
  public getParameters(): Record<string, Matrix> {
    const params: Record<string, Matrix> = { 'proj.w1': this.wProj1, 'proj.w2': this.wProj2 };
    for (const [name, M] of Object.entries(this.lm.getParameters())) params[`lm.${name}`] = M;
    return params;
  }

  public getParameterCount(): { projector: number; lm: number; vision: number } {
    const count = (ms: Matrix[]) => ms.reduce((s, M) => s + M.length * M[0].length, 0);
    return {
      projector: count([this.wProj1, this.wProj2]),
      lm: count(Object.values(this.lm.getParameters())),
      // only the image tower: the CLIP's text tower isn't part of the VLM
      vision: count(Object.entries(this.vision.getParameters()).filter(([n]) => n.startsWith('img.')).map(([, M]) => M)),
    };
  }

  // ── Picture → image tokens ──────────────────────────────────────────────

  /** The vision encoder's per-patch features: fixed for a given picture, since the encoder is frozen */
  public features(patches: Matrix): Matrix {
    return this.vision.patchFeatures(patches);
  }

  /** features → GELU(features · W1) · W2: one image token per patch */
  private project(features: Matrix) {
    const hiddenRaw = MatrixMath.matmul(features, this.wProj1);
    const hidden = MatrixMath.gelu(hiddenRaw);
    return { hiddenRaw, hidden, tokens: MatrixMath.matmul(hidden, this.wProj2) };
  }

  /** The image tokens for these features, or empty slots (zeros) when there is no picture yet */
  public imageTokens(features: Matrix | null): Matrix {
    return features ? this.project(features).tokens : MatrixMath.zeros(this.numPatches, this.config.dModel);
  }

  // ── Training ────────────────────────────────────────────────────────────

  /**
   * Gradient of the mean loss over a batch (each example's loss is the mean over its answer
   * words). In phase 0 the image slots are zeros, so the projector gets nothing.
   * `features[b]` are example b's vision features (computed once; the encoder is frozen).
   */
  public computeGradients(batch: VlmExample[], features: Matrix[], phase: VlmPhase): { loss: number; grads: Record<string, Matrix> } {
    const grads: Record<string, Matrix> = {};
    for (const [name, p] of Object.entries(this.getParameters())) grads[name] = MatrixMath.zeros(p.length, p[0].length);
    const T = MatrixMath.transpose;
    const mm = MatrixMath.matmul;
    let loss = 0;

    batch.forEach((ex, b) => {
      const proj = phase === 0 ? null : this.project(features[b]);
      const prefix = proj ? proj.tokens : this.imageTokens(null);
      const r = this.lm.computeGradients(ex.input, ex.target, { prefix, lossMask: ex.lossMask });
      loss += r.loss / batch.length;
      for (const [name, g] of Object.entries(r.grads)) {
        const acc = grads[`lm.${name}`];
        for (let i = 0; i < g.length; i++) for (let k = 0; k < g[i].length; k++) acc[i][k] += g[i][k] / batch.length;
      }
      if (!proj || !r.dPrefix) return;
      // Projector backward: tokens = hidden · W2, hidden = gelu(hiddenRaw), hiddenRaw = features · W1
      const dTokens = MatrixMath.scale(r.dPrefix, 1 / batch.length);
      MatrixMath.addInPlace(grads['proj.w2'], mm(T(proj.hidden), dTokens));
      const dHidden = mm(dTokens, T(this.wProj2));
      const dRaw = MatrixMath.geluBackward(proj.hiddenRaw, dHidden);
      MatrixMath.addInPlace(grads['proj.w1'], mm(T(features[b]), dRaw));
    });
    return { loss, grads };
  }

  /** Mean loss over a batch, forward only */
  public evaluateLoss(batch: VlmExample[], features: Matrix[], phase: VlmPhase): number {
    return batch.reduce((s, ex, b) => {
      const prefix = this.imageTokens(phase === 0 ? null : features[b]);
      return s + this.lm.evaluateLoss(ex.input, ex.target, { prefix, lossMask: ex.lossMask }).loss / batch.length;
    }, 0);
  }

  /** One step: backprop, clip the trained parameters' gradients to norm `maxGradNorm`, AdamW on them only */
  public trainStep(batch: VlmExample[], features: Matrix[], phase: VlmPhase, lr = 0.001, maxGradNorm = 1.0): VlmStepResult {
    const { loss, grads } = this.computeGradients(batch, features, phase);
    const trains = PHASE_TRAINS[phase];
    const trained = Object.keys(grads).filter(n => (n.startsWith('proj.') ? trains.projector : trains.lm));
    let sumSq = 0;
    for (const n of trained) for (const row of grads[n]) for (const v of row) sumSq += v * v;
    const gradNorm = Math.sqrt(sumSq);
    const clip = gradNorm > maxGradNorm ? maxGradNorm / gradNorm : 1;
    const params = this.getParameters();
    for (const n of trained) {
      const g = grads[n];
      if (clip !== 1) for (const row of g) for (let k = 0; k < row.length; k++) row[k] *= clip;
      this.optimizer.step(n, params[n], g, lr);
    }
    this.steps++;
    return { loss, gradNorm };
  }

  // ── Answering ───────────────────────────────────────────────────────────

  /**
   * Would greedy decoding produce exactly this answer? Checked in one forward pass ("teacher
   * forcing"): it would iff the most likely next word is the right one at every answer position.
   */
  public answersCorrectly(ex: VlmExample, imageTokens: Matrix): boolean {
    const P = imageTokens.length;
    const data = this.lm.inspectForwardPass(ex.input, ex.input.map(String), imageTokens);
    for (let i = 0; i < ex.target.length; i++) {
      if (ex.lossMask[i] && argmax(data.probabilities[P + i]) !== ex.target[i]) return false;
    }
    return true;
  }

  /** Answer a question greedily, word by word, until <end> (or `maxWords`) */
  public answer(imageTokens: Matrix, questionIds: number[], maxWords = 10): VlmAnswer {
    const ids = [...questionIds];
    const out: VlmAnswer = { words: [], ids: [], steps: [] };
    const room = this.lm.config.contextWindow - imageTokens.length;
    for (let k = 0; k < maxWords && ids.length < room; k++) {
      const data = this.lm.inspectForwardPass(ids, ids.map(id => VLM_VOCAB[id]), imageTokens);
      const next = argmax(data.probabilities[data.probabilities.length - 1]);
      out.steps.push(data);
      if (next === END_ID) break;
      out.ids.push(next);
      out.words.push(VLM_VOCAB[next]);
      ids.push(next);
    }
    return out;
  }

  /** The word embeddings closest (by cosine) to an image token: what the LM might "read" it as */
  public nearestWords(token: number[], k = 3): { word: string; sim: number }[] {
    const table = this.lm.getParameters().wTokenEmbed;
    const norm = (v: number[]) => Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    const nt = norm(token);
    return table
      .map((row, id) => ({ word: VLM_VOCAB[id], sim: row.reduce((s, x, d) => s + x * token[d], 0) / (nt * norm(row)) }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, k);
  }

  // ── Saving and copying ──────────────────────────────────────────────────

  public exportState(): VlmState {
    const weights: Record<string, Float64Array> = {};
    for (const [name, M] of Object.entries(this.getParameters())) weights[name] = Float64Array.from(M.flat());
    return { weights, optimizer: this.optimizer.exportState() };
  }

  /** Overwrite the projector and LM weights (and optimizer memory) in place; false if any shape differs */
  public importState(state: VlmState): boolean {
    const params = this.getParameters();
    const names = Object.keys(params);
    if (names.length !== Object.keys(state.weights).length) return false;
    for (const name of names) {
      const data = state.weights[name];
      if (!data || data.length !== params[name].length * params[name][0].length) return false;
    }
    for (const name of names) {
      const M = params[name];
      const data = state.weights[name];
      const cols = M[0].length;
      for (let r = 0; r < M.length; r++) for (let c = 0; c < cols; c++) M[r][c] = data[r * cols + c];
    }
    this.optimizer.importState(state.optimizer, params);
    return true;
  }

  /** An independent copy (sharing the same frozen vision encoder) */
  public clone(): MicroVlm {
    const copy = new MicroVlm(this.vision, this.config);
    copy.importState(this.exportState());
    copy.steps = this.steps;
    return copy;
  }
}
