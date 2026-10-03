import { LayerInspection } from '../../types';
import {
  BLOCK_WEIGHT_NAMES, BlockCache, BlockWeights, blockBackward, blockForward, randomBlockWeights
} from '../block';
import { Optimizer } from '../optimizer';
import { Matrix, MatrixMath } from '../tensor';

/**
 * MicroJepa: a tiny I-JEPA (Image Joint-Embedding Predictive Architecture).
 *
 * An image is cut into patches (see shapes.ts). Some patches are hidden (the *targets*);
 * the rest are the *context*. Three transformers take part:
 *
 *   context patches ─► context encoder ─► predictor (+ a mask token per hidden position) ─► predicted vectors
 *   all patches     ─► target encoder  ─► the vectors at the hidden positions ─────────────► (compare)
 *
 *   loss = mean squared distance between predicted and actual target vectors
 *
 * Unlike the GPT, nothing is ever predicted in pixel (or token) space: the model only has to
 * guess the hidden patches' *embeddings*, so it is free to ignore unpredictable detail (noise).
 *
 * The catch is that the targets are produced by a network that is also learning. If gradients
 * could reach both sides, the easiest way to zero the loss would be to map every image to the
 * same vector ("collapse"). I-JEPA prevents that with two tricks:
 *   - stop-gradient: the loss never sends gradients into the target encoder;
 *   - EMA: the target encoder is a slowly moving average of the context encoder,
 *     θ_target ← m·θ_target + (1 − m)·θ_context, with m close to 1.
 * `ablation` switches these off so you can watch what happens.
 *
 * Even with both tricks, a small JEPA can settle into *dimensional* collapse: the embeddings
 * stop being identical but only use a couple of directions (say, colour and "is this background").
 * A second, independent defence is to pay the encoder to keep its outputs spread out:
 * VICReg-style variance and covariance terms on a batch of image embeddings (`varWeight`, `covWeight`).
 *
 * All three transformers use the shared post-LN block (block.ts) with no causal mask:
 * every patch may attend to every other.
 */

/**
 * 'none'        the real recipe (stop-gradient + EMA target encoder)
 * 'no-ema'      targets come from the current context encoder, still with stop-gradient
 * 'no-stopgrad' targets come from the current context encoder AND the loss trains it through them
 */
export type JepaAblation = 'none' | 'no-ema' | 'no-stopgrad';

export interface JepaConfig {
  imageSize: number;   // pixels per side
  patchSize: number;   // pixels per patch side; (imageSize / patchSize)² patches
  dModel: number;      // encoder width (the size of each patch's embedding)
  numHeads: number;
  numLayers: number;
  mlpRatio: number;
  predDim: number;     // predictor width (narrower than the encoder, as in I-JEPA)
  predHeads: number;
  predLayers: number;
  numTargets: number;  // hidden rectangular blocks per image
  emaStart: number;    // EMA momentum m at step 0 …
  emaEnd: number;      // … rising linearly to this
  emaSteps: number;    // … over this many steps
  ablation: JepaAblation;
  varWeight: number;   // weight of the variance term (0 = off): each embedding dimension should vary across images
  covWeight: number;   // weight of the covariance term (0 = off): dimensions should not all say the same thing
}

export const DEFAULT_JEPA_CONFIG: JepaConfig = {
  imageSize: 16,
  patchSize: 4,
  dModel: 32,
  numHeads: 4,
  numLayers: 2,
  mlpRatio: 2,
  predDim: 16,
  predHeads: 2,
  predLayers: 2,
  numTargets: 2,
  emaStart: 0.996,
  emaEnd: 1.0,
  emaSteps: 5000,
  ablation: 'none',
  varWeight: 0,
  covWeight: 0,
};

/** Which patches (by index into the patch grid) the context encoder sees, and which are hidden */
export interface JepaMask {
  context: number[];
  targets: number[];
}

/**
 * I-JEPA-style multi-block masking: hide `numTargets` random rectangles (1 to grid/2 patches per
 * side, possibly overlapping); everything else is context. Redraws if under half the image
 * would be left as context.
 */
export function sampleMask(rand: () => number, grid: number, numTargets: number): JepaMask {
  const maxSide = Math.max(1, Math.round(grid / 2));
  for (;;) {
    const hidden = new Set<number>();
    for (let t = 0; t < numTargets; t++) {
      const h = 1 + Math.floor(rand() * maxSide);
      const w = 1 + Math.floor(rand() * maxSide);
      const top = Math.floor(rand() * (grid - h + 1));
      const left = Math.floor(rand() * (grid - w + 1));
      for (let y = top; y < top + h; y++) for (let x = left; x < left + w; x++) hidden.add(y * grid + x);
    }
    const all = Array.from({ length: grid * grid }, (_, i) => i);
    const context = all.filter(i => !hidden.has(i));
    if (context.length >= (grid * grid) / 2) return { context, targets: all.filter(i => hidden.has(i)) };
  }
}

/** Patch embedding + position embeddings + transformer blocks */
interface Encoder {
  wPatch: Matrix;        // [patchDim x dModel]: one patch's pixels → its first embedding
  wPos: Matrix;          // [numPatches x dModel]: where in the grid the patch sits
  blocks: BlockWeights[];
}

/** Narrow transformer that guesses the hidden patches' embeddings from the context's */
interface Predictor {
  wIn: Matrix;           // [dModel x predDim]: context embeddings into the predictor's width
  wPos: Matrix;          // [numPatches x predDim]: the predictor's own position embeddings
  maskToken: Matrix;     // [1 x predDim]: one learned vector standing in for every hidden patch
  blocks: BlockWeights[];
  wOut: Matrix;          // [predDim x dModel]: back to the encoder's width, to compare with targets
}

interface EncoderPass {
  idx: number[];
  patchesIn: Matrix;     // the selected patches' pixels
  embedded: Matrix;      // patch embedding + position, before the blocks
  layers: LayerInspection[];
  caches: BlockCache[];
  out: Matrix;           // [idx.length x dModel]
}

interface PredictorPass {
  ctxIn: Matrix;         // the context encoder's output (input to wIn)
  start: Matrix;         // [context rows; mask-token rows] entering the blocks
  layers: LayerInspection[];
  caches: BlockCache[];
  hiddenOut: Matrix;     // the blocks' output at the mask-token rows
  preds: Matrix;         // [targets x dModel]
}

/** Everything one image's forward pass computes, for the visualizers */
export interface JepaInspection {
  mask: JepaMask;
  context: { embedded: Matrix; layers: LayerInspection[]; out: Matrix };
  predictor: { start: Matrix; layers: LayerInspection[]; preds: Matrix };
  target: { embedded: Matrix; layers: LayerInspection[]; out: Matrix }; // all patches
  targets: Matrix;       // target.out at the hidden patches
  errors: number[];      // squared distance per hidden patch
  loss: number;
}

function encoderForward(enc: Encoder, patches: Matrix, idx: number[], numHeads: number): EncoderPass {
  const patchesIn = idx.map(i => patches[i]);
  const embedded = MatrixMath.matmul(patchesIn, enc.wPatch);
  for (let r = 0; r < idx.length; r++) {
    const row = embedded[r];
    const pos = enc.wPos[idx[r]];
    for (let d = 0; d < row.length; d++) row[d] += pos[d];
  }
  const layers: LayerInspection[] = [];
  const caches: BlockCache[] = [];
  let x = embedded;
  enc.blocks.forEach((w, l) => {
    const { inspection, cache } = blockForward(x, w, numHeads, false, l);
    layers.push(inspection);
    caches.push(cache);
    x = inspection.norm2Output;
  });
  return { idx, patchesIn, embedded, layers, caches, out: x };
}

/** Backprop dOut (∂L/∂encoder output) through the encoder, *adding* into grads[prefix + name] */
function encoderBackward(
  enc: Encoder, pass: EncoderPass, dOut: Matrix, numHeads: number,
  grads: Record<string, Matrix>, prefix: string
): void {
  let dX = dOut;
  for (let l = enc.blocks.length - 1; l >= 0; l--) {
    const back = blockBackward(dX, enc.blocks[l], pass.layers[l], pass.caches[l], numHeads);
    for (const name of BLOCK_WEIGHT_NAMES) MatrixMath.addInPlace(grads[`${prefix}${name}.${l}`], back.grads[name]);
    dX = back.dInput;
  }
  // embedded = patchesIn · W_patch + W_pos[idx]
  MatrixMath.addInPlace(grads[`${prefix}wPatch`], MatrixMath.matmul(MatrixMath.transpose(pass.patchesIn), dX));
  const gPos = grads[`${prefix}wPos`];
  for (let r = 0; r < pass.idx.length; r++) {
    const g = gPos[pass.idx[r]];
    for (let d = 0; d < g.length; d++) g[d] += dX[r][d];
  }
}

function predictorForward(pred: Predictor, ctxOut: Matrix, mask: JepaMask, numHeads: number): PredictorPass {
  // Context rows: the context encoder's embeddings, projected, plus their position
  const start = MatrixMath.matmul(ctxOut, pred.wIn);
  mask.context.forEach((p, r) => {
    const pos = pred.wPos[p];
    for (let d = 0; d < pos.length; d++) start[r][d] += pos[d];
  });
  // Hidden rows: the same mask token everywhere; only the position says which patch to guess
  for (const p of mask.targets) start.push(pred.maskToken[0].map((v, d) => v + pred.wPos[p][d]));

  const layers: LayerInspection[] = [];
  const caches: BlockCache[] = [];
  let x = start;
  pred.blocks.forEach((w, l) => {
    const { inspection, cache } = blockForward(x, w, numHeads, false, l);
    layers.push(inspection);
    caches.push(cache);
    x = inspection.norm2Output;
  });
  const hiddenOut = x.slice(mask.context.length);
  return { ctxIn: ctxOut, start, layers, caches, hiddenOut, preds: MatrixMath.matmul(hiddenOut, pred.wOut) };
}

/** Backprop dPreds through the predictor (adding into grads['pred.*']); returns ∂L/∂(context encoder output) */
function predictorBackward(
  pred: Predictor, pass: PredictorPass, mask: JepaMask, dPreds: Matrix, numHeads: number,
  grads: Record<string, Matrix>
): Matrix {
  const T = MatrixMath.transpose;
  const mm = MatrixMath.matmul;
  const nc = mask.context.length;

  // preds = hiddenOut · W_out; the context rows' outputs aren't used, so their gradient is 0
  MatrixMath.addInPlace(grads['pred.wOut'], mm(T(pass.hiddenOut), dPreds));
  const dHidden = mm(dPreds, T(pred.wOut));
  let dX: Matrix = [...MatrixMath.zeros(nc, pred.wIn[0].length), ...dHidden];

  for (let l = pred.blocks.length - 1; l >= 0; l--) {
    const back = blockBackward(dX, pred.blocks[l], pass.layers[l], pass.caches[l], numHeads);
    for (const name of BLOCK_WEIGHT_NAMES) MatrixMath.addInPlace(grads[`pred.${name}.${l}`], back.grads[name]);
    dX = back.dInput;
  }

  // Context rows: start = ctxOut · W_in + W_pos[p]
  const dCtxRows = dX.slice(0, nc);
  MatrixMath.addInPlace(grads['pred.wIn'], mm(T(pass.ctxIn), dCtxRows));
  // Hidden rows: start = maskToken + W_pos[p]
  const gPos = grads['pred.wPos'];
  const gMask = grads['pred.maskToken'][0];
  [...mask.context, ...mask.targets].forEach((p, r) => {
    for (let d = 0; d < gMask.length; d++) {
      gPos[p][d] += dX[r][d];
      if (r >= nc) gMask[d] += dX[r][d];
    }
  });
  return mm(dCtxRows, T(pred.wIn));
}

function randomEncoder(c: JepaConfig): Encoder {
  const patchDim = c.patchSize * c.patchSize * 3;
  const numPatches = (c.imageSize / c.patchSize) ** 2;
  const scale = Math.sqrt(2.0 / c.dModel);
  return {
    wPatch: MatrixMath.random(patchDim, c.dModel, Math.sqrt(2.0 / patchDim)),
    wPos: MatrixMath.random(numPatches, c.dModel, scale),
    blocks: Array.from({ length: c.numLayers }, () => randomBlockWeights(c.dModel, c.dModel * c.mlpRatio, scale)),
  };
}

function encoderMatrices(enc: Encoder): Record<string, Matrix> {
  const out: Record<string, Matrix> = { wPatch: enc.wPatch, wPos: enc.wPos };
  enc.blocks.forEach((b, l) => { for (const name of BLOCK_WEIGHT_NAMES) out[`${name}.${l}`] = b[name]; });
  return out;
}

const copyMatrix = (M: Matrix): Matrix => M.map(row => row.slice());

function cloneBlock(b: BlockWeights): BlockWeights {
  return {
    wQ: copyMatrix(b.wQ), wK: copyMatrix(b.wK), wV: copyMatrix(b.wV), wO: copyMatrix(b.wO),
    wMlp1: copyMatrix(b.wMlp1), wMlp2: copyMatrix(b.wMlp2),
  };
}

function cloneEncoder(enc: Encoder): Encoder {
  return { wPatch: copyMatrix(enc.wPatch), wPos: copyMatrix(enc.wPos), blocks: enc.blocks.map(cloneBlock) };
}

/**
 * VICReg's anti-collapse terms on a batch of n embeddings Z [n x d] (one row per image):
 *
 *   variance   (1/d) Σ_j max(0, 1 − std_j)        each dimension's std across images should reach 1
 *   covariance (1/d) Σ_{i≠j} C_ij²                 off-diagonal covariances should be 0 (decorrelated)
 *
 * with c = Z − mean(Z), C = cᵀc/(n−1), std_j = √(C_jj + ε). Returns the weighted sum and ∂/∂Z.
 * (Centering needs no extra term in the gradient: every column of c, and so of ∂/∂c, sums to 0.)
 */
export function varianceCovariance(Z: Matrix, varWeight: number, covWeight: number): { loss: number; dZ: Matrix } {
  const n = Z.length;
  const d = Z[0].length;
  const mean = new Array(d).fill(0);
  for (const z of Z) for (let j = 0; j < d; j++) mean[j] += z[j] / n;
  const c = Z.map(z => z.map((v, j) => v - mean[j]));
  const C = MatrixMath.scale(MatrixMath.matmul(MatrixMath.transpose(c), c), 1 / (n - 1));
  const std = C.map((row, j) => Math.sqrt(row[j] + 1e-4));

  let varLoss = 0;
  let covLoss = 0;
  for (let i = 0; i < d; i++) {
    varLoss += Math.max(0, 1 - std[i]) / d;
    for (let j = 0; j < d; j++) if (i !== j) covLoss += (C[i][j] * C[i][j]) / d;
  }

  // ∂cov/∂C_ij = 2·C_ij/d off the diagonal; through C = cᵀc/(n−1): ∂/∂c = c·(G + Gᵀ)/(n−1) = 2·c·G/(n−1)
  const G = C.map((row, i) => row.map((v, j) => (i === j ? 0 : (2 * v) / d)));
  const cG = MatrixMath.matmul(c, G);
  const dZ = c.map((row, b) =>
    row.map((cv, j) => {
      // ∂std_j/∂c_bj = c_bj / ((n−1)·std_j); the hinge only pushes while std_j < 1
      const dVar = std[j] < 1 ? -cv / (d * (n - 1) * std[j]) : 0;
      return varWeight * dVar + covWeight * (2 * cG[b][j]) / (n - 1);
    })
  );
  return { loss: varWeight * varLoss + covWeight * covLoss, dZ };
}

/** Mean of each column: one image's patch embeddings → one vector */
function meanRows(M: Matrix): number[] {
  const out = new Array(M[0].length).fill(0);
  for (const row of M) for (let d = 0; d < row.length; d++) out[d] += row[d] / M.length;
  return out;
}

function meanSquaredError(preds: Matrix, targets: Matrix): { loss: number; errors: number[] } {
  const errors = preds.map((row, j) => row.reduce((s, v, d) => s + (v - targets[j][d]) ** 2, 0));
  const count = preds.length * preds[0].length;
  return { loss: errors.reduce((s, e) => s + e, 0) / count, errors };
}

/** The training-recipe settings, which (unlike the architecture) can change mid-training */
export type JepaRecipe = Pick<JepaConfig, 'ablation' | 'varWeight' | 'covWeight'>;

export class MicroJepa {
  /** Replaced (never mutated) by setRecipe; the architecture fields never change */
  public config: JepaConfig;
  public readonly grid: number;          // patches per side
  public readonly numPatches: number;
  /** Optimizer steps taken so far (drives the EMA schedule) */
  public steps = 0;

  private context: Encoder;
  private target: Encoder;               // EMA copy of `context`; never receives gradients
  private predictor: Predictor;
  private optimizer = new Optimizer('adamw');

  constructor(config: JepaConfig) {
    if (config.imageSize % config.patchSize !== 0) throw new Error('imageSize must be a multiple of patchSize');
    if (config.dModel % config.numHeads !== 0) throw new Error('dModel must be divisible by numHeads');
    if (config.predDim % config.predHeads !== 0) throw new Error('predDim must be divisible by predHeads');
    this.config = config;
    this.grid = config.imageSize / config.patchSize;
    this.numPatches = this.grid * this.grid;

    this.context = randomEncoder(config);
    this.target = cloneEncoder(this.context);
    const p = config.predDim;
    const scale = Math.sqrt(2.0 / p);
    this.predictor = {
      wIn: MatrixMath.random(config.dModel, p, Math.sqrt(2.0 / config.dModel)),
      wPos: MatrixMath.random(this.numPatches, p, scale),
      maskToken: MatrixMath.random(1, p, scale),
      blocks: Array.from({ length: config.predLayers }, () => randomBlockWeights(p, p * config.mlpRatio, scale)),
      wOut: MatrixMath.random(p, config.dModel, scale),
    };
  }

  /** Every trainable matrix by name ('enc.*' context encoder, 'pred.*' predictor); live arrays */
  public getParameters(): Record<string, Matrix> {
    const params: Record<string, Matrix> = {};
    for (const [name, M] of Object.entries(encoderMatrices(this.context))) params[`enc.${name}`] = M;
    const pr = this.predictor;
    params['pred.wIn'] = pr.wIn;
    params['pred.wPos'] = pr.wPos;
    params['pred.maskToken'] = pr.maskToken;
    pr.blocks.forEach((b, l) => { for (const name of BLOCK_WEIGHT_NAMES) params[`pred.${name}.${l}`] = b[name]; });
    params['pred.wOut'] = pr.wOut;
    return params;
  }

  public getParameterCount(): number {
    return Object.values(this.getParameters()).reduce((s, M) => s + M.length * M[0].length, 0);
  }

  /**
   * Switch the anti-collapse recipe without touching the weights, e.g. remove stop-gradient
   * halfway through training to watch the encoder collapse. Under the ablations the EMA copy
   * isn't updated, so returning to the real recipe restarts it from the current encoder.
   */
  public setRecipe(recipe: JepaRecipe): void {
    if (recipe.ablation === 'none' && this.config.ablation !== 'none') this.target = cloneEncoder(this.context);
    this.config = { ...this.config, ...recipe };
  }

  /** An independent copy with the same weights and step count (e.g. to keep the untrained encoder as a baseline) */
  public clone(): MicroJepa {
    const copy = new MicroJepa(this.config);
    copy.context = cloneEncoder(this.context);
    copy.target = cloneEncoder(this.target);
    const pr = this.predictor;
    copy.predictor = {
      wIn: copyMatrix(pr.wIn),
      wPos: copyMatrix(pr.wPos),
      maskToken: copyMatrix(pr.maskToken),
      blocks: pr.blocks.map(cloneBlock),
      wOut: copyMatrix(pr.wOut),
    };
    copy.steps = this.steps;
    return copy;
  }

  /** EMA momentum m for the current step */
  public momentum(): number {
    const { emaStart, emaEnd, emaSteps } = this.config;
    return emaStart + (emaEnd - emaStart) * Math.min(1, this.steps / Math.max(1, emaSteps));
  }

  /** The encoder that produces targets: the EMA copy, or under an ablation the context encoder itself */
  private targetEncoder(): Encoder {
    return this.config.ablation === 'none' ? this.target : this.context;
  }

  private allPatches(): number[] {
    return Array.from({ length: this.numPatches }, (_, i) => i);
  }

  private forwardOne(patches: Matrix, mask: JepaMask) {
    const c = this.config;
    const ctx = encoderForward(this.context, patches, mask.context, c.numHeads);
    const pred = predictorForward(this.predictor, ctx.out, mask, c.predHeads);
    const tgt = encoderForward(this.targetEncoder(), patches, this.allPatches(), c.numHeads);
    const targets = mask.targets.map(i => tgt.out[i]);
    return { ctx, pred, tgt, targets, ...meanSquaredError(pred.preds, targets) };
  }

  private useRegularizer(batchSize: number): boolean {
    return batchSize > 1 && (this.config.varWeight > 0 || this.config.covWeight > 0);
  }

  /** Mean loss over a batch (forward only), including the variance/covariance term if enabled */
  public evaluateLoss(batch: Matrix[], masks: JepaMask[]): number {
    const fwd = batch.map((patches, b) => this.forwardOne(patches, masks[b]));
    let loss = fwd.reduce((s, f) => s + f.loss, 0) / batch.length;
    if (this.useRegularizer(batch.length)) {
      loss += varianceCovariance(fwd.map(f => meanRows(f.ctx.out)), this.config.varWeight, this.config.covWeight).loss;
    }
    return loss;
  }

  /**
   * Gradient of the mean batch loss with respect to every trainable matrix. With stop-gradient
   * (the default), the targets are treated as constants; under 'no-stopgrad' the gradient also
   * flows back through the target branch into the (shared) context encoder. The optional
   * variance/covariance term looks at the whole batch at once (one vector per image: the mean of
   * its context-encoder outputs), so every image's forward pass runs before any backward pass.
   */
  public computeGradients(
    batch: Matrix[], masks: JepaMask[]
  ): { loss: number; predLoss: number; regLoss: number; grads: Record<string, Matrix> } {
    const c = this.config;
    const grads: Record<string, Matrix> = {};
    for (const [name, p] of Object.entries(this.getParameters())) grads[name] = MatrixMath.zeros(p.length, p[0].length);

    const fwd = batch.map((patches, b) => this.forwardOne(patches, masks[b]));
    const predLoss = fwd.reduce((s, f) => s + f.loss, 0) / batch.length;
    const reg = this.useRegularizer(batch.length)
      ? varianceCovariance(fwd.map(f => meanRows(f.ctx.out)), c.varWeight, c.covWeight)
      : null;

    fwd.forEach((f, b) => {
      const mask = masks[b];
      // L = (1/B) · mean over hidden patches and dims of (pred − target)²  →  ∂L/∂pred = 2(pred − target)/(B·n·d)
      const norm = 2 / (batch.length * f.targets.length * c.dModel);
      const dPreds = f.pred.preds.map((row, j) => row.map((v, d) => (v - f.targets[j][d]) * norm));

      const dCtxOut = predictorBackward(this.predictor, f.pred, mask, dPreds, c.predHeads, grads);
      if (reg) {
        // z_b = mean of the context rows, so each row gets an equal share of ∂L/∂z_b
        const share = reg.dZ[b].map(v => v / dCtxOut.length);
        for (const row of dCtxOut) for (let d = 0; d < row.length; d++) row[d] += share[d];
      }
      encoderBackward(this.context, f.ctx, dCtxOut, c.numHeads, grads, 'enc.');

      if (c.ablation === 'no-stopgrad') {
        // ∂L/∂target = −∂L/∂pred at the hidden patches, 0 elsewhere
        const dTgt = MatrixMath.zeros(this.numPatches, c.dModel);
        mask.targets.forEach((p, j) => { dTgt[p] = dPreds[j].map(v => -v); });
        encoderBackward(this.context, f.tgt, dTgt, c.numHeads, grads, 'enc.');
      }
    });
    const regLoss = reg ? reg.loss : 0;
    return { loss: predLoss + regLoss, predLoss, regLoss, grads };
  }

  /**
   * One training step: backprop, clip to global grad-norm `maxGradNorm`, AdamW update,
   * then move the target encoder toward the context encoder (EMA).
   */
  public trainStep(batch: Matrix[], masks: JepaMask[], lr = 0.001, maxGradNorm = 1.0) {
    const { loss, predLoss, regLoss, grads } = this.computeGradients(batch, masks);

    let sumSq = 0;
    for (const g of Object.values(grads)) for (const row of g) for (const v of row) sumSq += v * v;
    const gradNorm = Math.sqrt(sumSq);
    const clip = gradNorm > maxGradNorm ? maxGradNorm / gradNorm : 1;

    const params = this.getParameters();
    for (const [name, g] of Object.entries(grads)) {
      if (clip !== 1) for (const row of g) for (let k = 0; k < row.length; k++) row[k] *= clip;
      this.optimizer.step(name, params[name], g, lr);
    }

    const m = this.momentum();
    if (this.config.ablation === 'none') {
      const online = encoderMatrices(this.context);
      for (const [name, T] of Object.entries(encoderMatrices(this.target))) {
        const C = online[name];
        for (let r = 0; r < T.length; r++) {
          const t = T[r];
          const cRow = C[r];
          for (let k = 0; k < t.length; k++) t[k] = m * t[k] + (1 - m) * cRow[k];
        }
      }
    }
    this.steps++;
    return { loss, predLoss, regLoss, gradNorm, momentum: m };
  }

  /** Embed every patch of an image: [numPatches x dModel] */
  public encode(patches: Matrix, which: 'target' | 'context' = 'target'): Matrix {
    const enc = which === 'target' ? this.targetEncoder() : this.context;
    return encoderForward(enc, patches, this.allPatches(), this.config.numHeads).out;
  }

  /** One image's full forward pass with every intermediate, for the visualizers */
  public inspect(patches: Matrix, mask: JepaMask): JepaInspection {
    const f = this.forwardOne(patches, mask);
    return {
      mask,
      context: { embedded: f.ctx.embedded, layers: f.ctx.layers, out: f.ctx.out },
      predictor: { start: f.pred.start, layers: f.pred.layers, preds: f.pred.preds },
      target: { embedded: f.tgt.embedded, layers: f.tgt.layers, out: f.tgt.out },
      targets: f.targets,
      errors: f.errors,
      loss: f.loss,
    };
  }
}
