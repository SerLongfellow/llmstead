import { LayerInspection } from '../../types';
import {
  BLOCK_WEIGHT_NAMES, BlockCache, BlockWeights, blockBackward, blockForward, randomBlockWeights
} from '../block';
import { Optimizer, OptimizerState } from '../optimizer';
import { Matrix, MatrixMath } from '../tensor';

/**
 * MicroClip: a tiny CLIP (Contrastive Language–Image Pre-training, OpenAI 2021).
 *
 * Two separate transformers ("towers") each turn their input into one vector of the same size:
 *
 *   picture → patches → image tower → mean of the outputs → projection → normalize → u
 *   caption → words   → text tower  → mean of the outputs → projection → normalize → v
 *
 * Neither tower ever sees the other's input. They're trained together on a batch of N
 * matching (picture, caption) pairs: every picture is scored against every caption,
 *
 *   S[i][j] = scale · (u_i · v_j)        (cosine similarity, times a learned temperature)
 *
 * and the loss asks each row (picture i: which of these N captions is mine?) and each column
 * (caption j: which picture is mine?) to pick the diagonal, as N-way classification with
 * cross-entropy. That is the "contrastive" part: a caption is learned as much from the N − 1
 * pictures it does *not* describe as from the one it does.
 *
 * Both towers use the shared pre-LN block (block.ts) without the causal mask: every patch can
 * look at every other patch, and every word at every other word.
 */

export type FalseNegatives = 'ignore' | 'multi-positive';

export interface ClipConfig {
  imageSize: number;    // pixels per side
  patchSize: number;    // pixels per patch side; (imageSize / patchSize)² patches
  dModel: number;       // width of both towers
  numHeads: number;
  imageLayers: number;
  textLayers: number;
  mlpRatio: number;
  dEmbed: number;       // size of the shared space both towers project into
  vocabSize: number;    // number of words the text tower knows
  maxWords: number;     // longest caption (number of text position embeddings)
  /**
   * When two pairs in a batch have the same caption, plain CLIP still calls the "other" one
   * wrong ('ignore'). 'multi-positive' counts every matching caption as right instead.
   */
  falseNegatives: FalseNegatives;
}

/** One training pair */
export interface ClipExample {
  patches: Matrix;   // [numPatches x patchDim]
  tokens: number[];  // caption word ids
  key: string;       // what the caption states (captionKey): same key = interchangeable captions
}

/** The learned numbers, as plain typed arrays (for the worker and for clone) */
export interface ClipState {
  weights: Record<string, Float64Array>;
  optimizer: OptimizerState;
}

/** The temperature can't exceed this (CLIP's own cap), or the softmax gets too sharp to train */
export const MAX_LOGIT_SCALE = 100;

interface Tower {
  blocks: BlockWeights[];
  wProj: Matrix;   // [dModel x dEmbed]
}

interface TowerPass {
  embedded: Matrix;          // tower input (before the blocks)
  layers: LayerInspection[];
  caches: BlockCache[];
  last: Matrix;              // last block's output (input to the final LayerNorm)
  finalNorm: Matrix;         // [seqLen x dModel]
  pooled: number[];          // mean of finalNorm's rows
  z: number[];               // pooled · W_proj, before normalizing
  norm: number;              // |z|
  unit: number[];            // z / |z|: the embedding
}

function towerForward(embedded: Matrix, tower: Tower, numHeads: number): TowerPass {
  const layers: LayerInspection[] = [];
  const caches: BlockCache[] = [];
  let x = embedded;
  tower.blocks.forEach((w, l) => {
    const { inspection, cache } = blockForward(x, w, numHeads, false, l);
    layers.push(inspection);
    caches.push(cache);
    x = inspection.blockOutput;
  });
  const finalNorm = MatrixMath.layerNorm(x);
  const pooled = meanRows(finalNorm);
  const z = vecMat(pooled, tower.wProj);
  const norm = Math.sqrt(z.reduce((s, v) => s + v * v, 0) + 1e-12);
  return { embedded, layers, caches, last: x, finalNorm, pooled, z, norm, unit: z.map(v => v / norm) };
}

/**
 * Backprop dUnit (∂L/∂unit embedding) through one tower, adding into grads[prefix + name].
 * Returns ∂L/∂(tower input).
 */
function towerBackward(
  pass: TowerPass, dUnit: number[], tower: Tower, numHeads: number,
  grads: Record<string, Matrix>, prefix: string
): Matrix {
  // unit = z / |z|  →  dz = (dUnit − unit·(unit · dUnit)) / |z|
  const dot = pass.unit.reduce((s, u, k) => s + u * dUnit[k], 0);
  const dz = dUnit.map((g, k) => (g - pass.unit[k] * dot) / pass.norm);

  // z = pooled · W_proj
  const gProj = grads[`${prefix}wProj`];
  for (let r = 0; r < pass.pooled.length; r++) {
    const row = gProj[r];
    const p = pass.pooled[r];
    for (let k = 0; k < dz.length; k++) row[k] += p * dz[k];
  }
  const dPooled = tower.wProj.map(row => row.reduce((s, w, k) => s + w * dz[k], 0));

  // pooled = mean of the rows: each row gets an equal share
  const n = pass.finalNorm.length;
  const dFinal = pass.finalNorm.map(() => dPooled.map(v => v / n));
  let dX = MatrixMath.layerNormBackward(pass.last, dFinal);

  for (let l = tower.blocks.length - 1; l >= 0; l--) {
    const back = blockBackward(dX, tower.blocks[l], pass.layers[l], pass.caches[l], numHeads);
    for (const name of BLOCK_WEIGHT_NAMES) MatrixMath.addInPlace(grads[`${prefix}${name}.${l}`], back.grads[name]);
    dX = back.dInput;
  }
  return dX;
}

function meanRows(M: Matrix): number[] {
  const out = new Array(M[0].length).fill(0);
  for (const row of M) for (let d = 0; d < row.length; d++) out[d] += row[d] / M.length;
  return out;
}

/** v · M for a vector v and matrix M [v.length x k] */
function vecMat(v: number[], M: Matrix): number[] {
  const out = new Array(M[0].length).fill(0);
  for (let r = 0; r < v.length; r++) {
    const row = M[r];
    const x = v[r];
    for (let k = 0; k < out.length; k++) out[k] += x * row[k];
  }
  return out;
}

const dotVec = (a: number[], b: number[]) => a.reduce((s, v, k) => s + v * b[k], 0);

/** What a batch's loss and similarity grid look like (also the Train tab's live matrix) */
export interface ClipBatchResult {
  loss: number;
  similarity: Matrix;   // cosine similarity u_i · v_j, [N x N]
  imageToText: number;  // fraction of rows whose best caption is a right one
  textToImage: number;  // fraction of columns whose best picture is a right one
}

export class MicroClip {
  public config: ClipConfig;
  public readonly numPatches: number;
  public readonly patchDim: number;
  /** Optimizer steps taken so far */
  public steps = 0;

  private wPatch: Matrix;      // [patchDim x dModel]: one patch's pixels → its first embedding
  private wImgPos: Matrix;     // [numPatches x dModel]: where in the grid the patch sits
  private image: Tower;
  private wTokenEmbed: Matrix; // [vocabSize x dModel]
  private wTxtPos: Matrix;     // [maxWords x dModel]
  private text: Tower;
  private logScale: Matrix;    // [1 x 1]: the temperature, stored as a log so it stays positive
  private optimizer = new Optimizer('adamw');

  constructor(config: ClipConfig) {
    if (config.imageSize % config.patchSize !== 0) throw new Error('imageSize must be a multiple of patchSize');
    if (config.dModel % config.numHeads !== 0) throw new Error('dModel must be divisible by numHeads');
    this.config = config;
    this.numPatches = (config.imageSize / config.patchSize) ** 2;
    this.patchDim = config.patchSize * config.patchSize * 3;

    const d = config.dModel;
    const scale = Math.sqrt(2.0 / d);
    const dMlp = d * config.mlpRatio;
    this.wPatch = MatrixMath.random(this.patchDim, d, Math.sqrt(2.0 / this.patchDim));
    this.wImgPos = MatrixMath.random(this.numPatches, d, scale);
    this.image = {
      blocks: Array.from({ length: config.imageLayers }, () => randomBlockWeights(d, dMlp, scale)),
      wProj: MatrixMath.random(d, config.dEmbed, Math.sqrt(1.0 / d)),
    };
    this.wTokenEmbed = MatrixMath.random(config.vocabSize, d, scale);
    this.wTxtPos = MatrixMath.random(config.maxWords, d, scale);
    this.text = {
      blocks: Array.from({ length: config.textLayers }, () => randomBlockWeights(d, dMlp, scale)),
      wProj: MatrixMath.random(d, config.dEmbed, Math.sqrt(1.0 / d)),
    };
    // CLIP starts at a temperature of 0.07, i.e. scale 1/0.07 ≈ 14
    this.logScale = [[Math.log(1 / 0.07)]];
  }

  /** Every trainable matrix by name ('img.*' image tower, 'txt.*' text tower); the live arrays */
  public getParameters(): Record<string, Matrix> {
    const params: Record<string, Matrix> = { 'img.wPatch': this.wPatch, 'img.wPos': this.wImgPos };
    this.image.blocks.forEach((b, l) => { for (const n of BLOCK_WEIGHT_NAMES) params[`img.${n}.${l}`] = b[n]; });
    params['img.wProj'] = this.image.wProj;
    params['txt.wTokenEmbed'] = this.wTokenEmbed;
    params['txt.wPos'] = this.wTxtPos;
    this.text.blocks.forEach((b, l) => { for (const n of BLOCK_WEIGHT_NAMES) params[`txt.${n}.${l}`] = b[n]; });
    params['txt.wProj'] = this.text.wProj;
    params.logScale = this.logScale;
    return params;
  }

  public getParameterCount(): number {
    return Object.values(this.getParameters()).reduce((s, M) => s + M.length * M[0].length, 0);
  }

  /** The current temperature multiplier (similarities are multiplied by this before the softmax) */
  public logitScale(): number {
    return Math.min(Math.exp(this.logScale[0][0]), MAX_LOGIT_SCALE);
  }

  /** Change the settings that don't affect the weights' shapes */
  public setFalseNegatives(mode: FalseNegatives): void {
    this.config = { ...this.config, falseNegatives: mode };
  }

  // ── The two towers ────────────────────────────────────────────────────────

  private imagePass(patches: Matrix): TowerPass {
    const embedded = MatrixMath.matmul(patches, this.wPatch);
    for (let p = 0; p < embedded.length; p++) {
      const row = embedded[p];
      const pos = this.wImgPos[p];
      for (let d = 0; d < row.length; d++) row[d] += pos[d];
    }
    return towerForward(embedded, this.image, this.config.numHeads);
  }

  private textPass(tokens: number[]): TowerPass {
    const ids = tokens.slice(0, this.config.maxWords);
    const embedded = ids.map((id, i) => this.wTokenEmbed[id].map((v, d) => v + this.wTxtPos[i][d]));
    return towerForward(embedded, this.text, this.config.numHeads);
  }

  /** Which captions in the batch count as right for each picture: rows sum to 1 */
  private targets(batch: ClipExample[]): Matrix {
    const n = batch.length;
    if (this.config.falseNegatives === 'ignore') {
      return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    }
    return batch.map(a => {
      const row = batch.map((b): number => (a.key === b.key ? 1 : 0));
      const count = row.reduce((s, v) => s + v, 0);
      return row.map(v => v / count);
    });
  }

  private forwardBatch(batch: ClipExample[]) {
    const imgs = batch.map(ex => this.imagePass(ex.patches));
    const txts = batch.map(ex => this.textPass(ex.tokens));
    const n = batch.length;
    const scale = this.logitScale();
    const G = imgs.map(a => txts.map(b => dotVec(a.unit, b.unit)));           // cosine similarities
    const S = MatrixMath.scale(G, scale);
    const rowP = MatrixMath.softmax(S, false);                                 // picture → which caption
    const colP = MatrixMath.transpose(MatrixMath.softmax(MatrixMath.transpose(S), false)); // caption → which picture
    const T = this.targets(batch);

    let rowLoss = 0;
    let colLoss = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (T[i][j] > 0) {
          rowLoss -= (T[i][j] * Math.log(Math.max(rowP[i][j], 1e-12))) / n;
          colLoss -= (T[i][j] * Math.log(Math.max(colP[i][j], 1e-12))) / n;
        }
      }
    }

    // Accuracy: does the best-scoring caption (or picture) state the right facts?
    let rowHits = 0;
    let colHits = 0;
    for (let i = 0; i < n; i++) {
      let best = 0;
      for (let j = 1; j < n; j++) if (G[i][j] > G[i][best]) best = j;
      if (batch[best].key === batch[i].key) rowHits++;
      let bestI = 0;
      for (let k = 1; k < n; k++) if (G[k][i] > G[bestI][i]) bestI = k;
      if (batch[bestI].key === batch[i].key) colHits++;
    }

    return {
      imgs, txts, G, rowP, colP, T, scale,
      result: { loss: (rowLoss + colLoss) / 2, similarity: G, imageToText: rowHits / n, textToImage: colHits / n } as ClipBatchResult,
    };
  }

  /** The batch's loss, accuracies and similarity grid, without training */
  public evaluateBatch(batch: ClipExample[]): ClipBatchResult {
    return this.forwardBatch(batch).result;
  }

  public evaluateLoss(batch: ClipExample[]): number {
    return this.forwardBatch(batch).result.loss;
  }

  /**
   * Gradient of the batch loss with respect to every trainable matrix.
   *
   * With L = ½(L_rows + L_cols) and each a mean cross-entropy over N rows (or columns):
   *   ∂L/∂S[i][j] = ½ · ((rowP[i][j] − T[i][j]) + (colP[i][j] − T[i][j])) / N
   * then S = scale·G, G[i][j] = u_i · v_j, and each u, v back through its tower.
   */
  public computeGradients(batch: ClipExample[]): ClipBatchResult & { grads: Record<string, Matrix> } {
    const grads: Record<string, Matrix> = {};
    for (const [name, p] of Object.entries(this.getParameters())) grads[name] = MatrixMath.zeros(p.length, p[0].length);

    const f = this.forwardBatch(batch);
    const n = batch.length;
    const dS = f.rowP.map((row, i) => row.map((p, j) => (p - f.T[i][j] + f.colP[i][j] - f.T[i][j]) / (2 * n)));

    // S = scale · G, scale = exp(logScale) (constant once it hits the cap)
    let dScale = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) dScale += dS[i][j] * f.G[i][j];
    if (Math.exp(this.logScale[0][0]) < MAX_LOGIT_SCALE) grads.logScale[0][0] = dScale * f.scale;
    const dG = MatrixMath.scale(dS, f.scale);

    const dim = this.config.dEmbed;
    const { numHeads } = this.config;
    for (let i = 0; i < n; i++) {
      // G[i][j] = u_i · v_j  →  ∂/∂u_i = Σ_j dG[i][j] v_j,  ∂/∂v_i = Σ_k dG[k][i] u_k
      const du = new Array(dim).fill(0);
      const dv = new Array(dim).fill(0);
      for (let j = 0; j < n; j++) {
        const vj = f.txts[j].unit;
        const uj = f.imgs[j].unit;
        for (let k = 0; k < dim; k++) {
          du[k] += dG[i][j] * vj[k];
          dv[k] += dG[j][i] * uj[k];
        }
      }

      // Image tower, then its input: embedded = patches · W_patch + W_pos
      const dImg = towerBackward(f.imgs[i], du, this.image, numHeads, grads, 'img.');
      MatrixMath.addInPlace(grads['img.wPatch'], MatrixMath.matmul(MatrixMath.transpose(batch[i].patches), dImg));
      MatrixMath.addInPlace(grads['img.wPos'], dImg);

      // Text tower, then its input: embedded_k = tokenEmbed[id_k] + W_pos[k]
      const dTxt = towerBackward(f.txts[i], dv, this.text, numHeads, grads, 'txt.');
      const ids = batch[i].tokens.slice(0, this.config.maxWords);
      const gTok = grads['txt.wTokenEmbed'];
      const gPos = grads['txt.wPos'];
      ids.forEach((id, k) => {
        for (let d = 0; d < dTxt[k].length; d++) {
          gTok[id][d] += dTxt[k][d];
          gPos[k][d] += dTxt[k][d];
        }
      });
    }
    return { ...f.result, grads };
  }

  /** One step: backprop, clip to global grad-norm `maxGradNorm`, AdamW update */
  public trainStep(batch: ClipExample[], lr = 0.001, maxGradNorm = 1.0): ClipBatchResult & { gradNorm: number } {
    const { grads, ...result } = this.computeGradients(batch);
    let sumSq = 0;
    for (const g of Object.values(grads)) for (const row of g) for (const v of row) sumSq += v * v;
    const gradNorm = Math.sqrt(sumSq);
    const clip = gradNorm > maxGradNorm ? maxGradNorm / gradNorm : 1;
    const params = this.getParameters();
    for (const [name, g] of Object.entries(grads)) {
      if (clip !== 1) for (const row of g) for (let k = 0; k < row.length; k++) row[k] *= clip;
      this.optimizer.step(name, params[name], g, lr);
    }
    this.steps++;
    return { ...result, gradNorm };
  }

  // ── Using the trained model ───────────────────────────────────────────────

  /** A picture's embedding (unit length) */
  public embedImage(patches: Matrix): number[] {
    return this.imagePass(patches).unit;
  }

  /** A caption's embedding (unit length); null for an empty caption */
  public embedText(tokens: number[]): number[] | null {
    return tokens.length ? this.textPass(tokens).unit : null;
  }

  /**
   * Everything the image tower computes for one picture, plus where in it a caption matches:
   * each patch's own output, projected and normalized like the whole picture's, dotted with
   * the caption's embedding. (A heuristic view: the model was only ever trained on the average.)
   */
  public inspectImage(patches: Matrix, textEmbedding: number[] | null = null) {
    const pass = this.imagePass(patches);
    const patchMatch = textEmbedding
      ? pass.finalNorm.map(row => {
          const z = vecMat(row, this.image.wProj);
          const norm = Math.sqrt(z.reduce((s, v) => s + v * v, 0) + 1e-12);
          return dotVec(z, textEmbedding) / norm;
        })
      : null;
    return { layers: pass.layers, embedding: pass.unit, patchMatch };
  }

  /** Everything the text tower computes for one caption */
  public inspectText(tokens: number[]) {
    const pass = this.textPass(tokens);
    return { layers: pass.layers, embedding: pass.unit };
  }

  // ── Saving and copying ────────────────────────────────────────────────────

  public exportState(): ClipState {
    const weights: Record<string, Float64Array> = {};
    for (const [name, M] of Object.entries(this.getParameters())) weights[name] = Float64Array.from(M.flat());
    return { weights, optimizer: this.optimizer.exportState() };
  }

  /** Overwrite the weights (and optimizer memory) in place; false (and no change) if any shape differs */
  public importState(state: ClipState): boolean {
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
      for (let r = 0; r < M.length; r++) {
        const row = M[r];
        for (let c = 0; c < cols; c++) row[c] = data[r * cols + c];
      }
    }
    this.optimizer.importState(state.optimizer, params);
    return true;
  }

  /** An independent copy with the same weights (e.g. to keep the untrained model as a baseline) */
  public clone(): MicroClip {
    const copy = new MicroClip(this.config);
    copy.importState(this.exportState());
    copy.steps = this.steps;
    return copy;
  }
}
