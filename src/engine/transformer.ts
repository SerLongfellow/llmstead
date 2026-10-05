import { LayerInspection, StepInspectionData, TransformerConfig } from '../types';
import { BLOCK_WEIGHT_NAMES, BlockCache, BlockWeights, blockBackward, blockForward } from './block';
import { Matrix, MatrixMath } from './tensor';
import { Optimizer, OptimizerState, OptimizerType } from './optimizer';

/** A trained model's learned numbers (see MicroTransformer.exportState) */
export interface ModelState {
  weights: Record<string, Float64Array>;
  optimizer: OptimizerState;
}

/**
 * Extra inputs for a vision-language model (both optional; without them a pass is the plain GPT's):
 * `prefix` rows sit in front of the tokens (see forward), and `lossMask[i]` says whether text
 * position i's prediction counts toward the loss (e.g. only the answer to a question).
 */
export interface PrefixOptions {
  prefix?: Matrix;
  lossMask?: boolean[];
}

/** Settings that change the shape of the weight matrices. Changing any of these requires a fresh model. */
export type ArchitectureConfig = Pick<
  TransformerConfig,
  'vocabSize' | 'contextWindow' | 'dModel' | 'numHeads' | 'numLayers' | 'mlpRatio'
>;

/**
 * Architecture: PRE-LayerNorm, the GPT-2 layout. Each half of a block reads a normalized copy
 * of its input, and its output is added back onto the residual stream, which is never
 * normalized itself until the final LayerNorm before the output head:
 *
 *   x ─┬─► LayerNorm ─► Attention ─► (+) ─┬─► LayerNorm ─► MLP ─► (+) ─► next block
 *      └──────────────────────────────┘   └────────────────────────┘
 *
 * The original Transformer was POST-LayerNorm (residual add, then normalize the sum), but
 * pre-LN trains more stably in deep stacks and is what GPT-2 and nearly every later LLM use.
 * Matching GPT-2 exactly is also what lets the GGUF export (gguf.ts) hand the weights to
 * llama.cpp and Ollama, which run GPT-2 with their own code. GPT-2 also has biases and a
 * learned LayerNorm scale and shift; this model leaves them out, and the export writes them
 * as zeros and ones, which changes nothing.
 *
 * The block itself (attention + MLP, forward and backward) lives in block.ts.
 */

export class MicroTransformer {
  public config: TransformerConfig;
  
  // Weight matrices
  private wTokenEmbed: Matrix; // [vocabSize x dModel]
  private wPosEmbed: Matrix;   // [contextWindow x dModel]
  
  // Layer weights [numLayers]
  private wQ: Matrix[]; // [dModel x dModel]
  private wK: Matrix[]; // [dModel x dModel]
  private wV: Matrix[]; // [dModel x dModel]
  private wO: Matrix[]; // [dModel x dModel]

  // MLP Layer weights
  private wMlp1: Matrix[]; // [dModel x (dModel * mlpRatio)]
  private wMlp2: Matrix[]; // [(dModel * mlpRatio) x dModel]

  // Output Head Unembedding Matrix
  private wHead: Matrix; // [dModel x vocabSize]

  private optimizer: Optimizer;

  constructor(config: TransformerConfig) {
    if (config.dModel % config.numHeads !== 0) {
      throw new Error(
        `d_model (${config.dModel}) must be divisible by numHeads (${config.numHeads}); ` +
        `otherwise d_model mod numHeads dimensions would be silently dropped when splitting into heads.`
      );
    }
    this.config = config;
    this.optimizer = new Optimizer(config.optimizer);
    
    // Scale initialization based on dModel dimension
    const scale = Math.sqrt(2.0 / config.dModel);
    const dMlp = config.dModel * config.mlpRatio;

    this.wTokenEmbed = MatrixMath.random(config.vocabSize, config.dModel, scale);
    this.wPosEmbed = MatrixMath.random(config.contextWindow, config.dModel, scale);
    this.wHead = MatrixMath.random(config.dModel, config.vocabSize, scale);

    this.wQ = [];
    this.wK = [];
    this.wV = [];
    this.wO = [];
    this.wMlp1 = [];
    this.wMlp2 = [];

    for (let l = 0; l < config.numLayers; l++) {
      this.wQ.push(MatrixMath.random(config.dModel, config.dModel, scale));
      this.wK.push(MatrixMath.random(config.dModel, config.dModel, scale));
      this.wV.push(MatrixMath.random(config.dModel, config.dModel, scale));
      this.wO.push(MatrixMath.random(config.dModel, config.dModel, scale));

      this.wMlp1.push(MatrixMath.random(config.dModel, dMlp, scale));
      this.wMlp2.push(MatrixMath.random(dMlp, config.dModel, scale));
    }
  }

  /** Swap optimizer (resets Adam moment estimates but keeps the learned weights) */
  public setOptimizer(type: OptimizerType): void {
    if (this.optimizer.type === type) return;
    this.optimizer = new Optimizer(type);
    this.config = { ...this.config, optimizer: type };
  }

  public getOptimizerType(): OptimizerType {
    return this.optimizer.type;
  }

  /** Calculate total trainable parameters */
  public getParameterCount(): number {
    const { vocabSize, contextWindow, dModel, numLayers, mlpRatio } = this.config;
    const dMlp = dModel * mlpRatio;

    const tokenEmbedParams = vocabSize * dModel;
    const posEmbedParams = contextWindow * dModel;
    
    // Per layer: Q, K, V, O (4 * dModel^2) + MLP1 (dModel * dMlp) + MLP2 (dMlp * dModel)
    const perLayerParams = 4 * dModel * dModel + 2 * dModel * dMlp;
    const headParams = dModel * vocabSize;

    return tokenEmbedParams + posEmbedParams + numLayers * perLayerParams + headParams;
  }

  /** Layer l's weights, as the shared block code expects them (the live arrays) */
  private blockWeights(l: number): BlockWeights {
    return { wQ: this.wQ[l], wK: this.wK[l], wV: this.wV[l], wO: this.wO[l], wMlp1: this.wMlp1[l], wMlp2: this.wMlp2[l] };
  }

  /**
   * Run a complete forward pass and capture all internal states for visualization.
   * `prefix`: see forward.
   */
  public inspectForwardPass(tokens: number[], tokenStrings: string[], prefix?: Matrix): StepInspectionData {
    return this.forward(tokens, tokenStrings, prefix).data;
  }

  /**
   * Forward pass that also returns the extra per-layer values the backward pass needs.
   *
   * `prefix` (optional, [P x dModel]): rows that take the first P positions instead of token
   * embeddings, with the tokens following from position P. This is how a vision-language model
   * feeds a picture to its language model: each image patch, projected to dModel, sits where a
   * token embedding would. Without it, the pass is exactly the plain GPT's.
   */
  private forward(tokens: number[], tokenStrings: string[], prefix?: Matrix): { data: StepInspectionData; caches: BlockCache[] } {
    const P = prefix ? prefix.length : 0;
    const seqLen = Math.min(tokens.length, this.config.contextWindow - P);
    const slicedTokens = tokens.slice(0, seqLen);
    const slicedStrings = tokenStrings.slice(0, seqLen);
    const dModel = this.config.dModel;
    const numHeads = this.config.numHeads;

    // 1. Embedding lookup
    const tokenEmbeddings: Matrix = MatrixMath.zeros(P + seqLen, dModel);
    const posEmbeddings: Matrix = MatrixMath.zeros(P + seqLen, dModel);
    const combinedEmbeddings: Matrix = MatrixMath.zeros(P + seqLen, dModel);

    // Prefix rows (e.g. a picture's projected patches) are used as given, plus their position
    for (let i = 0; i < P; i++) {
      for (let d = 0; d < dModel; d++) {
        const rowVal = prefix![i][d];
        const posVal = this.wPosEmbed[i][d];
        tokenEmbeddings[i][d] = rowVal;
        posEmbeddings[i][d] = posVal;
        combinedEmbeddings[i][d] = rowVal + posVal;
      }
    }

    for (let i = 0; i < seqLen; i++) {
      const p = P + i;
      const tokId = slicedTokens[i];
      for (let d = 0; d < dModel; d++) {
        const tokVal = tokId < this.config.vocabSize ? this.wTokenEmbed[tokId][d] : 0;
        const posVal = p < this.config.contextWindow ? this.wPosEmbed[p][d] : 0;
        tokenEmbeddings[p][d] = tokVal;
        posEmbeddings[p][d] = posVal;
        combinedEmbeddings[p][d] = tokVal + posVal;
      }
    }

    let x: Matrix = combinedEmbeddings;
    const layerInspections: LayerInspection[] = [];
    const caches: BlockCache[] = [];

    // 2. Transformer blocks (block.ts), causal: a position may only look at itself and earlier ones
    for (let l = 0; l < this.config.numLayers; l++) {
      const { inspection, cache } = blockForward(x, this.blockWeights(l), numHeads, true, l);
      layerInspections.push(inspection);
      caches.push(cache);
      x = inspection.blockOutput;
    }

    // Final LayerNorm before the unembedding head: under pre-LN the stream is never normalized
    // inside the blocks, so this is where it's brought back to a standard scale.
    const finalNorm = MatrixMath.layerNorm(x);

    // 3. Final Logits & Softmax Probabilities: [seqLen x dModel] * [dModel x vocabSize] -> [seqLen x vocabSize]
    const logits = MatrixMath.matmul(finalNorm, this.wHead);
    const probabilities = MatrixMath.softmax(logits, false);

    const data: StepInspectionData = {
      inputString: slicedStrings.join(''),
      // Prefix positions have no token id (-1) and are labelled ▣1, ▣2, …
      tokens: P ? [...new Array<number>(P).fill(-1), ...slicedTokens] : slicedTokens,
      tokenStrings: P ? [...Array.from({ length: P }, (_, i) => `▣${i + 1}`), ...slicedStrings] : slicedStrings,
      tokenEmbeddings,
      positionEmbeddings: posEmbeddings,
      combinedEmbeddings,
      layerInspections,
      finalNorm,
      logits,
      probabilities
    };
    return { data, caches };
  }

  /**
   * Mean cross-entropy of next-token predictions: L = −(1/N) Σ log P(target_i), over the N text
   * positions that count (all of them, unless `mask` leaves some out). Text position i is row
   * `offset + i` of `probs` (offset = the number of prefix rows).
   */
  private crossEntropy(probs: number[][], targetTokens: number[], seqLen: number, offset = 0, mask?: boolean[]): number {
    let totalLoss = 0;
    let count = 0;
    for (let i = 0; i < seqLen; i++) {
      if (mask && !mask[i]) continue;
      count++;
      const targetId = targetTokens[i];
      if (targetId < probs[offset + i].length) {
        totalLoss -= Math.log(Math.max(probs[offset + i][targetId], 1e-10));
      }
    }
    return count ? totalLoss / count : 0;
  }

  /** Forward pass only (no weight updates) — used for validation loss on held-out text. `opts`: see computeGradients. */
  public evaluateLoss(inputTokens: number[], targetTokens: number[], opts: PrefixOptions = {}): { loss: number; perplexity: number } {
    const P = opts.prefix ? opts.prefix.length : 0;
    const seqLen = Math.min(inputTokens.length, targetTokens.length, this.config.contextWindow - P);
    if (seqLen === 0) return { loss: 0, perplexity: 1 };
    const forward = this.inspectForwardPass(inputTokens.slice(0, seqLen), inputTokens.slice(0, seqLen).map(String), opts.prefix);
    const loss = this.crossEntropy(forward.probabilities, targetTokens, seqLen, P, opts.lossMask);
    return { loss, perplexity: Math.exp(loss) };
  }


  /**
   * Every trainable matrix by name (e.g. 'wQ.0' is layer 0's query projection).
   * These are the live arrays, not copies — the gradient checker perturbs them in place.
   */
  public getParameters(): Record<string, Matrix> {
    const params: Record<string, Matrix> = {
      wTokenEmbed: this.wTokenEmbed,
      wPosEmbed: this.wPosEmbed,
      wHead: this.wHead,
    };
    for (let l = 0; l < this.config.numLayers; l++) {
      params[`wQ.${l}`] = this.wQ[l];
      params[`wK.${l}`] = this.wK[l];
      params[`wV.${l}`] = this.wV[l];
      params[`wO.${l}`] = this.wO[l];
      params[`wMlp1.${l}`] = this.wMlp1[l];
      params[`wMlp2.${l}`] = this.wMlp2[l];
    }
    return params;
  }

  /**
   * Everything training has changed, as plain typed arrays that can be stored (the app keeps
   * it in IndexedDB) and loaded back exactly: every weight matrix, flattened row by row, plus
   * the optimizer's memory. The architecture itself lives in `config`.
   */
  public exportState(): ModelState {
    const weights: Record<string, Float64Array> = {};
    for (const [name, M] of Object.entries(this.getParameters())) weights[name] = Float64Array.from(M.flat());
    return { weights, optimizer: this.optimizer.exportState() };
  }

  /**
   * Load a state from exportState into this model, overwriting its weights in place. Every
   * matrix must exist with the same size, or nothing is changed and this returns false.
   */
  public importState(state: ModelState): boolean {
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
    // Adam's memory only makes sense for the same optimizer; otherwise it starts fresh
    if (state.optimizer.type === this.optimizer.type) this.optimizer.importState(state.optimizer, params);
    return true;
  }

  /**
   * Backpropagation: the exact gradient of the mean cross-entropy loss with respect
   * to every parameter, keyed like getParameters().
   *
   * The backward pass walks the forward pass in reverse. Each op receives dOut
   * (∂L/∂output) and turns it into ∂L/∂input plus ∂L/∂weights using its local rule:
   *
   *   matmul     C = A·B          dA = dC·Bᵀ,  dB = Aᵀ·dC
   *   residual   y = a + b        da = dy,     db = dy     (gradient is copied to both branches)
   *   LayerNorm, GELU, softmax    see MatrixMath.*Backward
   *   embedding  x_i = E[tok_i]   dE[tok_i] += dx_i        (scatter-add; repeated tokens accumulate)
   */
  public computeGradients(
    inputTokens: number[],
    targetTokens: number[],
    opts: PrefixOptions = {}
  ): { loss: number; perplexity: number; grads: Record<string, Matrix>; dPrefix: Matrix | null } {
    const P = opts.prefix ? opts.prefix.length : 0;
    const mask = opts.lossMask;
    const seqLen = Math.min(inputTokens.length, targetTokens.length, this.config.contextWindow - P);
    const { vocabSize, contextWindow, dModel, numHeads, numLayers } = this.config;
    const T = MatrixMath.transpose;
    const mm = MatrixMath.matmul;

    const grads: Record<string, Matrix> = {};
    for (const [name, p] of Object.entries(this.getParameters())) {
      grads[name] = MatrixMath.zeros(p.length, p[0].length);
    }
    let count = 0;
    for (let i = 0; i < seqLen; i++) if (!mask || mask[i]) count++;
    if (count === 0) return { loss: 0, perplexity: 1, grads, dPrefix: P ? MatrixMath.zeros(P, dModel) : null };

    const tokens = inputTokens.slice(0, seqLen);
    const { data: fwd, caches } = this.forward(tokens, tokens.map(String), opts.prefix);
    const loss = this.crossEntropy(fwd.probabilities, targetTokens, seqLen, P, mask);

    // ── Softmax + cross-entropy combined: ∂L/∂logits_i = (p_i − onehot(target_i)) / N
    // (prefix rows and masked-out positions predict nothing that counts, so their gradient is 0)
    const dLogits: Matrix = fwd.probabilities.map((row, r) => {
      const i = r - P;
      if (i < 0 || (mask && !mask[i])) return row.map(() => 0);
      return row.map((p, k) => (p - (k === targetTokens[i] ? 1 : 0)) / count);
    });

    // ── Output head: logits = finalNorm · W_head
    grads.wHead = mm(T(fwd.finalNorm), dLogits);
    const dFinalNorm = mm(dLogits, T(this.wHead));

    // ── Final LayerNorm. Its input is the last block's output (or the embeddings if 0 layers).
    const lastOut = numLayers > 0 ? fwd.layerInspections[numLayers - 1].blockOutput : fwd.combinedEmbeddings;
    let dX = MatrixMath.layerNormBackward(lastOut, dFinalNorm);

    // ── Transformer blocks, last to first (block.ts). dX enters as ∂L/∂(block output).
    for (let l = numLayers - 1; l >= 0; l--) {
      const back = blockBackward(dX, this.blockWeights(l), fwd.layerInspections[l], caches[l], numHeads);
      for (const name of BLOCK_WEIGHT_NAMES) grads[`${name}.${l}`] = back.grads[name];
      dX = back.dInput;
    }

    // ── Embeddings: x_p = tokEmbed[tok_i] + posEmbed[p], text position i sitting at p = P + i
    for (let i = 0; i < seqLen; i++) {
      const p = P + i;
      const tokId = tokens[i];
      for (let d = 0; d < dModel; d++) {
        if (tokId < vocabSize) grads.wTokenEmbed[tokId][d] += dX[p][d];
        if (p < contextWindow) grads.wPosEmbed[p][d] += dX[p][d];
      }
    }
    // ── Prefix rows: x_i = prefix_i + posEmbed[i], so ∂L/∂prefix_i is just dX_i
    let dPrefix: Matrix | null = null;
    if (P) {
      dPrefix = dX.slice(0, P).map(row => row.slice());
      for (let i = 0; i < P; i++) for (let d = 0; d < dModel; d++) grads.wPosEmbed[i][d] += dX[i][d];
    }

    return { loss, perplexity: Math.exp(loss), grads, dPrefix };
  }

  /**
   * One training step: forward, backward, then update every parameter via the optimizer.
   * Gradients are clipped to a global L2 norm of `maxGradNorm` first (standard practice —
   * it keeps one unlucky batch from blowing the weights up, especially under plain SGD).
   */
  public trainStep(
    inputTokens: number[],
    targetTokens: number[],
    lr: number = 0.01,
    maxGradNorm: number = 1.0
  ): { loss: number; perplexity: number; gradNorm: number } {
    const { loss, perplexity, grads } = this.computeGradients(inputTokens, targetTokens);

    let sumSq = 0;
    for (const g of Object.values(grads)) {
      for (const row of g) for (const v of row) sumSq += v * v;
    }
    const gradNorm = Math.sqrt(sumSq);
    const clip = gradNorm > maxGradNorm ? maxGradNorm / gradNorm : 1;

    const params = this.getParameters();
    for (const [name, g] of Object.entries(grads)) {
      // Scale in place: these gradients are fresh each step, so there's no need to copy them
      if (clip !== 1) for (const row of g) for (let c = 0; c < row.length; c++) row[c] *= clip;
      this.optimizer.step(name, params[name], g, lr);
    }

    return { loss, perplexity, gradNorm };
  }
}
