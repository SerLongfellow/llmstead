import { LayerInspection, StepInspectionData, TransformerConfig } from '../types';
import { Matrix, MatrixMath } from './tensor';
import { Optimizer, OptimizerState, OptimizerType } from './optimizer';

/** A trained model's learned numbers (see MicroTransformer.exportState) */
export interface ModelState {
  weights: Record<string, Float64Array>;
  optimizer: OptimizerState;
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
 */

/** Values from the forward pass that the visualizers don't need but backprop does */
interface LayerCache {
  input: Matrix;        // block input x (input to LayerNorm 1)
  mlpHiddenRaw: Matrix; // MLP pre-activation (input to GELU)
}

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

  /** Run a complete forward pass and capture all internal states for visualization */
  public inspectForwardPass(tokens: number[], tokenStrings: string[]): StepInspectionData {
    return this.forward(tokens, tokenStrings).data;
  }

  /** Forward pass that also returns the extra per-layer values the backward pass needs */
  private forward(tokens: number[], tokenStrings: string[]): { data: StepInspectionData; caches: LayerCache[] } {
    const seqLen = Math.min(tokens.length, this.config.contextWindow);
    const slicedTokens = tokens.slice(0, seqLen);
    const slicedStrings = tokenStrings.slice(0, seqLen);
    const dModel = this.config.dModel;
    const numHeads = this.config.numHeads;
    const headDim = dModel / numHeads; // exact: constructor guarantees divisibility

    // 1. Embedding lookup
    const tokenEmbeddings: Matrix = MatrixMath.zeros(seqLen, dModel);
    const posEmbeddings: Matrix = MatrixMath.zeros(seqLen, dModel);
    const combinedEmbeddings: Matrix = MatrixMath.zeros(seqLen, dModel);

    for (let i = 0; i < seqLen; i++) {
      const tokId = slicedTokens[i];
      for (let d = 0; d < dModel; d++) {
        const tokVal = tokId < this.config.vocabSize ? this.wTokenEmbed[tokId][d] : 0;
        const posVal = i < this.config.contextWindow ? this.wPosEmbed[i][d] : 0;
        tokenEmbeddings[i][d] = tokVal;
        posEmbeddings[i][d] = posVal;
        combinedEmbeddings[i][d] = tokVal + posVal;
      }
    }

    let x: Matrix = combinedEmbeddings;
    const layerInspections: LayerInspection[] = [];
    const caches: LayerCache[] = [];

    // 2. Transformer Block Execution
    for (let l = 0; l < this.config.numLayers; l++) {
      // Pre-LN: attention reads a normalized copy of the stream
      const norm1Output = MatrixMath.layerNorm(x);

      // Q, K, V projections: [seqLen x dModel] * [dModel x dModel] -> [seqLen x dModel]
      const fullQ = MatrixMath.matmul(norm1Output, this.wQ[l]);
      const fullK = MatrixMath.matmul(norm1Output, this.wK[l]);
      const fullV = MatrixMath.matmul(norm1Output, this.wV[l]);

      // Split into heads: [numHeads][seqLen][headDim]
      const queries: number[][][] = MatrixMath.zeros3D(numHeads, seqLen, headDim);
      const keys: number[][][] = MatrixMath.zeros3D(numHeads, seqLen, headDim);
      const values: number[][][] = MatrixMath.zeros3D(numHeads, seqLen, headDim);

      for (let h = 0; h < numHeads; h++) {
        for (let i = 0; i < seqLen; i++) {
          for (let d = 0; d < headDim; d++) {
            queries[h][i][d] = fullQ[i][h * headDim + d];
            keys[h][i][d] = fullK[i][h * headDim + d];
            values[h][i][d] = fullV[i][h * headDim + d];
          }
        }
      }

      // Compute Multi-Head Attention for each head
      const rawAttentionScores: number[][][] = MatrixMath.zeros3D(numHeads, seqLen, seqLen);
      const attentionWeights: number[][][] = MatrixMath.zeros3D(numHeads, seqLen, seqLen);
      const headOutputs: number[][][] = MatrixMath.zeros3D(numHeads, seqLen, headDim);

      for (let h = 0; h < numHeads; h++) {
        // Dot product Q_h * K_h^T / sqrt(headDim)
        const Q_h = queries[h];
        const K_h_T = MatrixMath.transpose(keys[h]);
        const scores = MatrixMath.scale(MatrixMath.matmul(Q_h, K_h_T), 1.0 / Math.sqrt(headDim));
        rawAttentionScores[h] = scores;

        // Causal Softmax
        const attnPattern = MatrixMath.softmax(scores, true);
        attentionWeights[h] = attnPattern;

        // Weighted sum of Values: [seqLen x seqLen] * [seqLen x headDim] -> [seqLen x headDim]
        headOutputs[h] = MatrixMath.matmul(attnPattern, values[h]);
      }

      // Concatenate Head outputs: [seqLen x dModel]
      const concatOutput: Matrix = MatrixMath.zeros(seqLen, dModel);
      for (let i = 0; i < seqLen; i++) {
        for (let h = 0; h < numHeads; h++) {
          for (let d = 0; d < headDim; d++) {
            concatOutput[i][h * headDim + d] = headOutputs[h][i][d];
          }
        }
      }

      // Project concatenated multi-head output back: [seqLen x dModel]
      const attnProjected = MatrixMath.matmul(concatOutput, this.wO[l]);

      // Residual add: attention's output is added onto the stream
      const afterAttention = MatrixMath.add(x, attnProjected);

      // MLP / Feed-Forward Network on a normalized copy: LN -> MLP1 -> GELU -> MLP2
      const norm2Output = MatrixMath.layerNorm(afterAttention);
      const mlpHiddenRaw = MatrixMath.matmul(norm2Output, this.wMlp1[l]);
      const mlpHidden = MatrixMath.gelu(mlpHiddenRaw);
      const mlpOutput = MatrixMath.matmul(mlpHidden, this.wMlp2[l]);

      // Residual add around the MLP gives the block output (not normalized)
      const blockOutput = MatrixMath.add(afterAttention, mlpOutput);

      layerInspections.push({
        layerIndex: l,
        queries,
        keys,
        values,
        rawAttentionScores,
        attentionWeights,
        headOutputs,
        concatOutput,
        norm1Output,
        afterAttention,
        norm2Output,
        mlpHidden,
        mlpOutput,
        blockOutput
      });

      // Pass output to next block
      caches.push({ input: x, mlpHiddenRaw });

      x = blockOutput;
    }

    // Final LayerNorm before the unembedding head: under pre-LN the stream is never normalized
    // inside the blocks, so this is where it's brought back to a standard scale.
    const finalNorm = MatrixMath.layerNorm(x);

    // 3. Final Logits & Softmax Probabilities: [seqLen x dModel] * [dModel x vocabSize] -> [seqLen x vocabSize]
    const logits = MatrixMath.matmul(finalNorm, this.wHead);
    const probabilities = MatrixMath.softmax(logits, false);

    const data: StepInspectionData = {
      inputString: slicedStrings.join(''),
      tokens: slicedTokens,
      tokenStrings: slicedStrings,
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
   * Cross-entropy of next-token predictions: L = −Σ wᵢ log P(target_i). With no weights every
   * position counts 1/N, the plain mean used in pre-training. Post-training passes its own:
   * zeros on prompt tokens (SFT), or signed weights that push a sequence up or down (DPO, RL).
   */
  private crossEntropy(probs: number[][], targetTokens: number[], seqLen: number, weights?: number[]): number {
    let totalLoss = 0;
    for (let i = 0; i < seqLen; i++) {
      const targetId = targetTokens[i];
      if (targetId < probs[i].length) {
        totalLoss -= (weights ? weights[i] : 1) * Math.log(Math.max(probs[i][targetId], 1e-10));
      }
    }
    return weights ? totalLoss : totalLoss / seqLen;
  }

  /** Forward pass only (no weight updates) — used for validation loss on held-out text */
  public evaluateLoss(
    inputTokens: number[],
    targetTokens: number[],
    weights?: number[]
  ): { loss: number; perplexity: number } {
    const seqLen = Math.min(inputTokens.length, targetTokens.length, this.config.contextWindow);
    if (seqLen === 0) return { loss: 0, perplexity: 1 };
    const forward = this.inspectForwardPass(inputTokens.slice(0, seqLen), inputTokens.slice(0, seqLen).map(String));
    const loss = this.crossEntropy(forward.probabilities, targetTokens, seqLen, weights);
    return { loss, perplexity: Math.exp(loss) };
  }

  /**
   * log P(tokens[i+1] | tokens[0..i]) for every position i, from one forward pass: entry i is
   * how likely the model found the token that actually came next. Summing a response's entries
   * gives the log-probability of the whole response, which DPO and RL compare.
   */
  public tokenLogProbs(tokens: number[]): number[] {
    const n = Math.min(tokens.length - 1, this.config.contextWindow);
    if (n <= 0) return [];
    const { probabilities } = this.inspectForwardPass(tokens.slice(0, n), tokens.slice(0, n).map(String));
    return probabilities.map((row, i) => Math.log(Math.max(row[tokens[i + 1]] ?? 0, 1e-10)));
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
   *
   * `weights` (one per position) turns the mean into a weighted sum, L = −Σ wᵢ log P(target_i)
   * (see crossEntropy). Only the first step below changes; the rest of the pass is the same.
   */
  public computeGradients(
    inputTokens: number[],
    targetTokens: number[],
    weights?: number[]
  ): { loss: number; perplexity: number; grads: Record<string, Matrix> } {
    const seqLen = Math.min(inputTokens.length, targetTokens.length, this.config.contextWindow);
    const { vocabSize, contextWindow, dModel, numHeads, numLayers } = this.config;
    const headDim = dModel / numHeads;
    const attnScale = 1.0 / Math.sqrt(headDim);
    const T = MatrixMath.transpose;
    const mm = MatrixMath.matmul;

    const grads: Record<string, Matrix> = {};
    for (const [name, p] of Object.entries(this.getParameters())) {
      grads[name] = MatrixMath.zeros(p.length, p[0].length);
    }
    if (seqLen === 0) return { loss: 0, perplexity: 1, grads };

    const tokens = inputTokens.slice(0, seqLen);
    const { data: fwd, caches } = this.forward(tokens, tokens.map(String));
    const loss = this.crossEntropy(fwd.probabilities, targetTokens, seqLen, weights);

    // ── Softmax + cross-entropy combined: ∂L/∂logits_i = wᵢ · (p_i − onehot(target_i)), wᵢ = 1/N by default
    const dLogits: Matrix = weights
      ? fwd.probabilities.map((row, i) => row.map((p, k) => weights[i] * (p - (k === targetTokens[i] ? 1 : 0))))
      : fwd.probabilities.map((row, i) => row.map((p, k) => (p - (k === targetTokens[i] ? 1 : 0)) / seqLen));

    // ── Output head: logits = finalNorm · W_head
    grads.wHead = mm(T(fwd.finalNorm), dLogits);
    const dFinalNorm = mm(dLogits, T(this.wHead));

    // ── Final LayerNorm. Its input is the last block's output (or the embeddings if 0 layers).
    const lastOut = numLayers > 0 ? fwd.layerInspections[numLayers - 1].blockOutput : fwd.combinedEmbeddings;
    let dX = MatrixMath.layerNormBackward(lastOut, dFinalNorm);

    // ── Transformer blocks, last to first. dX enters as ∂L/∂(block output).
    for (let l = numLayers - 1; l >= 0; l--) {
      const ins = fwd.layerInspections[l];
      const cache = caches[l];

      // blockOutput = afterAttention + mlpOut  → gradient flows to both the skip path and the MLP
      const dAfterAttn = dX.map(row => row.slice());
      const dMlpOut = dX;

      // mlpOut = gelu(mlpHiddenRaw) · W2
      grads[`wMlp2.${l}`] = mm(T(ins.mlpHidden), dMlpOut);
      const dMlpHidden = mm(dMlpOut, T(this.wMlp2[l]));

      // mlpHidden = gelu(mlpHiddenRaw)
      const dMlpRaw = MatrixMath.geluBackward(cache.mlpHiddenRaw, dMlpHidden);

      // mlpHiddenRaw = norm2 · W1
      grads[`wMlp1.${l}`] = mm(T(ins.norm2Output), dMlpRaw);
      const dNorm2 = mm(dMlpRaw, T(this.wMlp1[l]));

      // norm2 = LN(afterAttention): the MLP branch's gradient joins the skip path's
      MatrixMath.addInPlace(dAfterAttn, MatrixMath.layerNormBackward(ins.afterAttention, dNorm2));

      // afterAttention = x + attnProjected  → skip path and attention branch both get it
      const dInput = dAfterAttn.map(row => row.slice());
      const dAttnProjected = dAfterAttn;

      // attnProjected = concat · W_O
      grads[`wO.${l}`] = mm(T(ins.concatOutput), dAttnProjected);
      const dConcat = mm(dAttnProjected, T(this.wO[l]));

      // Per head: scores = Q_h·K_hᵀ·scale, A = causalSoftmax(scores), out_h = A·V_h
      const dQ = MatrixMath.zeros(seqLen, dModel);
      const dK = MatrixMath.zeros(seqLen, dModel);
      const dV = MatrixMath.zeros(seqLen, dModel);
      for (let h = 0; h < numHeads; h++) {
        const off = h * headDim;
        const dOut = dConcat.map(row => row.slice(off, off + headDim));
        const A = ins.attentionWeights[h];

        const dA = mm(dOut, T(ins.values[h]));
        const dV_h = mm(T(A), dOut);
        const dScores = MatrixMath.scale(MatrixMath.softmaxBackward(A, dA), attnScale);
        const dQ_h = mm(dScores, ins.keys[h]);
        const dK_h = mm(T(dScores), ins.queries[h]);

        for (let i = 0; i < seqLen; i++) {
          for (let d = 0; d < headDim; d++) {
            dQ[i][off + d] = dQ_h[i][d];
            dK[i][off + d] = dK_h[i][d];
            dV[i][off + d] = dV_h[i][d];
          }
        }
      }

      // Q = norm1·W_Q, K = norm1·W_K, V = norm1·W_V  → norm1 gets gradient from all three
      const n1T = T(ins.norm1Output);
      grads[`wQ.${l}`] = mm(n1T, dQ);
      grads[`wK.${l}`] = mm(n1T, dK);
      grads[`wV.${l}`] = mm(n1T, dV);
      const dNorm1 = mm(dQ, T(this.wQ[l]));
      MatrixMath.addInPlace(dNorm1, mm(dK, T(this.wK[l])));
      MatrixMath.addInPlace(dNorm1, mm(dV, T(this.wV[l])));

      // norm1 = LN(x)
      MatrixMath.addInPlace(dInput, MatrixMath.layerNormBackward(cache.input, dNorm1));

      dX = dInput;
    }

    // ── Embeddings: x_i = tokEmbed[tok_i] + posEmbed[i]
    for (let i = 0; i < seqLen; i++) {
      const tokId = tokens[i];
      for (let d = 0; d < dModel; d++) {
        if (tokId < vocabSize) grads.wTokenEmbed[tokId][d] += dX[i][d];
        if (i < contextWindow) grads.wPosEmbed[i][d] += dX[i][d];
      }
    }

    return { loss, perplexity: Math.exp(loss), grads };
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
    const gradNorm = this.applyGradients(grads, lr, maxGradNorm);
    return { loss, perplexity, gradNorm };
  }

  /**
   * Clip `grads` to a global L2 norm of `maxGradNorm`, then let the optimizer update every
   * parameter. Returns the norm before clipping. Post-training passes its own optimizer, so its
   * Adam memory starts fresh instead of carrying pre-training's.
   */
  public applyGradients(
    grads: Record<string, Matrix>,
    lr: number,
    maxGradNorm: number = 1.0,
    optimizer: Optimizer = this.optimizer
  ): number {
    const gradNorm = globalNorm(grads);
    const clip = gradNorm > maxGradNorm ? maxGradNorm / gradNorm : 1;

    const params = this.getParameters();
    for (const [name, g] of Object.entries(grads)) {
      // Scale in place: these gradients are fresh each step, so there's no need to copy them
      if (clip !== 1) for (const row of g) for (let c = 0; c < row.length; c++) row[c] *= clip;
      optimizer.step(name, params[name], g, lr);
    }
    return gradNorm;
  }
}

/** √(Σ g²) over every entry of every matrix */
export function globalNorm(grads: Record<string, Matrix>): number {
  let sumSq = 0;
  for (const g of Object.values(grads)) {
    for (const row of g) for (const v of row) sumSq += v * v;
  }
  return Math.sqrt(sumSq);
}
