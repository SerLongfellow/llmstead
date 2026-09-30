import { LayerInspection, StepInspectionData, TransformerConfig } from '../types';
import { Matrix, MatrixMath } from './tensor';
import { Optimizer, OptimizerType } from './optimizer';

/** Settings that change the shape of the weight matrices. Changing any of these requires a fresh model. */
export type ArchitectureConfig = Pick<
  TransformerConfig,
  'vocabSize' | 'contextWindow' | 'dModel' | 'numHeads' | 'numLayers' | 'mlpRatio'
>;

/**
 * Architecture: POST-LayerNorm (original "Attention Is All You Need" layout).
 *
 *   x ─► Attention ─► (+x) ─► LayerNorm ─► MLP ─► (+) ─► LayerNorm ─► next block
 *
 * GPT-2 and most modern LLMs use PRE-LayerNorm instead, where each sub-layer
 * normalizes its *input* and the residual stream itself is never normalized:
 *
 *   x ─► LayerNorm ─► Attention ─► (+x) ─► LayerNorm ─► MLP ─► (+) ─► next block
 *
 * Pre-LN trains more stably in deep stacks; post-LN is simpler to read.
 */

/** Values from the forward pass that the visualizers don't need but backprop does */
interface LayerCache {
  input: Matrix;        // block input x
  res1: Matrix;         // x + attention output (input to LayerNorm 1)
  mlpHiddenRaw: Matrix; // MLP pre-activation (input to GELU)
  res2: Matrix;         // norm1 + MLP output (input to LayerNorm 2)
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
      // Q, K, V projections: [seqLen x dModel] * [dModel x dModel] -> [seqLen x dModel]
      const fullQ = MatrixMath.matmul(x, this.wQ[l]);
      const fullK = MatrixMath.matmul(x, this.wK[l]);
      const fullV = MatrixMath.matmul(x, this.wV[l]);

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

      // Post-LN: residual add FIRST, then normalize the sum
      const res1 = MatrixMath.add(x, attnProjected);
      const norm1Output = MatrixMath.layerNorm(res1);

      // MLP / Feed-Forward Network on the normalized stream: MLP1 -> GELU -> MLP2
      const mlpHiddenRaw = MatrixMath.matmul(norm1Output, this.wMlp1[l]);
      const mlpHidden = MatrixMath.gelu(mlpHiddenRaw);
      const mlpOutput = MatrixMath.matmul(mlpHidden, this.wMlp2[l]);

      // Post-LN: residual add (around the MLP), then normalize again
      const res2 = MatrixMath.add(norm1Output, mlpOutput);
      const norm2Output = MatrixMath.layerNorm(res2);

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
        mlpHidden,
        mlpOutput,
        norm2Output
      });

      // Pass output to next block
      caches.push({ input: x, res1, mlpHiddenRaw, res2 });

      x = norm2Output;
    }

    // Final LayerNorm before the unembedding head. In this post-LN layout x is already
    // normalized by the last block, so this is ~identity; it becomes essential under pre-LN.
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

  /** Mean cross-entropy of next-token predictions: L = −(1/N) Σ log P(target_i) */
  private crossEntropy(probs: number[][], targetTokens: number[], seqLen: number): number {
    let totalLoss = 0;
    for (let i = 0; i < seqLen; i++) {
      const targetId = targetTokens[i];
      if (targetId < probs[i].length) {
        totalLoss -= Math.log(Math.max(probs[i][targetId], 1e-10));
      }
    }
    return totalLoss / seqLen;
  }

  /** Forward pass only (no weight updates) — used for validation loss on held-out text */
  public evaluateLoss(inputTokens: number[], targetTokens: number[]): { loss: number; perplexity: number } {
    const seqLen = Math.min(inputTokens.length, targetTokens.length, this.config.contextWindow);
    if (seqLen === 0) return { loss: 0, perplexity: 1 };
    const forward = this.inspectForwardPass(inputTokens.slice(0, seqLen), inputTokens.slice(0, seqLen).map(String));
    const loss = this.crossEntropy(forward.probabilities, targetTokens, seqLen);
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
    targetTokens: number[]
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
    const loss = this.crossEntropy(fwd.probabilities, targetTokens, seqLen);

    // ── Softmax + cross-entropy combined: ∂L/∂logits_i = (p_i − onehot(target_i)) / N
    const dLogits: Matrix = fwd.probabilities.map((row, i) =>
      row.map((p, k) => (p - (k === targetTokens[i] ? 1 : 0)) / seqLen)
    );

    // ── Output head: logits = finalNorm · W_head
    grads.wHead = mm(T(fwd.finalNorm), dLogits);
    const dFinalNorm = mm(dLogits, T(this.wHead));

    // ── Final LayerNorm. Its input is the last block's output (or the embeddings if 0 layers).
    const lastOut = numLayers > 0 ? fwd.layerInspections[numLayers - 1].norm2Output : fwd.combinedEmbeddings;
    let dX = MatrixMath.layerNormBackward(lastOut, dFinalNorm);

    // ── Transformer blocks, last to first. dX enters as ∂L/∂(block output).
    for (let l = numLayers - 1; l >= 0; l--) {
      const ins = fwd.layerInspections[l];
      const cache = caches[l];

      // norm2 = LN(res2)
      const dRes2 = MatrixMath.layerNormBackward(cache.res2, dX);

      // res2 = norm1 + mlpOut  → gradient flows to both the skip path and the MLP
      const dNorm1 = dRes2.map(row => row.slice());
      const dMlpOut = dRes2;

      // mlpOut = gelu(mlpHiddenRaw) · W2
      grads[`wMlp2.${l}`] = mm(T(ins.mlpHidden), dMlpOut);
      const dMlpHidden = mm(dMlpOut, T(this.wMlp2[l]));

      // mlpHidden = gelu(mlpHiddenRaw)
      const dMlpRaw = MatrixMath.geluBackward(cache.mlpHiddenRaw, dMlpHidden);

      // mlpHiddenRaw = norm1 · W1
      grads[`wMlp1.${l}`] = mm(T(ins.norm1Output), dMlpRaw);
      MatrixMath.addInPlace(dNorm1, mm(dMlpRaw, T(this.wMlp1[l])));

      // norm1 = LN(res1)
      const dRes1 = MatrixMath.layerNormBackward(cache.res1, dNorm1);

      // res1 = x + attnProjected  → skip path and attention branch both get dRes1
      const dInput = dRes1.map(row => row.slice());
      const dAttnProjected = dRes1;

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

      // Q = x·W_Q, K = x·W_K, V = x·W_V  → x gets gradient from all three projections
      const xT = T(cache.input);
      grads[`wQ.${l}`] = mm(xT, dQ);
      grads[`wK.${l}`] = mm(xT, dK);
      grads[`wV.${l}`] = mm(xT, dV);
      MatrixMath.addInPlace(dInput, mm(dQ, T(this.wQ[l])));
      MatrixMath.addInPlace(dInput, mm(dK, T(this.wK[l])));
      MatrixMath.addInPlace(dInput, mm(dV, T(this.wV[l])));

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

    let sumSq = 0;
    for (const g of Object.values(grads)) {
      for (const row of g) for (const v of row) sumSq += v * v;
    }
    const gradNorm = Math.sqrt(sumSq);
    const clip = gradNorm > maxGradNorm ? maxGradNorm / gradNorm : 1;

    const params = this.getParameters();
    for (const [name, g] of Object.entries(grads)) {
      const scaled = clip === 1 ? g : MatrixMath.scale(g, clip);
      this.optimizer.step(name, params[name], scaled, lr);
    }

    return { loss, perplexity, gradNorm };
  }
}
