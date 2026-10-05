import { LayerInspection } from '../types';
import { Matrix, MatrixMath } from './tensor';

/**
 * One transformer block, shared by the GPT (transformer.ts) and the vision-language model
 * (vl/clip.ts). PRE-LayerNorm, the GPT-2 layout: each half reads a normalized copy of the
 * stream and adds its output back onto it.
 *
 *   x ─┬─► LayerNorm ─► Attention ─► (+) ─┬─► LayerNorm ─► MLP ─► (+) ─► block output
 *      └──────────────────────────────┘   └────────────────────────┘
 *
 * `causal` decides whether a position may attend to later positions. A GPT predicting the next
 * token must not peek ahead (causal); an encoder that only summarizes a whole input (an image's
 * patches, or a caption) can let every position see every other.
 */

/** One block's weight matrices (no biases, no LayerNorm gain/bias) */
export interface BlockWeights {
  wQ: Matrix;    // [dModel x dModel]
  wK: Matrix;    // [dModel x dModel]
  wV: Matrix;    // [dModel x dModel]
  wO: Matrix;    // [dModel x dModel]
  wMlp1: Matrix; // [dModel x dMlp]
  wMlp2: Matrix; // [dMlp x dModel]
}

export const BLOCK_WEIGHT_NAMES = ['wQ', 'wK', 'wV', 'wO', 'wMlp1', 'wMlp2'] as const;

/** Values from the forward pass that the visualizers don't need but backprop does */
export interface BlockCache {
  input: Matrix;        // block input x (input to LayerNorm 1)
  mlpHiddenRaw: Matrix; // MLP pre-activation (input to GELU)
}

export function randomBlockWeights(dModel: number, dMlp: number, scale: number): BlockWeights {
  return {
    wQ: MatrixMath.random(dModel, dModel, scale),
    wK: MatrixMath.random(dModel, dModel, scale),
    wV: MatrixMath.random(dModel, dModel, scale),
    wO: MatrixMath.random(dModel, dModel, scale),
    wMlp1: MatrixMath.random(dModel, dMlp, scale),
    wMlp2: MatrixMath.random(dMlp, dModel, scale),
  };
}

/** Run one block on x [seqLen x dModel]; the output is inspection.blockOutput */
export function blockForward(
  x: Matrix, w: BlockWeights, numHeads: number, causal: boolean, layerIndex: number
): { inspection: LayerInspection; cache: BlockCache } {
  const seqLen = x.length;
  const dModel = x[0].length;
  const headDim = dModel / numHeads;

  // Pre-LN: attention reads a normalized copy of the stream
  const norm1Output = MatrixMath.layerNorm(x);

  // Q, K, V projections: [seqLen x dModel] * [dModel x dModel] -> [seqLen x dModel]
  const fullQ = MatrixMath.matmul(norm1Output, w.wQ);
  const fullK = MatrixMath.matmul(norm1Output, w.wK);
  const fullV = MatrixMath.matmul(norm1Output, w.wV);

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

    // Softmax over each row (with the causal mask for a GPT)
    const attnPattern = MatrixMath.softmax(scores, causal);
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
  const attnProjected = MatrixMath.matmul(concatOutput, w.wO);

  // Residual add: attention's output is added onto the stream
  const afterAttention = MatrixMath.add(x, attnProjected);

  // MLP / Feed-Forward Network on a normalized copy: LN -> MLP1 -> GELU -> MLP2
  const norm2Output = MatrixMath.layerNorm(afterAttention);
  const mlpHiddenRaw = MatrixMath.matmul(norm2Output, w.wMlp1);
  const mlpHidden = MatrixMath.gelu(mlpHiddenRaw);
  const mlpOutput = MatrixMath.matmul(mlpHidden, w.wMlp2);

  // Residual add around the MLP gives the block output (not normalized)
  const blockOutput = MatrixMath.add(afterAttention, mlpOutput);

  return {
    inspection: {
      layerIndex,
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
      blockOutput,
    },
    cache: { input: x, mlpHiddenRaw },
  };
}

/**
 * Backprop through one block. dX is ∂L/∂(block output); returns ∂L/∂(block input) and the
 * gradient of each of the block's weight matrices. (The causal mask needs no special case:
 * masked attention weights are 0, so softmaxBackward gives them 0 gradient.)
 */
export function blockBackward(
  dX: Matrix, w: BlockWeights, ins: LayerInspection, cache: BlockCache, numHeads: number
): { dInput: Matrix; grads: BlockWeights } {
  const seqLen = dX.length;
  const dModel = dX[0].length;
  const headDim = dModel / numHeads;
  const attnScale = 1.0 / Math.sqrt(headDim);
  const T = MatrixMath.transpose;
  const mm = MatrixMath.matmul;

  // blockOutput = afterAttention + mlpOut  → gradient flows to both the skip path and the MLP
  const dAfterAttn = dX.map(row => row.slice());
  const dMlpOut = dX;

  // mlpOut = gelu(mlpHiddenRaw) · W2
  const gMlp2 = mm(T(ins.mlpHidden), dMlpOut);
  const dMlpHidden = mm(dMlpOut, T(w.wMlp2));

  // mlpHidden = gelu(mlpHiddenRaw)
  const dMlpRaw = MatrixMath.geluBackward(cache.mlpHiddenRaw, dMlpHidden);

  // mlpHiddenRaw = norm2 · W1
  const gMlp1 = mm(T(ins.norm2Output), dMlpRaw);
  const dNorm2 = mm(dMlpRaw, T(w.wMlp1));

  // norm2 = LN(afterAttention): the MLP branch's gradient joins the skip path's
  MatrixMath.addInPlace(dAfterAttn, MatrixMath.layerNormBackward(ins.afterAttention, dNorm2));

  // afterAttention = x + attnProjected  → skip path and attention branch both get it
  const dInput = dAfterAttn.map(row => row.slice());
  const dAttnProjected = dAfterAttn;

  // attnProjected = concat · W_O
  const gO = mm(T(ins.concatOutput), dAttnProjected);
  const dConcat = mm(dAttnProjected, T(w.wO));

  // Per head: scores = Q_h·K_hᵀ·scale, A = softmax(scores), out_h = A·V_h
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
  const gQ = mm(n1T, dQ);
  const gK = mm(n1T, dK);
  const gV = mm(n1T, dV);
  const dNorm1 = mm(dQ, T(w.wQ));
  MatrixMath.addInPlace(dNorm1, mm(dK, T(w.wK)));
  MatrixMath.addInPlace(dNorm1, mm(dV, T(w.wV)));

  // norm1 = LN(x)
  MatrixMath.addInPlace(dInput, MatrixMath.layerNormBackward(cache.input, dNorm1));

  return { dInput, grads: { wQ: gQ, wK: gK, wV: gV, wO: gO, wMlp1: gMlp1, wMlp2: gMlp2 } };
}
