import { LayerInspection } from '../types';
import { Matrix, MatrixMath } from './tensor';

/**
 * One transformer block, shared by every model in the engine (the GPT's decoder and the
 * JEPA's encoders and predictor). POST-LayerNorm layout, no biases:
 *
 *   x ─► Attention ─► (+x) ─► LayerNorm ─► MLP ─► (+) ─► LayerNorm ─► output
 *
 * The only switch is `causal`: a GPT must not let position i look at later positions (it is
 * predicting them), while an encoder that sees a whole image lets every patch attend to every other.
 */

/** One block's weight matrices */
export interface BlockWeights {
  wQ: Matrix;    // [dModel x dModel]
  wK: Matrix;    // [dModel x dModel]
  wV: Matrix;    // [dModel x dModel]
  wO: Matrix;    // [dModel x dModel]
  wMlp1: Matrix; // [dModel x dMlp]
  wMlp2: Matrix; // [dMlp x dModel]
}

/** Values from the forward pass that the visualizers don't need but backprop does */
export interface BlockCache {
  input: Matrix;        // block input x
  res1: Matrix;         // x + attention output (input to LayerNorm 1)
  mlpHiddenRaw: Matrix; // MLP pre-activation (input to GELU)
  res2: Matrix;         // norm1 + MLP output (input to LayerNorm 2)
}

export const BLOCK_WEIGHT_NAMES = ['wQ', 'wK', 'wV', 'wO', 'wMlp1', 'wMlp2'] as const;

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

/** Forward through one block. The block's output is `inspection.norm2Output`. */
export function blockForward(
  x: Matrix,
  w: BlockWeights,
  numHeads: number,
  causal: boolean,
  layerIndex: number
): { inspection: LayerInspection; cache: BlockCache } {
  const seqLen = x.length;
  const dModel = x[0].length;
  const headDim = dModel / numHeads;

  // Q, K, V projections: [seqLen x dModel] * [dModel x dModel] -> [seqLen x dModel]
  const fullQ = MatrixMath.matmul(x, w.wQ);
  const fullK = MatrixMath.matmul(x, w.wK);
  const fullV = MatrixMath.matmul(x, w.wV);

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

    // Softmax (causal or not)
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

  // Post-LN: residual add FIRST, then normalize the sum
  const res1 = MatrixMath.add(x, attnProjected);
  const norm1Output = MatrixMath.layerNorm(res1);

  // MLP / Feed-Forward Network on the normalized stream: MLP1 -> GELU -> MLP2
  const mlpHiddenRaw = MatrixMath.matmul(norm1Output, w.wMlp1);
  const mlpHidden = MatrixMath.gelu(mlpHiddenRaw);
  const mlpOutput = MatrixMath.matmul(mlpHidden, w.wMlp2);

  // Post-LN: residual add (around the MLP), then normalize again
  const res2 = MatrixMath.add(norm1Output, mlpOutput);
  const norm2Output = MatrixMath.layerNorm(res2);

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
      mlpHidden,
      mlpOutput,
      norm2Output
    },
    cache: { input: x, res1, mlpHiddenRaw, res2 }
  };
}

/**
 * Backward through one block. dX enters as ∂L/∂(block output) and leaves as ∂L/∂(block input),
 * along with ∂L/∂(each weight). Local rules:
 *
 *   matmul     C = A·B          dA = dC·Bᵀ,  dB = Aᵀ·dC
 *   residual   y = a + b        da = dy,     db = dy     (gradient is copied to both branches)
 *   LayerNorm, GELU, softmax    see MatrixMath.*Backward
 */
export function blockBackward(
  dX: Matrix,
  w: BlockWeights,
  ins: LayerInspection,
  cache: BlockCache,
  numHeads: number
): { dInput: Matrix; grads: BlockWeights } {
  const seqLen = dX.length;
  const dModel = dX[0].length;
  const headDim = dModel / numHeads;
  const attnScale = 1.0 / Math.sqrt(headDim);
  const T = MatrixMath.transpose;
  const mm = MatrixMath.matmul;

  // norm2 = LN(res2)
  const dRes2 = MatrixMath.layerNormBackward(cache.res2, dX);

  // res2 = norm1 + mlpOut  → gradient flows to both the skip path and the MLP
  const dNorm1 = dRes2.map(row => row.slice());
  const dMlpOut = dRes2;

  // mlpOut = gelu(mlpHiddenRaw) · W2
  const gMlp2 = mm(T(ins.mlpHidden), dMlpOut);
  const dMlpHidden = mm(dMlpOut, T(w.wMlp2));

  // mlpHidden = gelu(mlpHiddenRaw)
  const dMlpRaw = MatrixMath.geluBackward(cache.mlpHiddenRaw, dMlpHidden);

  // mlpHiddenRaw = norm1 · W1
  const gMlp1 = mm(T(ins.norm1Output), dMlpRaw);
  MatrixMath.addInPlace(dNorm1, mm(dMlpRaw, T(w.wMlp1)));

  // norm1 = LN(res1)
  const dRes1 = MatrixMath.layerNormBackward(cache.res1, dNorm1);

  // res1 = x + attnProjected  → skip path and attention branch both get dRes1
  const dInput = dRes1.map(row => row.slice());
  const dAttnProjected = dRes1;

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

  // Q = x·W_Q, K = x·W_K, V = x·W_V  → x gets gradient from all three projections
  const xT = T(cache.input);
  const gQ = mm(xT, dQ);
  const gK = mm(xT, dK);
  const gV = mm(xT, dV);
  MatrixMath.addInPlace(dInput, mm(dQ, T(w.wQ)));
  MatrixMath.addInPlace(dInput, mm(dK, T(w.wK)));
  MatrixMath.addInPlace(dInput, mm(dV, T(w.wV)));

  return { dInput, grads: { wQ: gQ, wK: gK, wV: gV, wO: gO, wMlp1: gMlp1, wMlp2: gMlp2 } };
}
