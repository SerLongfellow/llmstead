import { MicroTransformer, globalNorm } from '../transformer';
import { Matrix, MatrixMath } from '../tensor';
import { Optimizer } from '../optimizer';

/** Which weight matrices get an adapter */
export type LoraTargets = 'attention' | 'all';

/**
 * LoRA (low-rank adaptation): freeze every weight and learn a small correction for some of them,
 *
 *   W = W₀ + s · A · B        A: [in × r], B: [r × out], s = α / r
 *
 * With r much smaller than the matrix, A and B together hold a few percent of W's numbers.
 * B starts at zero, so the model starts out exactly as it was.
 *
 * No new backward pass is needed. The model keeps using its live matrices, which this class
 * keeps equal to W₀ + s·A·B; backprop gives ∂L/∂W as usual, and the chain rule turns that into
 *
 *   ∂L/∂A = s · ∂L/∂W · Bᵀ        ∂L/∂B = s · Aᵀ · ∂L/∂W
 *
 * Every other gradient is thrown away, which is what "frozen" means.
 */
export class LoraAdapters {
  public readonly rank: number;
  public readonly scale: number;
  /** Adapted matrix names, e.g. 'wQ.0' */
  public readonly names: string[];
  private base: Record<string, Matrix> = {};
  public readonly A: Record<string, Matrix> = {};
  public readonly B: Record<string, Matrix> = {};

  constructor(model: MicroTransformer, opts: { rank: number; alpha?: number; targets?: LoraTargets }) {
    this.rank = opts.rank;
    this.scale = (opts.alpha ?? opts.rank) / opts.rank;
    const kinds = opts.targets === 'all' ? ['wQ', 'wK', 'wV', 'wO', 'wMlp1', 'wMlp2'] : ['wQ', 'wV'];
    const params = model.getParameters();
    this.names = Object.keys(params).filter(n => kinds.includes(n.split('.')[0]));
    for (const name of this.names) {
      const W = params[name];
      this.base[name] = W.map(row => row.slice());
      this.A[name] = MatrixMath.random(W.length, this.rank, 1 / Math.sqrt(W.length));
      this.B[name] = MatrixMath.zeros(this.rank, W[0].length);
    }
  }

  /** Numbers LoRA trains */
  public trainableCount(): number {
    return this.names.reduce((s, n) => s + this.A[n].length * this.rank + this.rank * this.B[n][0].length, 0);
  }

  /**
   * Turn the model's gradients into adapter gradients, clip them, step A and B, and rewrite the
   * adapted matrices. Returns the adapters' gradient norm before clipping.
   */
  public step(model: MicroTransformer, grads: Record<string, Matrix>, lr: number, optimizer: Optimizer, maxGradNorm = 1.0): number {
    const adapterGrads = this.adapterGradients(grads);
    const norm = globalNorm(adapterGrads);
    const clip = norm > maxGradNorm ? maxGradNorm / norm : 1;
    for (const [key, g] of Object.entries(adapterGrads)) {
      if (clip !== 1) for (const row of g) for (let c = 0; c < row.length; c++) row[c] *= clip;
      const [which, name] = key.split(':');
      optimizer.step(key, which === 'A' ? this.A[name] : this.B[name], g, lr);
    }
    this.write(model);
    return norm;
  }

  /** ∂L/∂A and ∂L/∂B from the model's ∂L/∂W, keyed 'A:wQ.0', 'B:wQ.0', … */
  public adapterGradients(grads: Record<string, Matrix>): Record<string, Matrix> {
    const T = MatrixMath.transpose;
    const out: Record<string, Matrix> = {};
    for (const name of this.names) {
      const dW = grads[name];
      out[`A:${name}`] = MatrixMath.scale(MatrixMath.matmul(dW, T(this.B[name])), this.scale);
      out[`B:${name}`] = MatrixMath.scale(MatrixMath.matmul(T(this.A[name]), dW), this.scale);
    }
    return out;
  }

  /** Set every adapted matrix to W₀ + s·A·B */
  public write(model: MicroTransformer) {
    const params = model.getParameters();
    for (const name of this.names) {
      const delta = MatrixMath.matmul(this.A[name], this.B[name]);
      const W = params[name];
      const W0 = this.base[name];
      for (let r = 0; r < W.length; r++) {
        const w = W[r];
        const w0 = W0[r];
        const d = delta[r];
        for (let c = 0; c < w.length; c++) w[c] = w0[c] + this.scale * d[c];
      }
    }
  }
}
