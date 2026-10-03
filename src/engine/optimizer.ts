import { Matrix } from './tensor';

export type OptimizerType = 'adamw' | 'sgd';

export interface OptimizerHyperparams {
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
}

/** Saved optimizer memory (see Optimizer.exportState); matrices are flattened row by row */
export interface OptimizerState {
  type: OptimizerType;
  t: Record<string, number>;
  m: Record<string, Float64Array>;
  v: Record<string, Float64Array>;
}

const DEFAULTS: OptimizerHyperparams = {
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
  weightDecay: 1e-4,
};

/**
 * Applies a gradient to a weight matrix in place.
 *
 * SGD:   w ← w − lr · (g + λ·w)                      (L2 penalty folded into the gradient)
 * AdamW: m ← β1·m + (1−β1)·g
 *        v ← β2·v + (1−β2)·g²
 *        w ← w − lr · ( m̂ / (√v̂ + ε) + λ·w )        (decoupled weight decay)
 *
 * Adam keeps one m and one v per parameter, so state is tracked per named matrix.
 */
export class Optimizer {
  public readonly type: OptimizerType;
  public readonly hp: OptimizerHyperparams;
  private m = new Map<string, Matrix>();
  private v = new Map<string, Matrix>();
  private t = new Map<string, number>();

  constructor(type: OptimizerType, hp: Partial<OptimizerHyperparams> = {}) {
    this.type = type;
    this.hp = { ...DEFAULTS, ...hp };
  }

  /** Number of update steps applied so far to the named parameter */
  public stepsFor(name: string): number {
    return this.t.get(name) ?? 0;
  }

  /**
   * The optimizer's memory, as plain arrays that can be saved: Adam's two running averages
   * (m, v) for every weight, and how many steps each matrix has taken (for bias correction).
   */
  public exportState(): OptimizerState {
    const flat = (M: Matrix) => Float64Array.from(M.flat());
    return {
      type: this.type,
      t: Object.fromEntries(this.t),
      m: Object.fromEntries([...this.m].map(([k, M]) => [k, flat(M)])),
      v: Object.fromEntries([...this.v].map(([k, M]) => [k, flat(M)])),
    };
  }

  /** Load state saved by exportState. `params` gives each matrix's shape. */
  public importState(state: OptimizerState, params: Record<string, Matrix>): void {
    const unflat = (data: Float64Array, like: Matrix): Matrix => {
      const cols = like[0].length;
      return like.map((_, r) => Array.from(data.subarray(r * cols, (r + 1) * cols)));
    };
    this.t = new Map(Object.entries(state.t));
    this.m = new Map(Object.entries(state.m).filter(([k]) => params[k]).map(([k, d]) => [k, unflat(d, params[k])]));
    this.v = new Map(Object.entries(state.v).filter(([k]) => params[k]).map(([k, d]) => [k, unflat(d, params[k])]));
  }

  public step(name: string, param: Matrix, grad: Matrix, lr: number): void {
    const { weightDecay } = this.hp;

    if (this.type === 'sgd') {
      for (let r = 0; r < param.length; r++) {
        const p = param[r];
        const g = grad[r];
        for (let c = 0; c < p.length; c++) {
          p[c] -= lr * (g[c] + weightDecay * p[c]);
        }
      }
      this.t.set(name, this.stepsFor(name) + 1);
      return;
    }

    // AdamW
    const { beta1, beta2, eps } = this.hp;
    let m = this.m.get(name);
    let v = this.v.get(name);
    if (!m || !v) {
      m = param.map(row => new Array(row.length).fill(0));
      v = param.map(row => new Array(row.length).fill(0));
      this.m.set(name, m);
      this.v.set(name, v);
    }
    const t = this.stepsFor(name) + 1;
    this.t.set(name, t);
    const bc1 = 1 - Math.pow(beta1, t);
    const bc2 = 1 - Math.pow(beta2, t);

    for (let r = 0; r < param.length; r++) {
      const p = param[r];
      const g = grad[r];
      const mr = m[r];
      const vr = v[r];
      for (let c = 0; c < p.length; c++) {
        mr[c] = beta1 * mr[c] + (1 - beta1) * g[c];
        vr[c] = beta2 * vr[c] + (1 - beta2) * g[c] * g[c];
        const mHat = mr[c] / bc1;
        const vHat = vr[c] / bc2;
        p[c] -= lr * (mHat / (Math.sqrt(vHat) + eps) + weightDecay * p[c]);
      }
    }
  }
}
