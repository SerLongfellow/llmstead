// Matrix operations & math helpers for the Micro-Transformer engine

export type Matrix = number[][]; // [rows][cols]
export type Vector = number[];

export class MatrixMath {
  /** Create a matrix initialized with zeros */
  public static zeros(rows: number, cols: number): Matrix {
    return Array.from({ length: rows }, () => new Array(cols).fill(0));
  }

  /** Create a 3D matrix initialized with zeros */
  public static zeros3D(dim1: number, dim2: number, dim3: number): number[][][] {
    return Array.from({ length: dim1 }, () =>
      Array.from({ length: dim2 }, () => new Array(dim3).fill(0))
    );
  }

  /** Initialize matrix with random uniform values in range [-scale, scale] */
  public static random(rows: number, cols: number, scale: number = 0.1): Matrix {
    const mat: Matrix = [];
    for (let r = 0; r < rows; r++) {
      const row: number[] = [];
      for (let c = 0; c < cols; c++) {
        // Gaussian-like approximation via Box-Muller transform for smooth weight init
        const u1 = Math.random() || 1e-7;
        const u2 = Math.random();
        const randStdNorm = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        row.push(randStdNorm * scale);
      }
      mat.push(row);
    }
    return mat;
  }

  /** Matrix Multiplication A [M x K] * B [K x N] -> C [M x N] */
  public static matmul(A: Matrix, B: Matrix): Matrix {
    const M = A.length;
    const K = A[0].length;
    const N = B[0].length;

    if (K !== B.length) {
      throw new Error(`Matrix dimensions mismatch: A[${M}x${K}] * B[${B.length}x${N}]`);
    }

    const C: Matrix = MatrixMath.zeros(M, N);
    for (let i = 0; i < M; i++) {
      for (let k = 0; k < K; k++) {
        const aVal = A[i][k];
        for (let j = 0; j < N; j++) {
          C[i][j] += aVal * B[k][j];
        }
      }
    }
    return C;
  }

  /** Transpose Matrix [M x N] -> [N x M] */
  public static transpose(A: Matrix): Matrix {
    const M = A.length;
    const N = A[0].length;
    const T: Matrix = MatrixMath.zeros(N, M);
    for (let r = 0; r < M; r++) {
      for (let c = 0; c < N; c++) {
        T[c][r] = A[r][c];
      }
    }
    return T;
  }

  /** Element-wise Addition: A + B */
  public static add(A: Matrix, B: Matrix): Matrix {
    const M = A.length;
    const N = A[0].length;
    const C: Matrix = MatrixMath.zeros(M, N);
    for (let r = 0; r < M; r++) {
      for (let c = 0; c < N; c++) {
        C[r][c] = A[r][c] + B[r][c];
      }
    }
    return C;
  }

  /** Scale matrix by scalar value */
  public static scale(A: Matrix, s: number): Matrix {
    return A.map(row => row.map(val => val * s));
  }

  /** Apply GELU activation element-wise */
  public static gelu(A: Matrix): Matrix {
    return A.map(row =>
      row.map(x => {
        // Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
      })
    );
  }

  /** Element-wise in-place accumulation: A += B */
  public static addInPlace(A: Matrix, B: Matrix): void {
    for (let r = 0; r < A.length; r++) {
      for (let c = 0; c < A[r].length; c++) {
        A[r][c] += B[r][c];
      }
    }
  }

  /**
   * GELU backward. For the tanh approximation g(x) = ½·x·(1 + tanh(u)), u = c·(x + a·x³):
   *   g'(x) = ½·(1 + tanh u) + ½·x·(1 − tanh² u)·c·(1 + 3a·x²)
   * Returns dX = dY ⊙ g'(X), where X is the *pre-activation* input.
   */
  public static geluBackward(X: Matrix, dY: Matrix): Matrix {
    const c = Math.sqrt(2.0 / Math.PI);
    const a = 0.044715;
    return X.map((row, r) =>
      row.map((x, j) => {
        const t = Math.tanh(c * (x + a * x * x * x));
        const dg = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * a * x * x);
        return dY[r][j] * dg;
      })
    );
  }

  /**
   * LayerNorm backward (no gain/bias), row by row. With y = (x − μ)/σ over n features:
   *   dx = (1/σ) · ( dy − mean(dy) − y · mean(dy ⊙ y) )
   * The two mean terms appear because every x_j also moves μ and σ.
   * X is the LayerNorm *input*; eps must match the forward pass.
   */
  public static layerNormBackward(X: Matrix, dY: Matrix, eps: number = 1e-5): Matrix {
    const n = X[0].length;
    return X.map((row, r) => {
      const mean = row.reduce((s, v) => s + v, 0) / n;
      const variance = row.reduce((s, v) => s + (v - mean) * (v - mean), 0) / n;
      const stdDev = Math.sqrt(variance + eps);
      const y = row.map(v => (v - mean) / stdDev);
      const dy = dY[r];
      let meanDy = 0;
      let meanDyY = 0;
      for (let j = 0; j < n; j++) {
        meanDy += dy[j];
        meanDyY += dy[j] * y[j];
      }
      meanDy /= n;
      meanDyY /= n;
      return y.map((yj, j) => (dy[j] - meanDy - yj * meanDyY) / stdDev);
    });
  }

  /**
   * Softmax backward, row by row. With p = softmax(s):
   *   ds_j = p_j · ( dp_j − Σ_k p_k·dp_k )
   * P is the softmax *output*. Causally masked entries have p = 0, so their gradient
   * is 0 automatically — the mask needs no special handling here.
   */
  public static softmaxBackward(P: Matrix, dP: Matrix): Matrix {
    return P.map((p, r) => {
      const dp = dP[r];
      let dot = 0;
      for (let k = 0; k < p.length; k++) dot += p[k] * dp[k];
      return p.map((pj, j) => pj * (dp[j] - dot));
    });
  }

  /** Layer Normalization along embedding dimension */
  public static layerNorm(X: Matrix, eps: number = 1e-5): Matrix {
    const seqLen = X.length;
    const dModel = X[0].length;
    const Y: Matrix = MatrixMath.zeros(seqLen, dModel);

    for (let i = 0; i < seqLen; i++) {
      // Mean
      const mean = X[i].reduce((sum, val) => sum + val, 0) / dModel;
      // Variance
      const variance = X[i].reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / dModel;
      const stdDev = Math.sqrt(variance + eps);

      for (let j = 0; j < dModel; j++) {
        Y[i][j] = (X[i][j] - mean) / stdDev;
      }
    }
    return Y;
  }

  /** Softmax along the last dimension with optional Causal Masking */
  public static softmax(X: Matrix, applyCausalMask: boolean = false): Matrix {
    const seqLen = X.length;
    const cols = X[0].length;
    const result: Matrix = MatrixMath.zeros(seqLen, cols);

    for (let i = 0; i < seqLen; i++) {
      let maxVal = -Infinity;
      for (let j = 0; j < cols; j++) {
        if (applyCausalMask && j > i) continue; // Mask future tokens
        if (X[i][j] > maxVal) maxVal = X[i][j];
      }

      let expSum = 0;
      const exps = new Array(cols).fill(0);
      for (let j = 0; j < cols; j++) {
        if (applyCausalMask && j > i) {
          exps[j] = 0;
        } else {
          exps[j] = Math.exp(X[i][j] - maxVal);
          expSum += exps[j];
        }
      }

      for (let j = 0; j < cols; j++) {
        result[i][j] = expSum > 0 ? exps[j] / expSum : 0;
      }
    }

    return result;
  }
}
