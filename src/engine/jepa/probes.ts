import { seededRandom } from '../datasets';
import { Optimizer } from '../optimizer';
import { Matrix, MatrixMath } from '../tensor';
import { MicroJepa } from './jepa';
import { COLOURS, SHAPES, ShapeImage, patchify } from './shapes';

/**
 * How do you tell whether a JEPA learned anything? It has no output to read: it only produces
 * embeddings. Two kinds of check:
 *
 *  1. Collapse metrics. A collapsed encoder maps every image to (nearly) the same vector, which
 *     makes the JEPA loss trivially small. Measured on a batch of image embeddings:
 *       - spread: the average standard deviation of each embedding dimension across images;
 *       - effective rank: how many directions the embeddings really use, between 1 (all on one
 *         line) and dModel (spread evenly in every direction).
 *
 *  2. Linear probes. Freeze the encoder, embed labelled images, and fit the simplest possible
 *     classifier (one matrix) to read off the shape, colour and position. If a linear map can
 *     recover them, the embeddings encode them. The same probe on raw pixels and on an untrained
 *     encoder gives the baselines to beat.
 *
 *  3. Few-label probes. With plenty of labels even raw pixels do well, which hides the difference
 *     between a good representation and a useless one. The case self-supervised learning is for is
 *     the opposite: lots of unlabelled images, very few labels. So the same probes are also fitted
 *     on only k labelled images per shape class (averaged over several random picks of the k).
 */

/**
 * One vector per image from the encoder's patch embeddings: 'mean' averages them over all patches
 * (what I-JEPA's evaluations do; loses where things are), 'concat' lays them end to end (keeps it).
 */
export function imageFeatures(model: MicroJepa, images: ShapeImage[], pool: 'mean' | 'concat' = 'mean'): Matrix {
  return images.map(img => {
    const out = model.encode(patchify(img, model.config.patchSize));
    if (pool === 'concat') return out.flat();
    const mean = new Array(out[0].length).fill(0);
    for (const row of out) for (let d = 0; d < row.length; d++) mean[d] += row[d] / out.length;
    return mean;
  });
}

/** Raw pixels as features (the baseline that needs no model at all) */
export function pixelFeatures(images: ShapeImage[]): Matrix {
  return images.map(img => img.pixels.slice());
}

/** Eigenvalues of a symmetric matrix (cyclic Jacobi rotations; fine for the ≤ 64×64 covariances here) */
export function symmetricEigenvalues(S: Matrix): number[] {
  const n = S.length;
  const a = S.map(row => row.slice());
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
    if (off < 1e-20) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-30) continue;
        // Rotate in the (p, q) plane by the angle that zeroes a[p][q]
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
      }
    }
  }
  return a.map((row, i) => row[i]);
}

export interface CollapseStats {
  spread: number;        // mean per-dimension standard deviation across images
  effectiveRank: number; // exp(entropy of the normalized covariance eigenvalues), 1…dModel
}

export function collapseStats(features: Matrix): CollapseStats {
  const n = features.length;
  const dims = features[0].length;
  const mean = new Array(dims).fill(0);
  for (const f of features) for (let d = 0; d < dims; d++) mean[d] += f[d] / n;
  const centred = features.map(f => f.map((v, d) => v - mean[d]));
  const cov = MatrixMath.scale(MatrixMath.matmul(MatrixMath.transpose(centred), centred), 1 / n);

  const spread = cov.reduce((s, row, d) => s + Math.sqrt(Math.max(0, row[d])), 0) / dims;
  const eig = symmetricEigenvalues(cov).map(v => Math.max(0, v));
  const total = eig.reduce((s, v) => s + v, 0);
  if (total <= 0) return { spread, effectiveRank: 1 };
  let entropy = 0;
  for (const v of eig) if (v > 0) entropy -= (v / total) * Math.log(v / total);
  return { spread, effectiveRank: Math.exp(entropy) };
}

/**
 * Standardize each feature with the training set's mean and std. The floor keeps a collapsed
 * dimension (std ≈ 0) from being blown back up to unit scale.
 */
function standardize(train: Matrix, test: Matrix): [Matrix, Matrix] {
  const dims = train[0].length;
  const mean = new Array(dims).fill(0);
  const std = new Array(dims).fill(0);
  for (const f of train) for (let d = 0; d < dims; d++) mean[d] += f[d] / train.length;
  for (const f of train) for (let d = 0; d < dims; d++) std[d] += (f[d] - mean[d]) ** 2 / train.length;
  for (let d = 0; d < dims; d++) std[d] = Math.max(Math.sqrt(std[d]), 1e-3);
  // A trailing 1 gives the linear map a bias
  const apply = (M: Matrix) => M.map(f => [...f.map((v, d) => (v - mean[d]) / std[d]), 1]);
  return [apply(train), apply(test)];
}

/**
 * Solve A·W = B for W, where A is symmetric positive definite (Cholesky: A = L·Lᵀ, then two
 * triangular solves per column of B).
 */
function solveSymmetric(A: Matrix, B: Matrix): Matrix {
  const n = A.length;
  const L = MatrixMath.zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i][j];
      const Li = L[i], Lj = L[j];
      for (let k = 0; k < j; k++) sum -= Li[k] * Lj[k];
      Li[j] = i === j ? Math.sqrt(Math.max(sum, 1e-12)) : sum / Lj[j];
    }
  }
  const W = MatrixMath.zeros(n, B[0].length);
  for (let c = 0; c < B[0].length; c++) {
    const y = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let sum = B[i][c];
      for (let k = 0; k < i; k++) sum -= L[i][k] * y[k];
      y[i] = sum / L[i][i];
    }
    for (let i = n - 1; i >= 0; i--) {
      let sum = y[i];
      for (let k = i + 1; k < n; k++) sum -= L[k][i] * W[k][c];
      W[i][c] = sum / L[i][i];
    }
  }
  return W;
}

function plusDiagonal(A: Matrix, lambda: number): Matrix {
  return A.map((row, i) => row.map((v, j) => (i === j ? v + lambda : v)));
}

/**
 * Ridge regression, W = argmin |X·W − Y|² + λ|W|², for any λ: returns a function of λ so the
 * expensive products are computed once. With fewer images than features it uses the equivalent
 * n×n "dual" form W = Xᵀ·(X·Xᵀ + λI)⁻¹·Y, which is much cheaper and better conditioned there.
 */
function ridgeSolver(X: Matrix, Y: Matrix): (lambda: number) => Matrix {
  const XT = MatrixMath.transpose(X);
  if (X.length < X[0].length) {
    const K = MatrixMath.matmul(X, XT);
    return lambda => MatrixMath.matmul(XT, solveSymmetric(plusDiagonal(K, lambda), Y));
  }
  const A = MatrixMath.matmul(XT, X);
  const B = MatrixMath.matmul(XT, Y);
  return lambda => solveSymmetric(plusDiagonal(A, lambda), B);
}

/**
 * Ridge with λ picked from a small grid by 4-fold cross-validation on the training images only.
 * A fixed λ can't suit both 80 labelled images and 1,536: with few images and hundreds of features
 * a weak penalty overfits wildly (test R² far below 0), while with many it underfits.
 */
function ridgeCV(X: Matrix, Y: Matrix): Matrix {
  const grid = [0.1, 1, 10, 100, 1000];
  const folds = 4;
  const errors = grid.map(() => 0);
  for (let f = 0; f < folds; f++) {
    const trainIdx = X.map((_, i) => i).filter(i => i % folds !== f);
    const heldIdx = X.map((_, i) => i).filter(i => i % folds === f);
    const solve = ridgeSolver(trainIdx.map(i => X[i]), trainIdx.map(i => Y[i]));
    const Xh = heldIdx.map(i => X[i]);
    grid.forEach((lambda, g) => {
      const pred = MatrixMath.matmul(Xh, solve(lambda));
      pred.forEach((row, r) => row.forEach((v, k) => { errors[g] += (v - Y[heldIdx[r]][k]) ** 2; }));
    });
  }
  return ridgeSolver(X, Y)(grid[errors.indexOf(Math.min(...errors))]);
}

/**
 * Fit one linear map W on the training features:
 *   'classify': softmax(X·W) against one-hot labels (cross-entropy, AdamW, full batch); returns test accuracy
 *   'regress':  X·W against real-valued targets by ridge regression (ridgeCV), with the targets
 *               centred first so the penalty doesn't pull predictions toward 0; returns test R².
 *               (Gradient descent here was unstable with hundreds of features: the same features
 *               could score 0.8 one run and below 0 the next.)
 */
function fitLinearProbe(
  Xtr: Matrix, Ytr: Matrix, Xte: Matrix, Yte: Matrix, kind: 'classify' | 'regress', iters = 400
): number {
  const XtrT = MatrixMath.transpose(Xtr);
  let W: Matrix;
  let offset = new Array(Ytr[0].length).fill(0);
  if (kind === 'regress') {
    offset = offset.map((_, k) => Ytr.reduce((s, y) => s + y[k], 0) / Ytr.length);
    W = ridgeCV(Xtr, Ytr.map(y => y.map((v, k) => v - offset[k])));
  } else {
    W = MatrixMath.zeros(Xtr[0].length, Ytr[0].length);
    const opt = new Optimizer('adamw', { weightDecay: 1e-3 });
    for (let it = 0; it < iters; it++) {
      const out = MatrixMath.softmax(MatrixMath.matmul(Xtr, W));
      // Softmax + cross-entropy: the gradient w.r.t. the logits is (probabilities − one-hot)/n
      const dOut = out.map((row, i) => row.map((v, k) => (v - Ytr[i][k]) / Xtr.length));
      opt.step('W', W, MatrixMath.matmul(XtrT, dOut), 0.02);
    }
  }

  const pred = MatrixMath.matmul(Xte, W).map(row => row.map((v, k) => v + offset[k]));
  if (kind === 'classify') {
    const argmax = (row: number[]) => row.indexOf(Math.max(...row));
    return pred.filter((row, i) => argmax(row) === argmax(Yte[i])).length / Xte.length;
  }
  // R² per output column, averaged: 1 − SSE / total variance
  let r2 = 0;
  for (let k = 0; k < Yte[0].length; k++) {
    const mean = Yte.reduce((s, y) => s + y[k], 0) / Yte.length;
    const sse = pred.reduce((s, p, i) => s + (p[k] - Yte[i][k]) ** 2, 0);
    const sst = Yte.reduce((s, y) => s + (y[k] - mean) ** 2, 0);
    r2 += 1 - sse / sst;
  }
  return r2 / Yte[0].length;
}

export interface ProbeResults {
  shape: number;    // test accuracy (chance = 1/4)
  colour: number;   // test accuracy (chance = 1/4)
  position: number; // test R² for the shape's centre (0 = no better than guessing the mean)
}

/**
 * Fit and score the probes. `labelled` picks which training images the probes may learn from
 * (default: all); the feature scaling still uses every training image, since that needs no labels.
 */
export function runProbes(
  trainX: Matrix, train: ShapeImage[], testX: Matrix, test: ShapeImage[], labelled?: number[]
): ProbeResults {
  const [XtrAll, Xte] = standardize(trainX, testX);
  const pick = labelled ?? train.map((_, i) => i);
  const Xtr = pick.map(i => XtrAll[i]);
  train = pick.map(i => train[i]);
  const oneHot = (k: number, n: number) => Array.from({ length: n }, (_, i) => (i === k ? 1 : 0));
  const labels = (imgs: ShapeImage[]) => ({
    shape: imgs.map(i => oneHot(i.labels.shape, SHAPES.length)),
    colour: imgs.map(i => oneHot(i.labels.colour, COLOURS.length)),
    position: imgs.map(i => [i.labels.x, i.labels.y]),
  });
  const ltr = labels(train);
  const lte = labels(test);
  return {
    shape: fitLinearProbe(Xtr, ltr.shape, Xte, lte.shape, 'classify'),
    colour: fitLinearProbe(Xtr, ltr.colour, Xte, lte.colour, 'classify'),
    position: fitLinearProbe(Xtr, ltr.position, Xte, lte.position, 'regress'),
  };
}

/** k random training images of each shape class */
function pickPerClass(train: ShapeImage[], perClass: number, rand: () => number): number[] {
  const picked: number[] = [];
  for (let cls = 0; cls < SHAPES.length; cls++) {
    const pool = train.map((img, i) => (img.labels.shape === cls ? i : -1)).filter(i => i >= 0);
    for (let k = 0; k < perClass && pool.length > 0; k++) picked.push(...pool.splice(Math.floor(rand() * pool.length), 1));
  }
  return picked;
}

/** Probes fitted on only `perClass` labelled images per shape class, averaged over `draws` random picks */
export function runFewLabelProbes(
  trainX: Matrix, train: ShapeImage[], testX: Matrix, test: ShapeImage[], perClass: number, draws = 5, seed = 99
): ProbeResults {
  const rand = seededRandom(seed);
  const sum: ProbeResults = { shape: 0, colour: 0, position: 0 };
  for (let d = 0; d < draws; d++) {
    const r = runProbes(trainX, train, testX, test, pickPerClass(train, perClass, rand));
    sum.shape += r.shape / draws;
    sum.colour += r.colour / draws;
    sum.position += r.position / draws;
  }
  return sum;
}
