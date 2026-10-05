type Vec = number[];

const add = (a: Vec, b: Vec, s = 1) => a.map((v, i) => v + s * b[i]);
const dot = (a: Vec, b: Vec) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const norm = (a: Vec) => Math.sqrt(dot(a, a));

/**
 * Top-2 principal components of the (centered) embeddings via power iteration on the
 * d × d covariance matrix. Returns the mean, the two directions and the share of total
 * variance each one explains.
 */
export function pca2(rows: Vec[]) {
  const d = rows[0].length;
  const mean = new Array(d).fill(0);
  for (const r of rows) for (let j = 0; j < d; j++) mean[j] += r[j] / rows.length;
  const X = rows.map(r => r.map((v, j) => v - mean[j]));
  const C: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const x of X) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) C[i][j] += x[i] * x[j];
  const trace = C.reduce((s, row, i) => s + row[i], 0) || 1;

  const components: { dir: Vec; share: number }[] = [];
  for (let k = 0; k < 2; k++) {
    let v: Vec = Array.from({ length: d }, (_, i) => Math.sin(i + 1 + k)); // deterministic start
    for (let it = 0; it < 200; it++) {
      let w = C.map(row => dot(row, v));
      for (const { dir } of components) w = add(w, dir, -dot(w, dir)); // stay orthogonal to earlier PCs
      const n = norm(w) || 1;
      v = w.map(x => x / n);
    }
    const lambda = dot(v, C.map(row => dot(row, v)));
    components.push({ dir: v, share: lambda / trace });
  }
  return { mean, components };
}
