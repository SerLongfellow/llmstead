import { Matrix } from './tensor';
import { MicroTransformer } from './transformer';

export interface GradCheckResult {
  name: string;          // parameter matrix, e.g. 'wQ.0'
  checked: number;       // entries sampled
  maxRelError: number;   // worst |analytic − numeric| / max(|analytic| + |numeric|, floor)
  maxAbsError: number;
}

/**
 * Finite-difference gradient check. For sampled weights w, nudge w by ±h and measure
 *
 *   numeric ∂L/∂w ≈ ( L(w + h) − L(w − h) ) / 2h       (central difference, error O(h²))
 *
 * then compare with the analytic gradient from backprop. With float64 and h ≈ 1e-5,
 * a correct backward pass typically agrees to a relative error below ~1e-6; a bug
 * usually shows up as errors near 1.
 */
export function checkGradients(
  model: MicroTransformer,
  inputTokens: number[],
  targetTokens: number[],
  opts: { samplesPerMatrix?: number; h?: number } = {}
): GradCheckResult[] {
  return checkGradientsOf({
    params: model.getParameters(),
    grads: model.computeGradients(inputTokens, targetTokens).grads,
    lossAt: () => model.evaluateLoss(inputTokens, targetTokens).loss,
  }, opts);
}

/** The same check for any model: its live parameters, its analytic gradients, and a loss function */
export function checkGradientsOf(
  model: { params: Record<string, Matrix>; grads: Record<string, Matrix>; lossAt: () => number },
  opts: { samplesPerMatrix?: number; h?: number } = {}
): GradCheckResult[] {
  const samples = opts.samplesPerMatrix ?? 25;
  const h = opts.h ?? 1e-5;
  const { params, grads, lossAt } = model;

  const results: GradCheckResult[] = [];
  for (const [name, param] of Object.entries(params)) {
    const rows = param.length;
    const cols = param[0].length;
    let maxRel = 0;
    let maxAbs = 0;
    const n = Math.min(samples, rows * cols);
    for (let s = 0; s < n; s++) {
      const r = Math.floor(Math.random() * rows);
      const c = Math.floor(Math.random() * cols);
      const orig = param[r][c];
      param[r][c] = orig + h;
      const lPlus = lossAt();
      param[r][c] = orig - h;
      const lMinus = lossAt();
      param[r][c] = orig;

      const numeric = (lPlus - lMinus) / (2 * h);
      const analytic = grads[name][r][c];
      const abs = Math.abs(analytic - numeric);
      // The floor keeps entries whose true gradient is ~0 (e.g. unused vocab rows) from reporting huge ratios.
      const rel = abs / Math.max(Math.abs(analytic) + Math.abs(numeric), 1e-8);
      maxAbs = Math.max(maxAbs, abs);
      maxRel = Math.max(maxRel, rel);
    }
    results.push({ name, checked: n, maxRelError: maxRel, maxAbsError: maxAbs });
  }
  return results;
}
