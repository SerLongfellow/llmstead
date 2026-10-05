import { MicroTransformer } from './transformer';
import { Matrix } from './tensor';
import { LoraAdapters } from './posttrain/lora';

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
 *
 * `weights` checks the weighted loss post-training uses (see computeGradients).
 */
export function checkGradients(
  model: MicroTransformer,
  inputTokens: number[],
  targetTokens: number[],
  opts: { samplesPerMatrix?: number; h?: number; weights?: number[] } = {}
): GradCheckResult[] {
  const { grads } = model.computeGradients(inputTokens, targetTokens, opts.weights);
  const lossAt = () => model.evaluateLoss(inputTokens, targetTokens, opts.weights).loss;
  return Object.entries(model.getParameters()).map(([name, param]) =>
    checkMatrix(name, param, grads[name], lossAt, () => {}, opts)
  );
}

/**
 * The same check for any model (the vision models use it): its live parameters, its analytic
 * gradients, and a function that recomputes the loss.
 */
export function checkGradientsOf(
  model: { params: Record<string, Matrix>; grads: Record<string, Matrix>; lossAt: () => number },
  opts: { samplesPerMatrix?: number; h?: number } = {}
): GradCheckResult[] {
  return Object.entries(model.params).map(([name, param]) => checkMatrix(name, param, model.grads[name], model.lossAt, () => {}, opts));
}

/**
 * The same check for LoRA's adapters: nudge an entry of A or B, rewrite W = W₀ + s·A·B, and
 * compare the loss change with the chain-rule gradient from LoraAdapters.adapterGradients.
 */
export function checkLoraGradients(
  model: MicroTransformer,
  lora: LoraAdapters,
  inputTokens: number[],
  targetTokens: number[],
  opts: { samplesPerMatrix?: number; h?: number; weights?: number[] } = {}
): GradCheckResult[] {
  lora.write(model);
  const { grads } = model.computeGradients(inputTokens, targetTokens, opts.weights);
  const adapterGrads = lora.adapterGradients(grads);
  const lossAt = () => model.evaluateLoss(inputTokens, targetTokens, opts.weights).loss;
  const rewrite = () => lora.write(model);
  return lora.names.flatMap(name => [
    checkMatrix(`A:${name}`, lora.A[name], adapterGrads[`A:${name}`], lossAt, rewrite, opts),
    checkMatrix(`B:${name}`, lora.B[name], adapterGrads[`B:${name}`], lossAt, rewrite, opts),
  ]);
}

function checkMatrix(
  name: string,
  param: Matrix,
  grad: Matrix,
  lossAt: () => number,
  afterNudge: () => void,
  opts: { samplesPerMatrix?: number; h?: number }
): GradCheckResult {
  const samples = opts.samplesPerMatrix ?? 25;
  const h = opts.h ?? 1e-5;
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
    afterNudge();
    const lPlus = lossAt();
    param[r][c] = orig - h;
    afterNudge();
    const lMinus = lossAt();
    param[r][c] = orig;
    afterNudge();

    const numeric = (lPlus - lMinus) / (2 * h);
    const analytic = grad[r][c];
    const abs = Math.abs(analytic - numeric);
    // The floor keeps entries whose true gradient is ~0 (e.g. unused vocab rows) from reporting huge ratios.
    const rel = abs / Math.max(Math.abs(analytic) + Math.abs(numeric), 1e-8);
    maxAbs = Math.max(maxAbs, abs);
    maxRel = Math.max(maxRel, rel);
  }
  return { name, checked: n, maxRelError: maxRel, maxAbsError: maxAbs };
}
