// Run with: npm run bench
// Times the training engine so speed changes can be measured, not guessed.
// Each measurement runs for a fixed wall-clock budget, so the script takes about the same
// time on fast and slow machines.
import { MicroTransformer } from '../src/engine/transformer';
import { MatrixMath } from '../src/engine/tensor';
import { TransformerConfig } from '../src/types';

const BUDGET_MS = 2000;

/** Call `fn` repeatedly for BUDGET_MS (after a short warm-up) and return milliseconds per call */
function msPerCall(fn: () => void): number {
  const warmUntil = performance.now() + 300;
  while (performance.now() < warmUntil) fn();
  let calls = 0;
  const t0 = performance.now();
  while (performance.now() - t0 < BUDGET_MS) {
    fn();
    calls++;
  }
  return (performance.now() - t0) / calls;
}

const base: TransformerConfig = {
  vocabSize: 120,
  contextWindow: 16,
  dModel: 32,
  numHeads: 2,
  numLayers: 2,
  mlpRatio: 4,
  learningRate: 0.001,
  optimizer: 'adamw',
};

const configs: { label: string; config: TransformerConfig }[] = [
  { label: 'default (d 32, 2 layers, ctx 16)', config: base },
  { label: 'bigger  (d 64, 4 layers, ctx 32)', config: { ...base, dModel: 64, numHeads: 4, numLayers: 4, contextWindow: 32 } },
];

for (const { label, config } of configs) {
  const model = new MicroTransformer(config);
  const seq = Array.from({ length: config.contextWindow + 1 }, (_, i) => (i * 7 + 3) % config.vocabSize);
  const input = seq.slice(0, -1);
  const target = seq.slice(1);

  const step = msPerCall(() => model.trainStep(input, target, config.learningRate));
  const forward = msPerCall(() => model.evaluateLoss(input, target));
  console.log(`${label}`);
  console.log(`  training step    ${step.toFixed(2).padStart(7)} ms   (${Math.round(1000 / step)} steps/s)`);
  console.log(`  forward only     ${forward.toFixed(2).padStart(7)} ms   (one validation window)`);
}

const A = MatrixMath.random(16, 32);
const B = MatrixMath.random(32, 128);
const mm = msPerCall(() => MatrixMath.matmul(A, B));
console.log(`matmul 16×32 · 32×128  ${(mm * 1000).toFixed(0).padStart(5)} µs`);
