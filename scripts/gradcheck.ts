// Run with: npm run gradcheck
// Verifies backprop against finite differences, then trains briefly to confirm loss falls.
import { MicroTransformer } from '../src/engine/transformer';
import { checkGradients } from '../src/engine/gradCheck';

const config = {
  vocabSize: 12,
  contextWindow: 8,
  dModel: 8,
  numHeads: 2,
  numLayers: 2,
  mlpRatio: 2,
  learningRate: 0.01,
  optimizer: 'adamw' as const,
};

const model = new MicroTransformer(config);
const seq = [1, 4, 2, 7, 4, 9, 3, 1, 5];
const input = seq.slice(0, 8);
const target = seq.slice(1, 9);

console.log('Gradient check (central differences, h = 1e-5)');
let worst = 0;
for (const r of checkGradients(model, input, target)) {
  worst = Math.max(worst, r.maxRelError);
  const flag = r.maxRelError < 1e-4 ? 'ok  ' : 'FAIL';
  console.log(`  ${flag} ${r.name.padEnd(12)} rel ${r.maxRelError.toExponential(2)}  abs ${r.maxAbsError.toExponential(2)}`);
}

console.log('\nOverfitting one sequence with AdamW (loss should approach 0):');
for (let step = 0; step <= 300; step++) {
  const { loss, gradNorm } = model.trainStep(input, target, 0.01);
  if (step % 50 === 0) console.log(`  step ${String(step).padStart(3)}  loss ${loss.toFixed(4)}  |g| ${gradNorm.toFixed(3)}`);
}

if (worst >= 1e-4) {
  console.error(`\nGradient check FAILED (worst relative error ${worst.toExponential(2)})`);
  process.exit(1);
}
console.log('\nGradient check passed.');
