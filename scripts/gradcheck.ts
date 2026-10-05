// Run with: npm run gradcheck
// Verifies backprop against finite differences (plain, with post-training's per-position loss
// weights, and through LoRA adapters), then trains briefly to confirm loss falls.
import { MicroTransformer } from '../src/engine/transformer';
import { GradCheckResult, checkGradients, checkLoraGradients } from '../src/engine/gradCheck';
import { LoraAdapters } from '../src/engine/posttrain/lora';
import { MatrixMath } from '../src/engine/tensor';

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
// Signed weights, like DPO's rejected responses and RL's below-average answers; zeros, like masked prompts
const weights = [0, 0, 0.4, -0.3, 0.25, 0, -0.5, 0.7];

let worst = 0;
function report(title: string, results: GradCheckResult[]) {
  console.log(title);
  for (const r of results) {
    worst = Math.max(worst, r.maxRelError);
    const flag = r.maxRelError < 1e-4 ? 'ok  ' : 'FAIL';
    console.log(`  ${flag} ${r.name.padEnd(12)} rel ${r.maxRelError.toExponential(2)}  abs ${r.maxAbsError.toExponential(2)}`);
  }
}

report('Gradient check (central differences, h = 1e-5)', checkGradients(model, input, target));
report('\nWith per-position loss weights', checkGradients(model, input, target, { weights }));

const lora = new LoraAdapters(model, { rank: 2, alpha: 4, targets: 'all' });
// B starts at zero, which would make ∂L/∂A zero too; give it values so both are tested
for (const name of lora.names) lora.B[name] = MatrixMath.random(2, lora.B[name][0].length, 0.3);
report('\nThrough LoRA adapters (weighted loss)', checkLoraGradients(model, lora, input, target, { weights }));

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
