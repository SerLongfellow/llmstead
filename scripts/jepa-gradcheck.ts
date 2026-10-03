// Run with: npm run jepa:gradcheck
// Verifies the JEPA's backprop (context encoder + predictor) against finite differences, once with
// stop-gradient (the real recipe: targets are constants) and once without it (the gradient also
// flows through the target branch), and once with the variance/covariance anti-collapse term on.
// Then it trains briefly on one batch to confirm the loss falls.
import { checkGradientsOf } from '../src/engine/gradCheck';
import { DEFAULT_JEPA_CONFIG, JepaAblation, MicroJepa, sampleMask } from '../src/engine/jepa/jepa';
import { patchify, randomShapeImage } from '../src/engine/jepa/shapes';
import { seededRandom } from '../src/engine/datasets';

const config = {
  ...DEFAULT_JEPA_CONFIG,
  imageSize: 12,
  patchSize: 4,   // 3×3 = 9 patches
  dModel: 8,
  numHeads: 2,
  numLayers: 2,
  predDim: 6,
  predHeads: 2,
  predLayers: 2,
  numTargets: 1,
};

const rand = seededRandom(3);
const batch = [0, 1, 2].map(() => patchify(randomShapeImage(rand, config.imageSize), config.patchSize));
const masks = batch.map(() => sampleMask(rand, config.imageSize / config.patchSize, config.numTargets));

let worst = 0;
const cases: { label: string; ablation: JepaAblation; varWeight: number; covWeight: number }[] = [
  { label: "ablation 'none'", ablation: 'none', varWeight: 0, covWeight: 0 },
  { label: "ablation 'no-stopgrad'", ablation: 'no-stopgrad', varWeight: 0, covWeight: 0 },
  { label: 'variance + covariance term', ablation: 'none', varWeight: 1, covWeight: 0.5 },
];
for (const { label, ...settings } of cases) {
  const model = new MicroJepa({ ...config, ...settings });
  console.log(`Gradient check, ${label} (central differences, h = 1e-5)`);
  const results = checkGradientsOf({
    params: model.getParameters(),
    grads: model.computeGradients(batch, masks).grads,
    lossAt: () => model.evaluateLoss(batch, masks),
  });
  for (const r of results) {
    worst = Math.max(worst, r.maxRelError);
    const flag = r.maxRelError < 1e-4 ? 'ok  ' : 'FAIL';
    console.log(`  ${flag} ${r.name.padEnd(16)} rel ${r.maxRelError.toExponential(2)}  abs ${r.maxAbsError.toExponential(2)}`);
  }
  console.log('');
}

console.log('Fitting one batch with AdamW (loss should fall):');
const model = new MicroJepa(config);
for (let step = 0; step <= 300; step++) {
  const { loss, gradNorm } = model.trainStep(batch, masks, 0.01);
  if (step % 50 === 0) console.log(`  step ${String(step).padStart(3)}  loss ${loss.toFixed(4)}  |g| ${gradNorm.toFixed(3)}`);
}

if (worst >= 1e-4) {
  console.error(`\nGradient check FAILED (worst relative error ${worst.toExponential(2)})`);
  process.exit(1);
}
console.log('\nGradient check passed.');
