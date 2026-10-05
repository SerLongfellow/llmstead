// Run with: npm run vlm:gradcheck
// Verifies the vision-language model's backprop against finite differences: the language model
// reading image tokens as a prefix with an answer-only loss mask, and through the prefix, the
// projector (which checks ∂L/∂prefix too). Phase 0 (empty image slots) and phase 2 are checked.
// Then it fits one batch to confirm the loss falls.
import { checkGradientsOf } from '../src/engine/gradCheck';
import { seededRandom } from '../src/engine/datasets';
import { VOCAB } from '../src/engine/vl/captions';
import { MicroClip } from '../src/engine/vl/clip';
import { DEFAULT_HELD_OUT } from '../src/engine/vl/captions';
import { MicroVlm, VlmPhase } from '../src/engine/vl/vlm';
import { VqaTask, sampleVlmBatch } from '../src/engine/vl/vlmData';

const vision = new MicroClip({
  imageSize: 24, patchSize: 8, dModel: 8, numHeads: 2, imageLayers: 1, textLayers: 1,
  mlpRatio: 2, dEmbed: 4, vocabSize: VOCAB.length, maxWords: 12, falseNegatives: 'ignore',
});
const rand = seededRandom(5);
const tasks: VqaTask[] = ['describe', 'colour', 'shape', 'where', 'size', 'there'];
const batch = sampleVlmBatch(rand, 8, tasks, DEFAULT_HELD_OUT, 3);
const features = batch.map(ex => vision.patchFeatures(ex.patches));

let worst = 0;
for (const phase of [0, 2] as VlmPhase[]) {
  const model = new MicroVlm(vision, { dModel: 8, numHeads: 2, numLayers: 2, mlpRatio: 2 });
  console.log(`Gradient check, phase ${phase} (central differences, h = 1e-5)`);
  const all = checkGradientsOf({
    params: model.getParameters(),
    grads: model.computeGradients(batch, features, phase).grads,
    lossAt: () => model.evaluateLoss(batch, features, phase),
  });
  for (const r of all) {
    // In phase 0 the projector isn't used, so both sides are 0 there
    worst = Math.max(worst, r.maxRelError);
    const flag = r.maxRelError < 1e-4 ? 'ok  ' : 'FAIL';
    console.log(`  ${flag} ${r.name.padEnd(16)} rel ${r.maxRelError.toExponential(2)}  abs ${r.maxAbsError.toExponential(2)}`);
  }
  console.log('');
}

console.log('Fitting one batch with AdamW, phase 2 (loss should fall):');
const model = new MicroVlm(vision, { dModel: 8, numHeads: 2, numLayers: 2, mlpRatio: 2 });
for (let step = 0; step <= 300; step++) {
  const r = model.trainStep(batch, features, 2, 0.01);
  if (step % 50 === 0) console.log(`  step ${String(step).padStart(3)}  loss ${r.loss.toFixed(4)}  |g| ${r.gradNorm.toFixed(3)}`);
}
batch.forEach((ex, b) => {
  const got = model.answer(model.imageTokens(features[b]), ex.questionIds).words.join(' ');
  console.log(`  "${ex.question}" → "${got}" (right: "${ex.answer}")`);
});

if (worst >= 1e-4) {
  console.error(`\nGradient check FAILED (worst relative error ${worst.toExponential(2)})`);
  process.exit(1);
}
console.log('\nGradient check passed.');
