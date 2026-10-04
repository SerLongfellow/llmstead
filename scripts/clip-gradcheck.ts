// Run with: npm run clip:gradcheck
// Verifies the vision-language model's backprop (both towers, the projections and the learned
// temperature) against finite differences, with plain CLIP targets and with 'multi-positive'
// targets on a batch that contains two pairs with the same caption. Then it fits one batch to
// confirm the loss falls.
import { checkGradientsOf } from '../src/engine/gradCheck';
import { seededRandom } from '../src/engine/datasets';
import { VOCAB, captionFor, captionKey, tokenizeCaption } from '../src/engine/vl/captions';
import { ClipConfig, ClipExample, FalseNegatives, MicroClip } from '../src/engine/vl/clip';
import { ShapeLabels, drawShapeImage, patchify, randomLabels } from '../src/engine/vl/shapes';

const config: ClipConfig = {
  imageSize: 12,
  patchSize: 4,   // 3×3 = 9 patches
  dModel: 8,
  numHeads: 2,
  imageLayers: 2,
  textLayers: 2,
  mlpRatio: 2,
  dEmbed: 4,
  vocabSize: VOCAB.length,
  maxWords: 12,
  falseNegatives: 'ignore',
};

const rand = seededRandom(7);
const labels: ShapeLabels[] = [0, 1, 2].map(() => randomLabels(rand));
labels.push({ ...labels[0] }); // a second pair with the same caption as the first
const batch: ClipExample[] = labels.map((l, i) => {
  const caption = captionFor(l, 'position', i % 2);
  return {
    patches: patchify(drawShapeImage(l, rand, config.imageSize), config.patchSize),
    tokens: tokenizeCaption(caption).ids,
    key: captionKey(l, 'position'),
  };
});

let worst = 0;
for (const falseNegatives of ['ignore', 'multi-positive'] as FalseNegatives[]) {
  const model = new MicroClip({ ...config, falseNegatives });
  console.log(`Gradient check, falseNegatives '${falseNegatives}' (central differences, h = 1e-5)`);
  const results = checkGradientsOf({
    params: model.getParameters(),
    grads: model.computeGradients(batch).grads,
    lossAt: () => model.evaluateLoss(batch),
  });
  for (const r of results) {
    worst = Math.max(worst, r.maxRelError);
    const flag = r.maxRelError < 1e-4 ? 'ok  ' : 'FAIL';
    console.log(`  ${flag} ${r.name.padEnd(18)} rel ${r.maxRelError.toExponential(2)}  abs ${r.maxAbsError.toExponential(2)}`);
  }
  console.log('');
}

console.log('Fitting one batch with AdamW (loss should fall):');
const model = new MicroClip(config);
for (let step = 0; step <= 300; step++) {
  const r = model.trainStep(batch, 0.01);
  if (step % 50 === 0) {
    console.log(`  step ${String(step).padStart(3)}  loss ${r.loss.toFixed(4)}  |g| ${r.gradNorm.toFixed(3)}  scale ${model.logitScale().toFixed(1)}  img→txt ${r.imageToText.toFixed(2)}`);
  }
}

if (worst >= 1e-4) {
  console.error(`\nGradient check FAILED (worst relative error ${worst.toExponential(2)})`);
  process.exit(1);
}
console.log('\nGradient check passed.');
