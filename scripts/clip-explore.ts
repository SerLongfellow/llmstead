// Run with: npm run clip -- --steps 5000 --every 500
// Trains the vision-language model (a tiny CLIP) like the app does and prints, every --every
// steps: the training loss, the temperature, and zero-shot accuracy on fixed test pictures
// (pick the best of every possible caption), split into colour + shape combinations seen in
// training and the held-out ones it never saw. Step 0 is the untrained model: the baseline.
// Options: --batch 16  --lr 0.001  --detail shape|position|full  --fn ignore|multi
//          --dmodel 32  --img-layers 2  --txt-layers 1  --patch 6  --seed 1  --no-held-out
import { seededRandom } from '../src/engine/datasets';
import { CaptionDetail } from '../src/engine/vl/captions';
import { MicroClip } from '../src/engine/vl/clip';
import {
  ClipData, DEFAULT_CLIP_CONFIG, DEFAULT_CLIP_DATA, ZeroShotScore, captionBank, chanceExact, sampleBatch, testPairs, zeroShot
} from '../src/engine/vl/clipData';

const args = process.argv.slice(2);
const arg = (name: string, fallback: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const steps = Number(arg('steps', '5000'));
const every = Number(arg('every', '500'));
const batchSize = Number(arg('batch', '16'));
const lr = Number(arg('lr', '0.001'));
const seed = Number(arg('seed', '1'));
const data: ClipData = {
  ...DEFAULT_CLIP_DATA,
  detail: arg('detail', DEFAULT_CLIP_DATA.detail) as CaptionDetail,
  patchSize: Number(arg('patch', String(DEFAULT_CLIP_DATA.patchSize))),
  heldOut: args.includes('--no-held-out') ? [] : DEFAULT_CLIP_DATA.heldOut,
};
const config = {
  ...DEFAULT_CLIP_CONFIG,
  patchSize: data.patchSize,
  dModel: Number(arg('dmodel', String(DEFAULT_CLIP_CONFIG.dModel))),
  imageLayers: Number(arg('img-layers', String(DEFAULT_CLIP_CONFIG.imageLayers))),
  textLayers: Number(arg('txt-layers', String(DEFAULT_CLIP_CONFIG.textLayers))),
  falseNegatives: arg('fn', 'ignore') === 'multi' ? 'multi-positive' as const : 'ignore' as const,
};

let s = seed * 7919;
Math.random = () => { s = (s * 1664525 + 1013904223) % 4294967296; return s / 4294967296; };
const model = new MicroClip(config);
const rand = seededRandom(seed);
const seen = testPairs(seededRandom(1001), data, 256, 'seen');
const heldOut = testPairs(seededRandom(1002), data, 128, 'held-out');

const pct = (v: number | null) => (v === null ? '   –' : `${(v * 100).toFixed(0).padStart(3)}%`);
const fmt = (z: ZeroShotScore) => `exact ${pct(z.exact)}  colour ${pct(z.colour)}  shape ${pct(z.shape)}  where ${pct(z.where)}  size ${pct(z.size)}`;

console.log(`detail '${data.detail}', batch ${batchSize}, lr ${lr}, falseNegatives '${config.falseNegatives}', ` +
  `${model.getParameterCount().toLocaleString()} parameters, held out: ${data.heldOut.length ? 'yes' : 'no'}`);
console.log(`chance (exact): ${pct(chanceExact(data.detail))}\n`);

const report = (step: number, loss: number | null, ms: number | null) => {
  const bank = captionBank(model, data.detail);
  const head = `step ${String(step).padStart(6)}  loss ${loss === null ? '   –  ' : loss.toFixed(3)}  scale ${model.logitScale().toFixed(1).padStart(5)}` +
    (ms === null ? '' : `  ${ms.toFixed(1)} ms/step`);
  console.log(head);
  console.log(`   seen      ${fmt(zeroShot(model, seen, data.detail, bank))}`);
  if (heldOut.length) console.log(`   held-out  ${fmt(zeroShot(model, heldOut, data.detail, bank))}`);
};

report(0, null, null);
let lossSum = 0;
let t0 = performance.now();
for (let step = 1; step <= steps; step++) {
  lossSum += model.trainStep(sampleBatch(rand, data, batchSize), lr).loss;
  if (step % every === 0) {
    const ms = (performance.now() - t0) / every;
    report(step, lossSum / every, ms);
    lossSum = 0;
    t0 = performance.now();
  }
}
