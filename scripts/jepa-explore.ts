// Run with: npm run jepa -- [--steps 5000] [--grid 4] [--batch 8] [--lr 0.001] [--every 1000]
//                         [--ablation none|no-ema|no-stopgrad|all] [--targets 2] [--ema 0.996] [--dim 32]
//                         [--shapes small|big] [--probe-n 1536] [--var 0] [--cov 0] [--no-baseline]
// Trains the tiny I-JEPA on procedurally drawn shapes and reports, every few steps: the loss,
// collapse metrics (embedding spread, effective rank) and linear-probe scores on frozen
// embeddings, next to a raw-pixel baseline. Probes run on the patch embeddings averaged into one
// vector ("mean") and laid end to end ("concat", keeps layout), each fitted with 5 or 20
// labelled images per shape class (averaged over 5 random picks) or with every probe image.
// Colour is ~100% for every feature set, so only the pixel baseline prints it.
// Effective rank is shown as "-" once the spread is ~0 (fully collapsed): it would only be
// measuring rounding noise. Step 0 is the untrained encoder. Math.random (weight init) is seeded,
// so runs are repeatable. Run in the foreground: Windows throttles background processes heavily.
import { DEFAULT_JEPA_CONFIG, JepaAblation, MicroJepa, sampleMask } from '../src/engine/jepa/jepa';
import { Matrix } from '../src/engine/tensor';
import { collapseStats, imageFeatures, pixelFeatures, probeImageSets, runFewLabelProbes, runProbes } from '../src/engine/jepa/probes';
import { BIG_SHAPES, SMALL_SHAPES, patchify, randomShapeImage } from '../src/engine/jepa/shapes';
import { seededRandom } from '../src/engine/datasets';

let seed = 12345;
Math.random = () => {
  seed = (seed * 1664525 + 1013904223) % 4294967296;
  return seed / 4294967296;
};

const args = process.argv.slice(2);
const arg = (name: string, fallback: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const STEPS = Number(arg('steps', '5000'));
const TARGETS = Number(arg('targets', String(DEFAULT_JEPA_CONFIG.numTargets)));
const EMA = Number(arg('ema', String(DEFAULT_JEPA_CONFIG.emaStart)));
const DIM = Number(arg('dim', String(DEFAULT_JEPA_CONFIG.dModel)));
const SIZES = arg('shapes', 'small') === 'big' ? BIG_SHAPES : SMALL_SHAPES;
const PROBE_N = Number(arg('probe-n', '1536'));
const VAR = Number(arg('var', '0'));
const COV = Number(arg('cov', '0'));
const GRID = Number(arg('grid', '4'));
const BATCH = Number(arg('batch', '8'));
const LR = Number(arg('lr', '0.001'));
const EVERY = Number(arg('every', '1000'));
const ablationArg = arg('ablation', 'none');
const ablations: JepaAblation[] = ablationArg === 'all' ? ['none', 'no-ema', 'no-stopgrad'] : [ablationArg as JepaAblation];

const PATCH = DEFAULT_JEPA_CONFIG.patchSize;
const IMAGE = GRID * PATCH;
const { train: probeTrain, test: probeTest } = probeImageSets(SIZES, PROBE_N, 512, IMAGE);

const pct = (v: number) => `${(v * 100).toFixed(0).padStart(3)}%`;
const fmt = (v: number) => v.toFixed(2).padStart(5);

console.log(`I-JEPA on ${IMAGE}×${IMAGE} shapes, ${GRID}×${GRID} patches of ${PATCH}×${PATCH}; batch ${BATCH}, lr ${LR}, ${STEPS} steps, ${TARGETS} target blocks, EMA from ${EMA}, d ${DIM}, ${SIZES === BIG_SHAPES ? 'big' : 'small'} shapes, ${PROBE_N} probe images, var ${VAR} cov ${COV}`);
/** Shape accuracy and position R² with 5 / 20 labels per class and with all of them, as table cells */
function probeCells(train: Matrix, test: Matrix): string {
  const few5 = runFewLabelProbes(train, probeTrain, test, probeTest, 5);
  const few20 = runFewLabelProbes(train, probeTrain, test, probeTest, 20);
  const all = runProbes(train, probeTrain, test, probeTest);
  return `   ${pct(few5.shape)} ${pct(few20.shape)} ${pct(all.shape)}   ${fmt(few20.position)} ${fmt(all.position)}`;
}
const HEADER = 'shape@5  @20 @all  pos@20  @all';

// --no-baseline skips the raw-pixel probes and the step-0 (untrained) row, which are identical
// for every run with the same --shapes/--grid; handy when running several settings side by side.
const BASELINE = !args.includes('--no-baseline');
if (BASELINE) {
  const pix = runProbes(pixelFeatures(probeTrain), probeTrain, pixelFeatures(probeTest), probeTest);
  console.log(`raw pixels: colour ${pct(pix.colour)}   (chance: 25% for shape and colour, 0 for position R²)`);
  console.log(`                                     ${HEADER}`);
  console.log(`  raw pixels                        ${probeCells(pixelFeatures(probeTrain), pixelFeatures(probeTest))}\n`);
}

for (const ablation of ablations) {
  seed = 12345;
  const model = new MicroJepa({
    ...DEFAULT_JEPA_CONFIG, imageSize: IMAGE, dModel: DIM, numTargets: TARGETS, emaStart: EMA, emaSteps: STEPS, ablation,
    varWeight: VAR, covWeight: COV,
  });
  const rand = seededRandom(7);
  console.log(`── ablation: ${ablation}  (${model.getParameterCount().toLocaleString()} trainable parameters)`);
  console.log('                                  ───────── mean-pooled ──────────    ──────────── concat ────────────');
  console.log(`   step    loss   spread  eff.rank   ${HEADER}    ${HEADER}   ms/step`);

  let lossSum = 0;
  let lossCount = 0;
  let ms = 0;
  for (let step = 0; step <= STEPS; step++) {
    if (step % EVERY === 0 && (step > 0 || BASELINE)) {
      const ftr = imageFeatures(model, probeTrain);
      const stats = collapseStats(ftr);
      const meanCells = probeCells(ftr, imageFeatures(model, probeTest));
      const concatCells = probeCells(imageFeatures(model, probeTrain, 'concat'), imageFeatures(model, probeTest, 'concat'));
      const loss = lossCount ? (lossSum / lossCount).toFixed(4) : '   -  ';
      const speed = lossCount ? (ms / lossCount).toFixed(1) : '-';
      console.log(
        `  ${String(step).padStart(5)}  ${loss}   ${stats.spread < 0.01 ? stats.spread.toExponential(0).padStart(5) : fmt(stats.spread)}   ${stats.spread < 0.01 ? "   - " : fmt(stats.effectiveRank)}    ` +
        `${meanCells} ${concatCells}   ${speed.padStart(6)}`
      );
      lossSum = lossCount = ms = 0;
    }
    if (step === STEPS) break;

    const images = Array.from({ length: BATCH }, () => randomShapeImage(rand, IMAGE, SIZES));
    const batch = images.map(img => patchify(img, PATCH));
    const masks = batch.map(() => sampleMask(rand, GRID, TARGETS));
    const t0 = performance.now();
    // The prediction loss only (without the variance/covariance term), so runs stay comparable
    const { predLoss: loss } = model.trainStep(batch, masks, LR);
    ms += performance.now() - t0;
    lossSum += loss;
    lossCount++;
  }
  console.log('');
}
