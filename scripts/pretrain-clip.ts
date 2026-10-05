// Run with: npm run pretrain-clip            (trains and writes public/models/clip-pretrained.{json,bin})
//           npm run pretrain-clip -- --check (loads the shipped files into the current engine and
//                                             confirms they still fit and score what the manifest
//                                             says, within 3 points; takes about a second)
// Trains the ready-made CLIP that the VLM mode uses by default: the default settings, a fixed
// seed, with the same engine code as the Train tab. Re-run it (and commit both files) whenever the
// CLIP's architecture or defaults change; `--check` tells you when that's needed.
// Options: --steps 6000  --seed 1
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { seededRandom } from '../src/engine/datasets';
import { MicroClip } from '../src/engine/vl/clip';
import { DEFAULT_CLIP_CONFIG, DEFAULT_CLIP_DATA, ZeroShotScore, sampleBatch } from '../src/engine/vl/clipData';
import { evalSets, runEval } from '../src/engine/vl/clipTraining';
import { PRETRAINED_BASE, PRETRAINED_FORMAT, PretrainedManifest, packWeights, unpackWeights } from '../src/engine/vl/pretrained';

const args = process.argv.slice(2);
const num = (name: string, fallback: number) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? Number(args[i + 1]) : fallback;
};
const outBase = `public/${PRETRAINED_BASE}`;
const data = DEFAULT_CLIP_DATA;
const pct = (v: number | null) => (v === null ? '–' : `${Math.round(v * 100)}%`);
const show = (label: string, z: ZeroShotScore | null) =>
  z && console.log(`  ${label.padEnd(9)} exact ${pct(z.exact)}  colour ${pct(z.colour)}  shape ${pct(z.shape)}  where ${pct(z.where)}`);
const score = (clip: MicroClip) => {
  const r = runEval(clip, evalSets(data), data, clip.steps);
  return { seen: r.seen, heldOut: r.heldOut };
};

if (args.includes('--check')) {
  const manifest: PretrainedManifest = JSON.parse(readFileSync(`${outBase}.json`, 'utf8'));
  const buf = readFileSync(`${outBase}.bin`);
  let clip: MicroClip;
  try {
    clip = unpackWeights(manifest, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  } catch (e) {
    console.error(`The shipped CLIP no longer fits this engine: ${(e as Error).message}\nRe-run: npm run pretrain-clip`);
    process.exit(1);
  }
  const now = score(clip);
  console.log(`Loaded ${outBase}.bin (${manifest.training.steps} steps). Zero-shot now:`);
  show('seen', now.seen);
  show('held-out', now.heldOut);
  // Allow a few points: a different Node version can flip a near-tie on one or two test pictures.
  // A real change to the engine moves the scores far more than that.
  const TOLERANCE = 0.03;
  const drift: string[] = [];
  for (const split of ['seen', 'heldOut'] as const) {
    const a = now[split];
    const b = manifest.scores[split];
    if (!a || !b) continue;
    for (const k of ['exact', 'colour', 'shape', 'where'] as const) {
      const d = Math.abs((a[k] ?? 0) - (b[k] ?? 0));
      if (d > TOLERANCE) drift.push(`${split} ${k}: ${pct(b[k])} in the manifest, ${pct(a[k])} now`);
    }
  }
  if (drift.length) {
    console.error(`\nScores moved by more than ${TOLERANCE * 100} points, so the engine computes something different now:\n  ${drift.join('\n  ')}\nRe-run: npm run pretrain-clip`);
    process.exit(1);
  }
  console.log('\nOK: it fits the current engine and scores what the manifest says.');
  process.exit(0);
}

const steps = num('steps', 6000);
const seed = num('seed', 1);
const batchSize = 16;
const learningRate = 0.001;
// Seed the weight initialization too, so the same command always produces the same file
let s = seed * 7919;
Math.random = () => { s = (s * 1664525 + 1013904223) % 4294967296; return s / 4294967296; };

const clip = new MicroClip({ ...DEFAULT_CLIP_CONFIG, patchSize: data.patchSize });
const rand = seededRandom(seed);
const t0 = performance.now();
for (let i = 1; i <= steps; i++) {
  const r = clip.trainStep(sampleBatch(rand, data, batchSize), learningRate);
  if (i % 1000 === 0) console.log(`step ${i}: loss ${r.loss.toFixed(3)} (${((performance.now() - t0) / 1000).toFixed(0)} s)`);
}

// Round to Float32 (what ships) and score the rounded model, so the manifest describes exactly the shipped weights
const { matrices, data: weights } = packWeights(clip);
const base: PretrainedManifest = {
  format: PRETRAINED_FORMAT,
  config: clip.config,
  data: { detail: data.detail, heldOut: data.heldOut, patchSize: data.patchSize },
  training: { seed, steps, batchSize, learningRate },
  matrices,
  scores: { seen: null as never, heldOut: null },
};
const shipped = unpackWeights(base, weights.buffer as ArrayBuffer);
const manifest: PretrainedManifest = { ...base, scores: score(shipped) };

mkdirSync('public/models', { recursive: true });
writeFileSync(`${outBase}.bin`, Buffer.from(weights.buffer));
writeFileSync(`${outBase}.json`, JSON.stringify(manifest, null, 2) + '\n');
console.log(`\nWrote ${outBase}.bin (${(weights.byteLength / 1024).toFixed(0)} KB, ${weights.length.toLocaleString()} weights) and .json. Zero-shot:`);
show('seen', manifest.scores.seen);
show('held-out', manifest.scores.heldOut);
