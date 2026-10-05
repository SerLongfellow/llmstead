// Run with: npm run vlm -- --clip-steps 3000 --phase0 1000 --phase1 1000 --phase2 4000
// Trains a CLIP (stage 1) like the app, then the vision-language model on top of its frozen
// image tower, phase by phase (0 text only, 1 align the projector, 2 instruction-tune), and
// prints after each phase how often each kind of question is answered exactly right: with the
// picture, and with a blank picture instead (if those match, the model isn't looking).
// Options: --batch 8  --lr 0.001 (--lr0/--lr1/--lr2 per phase)  --every 0 (also report every N steps within a phase)
//          --clip-steps 0 (a random, untrained vision encoder)  --seed 1  --no-held-out
//          --phase0/1/2 0 skips that phase
import { seededRandom } from '../src/engine/datasets';
import { MicroClip } from '../src/engine/vl/clip';
import { DEFAULT_CLIP_CONFIG, DEFAULT_CLIP_DATA, sampleBatch } from '../src/engine/vl/clipData';
import { DEFAULT_VLM_CONFIG, MicroVlm, VlmPhase } from '../src/engine/vl/vlm';
import { VQA_TASKS, VqaTask } from '../src/engine/vl/vlmData';
import { VlmEvalResult, runVlmEval, trainVlmChunk, vlmEvalSet } from '../src/engine/vl/vlmTraining';

const args = process.argv.slice(2);
const num = (name: string, fallback: number) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? Number(args[i + 1]) : fallback;
};
const seed = num('seed', 1);
let s = seed * 7919;
Math.random = () => { s = (s * 1664525 + 1013904223) % 4294967296; return s / 4294967296; };

const heldOut = args.includes('--no-held-out') ? [] : DEFAULT_CLIP_DATA.heldOut;
const clipData = { ...DEFAULT_CLIP_DATA, heldOut };
const clipSteps = num('clip-steps', 3000);
const batchSize = num('batch', 8);
const lr = num('lr', 0.001);
const lrFor = (phase: VlmPhase) => num(`lr${phase}`, lr);
const every = num('every', 0);
const phaseSteps: Record<VlmPhase, number> = { 0: num('phase0', 1000), 1: num('phase1', 1000), 2: num('phase2', 4000) };
const tasks: VqaTask[] = VQA_TASKS.map(t => t.id);

// ── Stage 1: the vision encoder ──
const vision = new MicroClip(DEFAULT_CLIP_CONFIG);
const clipRand = seededRandom(seed);
let t0 = performance.now();
for (let i = 1; i <= clipSteps; i++) {
  const r = vision.trainStep(sampleBatch(clipRand, clipData, 16), 0.001);
  if (i % 1000 === 0) console.log(`CLIP step ${i}: loss ${r.loss.toFixed(3)}`);
}
if (clipSteps) console.log(`CLIP: ${clipSteps} steps in ${((performance.now() - t0) / 1000).toFixed(0)} s\n`);
else console.log('CLIP: none (random, untrained vision encoder)\n');

// ── Stage 2 ──
const model = new MicroVlm(vision, DEFAULT_VLM_CONFIG);
const set = vlmEvalSet(clipData.patchSize, heldOut);
const data = { patchSize: clipData.patchSize, heldOut, tasks };
const rand = seededRandom(seed + 1);
const p = model.getParameterCount();
console.log(`VLM: projector ${p.projector}, language model ${p.lm}, frozen vision ${p.vision} parameters; batch ${batchSize}, lr ${lr}\n`);

const pct = (v: number | null) => (v === null ? '  –' : `${Math.round(v * 100)}%`.padStart(4));
const report = (r: VlmEvalResult, label: string) => {
  console.log(label);
  console.log('   task       seen  blank   never-seen  blank');
  for (const t of VQA_TASKS) {
    const x = r.tasks[t.id];
    console.log(`   ${t.id.padEnd(9)} ${pct(x.seen)}  ${pct(x.seenBlank)}        ${pct(x.heldOut)}  ${pct(x.heldOutBlank)}`);
  }
};

report(runVlmEval(model, set, 0, 0), 'Untrained');
let step = 0;
for (const phase of [0, 1, 2] as VlmPhase[]) {
  const n = phaseSteps[phase];
  if (!n) {
    console.log(`\nPhase ${phase}: skipped`);
    continue;
  }
  t0 = performance.now();
  let lossSum = 0;
  for (let i = 1; i <= n; i++) {
    lossSum += trainVlmChunk(model, data, { phase, batchSize, learningRate: lrFor(phase), budgetMs: 0, maxSteps: 1, rand }).lossSum;
    step++;
    if (every && i % every === 0 && i < n) report(runVlmEval(model, set, step, phase), `  phase ${phase}, step ${i}: loss ${(lossSum / i).toFixed(3)}`);
  }
  const ms = (performance.now() - t0) / n;
  report(runVlmEval(model, set, step, phase), `\nAfter phase ${phase} (${n} steps, mean loss ${(lossSum / n).toFixed(3)}, ${ms.toFixed(1)} ms/step)`);
}

// A few answers in words
console.log('\nSample answers (seen pictures):');
for (const pic of set.pictures.slice(0, 3)) {
  const tokens = model.imageTokens(model.features(pic.patches));
  for (const ex of pic.examples) {
    const got = model.answer(tokens, ex.questionIds).words.join(' ');
    console.log(`  ${ex.question.padEnd(28)} → ${got.padEnd(28)} ${got === ex.answer ? 'ok' : `(right: ${ex.answer})`}`);
  }
}
