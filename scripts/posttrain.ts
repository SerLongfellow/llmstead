// Run with: npm run posttrain [-- dataset-id ...] [--method sft,dpo,rl] [--steps N] [--lr X] [--lora R] [--beta B] [--kl B]
// Pre-trains each sample dataset like `npm run evals` (seeded, default model; the weights are
// cached in node_modules/.cache), then runs the Post-train tab's presets on a copy and prints
// what changed: the method's own metrics, the try-it prompts before and after, the benchmark,
// and the pre-training validation loss (the cost: forgetting).
import * as fs from 'fs';
import { MicroTransformer, ModelState } from '../src/engine/transformer';
import { BPETokenizer } from '../src/engine/bpeTokenizer';
import { SAMPLE_DATASETS, splitDataset } from '../src/engine/datasets';
import { BENCHMARK_SUITES, runBenchmarkSuite } from '../src/engine/benchmarks';
import { generateTokens } from '../src/engine/generate';
import { validationLoss } from '../src/engine/training';
import { makeTrainer } from '../src/engine/posttrain/examples';
import { LoraAdapters } from '../src/engine/posttrain/lora';
import { prepareSft, sftStep } from '../src/engine/posttrain/sft';
import { dpoStep, preparePairs, scorePair, summarize } from '../src/engine/posttrain/dpo';
import { greedyAccuracy, rlStep } from '../src/engine/posttrain/grpo';
import { DPO_PRESETS, RL_PRESETS, SFT_PRESETS } from '../src/engine/posttrain/presets';

let seed = 12345;
Math.random = () => {
  seed = (seed * 1664525 + 1013904223) % 4294967296;
  return seed / 4294967296;
};

const args = process.argv.slice(2);
const opt = (name: string, fallback: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : fallback;
};
const valued = new Set(['--task', '--method', '--steps', '--lr', '--lora', '--beta', '--kl', '--pretrain']);
const only = args.filter((a, i) => !a.startsWith('--') && !valued.has(args[i - 1]));
const methods = opt('method', 'sft,dpo,rl').split(',');
const loraRank = Number(opt('lora', '0'));
const PRETRAIN = Number(opt('pretrain', '8000'));

// The app's defaults (App.tsx)
const VOCAB = 120;
const CONTEXT = 16;
const LR = 0.001;

function pretrained(dsId: string, text: string, tokenizer: BPETokenizer, vocabSize: number): MicroTransformer {
  const model = new MicroTransformer({
    vocabSize, contextWindow: CONTEXT, dModel: 32, numHeads: 2, numLayers: 2, mlpRatio: 4, learningRate: LR, optimizer: 'adamw',
  });
  const file = `node_modules/.cache/posttrain-${dsId}-${PRETRAIN}.json`;
  if (fs.existsSync(file)) {
    const saved = JSON.parse(fs.readFileSync(file, 'utf8'));
    const toF = (o: Record<string, number[]>) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, Float64Array.from(v)]));
    model.importState({ weights: toF(saved.weights), optimizer: { type: 'adamw', t: saved.t, m: toF(saved.m), v: toF(saved.v) } });
    return model;
  }
  const train = tokenizer.encode(splitDataset(text).trainText).tokens;
  for (let step = 0; step < PRETRAIN; step++) {
    const i = Math.floor(Math.random() * Math.max(1, train.length - CONTEXT - 1));
    model.trainStep(train.slice(i, i + CONTEXT), train.slice(i + 1, i + CONTEXT + 1), LR);
  }
  const s: ModelState = model.exportState();
  const toA = (o: Record<string, Float64Array>) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, Array.from(v)]));
  fs.writeFileSync(file, JSON.stringify({ weights: toA(s.weights), t: s.optimizer.t, m: toA(s.optimizer.m), v: toA(s.optimizer.v) }));
  return model;
}

function clone(model: MicroTransformer): MicroTransformer {
  const c = new MicroTransformer(model.config);
  c.importState(model.exportState());
  return c;
}

const fmt = (x: number, d = 3) => x.toFixed(d);

for (const ds of SAMPLE_DATASETS) {
  if (only.length && !only.includes(ds.id)) continue;
  seed = 12345;
  const tokenizer = new BPETokenizer();
  tokenizer.train(ds.text, VOCAB);
  const vocabSize = tokenizer.getVocabSize();
  const split = splitDataset(ds.text);
  const valTokens = tokenizer.encode(split.valText).tokens;
  const base = pretrained(ds.id, ds.text, tokenizer, vocabSize);
  const eos = tokenizer.specialId('<EOS>');
  const suite = BENCHMARK_SUITES[ds.id];

  const bench = (m: MicroTransformer) => {
    const r = runBenchmarkSuite(suite, m, tokenizer, split.trainText).bySplit;
    return `seen ${r.seen.passed}/${r.seen.total} held-out ${r['held-out'].passed}/${r['held-out'].total}`;
  };
  const show = (m: MicroTransformer, prompt: string) => {
    const g = generateTokens(m, tokenizer, prompt, { maxTokens: 12, temperature: 0, stopToken: eos });
    return JSON.stringify(g.text) + (g.stopped ? ' <EOS>' : '');
  };
  const compare = (after: MicroTransformer, prompts: string[]) => {
    for (const p of prompts) console.log(`      ${JSON.stringify(p).padEnd(28)} base ${show(base, p).padEnd(26)} after ${show(after, p)}`);
  };
  const setup = (m: MicroTransformer) => {
    const lora = loraRank > 0 ? new LoraAdapters(m, { rank: loraRank, targets: 'all' }) : null;
    if (lora) console.log(`    LoRA r=${loraRank}: ${lora.trainableCount()} trainable of ${m.getParameterCount()}`);
    return makeTrainer(lora);
  };

  console.log(`\n${ds.name}   base: val loss ${fmt(validationLoss(base, valTokens, CONTEXT)!)}  benchmark ${bench(base)}`);

  if (methods.includes('sft') && SFT_PRESETS[ds.id]) {
    const preset = SFT_PRESETS[ds.id];
    const steps = Number(opt('steps', '300'));
    const lr = Number(opt('lr', '0.0003'));
    console.log(`\n  SFT: ${preset.title}  (${steps} steps, lr ${lr})`);
    const model = clone(base);
    const trainer = setup(model);
    const data = prepareSft(tokenizer, preset.examples, { contextWindow: CONTEXT, eos: true });
    const trunc = data.filter(d => d.truncated).length;
    if (trunc) console.log(`    ${trunc}/${data.length} examples truncated to fit the context`);
    for (let s = 1; s <= steps; s++) {
      const r = sftStep(model, trainer, data, { lr, batchSize: 4, maskPrompt: true });
      if (s % Math.ceil(steps / 5) === 0 || s === 1) console.log(`    step ${String(s).padStart(4)}  response loss ${fmt(r.loss)}  |g| ${fmt(r.gradNorm)}`);
    }
    console.log(`    after: val loss ${fmt(validationLoss(model, valTokens, CONTEXT)!)}  benchmark ${bench(model)}`);
    console.log('    training examples:');
    compare(model, preset.examples.slice(0, 3).map(e => e.prompt));
    console.log('    not in the examples:');
    compare(model, preset.tryPrompts);
  }

  if (methods.includes('dpo') && DPO_PRESETS[ds.id]) {
    const preset = DPO_PRESETS[ds.id];
    const steps = Number(opt('steps', '200'));
    const lr = Number(opt('lr', '0.0001'));
    const beta = Number(opt('beta', '0.1'));
    console.log(`\n  DPO: ${preset.title}  (${steps} steps, lr ${lr}, β ${beta})`);
    const model = clone(base);
    const trainer = setup(model);
    const data = preparePairs(tokenizer, base, preset.pairs, CONTEXT);
    for (let s = 1; s <= steps; s++) {
      dpoStep(model, trainer, data, { beta, lr, batchSize: 4 });
      if (s % Math.ceil(steps / 5) === 0 || s === 1) {
        const m = summarize(data.map(p => scorePair(model, p, beta)));
        console.log(`    step ${String(s).padStart(4)}  loss ${fmt(m.loss)}  margin ${fmt(m.margin)}  acc ${fmt(m.accuracy, 2)}  Δchosen ${fmt(m.chosenShift, 2)}  Δrejected ${fmt(m.rejectedShift, 2)}`);
      }
    }
    console.log(`    after: val loss ${fmt(validationLoss(model, valTokens, CONTEXT)!)}  benchmark ${bench(model)}`);
    console.log('    pair prompts:');
    compare(model, preset.pairs.slice(0, 3).map(p => p.prompt));
    console.log('    not in the pairs:');
    compare(model, preset.tryPrompts);
  }

  for (const preset of methods.includes('rl') ? RL_PRESETS[ds.id] ?? [] : []) {
    if (args.includes('--task') && opt('task', '') !== preset.id) continue;
    const steps = Number(opt('steps', '200'));
    const lr = Number(opt('lr', '0.001'));
    const klBeta = Number(opt('kl', '0.04'));
    console.log(`\n  RL: ${preset.title}  (${steps} steps, lr ${lr}, KL β ${klBeta}, 4 questions × 8 answers)`);
    const model = clone(base);
    const reference = clone(base);
    const trainer = setup(model);
    const acc = () => `greedy acc train ${fmt(greedyAccuracy(model, tokenizer, preset.train), 2)} held-out ${fmt(greedyAccuracy(model, tokenizer, preset.heldOut), 2)}`;
    console.log(`    step    0  ${acc()}`);
    let rewardSum = 0, noSig = 0, kl = 0, n = 0;
    const t0 = performance.now();
    for (let s = 1; s <= steps; s++) {
      const r = rlStep(model, reference, tokenizer, trainer, preset.train, { groupSize: 8, batchSize: 4, temperature: 1, klBeta, lr });
      rewardSum += r.reward; noSig += r.noSignalShare; kl += r.kl; n++;
      if (s % Math.ceil(steps / 5) === 0) {
        console.log(`    step ${String(s).padStart(4)}  mean reward ${fmt(rewardSum / n, 2)}  no-signal groups ${fmt(noSig / n, 2)}  KL ${fmt(kl / n, 4)}  ${acc()}`);
        rewardSum = noSig = kl = n = 0;
      }
    }
    console.log(`    ${fmt((performance.now() - t0) / steps, 1)} ms/step`);
    console.log(`    after: val loss ${fmt(validationLoss(model, valTokens, CONTEXT)!)}  benchmark ${bench(model)}`);
    compare(model, [...preset.train.slice(0, 3), ...preset.heldOut.slice(0, 4)].map(t => t.prompt));
  }
}
