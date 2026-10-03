// Run with: npm run evals [-- dataset-id ...] [-- --steps 2000,8000]
// Trains each sample dataset the way the Train tab does (default model, AdamW, random windows
// from the training split) and runs its benchmark at a few checkpoints. Math.random is seeded,
// so runs are repeatable and a dataset change can be judged by its effect on the benchmark.
import { MicroTransformer } from '../src/engine/transformer';
import { BPETokenizer } from '../src/engine/bpeTokenizer';
import { SAMPLE_DATASETS, splitDataset } from '../src/engine/datasets';
import { BENCHMARK_SUITES, runBenchmarkSuite } from '../src/engine/benchmarks';

let seed = 12345;
Math.random = () => {
  seed = (seed * 1664525 + 1013904223) % 4294967296;
  return seed / 4294967296;
};

const args = process.argv.slice(2);
const stepsArg = args.indexOf('--steps');
const checkpoints = stepsArg >= 0 ? args[stepsArg + 1].split(',').map(Number) : [2000, 8000];
const only = args.filter((a, i) => !a.startsWith('--') && args[i - 1] !== '--steps');

// The app's defaults (App.tsx)
const VOCAB = 120;
const CONTEXT = 16;
const LR = 0.001;

for (const ds of SAMPLE_DATASETS) {
  if (only.length && !only.includes(ds.id)) continue;
  const suite = BENCHMARK_SUITES[ds.id];
  if (!suite) continue;
  seed = 12345;

  const tokenizer = new BPETokenizer();
  tokenizer.train(ds.text, VOCAB);
  const vocabSize = Math.max(...tokenizer.encode(ds.text).tokens) + 1;
  const model = new MicroTransformer({
    vocabSize, contextWindow: CONTEXT, dModel: 32, numHeads: 2, numLayers: 2, mlpRatio: 4, learningRate: LR, optimizer: 'adamw',
  });
  const split = splitDataset(ds.text);
  const train = tokenizer.encode(split.trainText).tokens;
  console.log(`\n${ds.name}  (${train.length} training tokens, ${ds.text.length} chars)`);

  let step = 0;
  for (const target of checkpoints) {
    let lossSum = 0;
    const start = step;
    for (; step < target; step++) {
      const i = Math.floor(Math.random() * Math.max(1, train.length - CONTEXT - 1));
      lossSum += model.trainStep(train.slice(i, i + CONTEXT), train.slice(i + 1, i + CONTEXT + 1), LR).loss;
    }
    const r = runBenchmarkSuite(suite, model, tokenizer, split.trainText);
    const { seen, 'held-out': held } = r.bySplit;
    console.log(`  step ${String(target).padStart(6)}  loss ${(lossSum / (step - start)).toFixed(3)}  seen ${seen.passed}/${seen.total}  held-out ${held.passed}/${held.total}`);
    for (const x of r.results) {
      const tag = x.split === 'seen' ? 'seen' : 'held';
      console.log(`      ${x.passed ? 'PASS' : '    '} ${tag}  ${JSON.stringify(x.testCase.prompt).padEnd(52)} → ${JSON.stringify(x.actualOutput)}`);
    }
  }
}
