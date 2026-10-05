// Run with: npm run stories [-- --minutes 30 --profile storyteller --lr 0.001 --every 5]
// Trains a model profile on the TinyStories sample (public/datasets/tinystories.txt) the way the
// Train tab does, for a fixed wall-clock time, and every few minutes prints the training and
// validation loss plus two continuations of "Once upon a time" (greedy and T 0.7). Use it to
// check what the Setup tab's TinyStories advice promises. Math.random is seeded, so runs repeat.
import * as fs from 'fs';
import { MicroTransformer } from '../src/engine/transformer';
import { BPETokenizer } from '../src/engine/bpeTokenizer';
import { splitDataset } from '../src/engine/datasets';
import { trainChunk, validationLoss } from '../src/engine/training';
import { generateContinuation } from '../src/engine/generate';
import { MODEL_PROFILES } from '../src/engine/modelProfiles';

let seed = 12345;
Math.random = () => {
  seed = (seed * 1664525 + 1013904223) % 4294967296;
  return seed / 4294967296;
};

const args = process.argv.slice(2);
const arg = (name: string, fallback: string) => (args.includes(`--${name}`) ? args[args.indexOf(`--${name}`) + 1] : fallback);
const minutes = Number(arg('minutes', '30'));
const every = Number(arg('every', '5'));
const lr = Number(arg('lr', '0.001'));
const profile = MODEL_PROFILES.find(p => p.id === arg('profile', 'storyteller'));
if (!profile) throw new Error(`Unknown profile; try one of ${MODEL_PROFILES.map(p => p.id).join(', ')}`);

const text = fs.readFileSync('public/datasets/tinystories.txt', 'utf8');
const tokenizer = new BPETokenizer();
tokenizer.train(text, profile.shape.vocabSize);
const { trainText, valText } = splitDataset(text);
const train = tokenizer.encode(trainText).tokens;
const val = tokenizer.encode(valText).tokens;
const model = new MicroTransformer({ ...profile.shape, vocabSize: tokenizer.getVocabSize(), learningRate: lr, optimizer: 'adamw' });
const ctx = profile.shape.contextWindow;
console.log(`${profile.name}: ${model.getParameterCount().toLocaleString()} parameters, ${train.length.toLocaleString()} training tokens, ${minutes} min`);

const t0 = Date.now();
let step = 0;
let lossSum = 0;
let lossSteps = 0;
let nextReport = every * 60_000;
const report = () => {
  const sample = (temperature: number) => JSON.stringify(generateContinuation(model, tokenizer, 'Once upon a time', { maxTokens: 50, temperature }));
  console.log(`\n${((Date.now() - t0) / 60_000).toFixed(0).padStart(3)} min  step ${step}  train ${(lossSum / Math.max(1, lossSteps)).toFixed(3)}  val ${validationLoss(model, val, ctx)?.toFixed(3)}  (${((step * ctx) / train.length).toFixed(2)} passes over the data)`);
  console.log(`  greedy: ${sample(0.1)}`);
  console.log(`  T 0.7:  ${sample(0.7)}`);
  lossSum = 0;
  lossSteps = 0;
};
while (Date.now() - t0 < minutes * 60_000) {
  const r = trainChunk(model, train, val, { contextWindow: ctx, learningRate: lr, startStep: step, budgetMs: 1000, maxSteps: Infinity });
  step += r.steps;
  lossSum += r.lossSum;
  lossSteps += r.steps;
  if (Date.now() - t0 >= nextReport) {
    report();
    nextReport += every * 60_000;
  }
}
report();
