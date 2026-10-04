// Run with: npm run export-gguf [-- dataset-id] [-- --steps 8000] [-- --out file.gguf] [-- --check]
// Trains a sample dataset the way `npm run evals` does (seeded, default model) and writes it as a
// GGUF file for llama.cpp / Ollama, the same file the Train tab's Export button produces.
// --check also writes <out>.check.json: this engine's tokens and next-token probabilities for a
// few prompts, to compare against what llama.cpp computes from the file.
import { writeFileSync } from 'node:fs';
import { MicroTransformer } from '../src/engine/transformer';
import { BPETokenizer } from '../src/engine/bpeTokenizer';
import { SAMPLE_DATASETS, splitDataset, samplePromptFor } from '../src/engine/datasets';
import { exportGguf } from '../src/engine/gguf';

let seed = 12345;
Math.random = () => {
  seed = (seed * 1664525 + 1013904223) % 4294967296;
  return seed / 4294967296;
};

const args = process.argv.slice(2);
const flag = (name: string) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};
const steps = Number(flag('--steps') ?? 8000);
const datasetId = args.find((a, i) => !a.startsWith('--') && !['--steps', '--out'].includes(args[i - 1])) ?? SAMPLE_DATASETS[0].id;
const ds = SAMPLE_DATASETS.find(d => d.id === datasetId);
if (!ds) throw new Error(`Unknown dataset ${datasetId}; try one of ${SAMPLE_DATASETS.map(d => d.id).join(', ')}`);
const out = flag('--out') ?? `llmstead-${ds.id}.gguf`;

// The app's defaults (App.tsx)
const CONTEXT = 16;
const LR = 0.001;
const tokenizer = new BPETokenizer();
tokenizer.train(ds.text, 120);
const model = new MicroTransformer({
  vocabSize: tokenizer.getVocabSize(), contextWindow: CONTEXT, dModel: 32, numHeads: 2, numLayers: 2, mlpRatio: 4, learningRate: LR, optimizer: 'adamw',
});
const split = splitDataset(ds.text);
const train = tokenizer.encode(split.trainText).tokens;
let lossSum = 0;
for (let step = 0; step < steps; step++) {
  const i = Math.floor(Math.random() * Math.max(1, train.length - CONTEXT - 1));
  lossSum += model.trainStep(train.slice(i, i + CONTEXT), train.slice(i + 1, i + CONTEXT + 1), LR).loss;
}
const bytes = exportGguf(model, tokenizer, `LLMStead ${ds.name}`);
writeFileSync(out, bytes);
console.log(`${ds.name}: ${steps} steps, mean loss ${(lossSum / Math.max(1, steps)).toFixed(3)} → ${out} (${(bytes.length / 1024).toFixed(0)} KB)`);

if (args.includes('--check')) {
  // Prompts: the sample prompt plus a few validation-text windows (text the model never trained on)
  const val = split.valText;
  const texts = [samplePromptFor(ds), ...[0, 0.25, 0.5, 0.75].map(f => val.slice(Math.floor(f * val.length)).slice(0, 60))];
  const cases = texts.map(text => {
    const { tokens } = tokenizer.encode(text);
    const window = tokens.slice(0, CONTEXT);
    const fwd = model.inspectForwardPass(window, window.map(String));
    return { text, tokens, window, probs: fwd.probabilities[window.length - 1], allProbs: fwd.probabilities };
  });
  writeFileSync(`${out}.check.json`, JSON.stringify({ cases }));
  console.log(`wrote ${out}.check.json`);
}
