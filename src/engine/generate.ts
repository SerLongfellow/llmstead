import { MicroTransformer } from './transformer';
import { BPETokenizer } from './bpeTokenizer';

export interface GenerateOptions {
  maxTokens: number;
  /** T ≤ 0.1 → greedy argmax */
  temperature: number;
  /** Stop early once the decoded continuation reaches this many characters */
  stopAfterChars?: number;
}

/** Pick the next token id from a probability distribution */
export function sampleToken(probs: number[], temperature: number): number {
  if (temperature <= 0.1) {
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    return best;
  }
  // p^(1/T) renormalized == softmax(logits / T)
  const scaled = probs.map(p => Math.pow(Math.max(p, 1e-10), 1.0 / temperature));
  const sum = scaled.reduce((a, b) => a + b, 0);
  const r = Math.random() * sum;
  let acc = 0;
  for (let i = 0; i < scaled.length; i++) {
    acc += scaled[i];
    if (r <= acc) return i;
  }
  return scaled.length - 1;
}

/** Autoregressively extend `prompt`; returns only the newly generated text */
export function generateContinuation(
  model: MicroTransformer,
  tokenizer: BPETokenizer,
  prompt: string,
  opts: GenerateOptions
): string {
  const tokens = [...tokenizer.encode(prompt).tokens];
  let out = '';

  for (let step = 0; step < opts.maxTokens; step++) {
    const window = tokens.slice(-model.config.contextWindow);
    if (window.length === 0) break;
    const inspection = model.inspectForwardPass(window, window.map(t => tokenizer.decode([t])));
    const last = inspection.probabilities[inspection.probabilities.length - 1] || [];
    const next = sampleToken(last, opts.temperature);
    tokens.push(next);
    out += tokenizer.decode([next]);
    if (opts.stopAfterChars !== undefined && out.length >= opts.stopAfterChars) break;
  }
  return out;
}
