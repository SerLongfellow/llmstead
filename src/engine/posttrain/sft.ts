// Supervised fine-tuning: keep training on next-token prediction, but only on curated
// prompt → response examples, and (usually) only on the response's tokens.
import { MicroTransformer } from '../transformer';
import { BPETokenizer } from '../bpeTokenizer';
import { TokenizedExample, Trainer, buildExample, mean, responseMask, update } from './examples';

export interface SftExample {
  prompt: string;
  response: string;
}

export interface SftOptions {
  lr: number;
  /** Examples per step */
  batchSize: number;
  /**
   * Loss on response tokens only (the standard recipe). Off, the model is also trained to
   * predict the prompts, i.e. to write user questions, which isn't the behaviour we want.
   */
  maskPrompt: boolean;
}

export function prepareSft(
  tokenizer: BPETokenizer,
  examples: SftExample[],
  opts: { contextWindow: number; eos: boolean }
): TokenizedExample[] {
  return examples.map(e => buildExample(tokenizer, e.prompt, e.response, opts));
}

/**
 * Loss weights for one example: the mean over response positions (or over every position
 * when the prompt isn't masked), divided by the batch size so the step averages examples.
 */
export function sftWeights(ex: TokenizedExample, maskPrompt: boolean, batchSize: number): number[] {
  const mask = responseMask(ex);
  if (!maskPrompt) return mask.map(() => 1 / (mask.length * batchSize));
  const n = mask.reduce((a, b) => a + b, 0) || 1;
  return mask.map(m => m / (n * batchSize));
}

export interface SftStepResult {
  /** Mean response-token cross-entropy over the batch, before the update */
  loss: number;
  gradNorm: number;
}

export function sftStep(model: MicroTransformer, trainer: Trainer, data: TokenizedExample[], opts: SftOptions): SftStepResult {
  const batch = Array.from({ length: Math.min(opts.batchSize, data.length) }, () => data[Math.floor(Math.random() * data.length)]);
  // Report response loss either way, so the masked and unmasked runs can be compared
  const loss = mean(batch.map(ex => responseLoss(model, ex)));
  const { gradNorm } = update(
    model,
    trainer,
    batch.map(ex => ({ tokens: ex.tokens, weights: sftWeights(ex, opts.maskPrompt, batch.length) })),
    opts.lr
  );
  return { loss, gradNorm };
}

/** Mean −log P over an example's response tokens */
export function responseLoss(model: MicroTransformer, ex: TokenizedExample): number {
  const mask = responseMask(ex);
  const lps = model.tokenLogProbs(ex.tokens);
  const n = mask.reduce((a, b) => a + b, 0) || 1;
  return -lps.reduce((s, lp, i) => s + mask[i] * lp, 0) / n;
}
