// Turning prompt/response text into the token sequences post-training learns from, plus the one
// update step every method shares. SFT, DPO and RL differ only in the per-token weights they
// hand to update(); see MicroTransformer.computeGradients.
import { MicroTransformer } from '../transformer';
import { BPETokenizer } from '../bpeTokenizer';
import { Matrix } from '../tensor';
import { Optimizer } from '../optimizer';
import { LoraAdapters } from './lora';

/** A prompt and its response as one token sequence, and where the response starts */
export interface TokenizedExample {
  /** Prompt tokens, then response tokens (then <EOS> if asked for) */
  tokens: number[];
  /**
   * Index of the first token trained as response. The tokenizer has no notion of a prompt
   * boundary, so a token can straddle it (" = 2" holds the end of "1 + 1 = " and the answer);
   * such a token counts as response, since writing it means writing the answer. The prompt's
   * last token counts too: generation backs up over it and writes it again (token healing,
   * see generate.ts), so the model has to learn what follows the prompt from there.
   */
  responseStart: number;
  /** The start of the prompt was cut to fit the context window */
  truncated: boolean;
  /** Characters the tokenizer has never seen; they become <UNK> */
  unknownChars: string[];
}

/**
 * Tokenize `prompt + response` and find the response's first token by character position.
 * Sequences longer than the context window (+1, since the last token is only a target) lose
 * tokens from the start of the prompt.
 */
export function buildExample(
  tokenizer: BPETokenizer,
  prompt: string,
  response: string,
  opts: { contextWindow: number; eos?: boolean }
): TokenizedExample {
  const { tokens, tokenStrings } = tokenizer.encode(prompt + response);
  const unk = tokenizer.specialId('<UNK>');
  const unknownChars = [...new Set(tokenStrings.filter((_, i) => tokens[i] === unk))];

  // The first token reaching the prompt's last character: the healed token, or one straddling the boundary
  let responseStart = tokens.length;
  let end = 0;
  for (let i = 0; i < tokenStrings.length; i++) {
    end += tokenStrings[i].length;
    if (end >= prompt.length) {
      responseStart = Math.max(1, i);
      break;
    }
  }
  const all = opts.eos ? [...tokens, tokenizer.specialId('<EOS>')] : tokens;

  const cut = Math.max(0, all.length - (opts.contextWindow + 1));
  // Keep at least one prompt token, so the first response token has something to follow
  const start = Math.min(cut, Math.max(0, responseStart - 1));
  return {
    tokens: all.slice(start, start + opts.contextWindow + 1),
    responseStart: responseStart - start,
    truncated: start > 0 || all.length - start > opts.contextWindow + 1,
    unknownChars,
  };
}

/**
 * One weight per *position*: position i predicts tokens[i + 1], so it belongs to the response
 * when tokens[i + 1] does. 1 for response positions, 0 for prompt positions.
 */
export function responseMask(ex: { tokens: number[]; responseStart: number }): number[] {
  return ex.tokens.slice(1).map((_, i) => (i + 1 >= ex.responseStart ? 1 : 0));
}

/** Σ log P(token) over the response: how likely the model is to write this exact response */
export function responseLogProb(model: MicroTransformer, ex: { tokens: number[]; responseStart: number }): number {
  const mask = responseMask(ex);
  return model.tokenLogProbs(ex.tokens).reduce((s, lp, i) => s + mask[i] * lp, 0);
}

/** A sequence and the weight of each of its positions in the loss −Σ wᵢ log P(next token) */
export interface WeightedSequence {
  tokens: number[];
  weights: number[];
}

/**
 * Where post-training's updates go: either every weight of the model (full fine-tuning, with
 * its own fresh AdamW) or only the LoRA adapters, with the model's weights frozen.
 */
export interface Trainer {
  optimizer: Optimizer;
  lora: LoraAdapters | null;
}

export function makeTrainer(lora: LoraAdapters | null = null): Trainer {
  return { optimizer: new Optimizer('adamw', { weightDecay: 0 }), lora };
}

/**
 * Sum the gradients of every sequence's weighted loss and take one optimizer step. Returns the
 * gradient norm before clipping and each sequence's weighted loss (before the step).
 */
export function update(
  model: MicroTransformer,
  trainer: Trainer,
  items: WeightedSequence[],
  lr: number
): { gradNorm: number; losses: number[] } {
  let total: Record<string, Matrix> | null = null;
  const losses: number[] = [];
  for (const { tokens, weights } of items) {
    if (tokens.length < 2 || weights.every(w => w === 0)) {
      losses.push(0);
      continue;
    }
    const { loss, grads } = model.computeGradients(tokens.slice(0, -1), tokens.slice(1), weights);
    losses.push(loss);
    if (!total) total = grads;
    else for (const name in grads) addInto(total[name], grads[name]);
  }
  if (!total) return { gradNorm: 0, losses };
  const gradNorm = trainer.lora
    ? trainer.lora.step(model, total, lr, trainer.optimizer)
    : model.applyGradients(total, lr, 1.0, trainer.optimizer);
  return { gradNorm, losses };
}

function addInto(a: Matrix, b: Matrix) {
  for (let r = 0; r < a.length; r++) {
    const ar = a[r];
    const br = b[r];
    for (let c = 0; c < ar.length; c++) ar[c] += br[c];
  }
}

/** Mean of the numbers, 0 for none */
export const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
