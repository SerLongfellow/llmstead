import { MicroTransformer } from './transformer';
import { BPETokenizer } from './bpeTokenizer';

export interface GenerateOptions {
  maxTokens: number;
  /** T ≤ 0.1 → greedy argmax */
  temperature: number;
  /** Stop early once the decoded continuation reaches this many characters */
  stopAfterChars?: number;
  /** Token healing (see generateContinuation). On unless set to false. */
  tokenHealing?: boolean;
  /** Stop when the model emits this token id (post-training teaches it to end replies with <EOS>) */
  stopToken?: number;
}

/** What generateTokens produced, as token ids (what RL trains on) as well as text */
export interface GeneratedSequence {
  /** The prompt's tokens (minus the healed one) followed by every generated token */
  tokens: number[];
  /** Index of the first generated token (the one that re-writes the healed prompt text, if any) */
  responseStart: number;
  /** The newly generated text (the healed prompt text is not repeated) */
  text: string;
  /** Whether generation ended on stopToken */
  stopped: boolean;
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

/**
 * Autoregressively extend `prompt`; returns only the newly generated text.
 *
 * Token healing: a prompt usually ends partway through what the tokenizer would treat as
 * one token. In the training text "1 + 1 = 2" is tokenized as "1" | " + 1" | " = 2", so a
 * prompt ending in "= " leaves the model on a lone " = " token it has rarely seen in that
 * spot. Healing backs up over the prompt's last token and lets the model write that text
 * again itself, in whichever tokens it prefers (" = " may come back as " = 2", or as ")"
 * then ":\n    print(" for code), so the boundary looks like training data.
 */
export function generateContinuation(
  model: MicroTransformer,
  tokenizer: BPETokenizer,
  prompt: string,
  opts: GenerateOptions
): string {
  return generateTokens(model, tokenizer, prompt, opts).text;
}

/** generateContinuation, also returning the exact token ids (see GeneratedSequence) */
export function generateTokens(
  model: MicroTransformer,
  tokenizer: BPETokenizer,
  prompt: string,
  opts: GenerateOptions
): GeneratedSequence {
  const tokens = [...tokenizer.encode(prompt).tokens];
  // The backed-up text, and how much of it the model still has to reproduce
  let healed = '';
  let healedId = -1;
  if (opts.tokenHealing !== false && tokens.length > 1) {
    healedId = tokens.pop()!;
    healed = tokenizer.decode([healedId]);
  }
  let pending = healed;
  let out = '';
  const responseStart = tokens.length;
  let stopped = false;

  for (let step = 0; step < opts.maxTokens; step++) {
    const window = tokens.slice(-model.config.contextWindow);
    if (window.length === 0) break;
    const inspection = model.inspectForwardPass(window, window.map(t => tokenizer.decode([t])));
    let last = inspection.probabilities[inspection.probabilities.length - 1] || [];
    if (pending) {
      // Allowed: tokens that cover the rest of the backed-up text, or a piece of its start
      const rest = pending;
      last = last.map((p, id) => {
        const t = tokenizer.decode([id]);
        return t.length > 0 && (t.startsWith(rest) || rest.startsWith(t)) ? p : 0;
      });
      // Every allowed token underflowed to 0: fall back to the prompt's own last token
      if (pending === healed && last.every(p => p === 0)) last = last.map((_, id) => (id === healedId ? 1 : 0));
    }
    const next = sampleToken(last, opts.temperature);
    tokens.push(next);
    const text = tokenizer.decode([next]);
    out += text;
    if (next === opts.stopToken) {
      stopped = true;
      break;
    }
    pending = pending.startsWith(text) ? pending.slice(text.length) : '';
    if (opts.stopAfterChars !== undefined && out.length - healed.length >= opts.stopAfterChars) break;
  }
  return { tokens, responseStart, text: out.slice(healed.length), stopped };
}
