// Direct Preference Optimization: learn from pairs of responses where one is preferred, without
// a separate reward model. The frozen pre-trained model is the reference the policy is measured
// against, which is what keeps it from drifting off into text that merely games the comparison.
import { MicroTransformer } from '../transformer';
import { BPETokenizer } from '../bpeTokenizer';
import { TokenizedExample, Trainer, buildExample, mean, responseLogProb, responseMask, update } from './examples';

export interface PreferencePair {
  prompt: string;
  chosen: string;
  rejected: string;
}

export interface PreparedPair {
  pair: PreferencePair;
  chosen: TokenizedExample;
  rejected: TokenizedExample;
  /** The reference model's log P(response), computed once since it never changes */
  refChosen: number;
  refRejected: number;
}

export function preparePairs(
  tokenizer: BPETokenizer,
  reference: MicroTransformer,
  pairs: PreferencePair[],
  contextWindow: number
): PreparedPair[] {
  return pairs.map(pair => {
    const chosen = buildExample(tokenizer, pair.prompt, pair.chosen, { contextWindow });
    const rejected = buildExample(tokenizer, pair.prompt, pair.rejected, { contextWindow });
    return { pair, chosen, rejected, refChosen: responseLogProb(reference, chosen), refRejected: responseLogProb(reference, rejected) };
  });
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const softplus = (x: number) => (x > 30 ? x : Math.log1p(Math.exp(x)));

/** Where one pair stands now */
export interface PairScore {
  /** log P(chosen) − reference's: how far training has moved the preferred response */
  chosenShift: number;
  rejectedShift: number;
  /** β · (chosenShift − rejectedShift): DPO's implicit reward gap. Positive = prefers the chosen one. */
  margin: number;
  /** −log σ(margin) */
  loss: number;
}

export function scorePair(model: MicroTransformer, p: PreparedPair, beta: number): PairScore {
  const chosenShift = responseLogProb(model, p.chosen) - p.refChosen;
  const rejectedShift = responseLogProb(model, p.rejected) - p.refRejected;
  const margin = beta * (chosenShift - rejectedShift);
  return { chosenShift, rejectedShift, margin, loss: softplus(-margin) };
}

export interface DpoMetrics {
  loss: number;
  margin: number;
  /** Share of pairs whose margin is positive */
  accuracy: number;
  chosenShift: number;
  rejectedShift: number;
}

export function summarize(scores: PairScore[]): DpoMetrics {
  return {
    loss: mean(scores.map(s => s.loss)),
    margin: mean(scores.map(s => s.margin)),
    accuracy: mean(scores.map(s => (s.margin > 0 ? 1 : 0))),
    chosenShift: mean(scores.map(s => s.chosenShift)),
    rejectedShift: mean(scores.map(s => s.rejectedShift)),
  };
}

/**
 * One DPO step on a random batch of pairs. The loss is −log σ(margin), and its gradient with
 * respect to each response's log-probability is
 *
 *   ∂L/∂log P(chosen) = −β·σ(−margin)      ∂L/∂log P(rejected) = +β·σ(−margin)
 *
 * A log-probability is a sum over the response's tokens, so every response token gets the same
 * weight: +β·σ(−margin) on the chosen response (pushed up), the negative on the rejected one
 * (pushed down). σ(−margin) shrinks as the pair is learned, so settled pairs stop pulling.
 */
export function dpoStep(
  model: MicroTransformer,
  trainer: Trainer,
  data: PreparedPair[],
  opts: { beta: number; lr: number; batchSize: number }
): DpoMetrics & { gradNorm: number } {
  const batch = Array.from({ length: Math.min(opts.batchSize, data.length) }, () => data[Math.floor(Math.random() * data.length)]);
  const scores = batch.map(p => scorePair(model, p, opts.beta));
  const items = batch.flatMap((p, i) => {
    const pull = (opts.beta * sigmoid(-scores[i].margin)) / batch.length;
    return [
      { tokens: p.chosen.tokens, weights: responseMask(p.chosen).map(m => m * pull) },
      { tokens: p.rejected.tokens, weights: responseMask(p.rejected).map(m => -m * pull) },
    ];
  });
  const { gradNorm } = update(model, trainer, items, opts.lr);
  return { ...summarize(scores), gradNorm };
}
