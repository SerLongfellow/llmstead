// Reinforcement learning with a verifiable reward, in the style of GRPO (Group Relative Policy
// Optimization, used for DeepSeek-R1's reasoning). No labelled answers to imitate: the model
// writes several answers to the same question, a checker scores them, and answers that beat
// their group's average are made more likely.
import { MicroTransformer } from '../transformer';
import { BPETokenizer } from '../bpeTokenizer';
import { answerMatches } from '../benchmarks';
import { generateTokens } from '../generate';
import { Trainer, mean, update } from './examples';

/** A question with checkable answers */
export interface RlTask {
  prompt: string;
  /** Every acceptable answer ("7 is greater than " accepts 0 to 6) */
  answers: string[];
}

/** Does the reply start with one of the task's answers? (whole-word match, see answerMatches) */
export const isCorrect = (task: RlTask, text: string) => task.answers.some(a => answerMatches(text, a));

/** Characters to generate: the benchmark's budget (runBenchmarkSuite) for the longest answer */
const budgetFor = (task: RlTask) => Math.max(...task.answers.map(a => a.length)) + 4;

export interface RlSample {
  text: string;
  tokens: number[];
  responseStart: number;
  /** 1 if the checker accepts the answer, else 0 */
  reward: number;
  /** (reward − group mean) / group spread: how much this answer beat its siblings */
  advantage: number;
}

export interface RlGroup {
  task: RlTask;
  samples: RlSample[];
  /** Every sample got the same reward, so the group carries no signal (all advantages 0) */
  noSignal: boolean;
}

export interface RlOptions {
  /** Answers sampled per question */
  groupSize: number;
  /** Questions per step */
  batchSize: number;
  temperature: number;
  /** Strength of the pull back toward the reference model */
  klBeta: number;
  lr: number;
}

export interface RlStepResult {
  groups: RlGroup[];
  /** Mean reward of this step's samples (before the update) */
  reward: number;
  /** Share of groups that carried no signal */
  noSignalShare: number;
  /** Mean per-token KL(policy ‖ reference) estimate on the sampled answers */
  kl: number;
  gradNorm: number;
}

/** Sample groupSize answers for one task and score them */
export function sampleGroup(
  model: MicroTransformer,
  tokenizer: BPETokenizer,
  task: RlTask,
  groupSize: number,
  temperature: number
): RlGroup {
  const budget = budgetFor(task);
  const raw = Array.from({ length: groupSize }, () => {
    const g = generateTokens(model, tokenizer, task.prompt, { maxTokens: budget, temperature, stopAfterChars: budget });
    // Keep the last contextWindow + 1 tokens: the most a training pass can see, and all the model saw while writing
    const cut = Math.max(0, Math.min(g.tokens.length - (model.config.contextWindow + 1), g.responseStart - 1));
    return {
      text: g.text,
      tokens: g.tokens.slice(cut),
      responseStart: g.responseStart - cut,
      reward: isCorrect(task, g.text) ? 1 : 0,
    };
  });
  const mu = mean(raw.map(s => s.reward));
  const sd = Math.sqrt(mean(raw.map(s => (s.reward - mu) ** 2)));
  const noSignal = sd < 1e-9;
  return { task, noSignal, samples: raw.map(s => ({ ...s, advantage: noSignal ? 0 : (s.reward - mu) / sd })) };
}

/**
 * One step: sample a group per task, then raise the log-probability of each sampled answer in
 * proportion to its advantage. For a sampled token with probability π (reference: π_ref), the
 * loss per answer token is
 *
 *   −A · log π  +  β · k3,      k3 = π_ref/π − log(π_ref/π) − 1   (≥ 0; an estimate of KL)
 *
 * whose gradient with respect to log π is −A + β·(1 − π_ref/π), so each token's loss weight is
 * A − β·(1 − π_ref/π), averaged over the answer's tokens, the group and the batch.
 *
 * The update uses the samples it just drew, once, so the probability-ratio clipping of PPO/GRPO
 * (there to make reusing old samples safe) would never kick in and is left out.
 */
export function rlStep(
  model: MicroTransformer,
  reference: MicroTransformer,
  tokenizer: BPETokenizer,
  trainer: Trainer,
  tasks: RlTask[],
  opts: RlOptions
): RlStepResult {
  const batch = Array.from({ length: Math.min(opts.batchSize, tasks.length) }, () => tasks[Math.floor(Math.random() * tasks.length)]);
  const groups = batch.map(task => sampleGroup(model, tokenizer, task, opts.groupSize, opts.temperature));

  const kls: number[] = [];
  const items = groups.flatMap(g =>
    g.samples.map(s => {
      const lp = model.tokenLogProbs(s.tokens);
      const ref = reference.tokenLogProbs(s.tokens);
      const n = s.tokens.length - s.responseStart; // answer tokens
      const scale = 1 / (Math.max(1, n) * g.samples.length * groups.length);
      const weights = lp.map((l, i) => {
        if (i + 1 < s.responseStart) return 0;
        const ratio = Math.exp(ref[i] - l);
        kls.push(ratio - (ref[i] - l) - 1);
        return (s.advantage - opts.klBeta * (1 - ratio)) * scale;
      });
      return { tokens: s.tokens, weights };
    })
  );
  const { gradNorm } = update(model, trainer, items, opts.lr);
  return {
    groups,
    reward: mean(groups.flatMap(g => g.samples.map(s => s.reward))),
    noSignalShare: mean(groups.map(g => (g.noSignal ? 1 : 0))),
    kl: mean(kls),
    gradNorm,
  };
}

/** Greedy accuracy on a task list (temperature 0, like the benchmark) */
export function greedyAccuracy(model: MicroTransformer, tokenizer: BPETokenizer, tasks: RlTask[]): number {
  return mean(
    tasks.map(t => {
      const budget = budgetFor(t);
      const g = generateTokens(model, tokenizer, t.prompt, { maxTokens: budget, temperature: 0, stopAfterChars: budget });
      return isCorrect(t, g.text) ? 1 : 0;
    })
  );
}
