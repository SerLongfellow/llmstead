// Starter data for the Post-train tab, per dataset. Kept short on purpose: the default model
// reads only 16 tokens, and at vocab 120 most words are several tokens long.
import { SftExample } from './sft';
import { PreferencePair } from './dpo';
import { RlTask } from './grpo';
import { seededRandom } from '../datasets';

export interface SftPreset {
  title: string;
  /** What the base model does, and what to look for after */
  goal: string;
  examples: SftExample[];
  /** Prompts not in `examples`, to see whether the behaviour carries over */
  tryPrompts: string[];
}

export interface DpoPreset {
  title: string;
  goal: string;
  pairs: PreferencePair[];
  tryPrompts: string[];
}

// The math benchmark's held-out sums (2 + 1, 4 + 3, 1 + 3) never appear in post-training data
// either, so the benchmark's held-out column still measures generalization afterwards.
const HELD_OUT_SUMS = new Set(['2+1', '4+3', '1+3']);
const sumAllowed = (a: number, b: number) => !HELD_OUT_SUMS.has(`${a}+${b}`);

/** Single-digit sums allowed in post-training data, in a fixed shuffled order */
function shuffledSums(seed: number): [number, number][] {
  const all: [number, number][] = [];
  for (let a = 0; a <= 9; a++) for (let b = 0; b <= 9; b++) if (sumAllowed(a, b)) all.push([a, b]);
  const rand = seededRandom(seed);
  for (let i = all.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [all[i], all[j]] = [all[j], all[i]];
  }
  return all;
}

const sums = shuffledSums(7);

export const SFT_PRESETS: Record<string, SftPreset> = {
  'math-logic': {
    title: 'Answer in a new Q/A format, then stop',
    goal:
      'The base model has never seen "Q: … = ?" questions, and never stops writing. After SFT it should answer ' +
      'in the new format and end with <EOS>, including for sums that are not in the examples.',
    examples: sums.slice(0, 24).map(([a, b]) => ({ prompt: `Q: ${a} + ${b} = ?\nA: `, response: `${a + b}` })),
    tryPrompts: ['Q: 2 + 1 = ?\nA: ', 'Q: 4 + 3 = ?\nA: ', ...sums.slice(24, 28).map(([a, b]) => `Q: ${a} + ${b} = ?\nA: `)],
  },
  'qa-dialogue': {
    title: 'Short replies that end',
    goal:
      'The base model runs on past its answer into the next "User:" turn. After SFT it should give a short reply ' +
      'and stop.',
    examples: [
      { prompt: 'User: Hi!\nAssistant: ', response: 'Hi!' },
      { prompt: 'User: Hello!\nAssistant: ', response: 'Hello!' },
      { prompt: 'User: Hey!\nAssistant: ', response: 'Hey!' },
      { prompt: 'User: Thanks!\nAssistant: ', response: 'Any time!' },
      { prompt: 'User: Thank you!\nAssistant: ', response: 'Any time!' },
      { prompt: 'User: Bye!\nAssistant: ', response: 'Bye!' },
      { prompt: 'User: What is a token?\nAssistant: ', response: 'A piece of text.' },
      { prompt: 'User: What is a loss?\nAssistant: ', response: 'An error score.' },
      { prompt: 'User: What is a weight?\nAssistant: ', response: 'A learned number.' },
    ],
    tryPrompts: ['User: Howdy!\nAssistant: ', 'User: Thanks a lot!\nAssistant: ', 'User: What is a model?\nAssistant: '],
  },
  'code-python': {
    title: 'One-line functions that end',
    goal: 'The base model keeps writing after the function body. After SFT it should write one return line and stop.',
    examples: [
      { prompt: 'def add(a, b):\n    ', response: 'return a + b' },
      { prompt: 'def multiply(a, b):\n    ', response: 'return a * b' },
      { prompt: 'def square(x):\n    ', response: 'return x * x' },
      { prompt: 'def double(x):\n    ', response: 'return x + x' },
      { prompt: 'def first(a, b):\n    ', response: 'return a' },
      { prompt: 'def second(a, b):\n    ', response: 'return b' },
    ],
    tryPrompts: ['def subtract(a, b):\n    ', 'def triple(x):\n    ', 'def last(a, b):\n    '],
  },
  shakespeare: {
    title: 'Short speeches that end',
    goal: 'The base model rolls straight on into the next speaker. After SFT, a speech should end with <EOS>.',
    examples: [
      { prompt: 'ALL:\n', response: 'Speak, speak.' },
      { prompt: 'ALL:\n', response: 'Resolved, resolved.' },
      { prompt: 'ALL:\n', response: "We know't, we know't." },
      { prompt: 'ALL:\n', response: 'Come, come.' },
      { prompt: 'MENENIUS:\n', response: 'Peace, peace.' },
      { prompt: 'MARCIUS:\n', response: 'Go, get you home.' },
    ],
    tryPrompts: ['FIRST CITIZEN:\n', 'SECOND CITIZEN:\n', 'VOLUMNIA:\n'],
  },
};

export const DPO_PRESETS: Record<string, DpoPreset> = {
  'math-logic': {
    title: 'Prefer the right sum',
    goal:
      'Each pair is a correct and an off-by-one answer. The preference accuracy climbs, but a model that cannot add ' +
      'does not learn to from preferences: check the sums outside the pairs.',
    pairs: sums.slice(0, 24).map(([a, b], i) => ({
      prompt: `${a} + ${b} = `,
      chosen: `${a + b}`,
      rejected: `${a + b + (i % 2 === 0 ? 1 : a + b > 0 ? -1 : 1)}`,
    })),
    tryPrompts: ['2 + 1 = ', '4 + 3 = ', '1 + 3 = ', ...sums.slice(24, 27).map(([a, b]) => `${a} + ${b} = `)],
  },
  // The rejected responses below are what the pre-trained models typically write: preference
  // data is most useful when it pushes away from the model's own habits.
  'qa-dialogue': {
    title: 'Prefer a different greeting',
    goal:
      'The base model answers every greeting with "Hi there! How…". Each pair prefers another reply it also ' +
      'saw in training. Check greetings that are not in the pairs.',
    pairs: [
      { prompt: 'User: Hi!\nAssistant: ', chosen: 'Hi! What would you like to know?', rejected: 'Hi there! How can I help you today?' },
      { prompt: 'User: Hello!\nAssistant: ', chosen: 'Hi! What would you like to know?', rejected: 'Hi there! How can I help you today?' },
      { prompt: 'User: Hey!\nAssistant: ', chosen: 'Hi! What would you like to know?', rejected: 'Hi there! How can I help you today?' },
      { prompt: 'User: Good morning!\nAssistant: ', chosen: 'Hi! What would you like to know?', rejected: 'Hi there! How can I help you today?' },
    ],
    tryPrompts: ['User: Howdy!\nAssistant: ', 'User: Good evening!\nAssistant: ', 'User: Greetings!\nAssistant: '],
  },
  'code-python': {
    title: 'Stop writing a + b everywhere',
    goal:
      'The base model writes "return a + b" for almost any function. Each pair prefers the right body over that ' +
      'habit. Watch both log-probabilities: DPO only says which reply is worse, and if it pushes "a + b" down ' +
      'faster than it lifts the right body, the model can land on something that is neither.',
    pairs: [
      { prompt: 'def multiply(a, b):\n    ', chosen: 'return a * b', rejected: 'return a + b' },
      { prompt: 'def divide(a, b):\n    ', chosen: 'return a / b', rejected: 'return a + b' },
      { prompt: 'def first(a, b):\n    ', chosen: 'return a', rejected: 'return a + b' },
      { prompt: 'def second(a, b):\n    ', chosen: 'return b', rejected: 'return a + b' },
      { prompt: 'def square(x):\n    ', chosen: 'return x * x', rejected: 'return x + b' },
    ],
    tryPrompts: ['def subtract(a, b):\n    ', 'def times(a, b):\n    ', 'def add(a, b):\n    '],
  },
  shakespeare: {
    title: 'Prefer speech over stutter',
    goal:
      'A small model falls into loops like "we we we we". Each pair prefers a real line from the play over the ' +
      'loop. Check other speakers: a model this small tends to trade one loop for another.',
    pairs: [
      { prompt: 'FIRST CITIZEN:\n', chosen: 'We are accounted poor', rejected: 'We we we we we we' },
      { prompt: 'ALL:\n', chosen: 'Speak, speak.', rejected: 'What we we we we we' },
      { prompt: 'SECOND CITIZEN:\n', chosen: 'Would you proceed', rejected: 'I we we we we we' },
      { prompt: 'MENENIUS:\n', chosen: 'What work is there?', rejected: 'What we we we we we' },
    ],
    tryPrompts: ['MARCIUS:\n', 'VOLUMNIA:\n', 'COMINIUS:\n'],
  },
};

export interface RlPreset {
  id: string;
  title: string;
  goal: string;
  /** Questions RL samples from */
  train: RlTask[];
  /** Questions it never trains on, to see whether the gain carries over */
  heldOut: RlTask[];
}

const range = (from: number, to: number) => Array.from({ length: to - from + 1 }, (_, i) => from + i);

/**
 * RL presets (math only: RL needs answers a program can check). RL can only reinforce answers
 * the model sometimes gets right, so what matters is how often a sampled answer is right.
 * After 8k pre-training steps the default model gets "a is less than" right about 40% of the
 * time, with right and wrong answers mixed in almost every group: plenty of signal. Sums are
 * right only ~5–10% of the time, and most groups are all wrong.
 */
export const RL_PRESETS: Record<string, RlPreset[]> = {
  'math-logic': [
    {
      id: 'compare',
      title: 'Name a bigger or smaller number',
      goal:
        'Any number that fits is right, so there is no single answer to imitate: a job for RL rather than SFT. ' +
        'The training text only compares 0–9; the held-out questions use two-digit numbers. Watch which answers ' +
        'the model settles on, and whether they still work for the held-out questions.',
      train: range(0, 9).flatMap(a => [
        ...(a < 9 ? [{ prompt: `${a} is less than `, answers: range(a + 1, 99).map(String) }] : []),
        ...(a > 0 ? [{ prompt: `${a} is greater than `, answers: range(0, a - 1).map(String) }] : []),
      ]),
      heldOut: [12, 15, 21, 24, 30, 37, 45, 58].flatMap(a => [
        { prompt: `${a} is less than `, answers: range(a + 1, 99).map(String) },
        { prompt: `${a} is greater than `, answers: range(0, a - 1).map(String) },
      ]),
    },
    {
      id: 'arithmetic',
      title: 'Sums and differences (hard)',
      goal:
        'One right answer per question, and a small model rarely finds it, so most groups get no signal. Expect ' +
        'the reward to creep up while the model learns to bet on common answers rather than to add.',
      ...(() => {
        const tasks: RlTask[] = [];
        for (let a = 0; a <= 9; a++) {
          for (let b = 0; b <= 9; b++) {
            if (sumAllowed(a, b)) tasks.push({ prompt: `${a} + ${b} = `, answers: [`${a + b}`] });
            if (b <= a) tasks.push({ prompt: `${a} - ${b} = `, answers: [`${a - b}`] });
          }
        }
        const rand = seededRandom(3);
        for (let i = tasks.length - 1; i > 0; i--) {
          const j = Math.floor(rand() * (i + 1));
          [tasks[i], tasks[j]] = [tasks[j], tasks[i]];
        }
        return { train: tasks.slice(0, 120), heldOut: tasks.slice(120) };
      })(),
    },
  ],
};
