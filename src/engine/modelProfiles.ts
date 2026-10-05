// Model profiles: ready-made combinations of the Setup settings (vocabulary and architecture).
// Picking one is the same as moving those sliders by hand, so it also builds a fresh model.
import { TransformerConfig } from '../types';

export type ProfileShape = Pick<TransformerConfig, 'vocabSize' | 'contextWindow' | 'dModel' | 'numHeads' | 'numLayers' | 'mlpRatio'>;

export interface ModelProfile {
  id: string;
  name: string;
  /** One line on the card */
  tagline: string;
  /** What it's for, shown when it's selected */
  description: string;
  shape: ProfileShape;
}

// Parameter counts and speeds in the comments are for `npm run bench`-style timing in Node; the
// Setup tab measures the real speed on the visitor's device. The descriptions' claims come from
// 4,000 seeded steps per profile (lr 0.001) on Shakespeare / math / Q&A, best validation loss:
// Tiny 3.36 / 1.53 / 2.52, Wide 3.28 / 1.36 / 1.88, Deep 3.40 / 1.36 / 2.32; Small (vocab 200,
// so not comparable) bottomed out at step 3,000 on Shakespeare and was still falling slowly on math.
export const MODEL_PROFILES: ModelProfile[] = [
  {
    // ~33k parameters, ~130 steps/s
    id: 'tiny',
    name: 'Tiny',
    tagline: 'The default. Learns the small datasets in minutes.',
    description:
      'Small enough to train in a couple of minutes and to inspect every number. Right-sized for the built-in datasets, which are a few pages each.',
    shape: { vocabSize: 120, contextWindow: 16, dModel: 32, numHeads: 2, numLayers: 2, mlpRatio: 4 },
  },
  {
    // ~224k parameters, ~17 steps/s
    id: 'small',
    name: 'Small',
    tagline: 'About 7× the parameters, twice the context.',
    description:
      'Every dimension turned up: wider, deeper, a bigger vocabulary, and it reads twice as many tokens at once. Each step is ~8× slower, and on Shakespeare its validation loss turns back up after about 3,000 steps: more model than a few pages of text can feed. (Its loss isn\'t directly comparable with Tiny\'s: bigger tokens carry more each, so per-token loss runs higher.)',
    shape: { vocabSize: 200, contextWindow: 32, dModel: 64, numHeads: 4, numLayers: 4, mlpRatio: 4 },
  },
  {
    // ~137k parameters, ~34 steps/s
    id: 'wide',
    name: 'Wide & shallow',
    tagline: 'One block, wide vectors.',
    description:
      'Nearly the same parameter budget as Deep & narrow, spent on width instead of depth. Train both on the same dataset for the same number of steps and compare validation loss. At this size, width usually wins: in our runs it beat Deep & narrow on Shakespeare and Q&A and tied on math, and each step is faster.',
    shape: { vocabSize: 120, contextWindow: 32, dModel: 96, numHeads: 4, numLayers: 1, mlpRatio: 4 },
  },
  {
    // ~124k parameters, ~28 steps/s
    id: 'deep',
    name: 'Deep & narrow',
    tagline: 'Four blocks, narrow vectors.',
    description:
      'Nearly the same parameter budget as Wide & shallow, spent on depth instead of width. Real models are deep (GPT-2 small has 12 blocks) because stacked blocks can build on each other, but that pays off with much more data and training than here. On these datasets it tends to learn more slowly than Wide & shallow; on math it catches up after a few thousand steps.',
    shape: { vocabSize: 120, contextWindow: 32, dModel: 48, numHeads: 4, numLayers: 4, mlpRatio: 4 },
  },
  {
    // ~545k parameters, ~5 steps/s
    id: 'storyteller',
    name: 'Storyteller',
    tagline: 'For TinyStories: the biggest model here.',
    description:
      'A bigger vocabulary (whole common words become one token) and a 64-token context, enough for a sentence or two. Made for the TinyStories dataset, and slow: after 30 minutes it writes story phrases ("The bird was happy to play with her mom") but loops and invents words. Give it hours, with this tab open, for more.',
    shape: { vocabSize: 500, contextWindow: 64, dModel: 96, numHeads: 4, numLayers: 4, mlpRatio: 4 },
  },
];

/** Just the settings a profile sets */
export const shapeOf = ({ vocabSize, contextWindow, dModel, numHeads, numLayers, mlpRatio }: ProfileShape): ProfileShape => ({
  vocabSize, contextWindow, dModel, numHeads, numLayers, mlpRatio,
});

/** The profile whose settings exactly match these, if any (else the settings are "custom") */
export function matchingProfile(config: ProfileShape): ModelProfile | undefined {
  return MODEL_PROFILES.find(p => (Object.keys(p.shape) as (keyof ProfileShape)[]).every(k => p.shape[k] === config[k]));
}
