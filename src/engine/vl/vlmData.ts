import { Matrix } from '../tensor';
import { Combo, VOCAB, captionFor, isHeldOut, whereWords } from './captions';
import { COLOURS, SHAPES, SIZES, ShapeImage, ShapeLabels, drawShapeImage, patchify, randomLabels } from './shapes';
import { IMAGE_SIZE } from './clipData';

/**
 * Questions and answers about the shape pictures, for the vision-language model (VLM).
 *
 * Each training example is a picture plus a short exchange, laid out the way the language model
 * reads it: the picture's patches first (as extra "tokens"), then the question, then the answer:
 *
 *   [▣1 … ▣9] what colour is the shape ? red <end>
 *
 * The model is trained to predict every next word, but only the answer's words (and <end>)
 * count toward the loss: it should learn to answer, not to ask.
 */

export type VqaTask = 'describe' | 'colour' | 'shape' | 'where' | 'size' | 'there';

export const VQA_TASKS: { id: VqaTask; label: string; example: string }[] = [
  { id: 'describe', label: 'Describe', example: 'describe the picture : a red circle at the top left' },
  { id: 'colour', label: 'Colour', example: 'what colour is the shape ? red' },
  { id: 'shape', label: 'Shape', example: 'what shape is it ? circle' },
  { id: 'where', label: 'Where', example: 'where is the shape ? at the top left' },
  { id: 'size', label: 'Size', example: 'is the shape big ? no' },
  { id: 'there', label: 'Is there…', example: 'is there a blue square ? no' },
];

export const END = '<end>';

/** The caption words plus the question words; a word's index is its token id */
export const VLM_VOCAB: readonly string[] = [
  ...new Set([...VOCAB, 'what', 'colour', 'shape', 'where', 'it', 'describe', 'picture', 'yes', 'no', '?', ':', END]),
];
const WORD_ID = new Map(VLM_VOCAB.map((w, i) => [w, i]));
export const END_ID = WORD_ID.get(END)!;

/** Split text into known words' ids ("?" and ":" are words of their own). Unknown words are dropped and reported. */
export function tokenizeVlm(text: string): { ids: number[]; words: string[]; unknown: string[] } {
  const ids: number[] = [];
  const words: string[] = [];
  const unknown: string[] = [];
  const parts = text.toLowerCase().replace(/([?:])/g, ' $1 ').split(/\s+/).filter(Boolean);
  for (const raw of parts) {
    const w = raw === END ? raw : raw.replace(/[^a-z?:]/g, '');
    if (!w) continue;
    const id = WORD_ID.get(w);
    if (id === undefined) unknown.push(raw);
    else {
      ids.push(id);
      words.push(w);
    }
  }
  return { ids, words, unknown };
}

/** One question about one picture, and its right answer */
export function questionFor(task: VqaTask, labels: ShapeLabels, rand: () => number, heldOut: Combo[]): { question: string; answer: string } {
  switch (task) {
    case 'describe':
      return { question: 'describe the picture :', answer: captionFor(labels, 'position') };
    case 'colour':
      return { question: 'what colour is the shape ?', answer: COLOURS[labels.colour] };
    case 'shape':
      return { question: 'what shape is it ?', answer: SHAPES[labels.shape] };
    case 'where':
      return { question: 'where is the shape ?', answer: whereWords(labels) };
    case 'size':
      return { question: 'is the shape big ?', answer: SIZES[labels.size] === 'big' ? 'yes' : 'no' };
    case 'there': {
      // Half the time ask about what's there; otherwise change one fact (colour or shape),
      // never asking about a held-out combination
      if (rand() < 0.5) return { question: `is there a ${COLOURS[labels.colour]} ${SHAPES[labels.shape]} ?`, answer: 'yes' };
      for (;;) {
        const asked = { ...labels };
        if (rand() < 0.5) asked.colour = (labels.colour + 1 + Math.floor(rand() * (COLOURS.length - 1))) % COLOURS.length;
        else asked.shape = (labels.shape + 1 + Math.floor(rand() * (SHAPES.length - 1))) % SHAPES.length;
        if (!isHeldOut(asked, heldOut)) return { question: `is there a ${COLOURS[asked.colour]} ${SHAPES[asked.shape]} ?`, answer: 'no' };
      }
    }
  }
}

/** One training or test example, ready for the model */
export interface VlmExample {
  image: ShapeImage;
  patches: Matrix;
  task: VqaTask;
  question: string;
  answer: string;
  questionIds: number[];
  answerIds: number[];  // the answer's words, then <end>
  input: number[];      // question + answer, minus the last word
  target: number[];     // the same shifted by one: the word each position should predict
  lossMask: boolean[];  // true where the target is part of the answer
}

export function makeExample(image: ShapeImage, patchSize: number, task: VqaTask, rand: () => number, heldOut: Combo[]): VlmExample {
  const { question, answer } = questionFor(task, image.labels, rand, heldOut);
  const questionIds = tokenizeVlm(question).ids;
  const answerIds = [...tokenizeVlm(answer).ids, END_ID];
  const all = [...questionIds, ...answerIds];
  const input = all.slice(0, -1);
  const target = all.slice(1);
  // target[i] is word i + 1, which belongs to the answer once i + 1 ≥ the question's length
  const lossMask = target.map((_, i) => i + 1 >= questionIds.length);
  return { image, patches: patchify(image, patchSize), task, question, answer, questionIds, answerIds, input, target, lossMask };
}

/** A training batch: random pictures (never a held-out combination), each with a random question from `tasks` */
export function sampleVlmBatch(rand: () => number, patchSize: number, tasks: VqaTask[], heldOut: Combo[], n: number): VlmExample[] {
  return Array.from({ length: n }, () => {
    const labels = randomLabels(rand, l => !isHeldOut(l, heldOut));
    const task = tasks[Math.floor(rand() * tasks.length)];
    return makeExample(drawShapeImage(labels, rand, IMAGE_SIZE), patchSize, task, rand, heldOut);
  });
}

/** Longest question + answer + <end>, in words (sets the language model's context with the patches) */
export const MAX_QA_WORDS = 18;
