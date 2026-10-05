import { COLOURS, COLS, ROWS, SHAPES, SIZES, ShapeLabels } from './shapes';

/**
 * Captions for the shape pictures, written from their labels, and the word-level tokenizer
 * the text tower reads them with.
 *
 * Why words and not BPE (as on the GPT side)? The captions only ever use a couple of dozen
 * words, so every word can simply be its own token. That keeps the views readable: when the
 * Look inside tab shows which token matched which part of the picture, the token is "red",
 * not a fragment like "re" + "d ".
 */

/** How much a caption says about its picture */
export type CaptionDetail = 'shape' | 'position' | 'full';

export const CAPTION_DETAILS: { id: CaptionDetail; label: string; example: string }[] = [
  { id: 'shape', label: 'Colour + shape', example: 'a red circle' },
  { id: 'position', label: '+ where', example: 'a red circle at the top left' },
  { id: 'full', label: '+ size', example: 'a small red circle at the top left' },
];

/** Where the shape is, in words, by [row][col] of the 3×3 grid */
const WHERE: string[][] = [
  ['at the top left', 'at the top', 'at the top right'],
  ['on the left', 'in the middle', 'on the right'],
  ['at the bottom left', 'at the bottom', 'at the bottom right'],
];

/** Where the shape is, in words ("at the top left", "in the middle", …) */
export const whereWords = (labels: ShapeLabels) => WHERE[labels.row][labels.col];

/** Two ways to say the same thing, so the model can't just memorize word positions */
export const NUM_TEMPLATES = 2;

/** The caption of a picture with these labels. Template 0 is the plain form used for zero-shot tests. */
export function captionFor(labels: ShapeLabels, detail: CaptionDetail, template = 0): string {
  const size = detail === 'full' ? `${SIZES[labels.size]} ` : '';
  const thing = `${size}${COLOURS[labels.colour]} ${SHAPES[labels.shape]}`;
  const where = detail === 'shape' ? '' : WHERE[labels.row][labels.col];
  if (template === 0) return where ? `a ${thing} ${where}` : `a ${thing}`;
  return where ? `${where} there is a ${thing}` : `there is a ${thing}`;
}

/**
 * The facts a caption at this detail level states. Two pictures with the same key are described
 * by the same captions, so for the model they're indistinguishable matches.
 */
export function captionKey(labels: ShapeLabels, detail: CaptionDetail): string {
  const parts = [labels.colour, labels.shape];
  if (detail !== 'shape') parts.push(labels.row, labels.col);
  if (detail === 'full') parts.push(labels.size);
  return parts.join(',');
}

/** One label set per distinct caption at this detail level (unstated facts fixed at 0) */
export function allCaptionLabels(detail: CaptionDetail): ShapeLabels[] {
  const out: ShapeLabels[] = [];
  const rows = detail === 'shape' ? [1] : [0, 1, 2];
  const cols = detail === 'shape' ? [1] : [0, 1, 2];
  const sizes = detail === 'full' ? [0, 1] : [0];
  for (let colour = 0; colour < COLOURS.length; colour++)
    for (let shape = 0; shape < SHAPES.length; shape++)
      for (const row of rows) for (const col of cols) for (const size of sizes) out.push({ shape, colour, size, row, col });
  return out;
}

/** A colour + shape pairing that training never shows (neither the picture nor its caption) */
export interface Combo {
  colour: number;
  shape: number;
}

export const DEFAULT_HELD_OUT: Combo[] = [
  { colour: COLOURS.indexOf('green'), shape: SHAPES.indexOf('triangle') },
  { colour: COLOURS.indexOf('yellow'), shape: SHAPES.indexOf('cross') },
];

export const isHeldOut = (labels: ShapeLabels, heldOut: Combo[]) =>
  heldOut.some(c => c.colour === labels.colour && c.shape === labels.shape);

export const comboName = (c: Combo) => `${COLOURS[c.colour]} ${SHAPES[c.shape]}`;

/** Every word any caption can use; a word's index is its token id */
export const VOCAB: readonly string[] = [
  'a', 'there', 'is', 'at', 'on', 'in', 'the',
  ...COLOURS, ...SHAPES, ...SIZES,
  ...new Set([...ROWS, ...COLS]),
];

const WORD_ID = new Map(VOCAB.map((w, i) => [w, i]));

/** Split text into known words' ids. Unknown words are dropped and reported, not guessed at. */
export function tokenizeCaption(text: string): { ids: number[]; words: string[]; unknown: string[] } {
  const ids: number[] = [];
  const words: string[] = [];
  const unknown: string[] = [];
  for (const w of text.toLowerCase().split(/[^a-z]+/).filter(Boolean)) {
    const id = WORD_ID.get(w);
    if (id === undefined) unknown.push(w);
    else {
      ids.push(id);
      words.push(w);
    }
  }
  return { ids, words, unknown };
}

/** Longest caption, in words (sets the text tower's number of position embeddings) */
export const MAX_CAPTION_WORDS = 12;
