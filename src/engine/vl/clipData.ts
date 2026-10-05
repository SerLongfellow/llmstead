import {
  CaptionDetail, Combo, DEFAULT_HELD_OUT, MAX_CAPTION_WORDS, NUM_TEMPLATES, VOCAB,
  allCaptionLabels, captionFor, captionKey, isHeldOut, tokenizeCaption
} from './captions';
import { ClipConfig, ClipExample, MicroClip } from './clip';
import { seededRandom } from '../datasets';
import { COLOURS, SHAPES, ShapeImage, ShapeLabels, drawShapeImage, patchify, randomLabels } from './shapes';

/**
 * The training pairs and the tests: how a batch of (picture, caption) pairs is drawn, and
 * zero-shot classification, where the model picks the best caption for a picture out of every
 * caption it could have (no classifier is trained: it's the same similarity score as in training).
 */

export const IMAGE_SIZE = 24;

/** What the pictures and captions look like (the data side of the settings) */
export interface ClipData {
  detail: CaptionDetail;
  heldOut: Combo[];
  patchSize: number;
}

export const DEFAULT_CLIP_DATA: ClipData = { detail: 'position', heldOut: DEFAULT_HELD_OUT, patchSize: 8 };

export const DEFAULT_CLIP_CONFIG: ClipConfig = {
  imageSize: IMAGE_SIZE,
  patchSize: 8,
  dModel: 32,
  numHeads: 4,
  imageLayers: 2,
  textLayers: 1,
  mlpRatio: 2,
  dEmbed: 16,
  vocabSize: VOCAB.length,
  maxWords: MAX_CAPTION_WORDS,
  falseNegatives: 'ignore',
};

/** A picture with its caption, ready for the model */
export interface Pair {
  image: ShapeImage;
  caption: string;
  example: ClipExample;
}

export function makePair(labels: ShapeLabels, rand: () => number, data: ClipData, template = 0): Pair {
  const image = drawShapeImage(labels, rand, IMAGE_SIZE);
  const caption = captionFor(labels, data.detail, template);
  return {
    image,
    caption,
    example: { patches: patchify(image, data.patchSize), tokens: tokenizeCaption(caption).ids, key: captionKey(labels, data.detail) },
  };
}

/** A training batch: random pictures (never a held-out combination), each caption in a random phrasing */
export function sampleBatch(rand: () => number, data: ClipData, n: number): ClipExample[] {
  return Array.from({ length: n }, () => {
    const labels = randomLabels(rand, l => !isHeldOut(l, data.heldOut));
    return makePair(labels, rand, data, Math.floor(rand() * NUM_TEMPLATES)).example;
  });
}

/** Fixed test pictures: either combinations seen in training, or only the held-out ones */
export function testPairs(seed: () => number, data: ClipData, n: number, split: 'seen' | 'held-out'): Pair[] {
  if (split === 'held-out' && data.heldOut.length === 0) return [];
  return Array.from({ length: n }, () => {
    const labels = randomLabels(seed, l => isHeldOut(l, data.heldOut) === (split === 'held-out'));
    return makePair(labels, seed, data);
  });
}

/**
 * Eight pairs for the Train tab's similarity grid, chosen so that neighbours differ by one fact:
 * a red circle at the top left next to a red square there (shape), a blue circle there (colour)
 * and a red circle at the bottom right (where). The grid then shows which facts the model can
 * already tell apart. None of them is a held-out combination.
 */
export function gridPairs(data: ClipData): Pair[] {
  const c = (name: string) => COLOURS.indexOf(name as never);
  const sh = (name: string) => SHAPES.indexOf(name as never);
  const L = (colour: string, shape: string, row: number, col: number): ShapeLabels => ({ colour: c(colour), shape: sh(shape), size: 1, row, col });
  const rand = seededRandom(3001);
  return [
    L('red', 'circle', 0, 0), L('red', 'square', 0, 0), L('blue', 'circle', 0, 0), L('red', 'circle', 2, 2),
    L('yellow', 'triangle', 1, 1), L('yellow', 'square', 1, 1), L('green', 'cross', 1, 2), L('green', 'cross', 1, 0),
  ].map(l => makePair(l, rand, data));
}

/** Zero-shot scores: the fraction of pictures whose best caption gets each fact right */
export interface ZeroShotScore {
  count: number;
  exact: number;   // every stated fact right
  colour: number;
  shape: number;
  where: number | null; // null when captions don't say where
  size: number | null;  // null when captions don't say the size
}

/** Every possible caption at this detail level (plain phrasing), embedded once */
export function captionBank(model: MicroClip, detail: CaptionDetail): { labels: ShapeLabels; text: string; embedding: number[] }[] {
  return allCaptionLabels(detail).map(labels => {
    const text = captionFor(labels, detail);
    return { labels, text, embedding: model.embedText(tokenizeCaption(text).ids)! };
  });
}

const dot = (a: number[], b: number[]) => a.reduce((s, v, k) => s + v * b[k], 0);

export function zeroShot(model: MicroClip, pairs: Pair[], detail: CaptionDetail, bank = captionBank(model, detail)): ZeroShotScore {
  let exact = 0, colour = 0, shape = 0, where = 0, size = 0;
  for (const p of pairs) {
    const u = model.embedImage(p.example.patches);
    let best = bank[0];
    let bestScore = -Infinity;
    for (const c of bank) {
      const s = dot(u, c.embedding);
      if (s > bestScore) {
        bestScore = s;
        best = c;
      }
    }
    const t = p.image.labels;
    const b = best.labels;
    const okColour = b.colour === t.colour;
    const okShape = b.shape === t.shape;
    const okWhere = b.row === t.row && b.col === t.col;
    const okSize = b.size === t.size;
    if (okColour) colour++;
    if (okShape) shape++;
    if (okWhere) where++;
    if (okSize) size++;
    if (okColour && okShape && (detail === 'shape' || okWhere) && (detail !== 'full' || okSize)) exact++;
  }
  const n = Math.max(1, pairs.length);
  return {
    count: pairs.length,
    exact: exact / n,
    colour: colour / n,
    shape: shape / n,
    where: detail === 'shape' ? null : where / n,
    size: detail === 'full' ? size / n : null,
  };
}

/** Chance level for an exact zero-shot match: one over the number of possible captions */
export const chanceExact = (detail: CaptionDetail) => 1 / allCaptionLabels(detail).length;

