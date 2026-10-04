import { Matrix } from '../tensor';

/**
 * Tiny procedurally drawn pictures for the vision-language model: one coloured shape on a dark,
 * slightly noisy background. They're generated on the fly from a seeded random source, so there
 * are no image files to ship and the supply is endless. Every picture knows what it shows
 * (shape, colour, where, how big), which is what its caption is written from (captions.ts).
 *
 * The shape sits in one of 9 cells of a 3×3 grid (with a little jitter), so "top left" or
 * "in the middle" is always a fair description and every position is equally common.
 */

export const SHAPES = ['circle', 'square', 'triangle', 'cross'] as const;
export const COLOURS = ['red', 'green', 'blue', 'yellow'] as const;
export const SIZES = ['small', 'big'] as const;
export const ROWS = ['top', 'middle', 'bottom'] as const;
export const COLS = ['left', 'middle', 'right'] as const;

export type ShapeName = (typeof SHAPES)[number];
export type ColourName = (typeof COLOURS)[number];

const RGB: Record<ColourName, [number, number, number]> = {
  red: [0.9, 0.2, 0.2],
  green: [0.2, 0.8, 0.3],
  blue: [0.25, 0.4, 0.95],
  yellow: [0.95, 0.85, 0.2],
};
const BACKGROUND: [number, number, number] = [0.1, 0.1, 0.12];
const NOISE = 0.03;

/** Half-extent of the shape in pixels (at 24×24), by size */
const RADIUS: Record<(typeof SIZES)[number], [number, number]> = {
  small: [2.4, 3.0],
  big: [3.6, 4.2],
};
/** How far (in pixels) the centre may wander from its cell's centre */
const JITTER = 1;

export interface ShapeLabels {
  shape: number;  // index into SHAPES
  colour: number; // index into COLOURS
  size: number;   // index into SIZES
  row: number;    // index into ROWS
  col: number;    // index into COLS
}

export interface ShapeImage {
  size: number;     // width = height, in pixels
  pixels: number[]; // RGB in 0–1, row by row: pixels[(y * size + x) * 3 + channel]
  labels: ShapeLabels;
}

/** Is the point (dx, dy) from the shape's centre inside a shape of half-extent r? (dy points down) */
function inside(shape: number, dx: number, dy: number, r: number): boolean {
  switch (SHAPES[shape]) {
    case 'circle':
      return dx * dx + dy * dy <= r * r;
    case 'square':
      return Math.abs(dx) <= 0.85 * r && Math.abs(dy) <= 0.85 * r;
    case 'triangle': // pointing up: zero width at the top, full width at the bottom
      return dy >= -r && dy <= r && Math.abs(dx) <= (dy + r) / 2;
    case 'cross': {
      const arm = 0.35 * r;
      return (Math.abs(dx) <= arm && Math.abs(dy) <= r) || (Math.abs(dy) <= arm && Math.abs(dx) <= r);
    }
  }
  return false;
}

/** Draw a picture of exactly these labels. Each pixel is 2×2 supersampled so small shapes keep their outline. */
export function drawShapeImage(labels: ShapeLabels, rand: () => number, size = 24): ShapeImage {
  const scale = size / 24;
  const [rMin, rMax] = RADIUS[SIZES[labels.size]];
  const r = scale * (rMin + rand() * (rMax - rMin));
  const cell = size / 3;
  const cx = (labels.col + 0.5) * cell + (rand() * 2 - 1) * JITTER * scale;
  const cy = (labels.row + 0.5) * cell + (rand() * 2 - 1) * JITTER * scale;
  const fg = RGB[COLOURS[labels.colour]];

  const pixels = new Array(size * size * 3).fill(0);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let cover = 0;
      for (const sy of [0.25, 0.75]) {
        for (const sx of [0.25, 0.75]) {
          if (inside(labels.shape, x + sx - cx, y + sy - cy, r)) cover += 0.25;
        }
      }
      for (let c = 0; c < 3; c++) {
        const v = cover * fg[c] + (1 - cover) * BACKGROUND[c] + (rand() - 0.5) * 2 * NOISE;
        pixels[(y * size + x) * 3 + c] = Math.min(1, Math.max(0, v));
      }
    }
  }
  return { size, pixels, labels };
}

/** Random labels, all equally likely (`allowed` can reject some, e.g. held-out combinations) */
export function randomLabels(rand: () => number, allowed: (l: ShapeLabels) => boolean = () => true): ShapeLabels {
  for (;;) {
    const labels: ShapeLabels = {
      shape: Math.floor(rand() * SHAPES.length),
      colour: Math.floor(rand() * COLOURS.length),
      size: Math.floor(rand() * SIZES.length),
      row: Math.floor(rand() * ROWS.length),
      col: Math.floor(rand() * COLS.length),
    };
    if (allowed(labels)) return labels;
  }
}

/**
 * Cut an image into a grid of square patches, each flattened to one row: the image tower's
 * "tokens". A 24×24 image with 6×6 patches gives 16 rows of 6·6·3 = 108 numbers, ordered
 * left-to-right, top-to-bottom. Values are rescaled from 0–1 to −1…1 so they're centred on 0.
 */
export function patchify(img: ShapeImage, patchSize: number): Matrix {
  const grid = img.size / patchSize;
  const rows: Matrix = [];
  for (let py = 0; py < grid; py++) {
    for (let px = 0; px < grid; px++) {
      const row: number[] = [];
      for (let y = 0; y < patchSize; y++) {
        for (let x = 0; x < patchSize; x++) {
          const base = ((py * patchSize + y) * img.size + (px * patchSize + x)) * 3;
          for (let c = 0; c < 3; c++) row.push(img.pixels[base + c] * 2 - 1);
        }
      }
      rows.push(row);
    }
  }
  return rows;
}
