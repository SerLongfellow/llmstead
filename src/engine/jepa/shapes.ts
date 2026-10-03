import { Matrix } from '../tensor';

/**
 * Tiny procedurally drawn images for the JEPA: one coloured shape on a dark, slightly noisy
 * background. Generated on the fly from a seeded random source, so there are no files to ship
 * and the "dataset" is effectively infinite. Every image also knows what it shows (shape,
 * colour, position, size), which is what the linear probes test the learned embeddings against.
 */

export const SHAPES = ['circle', 'square', 'triangle', 'cross'] as const;
export const COLOURS = ['red', 'green', 'blue', 'yellow'] as const;

const RGB: Record<(typeof COLOURS)[number], [number, number, number]> = {
  red: [0.9, 0.2, 0.2],
  green: [0.2, 0.8, 0.3],
  blue: [0.25, 0.4, 0.95],
  yellow: [0.95, 0.85, 0.2],
};
const BACKGROUND: [number, number, number] = [0.1, 0.1, 0.12];
const NOISE = 0.03;

export interface ShapeLabels {
  shape: number;  // index into SHAPES
  colour: number; // index into COLOURS
  x: number;      // centre, as a fraction of the image width (0–1)
  y: number;      // centre, as a fraction of the image height (0–1)
  size: number;   // half-extent, as a fraction of the image size
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

/** Range of the shape's half-extent, as fractions of the image size (must stay below 0.47 to fit) */
export interface ShapeSizeRange {
  min: number;
  max: number;
}
export const SMALL_SHAPES: ShapeSizeRange = { min: 0.18, max: 0.3 }; // 6–10 px across at 16×16
export const BIG_SHAPES: ShapeSizeRange = { min: 0.28, max: 0.42 };  // 9–13 px across: more patches show the outline

/** Draw one random image. Each pixel is 2×2 supersampled so small shapes keep their outline. */
export function randomShapeImage(rand: () => number, size = 16, sizes: ShapeSizeRange = SMALL_SHAPES): ShapeImage {
  const shape = Math.floor(rand() * SHAPES.length);
  const colour = Math.floor(rand() * COLOURS.length);
  const r = size * (sizes.min + rand() * (sizes.max - sizes.min));
  const cx = r + 0.5 + rand() * (size - 2 * r - 1);
  const cy = r + 0.5 + rand() * (size - 2 * r - 1);
  const fg = RGB[COLOURS[colour]];

  const pixels = new Array(size * size * 3).fill(0);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let cover = 0;
      for (const sy of [0.25, 0.75]) {
        for (const sx of [0.25, 0.75]) {
          if (inside(shape, x + sx - cx, y + sy - cy, r)) cover += 0.25;
        }
      }
      for (let c = 0; c < 3; c++) {
        const v = cover * fg[c] + (1 - cover) * BACKGROUND[c] + (rand() - 0.5) * 2 * NOISE;
        pixels[(y * size + x) * 3 + c] = Math.min(1, Math.max(0, v));
      }
    }
  }
  return { size, pixels, labels: { shape, colour, x: cx / size, y: cy / size, size: r / size } };
}

/**
 * Cut an image into a grid of square patches, each flattened to one row: the transformer's
 * "tokens". A 16×16 image with 4×4 patches gives 16 rows of 4·4·3 = 48 numbers, ordered
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
