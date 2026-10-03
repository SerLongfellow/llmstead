import { DEFAULT_JEPA_CONFIG, JepaConfig, JepaRecipe } from '../../engine/jepa/jepa';
import { BIG_SHAPES, COLOURS, SHAPES, SMALL_SHAPES, ShapeImage, ShapeSizeRange } from '../../engine/jepa/shapes';

/** "red circle" */
export const describe = (img: ShapeImage) => `${COLOURS[img.labels.colour]} ${SHAPES[img.labels.shape]}`;

/** Everything the JEPA mode's Setup and Train tabs let you change */
export interface JepaSettings {
  shapes: 'small' | 'big';
  // Architecture: changing any of these builds a fresh model
  dModel: number;
  numHeads: number;
  numLayers: number;
  predDim: number;
  predLayers: number;
  // Safe to change while training: the weights are kept
  numTargets: number;
  batchSize: number;
  learningRate: number;
  recipe: JepaRecipe;
}

/** The anti-collapse term's weights when it's switched on (what worked best in the experiments) */
export const ANTI_COLLAPSE = { varWeight: 1, covWeight: 0.04 };

export const DEFAULT_JEPA_SETTINGS: JepaSettings = {
  shapes: 'small',
  dModel: 32,
  numHeads: 4,
  numLayers: 2,
  predDim: 16,
  predLayers: 2,
  numTargets: 2,
  batchSize: 8,
  learningRate: 0.0003,
  recipe: { ablation: 'none', varWeight: 0, covWeight: 0 },
};

export const IMAGE_SIZE = 16;
export const PATCH_SIZE = 4;
export const GRID = IMAGE_SIZE / PATCH_SIZE;

export const shapeSizes = (s: JepaSettings): ShapeSizeRange => (s.shapes === 'big' ? BIG_SHAPES : SMALL_SHAPES);

/**
 * The model config for these settings. I-JEPA ramps the EMA momentum to exactly 1 at the end of a
 * fixed-length run, which would freeze the target encoder for good; training here has no set end,
 * so it levels off at 0.999 instead.
 */
export function jepaConfigFor(s: JepaSettings): JepaConfig {
  return {
    ...DEFAULT_JEPA_CONFIG,
    imageSize: IMAGE_SIZE,
    patchSize: PATCH_SIZE,
    dModel: s.dModel,
    numHeads: s.numHeads,
    numLayers: s.numLayers,
    predDim: s.predDim,
    predHeads: 2,
    predLayers: s.predLayers,
    numTargets: s.numTargets,
    emaEnd: 0.999,
    emaSteps: 5000,
    ...s.recipe,
  };
}

/** Plain-language names for the three recipes */
export const RECIPES: { id: JepaRecipe['ablation']; label: string; short: string; description: string }[] = [
  {
    id: 'none',
    label: 'I-JEPA',
    short: 'EMA target + stop-gradient',
    description:
      'The real recipe. Targets come from a slowly updated copy of the encoder (EMA) and the loss never trains the target side (stop-gradient).',
  },
  {
    id: 'no-ema',
    label: 'No EMA',
    short: 'live target, stop-gradient',
    description:
      'Targets come from the encoder as it is right now instead of a slow copy. Stop-gradient still protects the target side.',
  },
  {
    id: 'no-stopgrad',
    label: 'No stop-gradient',
    short: 'the loss trains both sides',
    description:
      'The loss can move the targets too, so the easiest way to make it small is to give every image the same embedding. Watch it collapse.',
  },
];
