import { CaptionDetail, DEFAULT_HELD_OUT } from '../../engine/vl/captions';
import { ClipConfig, FalseNegatives } from '../../engine/vl/clip';
import { ClipData, DEFAULT_CLIP_CONFIG } from '../../engine/vl/clipData';

/** Everything the image + text mode lets you change */
export interface VisionSettings {
  // Set up (changing any of these starts the model over)
  detail: CaptionDetail;
  heldOut: boolean;        // keep two colour + shape combinations out of training
  patchSize: number;       // pixels per patch side (24×24 pictures)
  dModel: number;
  imageLayers: number;
  textLayers: number;
  // Train (safe to change mid-training)
  batchSize: number;
  learningRate: number;
  falseNegatives: FalseNegatives;
}

export const DEFAULT_VISION_SETTINGS: VisionSettings = {
  detail: 'position',
  heldOut: true,
  patchSize: DEFAULT_CLIP_CONFIG.patchSize,
  dModel: DEFAULT_CLIP_CONFIG.dModel,
  imageLayers: DEFAULT_CLIP_CONFIG.imageLayers,
  textLayers: DEFAULT_CLIP_CONFIG.textLayers,
  batchSize: 16,
  learningRate: 0.001,
  falseNegatives: 'ignore',
};

export const PATCH_SIZES = [6, 8, 12];
export const BATCH_SIZES = [4, 8, 16, 32];

export const clipConfigFor = (s: VisionSettings): ClipConfig => ({
  ...DEFAULT_CLIP_CONFIG,
  patchSize: s.patchSize,
  dModel: s.dModel,
  imageLayers: s.imageLayers,
  textLayers: s.textLayers,
  falseNegatives: s.falseNegatives,
});

export const dataFor = (s: VisionSettings): ClipData => ({
  detail: s.detail,
  heldOut: s.heldOut ? DEFAULT_HELD_OUT : [],
  patchSize: s.patchSize,
});
