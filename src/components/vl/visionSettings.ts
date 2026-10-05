import { CaptionDetail, DEFAULT_HELD_OUT } from '../../engine/vl/captions';
import { ClipConfig, FalseNegatives } from '../../engine/vl/clip';
import { ClipData, DEFAULT_CLIP_CONFIG } from '../../engine/vl/clipData';
import { DEFAULT_VLM_CONFIG, VlmConfig, VlmPhase } from '../../engine/vl/vlm';
import { VQA_TASKS, VqaTask } from '../../engine/vl/vlmData';

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

// ── Stage 2: the vision-language model ──

/** What the VLM stage lets you change */
export interface VlmSettings {
  // Set up (changing the encoder or the language model's size starts the VLM over)
  encoder: 'clip' | 'random';   // a frozen copy of your CLIP's image tower, or an untrained one
  dModel: number;
  numLayers: number;
  tasks: VqaTask[];              // question kinds trained on in phases 0 and 2
  // Train (safe to change between and during runs)
  phase: VlmPhase;
  batchSize: number;
  learningRate: number;
}

/** Each phase's starting learning rate: the projector alone (phase 1) learns faster with a bigger step */
export const PHASE_LR: Record<VlmPhase, number> = { 0: 0.001, 1: 0.003, 2: 0.001 };

export const DEFAULT_VLM_SETTINGS: VlmSettings = {
  encoder: 'clip',
  dModel: DEFAULT_VLM_CONFIG.dModel,
  numLayers: DEFAULT_VLM_CONFIG.numLayers,
  tasks: VQA_TASKS.map(t => t.id),
  phase: 0,
  batchSize: 8,
  learningRate: PHASE_LR[0],
};

export const vlmConfigFor = (s: VlmSettings): VlmConfig => ({ ...DEFAULT_VLM_CONFIG, dModel: s.dModel, numLayers: s.numLayers });
