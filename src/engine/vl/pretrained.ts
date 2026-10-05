import { CaptionDetail, Combo } from './captions';
import { ClipConfig, MicroClip } from './clip';
import { DEFAULT_CLIP_CONFIG, ZeroShotScore } from './clipData';

/**
 * The ready-made CLIP that ships with the site (public/models/), so the VLM can have trained eyes
 * the moment it's opened. It is produced by `npm run pretrain-clip` with this same engine code (a
 * fixed seed, the default settings), never edited by hand. Two files:
 *
 *   clip-pretrained.json  the manifest: format, architecture, every matrix's name and shape in
 *                         order, how it was trained, and its zero-shot scores
 *   clip-pretrained.bin   the weights as little-endian Float32, matrices in manifest order, row by row
 *
 * Loading checks everything against the current code before using it (format, architecture, each
 * matrix name and shape, exact byte length). If the engine changes and nobody re-runs the script,
 * the load fails cleanly and the VLM falls back to the visitor's own CLIP.
 */

export const PRETRAINED_FORMAT = 1;
export const PRETRAINED_BASE = 'models/clip-pretrained';

export interface PretrainedManifest {
  format: number;
  config: ClipConfig;
  data: { detail: CaptionDetail; heldOut: Combo[]; patchSize: number };
  training: { seed: number; steps: number; batchSize: number; learningRate: number };
  matrices: { name: string; rows: number; cols: number }[];
  scores: { seen: ZeroShotScore; heldOut: ZeroShotScore | null };
}

/** The weights as one Float32Array, in getParameters() order, plus their names and shapes */
export function packWeights(clip: MicroClip): { matrices: PretrainedManifest['matrices']; data: Float32Array } {
  const params = Object.entries(clip.getParameters());
  const matrices = params.map(([name, M]) => ({ name, rows: M.length, cols: M[0].length }));
  const data = new Float32Array(matrices.reduce((s, m) => s + m.rows * m.cols, 0));
  let k = 0;
  for (const [, M] of params) for (const row of M) for (const v of row) data[k++] = v;
  return { matrices, data };
}

/** Build a MicroClip from a manifest and its weights; throws with a readable reason if anything doesn't fit */
export function unpackWeights(manifest: PretrainedManifest, bytes: ArrayBuffer): MicroClip {
  if (manifest.format !== PRETRAINED_FORMAT) throw new Error(`format ${manifest.format}, this site reads ${PRETRAINED_FORMAT}`);
  const keys = Object.keys(DEFAULT_CLIP_CONFIG) as (keyof ClipConfig)[];
  if (keys.some(k => manifest.config[k] === undefined)) throw new Error('its settings are from a different version of the model');
  const clip = new MicroClip(manifest.config);
  const params = Object.entries(clip.getParameters());
  if (params.length !== manifest.matrices.length) throw new Error('it has a different set of weight matrices');
  params.forEach(([name, M], i) => {
    const m = manifest.matrices[i];
    if (m.name !== name || m.rows !== M.length || m.cols !== M[0].length) throw new Error(`matrix ${i} is ${m.name} ${m.rows}×${m.cols}, expected ${name} ${M.length}×${M[0].length}`);
  });
  const total = manifest.matrices.reduce((s, m) => s + m.rows * m.cols, 0);
  if (bytes.byteLength !== total * 4) throw new Error(`the weights file is ${bytes.byteLength} bytes, expected ${total * 4}`);
  const data = new Float32Array(bytes);
  let k = 0;
  for (const [, M] of params) for (const row of M) for (let c = 0; c < row.length; c++) row[c] = data[k++];
  if (!data.every(Number.isFinite)) throw new Error('the weights contain non-numbers');
  clip.steps = manifest.training.steps;
  return clip;
}

/** Fetch the shipped CLIP. Throws with a readable reason (e.g. a missing file comes back as the site's HTML page). */
export async function fetchPretrainedClip(): Promise<{ clip: MicroClip; manifest: PretrainedManifest }> {
  const base = `/${PRETRAINED_BASE}`; // the site is always served from the root
  const res = await fetch(`${base}.json`);
  if (!res.ok) throw new Error(`couldn't download it (${res.status})`);
  let manifest: PretrainedManifest;
  try {
    manifest = await res.json();
  } catch {
    throw new Error("its description file isn't there");
  }
  const bin = await fetch(`${base}.bin`);
  if (!bin.ok) throw new Error(`couldn't download its weights (${bin.status})`);
  return { clip: unpackWeights(manifest, await bin.arrayBuffer()), manifest };
}
