// Published architectures of real language models, for scale comparisons in the UI.
// Sources: GPT-2 paper (Radford et al. 2019), GPT-3 paper (Brown et al. 2020),
// Llama 2 paper (Touvron et al. 2023), Llama 3 / 3.1 model card and paper (Meta 2024).
// Labs behind current commercial models (GPT-4 and later, Claude, Gemini) don't publish
// parameter counts or architectures, so those appear only as a context-window range.

export interface ReferenceModel {
  name: string;
  year: number;
  params: number;          // total parameters
  dModel: number;
  layers: number;
  heads: number;
  contextWindow: number;   // tokens
  vocabSize: number;
  trainingTokens?: number; // tokens seen during pre-training, when published
}

export const REFERENCE_MODELS: ReferenceModel[] = [
  { name: 'GPT-2 small', year: 2019, params: 124e6, dModel: 768, layers: 12, heads: 12, contextWindow: 1024, vocabSize: 50257 },
  { name: 'GPT-3', year: 2020, params: 175e9, dModel: 12288, layers: 96, heads: 96, contextWindow: 2048, vocabSize: 50257, trainingTokens: 300e9 },
  { name: 'Llama 2 7B', year: 2023, params: 7e9, dModel: 4096, layers: 32, heads: 32, contextWindow: 4096, vocabSize: 32000, trainingTokens: 2e12 },
  { name: 'Llama 3 8B', year: 2024, params: 8e9, dModel: 4096, layers: 32, heads: 32, contextWindow: 8192, vocabSize: 128256, trainingTokens: 15e12 },
  { name: 'Llama 3.1 405B', year: 2024, params: 405e9, dModel: 16384, layers: 126, heads: 128, contextWindow: 131072, vocabSize: 128256, trainingTokens: 15.6e12 },
];

/** Undisclosed frontier models: only the publicly advertised context range is stated */
export const FRONTIER_NOTE = {
  name: 'Frontier commercial models (GPT, Claude, Gemini)',
  contextRange: 'roughly 200K to 1M+ tokens',
};

const byName = (n: string) => REFERENCE_MODELS.find(m => m.name === n)!;

/** A few reference values for one setting, for hints under the Setup sliders */
export const referenceHint = (key: 'dModel' | 'layers' | 'heads' | 'contextWindow' | 'vocabSize') =>
  ['GPT-2 small', 'Llama 3 8B', 'GPT-3']
    .map(byName)
    .map(m => `${m.name}: ${m[key].toLocaleString()}`)
    .join(' · ');

/** 1234 → "1.2K", 175e9 → "175B" */
export const compact = (n: number) => {
  const units: [number, string][] = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']];
  for (const [v, s] of units) {
    if (n >= v) {
      const x = n / v;
      return (x >= 100 ? x.toFixed(0) : x.toFixed(1).replace(/\.0$/, '')) + s;
    }
  }
  return Math.round(n).toString();
};
