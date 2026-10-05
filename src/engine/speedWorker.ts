// Measures how long one training step takes for a given model shape, on this device, in a
// background thread (the Setup tab uses it for its time estimates). It trains a throwaway model on
// made-up tokens, so the real model is never touched.
import { MicroTransformer } from './transformer';
import { TransformerConfig } from '../types';

export type ToSpeedWorker = { id: number; config: TransformerConfig };
export type FromSpeedWorker = { id: number; msPerStep: number };

/** Time spent measuring, after one warm-up step */
const BUDGET_MS = 600;

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<ToSpeedWorker>) => void) | null;
  postMessage(message: FromSpeedWorker): void;
};

scope.onmessage = e => {
  const { id, config } = e.data;
  const model = new MicroTransformer(config);
  const seq = Array.from({ length: config.contextWindow + 1 }, (_, i) => (i * 7 + 3) % config.vocabSize);
  const input = seq.slice(0, -1);
  const target = seq.slice(1);

  model.trainStep(input, target, config.learningRate); // warm-up: the first step is always slow
  let steps = 0;
  const t0 = performance.now();
  do {
    model.trainStep(input, target, config.learningRate);
    steps++;
  } while (performance.now() - t0 < BUDGET_MS || steps < 2);
  scope.postMessage({ id, msPerStep: (performance.now() - t0) / steps });
};
