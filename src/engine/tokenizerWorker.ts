// Trains a BPE tokenizer in a background thread. For a big dataset that takes seconds, which would
// freeze the page; see tokenizers.ts for how the page uses it.
import { BPETokenizer, TrainedTokenizer } from './bpeTokenizer';

export type ToTokenizerWorker = { id: number; corpus: string; vocabSize: number; alsoEncode: string[] };
export type FromTokenizerWorker = { id: number; trained: TrainedTokenizer };

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<ToTokenizerWorker>) => void) | null;
  postMessage(message: FromTokenizerWorker, transfer?: Transferable[]): void;
};

scope.onmessage = e => {
  const { id, corpus, vocabSize, alsoEncode } = e.data;
  const tokenizer = new BPETokenizer();
  tokenizer.train(corpus, vocabSize);
  const trained = tokenizer.exportTrained([corpus, ...alsoEncode]);
  scope.postMessage({ id, trained }, trained.encodings.map(enc => enc.tokens.buffer));
};
