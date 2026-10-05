// The page's tokenizers, one per (dataset text, vocabulary size).
//
// A small dataset's tokenizer trains in milliseconds, so App simply builds it when it needs it. A
// big one (TinyStories) takes several seconds, so before switching to it App calls
// prepareTokenizer, which trains it in a background thread (tokenizerWorker.ts) along with the
// dataset's train/validation encodings. The page stays responsive, and when App then asks for
// that tokenizer it's ready.
import { BPETokenizer } from './bpeTokenizer';
import { splitDataset } from './datasets';
import { FromTokenizerWorker, ToTokenizerWorker } from './tokenizerWorker';

/** Corpora at least this long are tokenized in the background */
export const BACKGROUND_TOKENIZE_CHARS = 200_000;

/** Prepared tokenizers, newest last; only a couple are kept (each holds its dataset's encodings) */
const prepared: { corpus: string; vocabSize: number; tokenizer: BPETokenizer }[] = [];
const KEEP = 2;

/** The tokenizer for this text and vocabulary size: the prepared one if there is, else trained now */
export function tokenizerFor(corpus: string, vocabSize: number): BPETokenizer {
  const ready = prepared.find(p => p.corpus === corpus && p.vocabSize === vocabSize);
  if (ready) return ready.tokenizer;
  const t = new BPETokenizer();
  t.train(corpus, vocabSize);
  return t;
}

let worker: Worker | null | undefined;
let nextId = 0;

/**
 * Get a big corpus's tokenizer ready in the background, so tokenizerFor returns it instantly.
 * Resolves right away for small corpora. If workers aren't available it resolves too, and
 * tokenizerFor trains on the page instead (slow, but it works).
 */
export async function prepareTokenizer(corpus: string, vocabSize: number): Promise<void> {
  if (corpus.length < BACKGROUND_TOKENIZE_CHARS || prepared.some(p => p.corpus === corpus && p.vocabSize === vocabSize)) return;
  if (worker === undefined) {
    try {
      worker = new Worker(new URL('./tokenizerWorker.ts', import.meta.url), { type: 'module' });
    } catch {
      worker = null;
    }
  }
  if (!worker) return;
  const w = worker;
  const id = ++nextId;
  // The Train and Post-train tabs encode the training and validation splits; do those here too
  const { trainText, valText } = splitDataset(corpus);
  const trained = await new Promise<FromTokenizerWorker['trained'] | null>(resolve => {
    const onMessage = (e: MessageEvent<FromTokenizerWorker>) => {
      if (e.data.id !== id) return;
      w.removeEventListener('message', onMessage);
      resolve(e.data.trained);
    };
    w.addEventListener('message', onMessage);
    w.addEventListener('error', () => resolve(null), { once: true });
    const message: ToTokenizerWorker = { id, corpus, vocabSize, alsoEncode: [trainText, valText] };
    w.postMessage(message);
  });
  if (!trained) return;
  prepared.push({ corpus, vocabSize, tokenizer: BPETokenizer.fromTrained(trained) });
  if (prepared.length > KEEP) prepared.shift();
}
