import { BPEMerge, BPETokenizerState } from '../types';

/** A trained tokenizer as plain data (see exportTrained), with some texts already encoded */
export interface TrainedTokenizer {
  vocab: string[];
  merges: BPEMerge[];
  mergeIds: { a: number; b: number; id: number; replaces: number }[];
  encodings: { text: string; tokens: Int32Array }[];
}

/** Texts at least this long have their encoding cached (a whole dataset, not a prompt) */
const CACHE_MIN_CHARS = 20_000;
const CACHE_ENTRIES = 4;

/**
 * One merge applied to ids[0..length), in place: every (a, b) pair, left to right, becomes newId,
 * and any `replaces` id (an older token with the same text) is renamed to newId. Returns the new length.
 */
function applyMerge(ids: Int32Array, length: number, a: number, b: number, newId: number, replaces: number): number {
  let write = 0;
  let read = 0;
  while (read < length) {
    const id = ids[read];
    if (id === a && read + 1 < length && ids[read + 1] === b) {
      ids[write++] = newId;
      read += 2;
    } else {
      ids[write++] = id === replaces ? newId : id;
      read++;
    }
  }
  return write;
}

export class BPETokenizer {
  private vocab: Map<number, string> = new Map();
  private tokenToId: Map<string, number> = new Map();
  private idToToken: Map<number, string> = new Map();
  private merges: BPEMerge[] = [];
  /** The same merges as ids, which is what encode replays */
  private mergeIds: { a: number; b: number; id: number; replaces: number }[] = [];
  private cache: Map<string, { tokens: number[]; tokenStrings: string[] }> = new Map();
  private isTrained: boolean = false;

  constructor() {
    this.reset();
  }

  public reset() {
    this.vocab.clear();
    this.tokenToId.clear();
    this.idToToken.clear();
    this.merges = [];
    this.mergeIds = [];
    this.cache.clear();
    this.isTrained = false;
  }

  /**
   * Train BPE tokenizer on text corpus up to targetVocabSize
   */
  public train(corpus: string, targetVocabSize: number = 100): BPETokenizerState {
    this.reset();

    // 1. Initialize base vocabulary with full standard ASCII character set + corpus unique chars
    const baseAscii = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?:;'\"-_\n\t()[]{}<>/=";
    const uniqueChars = Array.from(new Set(baseAscii + corpus));
    let currentId = 0;

    // Standard special tokens
    const specialTokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'];
    specialTokens.forEach(st => {
      this.vocab.set(currentId, st);
      this.tokenToId.set(st, currentId);
      this.idToToken.set(currentId, st);
      currentId++;
    });

    uniqueChars.forEach(ch => {
      if (!this.tokenToId.has(ch)) {
        this.vocab.set(currentId, ch);
        this.tokenToId.set(ch, currentId);
        this.idToToken.set(currentId, ch);
        currentId++;
      }
    });

    // 2. The corpus as one token id per character. Ids rather than strings because a 2 MB dataset
    // is millions of tokens, and comparing numbers is far faster than comparing strings.
    let ids = this.charIds(corpus);
    let length = ids.length;
    // Pair counts in a flat table: pair (a, b) lives at a * width + b
    const width = Math.max(targetVocabSize, currentId);
    const pairCounts = new Int32Array(width * width);

    let step = 1;
    while (this.vocab.size < targetVocabSize && length > 1) {
      // Count every adjacent pair
      pairCounts.fill(0);
      let maxFreq = 0;
      for (let i = 0; i < length - 1; i++) {
        const c = ++pairCounts[ids[i] * width + ids[i + 1]];
        if (c > maxFreq) maxFreq = c;
      }

      // If no pair appears more than once, stop training
      if (maxFreq <= 1) break;

      // Ties go to the pair that appears first in the text
      let first = 0;
      while (pairCounts[ids[first] * width + ids[first + 1]] !== maxFreq) first++;
      const a = ids[first];
      const b = ids[first + 1];
      const newToken = this.idToToken.get(a)! + this.idToToken.get(b)!;

      // Register new token in vocabulary. Two different merges can spell the same text ("ab" + "c"
      // and "a" + "bc"); the newest id then stands for that text, so older copies are renamed to it.
      const newTokenId = currentId++;
      const replaces = this.tokenToId.get(newToken);
      this.vocab.set(newTokenId, newToken);
      this.tokenToId.set(newToken, newTokenId);
      this.idToToken.set(newTokenId, newToken);

      this.merges.push({ step, pair: [this.idToToken.get(a)!, this.idToToken.get(b)!], newToken, freq: maxFreq });
      this.mergeIds.push({ a, b, id: newTokenId, replaces: replaces ?? -1 });

      length = applyMerge(ids, length, a, b, newTokenId, replaces ?? -1);
      step++;
    }

    this.isTrained = true;
    // What's left is the corpus encoded, which is usually the next thing asked for
    ids = ids.subarray(0, length);
    this.remember(corpus, { tokens: Array.from(ids), tokenStrings: Array.from(ids, id => this.idToToken.get(id)!) });

    return this.getState(corpus);
  }

  /**
   * Text → token ids. Starts from one token per character and replays the merges in the order they
   * were learned, so text is split exactly the way training split it.
   *
   * The result for a long text (a whole dataset) is cached and shared: don't modify its arrays.
   */
  public encode(text: string): { tokens: number[]; tokenStrings: string[] } {
    if (!text) return { tokens: [], tokenStrings: [] };
    const cached = this.cache.get(text);
    if (cached) return cached;

    let ids = this.charIds(text);
    let length = ids.length;
    for (const m of this.mergeIds) length = applyMerge(ids, length, m.a, m.b, m.id, m.replaces);
    ids = ids.subarray(0, length);

    // A character the vocabulary doesn't have becomes <UNK> (but is still shown as itself)
    const unk = this.tokenToId.get('<UNK>') ?? 1;
    const result = {
      tokens: Array.from(ids, id => (id < 0 ? unk : id)),
      tokenStrings: Array.from(ids, id => (id < 0 ? String.fromCharCode(-id - 1) : this.idToToken.get(id)!)),
    };
    this.remember(text, result);
    return result;
  }

  /** One id per character (UTF-16 unit, like text.split('')); unknown characters get -(char code + 1) */
  private charIds(text: string): Int32Array {
    const ids = new Int32Array(text.length);
    for (let i = 0; i < text.length; i++) ids[i] = this.tokenToId.get(text[i]) ?? -text.charCodeAt(i) - 1;
    return ids;
  }

  /** Keeps the encodings of a few long texts (datasets get encoded by several tabs) */
  private remember(text: string, result: { tokens: number[]; tokenStrings: string[] }) {
    if (text.length < CACHE_MIN_CHARS) return;
    if (this.cache.size >= CACHE_ENTRIES) this.cache.delete(this.cache.keys().next().value!);
    this.cache.set(text, result);
  }

  public decode(tokens: number[]): string {
    return tokens
      .map(id => this.idToToken.get(id) ?? '<UNK>')
      .join('')
      .replace(/<PAD>|<UNK>|<BOS>|<EOS>/g, '');
  }

  public getState(corpusSample: string = ''): BPETokenizerState {
    return {
      vocab: this.vocab,
      merges: this.merges,
      tokenToId: this.tokenToId,
      idToToken: this.idToToken,
      corpusSample
    };
  }

  public getVocabSize(): number {
    return this.vocab.size;
  }

  /**
   * Everything training learned, as plain data (so it can be posted from a worker), plus the
   * encodings of the given texts. Rebuild it with fromTrained.
   */
  public exportTrained(texts: string[] = []): TrainedTokenizer {
    return {
      vocab: Array.from(this.idToToken.values()),
      merges: this.merges,
      mergeIds: this.mergeIds,
      encodings: texts.map(text => ({ text, tokens: Int32Array.from(this.encode(text).tokens) })),
    };
  }

  /** The tokenizer exportTrained described, with its encodings already cached */
  public static fromTrained(data: TrainedTokenizer): BPETokenizer {
    const t = new BPETokenizer();
    // Ids in order, so a token text that two merges produced ends up pointing at the newer id, as in training
    data.vocab.forEach((token, id) => {
      t.vocab.set(id, token);
      t.idToToken.set(id, token);
      t.tokenToId.set(token, id);
    });
    t.merges = data.merges;
    t.mergeIds = data.mergeIds;
    t.isTrained = true;
    for (const { text, tokens } of data.encodings) {
      t.remember(text, { tokens: Array.from(tokens), tokenStrings: Array.from(tokens, id => t.idToToken.get(id)!) });
    }
    return t;
  }

  /** Id of a special token such as '<EOS>'. Pre-training never shows the model one; SFT can teach it to end a reply with '<EOS>'. */
  public specialId(name: '<PAD>' | '<UNK>' | '<BOS>' | '<EOS>'): number {
    return this.tokenToId.get(name)!;
  }
}
