import { BPEMerge, BPETokenizerState } from '../types';

export class BPETokenizer {
  private vocab: Map<number, string> = new Map();
  private tokenToId: Map<string, number> = new Map();
  private idToToken: Map<number, string> = new Map();
  private merges: BPEMerge[] = [];
  private isTrained: boolean = false;

  constructor() {
    this.reset();
  }

  public reset() {
    this.vocab.clear();
    this.tokenToId.clear();
    this.idToToken.clear();
    this.merges = [];
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

    // 2. Tokenize corpus into character token arrays
    let words = corpus.split('').map(char => char);

    let step = 1;
    while (this.vocab.size < targetVocabSize) {
      // Count frequency of adjacent pairs
      const pairFreqs: Map<string, { pair: [string, string]; count: number }> = new Map();

      for (let i = 0; i < words.length - 1; i++) {
        const pairKey = `${words[i]}|||${words[i + 1]}`;
        const existing = pairFreqs.get(pairKey);
        if (existing) {
          existing.count++;
        } else {
          pairFreqs.set(pairKey, { pair: [words[i], words[i + 1]], count: 1 });
        }
      }

      if (pairFreqs.size === 0) break;

      // Find pair with max frequency
      let maxFreq = 0;
      let bestPair: [string, string] | null = null;

      for (const item of pairFreqs.values()) {
        if (item.count > maxFreq) {
          maxFreq = item.count;
          bestPair = item.pair;
        }
      }

      // If no pair appears more than once, stop training
      if (!bestPair || maxFreq <= 1) break;

      const newToken = bestPair[0] + bestPair[1];
      
      // Register new token in vocabulary
      const newTokenId = currentId++;
      this.vocab.set(newTokenId, newToken);
      this.tokenToId.set(newToken, newTokenId);
      this.idToToken.set(newTokenId, newToken);

      const mergeInfo: BPEMerge = {
        step,
        pair: bestPair,
        newToken,
        freq: maxFreq
      };
      this.merges.push(mergeInfo);

      // Perform replace merge across word list
      const newWords: string[] = [];
      let i = 0;
      while (i < words.length) {
        if (i < words.length - 1 && words[i] === bestPair[0] && words[i + 1] === bestPair[1]) {
          newWords.push(newToken);
          i += 2;
        } else {
          newWords.push(words[i]);
          i++;
        }
      }
      words = newWords;
      step++;
    }

    this.isTrained = true;

    return this.getState(corpus);
  }

  public encode(text: string): { tokens: number[]; tokenStrings: string[] } {
    if (!text) return { tokens: [], tokenStrings: [] };

    // Initial character split
    let tokens = text.split('');

    // Apply merges sequentially
    for (const merge of this.merges) {
      const [first, second] = merge.pair;
      const combined = merge.newToken;
      
      const newTokens: string[] = [];
      let i = 0;
      while (i < tokens.length) {
        if (i < tokens.length - 1 && tokens[i] === first && tokens[i + 1] === second) {
          newTokens.push(combined);
          i += 2;
        } else {
          newTokens.push(tokens[i]);
          i++;
        }
      }
      tokens = newTokens;
    }

    // Convert string tokens to numerical IDs
    const tokenIds = tokens.map(t => {
      const id = this.tokenToId.get(t);
      return id !== undefined ? id : (this.tokenToId.get('<UNK>') ?? 1);
    });

    return { tokens: tokenIds, tokenStrings: tokens };
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

  /** Id of a special token such as '<EOS>'. Pre-training never shows the model one; SFT can teach it to end a reply with '<EOS>'. */
  public specialId(name: '<PAD>' | '<UNK>' | '<BOS>' | '<EOS>'): number {
    return this.tokenToId.get(name)!;
  }
}
