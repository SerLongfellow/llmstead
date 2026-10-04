export interface TransformerConfig {
  vocabSize: number;
  contextWindow: number;
  dModel: number;
  numHeads: number;
  numLayers: number;
  mlpRatio: number;
  learningRate: number;
  optimizer: 'adamw' | 'sgd';
}

export interface BPEMerge {
  step: number;
  pair: [string, string];
  newToken: string;
  freq: number;
}

export interface BPETokenizerState {
  vocab: Map<number, string>;
  merges: BPEMerge[];
  tokenToId: Map<string, number>;
  idToToken: Map<number, string>;
  corpusSample: string;
}

export interface LayerInspection {
  layerIndex: number;
  // Per head: [headIndex][seqLen][headDim]
  queries: number[][][];
  keys: number[][][];
  values: number[][][];
  // Per head: [headIndex][seqLen][seqLen]
  rawAttentionScores: number[][][];
  attentionWeights: number[][][]; // Softmax + Causal Masking applied
  headOutputs: number[][][]; // [headIndex][seqLen][headDim]
  concatOutput: number[][]; // [seqLen][dModel]
  norm1Output: number[][]; // [seqLen][dModel] LayerNorm of the block input: what attention reads
  afterAttention: number[][]; // [seqLen][dModel] block input + attention output (residual add)
  norm2Output: number[][]; // [seqLen][dModel] LayerNorm of afterAttention: what the MLP reads
  mlpHidden: number[][]; // [seqLen][dModel * mlpRatio]
  mlpOutput: number[][]; // [seqLen][dModel]
  blockOutput: number[][]; // [seqLen][dModel] afterAttention + MLP output (residual add)
}

export interface StepInspectionData {
  inputString: string;
  tokens: number[];
  tokenStrings: string[];
  tokenEmbeddings: number[][]; // [seqLen][dModel]
  positionEmbeddings: number[][]; // [seqLen][dModel]
  combinedEmbeddings: number[][]; // [seqLen][dModel]
  layerInspections: LayerInspection[];
  finalNorm: number[][]; // [seqLen][dModel]
  logits: number[][]; // [seqLen][vocabSize]
  probabilities: number[][]; // [seqLen][vocabSize]
}

export interface TrainingMetric {
  step: number;
  loss: number;
  perplexity: number;
  sampleText: string;
  timestamp: number;
}

export interface DatasetOption {
  id: string;
  name: string;
  description: string;
  text: string;
  category: 'literature' | 'code' | 'synthetic' | 'logic';
  /** A good example prompt: used as the default Pipeline input and generation seed */
  samplePrompt?: string;
}
