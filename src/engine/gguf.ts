// Export a trained model as a GGUF file: the single-file format llama.cpp, Ollama and LM Studio
// load. A GGUF file holds no code, only numbers plus labels. Its metadata says "this is a gpt2"
// and lists the vocabulary; the program reading it runs its own GPT-2 implementation on the
// weights. So the export only works because MicroTransformer computes exactly what GPT-2 does
// (pre-LayerNorm blocks, learned position embeddings, GELU, a final LayerNorm), minus the
// biases and LayerNorm scale/shift, which are written as zeros and ones.
//
// File layout (GGUF version 3, all little-endian):
//   "GGUF" · version · tensor count · metadata count
//   metadata: key, value type, value          (architecture, sizes, tokenizer)
//   tensor index: name, shape, type, offset   (where each tensor's numbers sit in the data section)
//   padding to 32 bytes, then every tensor's float32 numbers, each aligned to 32 bytes
import { MicroTransformer } from './transformer';
import { BPETokenizer } from './bpeTokenizer';
import { Matrix } from './tensor';

const ALIGNMENT = 32;

/**
 * Rows in the exported position table. llama.cpp looks positions up in this table and stops
 * with an error past its end, and the context is never as small as the model's: apps default to
 * thousands of tokens, and llama.cpp rounds any context up to at least 256 (Ollama's num_ctx 16
 * still ran a 256-token window). So the table is padded: positions past the trained ones reuse
 * the last trained row. Past contextWindow tokens the text degrades, but nothing crashes.
 */
const SERVED_CONTEXT = 4096;

/**
 * Chat apps (LM Studio, llama-server's web UI) format messages with the file's Jinja chat
 * template. This model has no chat format, so the template joins the messages as plain text
 * (system prompts dropped): a chat is one running text that the model keeps continuing.
 */
const RAW_CHAT_TEMPLATE = "{% for message in messages %}{% if message['role'] != 'system' %}{{ message['content'] }}{% endif %}{% endfor %}";

// GGUF metadata value types
const U32 = 4;
const I32 = 5;
const F32 = 6;
const BOOL = 7;
const STRING = 8;
const ARRAY = 9;

// llama.cpp token types
const TOKEN_NORMAL = 1;
const TOKEN_UNKNOWN = 2;
const TOKEN_CONTROL = 3;
const TOKEN_BYTE = 6;

/** SentencePiece writes spaces as ▁ (U+2581) in its vocabulary; llama.cpp expects the same */
const SPACE_MARK = '▁';

type MetaValue =
  | { type: typeof U32 | typeof I32 | typeof F32; value: number }
  | { type: typeof BOOL; value: boolean }
  | { type: typeof STRING; value: string }
  | { type: typeof ARRAY; itemType: typeof I32 | typeof F32 | typeof STRING; value: (number | string)[] };

interface TensorEntry {
  name: string;
  /** GGUF lists dimensions innermost first: a [rows × cols] row-major matrix is [cols, rows] */
  dims: number[];
  data: Float32Array;
}

/**
 * GGUF stores a linear layer the way PyTorch does: one row per *output*, so llama.cpp computes
 * W·x. This engine computes x·W with one row per *input*, so every projection is transposed.
 */
function transposed(M: Matrix): TensorEntry['data'] {
  const rows = M.length;
  const cols = M[0].length;
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[c * rows + r] = M[r][c];
  return out;
}

function flat(M: Matrix): Float32Array {
  return Float32Array.from(M.flat());
}

function filled(n: number, value: number): Float32Array {
  return new Float32Array(n).fill(value);
}

/**
 * The vocabulary for llama.cpp's SentencePiece-style tokenizer ("llama" model), which suits this
 * BPE: like it, it never splits text into words first, and it merges the highest-scoring adjacent
 * pair whose joined text is a token. Scoring each merged token by how early its merge was learned
 * makes it apply merges in the same order as BPETokenizer.encode. llama.cpp's GPT-2 tokenizer
 * would split text at spaces and punctuation first, so merges like " = 2" could never form.
 *
 * 256 byte tokens (<0x00>…<0xFF>) are appended: llama.cpp falls back to them for any character
 * missing from the vocabulary, where this engine would use <UNK>.
 */
function buildVocab(tokenizer: BPETokenizer) {
  const state = tokenizer.getState();
  const n = tokenizer.getVocabSize();
  const mergeStep = new Map(state.merges.map(m => [m.newToken, m.step]));
  const tokens: string[] = [];
  const scores: number[] = [];
  const types: number[] = [];
  for (let id = 0; id < n; id++) {
    const text = state.idToToken.get(id) ?? '';
    tokens.push(text.replace(/ /g, SPACE_MARK));
    scores.push(-(mergeStep.get(text) ?? 0));
    types.push(text === '<UNK>' ? TOKEN_UNKNOWN : /^<(PAD|BOS|EOS)>$/.test(text) ? TOKEN_CONTROL : TOKEN_NORMAL);
  }
  for (let b = 0; b < 256; b++) {
    tokens.push(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`);
    scores.push(0);
    types.push(TOKEN_BYTE);
  }
  const id = (t: string) => state.tokenToId.get(t) ?? 0;
  return { tokens, scores, types, unk: id('<UNK>'), pad: id('<PAD>'), bos: id('<BOS>'), eos: id('<EOS>') };
}

/** Build the GGUF file for `model`, ready to save as e.g. llmstead.gguf */
export function exportGguf(model: MicroTransformer, tokenizer: BPETokenizer, name = 'LLMStead'): Uint8Array<ArrayBuffer> {
  const { contextWindow, dModel, numHeads, numLayers, mlpRatio } = model.config;
  const dMlp = dModel * mlpRatio;
  const p = model.getParameters();
  const vocab = buildVocab(tokenizer);
  const nVocab = vocab.tokens.length;
  const nOwn = p.wTokenEmbed.length;

  // Byte tokens get <UNK>'s embedding and output column. <UNK> never appears as a training
  // target, so training has pushed its score low: the model won't predict byte tokens either.
  const unkRow = p.wTokenEmbed[vocab.unk];
  const tokenEmbed: Matrix = [...p.wTokenEmbed, ...Array.from({ length: nVocab - nOwn }, () => unkRow)];
  const head: Matrix = p.wHead.map(row => [...row, ...new Array(nVocab - nOwn).fill(row[vocab.unk])]);

  const servedContext = Math.max(contextWindow, SERVED_CONTEXT);
  const posTable: Matrix = Array.from({ length: servedContext }, (_, i) => p.wPosEmbed[Math.min(i, contextWindow - 1)]);

  const ones = () => filled(dModel, 1);
  const zeros = (n: number) => filled(n, 0);
  const tensors: TensorEntry[] = [
    { name: 'token_embd.weight', dims: [dModel, nVocab], data: flat(tokenEmbed) },
    { name: 'position_embd.weight', dims: [dModel, servedContext], data: flat(posTable) },
  ];
  for (let l = 0; l < numLayers; l++) {
    // GPT-2 keeps Q, K and V in one matrix: output rows 0..d are Q, then K, then V
    const qkv = new Float32Array(3 * dModel * dModel);
    qkv.set(transposed(p[`wQ.${l}`]), 0);
    qkv.set(transposed(p[`wK.${l}`]), dModel * dModel);
    qkv.set(transposed(p[`wV.${l}`]), 2 * dModel * dModel);
    const blk = `blk.${l}`;
    tensors.push(
      { name: `${blk}.attn_norm.weight`, dims: [dModel], data: ones() },
      { name: `${blk}.attn_norm.bias`, dims: [dModel], data: zeros(dModel) },
      { name: `${blk}.attn_qkv.weight`, dims: [dModel, 3 * dModel], data: qkv },
      { name: `${blk}.attn_qkv.bias`, dims: [3 * dModel], data: zeros(3 * dModel) },
      { name: `${blk}.attn_output.weight`, dims: [dModel, dModel], data: transposed(p[`wO.${l}`]) },
      { name: `${blk}.attn_output.bias`, dims: [dModel], data: zeros(dModel) },
      { name: `${blk}.ffn_norm.weight`, dims: [dModel], data: ones() },
      { name: `${blk}.ffn_norm.bias`, dims: [dModel], data: zeros(dModel) },
      { name: `${blk}.ffn_up.weight`, dims: [dModel, dMlp], data: transposed(p[`wMlp1.${l}`]) },
      { name: `${blk}.ffn_up.bias`, dims: [dMlp], data: zeros(dMlp) },
      { name: `${blk}.ffn_down.weight`, dims: [dMlp, dModel], data: transposed(p[`wMlp2.${l}`]) },
      { name: `${blk}.ffn_down.bias`, dims: [dModel], data: zeros(dModel) },
    );
  }
  tensors.push(
    { name: 'output_norm.weight', dims: [dModel], data: ones() },
    { name: 'output_norm.bias', dims: [dModel], data: zeros(dModel) },
    { name: 'output.weight', dims: [dModel, nVocab], data: transposed(head) },
  );

  const u32 = (value: number): MetaValue => ({ type: U32, value });
  const metadata: [string, MetaValue][] = [
    ['general.architecture', { type: STRING, value: 'gpt2' }],
    ['general.name', { type: STRING, value: name }],
    ['general.file_type', u32(0)], // 0 = every tensor is float32
    ['general.alignment', u32(ALIGNMENT)],
    ['gpt2.context_length', u32(servedContext)],
    ['llmstead.trained_context_length', u32(contextWindow)], // the real one; see SERVED_CONTEXT
    ['gpt2.embedding_length', u32(dModel)],
    ['gpt2.feed_forward_length', u32(dMlp)],
    ['gpt2.block_count', u32(numLayers)],
    ['gpt2.attention.head_count', u32(numHeads)],
    ['gpt2.attention.layer_norm_epsilon', { type: F32, value: 1e-5 }], // matches MatrixMath.layerNorm
    ['tokenizer.ggml.model', { type: STRING, value: 'llama' }],
    ['tokenizer.ggml.tokens', { type: ARRAY, itemType: STRING, value: vocab.tokens }],
    ['tokenizer.ggml.scores', { type: ARRAY, itemType: F32, value: vocab.scores }],
    ['tokenizer.ggml.token_type', { type: ARRAY, itemType: I32, value: vocab.types }],
    ['tokenizer.ggml.unknown_token_id', u32(vocab.unk)],
    ['tokenizer.ggml.padding_token_id', u32(vocab.pad)],
    ['tokenizer.ggml.bos_token_id', u32(vocab.bos)],
    ['tokenizer.ggml.eos_token_id', u32(vocab.eos)],
    // The model was trained on raw text windows: no start token, no space added in front
    ['tokenizer.ggml.add_bos_token', { type: BOOL, value: false }],
    ['tokenizer.ggml.add_eos_token', { type: BOOL, value: false }],
    ['tokenizer.ggml.add_space_prefix', { type: BOOL, value: false }],
    ['tokenizer.chat_template', { type: STRING, value: RAW_CHAT_TEMPLATE }],
  ];

  return writeGguf(metadata, tensors);
}

function writeGguf(metadata: [string, MetaValue][], tensors: TensorEntry[]): Uint8Array<ArrayBuffer> {
  const utf8 = new TextEncoder();
  const chunks: Uint8Array[] = [];
  let size = 0;
  const push = (bytes: Uint8Array) => {
    chunks.push(bytes);
    size += bytes.length;
  };
  const num = (bytes: number, set: (v: DataView) => void) => {
    const b = new Uint8Array(bytes);
    set(new DataView(b.buffer));
    push(b);
  };
  const u32 = (v: number) => num(4, d => d.setUint32(0, v, true));
  const i32 = (v: number) => num(4, d => d.setInt32(0, v, true));
  const f32 = (v: number) => num(4, d => d.setFloat32(0, v, true));
  const u64 = (v: number) => num(8, d => d.setBigUint64(0, BigInt(v), true));
  const str = (s: string) => {
    const b = utf8.encode(s);
    u64(b.length);
    push(b);
  };
  const pad = () => {
    const extra = (ALIGNMENT - (size % ALIGNMENT)) % ALIGNMENT;
    if (extra) push(new Uint8Array(extra));
  };
  const item = (type: number, v: number | string | boolean) => {
    if (type === U32) u32(v as number);
    else if (type === I32) i32(v as number);
    else if (type === F32) f32(v as number);
    else if (type === BOOL) push(new Uint8Array([v ? 1 : 0]));
    else str(v as string);
  };

  push(utf8.encode('GGUF'));
  u32(3);
  u64(tensors.length);
  u64(metadata.length);
  for (const [key, meta] of metadata) {
    str(key);
    u32(meta.type);
    if (meta.type === ARRAY) {
      u32(meta.itemType);
      u64(meta.value.length);
      for (const v of meta.value) item(meta.itemType, v);
    } else {
      item(meta.type, meta.value);
    }
  }

  let offset = 0;
  for (const t of tensors) {
    str(t.name);
    u32(t.dims.length);
    for (const d of t.dims) u64(d);
    u32(0); // 0 = float32
    u64(offset);
    offset += Math.ceil((t.data.length * 4) / ALIGNMENT) * ALIGNMENT;
  }
  pad();

  for (const t of tensors) {
    push(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength)); // float32 is little-endian on every browser platform
    pad();
  }

  const out = new Uint8Array(size);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}
