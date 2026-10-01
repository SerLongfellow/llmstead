# LLMStead

Local React + Vite + TypeScript site for learning how LLMs work: train a tiny transformer from scratch in the browser and inspect every step. Long-term goal: a detailed tutorial covering the pipeline from tokenization → training → fine-tuning/RLHF → benchmarks.

## Commands
- `npm run dev` — dev server
- `npm run build` — `tsc && vite build` (output in `dist/`, which is stale; rebuild)

## Layout
- `src/engine/` — hand-written, dependency-free ML code (intentional: it's meant to be read)
  - `tensor.ts` — `MatrixMath`: matmul, transpose, softmax (+causal mask), layerNorm (no gain/bias), GELU. Plain `number[][]`.
  - `transformer.ts` — `MicroTransformer`. **Post-LayerNorm** blocks (residual add → LN), no biases. `inspectForwardPass` captures every intermediate for the visualizers. `computeGradients` (backprop), `trainStep`, `evaluateLoss` (forward-only, for validation), `setOptimizer`. Throws if `dModel % numHeads !== 0`.
  - `optimizer.ts` — `Optimizer` (SGD with L2, AdamW with decoupled weight decay), per-named-matrix state.
  - `bpeTokenizer.ts` — character-level BPE with merge history (no pre-tokenization; merges can cross word boundaries).
  - `datasets.ts` — 4 small sample datasets + `splitDataset()` (last ~15% held out, cut at a paragraph/line break).
  - `generate.ts` — `sampleToken` (greedy when T ≤ 0.1) and `generateContinuation`.
  - `benchmarks.ts` — prompt→expected-answer suites per dataset; `runBenchmarkSuite` greedy-decodes and labels each case `seen` vs `held-out` by checking the actual training text.
- Tabs, in order (`TABS` in Navbar.tsx): Start here (`StartView`: intro, the 3-step path, suggested experiments) → ① Set up → ② Train → ③ Look inside (the Pipeline). Each step has a dismissible `GuideStrip` (what to do + Next button; hidden state in localStorage `llmstead.hideGuides`, re-enabled from Start here).
- `src/components/` — one component per tab: `SetupView` (numbered steps: 1 Data, 2 Tokenizer/vocab size, 3 Architecture, 4 Training settings, plus a sticky parameter breakdown; steps 1–3 reset the model, step 4 does not), `PipelineView`, `TrainingDashboard` (shows the dataset read-only; changing it is done in Setup) with `BenchmarkPanel` inside.
  - `PipelineView` — the whole forward pass as a strip of stage cards (text → tokens → embeddings → blocks → probabilities → next token) with mini heatmaps; click a stage for its full detail view (read-only BPE `MergeHistory`, embedding/prediction tables, per-head attention via `AttentionGrid`, Q/K/V scores), click a token to follow its row through every stage. Re-runs inspection on mount so it reflects the latest weights.
  - `ScaleComparison` (bottom of Setup) compares the model to published architectures in `src/engine/referenceModels.ts` (GPT-2, GPT-3, Llama 2/3/3.1); undisclosed commercial models appear only as a context-window range. The Setup sliders cite them too; scale comparisons are deliberately kept to Setup.
  - `EmbeddingSpace` (inside the Embeddings stage) — token math (`a + b − c`, nearest by cosine; multi-token terms are averaged, quote a term to keep spaces), nearest neighbours of the followed token, and a 2D PCA map of the vocabulary. It reads a **snapshot** of `wTokenEmbed` taken per run, because training mutates the live table in place (memos keyed on it would never refresh). Shared chip/cosine helpers live in `tokenUi.tsx`.
- `src/App.tsx` — owns state, including the dataset list (built-ins + custom text added in Setup). Tokenizer is rebuilt via `useMemo` from the selected dataset + `config.vocabSize` (the *target* vocab, set only by Setup's vocab slider; the tokenizer is deliberately not retrainable elsewhere, since a new vocabulary means a new model); never mutate it in place. The model is memoized **only on shape-changing settings**, so the learning rate and optimizer can change without resetting weights. Children get `effectiveConfig` (vocabSize = real tokenizer vocab).

## Backprop
`computeGradients` in `transformer.ts` is a full manual backward pass (per-op rules for LayerNorm, GELU and softmax live in `tensor.ts` as `*Backward`). `trainStep` clips to global grad-norm 1.0, then sends **every** parameter through `this.optimizer.step()`. `npm run gradcheck` (esbuild-bundles `scripts/gradcheck.ts`) compares every matrix against central finite differences (`src/engine/gradCheck.ts`); expect relative error ~1e-7. Consider making the derivation a tutorial tab.
- With real gradients, AdamW lr ≈ 0.001 works best; the app default is 0.001 and the Setup slider is linear 0.001–0.05 (a log-scale slider is a good follow-up; SGD likely needs 0.05–1, untested).
- Training windows are sampled at random from the train split (not slid sequentially).
- Continuous training runs as many steps as fit in ~100 ms per tick (`TICK_BUDGET_MS`), then records one chart point (mean loss); validation runs every 50 steps on ≤32 fixed windows. The Step button is still exactly one step.

## Other notes / ideas
- Shakespeare dataset is Coriolanus 1.1 + 1.3 (~15k chars, typed from memory, verse lines joined per speech); its validation split starts at scene 3.
- The other datasets are ~1.3–2.1k training tokens each (at vocab 120). math-logic is generated in `buildMathLogicText()` (seeded shuffle); each non-Shakespeare dataset keeps its original lines first (so benchmark "seen" cases stay in the train split) and deliberately omits the benchmark's held-out strings. Keep it that way when editing datasets or benchmarks.
- The Training Dashboard stays mounted (hidden with `display: none`) so its loss history survives tab switches and training keeps running in the background; the navbar shows a pulsing dot on Train while it runs. The other tabs are conditionally rendered.
- "Reset" on the training chart clears history only; weights re-init when the architecture, tokenizer or dataset changes.
- Future direction discussed: keep this in-browser toy engine for mechanics, and add a separate real-model tier (Python + PyTorch/Hugging Face backend, or transformers.js/WebGPU) for importing base models, LoRA/DPO fine-tuning, and benchmarking. Reuse the same visualizations across both.
