# LLMStead

A browser-based tool for learning how LLMs work. It trains a small transformer from scratch in the browser and lets you inspect each step: BPE tokenization, embeddings, attention, the forward pass, and training loss. All the ML code (tensor math, backprop, optimizers, tokenizer) is hand-written with no libraries, so it can be read end to end.

The long-term goal is a tutorial covering the full pipeline, from tokenization through training, fine-tuning, and benchmarking.

## Running it

```bash
npm install
npm run dev
```

`npm run gradcheck` verifies the backprop against finite differences.
