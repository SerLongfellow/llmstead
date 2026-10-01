# LLMStead

*Raise your own models.*

A browser-based tool for learning how LLMs work. It trains a small transformer from scratch in the browser and lets you inspect each step: BPE tokenization, embeddings, attention, the forward pass, and training loss. All the ML code (tensor math, backprop, optimizers, tokenizer) is hand-written with no libraries, so it can be read end to end. Homegrown, no frameworks added.

Pick a dataset, set up the tokenizer and architecture, then watch the model grow from random weights into something that (sort of) writes Shakespeare.

The long-term goal is a tutorial covering the full pipeline, from tokenization through training, fine-tuning, and benchmarking.

## Running it

```bash
npm install
npm run dev
```

`npm run gradcheck` verifies the backprop against finite differences.
