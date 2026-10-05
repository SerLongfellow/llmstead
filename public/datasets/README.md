# Downloadable datasets

Datasets too big to bundle with the site's code. The app downloads one only when someone picks it in Setup (see `DOWNLOADABLE_DATASETS` in `src/engine/datasets.ts`).

## tinystories.txt

The first 2,450 stories (1,999,485 bytes) of `TinyStoriesV2-GPT4-train.txt` from the TinyStories dataset:

- Source: https://huggingface.co/datasets/roneneldan/TinyStories
- Paper: Ronen Eldan and Yuanzhi Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" (2023), https://arxiv.org/abs/2305.07759
- License: Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0), https://cdla.dev/sharing-1-0/. This excerpt is shared under the same license.

Changes from the original: the `<|endoftext|>` separators became blank lines, curly quotes became straight quotes, en and em dashes became `-`, ellipsis characters became `...`, and runs of spaces, trailing spaces and blank lines inside a story were removed, so the text is plain ASCII and stories are separated by exactly one blank line. The stories themselves are unchanged and in their original order.
