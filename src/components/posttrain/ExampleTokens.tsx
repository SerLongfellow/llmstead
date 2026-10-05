import React from 'react';
import { BPETokenizer } from '../../engine/bpeTokenizer';
import { TokenizedExample, responseMask } from '../../engine/posttrain/examples';
import { TokenChip } from '../tokenUi';

/** Probability → background: rose when the model finds the token unlikely, emerald when likely */
export const probColor = (p: number) =>
  `color-mix(in srgb, var(--accent-emerald) ${Math.round(p * 100)}%, color-mix(in srgb, var(--accent-rose) 55%, transparent))`;

/**
 * One example as the model trains on it: prompt tokens greyed out (no loss), response tokens
 * coloured. With `probs` (P of each token given the ones before it), response tokens are
 * coloured by how likely the model now finds them.
 */
export const ExampleTokens: React.FC<{
  example: TokenizedExample;
  tokenizer: BPETokenizer;
  probs?: number[];
  /** Grey out the prompt (it gets no loss). Off when SFT trains on the prompt too. */
  maskPrompt?: boolean;
  tint?: string;
}> = ({ example, tokenizer, probs, maskPrompt = true, tint }) => {
  const mask = responseMask(example);
  const eos = tokenizer.specialId('<EOS>');
  return (
    <span style={{ display: 'inline-flex', flexWrap: 'wrap', gap: 3, alignItems: 'center' }}>
      {example.tokens.map((id, i) => {
        // Token i is predicted at position i − 1; the first token is only ever context
        const trained = i > 0 && (mask[i - 1] === 1 || !maskPrompt);
        const p = i > 0 ? probs?.[i - 1] : undefined;
        const text = id === eos ? '<EOS>' : tokenizer.decode([id]);
        return (
          <TokenChip
            key={i}
            text={text}
            bg={!trained ? 'transparent' : p !== undefined ? probColor(p) : tint ?? 'var(--emerald-soft)'}
            title={
              !trained
                ? 'Prompt: the model reads it, but is not trained to write it'
                : p !== undefined
                  ? `P = ${(p * 100).toFixed(1)}%`
                  : 'Trained on: the loss includes predicting this token'
            }
          />
        );
      })}
    </span>
  );
};

/** Warnings for one example: characters the tokenizer doesn't know, or a prompt cut to fit */
export const exampleWarnings = (ex: TokenizedExample) =>
  [
    ex.unknownChars.length > 0 && `unknown characters ${ex.unknownChars.map(c => JSON.stringify(c)).join(' ')} become <UNK>`,
    ex.truncated && 'too long for the context window: the start of the prompt is cut off',
  ].filter(Boolean) as string[];
