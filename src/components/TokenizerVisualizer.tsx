import React, { useState } from 'react';
import { BPETokenizerState } from '../types';
import { BPETokenizer } from '../engine/bpeTokenizer';
import { Binary, Play, RefreshCw, Layers } from 'lucide-react';
import { InfoTooltip } from './InfoTooltip';

interface TokenizerVisualizerProps {
  tokenizer: BPETokenizer;
  tokenizerState: BPETokenizerState;
  targetVocabSize: number;
  isCustomCorpus: boolean;
  onResetToDataset: () => void;
  onRetrainTokenizer: (corpus: string, targetVocabSize: number) => void;
}

export const TokenizerVisualizer: React.FC<TokenizerVisualizerProps> = ({
  tokenizer,
  tokenizerState,
  targetVocabSize,
  isCustomCorpus,
  onResetToDataset,
  onRetrainTokenizer,
}) => {
  const [corpusInput, setCorpusInput] = useState<string>(tokenizerState.corpusSample || 'The quick brown fox jumps over the lazy dog. To be or not to be, that is the question.');
  const [targetVocab, setTargetVocab] = useState<number>(targetVocabSize);
  const [testString, setTestString] = useState<string>('The question is whether to be or not to be!');

  const encodedTest = tokenizer.encode(testString);

  const handleTrain = () => {
    onRetrainTokenizer(corpusInput, targetVocab);
  };

  // Keep the textarea in sync when the tokenizer's corpus changes elsewhere (e.g. dataset switch)
  React.useEffect(() => {
    setCorpusInput(tokenizerState.corpusSample);
  }, [tokenizerState.corpusSample]);

  // Color palette for token chips
  const tokenColors = [
    'rgba(99, 102, 241, 0.25)',
    'rgba(16, 185, 129, 0.25)',
    'rgba(245, 158, 11, 0.25)',
    'rgba(6, 182, 212, 0.25)',
    'rgba(168, 85, 247, 0.25)',
    'rgba(244, 63, 94, 0.25)',
  ];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '20px' }}>
      {/* Left Column: BPE Corpus & Merge Rules */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Corpus Trainer Box */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Binary size={20} color="var(--primary)" />
              <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Train BPE Tokenizer</h2>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {isCustomCorpus && (
                <button className="btn-secondary" onClick={onResetToDataset} title="Retrain on the selected dataset's text">
                  Custom corpus · reset
                </button>
              )}
              <span className="badge badge-emerald">
                Vocab Size: {tokenizerState.vocab.size}
                {tokenizerState.vocab.size < targetVocabSize ? ` / ${targetVocabSize} target` : ''}
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>Training Corpus Text</label>
                <InfoTooltip
                  title="Training Corpus Text"
                  description="The raw sample text used by the BPE algorithm to discover high-frequency adjacent character pairs."
                  impact="Shapes the learned subword vocabulary (e.g. common prefixes, suffixes, and words)."
                />
              </div>
              <textarea
                rows={4}
                value={corpusInput}
                onChange={(e) => setCorpusInput(e.target.value)}
                style={{ width: '100%', fontSize: '0.85rem' }}
              />
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>Target Vocab Size</label>
                    <InfoTooltip
                      title="Target Vocabulary Size"
                      description="The maximum number of unique tokens (base characters + merged subwords) in the vocabulary."
                      impact="Larger vocabularies create longer subwords, reducing sequence length per sentence."
                    />
                  </div>
                  <span className="font-mono" style={{ fontSize: '0.8rem', color: 'var(--primary)' }}>{targetVocab}</span>
                </div>
                <input
                  type="range"
                  min={50}
                  max={300}
                  step={10}
                  value={targetVocab}
                  onChange={(e) => setTargetVocab(Number(e.target.value))}
                />
              </div>

              <button className="btn-primary" onClick={handleTrain} style={{ alignSelf: 'flex-end' }}>
                <RefreshCw size={16} /> Re-Train BPE
              </button>
            </div>
          </div>
        </div>

        {/* BPE Merge Rules History */}
        <div className="glass-panel" style={{ padding: '24px', flex: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Layers size={20} color="var(--accent-amber)" />
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>BPE Merge History</h3>
            </div>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              {tokenizerState.merges.length} Merges Learned
            </span>
          </div>

          <div style={{ maxHeight: '280px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px', paddingRight: '4px' }}>
            {tokenizerState.merges.length === 0 ? (
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No merges recorded yet. Click Re-Train BPE to generate merge pairs.</p>
            ) : (
              tokenizerState.merges.map((merge) => (
                <div
                  key={merge.step}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 12px',
                    background: 'rgba(15, 23, 42, 0.5)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    fontSize: '0.82rem'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="font-mono" style={{ color: 'var(--text-dim)', fontSize: '0.75rem' }}>#{merge.step}</span>
                    <span className="font-mono" style={{ color: 'var(--accent-cyan)' }}>"{merge.pair[0]}"</span>
                    <span style={{ color: 'var(--text-dim)' }}>+</span>
                    <span className="font-mono" style={{ color: 'var(--accent-cyan)' }}>"{merge.pair[1]}"</span>
                    <span style={{ color: 'var(--text-dim)' }}>➔</span>
                    <span className="font-mono" style={{ color: '#ffffff', fontWeight: 700, background: 'rgba(99, 102, 241, 0.3)', padding: '2px 6px', borderRadius: '4px' }}>
                      "{merge.newToken}"
                    </span>
                  </div>
                  <span className="badge badge-amber" style={{ fontSize: '0.7rem' }}>
                    Freq: {merge.freq}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Right Column: Live Encoding Visualizer */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Play size={20} color="var(--accent-emerald)" />
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Interactive Tokenizer Encoder</h2>
        </div>

        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>Test Input String</label>
            <InfoTooltip
              title="Test Input String"
              description="Any prompt or sentence to break down into subwords using the learned BPE merge rules."
              impact="Shows how text is tokenized into numeric Token IDs before entering embedding layers."
            />
          </div>
          <input
            type="text"
            value={testString}
            onChange={(e) => setTestString(e.target.value)}
            style={{ width: '100%', fontSize: '0.9rem' }}
          />
        </div>

        {/* Visual Token Chips Output */}
        <div>
          <p style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '10px' }}>
            Subword Token Breakdown ({encodedTest.tokens.length} tokens):
          </p>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', padding: '16px', background: 'rgba(15, 23, 42, 0.6)', borderRadius: '10px', border: '1px solid var(--border-color)', minHeight: '80px' }}>
            {encodedTest.tokens.map((tokId, idx) => {
              const str = encodedTest.tokenStrings[idx];
              const bg = tokenColors[idx % tokenColors.length];
              return (
                <div
                  key={idx}
                  style={{
                    background: bg,
                    border: '1px solid rgba(255, 255, 255, 0.15)',
                    borderRadius: '8px',
                    padding: '6px 12px',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '2px',
                    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.2)'
                  }}
                >
                  <span className="font-mono" style={{ fontSize: '0.95rem', fontWeight: 700, color: '#ffffff' }}>
                    "{str}"
                  </span>
                  <span className="font-mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    ID: {tokId}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Encoded Array representation */}
        <div style={{ background: 'rgba(15, 23, 42, 0.4)', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px' }}>Token ID Tensor Array [1 x SeqLen]:</p>
          <code className="font-mono" style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)' }}>
            [{encodedTest.tokens.join(', ')}]
          </code>
        </div>

        {/* Reconstructed Decoded Text */}
        <div style={{ background: 'rgba(15, 23, 42, 0.4)', padding: '14px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '4px' }}>Reconstructed (Decoded) Text:</p>
          <p style={{ fontSize: '0.9rem', fontWeight: 500, color: 'var(--accent-emerald)' }}>
            "{tokenizer.decode(encodedTest.tokens)}"
          </p>
        </div>
      </div>
    </div>
  );
};
