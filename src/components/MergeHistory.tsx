import React from 'react';
import { BPETokenizerState } from '../types';
import { Layers } from 'lucide-react';

/** Read-only list of the merges BPE learned, in order (shown in the Pipeline's Tokens stage). */
export const MergeHistory: React.FC<{ tokenizerState: BPETokenizerState }> = ({ tokenizerState }) => (
  <div style={{ background: 'rgba(15, 23, 42, 0.4)', border: '1px solid var(--border-color)', borderRadius: '10px', padding: '16px' }}>
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <Layers size={18} color="var(--accent-amber)" />
        <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>BPE Merge History</h3>
      </div>
      <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
        {tokenizerState.merges.length} merges on top of {tokenizerState.vocab.size - tokenizerState.merges.length} base characters
        = {tokenizerState.vocab.size} tokens
      </span>
    </div>

    <div style={{ maxHeight: '280px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px', paddingRight: '4px' }}>
      {tokenizerState.merges.length === 0 ? (
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No merges: the vocabulary is just single characters.</p>
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
              <span className="font-mono" style={{ color: 'var(--accent-cyan)', whiteSpace: 'pre' }}>"{merge.pair[0]}"</span>
              <span style={{ color: 'var(--text-dim)' }}>+</span>
              <span className="font-mono" style={{ color: 'var(--accent-cyan)', whiteSpace: 'pre' }}>"{merge.pair[1]}"</span>
              <span style={{ color: 'var(--text-dim)' }}>➔</span>
              <span className="font-mono" style={{ color: '#ffffff', fontWeight: 700, background: 'rgba(99, 102, 241, 0.3)', padding: '2px 6px', borderRadius: '4px', whiteSpace: 'pre' }}>
                "{merge.newToken}"
              </span>
            </div>
            <span className="badge badge-amber" style={{ fontSize: '0.7rem' }}>
              Seen {merge.freq}×
            </span>
          </div>
        ))
      )}
    </div>
  </div>
);
