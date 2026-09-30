import React, { useState } from 'react';
import { StepInspectionData, TransformerConfig } from '../types';
import { Layers, ArrowRight, Table, Sparkles, Activity } from 'lucide-react';

interface ForwardPassInspectorProps {
  inspectionData: StepInspectionData;
  config: TransformerConfig;
  testInput: string;
  setTestInput: (str: string) => void;
  onRunInspect: () => void;
}

export const ForwardPassInspector: React.FC<ForwardPassInspectorProps> = ({
  inspectionData,
  config,
  testInput,
  setTestInput,
  onRunInspect,
}) => {
  const [selectedLayerIndex, setSelectedLayerIndex] = useState<number>(0);

  const selectedLayer = inspectionData.layerInspections[selectedLayerIndex] || inspectionData.layerInspections[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Input Control Box */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
          <div style={{ flex: 1, minWidth: '280px' }}>
            <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>Input Prompt Sentence</label>
            <div style={{ display: 'flex', gap: '10px', marginTop: '6px' }}>
              <input
                type="text"
                value={testInput}
                onChange={(e) => setTestInput(e.target.value)}
                style={{ flex: 1, fontSize: '0.95rem' }}
              />
              <button className="btn-primary" onClick={onRunInspect}>
                <Sparkles size={16} /> Run Forward Pass
              </button>
            </div>
          </div>

          {/* Layer Tab selector */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginRight: '6px' }}>Inspect Layer:</span>
            {inspectionData.layerInspections.map((l, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedLayerIndex(idx)}
                style={{
                  padding: '6px 14px',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  background: selectedLayerIndex === idx ? 'var(--primary)' : 'rgba(15, 23, 42, 0.6)',
                  color: selectedLayerIndex === idx ? '#ffffff' : 'var(--text-muted)'
                }}
              >
                Layer {idx + 1}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Step 1: Token & Positional Embedding Lookup */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
          <Table size={20} color="var(--accent-cyan)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>
            1. Token + Positional Embedding Addition: <code className="font-mono" style={{ color: 'var(--primary)' }}>E = E_token + E_pos</code>
          </h3>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
                <th style={{ padding: '10px' }}>Pos</th>
                <th style={{ padding: '10px' }}>Token</th>
                <th style={{ padding: '10px' }}>ID</th>
                <th style={{ padding: '10px' }}>Token Embed Vector (E_token)</th>
                <th style={{ padding: '10px' }}>Pos Embed Vector (E_pos)</th>
                <th style={{ padding: '10px' }}>Combined Input (E)</th>
              </tr>
            </thead>
            <tbody>
              {inspectionData.tokens.map((tokId, idx) => {
                const str = inspectionData.tokenStrings[idx];
                const tokVec = inspectionData.tokenEmbeddings[idx] || [];
                const posVec = inspectionData.positionEmbeddings[idx] || [];
                const combVec = inspectionData.combinedEmbeddings[idx] || [];

                return (
                  <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <td style={{ padding: '10px' }} className="font-mono">{idx}</td>
                    <td style={{ padding: '10px' }}>
                      <span className="font-mono" style={{ fontWeight: 700, color: 'var(--accent-emerald)', background: 'rgba(16, 185, 129, 0.15)', padding: '2px 8px', borderRadius: '4px' }}>
                        "{str}"
                      </span>
                    </td>
                    <td style={{ padding: '10px', color: 'var(--text-muted)' }} className="font-mono">{tokId}</td>
                    <td style={{ padding: '10px', fontSize: '0.75rem', color: 'var(--accent-cyan)' }} className="font-mono">
                      [{tokVec.slice(0, 4).map(v => v.toFixed(3)).join(', ')}...]
                    </td>
                    <td style={{ padding: '10px', fontSize: '0.75rem', color: 'var(--accent-amber)' }} className="font-mono">
                      [{posVec.slice(0, 4).map(v => v.toFixed(3)).join(', ')}...]
                    </td>
                    <td style={{ padding: '10px', fontSize: '0.75rem', color: '#ffffff', fontWeight: 600 }} className="font-mono">
                      [{combVec.slice(0, 4).map(v => v.toFixed(3)).join(', ')}...]
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Step 2: Layer Q, K, V Projections & Multi-Head Outputs */}
      {selectedLayer && (
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
            <Layers size={20} color="var(--accent-purple)" />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>
              2. Layer {selectedLayerIndex + 1}: Query, Key, Value Tensors & Attention Concat
            </h3>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
            {/* Queries */}
            <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <p style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-cyan)', marginBottom: '8px' }}>
                Query Matrix Q = X · W_Q
              </p>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }} className="font-mono">
                {selectedLayer.queries[0]?.slice(0, 4).map((qHead, tIdx) => (
                  <p key={tIdx}>Token {tIdx} (Head 1): [{qHead.map(v => v.toFixed(2)).join(', ')}]</p>
                ))}
              </div>
            </div>

            {/* Keys */}
            <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <p style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-emerald)', marginBottom: '8px' }}>
                Key Matrix K = X · W_K
              </p>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }} className="font-mono">
                {selectedLayer.keys[0]?.slice(0, 4).map((kHead, tIdx) => (
                  <p key={tIdx}>Token {tIdx} (Head 1): [{kHead.map(v => v.toFixed(2)).join(', ')}]</p>
                ))}
              </div>
            </div>

            {/* Values */}
            <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '16px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <p style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-amber)', marginBottom: '8px' }}>
                Value Matrix V = X · W_V
              </p>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }} className="font-mono">
                {selectedLayer.values[0]?.slice(0, 4).map((vHead, tIdx) => (
                  <p key={tIdx}>Token {tIdx} (Head 1): [{vHead.map(v => v.toFixed(2)).join(', ')}]</p>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Step 3: Final Softmax Probabilities Distribution */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
          <Activity size={20} color="var(--primary)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>
            3. Final Logits & Softmax Token Output Distribution
          </h3>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-color)', textAlign: 'left', color: 'var(--text-muted)' }}>
                <th style={{ padding: '10px' }}>Input Context Token</th>
                <th style={{ padding: '10px' }}>Top Predicted Next Token</th>
                <th style={{ padding: '10px' }}>Top Probability</th>
                <th style={{ padding: '10px' }}>Logit Vector (First 4 entries)</th>
              </tr>
            </thead>
            <tbody>
              {inspectionData.tokens.map((tokId, idx) => {
                const probs = inspectionData.probabilities[idx] || [];
                const logits = inspectionData.logits[idx] || [];

                // Find top probability
                let maxP = 0;
                let topId = 0;
                probs.forEach((p, pIdx) => {
                  if (p > maxP) {
                    maxP = p;
                    topId = pIdx;
                  }
                });

                return (
                  <tr key={idx} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                    <td style={{ padding: '10px' }}>
                      <span className="font-mono" style={{ fontWeight: 700, color: 'var(--accent-cyan)' }}>
                        "{inspectionData.tokenStrings[idx]}"
                      </span>
                    </td>
                    <td style={{ padding: '10px' }}>
                      <span className="font-mono" style={{ fontWeight: 700, color: 'var(--accent-emerald)', background: 'rgba(16, 185, 129, 0.15)', padding: '2px 8px', borderRadius: '4px' }}>
                        Token #{topId}
                      </span>
                    </td>
                    <td style={{ padding: '10px', fontWeight: 700, color: '#ffffff' }} className="font-mono">
                      {(maxP * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '10px', fontSize: '0.75rem', color: 'var(--text-muted)' }} className="font-mono">
                      [{logits.slice(0, 4).map(v => v.toFixed(2)).join(', ')}...]
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
