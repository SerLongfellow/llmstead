import React from 'react';
import { TransformerConfig } from '../types';
import { Sliders, Info, HelpCircle } from 'lucide-react';
import { InfoTooltip } from './InfoTooltip';

interface ParameterTunerProps {
  config: TransformerConfig;
  onChangeConfig: (newConfig: TransformerConfig) => void;
  paramCount: number;
}

export const ParameterTuner: React.FC<ParameterTunerProps> = ({
  config,
  onChangeConfig,
  paramCount,
}) => {
  const updateField = (key: keyof TransformerConfig, value: number | string) => {
    onChangeConfig({
      ...config,
      [key]: value,
    });
  };

  const MAX_HEADS = 8;
  // Only head counts that divide d_model evenly — otherwise dimensions would be dropped
  const headOptions = (dModel: number) =>
    Array.from({ length: MAX_HEADS }, (_, i) => i + 1).filter(h => dModel % h === 0);

  const updateDModel = (dModel: number) => {
    let numHeads = config.numHeads;
    if (dModel % numHeads !== 0) {
      numHeads = Math.max(...headOptions(dModel).filter(h => h <= numHeads));
    }
    onChangeConfig({ ...config, dModel, numHeads });
  };

  const dMlp = config.dModel * config.mlpRatio;
  const headDim = config.dModel / config.numHeads;

  const tokenEmbedParams = config.vocabSize * config.dModel;
  const posEmbedParams = config.contextWindow * config.dModel;
  const qkvParams = 3 * config.dModel * config.dModel * config.numLayers;
  const outProjParams = config.dModel * config.dModel * config.numLayers;
  const mlpParams = 2 * config.dModel * dMlp * config.numLayers;
  const headParams = config.dModel * config.vocabSize;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
      {/* Hyperparameter Controls */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
          <Sliders size={20} color="var(--primary)" />
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Architecture Hyperparameters</h2>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
          {/* Embedding Dimension */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Embedding Dimension (d_model)</label>
                <InfoTooltip
                  title="Embedding Dimension (d_model)"
                  description="The length of the dense numerical vector assigned to each token. Higher dimensions capture richer semantic relationships."
                  impact="Quadratically increases QKV projection matrix sizes and total parameter count."
                />
              </div>
              <span className="font-mono" style={{ fontSize: '0.85rem', color: 'var(--accent-cyan)' }}>{config.dModel}</span>
            </div>
            <input
              type="range"
              min={8}
              max={128}
              step={8}
              value={config.dModel}
              onChange={(e) => updateDModel(Number(e.target.value))}
            />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>Width of vector representation per token</span>
          </div>

          {/* Context Window */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Context Window Length (Seq Len)</label>
                <InfoTooltip
                  title="Context Window Length (Seq Len)"
                  description="The maximum number of sequence tokens the model can process, attend to, and remember at one time."
                  impact="Quadratically increases attention matrix calculation memory size (N x N)."
                />
              </div>
              <span className="font-mono" style={{ fontSize: '0.85rem', color: 'var(--accent-emerald)' }}>{config.contextWindow}</span>
            </div>
            <input
              type="range"
              min={8}
              max={64}
              step={8}
              value={config.contextWindow}
              onChange={(e) => updateField('contextWindow', Number(e.target.value))}
            />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>Maximum tokens processed simultaneously</span>
          </div>

          {/* Attention Heads */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Attention Heads (H)</label>
                <InfoTooltip
                  title="Multi-Head Attention (H)"
                  description="Splits d_model into H parallel heads. Head 1 might focus on grammar while Head 2 focuses on subject-verb relationships."
                  impact="Splits head dimension (d_k = d_model / H) without increasing total parameter count!"
                />
              </div>
              <span className="font-mono" style={{ fontSize: '0.85rem', color: 'var(--accent-amber)' }}>
                {config.numHeads} heads ({headDim}d each)
              </span>
            </div>
            <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
              {headOptions(config.dModel).map(h => (
                <button
                  key={h}
                  className={h === config.numHeads ? 'btn-primary' : 'btn-secondary'}
                  style={{ padding: '4px 12px', minWidth: '40px' }}
                  onClick={() => updateField('numHeads', h)}
                >
                  {h}
                </button>
              ))}
            </div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
              Only divisors of d_model ({config.dModel}) are allowed, so every head gets an equal slice
            </span>
          </div>

          {/* Transformer Layers */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Transformer Blocks (Layers L)</label>
                <InfoTooltip
                  title="Transformer Stack Depth (L)"
                  description="The number of stacked attention + MLP blocks. Deeper layers form abstract reasoning and higher-level concepts."
                  impact="Linearly multiplies total parameter count and computation time per pass."
                />
              </div>
              <span className="font-mono" style={{ fontSize: '0.85rem', color: 'var(--accent-purple)' }}>{config.numLayers}</span>
            </div>
            <input
              type="range"
              min={1}
              max={4}
              step={1}
              value={config.numLayers}
              onChange={(e) => updateField('numLayers', Number(e.target.value))}
            />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>Stack depth of attention + MLP blocks</span>
          </div>

          {/* MLP Ratio */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>MLP Expansion Ratio</label>
                <InfoTooltip
                  title="MLP Expansion Ratio"
                  description="The width multiplier for the Feed-Forward Neural Network layer inside each transformer block (standard is 4x)."
                  impact="Provides non-linear capacity (GELU) for storing factual memories and associations."
                />
              </div>
              <span className="font-mono" style={{ fontSize: '0.85rem', color: 'var(--primary)' }}>{config.mlpRatio}x ({dMlp} hidden)</span>
            </div>
            <input
              type="range"
              min={2}
              max={4}
              step={1}
              value={config.mlpRatio}
              onChange={(e) => updateField('mlpRatio', Number(e.target.value))}
            />
          </div>

          {/* Learning Rate */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Learning Rate</label>
                <InfoTooltip
                  title="Optimizer Learning Rate"
                  description="The step size used when updating weight matrices during backpropagation."
                  impact="Too high causes training divergence/loss explosion; too low makes learning very slow."
                />
              </div>
              <span className="font-mono" style={{ fontSize: '0.85rem', color: 'var(--accent-rose)' }}>{config.learningRate}</span>
            </div>
            <input
              type="range"
              min={0.001}
              max={0.05}
              step={0.001}
              value={config.learningRate}
              onChange={(e) => updateField('learningRate', Number(e.target.value))}
            />
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>Safe to change mid-training — weights are kept</span>
          </div>

          {/* Optimizer */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Optimizer</label>
              <InfoTooltip
                title="Optimizer"
                description="SGD steps every weight by lr × gradient. AdamW keeps running averages of each weight's gradient (momentum) and squared gradient, giving every weight its own adaptive step size, and applies weight decay separately from the gradient."
                impact="AdamW usually converges much faster on transformers. Switching resets AdamW's running averages but keeps the learned weights."
              />
            </div>
            <div style={{ display: 'flex', gap: '6px' }}>
              {(['adamw', 'sgd'] as const).map(opt => (
                <button
                  key={opt}
                  className={config.optimizer === opt ? 'btn-primary' : 'btn-secondary'}
                  style={{ padding: '4px 14px' }}
                  onClick={() => updateField('optimizer', opt)}
                >
                  {opt === 'adamw' ? 'AdamW' : 'SGD'}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Parameter Count & Tensor Layout Breakdown */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            <Info size={20} color="var(--accent-cyan)" />
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Parameter Breakdown</h2>
          </div>

          {/* Big Total Parameter Card */}
          <div style={{
            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.2) 100%)',
            border: '1px solid rgba(99, 102, 241, 0.4)',
            borderRadius: '12px',
            padding: '20px',
            textAlign: 'center',
            marginBottom: '20px'
          }}>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Total Trainable Weights</p>
            <p className="font-mono" style={{ fontSize: '2.2rem', fontWeight: 800, color: '#ffffff', margin: '4px 0' }}>
              {paramCount.toLocaleString()}
            </p>
            <p style={{ fontSize: '0.75rem', color: 'var(--accent-cyan)' }}>
              Micro-Transformer Architecture (Nano Scale)
            </p>
          </div>

          {/* Table Breakdown */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: 'rgba(15, 23, 42, 0.5)', borderRadius: '6px', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Token Embeddings (V × d_model)</span>
              <span className="font-mono" style={{ fontWeight: 600 }}>{tokenEmbedParams.toLocaleString()}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: 'rgba(15, 23, 42, 0.5)', borderRadius: '6px', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Positional Embeddings (N × d_model)</span>
              <span className="font-mono" style={{ fontWeight: 600 }}>{posEmbedParams.toLocaleString()}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: 'rgba(15, 23, 42, 0.5)', borderRadius: '6px', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Q, K, V Projections (3 × d_model² × L)</span>
              <span className="font-mono" style={{ fontWeight: 600 }}>{qkvParams.toLocaleString()}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: 'rgba(15, 23, 42, 0.5)', borderRadius: '6px', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Output Attention Proj (d_model² × L)</span>
              <span className="font-mono" style={{ fontWeight: 600 }}>{outProjParams.toLocaleString()}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: 'rgba(15, 23, 42, 0.5)', borderRadius: '6px', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Feed-Forward MLP (2 × d_model × d_mlp × L)</span>
              <span className="font-mono" style={{ fontWeight: 600 }}>{mlpParams.toLocaleString()}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: 'rgba(15, 23, 42, 0.5)', borderRadius: '6px', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-muted)' }}>Output Unembedding Head (d_model × V)</span>
              <span className="font-mono" style={{ fontWeight: 600 }}>{headParams.toLocaleString()}</span>
            </div>
          </div>
        </div>

        <div style={{ marginTop: '20px', padding: '12px', background: 'rgba(99, 102, 241, 0.08)', border: '1px solid rgba(99, 102, 241, 0.2)', borderRadius: '8px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          <p>💡 <strong>Did you know?</strong> Hover over any <HelpCircle size={12} style={{ display: 'inline' }} /> icon next to a hyperparameter to reveal detailed explanations of what it does!</p>
        </div>
      </div>
    </div>
  );
};
