import React, { useState } from 'react';
import { DatasetOption, TransformerConfig } from '../types';
import { SAMPLE_DATASETS } from '../engine/datasets';
import { BPETokenizer } from '../engine/bpeTokenizer';
import { BookOpen, FileText, Activity, Hash, Plus, CheckCircle, Edit3 } from 'lucide-react';
import { InfoTooltip } from './InfoTooltip';

interface DatasetExplorerProps {
  selectedDataset: DatasetOption;
  onSelectDataset: (ds: DatasetOption) => void;
  tokenizer: BPETokenizer;
  config: TransformerConfig;
  onNavigateToTraining: () => void;
}

export const DatasetExplorer: React.FC<DatasetExplorerProps> = ({
  selectedDataset,
  onSelectDataset,
  tokenizer,
  config,
  onNavigateToTraining,
}) => {
  const [datasetsList, setDatasetsList] = useState<DatasetOption[]>(SAMPLE_DATASETS);
  const [activeDataset, setActiveDataset] = useState<DatasetOption>(selectedDataset);
  
  // Custom dataset creation state
  const [isCreatingCustom, setIsCreatingCustom] = useState<boolean>(false);
  const [customTitle, setCustomTitle] = useState<string>('My Custom Dataset');
  const [customCategory, setCustomCategory] = useState<'literature' | 'code' | 'synthetic' | 'logic'>('synthetic');
  const [customText, setCustomText] = useState<string>(
    'Artificial Intelligence and Machine Learning models predict tokens based on probability distributions learned during pre-training.'
  );

  const encoded = tokenizer.encode(activeDataset.text);
  const charCount = activeDataset.text.length;
  const wordCount = activeDataset.text.trim().split(/\s+/).length;
  const tokenCount = encoded.tokens.length;
  const compressionRatio = (charCount / Math.max(tokenCount, 1)).toFixed(2);

  const handleChooseDataset = (ds: DatasetOption) => {
    setActiveDataset(ds);
    onSelectDataset(ds);
  };

  const handleSaveCustomDataset = () => {
    if (!customTitle.trim() || !customText.trim()) return;

    const newDs: DatasetOption = {
      id: 'custom-' + Date.now(),
      name: customTitle.trim(),
      category: customCategory,
      description: 'User-provided custom text dataset (' + customText.length + ' chars)',
      text: customText
    };

    setDatasetsList(prev => [...prev, newDs]);
    setActiveDataset(newDs);
    onSelectDataset(newDs);
    setIsCreatingCustom(false);
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '24px' }}>
      {/* Left Column: Dataset List & Stats */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Dataset Selection Cards */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <BookOpen size={20} color="var(--accent-purple)" />
              <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Datasets</h2>
              <InfoTooltip
                title="Dataset Repository"
                description="Choose built-in micro-datasets or paste any custom text dataset of your choice."
                impact="Selecting a dataset loads its text into the BPE tokenizer and training loop."
              />
            </div>

            <button
              className="btn-secondary"
              onClick={() => setIsCreatingCustom(!isCreatingCustom)}
              style={{ padding: '6px 12px', fontSize: '0.8rem' }}
            >
              <Plus size={14} /> Add Custom Text
            </button>
          </div>

          {/* Inline Custom Dataset Creator */}
          {isCreatingCustom && (
            <div style={{ background: 'rgba(15, 23, 42, 0.8)', padding: '16px', borderRadius: '10px', border: '1px solid var(--primary)', marginBottom: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <h4 style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>Create Custom Dataset</h4>
              
              <div>
                <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Dataset Title</label>
                <input
                  type="text"
                  value={customTitle}
                  onChange={(e) => setCustomTitle(e.target.value)}
                  style={{ width: '100%', marginTop: '4px', fontSize: '0.85rem' }}
                />
              </div>

              <div>
                <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Paste Text Content</label>
                <textarea
                  rows={4}
                  value={customText}
                  onChange={(e) => setCustomText(e.target.value)}
                  style={{ width: '100%', marginTop: '4px', fontSize: '0.82rem' }}
                />
              </div>

              <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                <button className="btn-secondary" onClick={() => setIsCreatingCustom(false)} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
                  Cancel
                </button>
                <button className="btn-primary" onClick={handleSaveCustomDataset} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
                  Save Dataset
                </button>
              </div>
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {datasetsList.map((ds) => {
              const isCurrent = activeDataset.id === ds.id;
              return (
                <div
                  key={ds.id}
                  onClick={() => handleChooseDataset(ds)}
                  style={{
                    padding: '16px',
                    borderRadius: '10px',
                    border: '1px solid ' + (isCurrent ? 'var(--primary)' : 'var(--border-color)'),
                    background: isCurrent ? 'rgba(99, 102, 241, 0.18)' : 'rgba(15, 23, 42, 0.5)',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    boxShadow: isCurrent ? '0 4px 16px rgba(99, 102, 241, 0.3)' : 'none'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                    <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: isCurrent ? '#ffffff' : 'var(--text-main)' }}>
                      {ds.name}
                    </h3>
                    <span className="badge badge-purple" style={{ fontSize: '0.68rem' }}>
                      {ds.category}
                    </span>
                  </div>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    {ds.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Dataset Corpus Metadata */}
        <div className="glass-panel" style={{ padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
            <Hash size={20} color="var(--accent-cyan)" />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Corpus Statistics</h3>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Characters</p>
              <p className="font-mono" style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--accent-cyan)' }}>
                {charCount.toLocaleString()}
              </p>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Words</p>
              <p className="font-mono" style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--accent-emerald)' }}>
                {wordCount.toLocaleString()}
              </p>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>BPE Tokens</p>
              <p className="font-mono" style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--accent-amber)' }}>
                {tokenCount.toLocaleString()}
              </p>
            </div>

            <div style={{ background: 'rgba(15, 23, 42, 0.6)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Compression Ratio</p>
              <p className="font-mono" style={{ fontSize: '1.3rem', fontWeight: 700, color: 'var(--accent-purple)' }}>
                {compressionRatio}x
              </p>
            </div>
          </div>

          <button
            className="btn-primary"
            onClick={onNavigateToTraining}
            style={{ width: '100%', marginTop: '20px', justifyContent: 'center' }}
          >
            <Activity size={16} /> Train Model On This Dataset
          </button>
        </div>
      </div>

      {/* Right Column: Full Raw Text Viewer */}
      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <FileText size={20} color="var(--accent-emerald)" />
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>
              Raw Dataset Viewer: {activeDataset.name}
            </h2>
          </div>
          <span className="badge badge-emerald" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <CheckCircle size={12} /> Active for Training
          </span>
        </div>

        {/* Text Container with Line Numbers */}
        <div
          style={{
            flex: 1,
            minHeight: '380px',
            background: 'rgba(9, 13, 22, 0.9)',
            border: '1px solid var(--border-color)',
            borderRadius: '10px',
            padding: '16px',
            overflowY: 'auto',
            display: 'flex',
            gap: '16px',
            boxShadow: 'inset 0 2px 10px rgba(0, 0, 0, 0.5)'
          }}
        >
          {/* Line numbers column */}
          <div style={{ userSelect: 'none', color: 'var(--text-dim)', textAlign: 'right', fontSize: '0.85rem' }} className="font-mono">
            {activeDataset.text.split('\n').map((_, idx) => (
              <div key={idx}>{idx + 1}</div>
            ))}
          </div>

          {/* Text content column */}
          <pre
            className="font-mono"
            style={{
              margin: 0,
              fontSize: '0.88rem',
              color: 'var(--text-main)',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              lineHeight: 1.5,
              flex: 1
            }}
          >
            {activeDataset.text}
          </pre>
        </div>
      </div>
    </div>
  );
};
