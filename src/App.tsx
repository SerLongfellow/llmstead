import React, { useState, useMemo, useEffect } from 'react';
import { DatasetOption, StepInspectionData, TransformerConfig } from './types';
import { MicroTransformer } from './engine/transformer';
import { BPETokenizer } from './engine/bpeTokenizer';
import { SAMPLE_DATASETS } from './engine/datasets';
import { Navbar } from './components/Navbar';
import { SetupView } from './components/SetupView';
import { TrainingDashboard } from './components/TrainingDashboard';
import { PipelineView } from './components/PipelineView';

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('setup');
  // Built-in datasets plus any custom text added in Setup (kept here so it survives tab switches)
  const [datasets, setDatasets] = useState<DatasetOption[]>(SAMPLE_DATASETS);
  const [selectedDataset, setSelectedDataset] = useState<DatasetOption>(SAMPLE_DATASETS[0]);
  
  // Transformer Hyperparameters
  const [config, setConfig] = useState<TransformerConfig>({
    vocabSize: 120,
    contextWindow: 16,
    dModel: 32,
    numHeads: 2,
    numLayers: 2,
    mlpRatio: 4,
    learningRate: 0.001,
    optimizer: 'adamw'
  });

  // BPE Tokenizer Engine instance, trained on the selected dataset. config.vocabSize is the
  // *target* vocab size (set in Setup). Rebuilt (never mutated) when either changes, and
  // since the model's embedding table depends on the vocabulary, that also means a new model.
  const tokenizer = useMemo(() => {
    const t = new BPETokenizer();
    t.train(selectedDataset.text, config.vocabSize);
    return t;
  }, [selectedDataset, config.vocabSize]);

  const tokenizerState = useMemo(() => tokenizer.getState(selectedDataset.text), [tokenizer, selectedDataset]);

  // BPE can stop early (no pair repeats), so the real vocab may be smaller than the target
  const actualVocab = tokenizer.getVocabSize();
  const effectiveConfig = useMemo<TransformerConfig>(
    () => ({ ...config, vocabSize: actualVocab }),
    [config, actualVocab]
  );

  // Micro-Transformer Engine instance. Depends ONLY on shape-changing settings, so
  // tweaking learning rate or optimizer never throws away trained weights.
  const { contextWindow, dModel, numHeads, numLayers, mlpRatio } = config;
  const model = useMemo(
    () => new MicroTransformer({ ...config, vocabSize: actualVocab }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [actualVocab, contextWindow, dModel, numHeads, numLayers, mlpRatio, tokenizer]
  );

  // Optimizer is swappable in place (resets Adam moments, keeps weights)
  useEffect(() => {
    model.setOptimizer(config.optimizer);
  }, [model, config.optimizer]);

  const [testSentence, setTestSentence] = useState<string>('FIRST CITIZEN: Hear me speak.');

  // Step Inspection State
  const [inspectionData, setInspectionData] = useState<StepInspectionData>(() => {
    const encoded = tokenizer.encode(testSentence);
    return model.inspectForwardPass(encoded.tokens, encoded.tokenStrings);
  });

  // Re-run inspection whenever testSentence, model, or tokenizer changes
  const runInspection = () => {
    const encoded = tokenizer.encode(testSentence);
    const data = model.inspectForwardPass(encoded.tokens, encoded.tokenStrings);
    setInspectionData(data);
  };

  useEffect(() => {
    runInspection();
  }, [testSentence, model, tokenizer]);

  const handleSelectDataset = (ds: DatasetOption) => {
    setSelectedDataset(ds);
    setTestSentence(ds.text.slice(0, 30));
  };

  const handleAddDataset = (ds: DatasetOption) => {
    setDatasets(prev => [...prev, ds]);
    handleSelectDataset(ds);
  };

  const paramCount = model.getParameterCount();

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 20px 40px 20px' }}>
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        paramCount={paramCount}
        datasetName={selectedDataset.name}
      />

      <main style={{ minHeight: '80vh' }}>
        {activeTab === 'setup' && (
          <SetupView
            datasets={datasets}
            selectedDataset={selectedDataset}
            onSelectDataset={handleSelectDataset}
            onAddDataset={handleAddDataset}
            tokenizer={tokenizer}
            targetVocabSize={config.vocabSize}
            onChangeVocabSize={(vocabSize) => setConfig(prev => ({ ...prev, vocabSize }))}
            config={effectiveConfig}
            // Setup sees the effective vocab; the target vocab only changes via its own slider
            onChangeConfig={(next) => setConfig({ ...next, vocabSize: config.vocabSize })}
            paramCount={paramCount}
          />
        )}

        {activeTab === 'pipeline' && (
          <PipelineView
            inspectionData={inspectionData}
            config={effectiveConfig}
            tokenizer={tokenizer}
            testInput={testSentence}
            setTestInput={setTestSentence}
            onRunInspect={runInspection}
            tokenizerState={tokenizerState}
          />
        )}

        {activeTab === 'training' && (
          <TrainingDashboard
            model={model}
            tokenizer={tokenizer}
            config={effectiveConfig}
            selectedDataset={selectedDataset}
            onNavigateToSetup={() => setActiveTab('setup')}
          />
        )}
      </main>

      <footer style={{ marginTop: '40px', textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-dim)', borderTop: '1px solid var(--border-color)', paddingTop: '20px' }}>
        <p>LLM Breakdown • Visualizing Small Language Models & Attention Mechanics from Scratch</p>
      </footer>
    </div>
  );
}
