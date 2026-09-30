import React, { useState, useMemo, useEffect } from 'react';
import { DatasetOption, StepInspectionData, TransformerConfig } from './types';
import { MicroTransformer } from './engine/transformer';
import { BPETokenizer } from './engine/bpeTokenizer';
import { SAMPLE_DATASETS } from './engine/datasets';
import { Navbar } from './components/Navbar';
import { ParameterTuner } from './components/ParameterTuner';
import { TokenizerVisualizer } from './components/TokenizerVisualizer';
import { DatasetExplorer } from './components/DatasetExplorer';
import { ForwardPassInspector } from './components/ForwardPassInspector';
import { AttentionHeatmap } from './components/AttentionHeatmap';
import { TrainingDashboard } from './components/TrainingDashboard';

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('tuner');
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

  // Which text the tokenizer is trained on. Defaults to the selected dataset; the BPE tab
  // can override it with a custom corpus. config.vocabSize is the *target* vocab size.
  const [tokenizerCorpus, setTokenizerCorpus] = useState<string>(SAMPLE_DATASETS[0].text);

  // BPE Tokenizer Engine instance — rebuilt (never mutated) when its inputs change
  const tokenizer = useMemo(() => {
    const t = new BPETokenizer();
    t.train(tokenizerCorpus, config.vocabSize);
    return t;
  }, [tokenizerCorpus, config.vocabSize]);

  const tokenizerState = useMemo(() => tokenizer.getState(tokenizerCorpus), [tokenizer, tokenizerCorpus]);

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
    setTokenizerCorpus(ds.text);
    setTestSentence(ds.text.slice(0, 30));
  };

  // Retrain tokenizer on a custom corpus: update state and let the memo rebuild it
  const handleRetrainTokenizer = (corpus: string, targetVocabSize: number) => {
    setTokenizerCorpus(corpus);
    setConfig(prev => ({ ...prev, vocabSize: targetVocabSize }));
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
        {activeTab === 'tuner' && (
          <ParameterTuner
            config={effectiveConfig}
            // Tuner sees the effective vocab; keep the tokenizer's target vocab untouched
            onChangeConfig={(next) => setConfig({ ...next, vocabSize: config.vocabSize })}
            paramCount={paramCount}
          />
        )}

        {activeTab === 'bpe' && (
          <TokenizerVisualizer
            tokenizer={tokenizer}
            tokenizerState={tokenizerState}
            targetVocabSize={config.vocabSize}
            isCustomCorpus={tokenizerCorpus !== selectedDataset.text}
            onResetToDataset={() => setTokenizerCorpus(selectedDataset.text)}
            onRetrainTokenizer={handleRetrainTokenizer}
          />
        )}

        {activeTab === 'datasets' && (
          <DatasetExplorer
            selectedDataset={selectedDataset}
            onSelectDataset={handleSelectDataset}
            tokenizer={tokenizer}
            config={effectiveConfig}
            onNavigateToTraining={() => setActiveTab('training')}
          />
        )}

        {activeTab === 'inspector' && (
          <ForwardPassInspector
            inspectionData={inspectionData}
            config={effectiveConfig}
            testInput={testSentence}
            setTestInput={setTestSentence}
            onRunInspect={runInspection}
          />
        )}

        {activeTab === 'attention' && (
          <AttentionHeatmap
            inspectionData={inspectionData}
            config={effectiveConfig}
          />
        )}

        {activeTab === 'training' && (
          <TrainingDashboard
            model={model}
            tokenizer={tokenizer}
            config={effectiveConfig}
            selectedDataset={selectedDataset}
            onSelectDataset={handleSelectDataset}
            onUpdateInspection={setInspectionData}
          />
        )}
      </main>

      <footer style={{ marginTop: '40px', textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-dim)', borderTop: '1px solid var(--border-color)', paddingTop: '20px' }}>
        <p>LLM Breakdown • Visualizing Small Language Models & Attention Mechanics from Scratch</p>
      </footer>
    </div>
  );
}
