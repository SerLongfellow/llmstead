import React, { useState, useMemo, useEffect } from 'react';
import { DatasetOption, StepInspectionData, TransformerConfig } from './types';
import { MicroTransformer } from './engine/transformer';
import { BPETokenizer } from './engine/bpeTokenizer';
import { SAMPLE_DATASETS, samplePromptFor } from './engine/datasets';
import { Navbar } from './components/Navbar';
import { SetupView } from './components/SetupView';
import { TrainingDashboard } from './components/TrainingDashboard';
import { PipelineView } from './components/PipelineView';
import { StartView } from './components/StartView';
import { GuideStrip } from './components/GuideStrip';
import { WhatsNextView } from './components/WhatsNextView';
import { Github } from 'lucide-react';

// Other places to learn how transformers work, linked from the footer
const RESOURCES = [
  {
    name: 'Transformer Explainer',
    url: 'https://poloclub.github.io/transformer-explainer/',
    description: 'Interactive visualization of GPT-2 running live in your browser (Georgia Tech Polo Club)',
  },
  {
    name: 'Transformers from Scratch',
    url: 'https://brandonrohrer.com/transformers.html',
    description: 'A step-by-step written walkthrough of how transformers work, by Brandon Rohrer',
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('start');
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

  const [testSentence, setTestSentence] = useState<string>(() => samplePromptFor(SAMPLE_DATASETS[0]));

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
    setTestSentence(samplePromptFor(ds));
  };

  const handleAddDataset = (ds: DatasetOption) => {
    setDatasets(prev => [...prev, ds]);
    handleSelectDataset(ds);
  };

  const paramCount = model.getParameterCount();

  // Step guides at the top of each step; hiding them is remembered in this browser
  const [guidesHidden, setGuidesHidden] = useState<boolean>(() => {
    try {
      return localStorage.getItem('llmstead.hideGuides') === '1';
    } catch {
      return false;
    }
  });
  const setGuides = (hidden: boolean) => {
    setGuidesHidden(hidden);
    try {
      localStorage.setItem('llmstead.hideGuides', hidden ? '1' : '0');
    } catch {
      /* storage unavailable: the choice just won't persist */
    }
  };

  const [isTraining, setIsTraining] = useState<boolean>(false);

  return (
    <div style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 20px 40px 20px' }}>
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isTraining={isTraining}
      />

      <main style={{ minHeight: '80vh' }}>
        {activeTab === 'start' && (
          <StartView onNavigate={setActiveTab} guidesHidden={guidesHidden} onShowGuides={() => setGuides(false)} />
        )}

        {activeTab === 'setup' && (
          <>
            {!guidesHidden && (
              <GuideStrip step={1} title="Set up your model" next={{ label: 'Next: Train', onClick: () => setActiveTab('training') }} onHide={() => setGuides(true)}>
                Choose a dataset, then adjust the tokenizer and model size if you like. The defaults train well. Changing any setting
                here resets the model and starts training over.
              </GuideStrip>
            )}
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
          </>
        )}

        {/* Kept mounted (just hidden) so the loss chart survives tab switches and training
            keeps running in the background while you look inside the model */}
        <div style={{ display: activeTab === 'training' ? 'block' : 'none' }}>
          {!guidesHidden && (
            <GuideStrip step={2} title="Train it" next={{ label: 'Next: Look inside', onClick: () => setActiveTab('pipeline') }} onHide={() => setGuides(true)}>
              Press Start and watch the loss fall. Training keeps going if you switch tabs. When the curve levels off, run the
              benchmark, then look inside to see what changed.
            </GuideStrip>
          )}
          <TrainingDashboard
            model={model}
            tokenizer={tokenizer}
            config={effectiveConfig}
            selectedDataset={selectedDataset}
            onNavigateToSetup={() => setActiveTab('setup')}
            onTrainingChange={setIsTraining}
            onChangeLearningRate={(learningRate) => setConfig(prev => ({ ...prev, learningRate }))}
            onChangeOptimizer={(optimizer) => setConfig(prev => ({ ...prev, optimizer }))}
            visible={activeTab === 'training'}
          />
        </div>

        {activeTab === 'pipeline' && (
          <>
            {!guidesHidden && (
              <GuideStrip step={3} title="Look inside" next={{ label: "Next: What's next", onClick: () => setActiveTab('next') }} onHide={() => setGuides(true)}>
                Type a prompt, click a stage to see what happens to it there, and click a token to follow it through the model.
                Compare the same prompt before and after more training.
              </GuideStrip>
            )}
            <PipelineView
              inspectionData={inspectionData}
              config={effectiveConfig}
              tokenizer={tokenizer}
              testInput={testSentence}
              setTestInput={setTestSentence}
              onRunInspect={runInspection}
              tokenizerState={tokenizerState}
              model={model}
            />
          </>
        )}

        {activeTab === 'next' && <WhatsNextView onNavigate={setActiveTab} />}
      </main>

      <footer style={{ marginTop: '40px', textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-dim)', borderTop: '1px solid var(--border-color)', paddingTop: '20px' }}>
        <p>LLMStead • Raise your own models: a tiny transformer, built and trained from scratch in your browser</p>
        <p style={{ marginTop: '8px', display: 'flex', justifyContent: 'center', flexWrap: 'wrap', gap: '6px 14px' }}>
          <span>Additional resources:</span>
          {RESOURCES.map(r => (
            <a key={r.url} href={r.url} target="_blank" rel="noopener noreferrer" title={r.description} style={{ color: 'var(--text-muted)' }}>
              {r.name}
            </a>
          ))}
        </p>
        <p style={{ marginTop: '8px' }}>
          <a
            href="https://github.com/SerLongfellow/llmstead"
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: 'var(--text-muted)', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '6px' }}
          >
            <Github size={14} /> View the source on GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}
