import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { SavedSession, SaveStatus, TrainingHistory, loadSession, saveSession, clearSession } from './persistence';
import { DatasetOption, StepInspectionData, TransformerConfig } from './types';
import { MicroTransformer } from './engine/transformer';
import { BPETokenizer } from './engine/bpeTokenizer';
import { SAMPLE_DATASETS, samplePromptFor } from './engine/datasets';
import { ExportModal } from './components/ExportModal';
import { Navbar } from './components/Navbar';
import { SetupView } from './components/SetupView';
import { TrainingDashboard } from './components/TrainingDashboard';
import { PipelineView } from './components/PipelineView';
import { StartView } from './components/StartView';
import { GuideStrip } from './components/GuideStrip';
import { WhatsNextView } from './components/WhatsNextView';
import { PostTrainSession, PostTrainView } from './components/posttrain/PostTrainView';
import { Github, History, X } from 'lucide-react';

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

const DEFAULT_CONFIG: TransformerConfig = {
  vocabSize: 120,
  contextWindow: 16,
  dModel: 32,
  numHeads: 2,
  numLayers: 2,
  mlpRatio: 4,
  learningRate: 0.001,
  optimizer: 'adamw',
};

/** How often to autosave while something has changed (also on pause and when the tab is hidden) */
const AUTOSAVE_MS = 10_000;

/** Longest the first render waits for the web fonts before showing the app anyway */
const FONT_WAIT_MS = 600;

/** Reads the autosave (if any) once, then renders the app starting from it */
export default function App() {
  const [boot, setBoot] = useState<{ saved: SavedSession | null } | null>(null);
  useEffect(() => {
    // Also start the web fonts loading now and wait for them (briefly), so the app doesn't first
    // render in a fallback font and then reflow when Inter arrives. Never wait more than FONT_WAIT_MS.
    const fonts = Promise.race([
      Promise.all(['400 1em Inter', '700 1em Inter', '400 1em "JetBrains Mono"'].map(f => document.fonts?.load(f))).catch(() => {}),
      new Promise(resolve => setTimeout(resolve, FONT_WAIT_MS)),
    ]);
    Promise.all([loadSession(), fonts]).then(([saved]) => setBoot({ saved }));
  }, []);
  // Loading takes a few milliseconds; show just the page background (set in index.html) rather
  // than flash the defaults, then the app fades in (.app-shell)
  if (!boot) return null;
  return <Workbench saved={boot.saved} />;
}


type RestoreNote = { kind: 'restored'; step: number; savedAt: number } | { kind: 'discarded'; reason: 'vocabulary' | 'architecture' };

function Workbench({ saved }: { saved: SavedSession | null }) {
  // Built-in datasets plus any custom text added in Setup (kept here so it survives tab switches)
  const [datasets, setDatasets] = useState<DatasetOption[]>(() => [...SAMPLE_DATASETS, ...(saved?.customDatasets ?? [])]);
  const [selectedDataset, setSelectedDataset] = useState<DatasetOption>(
    () => datasets.find(d => d.id === saved?.selectedDatasetId) ?? SAMPLE_DATASETS[0]
  );

  // Transformer Hyperparameters
  const [config, setConfig] = useState<TransformerConfig>(() => saved?.config ?? DEFAULT_CONFIG);

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

  // Restore the autosaved weights, but only if they still fit: the rebuilt tokenizer must have the
  // same vocabulary (a site update can change a dataset) and every matrix the same shape.
  const restoredModel = useRef<MicroTransformer | null>(null);
  const [restoreNote, setRestoreNote] = useState<RestoreNote | null>(() => {
    if (!saved) return null;
    if (saved.outdatedModel) return saved.history.stepCount > 0 ? { kind: 'discarded', reason: 'architecture' } : null;
    const vocabulary = Array.from({ length: actualVocab }, (_, id) => tokenizer.decode([id]));
    const sameVocab = vocabulary.length === saved.vocabulary.length && vocabulary.every((t, i) => t === saved.vocabulary[i]);
    const m = new MicroTransformer({ ...config, vocabSize: actualVocab });
    if (!sameVocab || !m.importState(saved.model)) return { kind: 'discarded', reason: 'vocabulary' };
    restoredModel.current = m;
    return { kind: 'restored', step: saved.history.stepCount, savedAt: saved.savedAt };
  });
  const initialHistory = restoredModel.current ? saved!.history : null;
  // Coming back to a model you've trained? Open on the Train tab, where you left off
  const [activeTab, setActiveTab] = useState<string>(() => (initialHistory && initialHistory.stepCount > 0 ? 'training' : 'start'));
  // An untrained save (settings only) restores silently; "restored from step #0" says nothing useful
  useEffect(() => {
    if (restoreNote?.kind === 'restored' && restoreNote.step === 0) setRestoreNote(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Micro-Transformer Engine instance. Depends ONLY on shape-changing settings, so
  // tweaking learning rate or optimizer never throws away trained weights.
  // The first one built is the restored model, if there is one.
  const { contextWindow, dModel, numHeads, numLayers, mlpRatio } = config;
  const model = useMemo(
    () => restoredModel.current ?? new MicroTransformer({ ...config, vocabSize: actualVocab }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [actualVocab, contextWindow, dModel, numHeads, numLayers, mlpRatio, tokenizer]
  );
  useEffect(() => {
    restoredModel.current = null; // later settings changes build fresh models
  }, []);

  // Optimizer is swappable in place (resets Adam moments, keeps weights)
  useEffect(() => {
    model.setOptimizer(config.optimizer);
  }, [model, config.optimizer]);

  const [testSentence, setTestSentence] = useState<string>(() => saved?.testSentence ?? samplePromptFor(SAMPLE_DATASETS[0]));

  // Step Inspection State
  const [inspectionData, setInspectionData] = useState<StepInspectionData>(() => {
    const encoded = tokenizer.encode(testSentence);
    return model.inspectForwardPass(encoded.tokens, encoded.tokenStrings);
  });

  // Re-run inspection whenever testSentence, model, or tokenizer changes
  const runInspection = () => {
    // An empty prompt has no tokens to run (the forward pass needs at least one), so fall
    // back to the dataset's sample prompt; it always matches the current model's shape
    let encoded = tokenizer.encode(testSentence);
    if (encoded.tokens.length === 0) encoded = tokenizer.encode(samplePromptFor(selectedDataset));
    if (encoded.tokens.length === 0) return;
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
  const [isPostTraining, setIsPostTraining] = useState<boolean>(false);
  const [exportOpen, setExportOpen] = useState(false);

  // ── Post-training ─────────────────────────────────────────────────────────
  // Starting post-training freezes a copy of the pre-trained weights: the "before" for every
  // comparison, DPO's and RL's reference, and what Restore puts back. A new model ends it.
  const [postTrain, setPostTrain] = useState<PostTrainSession | null>(null);
  useEffect(() => setPostTrain(null), [model]);
  const beginPostTrain = useCallback((): PostTrainSession => {
    const base = model.exportState();
    const reference = new MicroTransformer(model.config);
    reference.importState(base);
    const session = { base, reference };
    setPostTrain(session);
    return session;
  }, [model]);
  const restorePreTrained = () => {
    if (postTrain) model.importState(postTrain.base);
    setPostTrain(null);
  };
  const closeExport = useCallback(() => setExportOpen(false), []);

  // ── Autosave ──────────────────────────────────────────────────────────────
  // Anything that changes the session marks it dirty; it's written every AUTOSAVE_MS, when
  // training pauses, and when the tab is hidden or closed (a write during unload isn't
  // guaranteed to finish, so the periodic save is the safety net).
  const [saveStatus, setSaveStatus] = useState<SaveStatus>(null);
  const history = useRef<TrainingHistory>(initialHistory ?? { stepCount: 0, lossHistory: [], valLossHistory: [], stepLabels: [] });
  const dirty = useRef(false);
  const savingOff = useRef(false);
  const latest = useRef({ config, datasets, selectedDataset, testSentence, tokenizer, model, actualVocab, postTrain });
  latest.current = { config, datasets, selectedDataset, testSentence, tokenizer, model, actualVocab, postTrain };

  const saveNow = () => {
    if (!dirty.current || savingOff.current) return;
    dirty.current = false;
    const s = latest.current;
    // exportState copies the weights synchronously, so this is a consistent snapshot even mid-training
    saveSession({
      config: s.config,
      selectedDatasetId: s.selectedDataset.id,
      customDatasets: s.datasets.filter(d => !SAMPLE_DATASETS.includes(d)),
      vocabulary: Array.from({ length: s.actualVocab }, (_, id) => s.tokenizer.decode([id])),
      testSentence: s.testSentence,
      // While post-training, save the pre-trained weights: post-training isn't saved, so a reload
      // comes back to the model it started from
      model: s.postTrain ? s.postTrain.base : s.model.exportState(),
      history: history.current,
    }).then(ok => setSaveStatus(ok ? { kind: 'saved', at: Date.now() } : { kind: 'unavailable' }));
  };

  const onHistoryChange = (h: TrainingHistory) => {
    history.current = h;
    dirty.current = true;
  };
  useEffect(() => {
    dirty.current = true;
  }, [config, datasets, selectedDataset, testSentence, model, postTrain]);
  useEffect(() => {
    if (!isTraining) saveNow();
  }, [isTraining]);
  useEffect(() => {
    const timer = window.setInterval(saveNow, AUTOSAVE_MS);
    const onVisibility = () => document.visibilityState === 'hidden' && saveNow();
    document.addEventListener('visibilitychange', onVisibility);
    window.addEventListener('pagehide', saveNow);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisibility);
      window.removeEventListener('pagehide', saveNow);
    };
  }, []);

  const startFresh = async () => {
    savingOff.current = true; // so the reload's pagehide doesn't save it all again
    await clearSession();
    window.location.reload();
  };

  return (
    <div className="app-shell" style={{ maxWidth: '1400px', margin: '0 auto', padding: '0 20px 40px 20px' }}>
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isTraining={isTraining}
        isPostTraining={isPostTraining}
        onExport={() => setExportOpen(true)}
      />
      <ExportModal
        open={exportOpen}
        onClose={closeExport}
        model={model}
        tokenizer={tokenizer}
        selectedDataset={selectedDataset}
        stepCount={history.current.stepCount}
        trainingBusy={isTraining}
      />

      <main style={{ minHeight: '80vh' }}>
        {restoreNote && (
          <div
            style={{
              display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', padding: '10px 16px', marginBottom: 20, borderRadius: 10,
              fontSize: '0.85rem', color: 'var(--text-muted)', background: 'var(--surface-inset)', border: '1px solid var(--border-color)',
            }}
          >
            <History size={16} color="var(--accent-emerald)" style={{ flexShrink: 0 }} />
            <span style={{ flex: 1, minWidth: 200 }}>
              {restoreNote.kind === 'restored' ? (
                <>
                  <b style={{ color: 'var(--text-main)' }}>Welcome back.</b> Your model was restored from step #{restoreNote.step.toLocaleString()}
                  {' '}(saved in this browser {new Date(restoreNote.savedAt).toLocaleString()}).
                </>
              ) : restoreNote.reason === 'architecture' ? (
                <>
                  Your saved model couldn't be restored: LLMStead now uses GPT-2's block layout (normalizing before each half
                  instead of after), so weights trained on the old layout don't fit. Your settings were kept; you're starting
                  with a fresh model.
                </>
              ) : (
                <>
                  Your saved model couldn't be restored: the site's data has changed since it was saved, so its weights no longer
                  match the vocabulary. You're starting with a fresh model.
                </>
              )}
            </span>
            {restoreNote.kind === 'restored' && (
              <button className="btn-secondary" onClick={startFresh} style={{ padding: '4px 10px', fontSize: '0.8rem' }}>
                Start fresh
              </button>
            )}
            <button
              onClick={() => setRestoreNote(null)}
              aria-label="Dismiss"
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-dim)', display: 'flex', padding: 2 }}
            >
              <X size={16} />
            </button>
          </div>
        )}

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
            initialHistory={initialHistory}
            onHistoryChange={onHistoryChange}
            saveStatus={saveStatus}
            postTrained={postTrain !== null}
            onRestoreBase={restorePreTrained}
            onKeepPostTrained={() => setPostTrain(null)}
            onNavigateToPostTrain={() => setActiveTab('posttrain')}
          />
        </div>

        {activeTab === 'pipeline' && (
          <>
            {!guidesHidden && (
              <GuideStrip step={3} title="Look inside" next={{ label: 'Next: Post-train', onClick: () => setActiveTab('posttrain') }} onHide={() => setGuides(true)}>
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

        {/* Kept mounted like Train, so post-training keeps running while you look elsewhere */}
        <div style={{ display: activeTab === 'posttrain' ? 'block' : 'none' }}>
          {!guidesHidden && (
            <GuideStrip step={4} title="Post-train it" next={{ label: "Next: What's next", onClick: () => setActiveTab('next') }} onHide={() => setGuides(true)}>
              Pre-training taught the model to continue text. Pick a method, look at its starter data, and press Start: then
              compare the model before and after, and watch what it costs. Restore brings the pre-trained model back.
            </GuideStrip>
          )}
          <PostTrainView
            model={model}
            tokenizer={tokenizer}
            config={effectiveConfig}
            selectedDataset={selectedDataset}
            session={postTrain}
            onBeginSession={beginPostTrain}
            onRestoreBase={restorePreTrained}
            pretrainBusy={isTraining}
            visible={activeTab === 'posttrain'}
            onRunningChange={setIsPostTraining}
            onNavigateToTrain={() => setActiveTab('training')}
          />
        </div>

        {activeTab === 'next' && <WhatsNextView onNavigate={setActiveTab} onExport={() => setExportOpen(true)} />}
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
