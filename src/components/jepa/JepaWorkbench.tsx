import React, { useEffect, useMemo, useState } from 'react';
import { MicroJepa } from '../../engine/jepa/jepa';
import { probeImageSets } from '../../engine/jepa/probes';
import { GuideStrip } from '../GuideStrip';
import { ModeSwitchProps, Navbar, NavTab } from '../Navbar';
import { JepaInsideView } from './JepaInsideView';
import { JepaNextView } from './JepaNextView';
import { JepaSetupView } from './JepaSetupView';
import { JepaStartView } from './JepaStartView';
import { JepaTrainView } from './JepaTrainView';
import { DEFAULT_JEPA_SETTINGS, IMAGE_SIZE, JepaSettings, jepaConfigFor, shapeSizes } from './jepaSettings';

const JEPA_TABS: readonly NavTab[] = [
  { id: 'start', label: 'Start here', step: null },
  { id: 'setup', label: 'Set up', step: 1 },
  { id: 'training', label: 'Train', step: 2 },
  { id: 'pipeline', label: 'Look inside', step: 3 },
  { id: 'next', label: "What's next", step: null },
];

/** Labelled images for the probes and the nearest-neighbour gallery (fixed, so runs are comparable) */
const PROBE_TRAIN = 512;
const PROBE_TEST = 256;

const GUIDES_KEY = 'llmstead.hideGuides'; // shared with the GPT side

interface JepaWorkbenchProps {
  active: boolean; // whether the JEPA side is the one showing
  modeSwitch: ModeSwitchProps;
  onTrainingChange: (training: boolean) => void;
}

/** The image-model side of the site: a tiny I-JEPA you can set up, train and look inside */
export const JepaWorkbench: React.FC<JepaWorkbenchProps> = ({ active, modeSwitch, onTrainingChange }) => {
  const [activeTab, setActiveTab] = useState('start');
  const [settings, setSettings] = useState<JepaSettings>(DEFAULT_JEPA_SETTINGS);
  const [isTraining, setIsTraining] = useState(false);
  useEffect(() => {
    onTrainingChange(isTraining);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTraining]);

  // "Start over" in the Train tab bumps this to get fresh random weights with the same settings
  const [generation, setGeneration] = useState(0);

  // The model depends ONLY on the architecture, so the learning rate, masking and the training
  // recipe can change mid-training without throwing the weights away
  const { dModel, numHeads, numLayers, predDim, predLayers } = settings;
  const model = useMemo(
    () => new MicroJepa(jepaConfigFor(settings)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [dModel, numHeads, numLayers, predDim, predLayers, generation]
  );
  // A frozen copy of the freshly initialized model: the "untrained encoder" baseline
  const untrained = useMemo(() => model.clone(), [model]);
  useEffect(() => {
    model.setRecipe(settings.recipe);
  }, [model, settings.recipe]);

  // Training images are drawn fresh every step; these fixed labelled sets are only for evaluation
  const probeSets = useMemo(() => probeImageSets(shapeSizes(settings), PROBE_TRAIN, PROBE_TEST, IMAGE_SIZE), [settings.shapes]);

  const [guidesHidden, setGuidesHidden] = useState<boolean>(() => {
    try {
      return localStorage.getItem(GUIDES_KEY) === '1';
    } catch {
      return false;
    }
  });
  const setGuides = (hidden: boolean) => {
    setGuidesHidden(hidden);
    try {
      localStorage.setItem(GUIDES_KEY, hidden ? '1' : '0');
    } catch {
      /* storage unavailable: the choice just won't persist */
    }
  };

  return (
    <>
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isTraining={isTraining}
        tabs={JEPA_TABS}
        modeSwitch={modeSwitch}
      />
      <main style={{ minHeight: '80vh' }}>
        {activeTab === 'start' && (
          <JepaStartView onNavigate={setActiveTab} guidesHidden={guidesHidden} onShowGuides={() => setGuides(false)} />
        )}

        {activeTab === 'setup' && (
          <>
            {!guidesHidden && (
              <GuideStrip step={1} title="Set up your JEPA" next={{ label: 'Next: Train', onClick: () => setActiveTab('training') }} onHide={() => setGuides(true)}>
                Look at the images it will learn from and how they're cut into patches and masked. The defaults train fine.
                Changing the model's size starts it over from random weights.
              </GuideStrip>
            )}
            <JepaSetupView
              settings={settings}
              onChange={setSettings}
              paramCount={model.getParameterCount()}
              sampleImages={probeSets.test.slice(0, 12)}
            />
          </>
        )}

        {/* Kept mounted (just hidden) so the charts survive tab switches and training keeps running */}
        <div style={{ display: activeTab === 'training' ? 'block' : 'none' }}>
          {!guidesHidden && (
            <GuideStrip step={2} title="Train it" next={{ label: 'Next: Look inside', onClick: () => setActiveTab('pipeline') }} onHide={() => setGuides(true)}>
              Press Start and watch two things: the prediction loss falling, and the collapse chart below it. Then run the probes to
              see what the embeddings actually learned. Try switching the recipe while it trains.
            </GuideStrip>
          )}
          <JepaTrainView
            model={model}
            untrained={untrained}
            settings={settings}
            onChange={setSettings}
            probeSets={probeSets}
            visible={active && activeTab === 'training'}
            isTraining={isTraining}
            setIsTraining={setIsTraining}
            onStartOver={() => setGeneration(g => g + 1)}
            onNavigateToSetup={() => setActiveTab('setup')}
          />
        </div>

        {activeTab === 'pipeline' && (
          <>
            {!guidesHidden && (
              <GuideStrip step={3} title="Look inside" next={{ label: "Next: What's next", onClick: () => setActiveTab('next') }} onHide={() => setGuides(true)}>
                Follow one image through the three networks: what the context encoder sees, what the predictor guesses for the
                hidden patches, and how close it gets. Then see which images the model thinks are alike.
              </GuideStrip>
            )}
            <JepaInsideView model={model} untrained={untrained} settings={settings} gallery={probeSets.test} />
          </>
        )}

        {activeTab === 'next' && <JepaNextView onNavigate={setActiveTab} />}
      </main>
    </>
  );
};
