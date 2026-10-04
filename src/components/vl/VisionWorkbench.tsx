import React, { useEffect, useMemo, useState } from 'react';
import { seededRandom } from '../../engine/datasets';
import { MicroClip } from '../../engine/vl/clip';
import { testPairs } from '../../engine/vl/clipData';
import { GuideStrip } from '../GuideStrip';
import { ModeSwitchProps, Navbar } from '../Navbar';
import { VisionInsideView } from './VisionInsideView';
import { VisionNextView } from './VisionNextView';
import { VisionSetupView } from './VisionSetupView';
import { VisionStartView } from './VisionStartView';
import { VisionTrainView } from './VisionTrainView';
import { DEFAULT_VISION_SETTINGS, VisionSettings, clipConfigFor, dataFor } from './visionSettings';

const GUIDES_KEY = 'llmstead.hideGuides'; // shared with the GPT side

/** Pictures for the gallery in Look inside (and the samples in Set up): fixed, so runs are comparable */
const GALLERY_SEEN = 48;
const GALLERY_HELD_OUT = 16;

interface VisionWorkbenchProps {
  active: boolean; // whether this side is the one showing
  modeSwitch: ModeSwitchProps;
  onTrainingChange: (training: boolean) => void;
}

/** The image + text side of the site: a tiny CLIP you can set up, train and look inside */
export const VisionWorkbench: React.FC<VisionWorkbenchProps> = ({ active, modeSwitch, onTrainingChange }) => {
  const [activeTab, setActiveTab] = useState('start');
  const [settings, setSettings] = useState<VisionSettings>(DEFAULT_VISION_SETTINGS);
  const [isTraining, setIsTraining] = useState(false);
  useEffect(() => {
    onTrainingChange(isTraining);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTraining]);

  // "Start over" in the Train tab bumps this to get fresh random weights with the same settings
  const [generation, setGeneration] = useState(0);
  // Bumped whenever the page's copy of the weights changes, so the views that read them refresh
  const [weightsVersion, setWeightsVersion] = useState(0);

  // A new model only when the architecture or the data changes; batch size, learning rate and
  // the false-negative rule can change mid-training without throwing the weights away
  const { detail, heldOut, patchSize, dModel, imageLayers, textLayers } = settings;
  const data = useMemo(() => dataFor(settings), [detail, heldOut, patchSize]); // eslint-disable-line react-hooks/exhaustive-deps
  const model = useMemo(
    () => new MicroClip(clipConfigFor(settings)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [data, dModel, imageLayers, textLayers, generation]
  );
  // A frozen copy of the freshly initialized model: the "untrained" baseline
  const untrained = useMemo(() => model.clone(), [model]);
  useEffect(() => {
    model.setFalseNegatives(settings.falseNegatives);
  }, [model, settings.falseNegatives]);

  const gallery = useMemo(
    () => [
      ...testPairs(seededRandom(2001), data, GALLERY_SEEN, 'seen'),
      ...testPairs(seededRandom(2002), data, GALLERY_HELD_OUT, 'held-out'),
    ],
    [data]
  );

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
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} isTraining={isTraining} modeSwitch={modeSwitch} />
      <main style={{ minHeight: '80vh' }}>
        {activeTab === 'start' && (
          <VisionStartView onNavigate={setActiveTab} guidesHidden={guidesHidden} onShowGuides={() => setGuides(false)} />
        )}

        {activeTab === 'setup' && (
          <>
            {!guidesHidden && (
              <GuideStrip step={1} title="Set up your model" next={{ label: 'Next: Train', onClick: () => setActiveTab('training') }} onHide={() => setGuides(true)}>
                Look at the pictures and captions it will learn from, and how each picture is cut into patches. The defaults train
                well. Changing anything here starts the model over from random weights.
              </GuideStrip>
            )}
            <VisionSetupView settings={settings} onChange={setSettings} paramCount={model.getParameterCount()} samples={gallery.slice(0, 12)} />
          </>
        )}

        {/* Kept mounted (just hidden) so the charts survive tab switches and training keeps running */}
        <div style={{ display: activeTab === 'training' ? 'block' : 'none' }}>
          {!guidesHidden && (
            <GuideStrip step={2} title="Train it" next={{ label: 'Next: Look inside', onClick: () => setActiveTab('pipeline') }} onHide={() => setGuides(true)}>
              Press Start and watch the grid of pictures × captions: the right matches light up along the diagonal. The test scores
              below show what it learns first, and whether it copes with colour + shape pairs it never saw.
            </GuideStrip>
          )}
          <VisionTrainView
            model={model}
            data={data}
            settings={settings}
            onChange={setSettings}
            visible={active && activeTab === 'training'}
            isTraining={isTraining}
            setIsTraining={setIsTraining}
            weightsVersion={weightsVersion}
            onWeightsChanged={() => setWeightsVersion(v => v + 1)}
            onStartOver={() => setGeneration(g => g + 1)}
            onNavigateToSetup={() => setActiveTab('setup')}
          />
        </div>

        {activeTab === 'pipeline' && (
          <>
            {!guidesHidden && (
              <GuideStrip step={3} title="Look inside" next={{ label: "Next: What's next", onClick: () => setActiveTab('next') }} onHide={() => setGuides(true)}>
                Search the pictures with your own words, see where in a picture a caption matches, and look at both towers'
                attention. Flip to the untrained model to compare.
              </GuideStrip>
            )}
            <VisionInsideView model={model} untrained={untrained} data={data} gallery={gallery} weightsVersion={weightsVersion} />
          </>
        )}

        {activeTab === 'next' && <VisionNextView onNavigate={setActiveTab} />}
      </main>
    </>
  );
};
