import React, { useEffect, useMemo, useState } from 'react';
import { seededRandom } from '../../engine/datasets';
import { MicroClip } from '../../engine/vl/clip';
import { ClipData, testPairs } from '../../engine/vl/clipData';
import { MicroVlm } from '../../engine/vl/vlm';
import { VlmDataSettings } from '../../engine/vl/vlmTraining';
import { GuideStrip } from '../GuideStrip';
import { ModeSwitchProps, Navbar } from '../Navbar';
import { VisionInsideView } from './VisionInsideView';
import { VisionNextView } from './VisionNextView';
import { VisionSetupView } from './VisionSetupView';
import { VisionStartView } from './VisionStartView';
import { VisionTrainView } from './VisionTrainView';
import { VlmInsideView } from './VlmInsideView';
import { StageBar, VisionStage } from './VlmPipeline';
import { VlmSetupView } from './VlmSetupView';
import { VlmTrainView } from './VlmTrainView';
import {
  DEFAULT_VISION_SETTINGS, DEFAULT_VLM_SETTINGS, VisionSettings, VlmSettings, clipConfigFor, dataFor, vlmConfigFor,
} from './visionSettings';

const GUIDES_KEY = 'llmstead.hideGuides'; // shared with the GPT side
const STAGE_KEY = 'llmstead.visionStage';

/** Pictures for the gallery in Look inside (and the samples in Set up): fixed, so runs are comparable */
const GALLERY_SEEN = 48;
const GALLERY_HELD_OUT = 16;

/** Below this many CLIP steps, the VLM's Setup suggests training the CLIP first */
export const CLIP_READY_STEPS = 1000;

interface VisionWorkbenchProps {
  active: boolean; // whether this side is the one showing
  modeSwitch: ModeSwitchProps;
  onTrainingChange: (training: boolean) => void;
}

/** The VLM's vision encoder: a frozen copy of the CLIP, and what it was trained on */
export interface VisionCopy {
  clip: MicroClip;
  step: number;
  data: ClipData;
}

/**
 * The image + text side of the site, in two stages:
 *   Match: a tiny CLIP learns which captions fit which pictures
 *   Describe: a tiny LLaVA-style VLM reads pictures through that CLIP's (frozen) image tower
 *      and answers questions about them
 */
export const VisionWorkbench: React.FC<VisionWorkbenchProps> = ({ active, modeSwitch, onTrainingChange }) => {
  const [activeTab, setActiveTab] = useState('start');
  const [stage, setStageState] = useState<VisionStage>(() => {
    try {
      return localStorage.getItem(STAGE_KEY) === 'describe' ? 'describe' : 'match';
    } catch {
      return 'match';
    }
  });
  const setStage = (s: VisionStage) => {
    setStageState(s);
    try {
      localStorage.setItem(STAGE_KEY, s);
    } catch {
      /* storage unavailable: the choice just won't persist */
    }
  };

  // ── Stage 1: the CLIP ──
  const [settings, setSettings] = useState<VisionSettings>(DEFAULT_VISION_SETTINGS);
  const [clipTraining, setClipTraining] = useState(false);
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
  // A frozen copy of the freshly initialized model: the "untrained" baseline (and the VLM's random encoder)
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

  // ── Stage 2: the VLM, built the first time Describe is opened ──
  const [vlmSettings, setVlmSettings] = useState<VlmSettings>(DEFAULT_VLM_SETTINGS);
  const [vlmTraining, setVlmTraining] = useState(false);
  const [vlmGeneration, setVlmGeneration] = useState(0);
  const [vlmWeightsVersion, setVlmWeightsVersion] = useState(0);
  // A copy, not the live CLIP: training the CLIP further mustn't change the VLM's eyes underneath it
  const [visionCopy, setVisionCopy] = useState<VisionCopy | null>(null);
  const takeVisionCopy = () => setVisionCopy({ clip: model.clone(), step: model.steps, data });
  useEffect(() => {
    if (stage === 'describe' && !visionCopy) takeVisionCopy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage]);

  const encoder = vlmSettings.encoder === 'clip' ? visionCopy : visionCopy && { clip: untrained, step: 0, data };
  const vlm = useMemo(
    () => (encoder ? new MicroVlm(encoder.clip, vlmConfigFor(vlmSettings)) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [encoder?.clip, vlmSettings.dModel, vlmSettings.numLayers, vlmGeneration]
  );
  const vlmUntrained = useMemo(() => vlm?.clone() ?? null, [vlm]);
  const vlmData: VlmDataSettings | null = encoder
    ? { patchSize: encoder.clip.config.patchSize, heldOut: encoder.data.heldOut, tasks: vlmSettings.tasks }
    : null;

  useEffect(() => {
    onTrainingChange(clipTraining || vlmTraining);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clipTraining, vlmTraining]);

  /** From the VLM's Setup: go train the CLIP first */
  const trainClipNow = () => {
    setStage('match');
    setActiveTab('training');
    setClipTraining(true);
  };

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

  const match = stage === 'match';
  const stageBar = <StageBar stage={stage} onChange={setStage} training={{ match: clipTraining, describe: vlmTraining }} />;
  const guide = (step: number, title: string, next: { label: string; tab: string }, text: React.ReactNode) =>
    !guidesHidden && (
      <GuideStrip step={step} title={title} next={{ label: next.label, onClick: () => setActiveTab(next.tab) }} onHide={() => setGuides(true)}>
        {text}
      </GuideStrip>
    );

  return (
    <>
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} isTraining={clipTraining || vlmTraining} modeSwitch={modeSwitch} />
      <main style={{ minHeight: '80vh' }}>
        {activeTab === 'start' && (
          <VisionStartView onNavigate={setActiveTab} onStage={setStage} guidesHidden={guidesHidden} onShowGuides={() => setGuides(false)} />
        )}

        {activeTab === 'setup' && (
          <>
            {stageBar}
            {match
              ? guide(1, 'Set up your CLIP', { label: 'Next: Train', tab: 'training' }, <>
                  Look at the pictures and captions it will learn from, and how each picture is cut into patches. The defaults train
                  well. Changing anything here starts the model over from random weights.
                </>)
              : guide(1, 'Set up your VLM', { label: 'Next: Train', tab: 'training' }, <>
                  Choose the eyes (your CLIP's image tower, frozen) and the questions it will learn to answer. Train the CLIP (Match)
                  first; the defaults are fine for everything else.
                </>)}
            {match ? (
              <VisionSetupView settings={settings} onChange={setSettings} paramCount={model.getParameterCount()} samples={gallery.slice(0, 12)} />
            ) : (
              vlm && encoder && (
                <VlmSetupView
                  settings={vlmSettings}
                  onChange={setVlmSettings}
                  vlm={vlm}
                  visionCopy={encoder}
                  clipSteps={model.steps}
                  clipReady={(visionCopy?.step ?? 0) >= CLIP_READY_STEPS}
                  onRefreshCopy={() => {
                    setVlmTraining(false);
                    takeVisionCopy();
                  }}
                  onTrainClip={trainClipNow}
                  sample={gallery[0]}
                />
              )
            )}
          </>
        )}

        {/* Both Train views are kept mounted (just hidden) so charts survive and training keeps running */}
        <div style={{ display: activeTab === 'training' ? 'block' : 'none' }}>
          {stageBar}
          <div style={{ display: match ? 'block' : 'none' }}>
            {guide(2, 'Train your CLIP', { label: 'Next: Look inside', tab: 'pipeline' }, <>
              Press Start and watch the grid of pictures × captions: the right matches light up along the diagonal. The test scores
              below show what it learns first, and whether it copes with colour + shape pairs it never saw.
            </>)}
            <VisionTrainView
              model={model}
              data={data}
              settings={settings}
              onChange={setSettings}
              visible={active && activeTab === 'training' && match}
              isTraining={clipTraining}
              setIsTraining={setClipTraining}
              weightsVersion={weightsVersion}
              onWeightsChanged={() => setWeightsVersion(v => v + 1)}
              onStartOver={() => setGeneration(g => g + 1)}
              onNavigateToSetup={() => setActiveTab('setup')}
            />
          </div>
          {vlm && vlmData && encoder && (
            <div style={{ display: match ? 'none' : 'block' }}>
              {guide(2, 'Train your VLM', { label: 'Next: Look inside', tab: 'pipeline' }, <>
                Run the three phases in order, like LLaVA: text only, then align the projector, then instruction-tune. After each one,
                compare answers with the picture and with a blank picture: the gap is how much it actually looks.
              </>)}
              <VlmTrainView
                vlm={vlm}
                data={vlmData}
                settings={vlmSettings}
                onChange={setVlmSettings}
                visible={active && activeTab === 'training' && !match}
                isTraining={vlmTraining}
                setIsTraining={setVlmTraining}
                onWeightsChanged={() => setVlmWeightsVersion(v => v + 1)}
                onStartOver={() => setVlmGeneration(g => g + 1)}
                onNavigateToSetup={() => setActiveTab('setup')}
              />
            </div>
          )}
        </div>

        {activeTab === 'pipeline' && (
          <>
            {stageBar}
            {match
              ? guide(3, 'Look inside your CLIP', { label: "Next: What's next", tab: 'next' }, <>
                  Search the pictures with your own words, see where in a picture a caption matches, and look at both towers'
                  attention. Flip to the untrained model to compare.
                </>)
              : guide(3, 'Look inside your VLM', { label: "Next: What's next", tab: 'next' }, <>
                  Pick a picture, ask a question, and click any word of the answer to see which patches the model was looking at when
                  it chose that word. Try the blank picture too.
                </>)}
            {match ? (
              <VisionInsideView model={model} untrained={untrained} data={data} gallery={gallery} weightsVersion={weightsVersion} />
            ) : (
              vlm && vlmUntrained && vlmData && (
                <VlmInsideView vlm={vlm} untrained={vlmUntrained} data={vlmData} gallery={gallery} weightsVersion={vlmWeightsVersion} />
              )
            )}
          </>
        )}

        {activeTab === 'next' && <VisionNextView onNavigate={setActiveTab} />}
      </main>
    </>
  );
};
