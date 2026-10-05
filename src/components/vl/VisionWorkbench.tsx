import React, { useEffect, useMemo, useRef, useState } from 'react';
import { seededRandom } from '../../engine/datasets';
import { MicroClip } from '../../engine/vl/clip';
import { ClipData, testPairs } from '../../engine/vl/clipData';
import { MicroVlm } from '../../engine/vl/vlm';
import { VlmDataSettings } from '../../engine/vl/vlmTraining';
import { PretrainedManifest, fetchPretrainedClip } from '../../engine/vl/pretrained';
import { GuideStrip } from '../GuideStrip';
import { IMAGE_TABS, ModeSwitchProps, Navbar } from '../Navbar';
import { VisionInsideView } from './VisionInsideView';
import { VisionNextView } from './VisionNextView';
import { VisionSetupView } from './VisionSetupView';
import { VisionStartView } from './VisionStartView';
import { VisionTrainView } from './VisionTrainView';
import { VlmInsideView } from './VlmInsideView';
import { VlmNextView } from './VlmNextView';
import { VlmSetupView } from './VlmSetupView';
import { VlmStartView } from './VlmStartView';
import { VlmTrainView } from './VlmTrainView';
import {
  DEFAULT_VISION_SETTINGS, DEFAULT_VLM_SETTINGS, VisionSettings, VlmSettings, clipConfigFor, dataFor, vlmConfigFor,
} from './visionSettings';

const GUIDES_KEY = 'llmstead.hideGuides'; // shared with the GPT side

/** Pictures for the gallery in Look inside (and the samples in Set up): fixed, so runs are comparable */
const GALLERY_SEEN = 48;
const GALLERY_HELD_OUT = 16;

/** Below this many CLIP steps, the VLM suggests training the CLIP first */
export const CLIP_READY_STEPS = 1000;

type ImageMode = 'clip' | 'vlm';

interface VisionWorkbenchProps {
  /** Which of the two image models is showing (null while the GPT is) */
  mode: ImageMode | null;
  modeSwitch: ModeSwitchProps;
  onTrainingChange: (training: { clip: boolean; vlm: boolean }) => void;
}

/** The ready-made CLIP shipped in public/models, fetched the first time the VLM mode is opened */
export type PretrainedEyes =
  | { status: 'loading' }
  | { status: 'ready'; clip: MicroClip; manifest: PretrainedManifest }
  | { status: 'failed'; reason: string };

/** The VLM's vision encoder: a frozen copy of the CLIP, and what it was trained on */
export interface VisionCopy {
  clip: MicroClip;
  step: number;
  data: ClipData;
}

/**
 * The two image models. They're separate modes in the navbar, each with its own tabs, but live
 * in one workbench because the second is built on the first:
 *   Images · CLIP: a tiny CLIP learns which captions fit which pictures
 *   Images → Text · VLM: a tiny LLaVA-style VLM reads pictures through a frozen copy of that
 *   CLIP's image tower and answers questions about them
 */
export const VisionWorkbench: React.FC<VisionWorkbenchProps> = ({ mode, modeSwitch, onTrainingChange }) => {
  // While the GPT is showing, keep the last image model rendered (hidden by the parent)
  const lastMode = useRef<ImageMode>(mode ?? 'clip');
  if (mode) lastMode.current = mode;
  const shown = lastMode.current;
  const [clipTab, setClipTab] = useState('start');
  const [vlmTab, setVlmTab] = useState('start');

  // ── The CLIP ──
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

  // ── The VLM, built the first time its mode is opened ──
  const [vlmSettings, setVlmSettings] = useState<VlmSettings>(DEFAULT_VLM_SETTINGS);
  const [vlmTraining, setVlmTraining] = useState(false);
  const [vlmGeneration, setVlmGeneration] = useState(0);
  const [vlmWeightsVersion, setVlmWeightsVersion] = useState(0);
  // A copy, not the live CLIP: training the CLIP further mustn't change the VLM's eyes underneath it
  const [visionCopy, setVisionCopy] = useState<VisionCopy | null>(null);
  const takeVisionCopy = () => setVisionCopy({ clip: model.clone(), step: model.steps, data });
  useEffect(() => {
    if (mode === 'vlm' && !visionCopy) takeVisionCopy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  // The ready-made eyes: fetched once, used by default; if they can't be loaded, fall back to your own CLIP
  const [pretrained, setPretrained] = useState<PretrainedEyes | null>(null);
  const pretrainedRequested = useRef(false);
  useEffect(() => {
    if (mode !== 'vlm' || pretrainedRequested.current) return;
    pretrainedRequested.current = true;
    setPretrained({ status: 'loading' });
    fetchPretrainedClip()
      .then(r => setPretrained({ status: 'ready', ...r }))
      .catch((e: Error) => setPretrained({ status: 'failed', reason: e.message }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);
  useEffect(() => {
    if (pretrained?.status === 'failed' && vlmSettings.encoder === 'pretrained') setVlmSettings(s => ({ ...s, encoder: 'clip' }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pretrained]);
  const pretrainedCopy = useMemo<VisionCopy | null>(
    () => (pretrained?.status === 'ready' ? { clip: pretrained.clip, step: pretrained.manifest.training.steps, data: pretrained.manifest.data } : null),
    [pretrained]
  );

  const encoder =
    vlmSettings.encoder === 'pretrained' ? pretrainedCopy : vlmSettings.encoder === 'clip' ? visionCopy : visionCopy && { clip: untrained, step: 0, data };
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
    onTrainingChange({ clip: clipTraining, vlm: vlmTraining });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clipTraining, vlmTraining]);

  /** From the VLM: switch to the CLIP's Train tab */
  const openClipTraining = (start: boolean) => {
    modeSwitch.onChange('clip');
    setClipTab('training');
    if (start) setClipTraining(true);
  };
  const openVlm = () => {
    modeSwitch.onChange('vlm');
    setVlmTab('start');
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
  const guide = (setTab: (t: string) => void, step: number, title: string, next: { label: string; tab: string }, text: React.ReactNode) =>
    !guidesHidden && (
      <GuideStrip step={step} title={title} next={{ label: next.label, onClick: () => setTab(next.tab) }} onHide={() => setGuides(true)}>
        {text}
      </GuideStrip>
    );

  const isClip = shown === 'clip';
  const tab = isClip ? clipTab : vlmTab;
  const setTab = isClip ? setClipTab : setVlmTab;

  return (
    <>
      <Navbar activeTab={tab} setActiveTab={setTab} isTraining={isClip ? clipTraining : vlmTraining} tabs={IMAGE_TABS} modeSwitch={modeSwitch} />

      {/* ── Images · CLIP ── */}
      <main style={{ minHeight: '80vh', display: isClip ? 'block' : 'none' }}>
        {clipTab === 'start' && (
          <VisionStartView onNavigate={setClipTab} onOpenVlm={openVlm} guidesHidden={guidesHidden} onShowGuides={() => setGuides(false)} />
        )}
        {clipTab === 'setup' && (
          <>
            {guide(setClipTab, 1, 'Set up your CLIP', { label: 'Next: Train', tab: 'training' }, <>
              Look at the pictures and captions it will learn from, and how each picture is cut into patches. The defaults train
              well. Changing anything here starts the model over from random weights.
            </>)}
            <VisionSetupView settings={settings} onChange={setSettings} paramCount={model.getParameterCount()} samples={gallery.slice(0, 12)} />
          </>
        )}
        {/* Kept mounted (just hidden) so the charts survive tab switches and training keeps running */}
        <div style={{ display: clipTab === 'training' ? 'block' : 'none' }}>
          {guide(setClipTab, 2, 'Train your CLIP', { label: 'Next: Look inside', tab: 'pipeline' }, <>
            Press Start and watch the grid of pictures × captions: the right matches light up along the diagonal. The test scores
            below show what it learns first, and whether it copes with colour + shape pairs it never saw.
          </>)}
          <VisionTrainView
            model={model}
            data={data}
            settings={settings}
            onChange={setSettings}
            visible={mode === 'clip' && clipTab === 'training'}
            isTraining={clipTraining}
            setIsTraining={setClipTraining}
            weightsVersion={weightsVersion}
            onWeightsChanged={() => setWeightsVersion(v => v + 1)}
            onStartOver={() => setGeneration(g => g + 1)}
            onNavigateToSetup={() => setClipTab('setup')}
          />
        </div>
        {clipTab === 'pipeline' && (
          <>
            {guide(setClipTab, 3, 'Look inside your CLIP', { label: "Next: What's next", tab: 'next' }, <>
              Search the pictures with your own words, see where in a picture a caption matches, and look at both towers'
              attention. Flip to the untrained model to compare.
            </>)}
            <VisionInsideView model={model} untrained={untrained} data={data} gallery={gallery} weightsVersion={weightsVersion} />
          </>
        )}
        {clipTab === 'next' && <VisionNextView onNavigate={setClipTab} onOpenVlm={openVlm} />}
      </main>

      {/* ── Images → Text · VLM ── */}
      {vlm && vlmUntrained && vlmData && encoder && (
        <main style={{ minHeight: '80vh', display: isClip ? 'none' : 'block' }}>
          {vlmTab === 'start' && (
            <VlmStartView
              onNavigate={setVlmTab}
              eyes={vlmSettings.encoder}
              pretrainedSteps={pretrainedCopy?.step ?? null}
              clipSteps={model.steps}
              clipReady={model.steps >= CLIP_READY_STEPS}
              onOpenClip={() => openClipTraining(false)}
              guidesHidden={guidesHidden}
              onShowGuides={() => setGuides(false)}
            />
          )}
          {vlmTab === 'setup' && (
            <>
              {guide(setVlmTab, 1, 'Set up your VLM', { label: 'Next: Train', tab: 'training' }, <>
                Choose its eyes (a frozen copy of your CLIP's image tower) and the questions it will learn to answer. Train the CLIP
                first; the defaults are fine for everything else.
              </>)}
              <VlmSetupView
                settings={vlmSettings}
                onChange={setVlmSettings}
                vlm={vlm}
                visionCopy={encoder}
                pretrained={pretrained}
                clipSteps={model.steps}
                clipReady={(visionCopy?.step ?? 0) >= CLIP_READY_STEPS}
                onRefreshCopy={() => {
                  setVlmTraining(false);
                  takeVisionCopy();
                }}
                onTrainClip={() => openClipTraining(true)}
                sample={gallery[0]}
              />
            </>
          )}
          <div style={{ display: vlmTab === 'training' ? 'block' : 'none' }}>
            {guide(setVlmTab, 2, 'Train your VLM', { label: 'Next: Look inside', tab: 'pipeline' }, <>
              Run the three phases in order, like LLaVA: text only, then align the projector, then instruction-tune. After each one,
              compare answers with the picture and with a blank picture: the gap is how much it actually looks.
            </>)}
            <VlmTrainView
              vlm={vlm}
              data={vlmData}
              settings={vlmSettings}
              onChange={setVlmSettings}
              visible={mode === 'vlm' && vlmTab === 'training'}
              isTraining={vlmTraining}
              setIsTraining={setVlmTraining}
              onWeightsChanged={() => setVlmWeightsVersion(v => v + 1)}
              onStartOver={() => setVlmGeneration(g => g + 1)}
              onNavigateToSetup={() => setVlmTab('setup')}
            />
          </div>
          {vlmTab === 'pipeline' && (
            <>
              {guide(setVlmTab, 3, 'Look inside your VLM', { label: "Next: What's next", tab: 'next' }, <>
                Pick a picture, ask a question, and click any word of the answer to see which patches the model was looking at when
                it chose that word. Try the blank picture too.
              </>)}
              <VlmInsideView vlm={vlm} untrained={vlmUntrained} data={vlmData} gallery={gallery} weightsVersion={vlmWeightsVersion} />
            </>
          )}
          {vlmTab === 'next' && <VlmNextView onNavigate={setVlmTab} />}
        </main>
      )}
      {!vlm && shown === 'vlm' && (
        <main style={{ minHeight: '80vh' }}>
          <div className="glass-panel" style={{ padding: 32, maxWidth: 640, margin: '40px auto', textAlign: 'center', color: 'var(--text-muted)' }}>
            Loading the ready-made eyes (a small trained CLIP, about 130 KB)…
          </div>
        </main>
      )}
    </>
  );
};
