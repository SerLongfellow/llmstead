import React from 'react';
import { AlertTriangle, ArrowRight, CheckCircle2, CircleSlash, FlaskConical, MessageSquareText } from 'lucide-react';
import { VlmPipeline } from './VlmPipeline';
import { VlmEncoder } from './visionSettings';

interface VlmStartViewProps {
  onNavigate: (tab: string) => void;
  eyes: VlmEncoder;                // which encoder the VLM is using
  pretrainedSteps: number | null;  // the ready-made CLIP's training steps, once loaded
  clipSteps: number;     // how far the CLIP (Images · CLIP mode) has trained
  clipReady: boolean;
  onOpenClip: () => void;
  guidesHidden: boolean;
  onShowGuides: () => void;
}

const STEPS = [
  {
    step: 1,
    tab: 'setup',
    title: 'Set up',
    text: 'Give it eyes (a frozen copy of your CLIP), choose the questions it learns to answer, and how big its language model is.',
  },
  {
    step: 2,
    tab: 'training',
    title: 'Train',
    text: 'Run the three phases, like LLaVA, and watch the answers start to depend on the picture instead of on habit.',
  },
  {
    step: 3,
    tab: 'pipeline',
    title: 'Look inside',
    text: 'Ask it a question about a picture and click each word of its answer to see which patches it was looking at.',
  },
];

const NOT_THIS = [
  {
    title: 'Not a chat model.',
    text: 'It answers six kinds of question about one shape, with about 30 words. Real vision-language models put a CLIP-style image tower in front of a large language model that already knows how to talk.',
  },
  {
    title: 'Not how LLaVA is trained.',
    text: "LLaVA's language model arrives already trained on huge amounts of text, and its phases are tuned on hundreds of thousands of real images. Here every part starts from nothing except the eyes you trained yourself.",
  },
];

const EXPERIMENTS = [
  {
    title: 'Does it actually look?',
    text: 'After phase 0 (text only) the answers are the same with the picture and with a blank one: it answers from habit. After phase 2 the two lines on the "Does it look?" chart split apart. In Look inside, pick the blank picture and ask anything.',
  },
  {
    title: 'Eyes that were never told about size',
    text: "Your CLIP's captions say colour, shape and place, but by default never size. Ask \"is the shape big?\": can the VLM read size out of features that were never trained to keep it?",
  },
  {
    title: 'Skip the phases',
    text: "Start over and go straight to phase 2. With a model this small it does about as well, because its language model has nothing to protect. LLaVA's phases are there for a big model that does.",
  },
  {
    title: 'Blurry eyes',
    text: 'In Set up, switch the eyes to an untrained encoder and train again. How far does the VLM get when the picture reaches it as noise?',
  },
];

export const VlmStartView: React.FC<VlmStartViewProps> = ({ onNavigate, eyes, pretrainedSteps, clipSteps, clipReady, onOpenClip, guidesHidden, onShowGuides }) => {
  const ok = eyes === 'pretrained' || (eyes === 'clip' && clipReady);
  return (
  <div style={{ maxWidth: 960, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>
    <div className="glass-panel" style={{ padding: '32px 32px 28px', display: 'flex', gap: 24, alignItems: 'flex-start', flexWrap: 'wrap' }}>
      <div style={{ background: 'var(--primary)', borderRadius: 14, padding: 14, display: 'flex', color: '#ffffff' }}>
        <MessageSquareText size={48} />
      </div>
      <div style={{ flex: 1, minWidth: 280 }}>
        <h2 style={{ fontSize: '1.6rem', fontWeight: 700, marginBottom: 8 }}>Teach a model to answer questions about pictures</h2>
        <p style={{ color: 'var(--text-muted)', lineHeight: 1.65 }}>
          A CLIP can tell which caption fits a picture, but it can't say anything. A <b style={{ color: 'var(--text-main)' }}>vision-language
          model</b> (VLM) can: ask "what colour is the shape?" and it writes the answer. Here you build a tiny one the way LLaVA (2023)
          was built, by giving a small GPT a pair of eyes: the image tower of a CLIP like the one in the Images · CLIP mode.
        </p>
      </div>
    </div>

    {/* The CLIP it depends on */}
    <div
      className="glass-panel"
      style={{ padding: '14px 20px', display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', borderColor: ok ? undefined : 'var(--amber-tint)' }}
    >
      {ok ? <CheckCircle2 size={18} color="var(--accent-emerald)" /> : <AlertTriangle size={18} color="var(--accent-amber)" />}
      <span style={{ flex: '1 1 300px', fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>
        {eyes === 'pretrained' ? (
          <>
            <b style={{ color: 'var(--text-main)' }}>Ready to go:</b> it starts with ready-made eyes, a CLIP trained with this site's own
            code{pretrainedSteps ? <> for {pretrainedSteps.toLocaleString()} steps</> : null}. For the full journey, train your own in
            Images · CLIP, then pick "Your CLIP" in Set up.
          </>
        ) : eyes === 'random' ? (
          <>It's using untrained eyes, for comparison. Pick the ready-made CLIP or your own in Set up.</>
        ) : clipReady ? (
          <>Your CLIP has trained for {clipSteps.toLocaleString()} steps, so it's ready to lend its eyes.</>
        ) : (
          <>
            <b style={{ color: 'var(--text-main)' }}>This model needs a trained CLIP first.</b> Yours is at step #
            {clipSteps.toLocaleString()}; give it a couple of minutes of training (a few thousand steps) in the Images · CLIP mode.
          </>
        )}
      </span>
      <button className={ok ? 'btn-secondary' : 'btn-primary'} onClick={onOpenClip} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
        {ok ? 'Open Images · CLIP' : 'Train the CLIP'} <ArrowRight size={14} />
      </button>
    </div>

    <div className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 14 }}>
      <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>How it works: three parts in a row</h3>
      <VlmPipeline phase={null} />
      <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
        The language model never sees pixels. The vision encoder turns the picture into one vector per patch, and the{' '}
        <b style={{ color: 'var(--text-main)' }}>projector</b> turns each of those into an <b style={{ color: 'var(--text-main)' }}>image
        token</b>: a vector the same size as a word embedding. The language model reads the image tokens exactly where it would read
        words, followed by the question, and writes the answer one word at a time, the same way the text model on this site predicts
        the next token. It learns to use the image tokens only because answering the questions needs them.
      </p>
    </div>

    <div>
      <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 12 }}>Three steps, left to right</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14 }}>
        {STEPS.map(s => (
          <button
            key={s.step}
            onClick={() => onNavigate(s.tab)}
            className="glass-panel"
            style={{ padding: 18, textAlign: 'left', cursor: 'pointer', color: 'var(--text-main)', display: 'flex', flexDirection: 'column', gap: 8 }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span className="font-mono" style={{ width: 24, height: 24, borderRadius: 12, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', fontWeight: 700, background: 'var(--primary)', color: '#ffffff' }}>
                {s.step}
              </span>
              <span style={{ fontWeight: 700 }}>{s.title}</span>
            </div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>{s.text}</span>
          </button>
        ))}
      </div>
    </div>

    <div className="glass-panel" style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
        <CircleSlash size={18} color="var(--accent-rose)" />
        <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>What this isn't</h3>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {NOT_THIS.map(n => (
          <div key={n.title} style={{ display: 'grid', gridTemplateColumns: '14px 1fr', gap: 10 }}>
            <span style={{ color: 'var(--accent-rose)', fontWeight: 700, lineHeight: 1.5 }}>×</span>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>
              <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>{n.title}</span> {n.text}
            </div>
          </div>
        ))}
      </div>
    </div>

    <div className="glass-panel" style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
        <FlaskConical size={18} color="var(--accent-amber)" />
        <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>Things to try</h3>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {EXPERIMENTS.map(e => (
          <div key={e.title} style={{ padding: '12px 14px', borderRadius: 8, background: 'var(--surface-inset)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontWeight: 600, marginBottom: 4 }}>{e.title}</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>{e.text}</div>
          </div>
        ))}
      </div>
    </div>

    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 16, flexWrap: 'wrap' }}>
      <button className="btn-primary" onClick={() => onNavigate('setup')} style={{ padding: '12px 22px', fontSize: '0.95rem' }}>
        Start: set up your VLM <ArrowRight size={16} />
      </button>
      {guidesHidden && (
        <button className="btn-secondary" onClick={onShowGuides}>
          Show the step guides again
        </button>
      )}
    </div>
  </div>
  );
};
