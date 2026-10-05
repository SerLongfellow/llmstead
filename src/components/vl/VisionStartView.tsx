import React from 'react';
import { ArrowRight, BookA, CircleSlash, FlaskConical, Images } from 'lucide-react';
import { THEME } from '../../styles/theme';
import { VisionStage, VlmPipeline } from './VlmPipeline';

interface VisionStartViewProps {
  onNavigate: (tab: string) => void;
  onStage: (stage: VisionStage) => void;
  guidesHidden: boolean;
  onShowGuides: () => void;
}

const STEPS = [
  {
    step: 1,
    tab: 'setup',
    title: 'Set up',
    text: 'See the pictures and captions (or questions) it learns from, how a picture is cut into patches, and how big the model is. The defaults train well.',
  },
  {
    step: 2,
    tab: 'training',
    title: 'Train',
    text: 'Match: watch the right picture + caption pairs light up. Describe: run the three phases and watch it start to use the picture.',
  },
  {
    step: 3,
    tab: 'pipeline',
    title: 'Look inside',
    text: 'Search the pictures with your own words, or ask the VLM a question and see which patches it looked at for each word of its answer.',
  },
];

const NOT_THIS: { title: string; text: string; link?: { label: string; tab: string } }[] = [
  {
    title: 'Not a model of real photos.',
    text: 'It only ever sees coloured shapes on a 24 × 24 grid, with captions made from about 20 words. The real CLIP learned from 400 million photos and their captions from the web.',
  },
  {
    title: 'Not a chat model.',
    text: "The Describe model answers six kinds of question about one shape, with about 30 words. Real vision-language models put a CLIP-style image tower in front of a large language model that already knows how to talk.",
    link: { label: 'See how they compare →', tab: 'next' },
  },
  {
    title: 'Not how CLIP is trained for real.',
    text: 'The real one compared batches of 32,768 pairs at a time on hundreds of GPUs. Here it is 16 pairs at a time in plain JavaScript on your CPU, so every step can be inspected.',
  },
];

const EXPERIMENTS = [
  {
    title: 'What does it learn first?',
    text: 'Train with the defaults and watch the test chart. Colour is right within a few hundred steps. Shape and position take longer, and the colour + shape pairs it never saw take longest, and stay the shakiest.',
  },
  {
    title: 'Low loss, missing facts',
    text: "In Setup, set the patch size to 6 pixels, then train. The loss gets low, yet shape stays close to a coin flip for thousands of steps: in a batch of 16, colour and position almost always tell every pair apart, and the loss only rewards telling pairs apart. Then switch the captions to just colour + shape, so shape is the only way to tell many pairs apart.",
  },
  {
    title: 'Green triangles it has never seen',
    text: "Training never shows a green triangle or a yellow cross. Compare the dashed line on the test chart with the solid one, and search for \"a green triangle\" in Look inside. Can it combine 'green' and 'triangle' on its own?",
  },
  {
    title: 'Does it actually look?',
    text: 'In Describe, after phase 0 (text only) the answers are the same with the picture and with a blank one: it answers from habit. After phase 2 the two lines on the "Does it look" chart split apart. In Look inside, pick the blank picture and ask anything.',
  },
  {
    title: 'Eyes that were never told about size',
    text: "Your CLIP's captions say colour, shape and place, but by default never size. Freeze its image tower into the VLM and ask \"is the shape big?\": can the VLM read size out of features that were never trained to keep it?",
  },
  {
    title: 'Why the patch size matters',
    text: 'At 8 pixels, each patch is exactly one of the 9 places a shape can sit, so a shape mostly lands in one patch. At 6, shapes get cut across patches and the image tower has to put them back together with attention. Compare the shape line on the test chart.',
  },
];

/** Two towers, one space: the diagram on Start here */
const TwoTowers: React.FC = () => {
  const box = (x: number, y: number, w: number, h: number, label: string, sub: string, color: string) => (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={10} fill={THEME.surface} stroke={color} strokeWidth={1.5} />
      <text x={x + w / 2} y={y + h / 2 - 4} textAnchor="middle" fontSize={14} fontWeight={700} fill={THEME.textMain}>{label}</text>
      <text x={x + w / 2} y={y + h / 2 + 14} textAnchor="middle" fontSize={11} fill={THEME.textMuted}>{sub}</text>
    </g>
  );
  const arrow = (x1: number, y1: number, x2: number, y2: number) => (
    <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={THEME.textDim} strokeWidth={1.5} markerEnd="url(#vl-arrow)" />
  );
  return (
    <svg viewBox="0 0 760 250" style={{ width: '100%', maxWidth: 760, display: 'block', margin: '0 auto' }} role="img" aria-label="A picture goes through the image tower and a caption through the text tower; both become vectors in one shared space, where matching pairs are pulled together.">
      <defs>
        <marker id="vl-arrow" viewBox="0 0 10 10" refX={9} refY={5} markerWidth={7} markerHeight={7} orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill={THEME.textDim} />
        </marker>
      </defs>
      {/* picture */}
      <rect x={20} y={30} width={60} height={60} rx={6} fill="#1a1a1f" />
      <circle cx={50} cy={60} r={17} fill="#e63333" />
      <text x={50} y={110} textAnchor="middle" fontSize={11} fill={THEME.textMuted}>picture</text>
      {/* caption */}
      <text x={50} y={175} textAnchor="middle" fontSize={12} fontFamily="JetBrains Mono, monospace" fill={THEME.emerald}>"a red</text>
      <text x={50} y={191} textAnchor="middle" fontSize={12} fontFamily="JetBrains Mono, monospace" fill={THEME.emerald}>circle"</text>
      <text x={50} y={212} textAnchor="middle" fontSize={11} fill={THEME.textMuted}>caption</text>

      {arrow(90, 60, 140, 60)}
      {arrow(90, 183, 140, 183)}
      {box(145, 25, 190, 70, 'Image tower', 'patches → transformer', THEME.cyan)}
      {box(145, 148, 190, 70, 'Text tower', 'words → transformer', THEME.emerald)}
      {arrow(340, 60, 420, 105)}
      {arrow(340, 183, 420, 140)}

      {/* shared space */}
      <rect x={425} y={50} width={315} height={150} rx={12} fill="none" stroke={THEME.purple} strokeDasharray="5 4" />
      <text x={582} y={40} textAnchor="middle" fontSize={12} fontWeight={700} fill={THEME.purple}>one shared space (16 numbers)</text>
      <circle cx={505} cy={112} r={7} fill={THEME.cyan} />
      <circle cx={530} cy={132} r={7} fill={THEME.emerald} />
      <line x1={510} y1={116} x2={525} y2={128} stroke={THEME.textMain} strokeWidth={1.5} />
      <text x={545} y={120} fontSize={11} fill={THEME.textMain}>match: pulled together</text>
      <circle cx={660} cy={80} r={6} fill={THEME.emerald} opacity={0.6} />
      <circle cx={690} cy={170} r={6} fill={THEME.emerald} opacity={0.6} />
      <text x={600} y={185} textAnchor="end" fontSize={11} fill={THEME.textDim}>other captions: pushed away</text>
    </svg>
  );
};

export const VisionStartView: React.FC<VisionStartViewProps> = ({ onNavigate, onStage, guidesHidden, onShowGuides }) => (
  <div style={{ maxWidth: 960, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>
    <div className="glass-panel" style={{ padding: '32px 32px 28px', display: 'flex', gap: 24, alignItems: 'flex-start', flexWrap: 'wrap' }}>
      <div style={{ background: 'var(--primary)', borderRadius: 14, padding: 14, display: 'flex', color: '#ffffff' }}>
        <Images size={48} />
      </div>
      <div style={{ flex: 1, minWidth: 280 }}>
        <h2 style={{ fontSize: '1.6rem', fontWeight: 700, marginBottom: 8 }}>Teach a model to match pictures and words</h2>
        <p style={{ color: 'var(--text-muted)', lineHeight: 1.65 }}>
          How does a model know that a photo shows "a dog on a beach"? Many vision-language models are built on a{' '}
          <b style={{ color: 'var(--text-main)' }}>CLIP</b>-style model (OpenAI, 2021): two transformers trained side by side, one
          reading pictures and one reading captions, until a picture and its caption land on nearly the same point in a shared
          space. Here you train a tiny one from scratch, on simple pictures of shapes, and watch it learn which words go with
          which pictures.
        </p>
      </div>
    </div>

    <div className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 14 }}>
      <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>How it learns: two towers, one space</h3>
      <TwoTowers />
      <div style={{ display: 'flex', gap: 10, padding: '10px 14px', borderRadius: 8, background: 'var(--surface-inset)', border: '1px solid var(--border-color)' }}>
        <BookA size={16} color="var(--accent-purple)" style={{ flexShrink: 0, marginTop: 2 }} />
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>
          <b style={{ color: 'var(--text-main)' }}>What's a "tower"?</b> One of the separate networks in a model that takes in more
          than one kind of input. Each is its own stack of transformer layers; papers usually draw them as tall stacks standing side
          by side, hence the name. CLIP has two, the two boxes in the middle of the diagram above: an{' '}
          <b style={{ color: 'var(--text-main)' }}>image tower</b> that reads the picture's patches, and a{' '}
          <b style={{ color: 'var(--text-main)' }}>text tower</b> that reads the caption's words. They share no weights and never
          see each other's input. Each ends in one vector, and the two vectors only meet at the very end, in the shared space on
          the right, where they're compared. (This design is also called a two-tower model or dual encoder.)
        </p>
      </div>
      <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
        Each training step takes a batch of pictures with their captions and scores{' '}
        <i>every</i> picture against <i>every</i> caption. The loss rewards the right pairs for scoring highest and every wrong
        pairing in the batch for scoring low. That's why it's called <b style={{ color: 'var(--text-main)' }}>contrastive</b>{' '}
        learning: a caption is learned as much from the pictures it doesn't describe as from the one it does.
      </p>
    </div>

    <div className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 14 }}>
      <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>Then a second model: from matching to describing</h3>
      <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
        CLIP can tell which caption fits, but it can't say anything. The second model, Describe, builds a tiny vision-language model the way LLaVA
        (2023) did: your CLIP's image tower, frozen, turns a picture into one vector per patch; a small <b style={{ color: 'var(--text-main)' }}>projector</b>{' '}
        turns each into an "image token" the size of a word embedding; and a GPT reads those tokens followed by a question, then
        writes the answer word by word.
      </p>
      <VlmPipeline phase={null} />
      <div>
        <button className="btn-secondary" onClick={() => { onStage('describe'); onNavigate('setup'); }}>
          Go to Describe <ArrowRight size={14} />
        </button>
      </div>
    </div>

    <div>
      <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 12 }}>The same three steps for each model</h3>
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
              {n.link && (
                <>
                  {' '}
                  <button
                    onClick={() => onNavigate(n.link!.tab)}
                    style={{ background: 'none', border: 'none', padding: 0, font: 'inherit', fontWeight: 600, color: 'var(--primary)', cursor: 'pointer' }}
                  >
                    {n.link.label}
                  </button>
                </>
              )}
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
      <button className="btn-primary" onClick={() => { onStage('match'); onNavigate('setup'); }} style={{ padding: '12px 22px', fontSize: '0.95rem' }}>
        Start with Match: set up your CLIP <ArrowRight size={16} />
      </button>
      {guidesHidden && (
        <button className="btn-secondary" onClick={onShowGuides}>
          Show the step guides again
        </button>
      )}
    </div>
  </div>
);
