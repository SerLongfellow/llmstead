import React from 'react';
import { ArrowRight, FlaskConical, Info } from 'lucide-react';
import { GoDeeper } from '../GoDeeper';

interface JepaStartViewProps {
  onNavigate: (tab: string) => void;
  guidesHidden: boolean;
  onShowGuides: () => void;
}

const STEPS = [
  {
    step: 1,
    tab: 'setup',
    title: 'Set up',
    text: 'Meet the images: one coloured shape on a 16×16 canvas, cut into 16 patches. Some patches get hidden, and the model has to guess what they are.',
  },
  {
    step: 2,
    tab: 'training',
    title: 'Train',
    text: 'Watch the prediction loss fall, and the collapse chart that tells you whether that means anything. Then probe what the embeddings learned.',
  },
  {
    step: 3,
    tab: 'pipeline',
    title: 'Look inside',
    text: 'Follow one image through the three networks, see how close each guess is, and find which images the model thinks are alike.',
  },
];

/** GPT vs JEPA, side by side */
const COMPARISON: [string, string, string][] = [
  ['Learns from', 'Text, one token at a time', 'Images, cut into patches'],
  ['Training task', 'Predict the next token', 'Predict the embeddings of hidden patches'],
  ['Predicts in', 'Token space (a probability for every word piece)', 'Embedding space (a list of numbers)'],
  ['Attention', 'Causal: only looks back', 'Every patch sees every other'],
  ['Networks', 'One transformer', 'Three: context encoder, predictor, target encoder'],
  ['Output you can read', 'Generated text', 'None. You judge it by probing its embeddings'],
];

const EXPERIMENTS = [
  {
    title: 'The loss lies',
    text: 'Train with the default recipe and watch the prediction loss fall to almost nothing. Now look at the collapse chart: the embeddings quietly squeeze into about two directions. A low loss only means the guesses match the targets, not that the targets are worth matching.',
  },
  {
    title: 'Pull out the safety pin',
    text: 'While it trains, switch the recipe to "No stop-gradient". The loss drops even further, and the embeddings shrink toward a single point: every image gets the same answer. This is collapse, the failure JEPA\'s design exists to prevent.',
  },
  {
    title: 'Pay it to stay spread out',
    text: 'Start over with the anti-collapse term switched on. The embeddings use many more directions, and the position probe beats the untrained encoder, especially with only a few labels.',
  },
  {
    title: 'What does "similar" mean to it?',
    text: 'In Look inside, check an image\'s nearest neighbours before and after training. Do they share its shape, its colour, or just where the shape sits?',
  },
];

export const JepaStartView: React.FC<JepaStartViewProps> = ({ onNavigate, guidesHidden, onShowGuides }) => (
  <div style={{ maxWidth: 960, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>
    {/* Intro */}
    <div className="glass-panel" style={{ padding: '32px 32px 28px' }}>
      <span className="badge badge-purple" style={{ marginBottom: 12 }}>Image model</span>
      <h2 style={{ fontSize: '1.6rem', fontWeight: 700, marginBottom: 8 }}>Raise a JEPA: learning by predicting what's hidden</h2>
      <p style={{ color: 'var(--text-muted)', lineHeight: 1.65, marginBottom: 12 }}>
        A JEPA (Joint-Embedding Predictive Architecture) learns about images without any labels. It hides part of an image and
        tries to predict the missing part, but not pixel by pixel: it predicts the hidden part's <i>embedding</i>, a short list of
        numbers that describes it. The idea, championed by Yann LeCun, is that predicting in this abstract space lets a model
        focus on what's predictable (there's a red square there) and ignore what isn't (the exact noise in each pixel).
      </p>
      <p style={{ color: 'var(--text-muted)', lineHeight: 1.65 }}>
        This is a tiny I-JEPA, the image version, built from the same transformer blocks as the text model and trained from
        scratch here in your browser. It's small enough to train in minutes and fully open to inspect.
      </p>
    </div>

    {/* How it works */}
    <div className="glass-panel" style={{ padding: '24px 28px' }}>
      <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: 14 }}>How it works</h3>
      <div style={{ display: 'flex', alignItems: 'stretch', gap: 10, flexWrap: 'wrap', marginBottom: 16 }}>
        {[
          { title: 'Context encoder', text: 'Sees only the visible patches and turns each into an embedding.', color: 'var(--primary)' },
          { title: 'Predictor', text: 'Given those embeddings and where the hidden patches are, guesses each hidden patch\'s embedding.', color: 'var(--accent-purple)' },
          { title: 'Target encoder', text: 'Sees the whole image and produces the "right answers" for the hidden patches.', color: 'var(--accent-amber)' },
          { title: 'Loss', text: 'How far each guess is from its answer. Training shrinks it.', color: 'var(--accent-rose)' },
        ].map((b, i) => (
          <React.Fragment key={b.title}>
            {i > 0 && <ArrowRight size={18} color="var(--text-dim)" style={{ alignSelf: 'center', flexShrink: 0 }} />}
            <div style={{ flex: '1 1 160px', padding: '12px 14px', borderRadius: 8, background: 'var(--surface-inset)', borderTop: `3px solid ${b.color}` }}>
              <p style={{ fontWeight: 700, fontSize: '0.9rem', marginBottom: 4 }}>{b.title}</p>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{b.text}</p>
            </div>
          </React.Fragment>
        ))}
      </div>
      <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
        <b style={{ color: 'var(--text-main)' }}>The catch:</b> the answers come from a network that is also learning. If training
        could change both sides freely, the easiest way to get a perfect score would be to give every image the same embedding.
        That's called <b style={{ color: 'var(--text-main)' }}>collapse</b>. I-JEPA prevents it with two tricks: the loss never
        trains the target encoder (<i>stop-gradient</i>), and the target encoder is a slowly moving average of the context
        encoder (<i>EMA</i>). You can switch both off and watch what happens.
      </p>
    </div>

    {/* Comparison */}
    <div className="glass-panel" style={{ padding: '24px 28px', overflowX: 'auto' }}>
      <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: 12 }}>Compared with the text model</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', minWidth: 520 }}>
        <thead>
          <tr style={{ color: 'var(--text-dim)', textAlign: 'left' }}>
            <th style={{ padding: '6px 10px 8px 0', fontWeight: 600 }} />
            <th style={{ padding: '6px 10px 8px', fontWeight: 600 }}>Text · GPT</th>
            <th style={{ padding: '6px 10px 8px', fontWeight: 600 }}>Images · JEPA</th>
          </tr>
        </thead>
        <tbody>
          {COMPARISON.map(([what, gpt, jepa]) => (
            <tr key={what} style={{ borderTop: '1px solid var(--border-color)' }}>
              <td style={{ padding: '8px 10px 8px 0', fontWeight: 600, whiteSpace: 'nowrap' }}>{what}</td>
              <td style={{ padding: '8px 10px', color: 'var(--text-muted)' }}>{gpt}</td>
              <td style={{ padding: '8px 10px', color: 'var(--text-muted)' }}>{jepa}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginTop: 10 }}>
        The building block is the same: both use the identical attention + MLP transformer block. What differs is what goes in,
        what's predicted, and how it's scored.
      </p>
    </div>

    {/* Steps */}
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 }}>
      {STEPS.map(s => (
        <button
          key={s.step}
          className="glass-panel"
          onClick={() => onNavigate(s.tab)}
          style={{ padding: '18px 20px', textAlign: 'left', cursor: 'pointer', color: 'inherit', font: 'inherit', display: 'flex', flexDirection: 'column', gap: 8 }}
        >
          <span style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span
              className="font-mono"
              style={{ width: 24, height: 24, borderRadius: 12, background: 'var(--primary)', color: '#ffffff', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', fontWeight: 700 }}
            >
              {s.step}
            </span>
            <span style={{ fontWeight: 700 }}>{s.title}</span>
          </span>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>{s.text}</span>
        </button>
      ))}
    </div>
    <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
      <button className="btn-primary" onClick={() => onNavigate('setup')}>
        Start: set up your JEPA <ArrowRight size={16} />
      </button>
      {guidesHidden && (
        <button className="btn-secondary" onClick={onShowGuides}>
          Show the step guides again
        </button>
      )}
    </div>

    {/* Experiments */}
    <div className="glass-panel" style={{ padding: '24px 28px' }}>
      <h3 style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
        <FlaskConical size={18} color="var(--accent-emerald)" /> Experiments to try
      </h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14 }}>
        {EXPERIMENTS.map(e => (
          <div key={e.title} style={{ padding: '14px 16px', borderRadius: 8, background: 'var(--surface-inset)', border: '1px solid var(--border-color)' }}>
            <p style={{ fontWeight: 700, fontSize: '0.9rem', marginBottom: 6 }}>{e.title}</p>
            <p style={{ fontSize: '0.825rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>{e.text}</p>
          </div>
        ))}
      </div>
    </div>

    {/* Honest limits */}
    <div className="glass-panel" style={{ padding: '20px 28px', display: 'flex', gap: 14, alignItems: 'flex-start' }}>
      <Info size={18} color="var(--accent-amber)" style={{ flexShrink: 0, marginTop: 2 }} />
      <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
        <p style={{ marginBottom: 8 }}>
          <b style={{ color: 'var(--text-main)' }}>What to expect.</b> With about 24,000 parameters and 16×16 images, this JEPA
          shows the mechanics (masking, prediction in embedding space, collapse and how to prevent it) but it won't learn
          much. In our tests it learns where the shape is, and never gets better than an untrained network at telling a
          circle from a square. Real JEPAs are millions of times larger and train for days on millions of photos or videos.
        </p>
        <GoDeeper
          links={[
            { label: 'I-JEPA paper', url: 'https://arxiv.org/abs/2301.08243' },
            { label: 'V-JEPA (video)', url: 'https://arxiv.org/abs/2404.08471' },
            { label: 'LeCun: A Path Towards Autonomous Machine Intelligence', url: 'https://openreview.net/forum?id=BZ5a1r-kVsf' },
            { label: 'VICReg (the anti-collapse term)', url: 'https://arxiv.org/abs/2105.04906' },
            { label: 'BYOL (EMA targets)', url: 'https://arxiv.org/abs/2006.07733' },
          ]}
        />
      </div>
    </div>
  </div>
);
