import React from 'react';
import { ArrowRight, CircleSlash, FlaskConical } from 'lucide-react';
import { ChickenLogo } from './ChickenLogo';

interface StartViewProps {
  onNavigate: (tab: string) => void;
  guidesHidden: boolean;
  onShowGuides: () => void;
}

const STEPS = [
  {
    step: 1,
    tab: 'setup',
    title: 'Set up',
    text: 'Pick the text your model will learn from, how it splits text into tokens, and how big the model is. The defaults are a fine place to start.',
  },
  {
    step: 2,
    tab: 'training',
    title: 'Train',
    text: 'Press Start and watch the loss fall as the model learns to predict the next token. Then run the benchmark to see what it picked up.',
  },
  {
    step: 3,
    tab: 'pipeline',
    title: 'Look inside',
    text: 'Type a prompt and follow it through every stage: tokens, embeddings, attention, and the final prediction.',
  },
];

const NOT_THIS = [
  {
    title: 'Not a useful model.',
    text: "With about 30,000 parameters and a few pages of text, it learns patterns it has seen, like 3 + 4 = 7, and little else. It can't hold a conversation, answer real questions, or reliably do sums it hasn't seen.",
  },
  {
    title: 'Not how real models are trained.',
    text: 'Real training runs on clusters of GPUs with frameworks like PyTorch, feeding in thousands of sequences at once for weeks. Here it is one short window at a time, in plain JavaScript on your CPU, so every step can be read and inspected.',
  },
  {
    title: 'Not the whole recipe behind chat assistants.',
    text: 'This covers pre-training: learning to predict the next token. Assistants like ChatGPT and Claude also go through fine-tuning and training on human feedback, which LLMStead doesn\'t do (yet).',
  },
  {
    title: 'Not an exact copy of a modern LLM.',
    text: 'The architecture is a simplified transformer. Modern models add refinements such as different normalization and position encodings, but the core ideas (tokens, embeddings, attention, next-token prediction) are the same.',
  },
];

const EXPERIMENTS = [
  {
    title: 'What does it learn first?',
    text: 'Train the math dataset (the default) for a couple of minutes, then run the benchmark. It gets sums it has seen, like 3 + 4 = 7, long before sums it has never seen, like 4 + 3. Memorizing comes before understanding.',
  },
  {
    title: 'Watch attention form',
    text: 'Open Look inside → Block 1 before you train, then again after. Attention starts spread evenly over every token and slowly develops favorites.',
  },
  {
    title: 'Memorizing vs. learning',
    text: 'Train on one of the small datasets and watch the two loss curves. When training loss keeps falling but validation loss stalls, the model is memorizing instead of generalizing.',
  },
];

export const StartView: React.FC<StartViewProps> = ({ onNavigate, guidesHidden, onShowGuides }) => (
  <div style={{ maxWidth: 960, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>
    {/* Intro */}
    <div className="glass-panel" style={{ padding: '32px 32px 28px', display: 'flex', gap: 24, alignItems: 'flex-start', flexWrap: 'wrap' }}>
      <div style={{ background: 'var(--primary)', borderRadius: 14, padding: 14, display: 'flex', color: '#ffffff' }}>
        <ChickenLogo size={48} />
      </div>
      <div style={{ flex: 1, minWidth: 280 }}>
        <h2 style={{ fontSize: '1.6rem', fontWeight: 700, marginBottom: 8 }}>Raise your own language model</h2>
        <p style={{ color: 'var(--text-muted)', lineHeight: 1.65 }}>
          LLMStead builds a tiny transformer, the same kind of model behind ChatGPT and Claude, and trains it from scratch
          right here in your browser. It's small enough to finish learning in minutes, and every step is open to inspect, so
          you can watch how a language model goes from random numbers to making real predictions. Nothing you do here
          leaves your computer.
        </p>
      </div>
    </div>

    {/* The path */}
    <div>
      <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 12 }}>How it works: three steps, left to right</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 14 }}>
        {STEPS.map(s => (
          <button
            key={s.step}
            onClick={() => onNavigate(s.tab)}
            className="glass-panel"
            style={{ padding: 18, textAlign: 'left', cursor: 'pointer', color: 'var(--text-main)', display: 'flex', flexDirection: 'column', gap: 8 }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span
                className="font-mono"
                style={{ width: 24, height: 24, borderRadius: 12, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', fontWeight: 700, background: 'var(--primary)', color: '#ffffff' }}
              >
                {s.step}
              </span>
              <span style={{ fontWeight: 700 }}>{s.title}</span>
            </div>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>{s.text}</span>
          </button>
        ))}
      </div>
    </div>

    {/* Expectations */}
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

    {/* Experiments */}
    <div className="glass-panel" style={{ padding: 24 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
        <FlaskConical size={18} color="var(--accent-amber)" />
        <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>Not sure what to try? A few first experiments</h3>
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
        Start: set up your model <ArrowRight size={16} />
      </button>
      {guidesHidden && (
        <button className="btn-secondary" onClick={onShowGuides}>
          Show the step guides again
        </button>
      )}
    </div>
  </div>
);
