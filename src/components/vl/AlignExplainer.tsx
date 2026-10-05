import React from 'react';
import { ArrowDown, Flame, Lightbulb, Snowflake, X } from 'lucide-react';
import { VlmEvalResult } from '../../engine/vl/vlmTraining';

interface AlignExplainerProps {
  /** The tests taken during phase 1 (the first is its starting baseline), to point at the reader's own numbers */
  phase1Evals: VlmEvalResult[];
}

const pct = (v: number | null) => (v === null ? '–' : `${Math.round(v * 100)}%`);

const Node: React.FC<{ title: string; sub: string; tone: 'frozen' | 'learns' | 'flow' | 'none' }> = ({ title, sub, tone }) => {
  const color = tone === 'learns' ? 'var(--accent-amber)' : tone === 'frozen' ? 'var(--accent-cyan)' : tone === 'none' ? 'var(--border-strong)' : 'var(--accent-purple)';
  return (
    <div style={{ padding: '6px 10px', borderRadius: 8, border: `1.5px solid ${color}`, background: 'var(--bg-card)', opacity: tone === 'none' ? 0.6 : 1 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontWeight: 700, fontSize: '0.8rem', color: 'var(--text-main)' }}>
        {tone === 'learns' && <Flame size={13} color={color} />}
        {tone === 'frozen' && <Snowflake size={13} color={color} />}
        {tone === 'none' && <X size={13} color={color} />}
        {title}
      </div>
      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', lineHeight: 1.4, marginTop: 2 }}>{sub}</div>
    </div>
  );
};

const Down = () => <ArrowDown size={14} color="var(--text-dim)" style={{ alignSelf: 'center' }} />;

/** What phase 1 ("align the projector") actually does, step by step, with the gradient's path. Sits inside the phase 1 card. */
export const AlignExplainer: React.FC<AlignExplainerProps> = ({ phase1Evals }) => {
  // The first phase-1 test is taken before any phase-1 training; quote only a later one
  const lastPhase1 = phase1Evals.length > 1 ? phase1Evals[phase1Evals.length - 1] : null;
  return (
  <details style={{ borderRadius: 8, background: 'var(--bg-card)', border: '1px solid var(--border-color)', padding: '8px 12px' }}>
    <summary style={{ cursor: 'pointer', fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-main)' }}>
      <Lightbulb size={14} color="var(--accent-amber)" style={{ verticalAlign: '-2px', margin: '0 6px 0 2px' }} />
      What does aligning actually do?
    </summary>

    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 12, fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
      <p>
        The <b style={{ color: 'var(--text-main)' }}>projector</b> is a small translator: the vision encoder and the language model
        were trained separately, so a patch vector from one means nothing to the other, even though both are 32 numbers. Phase 1
        trains that translator while everything around it stays fixed. One step:
      </p>
      <ol style={{ margin: 0, paddingLeft: 20, display: 'flex', flexDirection: 'column', gap: 4 }}>
        <li>A picture and its description, say "a red circle at the top left".</li>
        <li>The frozen vision encoder makes 9 patch vectors; the projector turns them into 9 image tokens.</li>
        <li>The frozen language model reads [image tokens] "describe the picture :" and tries to predict the description word by word.</li>
        <li>Its mistakes give a loss.</li>
        <li>
          Backpropagation sends the error back <i>through</i> the language model. Its weights are frozen, so they don't change, but
          the gradient passes through them and reaches the image tokens as "these should have been a bit different to make 'red'
          more likely".
        </li>
        <li>That gradient continues into the projector, and only the projector's weights are updated.</li>
      </ol>

      <div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          <Node title="Loss" sub="mistakes on the description" tone="flow" />
          <Down />
          <Node title="Language model" sub="frozen: passes the gradient through, doesn't change" tone="frozen" />
          <Down />
          <Node title="Image tokens" sub='"be more like this"' tone="flow" />
          <Down />
          <Node title="Projector" sub="the only part that learns" tone="learns" />
          <Down />
          <Node title="Vision encoder" sub="frozen, gets nothing" tone="none" />
        </div>
        <p style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: 4 }}>The gradient's path in phase 1, backwards from the loss.</p>
      </div>

      <p>
        <b style={{ color: 'var(--text-main)' }}>Why "align":</b> the language model is a fixed reader with fixed ideas of what pushes it
        toward saying "red" or "circle". The only way to lower the loss is to change what it's fed, so the projector is pushed until a
        red patch's image token lands where the reader already associates redness. Image tokens end up lined up with word
        embeddings. Think of a reader who only speaks English and can't be taught anything new, and an interpreter who looks at the
        photo and whispers to them: you score the interpreter on whether the reader describes the photo correctly, so the
        interpreter has to learn the reader's English.
      </p>
      <p>
        <b style={{ color: 'var(--text-main)' }}>Why only descriptions:</b> a description states every fact at once (colour, shape,
        place), so it's the densest picture-to-words lesson for the translator.
      </p>
      <p>
        <b style={{ color: 'var(--text-main)' }}>Why it doesn't teach answering questions:</b> in phase 0 the language model learned to
        answer "what colour is the shape ?" by guessing a common colour, never looking at the image slots, which were empty. Phase 1
        can make the image tokens mean "red", but that frozen habit still ignores them. Phase 2 unfreezes the language model so it
        can learn to use the image tokens for every kind of question.
        {lastPhase1 ? (
          <>
            {' '}In your phase 1 so far (step #{lastPhase1.step}): descriptions are right {pct(lastPhase1.tasks.describe.seen)} of
            the time with the picture ({pct(lastPhase1.tasks.describe.seenBlank)} with a blank one), while "what colour?" is at{' '}
            {pct(lastPhase1.tasks.colour.seen)} ({pct(lastPhase1.tasks.colour.seenBlank)} blank).
          </>
        ) : (
          <> Train phase 1 for a while and your own numbers for this will appear here.</>
        )}
      </p>
    </div>
  </details>
  );
};
