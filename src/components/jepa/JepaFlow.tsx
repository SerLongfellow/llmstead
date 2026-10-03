import React from 'react';
import { ArrowRight } from 'lucide-react';

/** Text for each box; Look inside fills these with the current image's numbers */
export interface JepaFlowText {
  visible: string;
  whole: string;
  context: string;
  predictor: string;
  target: string;
  loss: string;
  ema: string;
}

export const DEFAULT_FLOW_TEXT: JepaFlowText = {
  visible: 'Only the visible patches',
  whole: 'The whole image',
  context: 'Turns each visible patch into an embedding.',
  predictor: 'From those embeddings and the hidden positions, guesses each hidden patch\'s embedding.',
  target: 'Embeds every patch. Its embeddings of the hidden patches are the answers.',
  loss: 'How far each guess is from its answer. Training shrinks it.',
  ema: 'weights copied slowly (EMA), never trained by the loss',
};

const Box: React.FC<{ title: string; text: string; color: string; fill?: boolean }> = ({ title, text, color, fill = true }) => (
  <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--surface-inset)', borderTop: `3px solid ${color}`, height: fill ? '100%' : undefined, width: '100%' }}>
    <p style={{ fontWeight: 700, fontSize: '0.85rem', marginBottom: 3 }}>{title}</p>
    <p style={{ fontSize: '0.775rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>{text}</p>
  </div>
);

const Input: React.FC<{ text: string }> = ({ text }) => (
  <div style={{ fontSize: '0.775rem', color: 'var(--text-muted)', fontWeight: 600, textAlign: 'center', padding: '0 4px' }}>{text}</div>
);

const Arrow: React.FC<{ long?: boolean }> = ({ long }) => (
  <div style={{ display: 'flex', alignItems: 'center', color: 'var(--text-dim)', width: '100%' }}>
    {long && <div style={{ flex: 1, height: 2, background: 'var(--border-strong)', marginRight: -4 }} />}
    <ArrowRight size={16} style={{ flexShrink: 0, margin: long ? 0 : '0 auto' }} />
  </div>
);

/**
 * The I-JEPA training setup as two parallel branches that only meet at the loss:
 *
 *   visible patches → context encoder → predictor ─→ guesses ┐
 *                          ┆ EMA (weights, not data)          ├→ loss
 *   whole image     → target encoder ──────────────→ answers ┘
 */
export const JepaFlow: React.FC<{ text?: Partial<JepaFlowText> }> = ({ text }) => {
  const t = { ...DEFAULT_FLOW_TEXT, ...text };
  return (
    <div style={{ overflowX: 'auto' }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '90px 28px minmax(130px, 1fr) 28px minmax(150px, 1.2fr) 28px minmax(130px, 1fr)',
          gridTemplateRows: 'auto 26px auto',
          alignItems: 'center',
          rowGap: 4,
          minWidth: 640,
        }}
      >
        {/* Top branch: what the model sees */}
        <div style={{ gridColumn: 1, gridRow: 1 }}><Input text={t.visible} /></div>
        <div style={{ gridColumn: 2, gridRow: 1 }}><Arrow /></div>
        <div style={{ gridColumn: 3, gridRow: 1, alignSelf: 'stretch' }}><Box title="Context encoder" text={t.context} color="var(--primary)" /></div>
        <div style={{ gridColumn: 4, gridRow: 1 }}><Arrow /></div>
        <div style={{ gridColumn: 5, gridRow: 1, alignSelf: 'stretch' }}><Box title="Predictor" text={t.predictor} color="var(--accent-purple)" /></div>
        <div style={{ gridColumn: 6, gridRow: 1 }}>
          <Arrow />
          <p style={{ fontSize: '0.65rem', color: 'var(--text-dim)', textAlign: 'center' }}>guesses</p>
        </div>
        <div style={{ gridColumn: 7, gridRow: '1 / 4', alignSelf: 'stretch', display: 'flex', alignItems: 'center' }}>
          <Box title="Loss" text={t.loss} color="var(--accent-rose)" fill={false} />
        </div>

        {/* The only link between the encoders: a weight copy, not data */}
        <div
          style={{
            gridColumn: 3, gridRow: 2, alignSelf: 'stretch', display: 'flex', alignItems: 'center', gap: 6,
            borderLeft: '2px dashed var(--border-strong)', marginLeft: 20, paddingLeft: 8,
          }}
        >
          <span style={{ fontSize: '0.65rem', color: 'var(--text-dim)', lineHeight: 1.2 }}>{t.ema}</span>
        </div>

        {/* Bottom branch: where the answers come from */}
        <div style={{ gridColumn: 1, gridRow: 3 }}><Input text={t.whole} /></div>
        <div style={{ gridColumn: 2, gridRow: 3 }}><Arrow /></div>
        <div style={{ gridColumn: 3, gridRow: 3 }}><Box title="Target encoder" text={t.target} color="var(--accent-amber)" /></div>
        <div style={{ gridColumn: '4 / 7', gridRow: 3 }}>
          <Arrow long />
          <p style={{ fontSize: '0.65rem', color: 'var(--text-dim)', textAlign: 'center' }}>answers for the hidden patches</p>
        </div>
      </div>
    </div>
  );
};
