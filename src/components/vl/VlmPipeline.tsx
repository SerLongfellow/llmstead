import React from 'react';
import { ArrowRight, Flame, Link2, MessageSquareText, Snowflake } from 'lucide-react';
import { PHASE_TRAINS, VlmPhase } from '../../engine/vl/vlm';

interface VlmPipelineProps {
  /** Highlight what this phase trains; null shows the parts without a training state */
  phase: VlmPhase | null;
  params?: { vision: number; projector: number; lm: number };
}

const Part: React.FC<{ title: string; sub: string; params?: number; state: 'frozen' | 'training' | null; color: string }> = ({ title, sub, params, state, color }) => (
  <div
    style={{
      flex: '1 1 160px', minWidth: 150, padding: '12px 14px', borderRadius: 10, background: 'var(--surface-inset)',
      border: `1.5px solid ${state === 'training' ? 'var(--accent-amber)' : color}`,
      boxShadow: state === 'training' ? '0 0 0 3px var(--amber-soft)' : undefined,
      display: 'flex', flexDirection: 'column', gap: 4,
    }}
  >
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
      <span style={{ fontWeight: 700, fontSize: '0.9rem' }}>{title}</span>
      {state === 'frozen' && (
        <span title="Frozen: not updated in this phase" style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: '0.7rem', color: 'var(--accent-cyan)' }}>
          <Snowflake size={13} /> frozen
        </span>
      )}
      {state === 'training' && (
        <span title="Learning in this phase" style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: '0.7rem', color: 'var(--accent-amber)' }}>
          <Flame size={13} /> learning
        </span>
      )}
    </div>
    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.45 }}>{sub}</span>
    {params !== undefined && <span className="font-mono" style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{params.toLocaleString()} parameters</span>}
  </div>
);

/** The three parts of the VLM, left to right, and which of them learn in a phase */
export const VlmPipeline: React.FC<VlmPipelineProps> = ({ phase, params }) => {
  const trains = phase === null ? null : PHASE_TRAINS[phase];
  const arrow = <ArrowRight size={18} color="var(--text-dim)" style={{ flexShrink: 0, alignSelf: 'center' }} />;
  return (
    <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'stretch' }}>
      <Part
        title="Vision encoder"
        sub="Your CLIP's image tower: picture → one vector per patch. Always frozen."
        params={params?.vision}
        state={trains ? 'frozen' : null}
        color="var(--accent-cyan)"
      />
      {arrow}
      <Part
        title="Projector"
        sub="A small MLP: turns each patch vector into an 'image token', the size of a word embedding."
        params={params?.projector}
        state={trains ? (trains.projector ? 'training' : 'frozen') : null}
        color="var(--accent-purple)"
      />
      {arrow}
      <Part
        title="Language model"
        sub="A GPT: reads [image tokens + question] and writes the answer, one word at a time."
        params={params?.lm}
        state={trains ? (trains.lm ? 'training' : 'frozen') : null}
        color="var(--accent-emerald)"
      />
    </div>
  );
};

export type VisionStage = 'match' | 'describe';

/**
 * Switch between the two models of the Images + Text mode. Deliberately unnumbered (icons and
 * names instead): the navbar's numbered steps (Set up, Train, Look inside) apply to each of them.
 */
export const StageBar: React.FC<{ stage: VisionStage; onChange: (s: VisionStage) => void; training: Record<VisionStage, boolean> }> = ({ stage, onChange, training }) => {
  const stages: { id: VisionStage; icon: React.ReactNode; title: string; sub: string }[] = [
    { id: 'match', icon: <Link2 size={15} />, title: 'Match · CLIP', sub: 'which words fit which picture' },
    { id: 'describe', icon: <MessageSquareText size={15} />, title: 'Describe · VLM', sub: 'answers questions about a picture, built on the CLIP' },
  ];
  return (
    <div role="group" aria-label="Model" style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center', marginBottom: 16 }}>
      <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-dim)', marginRight: 2 }}>Model:</span>
      {stages.map((s, i) => {
        const active = s.id === stage;
        return (
          <React.Fragment key={s.id}>
            {i > 0 && <ArrowRight size={16} color="var(--text-dim)" />}
            <button
              onClick={() => onChange(s.id)}
              aria-pressed={active}
              style={{
                display: 'flex', alignItems: 'center', gap: 10, padding: '8px 14px', borderRadius: 10, cursor: 'pointer', textAlign: 'left',
                border: `1px solid ${active ? 'var(--primary)' : 'var(--border-color)'}`,
                background: active ? 'var(--primary-soft)' : 'var(--bg-card)', color: 'var(--text-main)',
              }}
            >
              <span style={{ display: 'inline-flex', color: active ? 'var(--primary)' : 'var(--text-muted)' }}>{s.icon}</span>
              <span style={{ lineHeight: 1.25 }}>
                <span style={{ display: 'block', fontWeight: 700, fontSize: '0.85rem' }}>{s.title}</span>
                <span style={{ display: 'block', fontSize: '0.7rem', color: 'var(--text-dim)' }}>{s.sub}</span>
              </span>
              {!active && training[s.id] && (
                <span title="Still training in the background" style={{ width: 7, height: 7, borderRadius: 4, background: 'var(--accent-emerald)', animation: 'pulse 1.2s ease-in-out infinite' }} />
              )}
            </button>
          </React.Fragment>
        );
      })}
    </div>
  );
};
