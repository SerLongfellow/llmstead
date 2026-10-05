import React from 'react';
import { ArrowRight, Flame, Snowflake } from 'lucide-react';
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
