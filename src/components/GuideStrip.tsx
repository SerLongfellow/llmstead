import React from 'react';
import { ArrowRight, Lightbulb, X } from 'lucide-react';

interface GuideStripProps {
  step: number;
  title: string;
  children: React.ReactNode;      // one or two sentences: what to do on this page
  next?: { label: string; onClick: () => void };
  onHide: () => void;
}

/** Slim "what do I do here?" banner shown at the top of each step. */
export const GuideStrip: React.FC<GuideStripProps> = ({ step, title, children, next, onHide }) => (
  <div
    style={{
      display: 'flex',
      alignItems: 'center',
      gap: 14,
      padding: '12px 16px',
      marginBottom: 20,
      borderRadius: 10,
      background: 'var(--primary-soft)',
      border: '1px solid color-mix(in srgb, var(--primary) 35%, transparent)',
    }}
  >
    <Lightbulb size={18} color="var(--primary)" style={{ flexShrink: 0 }} />
    <div style={{ flex: 1, fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
      <span style={{ fontWeight: 700, color: 'var(--text-main)', marginRight: 8 }}>
        Step {step}: {title}
      </span>
      {children}
    </div>
    {next && (
      <button className="btn-primary" onClick={next.onClick} style={{ flexShrink: 0, padding: '6px 12px', fontSize: '0.8rem' }}>
        {next.label} <ArrowRight size={14} />
      </button>
    )}
    <button
      onClick={onHide}
      title="Hide the step guides (you can turn them back on from Start here)"
      style={{ background: 'none', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', display: 'flex', padding: 4, flexShrink: 0 }}
    >
      <X size={16} />
    </button>
  </div>
);
