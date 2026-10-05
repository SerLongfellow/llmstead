import React from 'react';
import { Plus, RotateCcw, Trash2 } from 'lucide-react';
import { BPETokenizer } from '../../engine/bpeTokenizer';
import { buildExample } from '../../engine/posttrain/examples';
import { SftExample } from '../../engine/posttrain/sft';
import { PreferencePair } from '../../engine/posttrain/dpo';
import { ExampleTokens, exampleWarnings } from './ExampleTokens';

const fieldStyle: React.CSSProperties = { width: '100%', resize: 'vertical', fontSize: '0.8rem', lineHeight: 1.4, padding: '6px 8px' };

/** A small textarea for one field; newlines are real characters in the data */
const Field: React.FC<{ label: string; value: string; onChange: (v: string) => void; color?: string }> = ({ label, value, onChange, color }) => (
  <label style={{ display: 'flex', flexDirection: 'column', gap: 2, flex: '1 1 160px', minWidth: 0 }}>
    <span style={{ fontSize: '0.7rem', fontWeight: 600, color: color ?? 'var(--text-dim)' }}>{label}</span>
    <textarea
      className="font-mono"
      value={value}
      spellCheck={false}
      rows={Math.min(4, Math.max(1, value.split('\n').length))}
      onChange={e => onChange(e.target.value)}
      style={fieldStyle}
    />
  </label>
);

const Warnings: React.FC<{ items: string[] }> = ({ items }) =>
  items.length ? <p style={{ fontSize: '0.7rem', color: 'var(--accent-amber)', marginTop: 4 }}>{items.join(' · ')}</p> : null;

const RowShell: React.FC<{ children: React.ReactNode; onDelete: () => void }> = ({ children, onDelete }) => (
  <div style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--surface-inset)', border: '1px solid var(--border-color)', position: 'relative' }}>
    <button
      onClick={onDelete}
      aria-label="Remove"
      title="Remove"
      style={{ position: 'absolute', top: 6, right: 6, background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-dim)', display: 'flex', padding: 2 }}
    >
      <Trash2 size={14} />
    </button>
    {children}
  </div>
);

const EditorFooter: React.FC<{ onAdd: () => void; onReset: () => void; addLabel: string; disabled: boolean }> = ({ onAdd, onReset, addLabel, disabled }) => (
  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
    <button className="btn-secondary" onClick={onAdd} disabled={disabled} style={{ padding: '4px 10px', fontSize: '0.8rem' }}>
      <Plus size={14} /> {addLabel}
    </button>
    <button className="btn-secondary" onClick={onReset} disabled={disabled} style={{ padding: '4px 10px', fontSize: '0.8rem' }}>
      <RotateCcw size={14} /> Back to the starter set
    </button>
  </div>
);

export const SftEditor: React.FC<{
  examples: SftExample[];
  onChange: (examples: SftExample[]) => void;
  onReset: () => void;
  tokenizer: BPETokenizer;
  contextWindow: number;
  eos: boolean;
  maskPrompt: boolean;
  disabled: boolean;
}> = ({ examples, onChange, onReset, tokenizer, contextWindow, eos, maskPrompt, disabled }) => {
  const set = (i: number, patch: Partial<SftExample>) => onChange(examples.map((e, j) => (j === i ? { ...e, ...patch } : e)));
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 420, overflowY: 'auto', paddingRight: 4 }}>
        {examples.map((e, i) => {
          const ex = buildExample(tokenizer, e.prompt, e.response, { contextWindow, eos });
          return (
            <RowShell key={i} onDelete={() => onChange(examples.filter((_, j) => j !== i))}>
              <fieldset disabled={disabled} style={{ border: 'none', padding: 0, margin: 0, display: 'flex', gap: 8, flexWrap: 'wrap', paddingRight: 20 }}>
                <Field label="Prompt" value={e.prompt} onChange={prompt => set(i, { prompt })} />
                <Field label="Response" value={e.response} onChange={response => set(i, { response })} color="var(--accent-emerald)" />
              </fieldset>
              <div style={{ marginTop: 6 }}>
                <ExampleTokens example={ex} tokenizer={tokenizer} maskPrompt={maskPrompt} />
              </div>
              <Warnings items={exampleWarnings(ex)} />
            </RowShell>
          );
        })}
      </div>
      <EditorFooter onAdd={() => onChange([...examples, { prompt: '', response: '' }])} onReset={onReset} addLabel="Add an example" disabled={disabled} />
    </div>
  );
};

export const DpoEditor: React.FC<{
  pairs: PreferencePair[];
  onChange: (pairs: PreferencePair[]) => void;
  onReset: () => void;
  tokenizer: BPETokenizer;
  contextWindow: number;
  disabled: boolean;
}> = ({ pairs, onChange, onReset, tokenizer, contextWindow, disabled }) => {
  const set = (i: number, patch: Partial<PreferencePair>) => onChange(pairs.map((p, j) => (j === i ? { ...p, ...patch } : p)));
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 420, overflowY: 'auto', paddingRight: 4 }}>
        {pairs.map((p, i) => {
          const chosen = buildExample(tokenizer, p.prompt, p.chosen, { contextWindow });
          const rejected = buildExample(tokenizer, p.prompt, p.rejected, { contextWindow });
          return (
            <RowShell key={i} onDelete={() => onChange(pairs.filter((_, j) => j !== i))}>
              <fieldset disabled={disabled} style={{ border: 'none', padding: 0, margin: 0, display: 'flex', gap: 8, flexWrap: 'wrap', paddingRight: 20 }}>
                <Field label="Prompt" value={p.prompt} onChange={prompt => set(i, { prompt })} />
                <Field label="Chosen (preferred)" value={p.chosen} onChange={c => set(i, { chosen: c })} color="var(--accent-emerald)" />
                <Field label="Rejected" value={p.rejected} onChange={r => set(i, { rejected: r })} color="var(--accent-rose)" />
              </fieldset>
              <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 4 }}>
                <ExampleTokens example={chosen} tokenizer={tokenizer} />
                <ExampleTokens example={rejected} tokenizer={tokenizer} tint="color-mix(in srgb, var(--accent-rose) 25%, transparent)" />
              </div>
              <Warnings items={[...new Set([...exampleWarnings(chosen), ...exampleWarnings(rejected)])]} />
            </RowShell>
          );
        })}
      </div>
      <EditorFooter onAdd={() => onChange([...pairs, { prompt: '', chosen: '', rejected: '' }])} onReset={onReset} addLabel="Add a pair" disabled={disabled} />
    </div>
  );
};
