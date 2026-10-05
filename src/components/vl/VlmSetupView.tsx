import React, { useMemo } from 'react';
import { AlertTriangle, Cpu, Eye, MessageSquareText, RefreshCw } from 'lucide-react';
import { seededRandom } from '../../engine/datasets';
import { comboName } from '../../engine/vl/captions';
import { Pair } from '../../engine/vl/clipData';
import { MicroVlm } from '../../engine/vl/vlm';
import { VQA_TASKS, questionFor } from '../../engine/vl/vlmData';
import { InfoTooltip } from '../InfoTooltip';
import { ImageCanvas } from './ImageCanvas';
import type { VisionCopy } from './VisionWorkbench';
import { VlmPipeline } from './VlmPipeline';
import { VlmSettings } from './visionSettings';

interface VlmSetupViewProps {
  settings: VlmSettings;
  onChange: (s: VlmSettings) => void;
  vlm: MicroVlm;
  visionCopy: VisionCopy;   // the encoder in use (a CLIP copy, or the untrained one)
  clipSteps: number;        // where the live CLIP is now
  clipReady: boolean;       // whether the copy has trained enough to be worth using
  onRefreshCopy: () => void;
  onTrainClip: () => void;
  sample: Pair;
}

const Section: React.FC<{ n: number; title: string; icon: React.ReactNode; children: React.ReactNode }> = ({ n, title, icon, children }) => (
  <div className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 14 }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span className="font-mono" style={{ width: 24, height: 24, borderRadius: 12, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', fontWeight: 700, background: 'var(--primary)', color: '#ffffff' }}>
        {n}
      </span>
      {icon}
      <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>{title}</h3>
    </div>
    {children}
  </div>
);

const Choice = <T extends string | number>({ options, value, onPick }: { options: { id: T; label: string }[]; value: T; onPick: (v: T) => void }) => (
  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
    {options.map(o => (
      <button key={String(o.id)} className={o.id === value ? 'btn-primary' : 'btn-secondary'} style={{ padding: '5px 12px', fontSize: '0.8rem' }} onClick={() => onPick(o.id)}>
        {o.label}
      </button>
    ))}
  </div>
);

export const VlmSetupView: React.FC<VlmSetupViewProps> = ({
  settings, onChange, vlm, visionCopy, clipSteps, clipReady, onRefreshCopy, onTrainClip, sample,
}) => {
  const set = (patch: Partial<VlmSettings>) => onChange({ ...settings, ...patch });
  const params = vlm.getParameterCount();
  // One example of every question, about the sample picture
  const examples = useMemo(() => {
    const rand = seededRandom(9);
    return VQA_TASKS.map(t => ({ ...t, ...questionFor(t.id, sample.image.labels, rand, visionCopy.data.heldOut) }));
  }, [sample, visionCopy.data.heldOut]);
  const toggleTask = (id: (typeof VQA_TASKS)[number]['id']) => {
    const has = settings.tasks.includes(id);
    if (has && settings.tasks.length === 1) return; // keep at least one
    set({ tasks: has ? settings.tasks.filter(t => t !== id) : VQA_TASKS.map(t => t.id).filter(t => t === id || settings.tasks.includes(t)) });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1100, margin: '0 auto' }}>
      <div className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 14 }}>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          A vision-language model (VLM) is three parts in a row. The picture never reaches the language model as pixels: the vision
          encoder turns it into one vector per patch, and the projector turns each of those into an <b style={{ color: 'var(--text-main)' }}>image
          token</b>, a vector the same size as a word embedding. The language model reads them exactly where it would read words.
        </p>
        <VlmPipeline phase={null} params={params} />
      </div>

      <Section n={1} title="The eyes: which vision encoder" icon={<Eye size={18} color="var(--accent-cyan)" />}>
        <Choice
          options={[
            { id: 'clip', label: `Your CLIP (copy from step #${settings.encoder === 'clip' ? visionCopy.step.toLocaleString() : '…'})` },
            { id: 'random', label: 'An untrained encoder (to compare)' },
          ]}
          value={settings.encoder}
          onPick={encoder => set({ encoder })}
        />
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          LLaVA uses OpenAI's CLIP image tower the same way: trained first, on its own, then frozen. The VLM uses a{' '}
          <i>copy</i> of yours, so training the CLIP further doesn't change the VLM's eyes while it learns. Pictures are cut into{' '}
          {visionCopy.clip.config.patchSize}-pixel patches ({vlm.numPatches} image tokens)
          {visionCopy.data.heldOut.length ? <>, and {visionCopy.data.heldOut.map(comboName).join(' and ')} are never shown to either model</> : null}.
        </p>
        {settings.encoder === 'clip' && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <button className="btn-secondary" onClick={onRefreshCopy} disabled={clipSteps === visionCopy.step} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
              <RefreshCw size={14} /> Take a new copy (your CLIP is at step #{clipSteps.toLocaleString()})
            </button>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>A new copy starts the VLM over.</span>
          </div>
        )}
        {settings.encoder === 'clip' && !clipReady && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', padding: '10px 14px', borderRadius: 8, background: 'var(--amber-soft)', border: '1px solid var(--amber-tint)' }}>
            <AlertTriangle size={16} color="var(--accent-amber)" />
            <span style={{ flex: '1 1 260px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              This copy has barely trained, so the VLM would be looking through blurry eyes. Train your CLIP for a couple of minutes
              (a few thousand steps), then come back and take a new copy.
            </span>
            <button className="btn-primary" onClick={onTrainClip} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
              Train the CLIP now
            </button>
          </div>
        )}
      </Section>

      <Section n={2} title="The questions it learns to answer" icon={<MessageSquareText size={18} color="var(--accent-emerald)" />}>
        <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', alignItems: 'flex-start' }}>
          <ImageCanvas image={sample.image} size={120} />
          <div style={{ flex: '1 1 360px', display: 'flex', flexDirection: 'column', gap: 6 }}>
            {examples.map(ex => {
              const on = settings.tasks.includes(ex.id);
              return (
                <label key={ex.id} style={{ display: 'grid', gridTemplateColumns: '20px 90px 1fr', gap: 8, alignItems: 'center', cursor: 'pointer', opacity: on ? 1 : 0.5 }}>
                  <input type="checkbox" checked={on} onChange={() => toggleTask(ex.id)} />
                  <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>{ex.label}</span>
                  <span className="font-mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    {ex.question} <span style={{ color: 'var(--accent-emerald)' }}>{ex.answer}</span>
                  </span>
                </label>
              );
            })}
          </div>
        </div>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)', lineHeight: 1.6 }}>
          Each training example is [image tokens] + question + answer. Only the answer's words count toward the loss: it learns to
          answer, not to ask. The test always asks every kind, so you can see what an unchosen kind looks like. Note that your CLIP's
          captions may never have mentioned size: can a frozen encoder still tell big from small?
        </p>
      </Section>

      <Section n={3} title="Language model size" icon={<Cpu size={18} color="var(--accent-amber)" />}>
        <div style={{ display: 'flex', gap: 28, flexWrap: 'wrap' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Width (d_model)</label>
              <InfoTooltip
                title="Width"
                description="The size of every word embedding, and so of every image token the projector makes."
                impact="Changing it starts the VLM over."
              />
            </div>
            <Choice options={[16, 32, 48].map(d => ({ id: d, label: String(d) }))} value={settings.dModel} onPick={dModel => set({ dModel })} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
              <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>Layers</label>
              <InfoTooltip
                title="Layers"
                description="Transformer blocks in the language model. It's the same GPT as on the text side, with causal attention: each word may look at the image tokens and the words before it."
                impact="More layers can combine more, and each step is slower. Changing it starts the VLM over."
              />
            </div>
            <Choice options={[1, 2, 3].map(d => ({ id: d, label: String(d) }))} value={settings.numLayers} onPick={numLayers => set({ numLayers })} />
          </div>
        </div>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
          Projector {params.projector.toLocaleString()} + language model {params.lm.toLocaleString()} parameters learn; the vision
          encoder's {params.vision.toLocaleString()} stay frozen. LLaVA joined a CLIP image tower to a 7–13 billion parameter language
          model.
        </p>
      </Section>
    </div>
  );
};
