import React from 'react';
import { BookOpen, Cpu, Grid2x2 } from 'lucide-react';
import { CAPTION_DETAILS, DEFAULT_HELD_OUT, VOCAB, comboName } from '../../engine/vl/captions';
import { IMAGE_SIZE, Pair } from '../../engine/vl/clipData';
import { InfoTooltip } from '../InfoTooltip';
import { ImageCanvas } from './ImageCanvas';
import { PATCH_SIZES, VisionSettings } from './visionSettings';

interface VisionSetupViewProps {
  settings: VisionSettings;
  onChange: (s: VisionSettings) => void;
  paramCount: number;
  samples: Pair[];
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

const Choice = <T extends string | number | boolean>({ options, value, onPick }: { options: { id: T; label: string }[]; value: T; onPick: (v: T) => void }) => (
  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
    {options.map(o => (
      <button key={String(o.id)} className={o.id === value ? 'btn-primary' : 'btn-secondary'} style={{ padding: '5px 12px', fontSize: '0.8rem' }} onClick={() => onPick(o.id)}>
        {o.label}
      </button>
    ))}
  </div>
);

const Row: React.FC<{ label: string; tip: { title: string; description: string; impact: string }; children: React.ReactNode }> = ({ label, tip, children }) => (
  <div>
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
      <label style={{ fontSize: '0.85rem', fontWeight: 600 }}>{label}</label>
      <InfoTooltip {...tip} />
    </div>
    {children}
  </div>
);

export const VisionSetupView: React.FC<VisionSetupViewProps> = ({ settings, onChange, paramCount, samples }) => {
  const set = (patch: Partial<VisionSettings>) => onChange({ ...settings, ...patch });
  const grid = IMAGE_SIZE / settings.patchSize;
  const showcase = samples[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20, maxWidth: 1100, margin: '0 auto' }}>
      <Section n={1} title="Data: pictures and their captions" icon={<BookOpen size={18} color="var(--accent-purple)" />}>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
          Every picture is drawn on the spot: one of 4 shapes in one of 4 colours, small or big, in one of 9 places, on a noisy
          dark background, 24 × 24 pixels. Its caption is written from what was drawn, in one of two phrasings. There is no fixed
          dataset: every training step sees brand-new pictures.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 12 }}>
          {samples.map((p, i) => (
            <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'center', padding: 10, borderRadius: 8, background: 'var(--surface-inset)', border: '1px solid var(--border-color)' }}>
              <ImageCanvas image={p.image} size={72} />
              <span className="font-mono" style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textAlign: 'center' }}>{p.caption}</span>
            </div>
          ))}
        </div>

        <Row
          label="How much the captions say"
          tip={{
            title: 'Caption detail',
            description: 'What the captions mention. The model can only learn the facts its captions state: if no caption ever says where the shape is, nothing pushes it to notice.',
            impact: "More detail means more possible captions (16, 144 or 288) and more to learn. It also changes what's needed to tell a batch apart, which changes what gets learned first.",
          }}
        >
          <Choice
            options={CAPTION_DETAILS.map(d => ({ id: d.id, label: `${d.label}: "${d.example}"` }))}
            value={settings.detail}
            onPick={detail => set({ detail })}
          />
        </Row>

        <Row
          label="Hold out two combinations"
          tip={{
            title: 'Held-out combinations',
            description: `When on, training never shows a ${DEFAULT_HELD_OUT.map(comboName).join(' or a ')}: not the picture, not the caption. It does see green and triangles, yellow and crosses, just never together.`,
            impact: "The test scores report these separately. Getting them right means the model learned 'green' and 'triangle' as separate ideas it can combine, rather than memorizing each pair.",
          }}
        >
          <Choice
            options={[{ id: true, label: `On: never show ${DEFAULT_HELD_OUT.map(comboName).join(', ')}` }, { id: false, label: 'Off: show everything' }]}
            value={settings.heldOut}
            onPick={heldOut => set({ heldOut })}
          />
        </Row>

        <div>
          <p style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: 6 }}>The text tower's vocabulary: {VOCAB.length} words</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {VOCAB.map(w => (
              <span key={w} className="font-mono" style={{ fontSize: '0.75rem', padding: '2px 8px', borderRadius: 6, background: 'var(--surface-inset)', border: '1px solid var(--border-color)', color: 'var(--accent-emerald)' }}>
                {w}
              </span>
            ))}
          </div>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: 6 }}>
            Each word is one token. (The text model uses BPE, which splits text into pieces; with so few words, whole words keep
            every view readable.)
          </p>
        </div>
      </Section>

      <Section n={2} title="Patches: how the image tower reads a picture" icon={<Grid2x2 size={18} color="var(--accent-cyan)" />}>
        <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'center' }}>
          {showcase && <ImageCanvas image={showcase.image} size={192} patchSize={settings.patchSize} />}
          <div style={{ flex: '1 1 320px', display: 'flex', flexDirection: 'column', gap: 14 }}>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              A transformer reads a sequence of vectors. For text those are tokens; for a picture they're <b>patches</b>: the picture
              is cut into a {grid} × {grid} grid, and each {settings.patchSize} × {settings.patchSize}-pixel square becomes one
              vector of {settings.patchSize * settings.patchSize * 3} numbers (red, green and blue for each pixel). This is how
              Vision Transformers (ViT) work, the image half of CLIP included.
            </p>
            <Row
              label="Patch size"
              tip={{
                title: 'Patch size',
                description: 'Pixels per patch side. Smaller patches mean more of them (a longer sequence) and each one sees less of the picture.',
                impact: "At 8 pixels a patch is exactly one of the 9 places a shape can sit, so a shape mostly lands in one patch. At 6 pixels shapes get cut across patches and shape takes longer to learn; at 12 there are only 4 patches.",
              }}
            >
              <Choice
                options={PATCH_SIZES.map(p => ({ id: p, label: `${p} px → ${(IMAGE_SIZE / p) ** 2} patches` }))}
                value={settings.patchSize}
                onPick={patchSize => set({ patchSize })}
              />
            </Row>
          </div>
        </div>
      </Section>

      <Section n={3} title="Model size" icon={<Cpu size={18} color="var(--accent-amber)" />}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 18 }}>
          <Row
            label="Width (d_model)"
            tip={{
              title: 'Width',
              description: "The length of every vector inside both towers. Both towers then project down to a shared 16-number space where pictures and captions are compared.",
              impact: 'Wider holds more detail but each step is slower.',
            }}
          >
            <Choice options={[16, 32, 48].map(d => ({ id: d, label: String(d) }))} value={settings.dModel} onPick={dModel => set({ dModel })} />
          </Row>
          <Row
            label="Image tower layers"
            tip={{
              title: 'Image tower layers',
              description: 'Transformer blocks the patches go through. In each, every patch can look at every other patch (attention), so the tower can combine edges in different patches into a shape.',
              impact: 'More layers can learn more complex shapes, and slow each step down.',
            }}
          >
            <Choice options={[1, 2, 3].map(d => ({ id: d, label: String(d) }))} value={settings.imageLayers} onPick={imageLayers => set({ imageLayers })} />
          </Row>
          <Row
            label="Text tower layers"
            tip={{
              title: 'Text tower layers',
              description: "Transformer blocks the caption's words go through. Unlike the text model, nothing is masked: every word sees every other word.",
              impact: 'The captions are short and regular, so one layer is enough; more mostly costs time.',
            }}
          >
            <Choice options={[1, 2].map(d => ({ id: d, label: String(d) }))} value={settings.textLayers} onPick={textLayers => set({ textLayers })} />
          </Row>
        </div>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
          {paramCount.toLocaleString()} parameters. OpenAI's released CLIP models have from about 100 million to several hundred million.
        </p>
      </Section>
    </div>
  );
};
