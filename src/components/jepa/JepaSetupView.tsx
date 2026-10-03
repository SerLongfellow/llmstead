import React, { useMemo, useState } from 'react';
import { Shuffle } from 'lucide-react';
import { seededRandom } from '../../engine/datasets';
import { sampleMask } from '../../engine/jepa/jepa';
import { ShapeImage } from '../../engine/jepa/shapes';
import { InfoTooltip } from '../InfoTooltip';
import { ShapeCanvas } from './ShapeCanvas';
import { GRID, JepaSettings, PATCH_SIZE, describe } from './jepaSettings';

interface JepaSetupViewProps {
  settings: JepaSettings;
  onChange: (next: JepaSettings) => void;
  paramCount: number;
  sampleImages: ShapeImage[];
}

/** A row of buttons for picking one value */
function Choice<T extends string | number>({ value, options, onPick, format }: {
  value: T;
  options: readonly T[];
  onPick: (v: T) => void;
  format?: (v: T) => string;
}) {
  return (
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
      {options.map(o => (
        <button
          key={String(o)}
          className={o === value ? 'btn-primary' : 'btn-secondary'}
          onClick={() => onPick(o)}
          style={{ padding: '5px 12px', fontSize: '0.8rem' }}
        >
          {format ? format(o) : String(o)}
        </button>
      ))}
    </div>
  );
}

function Setting({ label, info, children }: { label: string; info?: { description: string; impact: string }; children: React.ReactNode }) {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
        <span style={{ fontSize: '0.8rem', fontWeight: 600 }}>{label}</span>
        {info && <InfoTooltip title={label} description={info.description} impact={info.impact} />}
      </div>
      {children}
    </div>
  );
}

function Section({ n, title, note, children }: { n: number; title: string; note?: string; children: React.ReactNode }) {
  return (
    <div className="glass-panel" style={{ padding: '20px 24px' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 14, flexWrap: 'wrap' }}>
        <span className="font-mono" style={{ color: 'var(--primary)', fontWeight: 700 }}>{n}</span>
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>{title}</h3>
        {note && <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>{note}</span>}
      </div>
      {children}
    </div>
  );
}

export const JepaSetupView: React.FC<JepaSetupViewProps> = ({ settings, onChange, paramCount, sampleImages }) => {
  const set = (patch: Partial<JepaSettings>) => onChange({ ...settings, ...patch });
  const example = sampleImages[0];

  // A few example masks for the current number of hidden blocks; Shuffle draws new ones
  const [maskSeed, setMaskSeed] = useState(1);
  const masks = useMemo(() => {
    const rand = seededRandom(maskSeed);
    return Array.from({ length: 4 }, () => sampleMask(rand, GRID, settings.numTargets));
  }, [maskSeed, settings.numTargets]);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr)', gap: 20, maxWidth: 1100, margin: '0 auto' }}>
      <Section n={1} title="Images" note="changing these keeps the model's weights">
        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: 14 }}>
          Every training step draws brand-new images: one shape (circle, square, triangle or cross) in one of four colours, at a
          random size and position, on a dark, slightly noisy background. There are no labels during training. The shape and
          colour names below are only used later, to test what the model learned.
        </p>
        <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-start', marginBottom: 16 }}>
          <Setting
            label="Shape size"
            info={{
              description: 'Small shapes are 6–10 pixels across; big ones 9–13, so more patches show part of the outline.',
              impact: 'Bigger shapes make shape easier to read off, even from raw pixels, which leaves less room to see what training adds.',
            }}
          >
            <Choice value={settings.shapes} options={['small', 'big'] as const} onPick={shapes => set({ shapes })} />
          </Setting>
        </div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {sampleImages.map((img, i) => (
            <figure key={i} style={{ textAlign: 'center' }}>
              <ShapeCanvas img={img} scale={4} />
              <figcaption style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: 4 }}>{describe(img)}</figcaption>
            </figure>
          ))}
        </div>
        {example && (
          <div style={{ display: 'flex', gap: 20, alignItems: 'center', flexWrap: 'wrap', marginTop: 20, paddingTop: 16, borderTop: '1px solid var(--border-color)' }}>
            <ShapeCanvas img={example} scale={10} patchSize={PATCH_SIZE} />
            <p style={{ flex: '1 1 280px', fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              <b style={{ color: 'var(--text-main)' }}>Patches are the tokens.</b> Each 16×16 image is cut into a {GRID}×{GRID} grid
              of {PATCH_SIZE}×{PATCH_SIZE}-pixel patches. A patch's {PATCH_SIZE * PATCH_SIZE} pixels × 3 colour channels make{' '}
              {PATCH_SIZE * PATCH_SIZE * 3} numbers, and a learned matrix turns those into the patch's first embedding, the way
              the text model looks up a token's embedding. A position embedding then says where in the grid the patch sits.
            </p>
          </div>
        )}
      </Section>

      <Section n={2} title="Masking" note="changing this keeps the model's weights">
        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: 14 }}>
          Each training image gets a fresh mask: a few rectangular blocks of patches are hidden (the <i>targets</i>, outlined in
          amber) and everything else is the <i>context</i>. The model sees the context and has to predict the targets' embeddings.
        </p>
        <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start', flexWrap: 'wrap' }}>
          <Setting
            label="Hidden blocks per image"
            info={{
              description: 'How many rectangles (1×1 to 2×2 patches each, possibly overlapping) to hide. At least half the image always stays visible.',
              impact: 'More hidden blocks make the task harder, since there is less context to predict from.',
            }}
          >
            <Choice value={settings.numTargets} options={[1, 2, 3, 4] as const} onPick={numTargets => set({ numTargets })} />
          </Setting>
          {example && (
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
              {masks.map((m, i) => (
                <ShapeCanvas key={i} img={example} scale={6} patchSize={PATCH_SIZE} mask={m} hidden="cover" />
              ))}
              <button className="btn-secondary" onClick={() => setMaskSeed(s => s + 1)} title="Draw new example masks" style={{ padding: '6px 10px' }}>
                <Shuffle size={14} />
              </button>
            </div>
          )}
        </div>
      </Section>

      <Section n={3} title="Model size" note="changing any of these starts the model over">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 18, marginBottom: 16 }}>
          <Setting
            label="Embedding size"
            info={{
              description: 'How many numbers describe each patch, in all three networks. The text model\'s d_model.',
              impact: 'Bigger embeddings can hold more, but cost more per step and need more training.',
            }}
          >
            <Choice value={settings.dModel} options={[16, 32, 48, 64] as const} onPick={dModel => set({ dModel })} />
          </Setting>
          <Setting
            label="Encoder layers"
            info={{ description: 'Transformer blocks in the context and target encoders (the same block as the text model, without the causal mask).', impact: 'More layers can combine information from more patches, at a higher cost per step.' }}
          >
            <Choice value={settings.numLayers} options={[1, 2, 3] as const} onPick={numLayers => set({ numLayers })} />
          </Setting>
          <Setting
            label="Attention heads"
            info={{ description: 'Each head is an independent attention pattern over the patches.', impact: 'More heads let a layer look at several things at once, each with fewer numbers.' }}
          >
            <Choice value={settings.numHeads} options={[2, 4] as const} onPick={numHeads => set({ numHeads })} />
          </Setting>
          <Setting
            label="Predictor width"
            info={{ description: 'The predictor works in its own, narrower space (I-JEPA uses a narrow predictor too), then maps its guesses back to embedding size.', impact: 'A predictor that is too strong can make the task too easy for the encoders to learn much.' }}
          >
            <Choice value={settings.predDim} options={[8, 16, 32] as const} onPick={predDim => set({ predDim })} />
          </Setting>
          <Setting label="Predictor layers">
            <Choice value={settings.predLayers} options={[1, 2] as const} onPick={predLayers => set({ predLayers })} />
          </Setting>
          <Setting
            label="Images per step"
            info={{ description: 'How many images each training step averages its gradient over (the batch size). Safe to change at any time.', impact: 'Bigger batches give smoother updates but each step takes longer. The anti-collapse term measures spread across this batch.' }}
          >
            <Choice value={settings.batchSize} options={[4, 8, 16] as const} onPick={batchSize => set({ batchSize })} />
          </Setting>
        </div>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          <span className="font-mono" style={{ color: 'var(--text-main)', fontWeight: 700 }}>{paramCount.toLocaleString()}</span> trainable
          parameters in the context encoder and predictor. The target encoder has another copy of the encoder's weights, but they
          are never trained directly: they follow the context encoder (EMA).
        </p>
      </Section>
    </div>
  );
};
