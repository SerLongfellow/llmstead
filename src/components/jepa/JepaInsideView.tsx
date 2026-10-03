import React, { useMemo, useState } from 'react';
import { ArrowRight, Eye, RefreshCw, Shuffle, Users } from 'lucide-react';
import { MicroJepa, sampleMask } from '../../engine/jepa/jepa';
import { imageFeatures } from '../../engine/jepa/probes';
import { ShapeImage, patchify, randomShapeImage } from '../../engine/jepa/shapes';
import { ShapeCanvas } from './ShapeCanvas';
import { GRID, IMAGE_SIZE, JepaSettings, PATCH_SIZE, describe, shapeSizes } from './jepaSettings';

interface JepaInsideViewProps {
  model: MicroJepa;
  untrained: MicroJepa;
  settings: JepaSettings;
  gallery: ShapeImage[]; // labelled images to search for nearest neighbours
}

const NEIGHBOURS = 8;

const cosine = (a: number[], b: number[]) => {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na * nb) || 1);
};

/** Just one patch of an image, as its own tiny image */
function cropPatch(img: ShapeImage, patch: number): ShapeImage {
  const px = (patch % GRID) * PATCH_SIZE;
  const py = Math.floor(patch / GRID) * PATCH_SIZE;
  const pixels: number[] = [];
  for (let y = 0; y < PATCH_SIZE; y++) {
    for (let x = 0; x < PATCH_SIZE; x++) {
      const i = ((py + y) * img.size + px + x) * 3;
      pixels.push(img.pixels[i], img.pixels[i + 1], img.pixels[i + 2]);
    }
  }
  return { size: PATCH_SIZE, pixels, labels: img.labels };
}

/** An embedding as a strip of cells: blue for positive, red for negative */
const VectorStrip: React.FC<{ values: number[]; label: string }> = ({ values, label }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
    <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', width: 56, flexShrink: 0 }}>{label}</span>
    <div style={{ display: 'flex', gap: 1, flex: 1 }} title={values.map(v => v.toFixed(2)).join(', ')}>
      {values.map((v, i) => {
        const t = Math.min(1, Math.abs(v) / 2);
        const c = v >= 0 ? 'var(--primary)' : 'var(--accent-rose)';
        return <span key={i} style={{ flex: 1, height: 14, minWidth: 3, background: `color-mix(in srgb, ${c} ${Math.round(t * 100)}%, var(--surface-inset))` }} />;
      })}
    </div>
  </div>
);

export const JepaInsideView: React.FC<JepaInsideViewProps> = ({ model, untrained, settings, gallery }) => {
  const sizes = shapeSizes(settings);
  const [img, setImg] = useState<ShapeImage>(() => randomShapeImage(Math.random, IMAGE_SIZE, sizes));
  const [mask, setMask] = useState(() => sampleMask(Math.random, GRID, settings.numTargets));
  const [showHidden, setShowHidden] = useState(false);
  // Training changes the weights in place, so Refresh re-runs everything on the current weights
  const [refresh, setRefresh] = useState(0);
  const [selected, setSelected] = useState<number | null>(null);
  const [layer, setLayer] = useState<number | null>(null);

  const ins = useMemo(() => model.inspect(patchify(img, PATCH_SIZE), mask), [model, img, mask, refresh]);

  const newImage = () => {
    setImg(randomShapeImage(Math.random, IMAGE_SIZE, sizes));
    setSelected(null);
  };
  const newMask = () => {
    setMask(sampleMask(Math.random, GRID, settings.numTargets));
    setSelected(null);
  };

  // Attention from the clicked patch, averaged over heads. A visible patch shows where the context
  // encoder looks from it; a hidden patch shows where the predictor looked to guess it.
  const isHidden = selected !== null && mask.targets.includes(selected);
  const layers = selected === null ? [] : isHidden ? ins.predictor.layers : ins.context.layers;
  const shownLayer = layer !== null && layer < layers.length ? layer : layers.length - 1;
  const heat = useMemo(() => {
    if (selected === null || layers.length === 0) return undefined;
    const order = isHidden ? [...mask.context, ...mask.targets] : mask.context;
    const row = isHidden ? mask.context.length + mask.targets.indexOf(selected) : mask.context.indexOf(selected);
    const heads = layers[shownLayer].attentionWeights;
    const weights = order.map((_, col) => heads.reduce((s, h) => s + h[row][col], 0) / heads.length);
    const max = Math.max(...weights);
    const out: (number | null)[] = new Array(GRID * GRID).fill(null);
    order.forEach((p, col) => { out[p] = weights[col] / (max || 1); });
    return out;
  }, [selected, isHidden, layers, shownLayer, mask]);

  // Nearest neighbours by embedding (each image's patch embeddings averaged), trained vs untrained
  const neighbours = useMemo(() => {
    const find = (m: MicroJepa) => {
      const [q] = imageFeatures(m, [img]);
      const feats = imageFeatures(m, gallery);
      return feats
        .map((f, i) => ({ img: gallery[i], sim: cosine(q, f) }))
        .sort((a, b) => b.sim - a.sim)
        .slice(0, NEIGHBOURS);
    };
    return { trained: find(model), untrained: find(untrained) };
  }, [model, untrained, img, gallery, refresh]);

  const avgLoss = ins.loss;
  const cosines = ins.predictor.preds.map((p, j) => cosine(p, ins.targets[j]));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Controls */}
      <div className="glass-panel" style={{ padding: '14px 20px', display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
        <span style={{ fontWeight: 700, marginRight: 6 }}>{describe(img)}</span>
        <button className="btn-secondary" onClick={newImage}><Shuffle size={15} /> New image</button>
        <button className="btn-secondary" onClick={newMask}><Shuffle size={15} /> New mask</button>
        <button className="btn-secondary" onClick={() => setShowHidden(s => !s)}>
          <Eye size={15} /> {showHidden ? 'Cover hidden patches' : 'Peek at hidden patches'}
        </button>
        <button className="btn-secondary" onClick={() => setRefresh(r => r + 1)} title="Re-run with the latest weights (training keeps changing them)">
          <RefreshCw size={15} /> Refresh (step #{model.steps})
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 420px), 1fr))', gap: 20 }}>
        {/* The image */}
        <div className="glass-panel" style={{ padding: '18px 22px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 6 }}>The image, masked</h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.5 }}>
            {mask.context.length} visible patches go to the context encoder; the {mask.targets.length} outlined in amber are hidden.{' '}
            <b style={{ color: 'var(--text-main)' }}>Click a patch</b> to see where the model looks from it.
          </p>
          <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', alignItems: 'flex-start' }}>
            <ShapeCanvas
              img={img}
              scale={16}
              patchSize={PATCH_SIZE}
              mask={mask}
              hidden={showHidden ? 'dim' : 'cover'}
              heat={heat}
              selectedPatch={selected}
              onPatchClick={p => setSelected(s => (s === p ? null : p))}
            />
            <div style={{ flex: '1 1 160px', fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>
              {selected === null ? (
                <p>Nothing selected. Visible patches show the context encoder's attention; hidden ones show the predictor's.</p>
              ) : (
                <>
                  <p style={{ marginBottom: 8 }}>
                    {isHidden ? (
                      <>Patch {selected} is <b style={{ color: 'var(--accent-amber)' }}>hidden</b>. Cyan shows where the predictor looked to guess its embedding (averaged over heads).</>
                    ) : (
                      <>Patch {selected} is <b style={{ color: 'var(--text-main)' }}>visible</b>. Cyan shows which visible patches the context encoder mixes into it (averaged over heads). It can't see the hidden ones.</>
                    )}
                  </p>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {layers.map((_, l) => (
                      <button
                        key={l}
                        className={l === shownLayer ? 'btn-primary' : 'btn-secondary'}
                        onClick={() => setLayer(l)}
                        style={{ padding: '3px 9px', fontSize: '0.75rem' }}
                      >
                        {isHidden ? 'Predictor' : 'Encoder'} layer {l + 1}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>
        </div>

        {/* The flow */}
        <div className="glass-panel" style={{ padding: '18px 22px' }}>
          <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 12 }}>Through the three networks</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[
              { title: 'Context encoder', color: 'var(--primary)', text: `Embeds the ${mask.context.length} visible patches (${settings.numLayers} layer${settings.numLayers > 1 ? 's' : ''}, ${settings.dModel} numbers each).` },
              { title: 'Predictor', color: 'var(--accent-purple)', text: `Reads those embeddings plus a placeholder "mask token" at each of the ${mask.targets.length} hidden positions, and outputs a guess for each.` },
              { title: 'Target encoder', color: 'var(--accent-amber)', text: `Embeds all ${GRID * GRID} patches with ${settings.recipe.ablation === 'none' ? `the slow EMA copy of the encoder (momentum ${model.momentum().toFixed(4)})` : 'the context encoder itself (the EMA copy is switched off)'}. Its embeddings of the hidden patches are the answers.` },
              { title: 'Loss for this image', color: 'var(--accent-rose)', text: `${avgLoss.toFixed(4)}: the mean squared difference between guesses and answers.` },
            ].map((b, i) => (
              <React.Fragment key={b.title}>
                {i > 0 && <ArrowRight size={14} color="var(--text-dim)" style={{ transform: 'rotate(90deg)', alignSelf: 'center' }} />}
                <div style={{ padding: '8px 12px', borderRadius: 8, background: 'var(--surface-inset)', borderLeft: `3px solid ${b.color}` }}>
                  <span style={{ fontWeight: 700, fontSize: '0.85rem' }}>{b.title}: </span>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{b.text}</span>
                </div>
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>

      {/* Guesses vs answers */}
      <div className="glass-panel" style={{ padding: '18px 22px' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 6 }}>Guess vs answer, for each hidden patch</h3>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 14, lineHeight: 1.5 }}>
          Each strip is one embedding: {settings.dModel} numbers, blue for positive and red for negative. Training pulls the guess
          toward the answer. Similarity is 1 when they point the same way and 0 when unrelated. Note the model never has to
          reproduce the pixels on the left, only their embedding.
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          {mask.targets.map((p, j) => (
            <div key={p} style={{ display: 'flex', gap: 14, alignItems: 'center', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, width: 60 }}>
                <ShapeCanvas img={cropPatch(img, p)} scale={12} />
                <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>patch {p}</span>
              </div>
              <div style={{ flex: '1 1 300px', display: 'flex', flexDirection: 'column', gap: 4 }}>
                <VectorStrip values={ins.predictor.preds[j]} label="Guess" />
                <VectorStrip values={ins.targets[j]} label="Answer" />
              </div>
              <div className="font-mono" style={{ fontSize: '0.8rem', width: 150, color: 'var(--text-muted)' }}>
                similarity <span style={{ color: cosines[j] > 0.8 ? 'var(--accent-emerald)' : cosines[j] > 0.4 ? 'var(--accent-amber)' : 'var(--accent-rose)' }}>{cosines[j].toFixed(2)}</span>
                <br />
                error {(ins.errors[j] / settings.dModel).toFixed(3)}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Neighbours */}
      <div className="glass-panel" style={{ padding: '18px 22px' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 6, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Users size={17} color="var(--accent-cyan)" /> Which images does it think are alike?
        </h3>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: 14, lineHeight: 1.5 }}>
          The {NEIGHBOURS} images (out of {gallery.length}) whose embeddings point most nearly the same way as this one's. What they
          have in common is what the embeddings encode: same colour? same shape? same place in the frame?
        </p>
        {(['trained', 'untrained'] as const).map(which => {
          const list = neighbours[which];
          const sameShape = list.filter(n => n.img.labels.shape === img.labels.shape).length;
          const sameColour = list.filter(n => n.img.labels.colour === img.labels.colour).length;
          return (
            <div key={which} style={{ marginBottom: 14 }}>
              <p style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: 8 }}>
                {which === 'trained' ? `Trained encoder (step #${model.steps})` : 'Untrained encoder'}
                <span style={{ fontWeight: 400, color: 'var(--text-dim)' }}>
                  {' '}· same shape {sameShape}/{NEIGHBOURS} · same colour {sameColour}/{NEIGHBOURS}
                </span>
              </p>
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                <figure style={{ textAlign: 'center', paddingRight: 10, borderRight: '1px solid var(--border-color)' }}>
                  <ShapeCanvas img={img} scale={4} />
                  <figcaption style={{ fontSize: '0.65rem', color: 'var(--text-dim)', marginTop: 3 }}>this image</figcaption>
                </figure>
                {list.map((n, i) => (
                  <figure key={i} style={{ textAlign: 'center' }}>
                    <ShapeCanvas img={n.img} scale={4} title={describe(n.img)} />
                    <figcaption className="font-mono" style={{ fontSize: '0.65rem', color: 'var(--text-dim)', marginTop: 3 }}>
                      {n.sim.toFixed(2)}
                    </figcaption>
                  </figure>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
