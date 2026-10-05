import React, { useMemo, useState } from 'react';
import { Eye, Map as MapIcon, Search, Type } from 'lucide-react';
import { pca2 } from '../../engine/pca';
import { VOCAB, captionFor, isHeldOut, tokenizeCaption } from '../../engine/vl/captions';
import { MicroClip } from '../../engine/vl/clip';
import { ClipData, Pair, captionBank } from '../../engine/vl/clipData';
import { COLOURS, SHAPES, ShapeLabels } from '../../engine/vl/shapes';
import { THEME } from '../../styles/theme';
import { AttentionGrid } from '../AttentionGrid';
import { InfoTooltip } from '../InfoTooltip';
import { PromptInput } from '../PromptInput';
import { ImageCanvas } from './ImageCanvas';

interface VisionInsideViewProps {
  model: MicroClip;
  untrained: MicroClip;
  data: ClipData;
  gallery: Pair[];
  weightsVersion: number; // changes whenever the trained weights do
}

const dot = (a: number[], b: number[]) => a.reduce((s, v, k) => s + v * b[k], 0);

/** The colour each shape is drawn in (for the map and chips) */
const SHAPE_RGB: Record<string, string> = { red: '#e63333', green: '#33cc4d', blue: '#4066f2', yellow: '#f2d933' };

const Panel: React.FC<{ icon: React.ReactNode; title: string; tip?: { title: string; description: string; impact: string }; children: React.ReactNode; wide?: boolean }> = ({
  icon, title, tip, children, wide,
}) => (
  <div className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 14, gridColumn: wide ? '1 / -1' : undefined, minWidth: 0 }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      {icon}
      <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>{title}</h3>
      {tip && <InfoTooltip {...tip} />}
    </div>
    {children}
  </div>
);

const Toggle = <T extends string | number>({ options, value, onPick }: { options: { id: T; label: string }[]; value: T; onPick: (v: T) => void }) => (
  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
    {options.map(o => (
      <button key={String(o.id)} className={o.id === value ? 'btn-primary' : 'btn-secondary'} style={{ padding: '3px 10px', fontSize: '0.75rem' }} onClick={() => onPick(o.id)}>
        {o.label}
      </button>
    ))}
  </div>
);

export const VisionInsideView: React.FC<VisionInsideViewProps> = ({ model, untrained, data, gallery, weightsVersion }) => {
  const [which, setWhich] = useState<'trained' | 'untrained'>('trained');
  const current = which === 'trained' ? model : untrained;
  const [query, setQuery] = useState('a red circle');
  const [chosen, setChosen] = useState<number | null>(null); // null: follow the top search result
  const [overlay, setOverlay] = useState<'match' | 'attention' | 'none'>('match');
  const [fromPatch, setFromPatch] = useState<number | null>(null);
  const [imgLayer, setImgLayer] = useState(0);
  const [txtLayer, setTxtLayer] = useState(0);
  const [txtHead, setTxtHead] = useState(0);

  const tokens = useMemo(() => tokenizeCaption(query), [query]);
  // Every picture's embedding, and the query's (recomputed when the weights change)
  const imageEmbeds = useMemo(
    () => gallery.map(p => current.embedImage(p.example.patches)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [current, gallery, weightsVersion]
  );
  const queryEmbed = useMemo(
    () => current.embedText(tokens.ids),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [current, tokens, weightsVersion]
  );
  const ranked = useMemo(
    () => (queryEmbed ? gallery.map((p, i) => ({ i, p, sim: dot(imageEmbeds[i], queryEmbed) })).sort((a, b) => b.sim - a.sim) : []),
    [gallery, imageEmbeds, queryEmbed]
  );

  const selected = chosen ?? ranked[0]?.i ?? 0;
  const setSelected = (i: number) => setChosen(i);
  const pick = gallery[selected] ?? gallery[0];
  const picture = useMemo(
    () => current.inspectImage(pick.example.patches, queryEmbed),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [current, pick, queryEmbed, weightsVersion]
  );
  const bank = useMemo(
    () => captionBank(current, data.detail),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [current, data.detail, weightsVersion]
  );
  const topCaptions = useMemo(
    () => bank.map(c => ({ text: c.text, sim: dot(picture.embedding, c.embedding) })).sort((a, b) => b.sim - a.sim).slice(0, 5),
    [bank, picture]
  );
  const textInspect = useMemo(
    () => (tokens.ids.length ? current.inspectText(tokens.ids) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [current, tokens, weightsVersion]
  );

  // The tint over the selected picture: where the query matches, or one patch's attention
  const grid = Math.sqrt(current.numPatches);
  const tint = useMemo(() => {
    if (overlay === 'match' && picture.patchMatch) {
      const v = picture.patchMatch;
      const lo = Math.min(...v);
      const hi = Math.max(...v);
      return v.map(x => (hi > lo ? (x - lo) / (hi - lo) : 0));
    }
    if (overlay === 'attention' && fromPatch !== null) {
      const layer = picture.layers[Math.min(imgLayer, picture.layers.length - 1)];
      const heads = layer.attentionWeights;
      const avg = heads[0][fromPatch].map((_, k) => heads.reduce((s, h) => s + h[fromPatch][k], 0) / heads.length);
      const hi = Math.max(...avg);
      return avg.map(x => x / (hi || 1));
    }
    return null;
  }, [overlay, picture, fromPatch, imgLayer]);

  // ── The shared space: pictures and plain colour + shape captions on one 2D map ──
  const mapCaptions = useMemo(() => {
    const out: { labels: ShapeLabels; text: string }[] = [];
    for (let colour = 0; colour < COLOURS.length; colour++)
      for (let shape = 0; shape < SHAPES.length; shape++) {
        const labels = { shape, colour, size: 0, row: 1, col: 1 };
        out.push({ labels, text: captionFor(labels, 'shape') });
      }
    return out;
  }, []);
  const map = useMemo(() => {
    const texts = mapCaptions.map(c => current.embedText(tokenizeCaption(c.text).ids)!);
    const rows = [...imageEmbeds, ...texts];
    const { mean, components } = pca2(rows);
    const project = (v: number[]) => components.map(c => dot(c.dir, v.map((x, k) => x - mean[k])));
    const pts = rows.map(project);
    const xs = pts.map(p => p[0]);
    const ys = pts.map(p => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const W = 640, H = 380, pad = 40;
    const sx = (x: number) => pad + ((x - x0) / (x1 - x0 || 1)) * (W - 2 * pad);
    const sy = (y: number) => pad + ((y - y0) / (y1 - y0 || 1)) * (H - 2 * pad);
    return {
      W, H,
      images: pts.slice(0, imageEmbeds.length).map(([x, y], i) => ({ x: sx(x), y: sy(y), pair: gallery[i], i })),
      texts: pts.slice(imageEmbeds.length).map(([x, y], i) => ({ x: sx(x), y: sy(y), ...mapCaptions[i] })),
      share: components[0].share + components[1].share,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current, imageEmbeds, mapCaptions]);

  const heldOutBadge = (p: Pair) =>
    isHeldOut(p.image.labels, data.heldOut) ? (
      <span className="badge badge-amber" style={{ fontSize: '0.6rem', padding: '1px 5px' }} title="A colour + shape combination training never showed">
        never seen
      </span>
    ) : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="glass-panel" style={{ padding: '12px 20px', display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Looking at:</span>
        <Toggle
          options={[{ id: 'trained', label: `Your model (step #${model.steps})` }, { id: 'untrained', label: 'The same model before training' }]}
          value={which}
          onPick={setWhich}
        />
        <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>Everything below updates as training runs.</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 460px), 1fr))', gap: 20 }}>
        {/* Search */}
        <Panel
          icon={<Search size={20} color="var(--primary)" />}
          title="Search the pictures with words"
          tip={{
            title: 'Text-to-image search',
            description: 'Your words go through the text tower to become one vector. Every picture already has a vector from the image tower. The pictures are sorted by how similar their vector is to yours (cosine similarity).',
            impact: 'This is how image search with CLIP works, at a much larger scale. Try words the captions never combine, like "a green triangle", or leave words out ("triangle").',
          }}
        >
          <PromptInput value={query} onChange={setQuery} placeholder="a blue square at the top left" />
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {VOCAB.map(w => (
              <button
                key={w}
                className="font-mono"
                onClick={() => setQuery(q => `${q.trim()} ${w}`.trim())}
                style={{ fontSize: '0.7rem', padding: '1px 7px', borderRadius: 6, cursor: 'pointer', background: 'var(--surface-inset)', border: '1px solid var(--border-color)', color: 'var(--accent-emerald)' }}
              >
                {w}
              </button>
            ))}
          </div>
          {tokens.unknown.length > 0 && (
            <p style={{ fontSize: '0.75rem', color: 'var(--accent-amber)' }}>
              Not in the vocabulary, so skipped: {tokens.unknown.map(w => `"${w}"`).join(', ')}
            </p>
          )}
          {ranked.length === 0 ? (
            <p style={{ fontSize: '0.85rem', color: 'var(--text-dim)' }}>Type at least one word from the vocabulary.</p>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(84px, 1fr))', gap: 8 }}>
              {ranked.slice(0, 12).map((r, k) => (
                <button
                  key={r.i}
                  onClick={() => setSelected(r.i)}
                  title={r.p.caption}
                  style={{
                    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, padding: 6, borderRadius: 8, cursor: 'pointer',
                    background: 'var(--surface-inset)', border: `1px solid ${r.i === selected ? 'var(--accent-cyan)' : 'var(--border-color)'}`, color: 'var(--text-main)',
                  }}
                >
                  <ImageCanvas image={r.p.image} size={64} />
                  <span className="font-mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    #{k + 1} · {r.sim.toFixed(2)}
                  </span>
                  {heldOutBadge(r.p)}
                </button>
              ))}
            </div>
          )}
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
            The best 12 of {gallery.length} test pictures, with their similarity to your words. Click one to look at it up close.
          </p>
        </Panel>

        {/* One picture */}
        <Panel
          icon={<Eye size={20} color="var(--accent-cyan)" />}
          title="One picture up close"
          tip={{
            title: 'Inside the image tower',
            description: '"Where your words match" takes each patch\'s own output vector, projects it like the whole picture\'s, and compares it with your words. "Attention" shows, for the patch you click, how much it looks at each other patch (averaged over the heads).',
            impact: 'The match map is a rough guide: training only ever compares the average of all patches with a caption, so a patch can light up for reasons that are hard to read.',
          }}
        >
          <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', alignItems: 'flex-start' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
              <ImageCanvas
                image={pick.image}
                size={200}
                patchSize={data.patchSize}
                overlay={tint}
                overlayColor={overlay === 'attention' ? THEME.cyan : THEME.amber}
                selectedPatch={overlay === 'attention' ? fromPatch : null}
                onPatchClick={p => {
                  setOverlay('attention');
                  setFromPatch(p);
                }}
              />
              <span className="font-mono" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{pick.caption}</span>
              {heldOutBadge(pick)}
            </div>
            <div style={{ flex: '1 1 200px', display: 'flex', flexDirection: 'column', gap: 10, minWidth: 0 }}>
              <Toggle
                options={[{ id: 'match', label: 'Where your words match' }, { id: 'attention', label: 'Attention' }, { id: 'none', label: 'Plain' }]}
                value={overlay}
                onPick={setOverlay}
              />
              {overlay === 'attention' && (
                <>
                  <Toggle options={picture.layers.map((_, l) => ({ id: l, label: `Layer ${l + 1}` }))} value={Math.min(imgLayer, picture.layers.length - 1)} onPick={setImgLayer} />
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                    {fromPatch === null
                      ? 'Click a patch to see which patches it looks at.'
                      : `Patch ${fromPatch + 1} of ${grid * grid} (outlined) attends to the tinted patches; brighter = more attention.`}
                  </p>
                </>
              )}
              {overlay === 'match' && (
                <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                  {queryEmbed ? <>Brighter patches match "{tokens.words.join(' ')}" more.</> : 'Type some words on the left first.'}
                </p>
              )}
              <div>
                <p style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: 6 }}>The captions it likes best for this picture</p>
                {topCaptions.map(c => (
                  <div key={c.text} style={{ display: 'grid', gridTemplateColumns: '1fr 44px', gap: 8, alignItems: 'center', marginBottom: 4 }}>
                    <div style={{ position: 'relative', padding: '3px 8px', borderRadius: 5, background: 'var(--surface-inset)', overflow: 'hidden' }}>
                      <div style={{ position: 'absolute', inset: 0, width: `${Math.max(0, c.sim) * 100}%`, background: c.text === captionFor(pick.image.labels, data.detail) ? 'var(--emerald-tint)' : 'var(--primary-soft)' }} />
                      <span className="font-mono" style={{ position: 'relative', fontSize: '0.72rem' }}>{c.text}</span>
                    </div>
                    <span className="font-mono" style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textAlign: 'right' }}>{c.sim.toFixed(2)}</span>
                  </div>
                ))}
                <p style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>Out of all {bank.length} possible captions. Green = the true one.</p>
              </div>
            </div>
          </div>
        </Panel>

        {/* Text tower */}
        <Panel
          wide
          icon={<Type size={20} color="var(--accent-emerald)" />}
          title="Inside the text tower: your words' attention"
          tip={{
            title: 'Text tower attention',
            description: 'For each word (row), how much it looks at every word (column) in one head of one layer. The tower then averages all the words\' outputs into one vector for the caption.',
            impact: 'Unlike the text model, nothing is masked: a word may look at words after it, because this tower never has to predict the next word.',
          }}
        >
          {textInspect ? (
            <>
              <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
                <Toggle options={textInspect.layers.map((_, l) => ({ id: l, label: `Layer ${l + 1}` }))} value={Math.min(txtLayer, textInspect.layers.length - 1)} onPick={setTxtLayer} />
                <Toggle
                  options={textInspect.layers[0].attentionWeights.map((_, h) => ({ id: h, label: `Head ${h + 1}` }))}
                  value={Math.min(txtHead, textInspect.layers[0].attentionWeights.length - 1)}
                  onPick={setTxtHead}
                />
              </div>
              <AttentionGrid
                causal={false}
                attentionMap={textInspect.layers[Math.min(txtLayer, textInspect.layers.length - 1)].attentionWeights[Math.min(txtHead, textInspect.layers[0].attentionWeights.length - 1)]}
                tokenStrings={tokens.words}
              />
            </>
          ) : (
            <p style={{ fontSize: '0.85rem', color: 'var(--text-dim)' }}>Type some words in the search box to see them here.</p>
          )}
        </Panel>

        {/* Shared space */}
        <Panel
          wide
          icon={<MapIcon size={20} color="var(--accent-purple)" />}
          title="One space for pictures and words"
          tip={{
            title: 'The shared embedding space',
            description: `Every test picture (a small shape) and the 16 plain captions like "a red circle" (text), squashed from 16 dimensions down to the 2 that keep the most spread (PCA). These 2 keep ${Math.round(map.share * 100)}% of it, so some closeness is lost.`,
            impact: "After training, pictures cluster by what their captions say, and each caption moves toward its pictures. Real CLIP models also keep pictures and captions in two separate regions, called the modality gap; you may see it here too.",
          }}
        >
          <svg viewBox={`0 0 ${map.W} ${map.H}`} style={{ width: '100%', maxHeight: 440, background: 'var(--surface-inset)', borderRadius: 10, border: '1px solid var(--border-color)' }}>
            {map.images.map(({ x, y, pair, i }) => (
              <g key={i} onClick={() => setSelected(i)} style={{ cursor: 'pointer' }}>
                <title>{pair.caption}</title>
                <Glyph shape={pair.image.labels.shape} x={x} y={y} r={6} fill={SHAPE_RGB[COLOURS[pair.image.labels.colour]]} ring={isHeldOut(pair.image.labels, data.heldOut)} selected={i === selected} />
              </g>
            ))}
            {map.texts.map(t => (
              <g key={t.text}>
                <circle cx={t.x} cy={t.y} r={2.5} fill={THEME.textMain} />
                <text x={t.x + 5} y={t.y + 4} fontSize={11} fontFamily="JetBrains Mono, monospace" fill={SHAPE_RGB[COLOURS[t.labels.colour]]} stroke={THEME.bg} strokeWidth={3} paintOrder="stroke">
                  {t.text.replace(/^a /, '')}
                </text>
              </g>
            ))}
          </svg>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
            Shapes are pictures (ringed = a never-seen combination; click one to look at it above). Text labels are captions.
          </p>
        </Panel>
      </div>
    </div>
  );
};

/** A small marker in the picture's own shape and colour */
const Glyph: React.FC<{ shape: number; x: number; y: number; r: number; fill: string; ring: boolean; selected: boolean }> = ({ shape, x, y, r, fill, ring, selected }) => {
  const stroke = selected ? THEME.cyan : ring ? THEME.amber : 'none';
  const sw = selected || ring ? 2 : 0;
  switch (SHAPES[shape]) {
    case 'circle':
      return <circle cx={x} cy={y} r={r} fill={fill} stroke={stroke} strokeWidth={sw} />;
    case 'square':
      return <rect x={x - r * 0.85} y={y - r * 0.85} width={r * 1.7} height={r * 1.7} fill={fill} stroke={stroke} strokeWidth={sw} />;
    case 'triangle':
      return <polygon points={`${x},${y - r} ${x + r},${y + r} ${x - r},${y + r}`} fill={fill} stroke={stroke} strokeWidth={sw} />;
    default: {
      const a = r * 0.35;
      const d = `M${x - a},${y - r}h${2 * a}v${r - a}h${r - a}v${2 * a}h${a - r}v${r - a}h${-2 * a}v${a - r}h${a - r}v${-2 * a}h${r - a}z`;
      return <path d={d} fill={fill} stroke={stroke} strokeWidth={sw} />;
    }
  }
};
