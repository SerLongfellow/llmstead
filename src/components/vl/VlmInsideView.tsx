import React, { useMemo, useState } from 'react';
import { Eye, MessageSquareText, Sparkles } from 'lucide-react';
import { seededRandom } from '../../engine/datasets';
import { isHeldOut } from '../../engine/vl/captions';
import { IMAGE_SIZE, Pair } from '../../engine/vl/clipData';
import { COLOURS, SHAPES, ShapeLabels, drawBlankImage, patchify } from '../../engine/vl/shapes';
import { MicroVlm } from '../../engine/vl/vlm';
import { END, VLM_VOCAB, VQA_TASKS, questionFor, tokenizeVlm } from '../../engine/vl/vlmData';
import { VlmDataSettings } from '../../engine/vl/vlmTraining';
import { THEME } from '../../styles/theme';
import { AttentionGrid } from '../AttentionGrid';
import { InfoTooltip } from '../InfoTooltip';
import { PromptInput } from '../PromptInput';
import { ImageCanvas } from './ImageCanvas';

interface VlmInsideViewProps {
  vlm: MicroVlm;
  untrained: MicroVlm;
  data: VlmDataSettings;
  gallery: Pair[];
  weightsVersion: number;
}

const PRESETS = ['describe the picture :', 'what colour is the shape ?', 'what shape is it ?', 'where is the shape ?', 'is the shape big ?', 'is there a red circle ?'];
const BLANK = -1;

const Toggle = <T extends string | number>({ options, value, onPick }: { options: { id: T; label: string }[]; value: T; onPick: (v: T) => void }) => (
  <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
    {options.map(o => (
      <button key={String(o.id)} className={o.id === value ? 'btn-primary' : 'btn-secondary'} style={{ padding: '3px 10px', fontSize: '0.75rem' }} onClick={() => onPick(o.id)}>
        {o.label}
      </button>
    ))}
  </div>
);

/** The right answer to a question about a picture, if the question is one of the kinds trained on */
function rightAnswer(question: string, labels: ShapeLabels): string | null {
  const q = tokenizeVlm(question).words.join(' ');
  const there = /^is there a (\w+) (\w+) \?$/.exec(q);
  if (there) {
    const colour = COLOURS.indexOf(there[1] as never);
    const shape = SHAPES.indexOf(there[2] as never);
    if (colour < 0 || shape < 0) return null;
    return colour === labels.colour && shape === labels.shape ? 'yes' : 'no';
  }
  for (const t of VQA_TASKS) {
    if (t.id === 'there') continue;
    const qa = questionFor(t.id, labels, Math.random, []);
    if (tokenizeVlm(qa.question).words.join(' ') === q) return qa.answer;
  }
  return null;
}

export const VlmInsideView: React.FC<VlmInsideViewProps> = ({ vlm, untrained, data, gallery, weightsVersion }) => {
  const [which, setWhich] = useState<'trained' | 'untrained'>('trained');
  const current = which === 'trained' ? vlm : untrained;
  const [picked, setPicked] = useState(0);
  const [question, setQuestion] = useState(PRESETS[1]);
  const [stepIdx, setStepIdx] = useState(0);
  const [layer, setLayer] = useState(0);
  const [showGrid, setShowGrid] = useState(false);
  const [gridHead, setGridHead] = useState(0);
  const [showWords, setShowWords] = useState(false);

  const blank = useMemo(() => drawBlankImage(seededRandom(4003), IMAGE_SIZE), []);
  const image = picked === BLANK ? blank : gallery[picked].image;
  const patches = useMemo(() => patchify(image, data.patchSize), [image, data.patchSize]);
  const tokens = useMemo(
    () => current.imageTokens(current.features(patches)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [current, patches, weightsVersion]
  );
  const q = useMemo(() => tokenizeVlm(question), [question]);
  const answer = useMemo(
    () => current.answer(tokens, q.ids),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [current, tokens, q, weightsVersion]
  );
  const right = picked === BLANK ? null : rightAnswer(question, image.labels);
  const said = answer.words.join(' ');

  // The answer's words, plus <end> if it chose to stop; each was chosen in one step
  const chips = [...answer.words, ...(answer.steps.length > answer.words.length ? [END] : [])];
  const k = Math.min(stepIdx, Math.max(0, answer.steps.length - 1));
  const step = answer.steps[k];
  const P = current.numPatches;
  const L = step ? Math.min(layer, step.layerInspections.length - 1) : 0;

  // Where the step's last position (the one choosing the next word) looks, averaged over heads
  const look = useMemo(() => {
    if (!step) return null;
    const heads = step.layerInspections[L].attentionWeights;
    const last = step.tokens.length - 1;
    const avg = heads[0][last].map((_, c) => heads.reduce((s, h) => s + h[last][c], 0) / heads.length);
    const imagePart = avg.slice(0, P);
    const max = Math.max(...imagePart) || 1;
    return {
      overlay: imagePart.map(v => v / max),
      imageShare: imagePart.reduce((s, v) => s + v, 0),
      words: avg.slice(P).map((v, i) => ({ word: step.tokenStrings[P + i], v })),
      top: step.probabilities[last]
        .map((p, id) => ({ word: VLM_VOCAB[id], p }))
        .sort((a, b) => b.p - a.p)
        .slice(0, 5),
    };
  }, [step, L, P]);

  const readsLike = useMemo(
    () => (showWords ? tokens.map(t => current.nearestWords(t, 1)[0]) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [showWords, tokens, current, weightsVersion]
  );

  const grid = Math.sqrt(P);
  const heldOutBadge = (p: Pair) =>
    isHeldOut(p.image.labels, data.heldOut) ? (
      <span className="badge badge-amber" style={{ fontSize: '0.55rem', padding: '0 4px' }} title="A colour + shape combination neither model was trained on">
        never seen
      </span>
    ) : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="glass-panel" style={{ padding: '12px 20px', display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Looking at:</span>
        <Toggle options={[{ id: 'trained', label: `Your VLM (step #${vlm.steps})` }, { id: 'untrained', label: 'The same VLM before training' }]} value={which} onPick={setWhich} />
        <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>Everything below updates as training runs.</span>
      </div>

      {/* Picture + question + answer */}
      <div className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <MessageSquareText size={20} color="var(--accent-emerald)" />
          <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Ask about a picture</h3>
        </div>
        <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4 }}>
          <button
            onClick={() => setPicked(BLANK)}
            title="A blank picture: just the background"
            style={{ flexShrink: 0, padding: 3, borderRadius: 7, cursor: 'pointer', background: 'var(--surface-inset)', border: `2px solid ${picked === BLANK ? 'var(--accent-cyan)' : 'var(--border-color)'}`, color: 'var(--text-dim)', fontSize: '0.6rem' }}
          >
            <ImageCanvas image={blank} size={44} />
            blank
          </button>
          {gallery.slice(0, 40).map((p, i) => (
            <button
              key={i}
              onClick={() => setPicked(i)}
              title={p.caption}
              style={{ flexShrink: 0, padding: 3, borderRadius: 7, cursor: 'pointer', background: 'var(--surface-inset)', border: `2px solid ${picked === i ? 'var(--accent-cyan)' : 'var(--border-color)'}`, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}
            >
              <ImageCanvas image={p.image} size={44} />
              {heldOutBadge(p)}
            </button>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-start' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
            <ImageCanvas image={image} size={216} patchSize={data.patchSize} overlay={look?.overlay ?? null} overlayColor={THEME.cyan} />
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', maxWidth: 216, textAlign: 'center' }}>
              Tint: where the model looked when it chose <b style={{ color: 'var(--text-main)' }}>{chips[k] ?? '…'}</b> (layer {L + 1}, all heads)
            </span>
            {readsLike && (
              <div style={{ display: 'grid', gridTemplateColumns: `repeat(${grid}, 1fr)`, gap: 3, width: 216 }}>
                {readsLike.map((w, i) => (
                  <span key={i} className="font-mono" title={`Image token ${i + 1}: closest word embedding "${w.word}" (cosine ${w.sim.toFixed(2)})`} style={{ fontSize: '0.6rem', textAlign: 'center', padding: '2px 0', borderRadius: 4, background: 'var(--surface-inset)', color: 'var(--accent-purple)', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {w.word}
                  </span>
                ))}
              </div>
            )}
            <label style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
              <input type="checkbox" checked={showWords} onChange={e => setShowWords(e.target.checked)} />
              What each image token is closest to
              <InfoTooltip
                title="Image tokens as words"
                description="For each patch's image token, the word whose embedding points in the most similar direction (cosine). The language model reads image tokens in the same space as words."
                impact="Don't expect neat labels: image tokens only have to be useful to the model, not to look like any one word. Some do drift toward words like a colour, or a position."
              />
            </label>
          </div>

          <div style={{ flex: '1 1 340px', display: 'flex', flexDirection: 'column', gap: 12, minWidth: 0 }}>
            <PromptInput value={question} onChange={v => { setQuestion(v); setStepIdx(0); }} placeholder="what colour is the shape ?" />
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              {PRESETS.map(p => (
                <button key={p} className="font-mono" onClick={() => { setQuestion(p); setStepIdx(0); }} style={{ fontSize: '0.7rem', padding: '2px 8px', borderRadius: 6, cursor: 'pointer', background: 'var(--surface-inset)', border: `1px solid ${p === question ? 'var(--primary)' : 'var(--border-color)'}`, color: 'var(--text-muted)' }}>
                  {p}
                </button>
              ))}
            </div>
            {q.unknown.length > 0 && (
              <p style={{ fontSize: '0.75rem', color: 'var(--accent-amber)' }}>Not in the vocabulary, so skipped: {q.unknown.map(w => `"${w}"`).join(', ')}</p>
            )}

            <div>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: 6 }}>Its answer, one word at a time (click a word):</p>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', alignItems: 'center' }}>
                {chips.length === 0 && <span style={{ fontSize: '0.85rem', color: 'var(--text-dim)' }}>(nothing yet)</span>}
                {chips.map((w, i) => (
                  <button
                    key={i}
                    className="font-mono"
                    onClick={() => setStepIdx(i)}
                    style={{
                      fontSize: '0.95rem', padding: '4px 10px', borderRadius: 7, cursor: 'pointer',
                      background: i === k ? 'var(--primary-soft)' : 'var(--surface-inset)',
                      border: `1px solid ${i === k ? 'var(--primary)' : 'var(--border-color)'}`,
                      color: w === END ? 'var(--text-dim)' : 'var(--text-main)',
                    }}
                  >
                    {w}
                  </button>
                ))}
              </div>
              {right !== null && (
                <p style={{ fontSize: '0.8rem', marginTop: 8, color: said === right ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                  {said === right ? '✓ Right' : <>✗ The right answer is "{right}"</>}
                </p>
              )}
              {picked === BLANK && (
                <p style={{ fontSize: '0.8rem', marginTop: 8, color: 'var(--text-dim)' }}>
                  There's nothing to see in a blank picture, so whatever it says comes from habit: what's usually the answer.
                </p>
              )}
            </div>

            {look && (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14 }}>
                <div>
                  <p style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: 6 }}>
                    Attention when choosing "{chips[k]}"
                    <span style={{ fontWeight: 400, color: 'var(--text-dim)' }}> · {Math.round(look.imageShare * 100)}% on the picture</span>
                  </p>
                  <Toggle options={(step?.layerInspections ?? []).map((_, l) => ({ id: l, label: `Layer ${l + 1}` }))} value={L} onPick={setLayer} />
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 2, marginTop: 8 }}>
                    {look.words.map((w, i) => (
                      <div key={i} style={{ display: 'grid', gridTemplateColumns: '80px 1fr 36px', gap: 6, alignItems: 'center' }}>
                        <span className="font-mono" style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textAlign: 'right' }}>{w.word}</span>
                        <div style={{ height: 6, borderRadius: 3, background: 'var(--surface-inset)' }}>
                          <div style={{ width: `${w.v * 100}%`, height: '100%', borderRadius: 3, background: 'var(--accent-emerald)' }} />
                        </div>
                        <span className="font-mono" style={{ fontSize: '0.65rem', color: 'var(--text-dim)' }}>{Math.round(w.v * 100)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <p style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: 6 }}>Its top next words at that point</p>
                  {look.top.map(t => (
                    <div key={t.word} style={{ display: 'grid', gridTemplateColumns: '64px 1fr 40px', gap: 6, alignItems: 'center', marginBottom: 3 }}>
                      <span className="font-mono" style={{ fontSize: '0.75rem', textAlign: 'right' }}>{t.word}</span>
                      <div style={{ height: 8, borderRadius: 4, background: 'var(--surface-inset)' }}>
                        <div style={{ width: `${t.p * 100}%`, height: '100%', borderRadius: 4, background: 'var(--primary)' }} />
                      </div>
                      <span className="font-mono" style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{(t.p * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Full attention grid */}
      {step && (
        <div className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <Eye size={20} color="var(--accent-cyan)" />
            <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>The whole sequence's attention</h3>
            <InfoTooltip
              title="Image tokens + words, one sequence"
              description="▣1…▣9 are the picture's image tokens, then the question and the answer so far, all in one causal sequence: each position may look at itself and everything before it. Words can look back at the picture; the picture can't look ahead at the words."
              impact="Look for answer words with bright cells in the ▣ columns: that's the language model reading the picture."
            />
            <button className="btn-secondary" onClick={() => setShowGrid(!showGrid)} style={{ marginLeft: 'auto', padding: '4px 10px', fontSize: '0.75rem' }}>
              {showGrid ? 'Hide' : 'Show'} the grid
            </button>
          </div>
          {showGrid && (
            <>
              <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
                <Toggle options={step.layerInspections.map((_, l) => ({ id: l, label: `Layer ${l + 1}` }))} value={L} onPick={setLayer} />
                <Toggle
                  options={step.layerInspections[0].attentionWeights.map((_, h) => ({ id: h, label: `Head ${h + 1}` }))}
                  value={Math.min(gridHead, step.layerInspections[0].attentionWeights.length - 1)}
                  onPick={setGridHead}
                />
              </div>
              <AttentionGrid
                attentionMap={step.layerInspections[L].attentionWeights[Math.min(gridHead, step.layerInspections[0].attentionWeights.length - 1)]}
                tokenStrings={step.tokenStrings}
                highlightRow={step.tokenStrings.length - 1}
              />
            </>
          )}
          <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)', display: 'flex', gap: 6, alignItems: 'center' }}>
            <Sparkles size={13} /> Showing the step that chose "{chips[k]}". The last row is the position that made that choice.
          </p>
        </div>
      )}
    </div>
  );
};
