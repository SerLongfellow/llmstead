import React from 'react';
import { Turtle, StepForward, ArrowRight, ArrowDown } from 'lucide-react';
import { TokenChip } from './tokenUi';

/** Everything one training step did, captured so it can be replayed slowly */
export interface SlowStep {
  stepNumber: number;
  windowStart: number;    // index of the first input token in the training split
  totalTokens: number;    // size of the training split
  inputStrs: string[];
  targetStrs: string[];
  predStrs: string[];     // the model's top guess at each position (before the update)
  predProbs: number[];    // ...and its probability
  probsBefore: number[];  // probability given to the correct next token, before the update
  probsAfter: number[];   // ...and after
  losses: number[];       // per-position error: −log(probsBefore)
  lossBefore: number;     // mean of `losses` (this is the loss on the chart)
  lossAfter: number;      // same window, re-checked after the update
  gradGroups: { name: string; norm: number }[]; // how hard each layer was pushed
  learningRate: number;
}

/** Stage numbers: 1 Sample, 2 Forward, 3 Compare, 4 Backward, 5 Update */
export const STAGES = ['Sample', 'Forward', 'Compare', 'Backward', 'Update'] as const;

interface SlowMotionPanelProps {
  step: SlowStep | null;
  stage: number;            // 0 = nothing yet, 1–5 = how far the replay has got
  playing: boolean;
  spotlight: number;        // which position the Compare/Update stages focus on
  onSpotlight: (i: number) => void;
  onNextStage: () => void;  // used when paused
}

const pct = (p: number) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;

/** Green for a small error, through amber, to red for a big miss (an error of 4 ≈ the right answer got 2%) */
const errorColor = (loss: number) => {
  const t = Math.min(1, loss / 4);
  return t < 0.5
    ? `color-mix(in srgb, var(--accent-amber) ${Math.round(t * 200)}%, var(--accent-emerald))`
    : `color-mix(in srgb, var(--accent-rose) ${Math.round((t - 0.5) * 200)}%, var(--accent-amber))`;
};

/** How good was the guess, judged by the probability the right answer got */
const verdict = (p: number) =>
  p >= 0.5 ? { text: '✓ good guess', color: 'var(--accent-emerald)' }
    : p >= 0.1 ? { text: '~ unsure', color: 'var(--accent-amber)' }
    : { text: '✗ big miss', color: 'var(--accent-rose)' };

/** "▼ better" / "▲ worse" tag for a before → after change */
const Change: React.FC<{ better: boolean; arrowUp: boolean }> = ({ better, arrowUp }) => (
  <span style={{ fontSize: '0.75rem', fontWeight: 700, marginLeft: 8, color: better ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
    {arrowUp ? '▲' : '▼'} {better ? 'better' : 'worse'}
  </span>
);

const Caption: React.FC<{ n: number; stage: number; children: React.ReactNode }> = ({ n, stage, children }) => (
  <div style={{ display: 'flex', gap: 10, alignItems: 'baseline', opacity: stage >= n ? 1 : 0.35, transition: 'opacity 0.3s' }}>
    <span
      className="font-mono"
      style={{
        flexShrink: 0,
        width: 22,
        height: 22,
        borderRadius: 11,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '0.72rem',
        fontWeight: 700,
        background: stage === n ? (n === 4 ? 'var(--accent-rose)' : 'var(--primary)') : 'var(--bg-card-hover)',
        color: stage === n ? '#ffffff' : 'var(--text-muted)',
      }}
    >
      {n}
    </span>
    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.55 }}>
      <b style={{ color: 'var(--text-main)' }}>{STAGES[n - 1]}.</b> {children}
    </div>
  </div>
);

export const SlowMotionPanel: React.FC<SlowMotionPanelProps> = ({ step, stage, playing, spotlight, onSpotlight, onNextStage }) => {
  const header = (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <Turtle size={20} color="var(--accent-emerald)" />
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>Slow motion: one training step</h3>
        {step && <span className="badge badge-primary font-mono">Step #{step.stepNumber}</span>}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
        {STAGES.map((s, i) => (
          <span
            key={s}
            style={{
              fontSize: '0.75rem',
              fontWeight: 600,
              padding: '3px 10px',
              borderRadius: 999,
              border: '1px solid var(--border-color)',
              background: stage === i + 1 ? (i === 3 ? 'var(--accent-rose)' : 'var(--primary)') : 'transparent',
              color: stage === i + 1 ? '#ffffff' : stage > i + 1 ? 'var(--text-main)' : 'var(--text-dim)',
            }}
          >
            {i + 1}. {s}
          </span>
        ))}
        {!playing && (
          <button className="btn-primary" onClick={onNextStage} style={{ padding: '5px 12px', fontSize: '0.8rem', marginLeft: 6 }}>
            <StepForward size={14} /> {!step || stage >= 5 ? 'Run a step' : 'Next stage'}
          </button>
        )}
      </div>
    </div>
  );

  if (!step) {
    return (
      <div className="glass-panel" style={{ padding: 24, gridColumn: '1 / -1' }}>
        {header}
        <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
          Press <b>Start</b> to watch training one step at a time, or <b>Run a step</b> to go through it stage by stage.
        </p>
      </div>
    );
  }

  const n = step.inputStrs.length;
  const s = Math.min(spotlight, n - 1);
  const maxLoss = Math.max(...step.losses, 1e-9);
  const maxGrad = Math.max(...step.gradGroups.map(g => g.norm), 1e-9);
  const context = step.inputStrs.slice(0, s + 1).join(''); // everything the guess can see
  const improved = step.probsAfter[s] >= step.probsBefore[s];
  const cols = { display: 'grid', gridTemplateColumns: `72px repeat(${n}, minmax(40px, 1fr))`, gap: 4, alignItems: 'center' } as const;

  // Model diagram boxes, left to right; the backward pass lights them right to left
  const boxes = ['Tokens', 'Embedding', ...step.gradGroups.filter(g => g.name.startsWith('Block')).map(g => g.name), 'Prediction', 'Loss'];
  const gradFor = (box: string) =>
    box === 'Embedding' ? step.gradGroups.find(g => g.name === 'Embedding')
      : box === 'Prediction' ? step.gradGroups.find(g => g.name === 'Output head')
      : step.gradGroups.find(g => g.name === box);

  return (
    <div className="glass-panel" style={{ padding: 24, gridColumn: '1 / -1' }}>
      {header}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
        {/* 1. Sample */}
        <Caption n={1} stage={stage}>
          A window of {n} tokens, taken from position {step.windowStart.toLocaleString()} of the{' '}
          {step.totalTokens.toLocaleString()}-token training text. The targets are the same text shifted one token: at
          every position, the model has to predict what comes next. All {n} predictions are made at once.
        </Caption>
        <details style={{ marginLeft: 32, marginTop: -8, opacity: stage >= 1 ? 1 : 0.35 }}>
          <summary style={{ cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600, color: 'var(--primary)' }}>
            Why one window at a time, and why all {n} predictions at once?
          </summary>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 8, fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.6, maxWidth: 900 }}>
            <p>
              <b style={{ color: 'var(--text-main)' }}>Why a window?</b> The model can only look at {n} tokens at once. That's
              its context window (real models handle many thousands). So training feeds it bite-sized chunks, and picking a
              random chunk each step means it practices on a different part of the text every time.
            </p>
            <p>
              <b style={{ color: 'var(--text-main)' }}>Each guess uses everything before it.</b> The guess in a column isn't
              based on just the token above it. Attention blends in all the earlier tokens in the window, so the guess in
              column 5 is based on tokens 1–5 together. (The highlighted span below shows this for whichever column you pick.)
            </p>
            <p>
              <b style={{ color: 'var(--text-main)' }}>So one window holds {n} questions.</b> Take the text "3 + 4 = 7": what
              comes after "3"? After "3 +"? After "3 + 4"? Instead of running the model once per question, it runs once on
              the whole window, with every position blindfolded to everything on its right (the causal mask). The guess after
              "3 + 4" can't peek at the "= 7" that follows, so it comes out exactly as if the model had only been shown
              "3 + 4". One run, {n} fair questions, which is a big part of why transformers are fast to train.
            </p>
            <p>
              <b style={{ color: 'var(--text-main)' }}>Generating is different.</b> The model still makes a guess at every
              position; generating just keeps the last one. When the model writes new text, the next token doesn't exist
              yet, so it has to go one token at a time: predict, append, repeat. You can try that in Look inside → Next
              token.
            </p>
          </div>
        </details>
        <div style={{ overflowX: 'auto', opacity: stage >= 1 ? 1 : 0.35 }}>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: '0 0 8px 76px' }}>
            The guess in column {s + 1} uses <b style={{ color: 'var(--primary)' }}>all {s + 1} highlighted tokens</b>, not
            just the one above it. Tokens to its right are hidden from it. Click any column to compare.
          </p>
          <div style={{ ...cols, minWidth: 72 + n * 44 }}>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>Input</span>
            {step.inputStrs.map((t, i) => (
              <div
                key={i}
                style={{ display: 'flex', flexDirection: 'column', alignItems: 'stretch', gap: 3, opacity: i <= s ? 1 : 0.25, transition: 'opacity 0.2s' }}
              >
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                  <TokenChip text={t} active={i === s} onClick={() => onSpotlight(i)} />
                </div>
                {/* the span of tokens the spotlighted guess can see */}
                <div style={{ height: 3, borderRadius: 2, background: i <= s ? 'var(--primary)' : 'transparent' }} />
              </div>
            ))}
            <span />
            {step.inputStrs.map((_, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'center', color: i === s ? 'var(--primary)' : 'var(--text-dim)', opacity: i === s ? 1 : 0.4 }}>
                <ArrowDown size={i === s ? 14 : 12} />
              </div>
            ))}
            {/* The model's top guess at each position appears once the forward pass has run */}
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', opacity: stage >= 2 ? 1 : 0 }}>Guess (model)</span>
            {step.predStrs.map((t, i) => {
              const right = t === step.targetStrs[i];
              return (
                <div
                  key={i}
                  title={`The model's top guess: ${(step.predProbs[i] * 100).toFixed(1)}% sure${right ? ', and it matches the target' : ''}`}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: 1,
                    opacity: stage >= 2 ? (i === s ? 1 : 0.55) : 0,
                    transition: 'opacity 0.4s',
                  }}
                >
                  <TokenChip
                    text={t}
                    bg={right ? 'var(--emerald-tint)' : 'var(--primary-tint)'}
                    active={i === s}
                    onClick={() => onSpotlight(i)}
                  />
                  <span className="font-mono" style={{ fontSize: '0.6rem', color: 'var(--text-dim)' }}>{pct(step.predProbs[i])}</span>
                </div>
              );
            })}
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>Target (next)</span>
            {step.targetStrs.map((t, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'center', opacity: i === s ? 1 : 0.55 }}>
                <TokenChip text={t} bg="var(--emerald-soft)" active={i === s} onClick={() => onSpotlight(i)} />
              </div>
            ))}
            {/* Per-position error bars appear at the Compare stage */}
            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', opacity: stage >= 3 ? 1 : 0, lineHeight: 1.3 }}>
              Error<br />(lower = better)
            </span>
            {step.losses.map((l, i) => (
              <div
                key={i}
                onClick={() => onSpotlight(i)}
                title={`−log(${pct(step.probsBefore[i])}) = ${l.toFixed(2)}`}
                style={{ height: 52, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', alignItems: 'center', cursor: 'pointer', opacity: stage >= 3 ? 1 : 0, transition: 'opacity 0.4s' }}
              >
                <div
                  style={{
                    width: '60%',
                    height: `${Math.max(4, (l / maxLoss) * 40)}px`,
                    borderRadius: 3,
                    background: errorColor(l),
                    outline: i === s ? '2px solid var(--text-main)' : undefined,
                    outlineOffset: 2,
                  }}
                />
                <span className="font-mono" style={{ fontSize: '0.62rem', color: 'var(--text-dim)' }}>{l.toFixed(1)}</span>
              </div>
            ))}
          </div>
        </div>

        {/* 2. Forward / 4. Backward share one model diagram */}
        <Caption n={2} stage={stage}>
          All {n} tokens flow through the model, left to right, and come out as a probability for every token in the
          vocabulary, at every position.
        </Caption>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 6, flexWrap: 'wrap', opacity: stage >= 2 ? 1 : 0.35 }}>
          {boxes.map((b, i) => {
            const g = gradFor(b);
            const fwd = stage === 2;
            const bwd = stage === 4;
            const delay = fwd ? i * 0.22 : bwd ? (boxes.length - 1 - i) * 0.22 : 0;
            return (
              <React.Fragment key={`${step.stepNumber}-${stage}-${b}`}>
                {i > 0 && (
                  <div style={{ paddingTop: 10, color: bwd ? 'var(--accent-rose)' : 'var(--text-dim)', transform: bwd ? 'scaleX(-1)' : undefined }}>
                    <ArrowRight size={16} />
                  </div>
                )}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'stretch', minWidth: 92 }}>
                  <div
                    className={fwd ? 'pulse-fwd' : bwd ? 'pulse-bwd' : undefined}
                    style={{
                      animationDelay: `${delay}s`,
                      padding: '8px 12px',
                      borderRadius: 8,
                      textAlign: 'center',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      border: '1px solid var(--border-color)',
                      background: 'var(--surface-inset)',
                    }}
                  >
                    {b}
                  </div>
                  {/* Gradient size per layer, shown from the Backward stage on */}
                  {g && (
                    <div style={{ opacity: stage >= 4 ? 1 : 0, transition: 'opacity 0.4s' }} title={`gradient size ${g.norm.toFixed(3)}`}>
                      <div style={{ height: 6, background: 'var(--surface-inset)', borderRadius: 3 }}>
                        <div style={{ width: `${(g.norm / maxGrad) * 100}%`, height: '100%', background: 'var(--accent-rose)', borderRadius: 3 }} />
                      </div>
                    </div>
                  )}
                </div>
              </React.Fragment>
            );
          })}
        </div>

        {/* 3. Compare */}
        <Caption n={3} stage={stage}>
          The objective: at each position, the error is −log(the probability the model gave the right answer). A confident
          right answer costs almost nothing; a low-probability one costs a lot. Click any column to inspect it.
        </Caption>
        <div
          style={{
            opacity: stage >= 3 ? 1 : 0.35,
            padding: '10px 14px',
            borderRadius: 8,
            background: 'var(--surface-inset)',
            border: '1px solid var(--border-color)',
            fontSize: '0.85rem',
            lineHeight: 1.9,
            color: 'var(--text-muted)',
          }}
        >
          {stage >= 3 ? (
            <>
              After <TokenChip text={context} active /> the model's top guess was <TokenChip text={step.predStrs[s]} bg="var(--primary-tint)" />{' '}
              ({pct(step.predProbs[s])}). The right answer <TokenChip text={step.targetStrs[s]} bg="var(--emerald-tint)" /> got{' '}
              <b className="font-mono" style={{ color: 'var(--text-main)' }}>{pct(step.probsBefore[s])}</b>, so the error is −log(
              {step.probsBefore[s].toFixed(3)}) = <b className="font-mono" style={{ color: errorColor(step.losses[s]) }}>{step.losses[s].toFixed(2)}</b>{' '}
              <b style={{ color: verdict(step.probsBefore[s]).color }}>{verdict(step.probsBefore[s]).text}</b>.
              The average of all {n} errors is the loss: <b className="font-mono" style={{ color: 'var(--text-main)' }}>{step.lossBefore.toFixed(3)}</b>.
            </>
          ) : (
            'Waiting for the forward pass…'
          )}
        </div>

        {/* 4. Backward */}
        <Caption n={4} stage={stage}>
          Backpropagation sends the error back through the model, right to left, working out for every one of the model's
          weights which direction would lower the loss. The red bars show how hard each part of the model is being pushed.
        </Caption>

        {/* 5. Update */}
        <Caption n={5} stage={stage}>
          The optimizer nudges every weight a small step in that direction (learning rate {step.learningRate}). Checking the
          same window again, after the update:
        </Caption>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', opacity: stage >= 5 ? 1 : 0.35 }}>
          <div style={{ flex: '1 1 260px', padding: '10px 14px', borderRadius: 8, background: 'var(--surface-inset)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Probability of the right answer at the spotlighted position</div>
            <div className="font-mono" style={{ fontSize: '1.2rem', fontWeight: 700 }}>
              {pct(step.probsBefore[s])} <span style={{ color: 'var(--text-dim)' }}>→</span>{' '}
              <span style={{ color: stage >= 5 ? (improved ? 'var(--accent-emerald)' : 'var(--accent-rose)') : undefined }}>
                {stage >= 5 ? pct(step.probsAfter[s]) : '?'}
              </span>
              {stage >= 5 && <Change better={improved} arrowUp={improved} />}
            </div>
          </div>
          <div style={{ flex: '1 1 260px', padding: '10px 14px', borderRadius: 8, background: 'var(--surface-inset)', border: '1px solid var(--border-color)' }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Loss on this window</div>
            <div className="font-mono" style={{ fontSize: '1.2rem', fontWeight: 700 }}>
              {step.lossBefore.toFixed(3)} <span style={{ color: 'var(--text-dim)' }}>→</span>{' '}
              <span style={{ color: stage >= 5 ? (step.lossAfter <= step.lossBefore ? 'var(--accent-emerald)' : 'var(--accent-rose)') : undefined }}>
                {stage >= 5 ? step.lossAfter.toFixed(3) : '?'}
              </span>
              {stage >= 5 && <Change better={step.lossAfter <= step.lossBefore} arrowUp={step.lossAfter > step.lossBefore} />}
            </div>
          </div>
        </div>
        {stage >= 5 && (
          <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
            One step makes a small difference. Training repeats this loop thousands of times, on a different window each time.
            {!improved && ' (Sometimes one position gets slightly worse while the window as a whole improves.)'}
          </p>
        )}
      </div>
    </div>
  );
};
