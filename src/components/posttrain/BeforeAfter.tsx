import React, { useEffect, useRef, useState } from 'react';
import { GitCompare, Play } from 'lucide-react';
import { MicroTransformer } from '../../engine/transformer';
import { BPETokenizer } from '../../engine/bpeTokenizer';
import { generateTokens } from '../../engine/generate';
import { BENCHMARK_SUITES, BenchmarkSuiteResult, runBenchmarkSuite } from '../../engine/benchmarks';
import { PromptInput } from '../PromptInput';
import { showTok } from '../tokenUi';

/** Greedy reply, stopping at <EOS> (which post-training may have taught) */
function reply(model: MicroTransformer, tokenizer: BPETokenizer, prompt: string) {
  const g = generateTokens(model, tokenizer, prompt, { maxTokens: 16, temperature: 0, stopToken: tokenizer.specialId('<EOS>') });
  return { text: g.text, stopped: g.stopped };
}

interface Row {
  prompt: string;
  base: { text: string; stopped: boolean };
  now: { text: string; stopped: boolean };
}

const Reply: React.FC<{ r: { text: string; stopped: boolean } }> = ({ r }) => (
  <span className="font-mono" style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
    {showTok(r.text)}
    {r.stopped && <span className="badge badge-emerald" style={{ marginLeft: 6, fontSize: '0.65rem' }}>&lt;EOS&gt;</span>}
  </span>
);

const Score: React.FC<{ r: BenchmarkSuiteResult | null }> = ({ r }) =>
  r ? (
    <span className="font-mono">
      seen {r.bySplit.seen.passed}/{r.bySplit.seen.total} · held-out {r.bySplit['held-out'].passed}/{r.bySplit['held-out'].total}
    </span>
  ) : (
    <span style={{ color: 'var(--text-dim)' }}>—</span>
  );

/**
 * The starting model (frozen) next to the post-trained one: the same prompts, greedy replies,
 * and the pre-training benchmark for both. Re-runs when `refreshKey` changes (each pause), but
 * only while visible: the tab stays mounted, and every Setup change builds a new model, so running
 * hidden would generate dozens of replies (and two benchmarks) per slider step.
 */
export const BeforeAfter: React.FC<{
  base: MicroTransformer;
  current: MicroTransformer;
  tokenizer: BPETokenizer;
  datasetId: string;
  trainText: string;
  defaultPrompts: string[];
  refreshKey: number;
  /** Nothing has been post-trained yet, so both columns would be the same model */
  untouched: boolean;
  /** The Post-train tab is showing; changes while hidden are caught up on when it's next shown */
  visible: boolean;
}> = ({ base, current, tokenizer, datasetId, trainText, defaultPrompts, refreshKey, untouched, visible }) => {
  const [promptText, setPromptText] = useState(() => defaultPrompts.map(p => JSON.stringify(p)).join('\n'));
  useEffect(() => setPromptText(defaultPrompts.map(p => JSON.stringify(p)).join('\n')), [defaultPrompts]);
  const [rows, setRows] = useState<Row[]>([]);
  const [bench, setBench] = useState<{ base: BenchmarkSuiteResult | null; now: BenchmarkSuiteResult | null }>({ base: null, now: null });
  const suite = BENCHMARK_SUITES[datasetId];

  // One prompt per line, written as a JSON string so spaces and \n are visible and exact
  const prompts = promptText
    .split('\n')
    .map(line => line.trim())
    .filter(Boolean)
    .map(line => {
      try {
        const v = JSON.parse(line);
        return typeof v === 'string' ? v : line;
      } catch {
        return line;
      }
    });

  const run = () => {
    setRows(prompts.map(prompt => ({ prompt, base: reply(base, tokenizer, prompt), now: reply(current, tokenizer, prompt) })));
    if (suite) {
      setBench({ base: runBenchmarkSuite(suite, base, tokenizer, trainText), now: runBenchmarkSuite(suite, current, tokenizer, trainText) });
    }
  };
  const stale = useRef(true);
  useEffect(() => {
    stale.current = true;
  }, [refreshKey, base, current, tokenizer]);
  useEffect(() => {
    if (!visible || !stale.current) return;
    stale.current = false;
    run();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, refreshKey, base, current, tokenizer]);

  return (
    <div className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <GitCompare size={20} color="var(--accent-cyan)" />
        <h3 style={{ fontSize: '1.05rem', fontWeight: 700, flex: 1 }}>Before and after</h3>
        <button className="btn-secondary" onClick={run} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
          <Play size={14} /> Compare now
        </button>
      </div>
      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
        The pre-trained model, frozen when post-training started, next to the model now, both answering greedily (always the
        most likely token). {untouched && 'Nothing has been post-trained yet, so the two columns match.'} Updates whenever
        post-training pauses.
      </p>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
          <thead>
            <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
              <th style={{ padding: '6px 8px' }}>Prompt</th>
              <th style={{ padding: '6px 8px' }}>Pre-trained</th>
              <th style={{ padding: '6px 8px' }}>Now</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} style={{ borderTop: '1px solid var(--border-color)', verticalAlign: 'top' }}>
                <td className="font-mono" style={{ padding: '6px 8px', color: 'var(--text-muted)', whiteSpace: 'pre-wrap' }}>{showTok(r.prompt)}</td>
                <td style={{ padding: '6px 8px', color: 'var(--text-muted)' }}><Reply r={r.base} /></td>
                <td style={{ padding: '6px 8px', fontWeight: r.now.text !== r.base.text ? 600 : 400 }}><Reply r={r.now} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {suite && (
        <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', fontSize: '0.8rem', padding: '10px 12px', borderRadius: 8, background: 'var(--surface-inset)' }}>
          <span style={{ fontWeight: 600 }}>Pre-training benchmark</span>
          <span>Pre-trained: <Score r={bench.base} /></span>
          <span>Now: <Score r={bench.now} /></span>
        </div>
      )}

      <details>
        <summary style={{ fontSize: '0.8rem', color: 'var(--text-muted)', cursor: 'pointer' }}>Edit the prompts</summary>
        <div style={{ marginTop: 8 }}>
          <PromptInput value={promptText} onChange={setPromptText} onSubmit={run} style={{ fontSize: '0.8rem' }} />
          <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
            One prompt per line, in quotes so trailing spaces and line breaks (\n) are exact.
          </span>
        </div>
      </details>
    </div>
  );
};
