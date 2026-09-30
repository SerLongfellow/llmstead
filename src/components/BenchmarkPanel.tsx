import React, { useState, useEffect } from 'react';
import { DatasetOption } from '../types';
import { MicroTransformer } from '../engine/transformer';
import { BPETokenizer } from '../engine/bpeTokenizer';
import { BENCHMARK_SUITES, BenchmarkSuiteResult, BenchmarkSplit, runBenchmarkSuite } from '../engine/benchmarks';
import { ClipboardCheck, Play } from 'lucide-react';
import { InfoTooltip } from './InfoTooltip';

interface BenchmarkPanelProps {
  model: MicroTransformer;
  tokenizer: BPETokenizer;
  selectedDataset: DatasetOption;
  /** The exact text the model trains on — used to label each case seen / held-out */
  trainText: string;
  stepCount: number;
}

const SPLIT_STYLE: Record<BenchmarkSplit, { label: string; color: string; badge: string }> = {
  'seen': { label: 'Seen in training', color: 'var(--accent-purple)', badge: 'badge-purple' },
  'held-out': { label: 'Held-out', color: 'var(--accent-amber)', badge: 'badge-amber' },
};

const show = (s: string) => JSON.stringify(s).slice(1, -1); // make \n and spaces visible

export const BenchmarkPanel: React.FC<BenchmarkPanelProps> = ({
  model,
  tokenizer,
  selectedDataset,
  trainText,
  stepCount,
}) => {
  const suite = BENCHMARK_SUITES[selectedDataset.id];
  const [result, setResult] = useState<BenchmarkSuiteResult | null>(null);
  const [ranAtStep, setRanAtStep] = useState<number | null>(null);

  // Results belong to a specific model + dataset; clear them when either changes
  useEffect(() => {
    setResult(null);
    setRanAtStep(null);
  }, [model, selectedDataset, tokenizer]);

  const run = () => {
    if (!suite) return;
    setResult(runBenchmarkSuite(suite, model, tokenizer, trainText));
    setRanAtStep(stepCount);
  };

  return (
    <div className="glass-panel" style={{ padding: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <ClipboardCheck size={20} color="var(--accent-cyan)" />
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Benchmarks: Memorization vs Generalization</h2>
          <InfoTooltip
            title="Seen vs Held-out"
            description="Each case is a prompt and an expected answer, scored by greedy decoding. 'Seen' cases appear word-for-word in the training split, so passing them may just be recall. 'Held-out' cases never appear in training — passing them means the model learned a pattern."
            impact="A big gap between the two scores is the signature of memorization. Real benchmarks guard against this by keeping test data out of training (and checking for contamination)."
          />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {ranAtStep !== null && (
            <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>
              Ran at step #{ranAtStep}{ranAtStep !== stepCount ? ` · model now at #${stepCount}` : ''}
            </span>
          )}
          <button className="btn-primary" onClick={run} disabled={!suite}>
            <Play size={16} /> Run Benchmarks
          </button>
        </div>
      </div>

      {!suite && (
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No benchmark suite defined for this dataset yet.</p>
      )}

      {suite && !result && (
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          {suite.cases.length} cases for {suite.datasetName}. Train for a while, then run to compare how the model does on text it has seen versus text it hasn't.
        </p>
      )}

      {result && (
        <>
          {/* Score cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginBottom: '16px' }}>
            {(['seen', 'held-out'] as const).map(split => {
              const sc = result.bySplit[split];
              const st = SPLIT_STYLE[split];
              return (
                <div key={split} style={{ padding: '14px 16px', borderRadius: '10px', background: 'rgba(15, 23, 42, 0.5)', border: `1px solid ${st.color}` }}>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{st.label}</p>
                  <p className="font-mono" style={{ fontSize: '1.6rem', fontWeight: 800, color: st.color }}>
                    {sc.total === 0 ? '—' : `${sc.accuracy.toFixed(0)}%`}
                  </p>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>{sc.passed} / {sc.total} passed</p>
                </div>
              );
            })}
          </div>

          {/* Per-case table */}
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
              <thead>
                <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
                  <th style={{ padding: '6px 8px' }}>Split</th>
                  <th style={{ padding: '6px 8px' }}>Prompt</th>
                  <th style={{ padding: '6px 8px' }}>Expected</th>
                  <th style={{ padding: '6px 8px' }}>Model output</th>
                  <th style={{ padding: '6px 8px' }}>Result</th>
                </tr>
              </thead>
              <tbody>
                {result.results.map(r => (
                  <tr key={r.testCase.id} style={{ borderTop: '1px solid var(--border-color)' }}>
                    <td style={{ padding: '6px 8px' }}>
                      <span className={`badge ${SPLIT_STYLE[r.split].badge}`} style={{ fontSize: '0.65rem' }}>{r.split}</span>
                    </td>
                    <td className="font-mono" style={{ padding: '6px 8px', color: 'var(--text-muted)' }}>{show(r.testCase.prompt)}</td>
                    <td className="font-mono" style={{ padding: '6px 8px', color: 'var(--accent-emerald)' }}>{show(r.testCase.expectedOutput)}</td>
                    <td className="font-mono" style={{ padding: '6px 8px' }}>{show(r.actualOutput)}</td>
                    <td style={{ padding: '6px 8px', fontWeight: 700, color: r.passed ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                      {r.passed ? 'PASS' : 'FAIL'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
};
