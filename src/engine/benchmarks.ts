import { MicroTransformer } from './transformer';
import { BPETokenizer } from './bpeTokenizer';
import { generateContinuation } from './generate';

export interface BenchmarkTestCase {
  id: string;
  prompt: string;
  expectedOutput: string;
}

export interface BenchmarkSuite {
  datasetId: string;
  datasetName: string;
  cases: BenchmarkTestCase[];
}

/**
 * 'seen'     — prompt + answer appears verbatim in the training split, so passing
 *              can be pure memorization.
 * 'held-out' — never shown during training; passing requires generalizing.
 * Assigned at runtime by checking the case against the actual training text.
 */
export type BenchmarkSplit = 'seen' | 'held-out';

export interface BenchmarkResult {
  testCase: BenchmarkTestCase;
  split: BenchmarkSplit;
  actualOutput: string;
  passed: boolean;
}

export interface SplitScore {
  total: number;
  passed: number;
  accuracy: number; // 0 to 100%
}

export interface BenchmarkSuiteResult {
  datasetId: string;
  datasetName: string;
  totalCases: number;
  passedCases: number;
  accuracy: number; // 0 to 100%
  bySplit: Record<BenchmarkSplit, SplitScore>;
  results: BenchmarkResult[];
}

export const BENCHMARK_SUITES: Record<string, BenchmarkSuite> = {
  'math-logic': {
    datasetId: 'math-logic',
    datasetName: 'Synthetic Math & Logic',
    cases: [
      { id: 'm1', prompt: '1 + 1 = ', expectedOutput: '2' },
      { id: 'm2', prompt: '1 + 2 = ', expectedOutput: '3' },
      { id: 'm3', prompt: '2 + 2 = ', expectedOutput: '4' },
      { id: 'm4', prompt: '2 + 3 = ', expectedOutput: '5' },
      { id: 'm5', prompt: '3 + 3 = ', expectedOutput: '6' },
      { id: 'm6', prompt: 'If A then B. A is true. Therefore ', expectedOutput: 'B' },
      { id: 'm7', prompt: 'If B then C. B is true. Therefore ', expectedOutput: 'C' },
      { id: 'm8', prompt: 'Dog is an animal. Animal has four legs. Dog has ', expectedOutput: 'four' },
      // Not in the corpus: commutativity, new sums, new rule chains, new nouns
      { id: 'm9', prompt: '2 + 1 = ', expectedOutput: '3' },
      { id: 'm10', prompt: '4 + 3 = ', expectedOutput: '7' },
      { id: 'm11', prompt: '1 + 3 = ', expectedOutput: '4' },
      { id: 'm12', prompt: 'If D then E. D is true. Therefore ', expectedOutput: 'E' },
      { id: 'm13', prompt: 'Cow is an animal. Animal has four legs. Cow has ', expectedOutput: 'four' },
    ]
  },
  'shakespeare': {
    datasetId: 'shakespeare',
    datasetName: 'Tiny Shakespeare',
    cases: [
      // Patterns that REPEAT in the training text. A small model learns frequent
      // patterns long before it can memorize a line that appears only once.
      { id: 's1', prompt: 'FIRST ', expectedOutput: 'CITIZEN' },
      { id: 's2', prompt: 'SECOND ', expectedOutput: 'CITIZEN' },
      { id: 's3', prompt: 'Caius ', expectedOutput: 'Marcius' },
      { id: 's4', prompt: 'Noble ', expectedOutput: 'Marcius' },
      { id: 's5', prompt: 'What\'s the ', expectedOutput: 'matter' },
      { id: 's6', prompt: 'the Volsces are in ', expectedOutput: 'arms' },
      // Not in the corpus: the same patterns behind a new lead-in
      { id: 's7', prompt: 'THIRD ', expectedOutput: 'CITIZEN' },
      { id: 's8', prompt: 'I say, Caius ', expectedOutput: 'Marcius' },
      { id: 's9', prompt: 'Tell me, what\'s the ', expectedOutput: 'matter' },
      { id: 's10', prompt: 'Hark! the Volsces are in ', expectedOutput: 'arms' },
    ]
  },
  'code-python': {
    datasetId: 'code-python',
    datasetName: 'Python Micro Snippets',
    cases: [
      { id: 'c1', prompt: 'def add(a, b):\n    ', expectedOutput: 'return' },
      { id: 'c2', prompt: 'def multiply(a, b):\n    ', expectedOutput: 'return' },
      { id: 'c3', prompt: 'for i in range(5):\n    ', expectedOutput: 'print' },
      { id: 'c4', prompt: 'x = 10\nif x > 5:\n    ', expectedOutput: 'print' },
      // Not in the corpus: same structure, new names and values
      { id: 'c5', prompt: 'def subtract(a, b):\n    ', expectedOutput: 'return' },
      { id: 'c6', prompt: 'for j in range(3):\n    ', expectedOutput: 'print' },
      { id: 'c7', prompt: 'y = 2\nif y > 1:\n    ', expectedOutput: 'print' },
    ]
  },
  'qa-dialogue': {
    datasetId: 'qa-dialogue',
    datasetName: 'Simple Q&A Conversations',
    cases: [
      { id: 'q1', prompt: 'User: Hello!\nAssistant: ', expectedOutput: 'Hi' },
      { id: 'q2', prompt: 'User: What is an LLM?\nAssistant: ', expectedOutput: 'An' },
      { id: 'q3', prompt: 'User: What is attention?\nAssistant: ', expectedOutput: 'Attention' },
      // Not in the corpus: rephrased questions
      { id: 'q4', prompt: 'User: Hi!\nAssistant: ', expectedOutput: 'Hi' },
      { id: 'q5', prompt: 'User: What is a token?\nAssistant: ', expectedOutput: 'A' },
    ]
  }
};

/** Label a case by whether its full text appears in the training split */
export function classifyCase(testCase: BenchmarkTestCase, trainText: string): BenchmarkSplit {
  return trainText.includes(testCase.prompt + testCase.expectedOutput) ? 'seen' : 'held-out';
}

function score(results: BenchmarkResult[]): SplitScore {
  const total = results.length;
  const passed = results.filter(r => r.passed).length;
  return { total, passed, accuracy: total === 0 ? 0 : (passed / total) * 100 };
}

/**
 * Greedy-decode a short continuation for every case and check whether it starts
 * with the expected answer (leading whitespace ignored).
 */
export function runBenchmarkSuite(
  suite: BenchmarkSuite,
  model: MicroTransformer,
  tokenizer: BPETokenizer,
  trainText: string
): BenchmarkSuiteResult {
  const results: BenchmarkResult[] = suite.cases.map(testCase => {
    const budget = testCase.expectedOutput.length + 4;
    const actualOutput = generateContinuation(model, tokenizer, testCase.prompt, {
      maxTokens: budget,
      temperature: 0, // greedy → deterministic, repeatable scores
      stopAfterChars: budget,
    });
    return {
      testCase,
      split: classifyCase(testCase, trainText),
      actualOutput,
      passed: actualOutput.trimStart().startsWith(testCase.expectedOutput),
    };
  });

  const overall = score(results);
  return {
    datasetId: suite.datasetId,
    datasetName: suite.datasetName,
    totalCases: overall.total,
    passedCases: overall.passed,
    accuracy: overall.accuracy,
    bySplit: {
      'seen': score(results.filter(r => r.split === 'seen')),
      'held-out': score(results.filter(r => r.split === 'held-out')),
    },
    results,
  };
}
