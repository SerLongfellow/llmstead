import React from 'react';
import { Compass, Layers, MessageSquare, ThumbsUp, ClipboardCheck, Rocket, ArrowRight, BookOpen } from 'lucide-react';
import { GoDeeper, ExternalLink } from './GoDeeper';
import { KeyPanel } from './KeyPanel';

interface WhatsNextViewProps {
  onNavigate: (tab: string) => void;
}

/** One line of a stage's example box. `tone` colors the label: good, bad, or neutral. */
interface ExampleLine {
  label: string;
  text: string;
  tone?: 'good' | 'bad';
}

interface Stage {
  id: string;
  pill: string;
  title: string;
  icon: React.ReactNode;
  tagline: string;
  /** Plain-language explanation, one paragraph per entry */
  body: React.ReactNode[];
  /** Jargon and acronyms used in this stage, explained for newcomers */
  terms: [term: string, meaning: React.ReactNode][];
  here: string[];
  real: string[];
  example: { title: string; lines: ExampleLine[]; note?: string };
  action?: { label: string; tab: string };
  links: ExternalLink[];
}

const STAGES: Stage[] = [
  {
    id: 'scale',
    pill: 'Scale up',
    title: 'Scale up pre-training',
    icon: <Layers size={20} color="var(--accent-purple)" />,
    tagline: 'The same next-token objective you just trained, with vastly more data, parameters and compute.',
    body: [
      <>
        Nothing about the goal changes: a frontier model's pre-training is still “predict the next token”, scored with the
        same loss you watched fall. What changes is scale, and with scale come abilities that a small model never shows,
        like translating, summarizing or writing working code without being taught any of them directly.
      </>,
      <>
        <b>Scaling laws</b> make this predictable: loss falls smoothly as you add parameters, data and compute, so labs can
        forecast a big run from small ones. The Chinchilla paper found that, for a fixed compute budget, a model should see
        roughly <b>20 tokens per parameter</b>. Many newer models go far past that on purpose: a smaller model trained
        longer is cheaper to run once millions of people use it.
      </>,
      <>
        The architecture gets refinements rather than a redesign. Most current models normalize <i>before</i> each half of
        the block (pre-LN, usually RMSNorm), encode position by rotating the queries and keys (RoPE) instead of adding a
        position table, and use a gated MLP (SwiGLU). Near the end, many runs switch to a smaller set of high-quality data
        and stretch the context window to tens or hundreds of thousands of tokens.
      </>,
    ],
    terms: [
      ['Parameters', 'The model\'s learned numbers (its weights). "405B" in a model name means 405 billion of them.'],
      ['CPU / GPU', 'A CPU is your computer\'s general-purpose processor; LLMStead trains on it. A GPU (graphics processing unit) is a chip built to do thousands of multiplications at once, which is what training mostly is.'],
      ['pre-LN', 'Normalizing the vector before each half of a transformer block instead of after it (LLMStead normalizes after, "post-LN"). Deep models train more stably this way.'],
      ['RMSNorm', 'Root-mean-square normalization: a cheaper layer norm that only rescales the vector to a standard size, skipping the "subtract the average" step.'],
      ['RoPE', 'Rotary position embedding: instead of adding a position vector (as LLMStead does), each query and key is rotated by an angle that depends on its position, so attention can tell how far apart two tokens are.'],
      ['MLP / SwiGLU', 'The MLP is the per-token network in each block (see Look inside). SwiGLU is a "gated" version: two projections side by side, one of which decides how much of the other to let through, number by number.'],
    ],
    here: [
      'About 30,000 parameters',
      'About 10,000 tokens of training text',
      'One short window per step, on your CPU, in minutes',
      'Text is whatever dataset you picked in Setup',
    ],
    real: [
      'Billions to hundreds of billions of parameters',
      'Trillions of tokens (Llama 3: about 15 trillion)',
      'Millions of tokens per step, across thousands of GPUs (16,000+ for Llama 3.1 405B), for weeks to months',
      'Web pages, books, code and math, heavily filtered and deduplicated',
    ],
    example: {
      title: 'What you get at the end: a base model',
      lines: [
        { label: 'Prompt', text: 'The capital of France is' },
        { label: 'Base model', text: ' Paris. The capital of Germany is Berlin. The capital of Italy is Rome.', tone: 'good' },
      ],
      note: "A base model knows a lot, but it's a document continuer: it writes whatever text is likely to come next.",
    },
    action: { label: 'Compare your model to real ones in Setup', tab: 'setup' },
    links: [
      { label: 'Chinchilla: compute-optimal training (Hoffmann et al.)', url: 'https://arxiv.org/abs/2203.15556' },
      { label: 'Scaling laws (Kaplan et al.)', url: 'https://arxiv.org/abs/2001.08361' },
      { label: 'FineWeb: building a pre-training dataset', url: 'https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1' },
      { label: "Let's build GPT (Andrej Karpathy, video)", url: 'https://www.youtube.com/watch?v=kCc8FmEb1nY' },
      { label: 'nanoGPT', url: 'https://github.com/karpathy/nanoGPT' },
    ],
  },
  {
    id: 'sft',
    pill: 'Fine-tune',
    title: 'Supervised fine-tuning (SFT)',
    icon: <MessageSquare size={20} color="var(--primary)" />,
    tagline: 'Teach the base model to answer instead of continue, by training it on example conversations.',
    body: [
      <>
        Ask a base model a question and it often doesn't answer. It continues the document, and on the web a question is
        often followed by more questions. To get an assistant, you keep training on a much smaller set of example
        conversations, written by people or generated by other models, showing exactly how a good reply looks.
      </>,
      <>
        Training works exactly as on the Train tab: predict the next token, measure the loss, backpropagate. Two things
        differ. Conversations are wrapped in a <b>chat template</b>, with special tokens added to the vocabulary that mark
        where each turn starts and ends. And the loss only counts the <b>assistant's</b> tokens, so the model learns to
        write replies, not to imitate users.
      </>,
      <>
        LLMStead's <b>Simple Q&A Conversations</b> dataset is a tiny taste of this: it's formatted as <code>User:</code> /{' '}
        <code>Assistant:</code> turns, so a model trained on it learns to answer after <code>Assistant:</code>. The
        difference is that it learns that format from scratch, instead of starting from a base model that already knows
        language.
      </>,
    ],
    terms: [
      ['SFT', 'Supervised fine-tuning. "Fine-tuning": continuing to train an already-trained model on a smaller, specialized dataset. "Supervised": every example comes with the exact answer to imitate.'],
      ['Base model', 'The model straight out of pre-training: it continues text, but hasn\'t been taught to follow instructions or answer questions.'],
      ['Special token', 'A token added to the vocabulary that never appears in ordinary text, like <|user|>, used to mark structure such as where a turn starts and ends.'],
      ['Chat template', 'The fixed recipe that turns a conversation (who said what) into one flat sequence of tokens, special tokens included.'],
    ],
    here: [
      'Q&A dataset uses plain "User:" and "Assistant:" text',
      'Trained from random weights on the conversations alone',
      'Loss counts every token, including the user turns',
    ],
    real: [
      'Starts from a pre-trained base model, so it already knows language and facts',
      'Tens of thousands to millions of example conversations',
      'Special tokens mark the turns; only assistant tokens count toward the loss',
      'Hours to days, a tiny fraction of pre-training compute',
    ],
    example: {
      title: 'Same prompt, before and after SFT',
      lines: [
        { label: 'Prompt', text: 'What is the boiling point of water?' },
        { label: 'Base model', text: ' What is the freezing point of water? What is the density of water? These questions…', tone: 'bad' },
        { label: 'After SFT', text: 'Water boils at 100 °C (212 °F) at sea level. At higher altitudes it boils at a lower temperature.', tone: 'good' },
        { label: 'Under the hood', text: '<|user|>What is the boiling point of water?<|end|><|assistant|>Water boils at…<|end|>' },
      ],
      note: 'The exact special tokens differ from model to model; the chat template turns a conversation into this flat token sequence.',
    },
    action: { label: 'Try the Q&A dataset in Setup', tab: 'setup' },
    links: [
      { label: 'InstructGPT: following instructions with human feedback (Ouyang et al.)', url: 'https://arxiv.org/abs/2203.02155' },
      { label: 'Chat templates (Hugging Face docs)', url: 'https://huggingface.co/docs/transformers/main/en/chat_templating' },
    ],
  },
  {
    id: 'preference',
    pill: 'Preference & RL',
    title: 'Preference tuning and reinforcement learning',
    icon: <ThumbsUp size={20} color="var(--accent-emerald)" />,
    tagline: 'Push the model toward answers people prefer, and toward answers that are actually correct.',
    body: [
      <>
        SFT shows the model good answers, but it's hard to write an example for every situation, and “good” is often a
        judgment call. So the next step trains on <b>comparisons</b>: show the model's two answers to the same prompt and
        record which one is better.
      </>,
      <>
        <b>RLHF</b> (reinforcement learning from human feedback) trains a separate <b>reward model</b> on those comparisons,
        so it can score any answer. The assistant then generates answers, gets scored, and is nudged toward higher scores,
        with a penalty for drifting too far from the SFT model so it doesn't learn to game the scorer.{' '}
        <b>DPO</b> (direct preference optimization) skips the reward model and trains on the pairs directly.{' '}
        <b>Constitutional AI</b> replaces many of the human judgments with AI feedback guided by a written set of
        principles.
      </>,
      <>
        For tasks with checkable answers, like math and code, you don't need a judge at all: run the tests or check the
        number. <b>Reinforcement learning with verifiable rewards</b> lets a model try many approaches and rewards the
        ones that reach the right answer. This is how “reasoning” models learn to think step by step before answering.
        LLMStead's math dataset is exactly this kind of data: every answer can be checked automatically.
      </>,
    ],
    terms: [
      ['RL', 'Reinforcement learning: learning from a score (a "reward") for what the model did, instead of from an example answer to copy.'],
      ['RLHF', 'Reinforcement learning from human feedback: the reward comes from a reward model trained on people\'s choices between pairs of answers.'],
      ['Reward model', 'A separate network that reads a prompt and an answer and outputs one number: how much people would likely prefer that answer.'],
      ['DPO', 'Direct preference optimization: skips the reward model and adjusts the model straight from the pairs, raising the probability of the preferred answer relative to the rejected one.'],
      ['RLAIF', 'Reinforcement learning from AI feedback: an AI model does the comparing, guided by a written list of principles. Constitutional AI is Anthropic\'s version.'],
      ['Verifiable reward', 'A reward a program can check automatically: run the code\'s tests, or compare the final number with the right answer.'],
      ['Reasoning model', 'A model trained (often with verifiable rewards) to write out intermediate steps before its final answer, sometimes called "thinking".'],
    ],
    here: [
      'Only learns from the text it was shown',
      'No notion of a better or worse answer, only a likely next token',
      "The benchmark checks answers, but the result doesn't feed back into training",
    ],
    real: [
      'Comparisons from people (or AI) between pairs of answers',
      'A reward model, or DPO directly, turns them into a training signal',
      'Verifiable rewards for math and code: automatic checks become the reward',
      'Trains helpfulness, honesty and safety behavior, not just knowledge',
    ],
    example: {
      title: 'One comparison, as a labeler sees it',
      lines: [
        { label: 'Prompt', text: 'My code throws "list index out of range". What does that mean?' },
        { label: 'Preferred', text: "It means you're reading a position the list doesn't have, like item 5 of a 5-item list (positions run 0–4). Check your loop bounds.", tone: 'good' },
        { label: 'Rejected', text: 'That is an error. You should fix your code so the error does not happen.', tone: 'bad' },
      ],
      note: 'Thousands of pairs like this, from many people, teach the reward model what "better" means.',
    },
    links: [
      { label: 'Illustrating RLHF (Hugging Face)', url: 'https://huggingface.co/blog/rlhf' },
      { label: 'DPO paper (Rafailov et al.)', url: 'https://arxiv.org/abs/2305.18290' },
      { label: 'Constitutional AI (Anthropic)', url: 'https://arxiv.org/abs/2212.08073' },
      { label: 'DeepSeek-R1: reasoning via RL', url: 'https://arxiv.org/abs/2501.12948' },
    ],
  },
  {
    id: 'evaluate',
    pill: 'Evaluate',
    title: 'Evaluate',
    icon: <ClipboardCheck size={20} color="var(--accent-amber)" />,
    tagline: 'Measure what the model can actually do, on questions it has never seen.',
    body: [
      <>
        Training loss says how well a model predicts its training text, not whether it's useful. So models are tested on{' '}
        <b>benchmarks</b>: fixed sets of questions with known answers, such as MMLU (multiple-choice exam questions across
        57 subjects), GSM8K (grade-school math word problems), HumanEval (writing small functions that must pass tests) and
        SWE-bench (fixing real bugs in open-source projects).
      </>,
      <>
        The big pitfall is <b>contamination</b>: if benchmark questions leaked into the training data, a high score may
        just be memorization. That's exactly what LLMStead's benchmark shows you, by labeling each case as <i>seen</i> (the
        answer is in the training text) or <i>held-out</i>. Real labs check for overlap, keep private test sets, and also
        rely on human comparisons, such as arenas where people vote between two anonymous models, plus safety testing and
        red-teaming to find harmful behavior before release.
      </>,
    ],
    terms: [
      ['Benchmark', 'A fixed set of questions with known answers, scored automatically, so different models can be compared on the same test.'],
      ['MMLU', 'Massive Multitask Language Understanding: about 16,000 multiple-choice exam questions across 57 subjects, from law to physics.'],
      ['GSM8K', 'Grade School Math 8K: about 8,500 grade-school math word problems. The answer is a number, so it\'s easy to check.'],
      ['HumanEval', '164 small programming tasks; the model writes a function, and it passes only if the function passes hidden tests.'],
      ['SWE-bench', 'Software engineering benchmark: real bug reports from open-source Python projects. The model\'s fix counts only if the project\'s own tests pass.'],
      ['Contamination', 'When test questions (or their answers) leaked into the training data, so a high score may be memorization.'],
      ['Red-teaming', 'Deliberately trying to make a model misbehave before release, borrowed from security, where the "red team" plays the attacker.'],
    ],
    here: [
      'A small prompt → expected-answer suite per dataset',
      'Each case labeled seen vs held-out by searching the training text',
      'Greedy decoding; a case passes if the output starts with the expected answer as a whole word',
    ],
    real: [
      'Dozens of public benchmarks plus private internal ones',
      'Contamination checks against the pre-training data',
      'Human preference votes and side-by-side comparisons',
      'Safety evaluations and red-teaming before release',
    ],
    example: {
      title: 'Why the seen / held-out split matters',
      lines: [
        { label: 'Seen', text: '1 + 2 = 3  ✓  (this exact line is in the training text, many times)', tone: 'good' },
        { label: 'Held-out', text: '2 + 1 = 3  ✗  (never in the training text)', tone: 'bad' },
      ],
      note: 'A model that only passes the seen cases has memorized, not learned. Real benchmarks have the same problem at a much larger scale.',
    },
    action: { label: 'Run the benchmark on the Train tab', tab: 'training' },
    links: [
      { label: 'MMLU paper (Hendrycks et al.)', url: 'https://arxiv.org/abs/2009.03300' },
      { label: 'GSM8K paper (Cobbe et al.)', url: 'https://arxiv.org/abs/2110.14168' },
      { label: 'LMArena: human preference leaderboard', url: 'https://lmarena.ai/' },
    ],
  },
  {
    id: 'deploy',
    pill: 'Deploy',
    title: 'Deploy',
    icon: <Rocket size={20} color="var(--accent-rose)" />,
    tagline: 'Make a trained model fast and cheap enough to answer millions of people.',
    body: [
      <>
        A finished model still has to run. <b>Quantization</b> stores each weight in fewer bits (8 or 4 instead of 16), so
        the model needs a fraction of the memory with only a small loss in quality. That's also how smaller open models fit
        on a laptop.
      </>,
      <>
        Generating text means running the model once per new token. The <b>KV cache</b> avoids redoing work: the keys and
        values you saw in Look inside are stored for every earlier token, so each new token only computes its own query,
        key and value and attends over the cache. Servers also <b>batch</b> many users' requests together so the GPU stays
        busy.
      </>,
      <>
        Finally comes <b>sampling</b>: picking the next token from the probabilities. That's the temperature slider on the
        Train tab. Low temperature picks the likeliest token almost every time; higher temperature adds variety. Deployed
        assistants also add system prompts, tool use and safety filters around the model itself.
      </>,
    ],
    terms: [
      ['Bit', 'One binary digit. Fewer bits per weight means each number is stored more coarsely, but the whole model takes less memory.'],
      ['Quantization', 'Rounding every weight to a coarser format after training (16 bits down to 8 or 4), trading a little accuracy for a much smaller, faster model.'],
      ['KV cache', 'KV stands for keys and values, the attention vectors from Look inside. Saving them for earlier tokens means each new token only computes its own.'],
      ['Batching', 'Running many requests through the model together, since a GPU is far faster doing many multiplications at once than one at a time.'],
      ['Temperature', 'How much to flatten or sharpen the probabilities before picking the next token. Low: almost always the top choice. High: more variety.'],
      ['System prompt', 'Instructions placed before the conversation that the user usually doesn\'t see, like "You are a helpful assistant".'],
      ['Tool use', 'The model writes a structured request (search the web, run code); the app performs it and feeds the result back as more text.'],
    ],
    here: [
      'Weights are full 64-bit JavaScript numbers',
      'Generation re-runs the whole sequence for each new token',
      'One user, one request, on your CPU',
    ],
    real: [
      'Weights quantized to 8 or 4 bits',
      'KV cache, so each new token only does a little new work',
      'Requests from many users batched together on GPUs',
      'System prompts, tools and safety filters around the model',
    ],
    example: {
      title: 'Same weights, different temperature',
      lines: [
        { label: 'T = 0 (greedy)', text: 'The cat sat on the mat. The cat sat on the mat. The cat sat…' },
        { label: 'T = 0.7', text: 'The cat sat on the windowsill, watching the rain.', tone: 'good' },
        { label: 'T = 1.5', text: 'The cat sat quantum orange beneath Tuesday’s spoon.', tone: 'bad' },
      ],
      note: 'Illustrative outputs: too cold repeats itself, too hot loses the thread.',
    },
    action: { label: 'Try the temperature slider on the Train tab', tab: 'training' },
    links: [
      { label: 'Quantization overview (Hugging Face docs)', url: 'https://huggingface.co/docs/transformers/main/en/quantization/overview' },
      { label: 'KV cache from scratch (Hugging Face)', url: 'https://huggingface.co/blog/kv-cache' },
    ],
  },
];

const TONE_COLOR = { good: 'var(--accent-emerald)', bad: 'var(--accent-rose)' };

const scrollToStage = (id: string) => document.getElementById(`next-${id}`)?.scrollIntoView({ block: 'start' });

const Pill: React.FC<{ label: string; here?: boolean; onClick?: () => void }> = ({ label, here, onClick }) => (
  <button
    onClick={onClick}
    disabled={!onClick}
    style={{
      padding: '6px 12px',
      borderRadius: 999,
      fontSize: '0.8rem',
      fontWeight: 600,
      cursor: onClick ? 'pointer' : 'default',
      border: `1px solid ${here ? 'var(--primary)' : 'var(--border-color)'}`,
      background: here ? 'var(--primary)' : 'var(--surface-inset)',
      color: here ? '#ffffff' : 'var(--text-main)',
    }}
  >
    {label}
    {here && <span style={{ fontWeight: 400, opacity: 0.85 }}> · you are here</span>}
  </button>
);

const BulletList: React.FC<{ heading: string; items: string[]; accent: string }> = ({ heading, items, accent }) => (
  <div style={{ background: 'var(--surface-inset)', border: '1px solid var(--border-color)', borderRadius: 8, padding: '12px 14px' }}>
    <div style={{ fontSize: '0.7rem', fontWeight: 700, color: accent, textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 8 }}>
      {heading}
    </div>
    <ul style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 4, fontSize: '0.83rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
      {items.map(item => <li key={item}>{item}</li>)}
    </ul>
  </div>
);

const StageCard: React.FC<{ stage: Stage; index: number; onNavigate: (tab: string) => void }> = ({ stage, index, onNavigate }) => (
  <section id={`next-${stage.id}`} className="glass-panel" style={{ padding: 24, scrollMarginTop: 16, display: 'flex', flexDirection: 'column', gap: 14 }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span
        className="font-mono"
        style={{ width: 24, height: 24, borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', fontWeight: 700, background: 'var(--bg-card-hover)', color: 'var(--text-muted)', flexShrink: 0 }}
      >
        {index + 1}
      </span>
      {stage.icon}
      <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>{stage.title}</h2>
    </div>
    <p style={{ fontWeight: 600, color: 'var(--text-main)', lineHeight: 1.5 }}>{stage.tagline}</p>
    {stage.body.map((para, i) => (
      <p key={i} style={{ fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: 1.65 }}>{para}</p>
    ))}
    <KeyPanel title="Terms used here" items={stage.terms} />

    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))', gap: 12 }}>
      <BulletList heading="In LLMStead" items={stage.here} accent="var(--accent-amber)" />
      <BulletList heading="In a real pipeline" items={stage.real} accent="var(--primary)" />
    </div>

    <div style={{ borderRadius: 8, border: '1px solid var(--border-color)', overflow: 'hidden' }}>
      <div style={{ padding: '8px 14px', fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-dim)', background: 'var(--surface-inset)', borderBottom: '1px solid var(--border-color)' }}>
        Example · {stage.example.title}
      </div>
      <div style={{ padding: '10px 14px', display: 'grid', gridTemplateColumns: 'minmax(80px, max-content) 1fr', gap: '8px 14px', fontSize: '0.83rem', lineHeight: 1.5 }}>
        {stage.example.lines.map(line => (
          <React.Fragment key={line.label}>
            <span style={{ fontWeight: 600, color: line.tone ? TONE_COLOR[line.tone] : 'var(--text-dim)' }}>{line.label}</span>
            <span className="font-mono" style={{ color: 'var(--text-main)', whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: '0.8rem' }}>{line.text}</span>
          </React.Fragment>
        ))}
      </div>
      {stage.example.note && (
        <div style={{ padding: '0 14px 10px', fontSize: '0.75rem', color: 'var(--text-dim)', lineHeight: 1.5 }}>{stage.example.note}</div>
      )}
    </div>

    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
      <GoDeeper links={stage.links} />
      {stage.action && (
        <button className="btn-secondary" onClick={() => onNavigate(stage.action!.tab)} style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
          {stage.action.label} <ArrowRight size={14} />
        </button>
      )}
    </div>
  </section>
);

export const WhatsNextView: React.FC<WhatsNextViewProps> = ({ onNavigate }) => (
  <div style={{ maxWidth: 960, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 24 }}>
    {/* Intro + where you are in the pipeline */}
    <div className="glass-panel" style={{ padding: '28px 28px 24px', display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <Compass size={22} color="var(--primary)" />
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700 }}>What's next: from base model to assistant</h2>
      </div>
      <p style={{ color: 'var(--text-muted)', lineHeight: 1.65 }}>
        What you trained here is a <b>base model</b>: a next-token predictor that has soaked up the patterns in its training
        text. Every chat assistant starts as one of these, just far bigger. The stages below are what happens next in a real
        pipeline. Notice how little the architecture changes from here on: almost everything that turns a base model into
        an assistant comes from <b>different data and different training goals</b>.
      </p>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
        <Pill label="Pre-training" here />
        {STAGES.map(s => (
          <React.Fragment key={s.id}>
            <ArrowRight size={14} color="var(--text-dim)" />
            <Pill label={s.pill} onClick={() => scrollToStage(s.id)} />
          </React.Fragment>
        ))}
      </div>
    </div>

    {STAGES.map((stage, i) => (
      <StageCard key={stage.id} stage={stage} index={i} onNavigate={onNavigate} />
    ))}

    {/* Closing */}
    <div className="glass-panel" style={{ padding: 24, display: 'flex', gap: 14, alignItems: 'flex-start' }}>
      <BookOpen size={20} color="var(--accent-purple)" style={{ flexShrink: 0, marginTop: 2 }} />
      <div style={{ fontSize: '0.88rem', color: 'var(--text-muted)', lineHeight: 1.65 }}>
        <b style={{ color: 'var(--text-main)' }}>Keep going.</b> The footer lists more interactive explainers, and every
        stage above links to the papers and write-ups behind it. LLMStead may grow a second tier someday, where you
        fine-tune a real small model, but the mechanics you saw here (tokens, embeddings, attention, the loss and
        backpropagation) are the same ones every stage above is built on.
      </div>
    </div>
  </div>
);
