import React, { useEffect, useState } from 'react';
import { Download, Copy, Check, X } from 'lucide-react';
import { DatasetOption } from '../types';
import { MicroTransformer } from '../engine/transformer';
import { BPETokenizer } from '../engine/bpeTokenizer';
import { exportGguf } from '../engine/gguf';
import { samplePromptFor } from '../engine/datasets';
import { InfoTooltip } from './InfoTooltip';
import { GoDeeper } from './GoDeeper';

interface ExportModalProps {
  open: boolean;
  onClose: () => void;
  model: MicroTransformer;
  tokenizer: BPETokenizer;
  selectedDataset: DatasetOption;
  stepCount: number;
  /** While the background worker trains, the page's model is a stale copy */
  trainingBusy: boolean;
}

function download(name: string, data: BlobPart, type: string) {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const codeStyle: React.CSSProperties = {
  margin: 0,
  padding: '12px 14px',
  background: 'var(--bg)',
  border: '1px solid var(--border-color)',
  borderRadius: 8,
  fontSize: '0.8rem',
  color: 'var(--text-main)',
  whiteSpace: 'pre-wrap',
  wordBreak: 'break-word',
  lineHeight: 1.5,
};

const CodeBlock: React.FC<{ text: string }> = ({ text }) => {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }, () => {});
  };
  return (
    <div style={{ position: 'relative' }}>
      <pre className="font-mono" style={codeStyle}>{text}</pre>
      <button
        className="btn-secondary"
        onClick={copy}
        title="Copy"
        style={{ position: 'absolute', top: 6, right: 6, padding: '4px 8px', fontSize: '0.75rem' }}
      >
        {copied ? <Check size={13} /> : <Copy size={13} />}
      </button>
    </div>
  );
};

/**
 * Ollama model names allow lowercase letters, digits, "-", "_" and "."; the same name makes a safe
 * file name everywhere. Anything else becomes "-".
 */
function cleanName(raw: string): string {
  return raw.trim().toLowerCase().replace(/.gguf$/, '').replace(/[^a-z0-9._-]+/g, '-').replace(/^[-._]+|[-._]+$/g, '');
}

const Step: React.FC<{ n: number; children: React.ReactNode }> = ({ n, children }) => (
  <div style={{ display: 'flex', gap: 10, alignItems: 'baseline' }}>
    <span className="font-mono" style={{ color: 'var(--accent-cyan)', fontWeight: 700, fontSize: '0.85rem' }}>{n}.</span>
    <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 6, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
      {children}
    </div>
  </div>
);


type AppId = 'ollama' | 'lmstudio' | 'llamacpp';
const APPS: { id: AppId; label: string }[] = [
  { id: 'ollama', label: 'Ollama' },
  { id: 'lmstudio', label: 'LM Studio' },
  { id: 'llamacpp', label: 'llama.cpp' },
];
const APP_KEY = 'llmstead.exportApp';

function savedApp(): AppId {
  try {
    const v = localStorage.getItem(APP_KEY);
    return APPS.some(a => a.id === v) ? (v as AppId) : 'ollama';
  } catch {
    return 'ollama';
  }
}

/** Shared generation setting, so every app's instructions match */
const TEMPERATURE = 0.7;

/**
 * A dialog, opened from the navbar's Export button, to download the trained model as a GGUF file
 * and run it in Ollama, LM Studio or llama.cpp. The settings matter: the model only knows
 * contextWindow positions (the file pads its position table so longer contexts don't crash, but
 * only the trained ones match the browser), and it never learned to stop.
 */
export const ExportModal: React.FC<ExportModalProps> = ({ open, onClose, model, tokenizer, selectedDataset, stepCount, trainingBusy }) => {
  // null = not edited, so the name follows the dataset
  const [customName, setCustomName] = useState<string | null>(null);
  const [app, setApp] = useState<AppId>(savedApp);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);
  if (!open) return null;

  const chooseApp = (id: AppId) => {
    setApp(id);
    try {
      localStorage.setItem(APP_KEY, id);
    } catch {
      // private window: the choice just isn't remembered
    }
  };

  const { contextWindow, numLayers } = model.config;
  const defaultName = `llmstead-${selectedDataset.id}`;
  const slug = cleanName(customName ?? '') || defaultName;
  const file = `${slug}.gguf`;
  const prompt = samplePromptFor(selectedDataset).trimEnd();
  // Past its window the model runs on reused positions (see the note at the bottom), so replies
  // stop about where its memory ends. It also never learned to stop by itself.
  const maxReply = contextWindow;

  const blocked = numLayers === 0
    ? 'llama.cpp and the apps built on it need at least one transformer block. Add one in Set up.'
    : trainingBusy
      ? 'Pause training first, so the export gets the latest weights.'
      : null;

  const exportModel = () => {
    const bytes = exportGguf(model, tokenizer, slug);
    download(file, bytes, 'application/octet-stream');
  };

  const modelfile = [
    `FROM ./${file}`,
    '',
    '# No chat formatting: the model just continues your text',
    'TEMPLATE """{{ .Prompt }}"""',
    '',
    `# It was trained on ${contextWindow}-token windows: longer prompts keep only their end`,
    `PARAMETER num_ctx ${contextWindow}`,
    "# It never learned to stop, and gets worse past its window, so keep replies short",
    `PARAMETER num_predict ${maxReply}`,
    `PARAMETER temperature ${TEMPERATURE}`,
    '# Ollama penalizes repeated tokens by default, which would skew its predictions',
    'PARAMETER repeat_penalty 1',
  ].join('\n');

  const settingsList = (items: [string, string][]) => (
    <ul style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 3 }}>
      {items.map(([k, v]) => (
        <li key={k}><b style={{ color: 'var(--text-main)', fontWeight: 600 }}>{k}</b>: {v}</li>
      ))}
    </ul>
  );

  const appSteps: Record<AppId, React.ReactNode> = {
    ollama: (
      <>
        <Step n={2}>
          <span>
            In the folder with the file, save this as <span className="font-mono">Modelfile</span> (or{' '}
            <a href="#" onClick={e => { e.preventDefault(); download('Modelfile', modelfile + '\n', 'text/plain'); }}>download it</a>):
          </span>
          <CodeBlock text={modelfile} />
        </Step>
        <Step n={3}>
          <span>From that folder, import it and give it a prompt:</span>
          <CodeBlock text={`ollama create ${slug} -f Modelfile\nollama run ${slug} ${JSON.stringify(prompt)}`} />
        </Step>
      </>
    ),
    lmstudio: (
      <>
        <Step n={2}>
          <span>
            Import the file with LM Studio's command-line tool and follow its prompts (or move the file to{' '}
            <span className="font-mono" style={{ overflowWrap: "anywhere" }}>~/.lmstudio/models/llmstead/{slug}/</span> yourself):
          </span>
          <CodeBlock text={`lms import ./${file}`} />
        </Step>
        <Step n={3}>
          <span>Load {slug} in the chat view with these settings:</span>
          {settingsList([
            ['Context Length', `${contextWindow} (in the load settings), the window size it was trained on`],
            ['Limit Response Length', `on, ${maxReply} tokens: it never learned to stop, and gets worse past its window`],
            ['Temperature', String(TEMPERATURE)],
            ['Repeat Penalty', "off (1), so its predictions aren't skewed"],
          ])}
          <span>
            Then type a prompt such as <span className="font-mono">{prompt}</span>. There's no chat formatting: your messages
            and its replies are joined into one running text that it keeps continuing, so start a new chat for each new prompt.
          </span>
        </Step>
      </>
    ),
    llamacpp: (
      <>
        <Step n={2}>
          <span>Start llama.cpp's server with the file:</span>
          <CodeBlock text={`llama-server -m ./${file}`} />
        </Step>
        <Step n={3}>
          <span>
            Open <span className="font-mono">http://localhost:8080</span> for a chat page, or ask for a plain continuation
            from another terminal:
          </span>
          <CodeBlock
            text={`curl http://localhost:8080/completion -d '${JSON.stringify({ prompt, n_predict: maxReply, temperature: TEMPERATURE })}'`}
          />
        </Step>
      </>
    ),
  };

  const tabStyle = (active: boolean): React.CSSProperties => ({
    padding: '6px 14px',
    borderRadius: 6,
    border: '1px solid var(--border-color)',
    fontSize: '0.8rem',
    fontWeight: 600,
    cursor: 'pointer',
    background: active ? 'var(--primary)' : 'var(--surface-inset)',
    color: active ? '#ffffff' : 'var(--text-muted)',
  });

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 50, display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
        padding: '6vh 16px', overflowY: 'auto', background: 'color-mix(in srgb, var(--bg) 70%, transparent)', backdropFilter: 'blur(3px)',
      }}
    >
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Export your model"
      className="glass-panel"
      onClick={e => e.stopPropagation()}
      style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: 16, width: '100%', maxWidth: 760, background: 'var(--bg-card)' }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Download size={20} color="var(--accent-emerald)" />
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Run your model outside LLMStead</h2>
          <InfoTooltip
            title="GGUF"
            description="GGUF is the single-file model format used by llama.cpp and the apps built on it, like Ollama and LM Studio. It holds the weights plus labels: which architecture this is, its sizes, and the tokenizer's vocabulary. There's no code inside; the app runs its own implementation of the named architecture."
            impact="This model is laid out exactly like GPT-2 (minus biases, which are exported as zeros), so it's exported as a gpt2 model and gives the same predictions in llama.cpp, up to rounding."
          />
        </div>
        <button className="btn-secondary" onClick={onClose} title="Close (Esc)" style={{ padding: '6px 8px' }}>
          <X size={16} />
        </button>
      </div>

      <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
        Real model releases ship the same way: a file of weights that any compatible app can load. Your model downloads as
        one <b>GGUF</b> file, which runs in <a href="https://ollama.com" target="_blank" rel="noreferrer">Ollama</a>,{' '}
        <a href="https://lmstudio.ai" target="_blank" rel="noreferrer">LM Studio</a> and{' '}
        <a href="https://github.com/ggml-org/llama.cpp" target="_blank" rel="noreferrer">llama.cpp</a>, the engine underneath both.
      </p>

      <label style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
        Model name
        <input
          type="text"
          value={customName ?? defaultName}
          onChange={e => setCustomName(e.target.value)}
          onBlur={() => customName !== null && setCustomName(slug === defaultName ? null : slug)}
          placeholder={defaultName}
          spellCheck={false}
          className="font-mono"
          style={{
            flex: '1 1 220px', minWidth: 0, padding: '6px 10px', fontSize: '0.85rem', borderRadius: 6,
            border: '1px solid var(--border-color)', background: 'var(--bg)', color: 'var(--text-main)',
          }}
        />
        {customName !== null && (
          <button className="btn-secondary" onClick={() => setCustomName(null)} style={{ padding: '4px 10px', fontSize: '0.75rem' }}>
            Reset
          </button>
        )}
      </label>
      <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: -10 }}>
        Used for the file and the name in the app: lowercase letters, digits, "-", "_" and "." (other characters become "-").
      </span>

      <Step n={1}>
        <span>Download the model ({selectedDataset.name}, step #{stepCount.toLocaleString()}):</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <button className="btn-primary" onClick={exportModel} disabled={blocked !== null} style={{ maxWidth: "100%", overflowWrap: "anywhere" }}>
            <Download size={16} /> Download {file}
          </button>
          {blocked && <span style={{ fontSize: '0.8rem', color: 'var(--accent-amber)' }}>{blocked}</span>}
        </div>
      </Step>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginRight: 4 }}>Run it in:</span>
        {APPS.map(a => (
          <button key={a.id} onClick={() => chooseApp(a.id)} style={tabStyle(app === a.id)} aria-pressed={app === a.id}>
            {a.label}
          </button>
        ))}
      </div>
      {appSteps[app]}

      <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)', lineHeight: 1.6 }}>
        Two differences from the playground. These apps don't do <b>token healing</b>, so a prompt ending in a space (like
        "= ") can throw the model off; leave the trailing space out. And the playground re-reads only the last{' '}
        {contextWindow} tokens before each new one, so the model always sees positions it was trained on. These apps
        instead keep every token in memory (the <b>KV cache</b>), so once prompt and reply pass {contextWindow} tokens,
        later tokens all reuse the last position the model learned. It keeps going, but the text falls apart, which is
        why the replies above stop at {maxReply} tokens.
      </p>
      <GoDeeper
        links={[
          { label: 'GGUF format spec', url: 'https://github.com/ggml-org/ggml/blob/master/docs/gguf.md' },
          { label: 'Ollama Modelfile reference', url: 'https://github.com/ollama/ollama/blob/main/docs/modelfile.md' },
          { label: 'Importing models into LM Studio', url: 'https://lmstudio.ai/docs/app/advanced/import-model' },
          { label: 'llama.cpp server', url: 'https://github.com/ggml-org/llama.cpp/tree/master/tools/server' },
        ]}
      />
    </div>
    </div>
  );
};
