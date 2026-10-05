import React from 'react';
import { Home, Compass, Download, Type, Images, MessageSquareText } from 'lucide-react';
import { ChickenLogo } from './ChickenLogo';

export interface NavTab {
  id: string;
  label: string;
  step: number | null;
}

/** Which model the site is showing: the text GPT, the image + text CLIP, or the vision-language model built on it */
export type ModelMode = 'gpt' | 'clip' | 'vlm';

export interface ModeSwitchProps {
  mode: ModelMode;
  onChange: (mode: ModelMode) => void;
  /** Whether each mode is training right now (it keeps going while the other is shown) */
  training: Record<ModelMode, boolean>;
}

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isTraining: boolean;
  /** Post-training is running (a pulsing dot on its tab, like Train's) */
  isPostTraining?: boolean;
  tabs?: readonly NavTab[];
  modeSwitch: ModeSwitchProps;
  /** Opens the export dialog (download the model as a GGUF file for Ollama); no button without it */
  onExport?: () => void;
}

const MODES: { id: ModelMode; label: string; hint: string; icon: React.ReactNode }[] = [
  { id: 'gpt', label: 'Text · GPT', hint: 'A tiny language model that learns to predict the next token', icon: <Type size={14} /> },
  { id: 'clip', label: 'Images · CLIP', hint: 'A tiny CLIP that learns which captions match which pictures', icon: <Images size={14} /> },
  { id: 'vlm', label: 'Images → Text · VLM', hint: 'A tiny vision-language model, built on your CLIP, that answers questions about pictures', icon: <MessageSquareText size={14} /> },
];

/** Switch between the models. Each stays loaded once opened, so switching never loses training progress. */
const ModeSwitch: React.FC<ModeSwitchProps> = ({ mode, onChange, training }) => (
  <div role="group" aria-label="Model" style={{ display: 'flex', gap: 4, background: 'var(--surface-inset)', padding: 3, borderRadius: 8, border: '1px solid var(--border-color)' }}>
    {MODES.map(m => {
      const active = m.id === mode;
      return (
        <button
          key={m.id}
          onClick={() => onChange(m.id)}
          title={m.hint}
          aria-pressed={active}
          style={{
            display: 'flex', alignItems: 'center', gap: 6, padding: '4px 10px', borderRadius: 6, border: 'none', cursor: 'pointer',
            fontSize: '0.75rem', fontWeight: 600,
            background: active ? 'var(--primary)' : 'transparent',
            color: active ? '#ffffff' : 'var(--text-muted)',
          }}
        >
          {m.icon}
          {m.label}
          {!active && training[m.id] && (
            <span
              title="Still training in the background"
              style={{ width: 6, height: 6, borderRadius: 3, background: 'var(--accent-emerald)', animation: 'pulse 1.2s ease-in-out infinite' }}
            />
          )}
        </button>
      );
    })}
  </div>
);

/** The text model's path, in order. Start is the intro, the middle four are numbered steps, and What's next is reading. */
export const TABS: readonly NavTab[] = [
  { id: 'start', label: 'Start here', step: null },
  { id: 'setup', label: 'Set up', step: 1 },
  { id: 'training', label: 'Train', step: 2 },
  { id: 'pipeline', label: 'Look inside', step: 3 },
  { id: 'posttrain', label: 'Post-train', step: 4 },
  { id: 'next', label: "What's next", step: null },
];

/** The image models' path: the same steps without post-training, which is about text models */
export const IMAGE_TABS: readonly NavTab[] = TABS.filter(t => t.id !== 'posttrain');

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  isTraining,
  isPostTraining = false,
  tabs = TABS,
  modeSwitch,
  onExport,
}) => {

  return (
    <header className="glass-panel" style={{ borderRadius: '0 0 10px 10px', borderTop: 'none', padding: '12px 24px', marginBottom: '24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        {/* Brand Logo & Title */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
          <div style={{
            background: 'var(--primary)',
            padding: '8px',
            borderRadius: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}>
            <ChickenLogo size={22} style={{ color: '#ffffff' }} />
          </div>
          <div>
            <h1 style={{ fontSize: '1.2rem', fontWeight: 700, letterSpacing: '-0.01em', color: 'var(--text-main)' }}>
              LLMStead
            </h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Raise your own models
            </p>
          </div>
          <ModeSwitch {...modeSwitch} />
        </div>

        {/* Tab Navigation */}
        <nav style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', background: 'var(--surface-inset)', padding: '4px', borderRadius: '10px', border: '1px solid var(--border-color)' }}>
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 14px',
                  borderRadius: '7px',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  background: isActive ? 'var(--bg-card-hover)' : 'transparent',
                  color: isActive ? 'var(--text-main)' : 'var(--text-muted)',
                  boxShadow: 'none'
                }}
              >
                {tab.step === null ? (
                  tab.id === 'start' ? <Home size={15} /> : <Compass size={15} />
                ) : (
                  <span
                    className="font-mono"
                    style={{
                      width: 18,
                      height: 18,
                      borderRadius: 9,
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      background: isActive ? 'var(--primary)' : 'var(--bg-card-hover)',
                      color: isActive ? '#ffffff' : 'var(--text-muted)',
                    }}
                  >
                    {tab.step}
                  </span>
                )}
                <span>{tab.label}</span>
                {((tab.id === 'training' && isTraining) || (tab.id === 'posttrain' && isPostTraining)) && (
                  <span
                    title={tab.id === 'training' ? 'Training is running' : 'Post-training is running'}
                    style={{ width: 7, height: 7, borderRadius: 4, background: 'var(--accent-emerald)', animation: 'pulse 1.2s ease-in-out infinite' }}
                  />
                )}
              </button>
            );
          })}
        </nav>

        {onExport && (
          <button className="btn-secondary" onClick={onExport} title="Download your model to run in Ollama or llama.cpp">
            <Download size={15} /> Export Model
          </button>
        )}
      </div>
    </header>
  );
};
