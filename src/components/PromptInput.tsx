import React from 'react';

/** Apple keyboards use ⌘ where everyone else uses Ctrl. userAgentData is newer; platform is the fallback. */
const IS_APPLE =
  typeof navigator !== 'undefined' &&
  /Mac|iPhone|iPad|iPod/i.test(
    (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData?.platform || navigator.platform || navigator.userAgent
  );

/** The run shortcut as it's written on this keyboard */
export const RUN_SHORTCUT = IS_APPLE ? '⌘ Enter' : 'Ctrl Enter';

/** The run shortcut as a small key chip, for button labels */
export const ShortcutKey: React.FC = () => <kbd className="kbd">{RUN_SHORTCUT}</kbd>;

/**
 * Multi-line prompt box. Prompts often need real line breaks (the Q&A data is
 * "User: …\nAssistant: …"), and a single-line <input> silently strips them, so this is a
 * textarea: Enter types a newline, Ctrl/⌘+Enter runs `onSubmit`. It grows with its content.
 */
export const PromptInput: React.FC<{
  value: string;
  onChange: (value: string) => void;
  onSubmit?: () => void;
  placeholder?: string;
  style?: React.CSSProperties;
}> = ({ value, onChange, onSubmit, placeholder, style }) => (
  <textarea
    value={value}
    placeholder={placeholder}
    onChange={e => onChange(e.target.value)}
    onKeyDown={e => {
      if (onSubmit && e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        onSubmit();
      }
    }}
    rows={Math.min(6, Math.max(2, value.split('\n').length))}
    spellCheck={false}
    className="font-mono"
    style={{ width: '100%', resize: 'vertical', lineHeight: 1.5, ...style }}
  />
);

/** `live`: the box has no Run button because its view updates as you type */
export const PromptHint: React.FC<{ live?: boolean }> = ({ live }) => (
  <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
    Enter adds a new line (the model reads line breaks like any other character)
    {live ? '. Everything below updates as you type.' : <>; {RUN_SHORTCUT} runs.</>}
  </span>
);
