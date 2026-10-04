// Autosave: the current model, its settings and its training history, kept in this browser's
// IndexedDB so a reload picks up where you left off. One record, overwritten on every save.
//
// IndexedDB rather than localStorage because it stores typed arrays directly (exact values, no
// JSON) and allows far more than localStorage's ~5 MB. Every function here fails softly: in a
// private window or with storage blocked, saving is simply unavailable.
import { DatasetOption, TransformerConfig } from './types';
import { ModelState } from './engine/transformer';

/** Bump when the saved shape changes; older saves are then ignored */
const SAVE_VERSION = 2;
/**
 * Version 1 saves came from the post-LayerNorm model. Their settings and datasets still load,
 * but the weights were trained for a different architecture, so they're marked outdated.
 */
const POST_LN_VERSION = 1;
const DB_NAME = 'llmstead';
const STORE = 'autosave';
const KEY = 'latest';

export interface TrainingHistory {
  stepCount: number;
  lossHistory: number[];
  valLossHistory: (number | null)[];
  stepLabels: string[];
}

export interface SavedSession {
  version: number;
  savedAt: number;
  config: TransformerConfig;            // target vocab size, as set in Setup
  selectedDatasetId: string;
  customDatasets: DatasetOption[];      // text added in Setup, so the dataset list survives too
  /**
   * The tokenizer's vocabulary (token text by id). The tokenizer is rebuilt from the dataset on
   * load; if a site update changed the dataset, the rebuilt vocabulary won't match and the saved
   * weights would map to the wrong tokens, so the save is discarded instead.
   */
  vocabulary: string[];
  testSentence: string;
  model: ModelState;
  history: TrainingHistory;
  /** Set on load for saves whose weights belong to an older architecture (see POST_LN_VERSION) */
  outdatedModel?: boolean;
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') return reject(new Error('IndexedDB unavailable'));
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
    req.onblocked = () => reject(new Error('IndexedDB blocked'));
  });
}

async function run<T>(mode: IDBTransactionMode, op: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openDb();
  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = db.transaction(STORE, mode);
      const req = op(tx.objectStore(STORE));
      tx.oncomplete = () => resolve(req.result);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  } finally {
    db.close();
  }
}

/** The saved session, or null if there is none, it's from an older version, or storage is unavailable */
export async function loadSession(): Promise<SavedSession | null> {
  try {
    const saved = (await run('readonly', s => s.get(KEY))) as SavedSession | undefined;
    if (saved && saved.version === POST_LN_VERSION) return { ...saved, outdatedModel: true };
    return saved && saved.version === SAVE_VERSION ? saved : null;
  } catch {
    return null;
  }
}

/** Returns false if the browser wouldn't store it (private window, storage blocked, disk full) */
export async function saveSession(session: Omit<SavedSession, 'version' | 'savedAt'>): Promise<boolean> {
  try {
    await run('readwrite', s => s.put({ ...session, version: SAVE_VERSION, savedAt: Date.now() }, KEY));
    return true;
  } catch {
    return false;
  }
}

export async function clearSession(): Promise<void> {
  try {
    await run('readwrite', s => s.delete(KEY));
  } catch {
    /* nothing saved, or storage unavailable: either way there's nothing to clear */
  }
}

/** What the Train tab shows about saving: last successful save, or that the browser refused */
export type SaveStatus = { kind: 'saved'; at: number } | { kind: 'unavailable' } | null;
