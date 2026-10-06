import React from 'react';
import { ArrowRight, Box, Brain, Compass, Repeat, Wrench } from 'lucide-react';
import { GoDeeper } from '../GoDeeper';

/**
 * The JEPA side's "What's next": what you keep after training, LeCun's case for world models,
 * how a JEPA becomes one, and where the research stands. Static content.
 */

const Section: React.FC<{ icon: React.ReactNode; title: string; children: React.ReactNode }> = ({ icon, title, children }) => (
  <div className="glass-panel" style={{ padding: '24px 28px' }}>
    <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: 12, display: 'flex', alignItems: 'center', gap: 8 }}>
      {icon} {title}
    </h3>
    <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.65, display: 'flex', flexDirection: 'column', gap: 10 }}>
      {children}
    </div>
  </div>
);

const B: React.FC<{ children: React.ReactNode }> = ({ children }) => <b style={{ color: 'var(--text-main)' }}>{children}</b>;

const Pill: React.FC<{ text: string; color: string; sub?: string }> = ({ text, color, sub }) => (
  <div style={{ padding: '8px 12px', borderRadius: 8, background: 'var(--surface-inset)', borderTop: `3px solid ${color}`, textAlign: 'center', minWidth: 100 }}>
    <div style={{ fontWeight: 700, fontSize: '0.8rem', color: 'var(--text-main)' }}>{text}</div>
    {sub && <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{sub}</div>}
  </div>
);

const Arr = () => <ArrowRight size={16} color="var(--text-dim)" style={{ flexShrink: 0 }} />;

/** LeCun's 2022 prescriptions, as "move away from → toward" pairs */
const SHIFTS: [string, string, string][] = [
  ['Generative models', 'Joint-embedding architectures', 'Predict in embedding space instead of reproducing every pixel or word.'],
  ['Probabilistic models', 'Energy-based models', 'Score how compatible two things are, rather than putting a probability on every possible outcome.'],
  ['Contrastive methods', 'Regularized methods', 'Prevent collapse by keeping embeddings spread out (like the VICReg term in the Train tab), not by pushing apart pairs of "negative" examples.'],
  ['Reinforcement learning', 'Model-predictive control', 'Plan by imagining outcomes with a world model; fall back on trial-and-error learning only when the model is wrong.'],
];

/** What a trained encoder gets used for, each with an example and why you'd skip the LLM there */
const USES: { title: string; example: string; whyNotLlm: string }[] = [
  {
    title: 'New tasks from a few labels',
    example:
      'A factory has millions of unlabelled photos of circuit boards and 80 labelled photos of bad solder joints. A frozen encoder plus a linear head trained on those 80 checks boards at camera speed, on a box beside the line. Medical imaging works the same way: pathology models pre-trained without labels on huge slide collections then learn to grade tumours from a few expert-labelled cases.',
    whyNotLlm:
      'The difference is a subtle visual pattern with no everyday name, the head gives the same calibrated score every time, and fifty images a second through a chatbot would be slow and costly.',
  },
  {
    title: 'Maps and coordinates (patch-level tasks)',
    example:
      'Each patch embedding keeps where it came from, so a light head on top can label every patch: forest canopy height from satellite photos, tumour outlines in a scan, or the point on an object where a robot gripper should close.',
    whyNotLlm: 'The answer is a map or a pixel position, not a sentence, and language is a lossy channel for "exactly where".',
  },
  {
    title: 'A head start for fine-tuning',
    example:
      'A conservation group fine-tunes a pre-trained encoder on 5,000 labelled camera-trap photos of local species, instead of the millions a model trained from scratch would need.',
    whyNotLlm: 'You end up owning a small model you can run offline in the field and retrain whenever a new species turns up.',
  },
  {
    title: 'Search and similarity',
    example:
      '"Find products that look like this photo", "more photos like this one", spotting near-duplicates in a dataset, or flagging a camera frame whose embedding is far from everything normal. The neighbours panel in Look inside is this, in miniature.',
    whyNotLlm: 'Each image is embedded once, in milliseconds, and searching millions of stored vectors is fast. An LLM would have to look at the images again for every query.',
  },
  {
    title: 'The eyes of a bigger system',
    example:
      'Vision-language models are mostly an image encoder feeding a language model: LLaVA, for example, puts an LLM on top of a CLIP encoder, and V-JEPA 2 was paired with an LLM to answer questions about videos. V-JEPA 2-AC kept its encoder frozen and trained a new action-conditioned predictor on top to plan robot movements.',
    whyNotLlm:
      'Here the encoder is part of the LLM system. The question is whether language belongs in the loop at all: a robot choosing motor commands 30 times a second gains little from turning what it sees into text first.',
  },
];

/** Encoder + small head vs a multimodal LLM */
const VERSUS: [string, string, string][] = [
  ['Labels needed', 'A handful per task', 'None: you describe the task in a prompt'],
  ['Output', 'Numbers: scores, maps, coordinates, vectors', 'Text'],
  ['Speed and cost', 'Milliseconds; small enough to run on a device', 'Much slower; billions of parameters, usually on a server'],
  ['Consistency', 'Deterministic, calibrated scores', 'Varies with the prompt and sampling'],
  ['Data with little text about it online (scans, satellite bands, sensors)', 'Can pre-train on your raw data alone', 'Knows it mostly second-hand, through text'],
  ['Fine visual detail', 'Kept in the embeddings', 'Squeezed through words'],
  ['Open-ended questions and world knowledge', 'No', 'Its strength'],
];

const TIMELINE: { when: string; what: string; text: string }[] = [
  { when: '2022', what: 'The position paper', text: 'LeCun lays out the plan in "A Path Towards Autonomous Machine Intelligence", with JEPA as its core building block.' },
  { when: '2023', what: 'I-JEPA', text: 'The image version, the one built here: learns image features by predicting embeddings of hidden blocks.' },
  { when: '2024', what: 'V-JEPA', text: 'The same idea on video: hide regions across space and time, predict their embeddings.' },
  {
    when: '2025',
    what: 'V-JEPA 2 and V-JEPA 2-AC',
    text: 'Pre-trained on over a million hours of internet video. The action-conditioned version (AC) adds a predictor post-trained on under 62 hours of robot video, and plans pick-and-place moves on robot arms in labs it had never seen, given only a picture of the goal.',
  },
  { when: '2025', what: 'LLM-JEPA', text: 'Brings a JEPA-style embedding-prediction objective into language model training, alongside the usual next-token loss.' },
  { when: '2025–26', what: 'AMI Labs', text: 'LeCun left Meta at the end of 2025 and founded AMI Labs in Paris in January 2026 to build world models full time.' },
];

export const JepaNextView: React.FC<{ onNavigate: (tab: string) => void }> = ({ onNavigate }) => (
  <div style={{ maxWidth: 960, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 20 }}>
    <div className="glass-panel" style={{ padding: '28px 32px' }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: 8 }}>What's next: from a JEPA to a world model</h2>
      <p style={{ color: 'var(--text-muted)', lineHeight: 1.65 }}>
        What you trained here learns to describe images. The reason JEPAs get so much attention is a bigger bet: that the same
        recipe, applied to video and to the effects of actions, can give machines a <i>world model</i>, an internal simulator of
        how things behave, which Yann LeCun argues is the missing piece for AI that can plan and reason about the physical world.
      </p>
    </div>

    <Section icon={<Box size={18} color="var(--accent-emerald)" />} title="What you keep after training">
      <p>
        The three networks aren't all part of the product. For I-JEPA, the goal is a good <B>encoder</B>: a network that turns an
        image into embeddings that make later tasks easy. After pre-training you keep <B>one encoder</B> (usually the slow EMA
        copy, the target encoder; the two end up nearly identical anyway), freeze it or fine-tune it, and put a small task-specific
        head on top, such as a classifier.
      </p>
      <p>
        The <B>predictor is thrown away</B>. Like the second encoder, it's scaffolding: its job was to turn "guess the hidden
        patch" into a training signal that shapes the encoder. The probes and nearest neighbours in this site work exactly like that:
        they read only the target encoder's embeddings and never touch the predictor.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.825rem', minWidth: 520 }}>
          <thead>
            <tr style={{ color: 'var(--text-dim)', textAlign: 'left' }}>
              <th style={{ padding: '6px 10px 6px 0', fontWeight: 600 }}>Goal</th>
              <th style={{ padding: '6px 10px', fontWeight: 600 }}>Kept after training</th>
              <th style={{ padding: '6px 10px', fontWeight: 600 }}>Discarded</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderTop: '1px solid var(--border-color)' }}>
              <td style={{ padding: '8px 10px 8px 0', color: 'var(--text-main)', fontWeight: 600 }}>Learn image features (I-JEPA, this site)</td>
              <td style={{ padding: '8px 10px' }}>One encoder</td>
              <td style={{ padding: '8px 10px' }}>The predictor and the other encoder</td>
            </tr>
            <tr style={{ borderTop: '1px solid var(--border-color)' }}>
              <td style={{ padding: '8px 10px 8px 0', color: 'var(--text-main)', fontWeight: 600 }}>World model for planning (V-JEPA 2-AC)</td>
              <td style={{ padding: '8px 10px' }}>The encoder <i>and</i> a predictor that also takes actions</td>
              <td style={{ padding: '8px 10px' }}>Training-only parts</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        That second row is where the predictor stops being scaffolding and becomes the point. It has to "speak the encoder's
        language" (read embeddings, output embeddings in the same space), and that is exactly what lets it serve as a simulator.
      </p>
    </Section>

    <Section icon={<Wrench size={18} color="var(--accent-cyan)" />} title="What's an encoder good for?">
      <p>
        On its own, the kept encoder can't name, draw or decide anything. It turns an image into embeddings: one per patch (which
        keep where things are) and usually one for the whole image (their average). Everything useful comes from what you put on
        top. Note that "vs an LLM" below really means a multimodal LLM, a chatbot you can show images to, and those contain an
        image encoder like this one inside.
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 10 }}>
        {USES.map(u => (
          <div key={u.title} style={{ padding: '12px 14px', borderRadius: 8, background: 'var(--surface-inset)', border: '1px solid var(--border-color)' }}>
            <p style={{ fontWeight: 700, color: 'var(--text-main)', fontSize: '0.85rem', marginBottom: 4 }}>{u.title}</p>
            <p style={{ fontSize: '0.8rem', lineHeight: 1.55, marginBottom: 6 }}>{u.example}</p>
            <p style={{ fontSize: '0.775rem', lineHeight: 1.5, color: 'var(--text-dim)' }}>
              <b style={{ color: 'var(--text-muted)' }}>Why not just ask an LLM?</b> {u.whyNotLlm}
            </p>
          </div>
        ))}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.825rem', minWidth: 560 }}>
          <thead>
            <tr style={{ color: 'var(--text-dim)', textAlign: 'left' }}>
              <th style={{ padding: '6px 10px 6px 0', fontWeight: 600 }} />
              <th style={{ padding: '6px 10px', fontWeight: 600 }}>Encoder + small head</th>
              <th style={{ padding: '6px 10px', fontWeight: 600 }}>Multimodal LLM</th>
            </tr>
          </thead>
          <tbody>
            {VERSUS.map(([what, enc, llm]) => (
              <tr key={what} style={{ borderTop: '1px solid var(--border-color)' }}>
                <td style={{ padding: '8px 10px 8px 0', color: 'var(--text-main)', fontWeight: 600 }}>{what}</td>
                <td style={{ padding: '8px 10px' }}>{enc}</td>
                <td style={{ padding: '8px 10px' }}>{llm}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p>
        <B>When the LLM wins:</B> you have no labels and need an answer today; the task is open-ended ("what's wrong in this
        photo?"); it needs knowledge of the world ("is this mushroom edible?"); or the volume is small enough that speed and cost
        don't matter. A common path is to prototype with a multimodal LLM, then switch to an encoder and a small head once the
        task is clear and it has to be fast, cheap and repeatable. Often the answer is both: the encoder as eyes, the language
        model for reasoning.
      </p>
      <p>
        <B>What it can't promise:</B> it learns whatever made the hidden patches predictable, which may not be what you care
        about. The one trained here picks up where the shape is, but not which shape it is.
      </p>
    </Section>

    <Section icon={<Brain size={18} color="var(--accent-purple)" />} title="Why LeCun thinks AI needs world models">
      <p>
        <B>Babies learn physics by watching.</B> In their first months, long before language, infants pick up that objects
        don't vanish when hidden and that unsupported things fall. They learn it mostly by observing, with very few
        "labels". LeCun's back-of-the-envelope estimate is that a four-year-old has taken in about as much data through vision as
        the largest language models have read as text. Text, in his view, is a thin, already-abstracted slice of experience.
      </p>
      <p>
        <B>Acting well needs predicting consequences.</B> A language model predicts the next word. To plan, a system needs to
        predict what happens to the world if it does something, try alternatives in its head, and pick the one that leads where it
        wants to go. That internal "what if" engine is the world model. Psychologists' System 1 and System 2 map onto his Mode 1
        (react directly) and Mode 2 (plan by searching with the world model).
      </p>
      <p>
        <B>Predict in embedding space, not pixels.</B> The future is full of detail no one could predict: the exact shimmer of
        leaves, the precise frame where a ball lands. A model that must generate every pixel spends its capacity on that detail,
        and when it's unsure it averages possibilities into blur. A JEPA lets the encoder drop what can't be predicted and keep
        what can. This is the same reason the JEPA here never reconstructs pixels.
      </p>
      <p>
        <B>His prescriptions.</B> In the 2022 paper and many talks since, he argues for four shifts:
      </p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 10 }}>
        {SHIFTS.map(([from, to, why]) => (
          <div key={from} style={{ padding: '10px 12px', borderRadius: 8, background: 'var(--surface-inset)', border: '1px solid var(--border-color)' }}>
            <p style={{ fontSize: '0.8rem', marginBottom: 4 }}>
              <span style={{ textDecoration: 'line-through', color: 'var(--text-dim)' }}>{from}</span>
              <span style={{ color: 'var(--text-dim)' }}> → </span>
              <b style={{ color: 'var(--text-main)' }}>{to}</b>
            </p>
            <p style={{ fontSize: '0.775rem', lineHeight: 1.5 }}>{why}</p>
          </div>
        ))}
      </div>
      <p>
        In the full design, a JEPA world model sits alongside other modules: <B>perception</B> (the encoder), a <B>cost</B> that
        scores how good a predicted state is, an <B>actor</B> that proposes actions, <B>short-term memory</B>, and a{' '}
        <B>configurator</B> that sets the others up for the task at hand. He also proposes stacking JEPAs into a hierarchy
        (H-JEPA): low levels predict short, detailed futures; higher levels predict longer, more abstract ones.
      </p>
      <p>
        <B>The other side.</B> This is a research bet, not settled science. His argument that long autoregressive outputs
        inevitably drift into errors is disputed, and language models have kept getting better at reasoning. Other labs pursue
        world models the generative way, producing video frame by frame (Google DeepMind's Genie models are an example). Whether
        world models must avoid generation is one of the field's open arguments.
      </p>
    </Section>

    <Section icon={<Repeat size={18} color="var(--primary)" />} title="How a JEPA becomes a world model">
      <p>
        Swap "hidden patch" for "next moment", and give the predictor one more input: the <B>action</B>. Now it answers "if I do
        this, what will the world look like next?", in embedding space.
      </p>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', padding: '6px 0' }}>
        <Pill text="Camera now" color="var(--text-dim)" />
        <Arr />
        <Pill text="Encoder" sub="state now" color="var(--primary)" />
        <span style={{ color: 'var(--text-dim)', fontWeight: 700 }}>+</span>
        <Pill text="Action" sub="e.g. move the arm" color="var(--accent-emerald)" />
        <Arr />
        <Pill text="Predictor" sub="predicted next state" color="var(--accent-purple)" />
        <Arr />
        <Pill text="Cost" sub="distance to the goal's embedding" color="var(--accent-rose)" />
      </div>
      <p>
        <B>Planning</B> then works like this (model-predictive control): encode the current view and a picture of the goal;
        imagine many candidate action sequences by running the predictor forward; score each by how close its final predicted
        state lands to the goal; carry out just the first action of the best one; look again, and repeat. V-JEPA 2-AC plans
        pick-and-place moves on real robot arms this way.
      </p>
      <p>
        Here, the images are still pictures and there are no actions, so this site's JEPA can't be a world model. A natural next
        experiment would be shapes that move when "pushed", with the push as the action. A predictor trained on that could plan
        how to get a shape from one place to another.
      </p>
    </Section>

    <Section icon={<Compass size={18} color="var(--accent-amber)" />} title="Where it stands">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {TIMELINE.map(t => (
          <div key={t.what} style={{ display: 'flex', gap: 14, alignItems: 'baseline' }}>
            <span className="font-mono" style={{ width: 64, flexShrink: 0, color: 'var(--accent-amber)', fontSize: '0.8rem' }}>{t.when}</span>
            <p>
              <B>{t.what}.</B> {t.text}
            </p>
          </div>
        ))}
      </div>
      <p>
        <B>Still open:</B> planning over long horizons (current demos are short, simple manipulations); building the hierarchy
        of H-JEPA in practice; learning the cost module rather than hand-specifying goals; and collapse, the problem you watched
        in the Train tab, which every non-generative method has to keep solving at scale.
      </p>
      <GoDeeper
        links={[
          { label: 'LeCun (2022): A Path Towards Autonomous Machine Intelligence', url: 'https://openreview.net/forum?id=BZ5a1r-kVsf' },
          { label: 'I-JEPA', url: 'https://arxiv.org/abs/2301.08243' },
          { label: 'V-JEPA', url: 'https://arxiv.org/abs/2404.08471' },
          { label: 'V-JEPA 2', url: 'https://arxiv.org/abs/2506.09985' },
          { label: 'LLM-JEPA', url: 'https://arxiv.org/abs/2509.14252' },
          { label: 'VICReg', url: 'https://arxiv.org/abs/2105.04906' },
        ]}
      />
    </Section>

    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
      <button className="btn-secondary" onClick={() => onNavigate('training')}>Back to training</button>
      <button className="btn-secondary" onClick={() => onNavigate('start')}>Start here</button>
    </div>
  </div>
);
