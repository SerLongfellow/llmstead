import React from 'react';
import { ArrowLeft, Layers, MessageSquareText, Scale, Search } from 'lucide-react';
import { GoDeeper } from '../GoDeeper';

interface VisionNextViewProps {
  onNavigate: (tab: string) => void;
}

const STAGES: { icon: React.ReactNode; title: string; body: React.ReactNode; links: { label: string; url: string }[] }[] = [
  {
    icon: <Scale size={20} color="var(--accent-amber)" />,
    title: 'Scale: the real CLIP',
    body: (
      <>
        <p>
          OpenAI's CLIP (2021) used the same recipe as this page: an image tower and a text tower, trained so that matching pairs
          score highest in a batch. The differences are size. It learned from 400 million image + caption pairs collected from
          the web, compared 32,768 pairs in every batch, and used Vision Transformers and a GPT-style text transformer with
          hundreds of millions of parameters.
        </p>
        <p>
          The payoff was zero-shot classification: to sort photos into categories it was never trained on, write each category
          as a caption ("a photo of a dog") and pick the closest one, exactly like the test on the Train tab.
        </p>
      </>
    ),
    links: [
      { label: 'CLIP paper (Radford et al., 2021)', url: 'https://arxiv.org/abs/2103.00020' },
      { label: 'OpenAI: CLIP', url: 'https://openai.com/index/clip/' },
    ],
  },
  {
    icon: <Layers size={20} color="var(--accent-purple)" />,
    title: 'A better loss: SigLIP',
    body: (
      <p>
        The softmax loss used here (and in CLIP) asks every row and column of the batch's grid to pick one winner, so every
        pair's loss depends on the whole batch. Google's SigLIP (2023) scores each cell on its own instead: a yes/no question,
        "is this caption this picture's?", with a sigmoid. That removes the batch-wide normalization, makes very large batches
        cheaper, and works better than the softmax loss when batches are small.
      </p>
    ),
    links: [{ label: 'SigLIP paper (Zhai et al., 2023)', url: 'https://arxiv.org/abs/2303.15343' }],
  },
  {
    icon: <Search size={20} color="var(--primary)" />,
    title: 'What CLIP-style models get used for',
    body: (
      <p>
        Searching photo libraries with words (the search box in Look inside), filtering and labelling huge image datasets,
        zero-shot classifiers, and steering image generators: early text-to-image models used CLIP's text tower to turn a prompt
        into the vector the image is generated from.
      </p>
    ),
    links: [],
  },
  {
    icon: <MessageSquareText size={20} color="var(--accent-emerald)" />,
    title: 'Your Describe model vs LLaVA',
    body: (
      <>
        <p>
          Your Describe model has the same three parts as LLaVA (2023): a CLIP image tower kept frozen, a <b>projector</b> that turns each
          patch's vector into an image token, and a GPT-style language model that reads those tokens in front of the question. The
          first LLaVA used a single linear layer as its projector; LLaVA-1.5 switched to a small two-layer MLP, as here.
        </p>
        <p>
          The scale is what differs. LLaVA's language model (Vicuna) has 7 to 13 billion parameters and arrives already trained on
          vast amounts of text; its first phase aligned the projector on about 600,000 image-caption pairs, and the second tuned it
          on about 158,000 conversations about images. Your language model starts from nothing, which is why its phase 0 exists and
          why skipping the phases costs it little: it has nothing to protect.
        </p>
      </>
    ),
    links: [
      { label: 'LLaVA paper (Liu et al., 2023)', url: 'https://arxiv.org/abs/2304.08485' },
      { label: 'LLaVA-1.5 (Liu et al., 2023)', url: 'https://arxiv.org/abs/2310.03744' },
    ],
  },
  {
    icon: <Layers size={20} color="var(--accent-cyan)" />,
    title: 'Other ways to join pictures and words',
    body: (
      <p>
        Putting image tokens in front of the text is the simplest design and the most common today, but not the only one.
        Flamingo (DeepMind, 2022) keeps the picture out of the sequence and lets the language model look at it through extra
        cross-attention layers. BLIP-2 (2023) squeezes the picture into a few dozen learned query tokens (a "Q-Former") before the
        language model sees it. Newer models also cut large pictures into tiles so fine detail survives, and train the vision
        encoder further instead of keeping it frozen.
      </p>
    ),
    links: [
      { label: 'Flamingo (Alayrac et al., 2022)', url: 'https://arxiv.org/abs/2204.14198' },
      { label: 'BLIP-2 (Li et al., 2023)', url: 'https://arxiv.org/abs/2301.12597' },
    ],
  },
];

export const VisionNextView: React.FC<VisionNextViewProps> = ({ onNavigate }) => (
  <div style={{ maxWidth: 960, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 20 }}>
    <div className="glass-panel" style={{ padding: '28px 32px' }}>
      <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: 8 }}>From matching to talking, at full size</h2>
      <p style={{ color: 'var(--text-muted)', lineHeight: 1.65 }}>
        You trained a model that knows which words fit which pictures, then gave it a voice. Here's how the same ideas work at full
        size.
      </p>
    </div>

    {STAGES.map(s => (
      <div key={s.title} className="glass-panel" style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {s.icon}
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>{s.title}</h3>
        </div>
        <div style={{ fontSize: '0.9rem', color: 'var(--text-muted)', lineHeight: 1.65, display: 'flex', flexDirection: 'column', gap: 10 }}>{s.body}</div>
        {s.links.length > 0 && <GoDeeper links={s.links} />}
      </div>
    ))}

    <div style={{ display: 'flex', justifyContent: 'center' }}>
      <button className="btn-secondary" onClick={() => onNavigate('pipeline')}>
        <ArrowLeft size={16} /> Back to Look inside
      </button>
    </div>
  </div>
);
