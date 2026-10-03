import React, { useEffect, useRef } from 'react';
import { JepaMask } from '../../engine/jepa/jepa';
import { ShapeImage } from '../../engine/jepa/shapes';
import { THEME, withAlpha } from '../../styles/theme';

interface ShapeCanvasProps {
  img: ShapeImage;
  scale: number;             // screen pixels per image pixel
  patchSize?: number;        // draws the patch grid when set
  mask?: JepaMask;           // outlines the hidden (target) patches
  hidden?: 'dim' | 'cover';  // how hidden patches look: dimmed, or covered up entirely
  heat?: (number | null)[];  // per patch, 0–1: an overlay such as attention weights (null = none)
  selectedPatch?: number | null;
  onPatchClick?: (patch: number) => void;
  title?: string;
}

/**
 * Draws one shape image pixel by pixel (no smoothing, so you see the real 16×16 pixels), with
 * optional patch grid, mask and per-patch overlay. Click a patch to select it.
 */
export const ShapeCanvas: React.FC<ShapeCanvasProps> = ({
  img, scale, patchSize, mask, hidden = 'dim', heat, selectedPatch, onPatchClick, title,
}) => {
  const ref = useRef<HTMLCanvasElement>(null);
  const px = img.size * scale;

  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, px, px);
    for (let y = 0; y < img.size; y++) {
      for (let x = 0; x < img.size; x++) {
        const i = (y * img.size + x) * 3;
        const [r, g, b] = [img.pixels[i], img.pixels[i + 1], img.pixels[i + 2]].map(v => Math.round(v * 255));
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x * scale, y * scale, scale, scale);
      }
    }
    if (!patchSize) return;
    const grid = img.size / patchSize;
    const cell = patchSize * scale;
    const at = (p: number) => [(p % grid) * cell, Math.floor(p / grid) * cell] as const;

    heat?.forEach((v, p) => {
      if (v === null) return;
      const [x, y] = at(p);
      ctx.fillStyle = withAlpha(THEME.cyan, 0.08 + 0.72 * Math.max(0, Math.min(1, v)));
      ctx.fillRect(x, y, cell, cell);
    });

    mask?.targets.forEach(p => {
      const [x, y] = at(p);
      ctx.fillStyle = hidden === 'cover' ? THEME.surface : 'rgba(0, 0, 0, 0.55)';
      ctx.fillRect(x, y, cell, cell);
      if (hidden === 'cover') {
        ctx.strokeStyle = withAlpha(THEME.amber, 0.35);
        ctx.lineWidth = 1;
        for (let d = -cell; d < cell; d += 6) {
          ctx.beginPath();
          ctx.moveTo(x + Math.max(0, d), y + Math.max(0, -d));
          ctx.lineTo(x + Math.min(cell, cell + d), y + Math.min(cell, cell - d));
          ctx.stroke();
        }
      }
    });

    // Grid lines, then the outlines on top
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;
    for (let k = 1; k < grid; k++) {
      ctx.beginPath();
      ctx.moveTo(k * cell + 0.5, 0);
      ctx.lineTo(k * cell + 0.5, px);
      ctx.moveTo(0, k * cell + 0.5);
      ctx.lineTo(px, k * cell + 0.5);
      ctx.stroke();
    }
    mask?.targets.forEach(p => {
      const [x, y] = at(p);
      ctx.strokeStyle = THEME.amber;
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
    });
    if (selectedPatch !== null && selectedPatch !== undefined) {
      const [x, y] = at(selectedPatch);
      ctx.strokeStyle = THEME.textMain;
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
    }
  }, [img, scale, px, patchSize, mask, hidden, heat, selectedPatch]);

  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onPatchClick || !patchSize) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const grid = img.size / patchSize;
    const gx = Math.floor(((e.clientX - rect.left) / rect.width) * grid);
    const gy = Math.floor(((e.clientY - rect.top) / rect.height) * grid);
    if (gx >= 0 && gx < grid && gy >= 0 && gy < grid) onPatchClick(gy * grid + gx);
  };

  return (
    <canvas
      ref={ref}
      width={px}
      height={px}
      title={title}
      onClick={onClick}
      style={{
        width: px,
        maxWidth: '100%',
        aspectRatio: '1',
        imageRendering: 'pixelated',
        borderRadius: 6,
        border: '1px solid var(--border-color)',
        cursor: onPatchClick ? 'pointer' : 'default',
        display: 'block',
      }}
    />
  );
};
