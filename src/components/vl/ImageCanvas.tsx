import React, { useEffect, useRef } from 'react';
import { ShapeImage } from '../../engine/vl/shapes';
import { THEME, rgb } from '../../styles/theme';

interface ImageCanvasProps {
  image: ShapeImage;
  size: number;              // displayed width = height, in CSS pixels
  patchSize?: number;        // draw the patch grid (pixels per patch side)
  overlay?: number[] | null; // one value per patch in 0–1, drawn as a tint over the picture
  overlayColor?: string;     // '#rrggbb'
  selectedPatch?: number | null;
  onPatchClick?: (patch: number) => void;
  title?: string;
}

/** One picture, drawn pixel by pixel (no smoothing), optionally with its patch grid and a per-patch tint */
export const ImageCanvas: React.FC<ImageCanvasProps> = ({
  image, size, patchSize, overlay, overlayColor = THEME.amber, selectedPatch, onPatchClick, title,
}) => {
  const ref = useRef<HTMLCanvasElement>(null);
  const dpr = typeof window !== 'undefined' ? Math.min(2, window.devicePixelRatio || 1) : 1;
  const px = Math.round(size * dpr);

  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    const n = image.size;
    const cell = px / n;
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const i = (y * n + x) * 3;
        const [r, g, b] = [image.pixels[i], image.pixels[i + 1], image.pixels[i + 2]].map(v => Math.round(v * 255));
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(Math.floor(x * cell), Math.floor(y * cell), Math.ceil(cell), Math.ceil(cell));
      }
    }
    if (!patchSize) return;
    const grid = n / patchSize;
    const pCell = px / grid;
    if (overlay) {
      const [r, g, b] = rgb(overlayColor);
      overlay.forEach((v, p) => {
        ctx.fillStyle = `rgba(${r},${g},${b},${Math.max(0, Math.min(1, v)) * 0.6})`;
        ctx.fillRect((p % grid) * pCell, Math.floor(p / grid) * pCell, pCell, pCell);
      });
    }
    ctx.strokeStyle = 'rgba(255,255,255,0.22)';
    ctx.lineWidth = Math.max(1, dpr);
    for (let k = 1; k < grid; k++) {
      ctx.beginPath();
      ctx.moveTo(k * pCell, 0);
      ctx.lineTo(k * pCell, px);
      ctx.moveTo(0, k * pCell);
      ctx.lineTo(px, k * pCell);
      ctx.stroke();
    }
    if (selectedPatch !== null && selectedPatch !== undefined) {
      ctx.strokeStyle = THEME.cyan;
      ctx.lineWidth = 2.5 * dpr;
      ctx.strokeRect((selectedPatch % grid) * pCell + dpr, Math.floor(selectedPatch / grid) * pCell + dpr, pCell - 2 * dpr, pCell - 2 * dpr);
    }
  }, [image, px, patchSize, overlay, overlayColor, selectedPatch, dpr]);

  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onPatchClick || !patchSize) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const grid = image.size / patchSize;
    const col = Math.min(grid - 1, Math.floor(((e.clientX - rect.left) / rect.width) * grid));
    const row = Math.min(grid - 1, Math.floor(((e.clientY - rect.top) / rect.height) * grid));
    onPatchClick(row * grid + col);
  };

  return (
    <canvas
      ref={ref}
      width={px}
      height={px}
      title={title}
      onClick={onClick}
      style={{ width: size, height: size, borderRadius: 6, imageRendering: 'pixelated', cursor: onPatchClick ? 'pointer' : undefined, display: 'block' }}
    />
  );
};
