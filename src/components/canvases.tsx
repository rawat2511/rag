import { useEffect, useRef } from 'react';

function setupCanvas(canvas: HTMLCanvasElement, w: number, h: number) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const g = canvas.getContext('2d')!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  return g;
}

/* ───────────────────────── Vector strip ───────────────────────── */

/** A 384-dim embedding drawn as a heat strip: blue = negative, orange = positive. */
export function VectorStrip({ vec, cols = 64, cell = 6 }: { vec: Float32Array | null; cols?: number; cell?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const dim = vec?.length ?? 384;
  const rows = Math.ceil(dim / cols);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const g = setupCanvas(canvas, cols * cell, rows * cell);
    for (let i = 0; i < dim; i++) {
      const v = vec ? Math.max(-1, Math.min(1, vec[i] / 0.11)) : 0;
      const a = Math.abs(v);
      g.fillStyle = vec
        ? v < 0
          ? `rgba(90,169,255,${0.12 + a * 0.88})`
          : `rgba(255,150,90,${0.12 + a * 0.88})`
        : '#222a42';
      g.fillRect((i % cols) * cell, Math.floor(i / cols) * cell, cell - 1, cell - 1);
    }
  }, [vec, cols, cell, rows, dim]);
  return <canvas ref={ref} className="vector-strip" />;
}
