import { useEffect, useMemo, useRef, useState } from 'react';
import type { Chunk } from '../lib/types';
import type { Projection } from '../lib/vector';

export const COLORS = {
  dense: '#5aa9ff',
  sparse: '#f5b54a',
  fuse: '#b38cff',
  rerank: '#45d6a0',
  final: '#ffd76a',
};

/** Chunks are coloured by their position in the document (start = teal, end = pink). */
export function chunkColor(i: number, n: number, alpha = 1): string {
  const hue = 175 + 165 * (n <= 1 ? 0 : i / (n - 1));
  return `hsla(${hue}, 72%, 62%, ${alpha})`;
}

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

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

/* ───────────────────────── Chunk grid ───────────────────────── */

interface ChunkGridProps {
  n: number;
  revealed: number;
  embedded: number;
  selected: number | null;
  hover: number | null;
  finalIds: number[];
  onSelect: (id: number) => void;
  onHover: (id: number | null) => void;
}

/** One tile per chunk: grey once chunked, coloured once its embedding exists. */
export function ChunkGrid({ n, revealed, embedded, selected, hover, finalIds, onSelect, onHover }: ChunkGridProps) {
  const [wrapRef, width] = useWidth<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const layout = useMemo(() => {
    if (!width || !n) return { tile: 0, cols: 1, rows: 0 };
    let tile = 26;
    for (; tile > 5; tile--) {
      const cols = Math.max(1, Math.floor(width / tile));
      if (Math.ceil(n / cols) * tile <= 230) break;
    }
    const cols = Math.max(1, Math.floor(width / tile));
    return { tile, cols, rows: Math.ceil(n / cols) };
  }, [width, n]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !layout.tile) return;
    const { tile, cols, rows } = layout;
    const g = setupCanvas(canvas, width, rows * tile);
    const gap = tile > 10 ? 2 : 1;
    const finals = new Set(finalIds);
    for (let i = 0; i < Math.min(revealed, n); i++) {
      const x = (i % cols) * tile;
      const y = Math.floor(i / cols) * tile;
      g.fillStyle = i < embedded ? chunkColor(i, n) : '#2a3350';
      g.fillRect(x, y, tile - gap, tile - gap);
    }
    const outline = (i: number, color: string, lw: number) => {
      const x = (i % cols) * tile;
      const y = Math.floor(i / cols) * tile;
      g.strokeStyle = color;
      g.lineWidth = lw;
      g.strokeRect(x - 0.5, y - 0.5, tile - gap + 1, tile - gap + 1);
    };
    for (const id of finals) outline(id, COLORS.final, 2.5);
    if (hover !== null && hover < revealed) outline(hover, '#ffffff', 1.5);
    if (selected !== null && selected < revealed) outline(selected, '#ffffff', 2.5);
  }, [layout, width, n, revealed, embedded, selected, hover, finalIds]);

  const idAt = (e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const col = Math.floor((e.clientX - rect.left) / layout.tile);
    const row = Math.floor((e.clientY - rect.top) / layout.tile);
    const id = row * layout.cols + col;
    return col < layout.cols && id >= 0 && id < Math.min(revealed, n) ? id : null;
  };

  return (
    <div ref={wrapRef} className="chunk-grid">
      <canvas
        ref={canvasRef}
        onMouseMove={(e) => onHover(idAt(e))}
        onMouseLeave={() => onHover(null)}
        onClick={(e) => {
          const id = idAt(e);
          if (id !== null) onSelect(id);
        }}
      />
    </div>
  );
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

/* ───────────────────────── Embedding map ───────────────────────── */

interface MapProps {
  proj: Projection;
  chunks: Chunk[];
  selected: number | null;
  hover: number | null;
  queryXY?: [number, number];
  denseIds: number[];
  sparseIds: number[];
  finalIds: number[];
  onSelect: (id: number) => void;
  onHover: (id: number | null) => void;
}

/** 2D PCA projection of every chunk embedding, with the query and its hits overlaid. */
export function EmbeddingMap(props: MapProps) {
  const { proj, chunks, selected, hover, queryXY, denseIds, sparseIds, finalIds, onSelect, onHover } = props;
  const [wrapRef, width] = useWidth<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; id: number } | null>(null);
  const height = 400;
  const n = chunks.length;

  const toScreen = useMemo(() => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    const grow = (x: number, y: number) => {
      minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y);
    };
    for (let i = 0; i < n; i++) grow(proj.coords[i * 2], proj.coords[i * 2 + 1]);
    if (queryXY) grow(queryXY[0], queryXY[1]);
    const pad = 28;
    const sx = (width - pad * 2) / (maxX - minX || 1);
    const sy = (height - pad * 2) / (maxY - minY || 1);
    return (x: number, y: number): [number, number] => [pad + (x - minX) * sx, height - pad - (y - minY) * sy];
  }, [proj, n, queryXY, width]);

  const pos = (id: number) => toScreen(proj.coords[id * 2], proj.coords[id * 2 + 1]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !width) return;
    const g = setupCanvas(canvas, width, height);
    const hasQuery = !!queryXY;
    const r = n > 1200 ? 2.2 : n > 300 ? 3 : 4.5;

    for (let i = 0; i < n; i++) {
      const [x, y] = pos(i);
      g.fillStyle = chunkColor(i, n, hasQuery ? 0.3 : 0.85);
      g.beginPath();
      g.arc(x, y, r, 0, Math.PI * 2);
      g.fill();
    }

    const ring = (id: number, color: string, radius: number) => {
      const [x, y] = pos(id);
      g.strokeStyle = color;
      g.lineWidth = 2;
      g.beginPath();
      g.arc(x, y, radius, 0, Math.PI * 2);
      g.stroke();
    };
    for (const id of sparseIds) ring(id, COLORS.sparse, r + 5.5);
    for (const id of denseIds) ring(id, COLORS.dense, r + 3);

    if (queryXY) {
      const [qx, qy] = toScreen(queryXY[0], queryXY[1]);
      for (const id of finalIds) {
        const [x, y] = pos(id);
        g.strokeStyle = 'rgba(255,215,106,0.55)';
        g.lineWidth = 1.5;
        g.beginPath();
        g.moveTo(qx, qy);
        g.lineTo(x, y);
        g.stroke();
        g.fillStyle = COLORS.final;
        g.beginPath();
        g.arc(x, y, r + 2, 0, Math.PI * 2);
        g.fill();
      }
      // query marker: a white diamond
      g.fillStyle = '#ffffff';
      g.strokeStyle = '#0b0f1a';
      g.lineWidth = 2;
      g.beginPath();
      g.moveTo(qx, qy - 9); g.lineTo(qx + 9, qy); g.lineTo(qx, qy + 9); g.lineTo(qx - 9, qy);
      g.closePath();
      g.fill();
      g.stroke();
      g.font = '600 12px system-ui, sans-serif';
      g.fillStyle = '#ffffff';
      g.fillText('query', qx + 13, qy + 4);
    }

    for (const [id, lw] of [[hover, 1.5], [selected, 2.5]] as const) {
      if (id === null || id >= n) continue;
      const [x, y] = pos(id);
      g.fillStyle = chunkColor(id, n);
      g.strokeStyle = '#ffffff';
      g.lineWidth = lw;
      g.beginPath();
      g.arc(x, y, r + 3, 0, Math.PI * 2);
      g.fill();
      g.stroke();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toScreen, width, n, selected, hover, queryXY, denseIds, sparseIds, finalIds]);

  const nearest = (e: React.MouseEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best = -1;
    let bestD = 14 * 14;
    for (let i = 0; i < n; i++) {
      const [x, y] = pos(i);
      const d = (x - mx) ** 2 + (y - my) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    return best >= 0 ? { id: best, x: mx, y: my } : null;
  };

  return (
    <div ref={wrapRef} className="embed-map">
      <canvas
        ref={canvasRef}
        onMouseMove={(e) => {
          const hit = nearest(e);
          setTip(hit);
          onHover(hit ? hit.id : null);
        }}
        onMouseLeave={() => { setTip(null); onHover(null); }}
        onClick={(e) => {
          const hit = nearest(e);
          if (hit) onSelect(hit.id);
        }}
      />
      {tip && (
        <div
          className="map-tip"
          style={{ left: Math.min(tip.x + 14, Math.max(0, width - 290)), top: Math.min(tip.y + 14, height - 110) }}
        >
          <b>Chunk {tip.id + 1}</b> · p.{chunks[tip.id].pageStart}
          <p>{chunks[tip.id].text.slice(0, 150)}…</p>
        </div>
      )}
    </div>
  );
}
