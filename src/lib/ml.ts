import type { Device } from './types';

/* eslint-disable @typescript-eslint/no-explicit-any */
interface Pending {
  resolve: (value: any) => void;
  reject: (err: Error) => void;
  onEvent?: (event: any) => void;
}

/** Promise-based client for the ML web worker. */
class MLClient {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  onDownload: ((model: string, loaded: number, total: number) => void) | null = null;

  constructor() {
    this.worker = new Worker(new URL('../workers/ml.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent) => {
      const m = e.data;
      if (m.type === 'download') {
        this.onDownload?.(m.model, m.loaded, m.total);
        return;
      }
      const p = this.pending.get(m.rid);
      if (!p) return;
      if (m.type === 'event') p.onEvent?.(m);
      else if (m.type === 'done') {
        this.pending.delete(m.rid);
        p.resolve(m.result);
      } else if (m.type === 'error') {
        this.pending.delete(m.rid);
        p.reject(new Error(m.message));
      }
    };
    this.worker.onerror = (e) => {
      const err = new Error(e.message || 'The ML worker crashed.');
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
    };
  }

  private call<T>(message: Record<string, unknown>, onEvent?: (event: any) => void): Promise<T> {
    const rid = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(rid, { resolve, reject, onEvent });
      this.worker.postMessage({ ...message, rid });
    });
  }

  /** Embed texts in batches. `onBatch` receives each batch's vectors as they are produced. */
  embed(
    texts: string[],
    device: Device,
    onStart: (device: Device) => void,
    onBatch: (start: number, data: Float32Array) => void,
  ): Promise<{ device: Device }> {
    return this.call({ type: 'embed', texts, device }, (ev) => {
      if (ev.event === 'start') onStart(ev.device);
      else onBatch(ev.start, ev.data);
    });
  }

  embedQuery(text: string, device: Device): Promise<Float32Array> {
    return this.call({ type: 'embedQuery', text, device });
  }

  loadReranker(): Promise<void> {
    return this.call({ type: 'loadReranker' });
  }

  /** Score (query, doc) pairs with the cross-encoder. Returns raw logits. */
  rerank(query: string, docs: string[], onBatch: (start: number, scores: number[]) => void): Promise<number[]> {
    return this.call({ type: 'rerank', query, docs }, (ev) => onBatch(ev.start, ev.scores));
  }

  cancelEmbedding(): void {
    this.worker.postMessage({ type: 'cancel', rid: 0 });
  }
}

export const ml = new MLClient();

export async function hasWebGPU(): Promise<boolean> {
  try {
    const gpu = (navigator as any).gpu;
    return !!gpu && !!(await gpu.requestAdapter());
  } catch {
    return false;
  }
}
