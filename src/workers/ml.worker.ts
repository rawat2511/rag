/// Runs both neural models off the main thread so the UI keeps animating:
///  - a bi-encoder (MiniLM) that turns text into 384-dim vectors
///  - a cross-encoder (MS MARCO MiniLM) that scores (query, passage) pairs for re-ranking
import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  env,
  pipeline,
} from '@huggingface/transformers';
import { EMBED_MODEL, RERANK_MODEL } from '../config';
import type { Device } from '../lib/types';

env.allowLocalModels = false;

/* eslint-disable @typescript-eslint/no-explicit-any */
const ctx = self as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent) => void) | null;
};

const extractors = new Map<Device, any>();
let rerankerPromise: Promise<{ tokenizer: any; model: any }> | null = null;
let embedJob = 0;

const downloadProgress = (model: string) => (p: any) => {
  if (p.status === 'progress' && typeof p.file === 'string' && p.file.endsWith('.onnx') && p.total) {
    ctx.postMessage({ type: 'download', model, loaded: p.loaded, total: p.total });
  }
};

async function getExtractor(preferred: Device): Promise<{ extractor: any; device: Device }> {
  const cached = extractors.get(preferred);
  if (cached) return { extractor: cached, device: preferred };
  if (preferred === 'webgpu') {
    try {
      const extractor = await pipeline('feature-extraction', EMBED_MODEL, {
        device: 'webgpu',
        dtype: 'fp32',
        progress_callback: downloadProgress('embedder'),
      });
      extractors.set('webgpu', extractor);
      return { extractor, device: 'webgpu' };
    } catch (err) {
      console.warn('WebGPU unavailable, falling back to WASM', err);
    }
  }
  let extractor = extractors.get('wasm');
  if (!extractor) {
    extractor = await pipeline('feature-extraction', EMBED_MODEL, {
      device: 'wasm',
      dtype: 'q8',
      progress_callback: downloadProgress('embedder'),
    });
    extractors.set('wasm', extractor);
  }
  return { extractor, device: 'wasm' };
}

function getReranker() {
  rerankerPromise ??= (async () => {
    const tokenizer = await AutoTokenizer.from_pretrained(RERANK_MODEL);
    const model = await AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL, {
      device: 'wasm',
      dtype: 'q8',
      progress_callback: downloadProgress('reranker'),
    });
    return { tokenizer, model };
  })();
  rerankerPromise.catch(() => (rerankerPromise = null));
  return rerankerPromise;
}

ctx.onmessage = async (e: MessageEvent) => {
  const m = e.data;
  const rid: number = m.rid;
  try {
    switch (m.type) {
      case 'cancel':
        embedJob++;
        break;

      case 'embed': {
        const job = ++embedJob;
        const texts: string[] = m.texts;
        const { extractor, device } = await getExtractor(m.device);
        ctx.postMessage({ type: 'event', rid, event: 'start', device });
        const batchSize = device === 'webgpu' ? 16 : 8;
        for (let i = 0; i < texts.length; i += batchSize) {
          if (job !== embedJob) {
            ctx.postMessage({ type: 'error', rid, message: 'cancelled' });
            return;
          }
          const out = await extractor(texts.slice(i, i + batchSize), { pooling: 'mean', normalize: true });
          const data = new Float32Array(out.data as Float32Array);
          ctx.postMessage({ type: 'event', rid, event: 'batch', start: i, data }, [data.buffer]);
        }
        ctx.postMessage({ type: 'done', rid, result: { device } });
        break;
      }

      case 'embedQuery': {
        const { extractor } = await getExtractor(m.device);
        const out = await extractor([m.text], { pooling: 'mean', normalize: true });
        const data = new Float32Array(out.data as Float32Array);
        ctx.postMessage({ type: 'done', rid, result: data }, [data.buffer]);
        break;
      }

      case 'loadReranker':
        await getReranker();
        ctx.postMessage({ type: 'done', rid, result: null });
        break;

      case 'rerank': {
        const { tokenizer, model } = await getReranker();
        const docs: string[] = m.docs;
        const logits: number[] = [];
        const batchSize = 4;
        for (let i = 0; i < docs.length; i += batchSize) {
          const batch = docs.slice(i, i + batchSize);
          const inputs = tokenizer(new Array(batch.length).fill(m.query), {
            text_pair: batch,
            padding: true,
            truncation: true,
          });
          const output = await model(inputs);
          const scores = Array.from(output.logits.data as Float32Array);
          logits.push(...scores);
          ctx.postMessage({ type: 'event', rid, event: 'batch', start: i, scores });
        }
        ctx.postMessage({ type: 'done', rid, result: logits });
        break;
      }
    }
  } catch (err) {
    ctx.postMessage({ type: 'error', rid, message: err instanceof Error ? err.message : String(err) });
  }
};
