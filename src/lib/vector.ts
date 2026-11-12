import type { FusedHit, Hit } from './types';

/** Brute-force cosine search. Vectors are L2-normalised, so cosine = dot product. */
export function topKDense(emb: Float32Array, n: number, dim: number, q: Float32Array, k: number): Hit[] {
  const scores = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const off = i * dim;
    for (let d = 0; d < dim; d++) s += emb[off + d] * q[d];
    scores[i] = s;
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
  return order.slice(0, k).map((id, i) => ({ id, score: scores[id], rank: i + 1 }));
}

/** Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank). */
export function rrf(dense: Hit[], sparse: Hit[], k = 60): FusedHit[] {
  const merged = new Map<number, FusedHit>();
  const get = (id: number) => {
    let hit = merged.get(id);
    if (!hit) merged.set(id, (hit = { id, score: 0, rank: 0 }));
    return hit;
  };
  for (const h of dense) {
    const hit = get(h.id);
    hit.denseRank = h.rank;
    hit.score += 1 / (k + h.rank);
  }
  for (const h of sparse) {
    const hit = get(h.id);
    hit.sparseRank = h.rank;
    hit.score += 1 / (k + h.rank);
  }
  return [...merged.values()].sort((a, b) => b.score - a.score).map((h, i) => ({ ...h, rank: i + 1 }));
}

export interface Projection {
  mean: Float32Array;
  axes: Float32Array[];
  /** x,y,z triples for every chunk. */
  coords: Float32Array;
}

const COMPONENTS = 3;

/** Project embeddings to 3D with PCA (top principal components via power iteration). */
export function pca3d(emb: Float32Array, n: number, dim: number): Projection {
  const mean = new Float32Array(dim);
  for (let i = 0; i < n; i++) for (let d = 0; d < dim; d++) mean[d] += emb[i * dim + d];
  for (let d = 0; d < dim; d++) mean[d] /= Math.max(1, n);

  const axes: Float32Array[] = [];
  for (let c = 0; c < COMPONENTS; c++) {
    let v = new Float32Array(dim);
    for (let d = 0; d < dim; d++) v[d] = Math.sin((d + 1) * (c + 1) * 12.9898) + 0.01;
    for (let iter = 0; iter < 40; iter++) {
      const w = new Float32Array(dim);
      let meanDot = 0;
      for (let d = 0; d < dim; d++) meanDot += mean[d] * v[d];
      let sumS = 0;
      for (let i = 0; i < n; i++) {
        const off = i * dim;
        let s = -meanDot;
        for (let d = 0; d < dim; d++) s += emb[off + d] * v[d];
        sumS += s;
        for (let d = 0; d < dim; d++) w[d] += s * emb[off + d];
      }
      for (let d = 0; d < dim; d++) w[d] -= sumS * mean[d];
      for (const prev of axes) {
        let dot = 0;
        for (let d = 0; d < dim; d++) dot += w[d] * prev[d];
        for (let d = 0; d < dim; d++) w[d] -= dot * prev[d];
      }
      let norm = 0;
      for (let d = 0; d < dim; d++) norm += w[d] * w[d];
      norm = Math.sqrt(norm);
      if (norm < 1e-9) break;
      for (let d = 0; d < dim; d++) w[d] /= norm;
      v = w;
    }
    let vn = 0;
    for (let d = 0; d < dim; d++) vn += v[d] * v[d];
    vn = Math.sqrt(vn) || 1;
    for (let d = 0; d < dim; d++) v[d] /= vn;
    axes.push(v);
  }

  const proj: Projection = { mean, axes, coords: new Float32Array(n * COMPONENTS) };
  for (let i = 0; i < n; i++) proj.coords.set(project(proj, emb.subarray(i * dim, (i + 1) * dim)), i * COMPONENTS);
  return proj;
}

export function project(p: Projection, v: Float32Array): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  for (let d = 0; d < v.length; d++) {
    const c = v[d] - p.mean[d];
    for (let a = 0; a < COMPONENTS; a++) out[a] += c * p.axes[a][d];
  }
  return out;
}
