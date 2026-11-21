import * as THREE from 'three';
import {
  COL, PALETTE, Dots, Lines, QueryMarker, clamp01, clip, clusters, ease, gauss, kmeans, label, nearest, panel, ramp, rect, rng, seg, setPanel, fade, argmin,
} from './kit';

export interface Step { title: string; detail: string; dur: number }
export interface FrameCtx { camera: THREE.Camera; now: number }
export interface Built { group: THREE.Group; update(t: number, ctx: FrameCtx): string }
export interface Rating { speed: number; recall: number; memory: number; build: number }

export interface AlgoDef {
  id: string;
  name: string;
  full: string;
  family: string;
  tagline: string;
  steps: Step[];
  camera: [number, number, number];
  target?: [number, number, number];
  spin?: boolean;
  rating: Rating;
  complexity: string;
  params: string[];
  usedIn: string;
  bestFor: string;
  watchOut: string;
  build(T: number[]): Built;
}

const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const HALF: [number, number, number] = [4.4, 2.4, 3];
const tmp = new THREE.Color();
const tmp2 = new THREE.Color();
const pulse = (now: number, f = 3) => 0.5 + 0.5 * Math.sin(now * f);

/* ═════════════════════════ 1 · Flat ═════════════════════════ */

function buildFlat(T: number[]): Built {
  const g = new THREE.Group();
  const N = 220;
  const { pts } = clusters(N, 6, 11, HALF);
  const q = V(1.2, 0.5, 0.6);
  const top = new Set(nearest(pts, q, 6));
  const dmax = Math.max(...pts.map((p) => p.distanceTo(q)));
  const dots = new Dots(N);
  const lines = new Lines(N + 8);
  const qm = new QueryMarker();
  g.add(dots.mesh, lines.obj, qm.group);
  const qStart = q.clone().add(V(0, 3, 0));
  const qp = V();

  return {
    group: g,
    update(t, { camera, now }) {
      const appear = seg(t, T[0], T[1] - 0.2);
      const scan = seg(t, T[2], T[3]) * N;
      const pick = ease(seg(t, T[3], T[3] + 0.9));
      lines.reset();
      for (let i = 0; i < N; i++) {
        const a = ease(seg(appear, (i / N) * 0.6, (i / N) * 0.6 + 0.4));
        const close = 1 - pts[i].distanceTo(q) / dmax;
        let s = 0.075 * a;
        let gain = 1;
        tmp.copy(COL.idle);
        if (i < scan) {
          const fresh = clamp01(1 - (scan - i) / 14);
          const isTop = top.has(i);
          tmp.lerp(COL.dense, 0.3 + 0.45 * close);
          gain = 1 + fresh * 2;
          s *= 1 + fresh * 0.6;
          if (isTop) { tmp.lerp(COL.final, pick); gain += 1.6 * pick; s *= 1 + 0.9 * pick; }
          else tmp.lerp(COL.dim, pick * 0.75);
          tmp2.copy(COL.dense).lerp(COL.final, isTop ? pick : 0);
          const k = (0.06 + 0.45 * close * close + fresh * 0.6) * (1 - pick) + (isTop ? 1.3 * pick : 0);
          lines.add(q, pts[i], tmp2, k);
        }
        dots.set(i, pts[i], s, tmp, gain);
      }
      dots.flush();
      lines.flush();
      const qa = seg(t, T[1], T[1] + 0.8);
      qm.update(qp.lerpVectors(qStart, q, ease(qa)), ease(qa * 2), camera, now);
      return t < T[2] ? `Vectors stored: ${N} · index overhead: none` : `Distance computations: ${Math.min(N, Math.floor(scan))} / ${N}`;
    },
  };
}

/* ═════════════════════════ 2 · IVF ═════════════════════════ */

function buildIVF(T: number[]): Built {
  const g = new THREE.Group();
  const N = 240;
  const K = 8;
  const NPROBE = 2;
  const { pts } = clusters(N, 6, 23, HALF);
  const km = kmeans(pts, K, 5, 6);
  const { history, assign, centroids } = km;
  // Put the query between two cells so the recall trade-off is visible.
  const c0 = 0;
  const c1 = nearest(centroids, centroids[c0], 2)[1];
  const q = centroids[c0].clone().lerp(centroids[c1], 0.4).add(V(0, 0.35, 0.2));
  const probe = nearest(centroids, q, NPROBE);
  const probeSet = new Set(probe);
  const members = pts.map((_, i) => i).filter((i) => probeSet.has(assign[i]));
  const found = nearest(pts, q, 6, members);
  const foundSet = new Set(found);
  const missed = nearest(pts, q, 6).filter((i) => !foundSet.has(i));
  const dmax = Math.max(...pts.map((p) => p.distanceTo(q)));

  const dots = new Dots(N);
  const cents = new Dots(K, new THREE.OctahedronGeometry(1));
  const lines = new Lines(N * 2 + K + 20);
  const qm = new QueryMarker();
  g.add(dots.mesh, cents.mesh, lines.obj, qm.group);
  const cur = centroids.map(() => V());

  return {
    group: g,
    update(t, { camera, now }) {
      const appear = seg(t, T[0], T[0] + 0.6);
      const it = seg(t, T[0] + 0.3, T[1] - 0.2) * (history.length - 1);
      const i0 = Math.min(Math.floor(it), history.length - 2);
      const f = ease(it - i0);
      for (let j = 0; j < K; j++) cur[j].lerpVectors(history[i0][j], history[i0 + 1][j], f);
      const colorAmt = 0.25 + 0.6 * seg(t, T[0] + 0.3, T[2]);
      const assignLines = seg(t, T[1], T[2]);
      const probeF = ease(seg(t, T[2] + 0.4, T[2] + 1.4));
      const scan = seg(t, T[3], T[4]) * members.length;
      const pick = ease(seg(t, T[4], T[4] + 0.8));
      const scanned = new Set(members.slice(0, Math.floor(scan)));

      lines.reset();
      for (let i = 0; i < N; i++) {
        const cell = t < T[1] ? argmin(cur, pts[i]) : assign[i];
        tmp.copy(COL.idle).lerp(PALETTE[cell], colorAmt);
        let gain = 1;
        let s = 0.07 * ease(seg(appear, (i / N) * 0.5, (i / N) * 0.5 + 0.5));
        const probed = probeSet.has(cell);
        if (!probed) tmp.lerp(COL.dim, probeF * 0.8);
        if (scanned.has(i)) gain = 1.4;
        if (pick > 0) {
          if (foundSet.has(i)) { tmp.lerp(COL.final, pick); gain = 1 + 1.8 * pick; s *= 1 + pick; }
          else if (missed.includes(i)) { tmp.lerp(COL.danger, pick); gain = 1 + pick * (1 + pulse(now, 6)); s *= 1 + 0.8 * pick; }
        }
        dots.set(i, pts[i], s, tmp, gain);
        // cell membership lines
        const fi = ease(seg(assignLines, (i / N) * 0.6, (i / N) * 0.6 + 0.4));
        const k = (0.32 - 0.18 * seg(t, T[2] - 0.4, T[2])) * (probed ? 1 : 1 - probeF);
        lines.grow(centroids[cell], pts[i], fi, PALETTE[cell], k);
      }
      for (let j = 0; j < K; j++) {
        const s = 0.17 * ease(seg(t, T[0], T[0] + 0.5)) * (probeSet.has(j) ? 1 + 0.5 * probeF : 1 - 0.3 * probeF);
        cents.set(j, cur[j], s, PALETTE[j], probeSet.has(j) ? 1.6 + probeF : 1.6 - probeF);
      }
      // query → centroids
      const qa = ease(seg(t, T[2], T[2] + 0.5));
      for (let j = 0; j < K; j++) {
        const pr = probeSet.has(j);
        tmp2.copy(COL.white).lerp(COL.fuse, pr ? probeF : 0);
        lines.add(q, centroids[j], tmp2, qa * (pr ? 0.45 + 0.8 * probeF : 0.45 - 0.33 * probeF) * (1 - 0.6 * pick));
      }
      // query → scanned members
      members.forEach((i, j) => {
        if (j >= scan) return;
        const close = 1 - pts[i].distanceTo(q) / dmax;
        const fresh = clamp01(1 - (scan - j) / 10);
        const hit = foundSet.has(i);
        tmp2.copy(COL.dense).lerp(COL.final, hit ? pick : 0);
        lines.add(q, pts[i], tmp2, (0.08 + 0.4 * close * close + fresh * 0.5) * (1 - pick) + (hit ? 1.3 * pick : 0));
      });
      for (const i of missed) lines.add(q, pts[i], COL.danger, pick * (0.3 + 0.5 * pulse(now, 6)));
      dots.flush();
      cents.flush();
      lines.flush();
      qm.update(q, qa, camera, now);
      if (t < T[2]) return `Cells (nlist): ${K} · vectors filed: ${N}`;
      const comps = K + Math.floor(scan);
      return `Distance computations: ${K} centroids + ${Math.floor(scan)} vectors = ${comps} / ${N}` + (pick > 0.5 && missed.length ? ` · missed ${missed.length} true neighbour${missed.length > 1 ? 's' : ''} (red)` : '');
    },
  };
}

/* ═════════════════════════ 3 · HNSW ═════════════════════════ */

function buildHNSW(T: number[]): Built {
  const g = new THREE.Group();
  const r = rng(31);
  const N = 140;
  const centers = Array.from({ length: 6 }, () => new THREE.Vector2((r() * 2 - 1) * 3.8, (r() * 2 - 1) * 2.4));
  const P = Array.from({ length: N }, () => {
    const c = centers[Math.floor(r() * centers.length)];
    return new THREE.Vector2(THREE.MathUtils.clamp(c.x + gauss(r) * 1.1, -4.8, 4.8), THREE.MathUtils.clamp(c.y + gauss(r) * 0.85, -3.1, 3.1));
  });
  const order = P.map((_, i) => i).sort(() => r() - 0.5);
  const level = new Array(N).fill(0);
  order.forEach((id, k) => { level[id] = k < 6 ? 2 : k < 28 ? 1 : 0; });
  const LY = [-2.1, 0, 2.1];
  const pos = (i: number, L: number) => V(P[i].x, LY[L], P[i].y);
  const layerIds = [0, 1, 2].map((L) => P.map((_, i) => i).filter((i) => level[i] >= L));
  const edges: [number, number][][] = [[], [], []];
  const adj = [0, 1, 2].map(() => new Map<number, number[]>());
  for (let L = 0; L < 3; L++) {
    const ids = layerIds[L];
    const M = L === 0 ? 4 : 3;
    const seen = new Set<string>();
    for (const i of ids) adj[L].set(i, []);
    for (const i of ids) {
      const nb = ids.filter((j) => j !== i).sort((a, b) => P[a].distanceToSquared(P[i]) - P[b].distanceToSquared(P[i])).slice(0, M);
      for (const j of nb) {
        const key = i < j ? `${i}-${j}` : `${j}-${i}`;
        if (seen.has(key)) continue;
        seen.add(key);
        edges[L].push([i, j]);
        adj[L].get(i)!.push(j);
        adj[L].get(j)!.push(i);
      }
    }
  }
  const q2 = new THREE.Vector2(3.1, 1.9);
  const entry = [...layerIds[2]].sort((a, b) => P[b].distanceToSquared(q2) - P[a].distanceToSquared(q2))[0];
  const path: { node: number; L: number; evals: number }[] = [];
  let cur = entry;
  let evals = 0;
  for (let L = 2; L >= 0; L--) {
    path.push({ node: cur, L, evals });
    for (;;) {
      let best = cur;
      for (const nb of adj[L].get(cur)!) {
        evals++;
        if (P[nb].distanceToSquared(q2) < P[best].distanceToSquared(q2)) best = nb;
      }
      if (best === cur) break;
      cur = best;
      path.push({ node: cur, L, evals });
    }
  }
  const z0 = path.findIndex((p) => p.L === 0);
  const S = path.length - 1;
  const pathPos = path.map((p) => pos(p.node, p.L));
  const qL = LY.map((y) => V(q2.x, y, q2.y));
  const top = nearest(layerIds[0].map((i) => pos(i, 0)), qL[0], 6).map((k) => layerIds[0][k]);
  const topSet = new Set(top);

  const dots = [0, 1, 2].map((L) => new Dots(layerIds[L].length));
  const ghosts = new Dots(3);
  const walker = new Dots(1);
  const lines = new Lines(edges.flat().length + N + 60);
  const qm = new QueryMarker(0.12);
  const qTop = V(q2.x, LY[2] + 1.3, q2.y);
  const tint = [COL.dense, COL.fuse, PALETTE[3]];
  const planes = LY.map((y, L) => panel(rect(V(0, y, 0), V(1, 0, 0), V(0, 0, 1), 5.3, 3.5), tint[L], 0.05));
  const labels = LY.map((y, L) => {
    const s = label(`Layer ${L} · ${layerIds[L].length} nodes`, '#c9cde0', 0.3);
    s.position.set(-5.3, y + 0.3, 3.5);
    s.center.set(0, 0.5);
    return s;
  });
  g.add(...planes.map((p) => p.group), ...labels, ...dots.map((d) => d.mesh), ghosts.mesh, walker.mesh, lines.obj, qm.group);

  return {
    group: g,
    update(t, { camera, now }) {
      lines.reset();
      const searching = t >= T[2];
      const u = t < T[2] ? 0 : t < T[3] ? seg(t, T[2], T[3]) * z0 : z0 + seg(t, T[3], T[4] - 0.3) * (S - z0);
      const hop = Math.min(Math.floor(u), S);
      const visited = new Set(path.slice(0, hop + 1).map((p) => `${p.L}:${p.node}`));
      const curNode = path[hop];
      const pick = ease(seg(t, T[4], T[4] + 0.8));
      for (let L = 0; L < 3; L++) {
        const a = ease(seg(t, T[0] + L * 0.65, T[0] + L * 0.65 + 1.1));
        setPanel(planes[L], a, searching && curNode.L === L ? 1.8 : 1);
        fade(labels[L], a, 0.85);
        layerIds[L].forEach((id, k) => {
          let gain = 1;
          let s = 0.07 * a * (L ? 1.25 : 1);
          tmp.copy(COL.idle).lerp(tint[L], 0.55);
          if (t >= T[1] && L === 2 && id === entry) { tmp.copy(COL.sparse); gain = 1.4 + pulse(now) * 1.2; s *= 1.4; }
          if (searching && visited.has(`${L}:${id}`)) { tmp.copy(COL.final); gain = 2; s *= 1.5; }
          if (L === 0 && pick > 0) {
            if (topSet.has(id)) { tmp.lerp(COL.final, pick); gain = 1 + 1.8 * pick; s *= 1 + 0.8 * pick; }
            else tmp.lerp(COL.dim, 0.5 * pick);
          }
          dots[L].set(k, pos(id, L), s, tmp, gain);
        });
        const eg = ease(seg(t, T[0] + L * 0.65 + 0.3, T[0] + L * 0.65 + 1.4));
        edges[L].forEach(([i, j], e) => {
          const f = ease(seg(eg, (e / edges[L].length) * 0.5, (e / edges[L].length) * 0.5 + 0.5));
          const active = searching && curNode.L === L && (i === curNode.node || j === curNode.node) && t < T[4];
          lines.grow(pos(i, L), pos(j, L), f, active ? COL.sparse : tint[L], active ? 1 : searching ? 0.16 : 0.28);
        });
      }
      // express-lane links between copies of the same node
      for (let i = 0; i < N; i++) if (level[i] > 0) lines.add(pos(i, 0), pos(i, level[i]), COL.white, 0.07 * seg(t, T[0] + 1, T[1]));
      // query ghosts on every layer
      const qa = ease(seg(t, T[1], T[1] + 0.6));
      for (let L = 0; L < 3; L++) ghosts.set(L, qL[L], 0.07 * qa, COL.white, 2.6);
      lines.add(qTop, qL[0], COL.white, 0.22 * qa);
      // walker + trail
      if (searching) {
        for (let s = 0; s < S; s++) {
          if (s < Math.floor(u)) lines.add(pathPos[s], pathPos[s + 1], COL.final, 1.4);
          else if (s === Math.floor(u)) lines.grow(pathPos[s], pathPos[s + 1], u - s, COL.final, 1.4);
        }
        const s0 = Math.min(Math.floor(u), S - 1);
        const w = pathPos[s0].clone().lerp(pathPos[Math.min(s0 + 1, S)], u >= S ? 1 : u - s0);
        walker.set(0, w, 0.12 * (1 - pick), COL.final, 3);
      } else walker.set(0, qTop, 0, COL.final);
      for (const id of top) lines.add(qL[0], pos(id, 0), COL.final, 1.2 * pick);
      dots.forEach((d) => d.flush());
      ghosts.flush();
      walker.flush();
      lines.flush();
      qm.update(qTop, qa, camera, now);
      if (!searching) return `Graph links: ${edges.flat().length} · levels: 3`;
      return `Hops: ${hop} · distance computations: ${curNode.evals} / ${N}`;
    },
  };
}

/* ═════════════════════════ 4 · Product Quantization ═════════════════════════ */

function buildPQ(T: number[]): Built {
  const g = new THREE.Group();
  const r = rng(47);
  const N = 90;
  const M = 4;
  const KS = 8;
  const TX = [-3.45, -1.15, 1.15, 3.45];
  const TY = -0.7;
  const TS = 0.9;
  const STRIP_Y = 2.3;
  const sub: THREE.Vector3[][] = [];
  for (let j = 0; j < M; j++) {
    const cs = Array.from({ length: 4 }, () => V((r() * 2 - 1) * 0.75, (r() * 2 - 1) * 0.75, 0));
    sub.push(Array.from({ length: N }, () => {
      const c = cs[Math.floor(r() * cs.length)];
      return V(THREE.MathUtils.clamp(c.x + gauss(r) * 0.3, -1.05, 1.05), THREE.MathUtils.clamp(c.y + gauss(r) * 0.3, -1.05, 1.05), 0);
    }));
  }
  const cb = sub.map((s, j) => kmeans(s, KS, 100 + j * 7, 8));
  const E = 0;
  const code = cb.map((c) => c.assign[E]);
  const qsub = sub.map((s) => s[7].clone().add(V(0.18, -0.12, 0)));
  const tp = (j: number, v: THREE.Vector3) => V(TX[j] + v.x * TS, TY + v.y * TS, 0.02);

  const tiles = TX.map((x) => panel(rect(V(x, TY, -0.02), V(1, 0, 0), V(0, 1, 0), 1.06, 1.06), PALETTE[1], 0.06));
  const tileLabels = TX.map((x, j) => {
    const s = label(`sub-space ${j + 1} · dims ${2 * j + 1}–${2 * j + 2}`, '#a6a8b3', 0.21);
    s.position.set(x, TY - 1.35, 0);
    return s;
  });
  const codeLabels = code.map((c, j) => {
    const s = label(`#${c}`, '#ffffff', 0.36);
    s.position.set(TX[j], STRIP_Y + 0.62, 0);
    return s;
  });
  const memA = label('1 vector = 8 floats × 4 B = 32 bytes', '#d6d5d0', 0.3);
  memA.position.set(0, STRIP_Y + 1.25, 0);
  const memB = label('1 vector = 4 codes × 1 B = 4 bytes', '#ffd95e', 0.3);
  memB.position.set(0, STRIP_Y + 1.25, 0);
  const sum = label(`d(q, x) ≈ T₁[#${code[0]}] + T₂[#${code[1]}] + T₃[#${code[2]}] + T₄[#${code[3]}]`, '#ffd95e', 0.3);
  sum.position.set(0, TY - 1.85, 0);

  const strip = new Dots(8, new THREE.BoxGeometry(1, 1, 1));
  const subDots = new Dots(N * M);
  const cents = new Dots(M * KS, new THREE.OctahedronGeometry(1));
  const qDots = new Dots(M);
  const lines = new Lines(M * KS + 40);
  g.add(...tiles.map((p) => p.group), ...tileLabels, ...codeLabels, memA, memB, sum, strip.mesh, subDots.mesh, cents.mesh, qDots.mesh, lines.obj);

  const vals = Array.from({ length: 8 }, (_, d) => (d % 2 ? sub[d >> 1][E].y : sub[d >> 1][E].x));

  return {
    group: g,
    update(t) {
      lines.reset();
      const stripIn = ease(seg(t, T[0], T[0] + 0.5));
      const split = ease(seg(t, T[0] + 0.5, T[0] + 1.3));
      const tilesIn = ease(seg(t, T[0] + 0.9, T[1]));
      const learn = seg(t, T[1], T[2]);
      const snap = ease(seg(t, T[2], T[2] + 1.4));
      const encode = ease(seg(t, T[2] + 0.6, T[3]));
      const qIn = ease(seg(t, T[3], T[3] + 0.5));
      const table = seg(t, T[3] + 0.3, T[4]);
      const pick = ease(seg(t, T[4], T[4] + 0.9));

      for (let d = 0; d < 8; d++) {
        const j = d >> 1;
        const x0 = (d - 3.5) * 0.5;
        const x1 = TX[j] + (d % 2 ? 0.26 : -0.26);
        const p = V(THREE.MathUtils.lerp(x0, x1, split), STRIP_Y, 0);
        p.x = THREE.MathUtils.lerp(p.x, TX[j], encode);
        const h = 0.25 + Math.abs(vals[d]) * 0.55;
        ramp((vals[d] + 1.1) / 2.2, tmp).lerp(PALETTE[code[j]], encode);
        strip.set(d, p, 1, tmp, 1 + encode * 0.8);
        strip.scl[d].set(THREE.MathUtils.lerp(0.4, 0.46, encode) * stripIn, THREE.MathUtils.lerp(h, 0.46, encode) * stripIn, THREE.MathUtils.lerp(0.4, 0.46, encode) * stripIn);
      }
      for (let j = 0; j < M; j++) {
        setPanel(tiles[j], tilesIn);
        fade(tileLabels[j], tilesIn, 0.9);
        fade(codeLabels[j], encode);
        const ex = tp(j, sub[j][E]).lerp(tp(j, cb[j].centroids[code[j]]), snap);
        const from = V(TX[j], STRIP_Y - 0.45, 0);
        lines.grow(from, ex, seg(t, T[0] + 1.1, T[1]), encode > 0 ? tmp.copy(COL.white).lerp(PALETTE[code[j]], encode) : COL.white, 0.5 * (1 - 0.6 * pick));
        for (let i = 0; i < N; i++) {
          const c = cb[j].assign[i];
          const k = j * N + i;
          const st = ease(seg(learn, 0.15 + (i / N) * 0.4, 0.35 + (i / N) * 0.4));
          tmp.copy(COL.idle).lerp(PALETTE[c], st * 0.85);
          const sp = ease(seg(snap, (i / N) * 0.4, (i / N) * 0.4 + 0.6));
          const p = tp(j, sub[j][i]).lerp(tp(j, cb[j].centroids[c]), sp * 0.92);
          const isE = i === E;
          subDots.set(k, isE ? ex : p, (isE ? 0.085 : 0.045 * ease(seg(t, T[1] - 0.3, T[1] + 0.3))) * tilesIn, isE ? COL.white : tmp, isE ? 2.4 : 1);
        }
        for (let c = 0; c < KS; c++) {
          const grow = ease(seg(learn, c / KS * 0.5, c / KS * 0.5 + 0.5));
          const chosen = c === code[j];
          cents.set(j * KS + c, tp(j, cb[j].centroids[c]).add(V(0, 0, 0.05)), 0.1 * grow * (chosen && pick ? 1 + 0.6 * pick : 1), PALETTE[c], 1.5 + (chosen ? pick * 1.5 : 0));
          // lookup-table lines from the query sub-vector to every centroid
          const lf = ease(seg(table, c / KS * 0.6, c / KS * 0.6 + 0.4));
          tmp2.copy(COL.white).lerp(COL.final, chosen ? pick : 0);
          lines.grow(tp(j, qsub[j]), tp(j, cb[j].centroids[c]), lf, tmp2, chosen ? 0.45 + pick : 0.45 * (1 - 0.8 * pick));
        }
        qDots.set(j, tp(j, qsub[j]).add(V(0, 0, 0.06)), 0.085 * qIn, COL.white, 3);
      }
      fade(memA, stripIn * (1 - encode));
      fade(memB, encode);
      fade(sum, pick);
      strip.flush();
      subDots.flush();
      cents.flush();
      qDots.flush();
      lines.flush();
      if (t < T[2]) return 'Memory per vector: 32 bytes (8 × float32)';
      if (t < T[3]) return 'Memory per vector: 4 bytes — 8× smaller (real systems: 96× for 768-d)';
      return `Lookup table: ${M} × ${KS} distances, computed once per query and reused for every vector`;
    },
  };
}

/* ═════════════════════════ 5 · LSH ═════════════════════════ */

function buildLSH(T: number[]): Built {
  const g = new THREE.Group();
  const N = 230;
  const { pts, centers } = clusters(N, 6, 61, HALF);
  const r = rng(9);
  const normals = [V(0.9, 0.25, 0.35), V(-0.2, 0.95, 0.2), V(0.35, -0.15, 0.92)].map((n) => n.add(V(r() - 0.5, r() - 0.5, r() - 0.5).multiplyScalar(0.5)).normalize());
  const offs = normals.map(() => (r() - 0.5) * 1.2);
  const hash = (p: THREE.Vector3) => normals.reduce((h, n, k) => h | ((n.dot(p) - offs[k] > 0 ? 1 : 0) << k), 0);
  const codes = pts.map(hash);
  const q = centers[2].clone().add(V(0.3, 0.2, -0.2));
  const qc = hash(q);
  const bucket = pts.map((_, i) => i).filter((i) => codes[i] === qc);
  const found = nearest(pts, q, 6, bucket);
  const foundSet = new Set(found);
  const missed = nearest(pts, q, 6).filter((i) => !foundSet.has(i));
  const bits = (c: number) => [0, 1, 2].map((k) => (c >> k) & 1).join('');
  const pcol = [PALETTE[1], PALETTE[4], PALETTE[3]];

  const planes = normals.map((n, k) => {
    const u = V(0, 1, 0).cross(n).normalize();
    if (u.lengthSq() < 0.1) u.set(1, 0, 0);
    const v = n.clone().cross(u).normalize();
    const c = n.clone().multiplyScalar(offs[k]);
    const p = panel(rect(V(), u, v, 4.2, 3.1), pcol[k], 0.08);
    p.group.position.copy(c);
    return { p, c, n, tip: label(`h${'₁₂₃'[k]}`, '#ffffff', 0.34) };
  });
  planes.forEach((pl) => pl.tip.position.copy(pl.c).addScaledVector(pl.n, 1.45));
  const qLabel = label(`bucket ${bits(qc)}`, '#ffffff', 0.32);
  qLabel.position.copy(q).add(V(0, 0.55, 0));

  const dots = new Dots(N);
  const lines = new Lines(N + 20);
  const qm = new QueryMarker();
  g.add(...planes.flatMap((pl) => [pl.p.group, pl.tip]), qLabel, dots.mesh, lines.obj, qm.group);
  const dmax = Math.max(...pts.map((p) => p.distanceTo(q)));

  return {
    group: g,
    update(t, { camera, now }) {
      lines.reset();
      const appear = seg(t, T[0], T[1] - 0.2);
      const hashF = seg(t, T[2], T[3] - 0.2);
      const qa = ease(seg(t, T[3], T[3] + 0.5));
      const dimF = ease(seg(t, T[3] + 0.4, T[4]));
      const scan = seg(t, T[4], T[4] + 1.3) * bucket.length;
      const pick = ease(seg(t, T[4] + 1.3, T[4] + 2));
      planes.forEach((pl, k) => {
        const a = ease(seg(t, T[1] + k * 0.75, T[1] + k * 0.75 + 0.9));
        pl.p.group.scale.setScalar(0.3 + 0.7 * a);
        setPanel(pl.p, a * (1 - 0.45 * seg(t, T[2], T[3])));
        fade(pl.tip, a * (1 - dimF * 0.6));
        lines.grow(pl.c, pl.c.clone().addScaledVector(pl.n, 1.2), a, pcol[k], 1.2 * (1 - dimF * 0.6));
      });
      for (let i = 0; i < N; i++) {
        const a = ease(seg(appear, (i / N) * 0.6, (i / N) * 0.6 + 0.4));
        const hf = ease(seg(hashF, (i / N) * 0.6, (i / N) * 0.6 + 0.4));
        tmp.copy(COL.idle).lerp(PALETTE[codes[i]], hf * 0.9);
        let gain = 1;
        let s = 0.07 * a;
        const inB = codes[i] === qc;
        if (!inB) tmp.lerp(COL.dim, dimF * 0.85);
        else gain = 1 + 0.6 * dimF;
        if (pick > 0) {
          if (foundSet.has(i)) { tmp.lerp(COL.final, pick); gain = 1 + 1.8 * pick; s *= 1 + pick; }
          else if (missed.includes(i)) { tmp.lerp(COL.danger, pick); gain = 1 + pick * (1 + pulse(now, 6)); s *= 1 + 0.6 * pick; }
        }
        dots.set(i, pts[i], s, tmp, gain);
      }
      bucket.forEach((i, j) => {
        if (j >= scan) return;
        const close = 1 - pts[i].distanceTo(q) / dmax;
        const hit = foundSet.has(i);
        tmp2.copy(PALETTE[qc]).lerp(COL.final, hit ? pick : 0);
        lines.add(q, pts[i], tmp2, (0.15 + 0.5 * close * close) * (1 - pick) + (hit ? 1.3 * pick : 0));
      });
      for (const i of missed) lines.add(q, pts[i], COL.danger, pick * (0.3 + 0.5 * pulse(now, 6)));
      fade(qLabel, qa);
      dots.flush();
      lines.flush();
      qm.update(q, qa, camera, now);
      if (t < T[3]) return `Hyperplanes: 3 → ${2 ** 3} buckets`;
      return `3 dot products to hash, then ${Math.floor(scan)} / ${N} distance computations` + (pick > 0.5 && missed.length ? ` · missed ${missed.length} (red) — fix with more hash tables` : '');
    },
  };
}

/* ═════════════════════════ 6 · Annoy (random projection trees) ═════════════════════════ */

interface TreeNode {
  ids: number[];
  depth: number;
  lo: number;
  hi: number;
  cons: { n: THREE.Vector3; d: number }[];
  split?: { n: THREE.Vector3; d: number; a: number; b: number; panel: ReturnType<typeof panel> };
  left?: TreeNode;
  right?: TreeNode;
}

function buildAnnoy(T: number[]): Built {
  const g = new THREE.Group();
  const N = 200;
  const { pts, centers } = clusters(N, 7, 73, HALF);
  const H = V(4.8, 2.8, 3.3);
  const r = rng(17);
  const box = [
    { n: V(1, 0, 0), d: -H.x }, { n: V(-1, 0, 0), d: -H.x },
    { n: V(0, 1, 0), d: -H.y }, { n: V(0, -1, 0), d: -H.y },
    { n: V(0, 0, 1), d: -H.z }, { n: V(0, 0, -1), d: -H.z },
  ];
  const depthTint = [COL.dense, COL.fuse, PALETTE[3], COL.rerank];
  const levels: TreeNode[][] = [[], [], [], [], []];
  const leaves: TreeNode[] = [];
  const grow = (node: TreeNode) => {
    levels[node.depth].push(node);
    if (node.depth >= 4 || node.ids.length < 8) { leaves.push(node); return; }
    const a = node.ids[Math.floor(r() * node.ids.length)];
    let b = a;
    while (b === a) b = node.ids[Math.floor(r() * node.ids.length)];
    const n = pts[b].clone().sub(pts[a]).normalize();
    const d = n.dot(pts[a].clone().add(pts[b]).multiplyScalar(0.5));
    const L = node.ids.filter((i) => n.dot(pts[i]) < d);
    const R = node.ids.filter((i) => n.dot(pts[i]) >= d);
    if (!L.length || !R.length) { leaves.push(node); return; }
    const u = Math.abs(n.y) < 0.9 ? V(0, 1, 0).cross(n).normalize() : V(1, 0, 0).cross(n).normalize();
    const v = n.clone().cross(u).normalize();
    let poly = rect(n.clone().multiplyScalar(d), u, v, 20, 20);
    for (const c of [...box, ...node.cons]) poly = clip(poly, c.n, c.d);
    const pn = panel(poly, depthTint[node.depth], node.depth === 0 ? 0.06 : 0.09);
    g.add(pn.group);
    node.split = { n, d, a, b, panel: pn };
    const mid = (node.lo + node.hi) / 2;
    node.left = { ids: L, depth: node.depth + 1, lo: node.lo, hi: mid, cons: [...node.cons, { n: n.clone().negate(), d: -d }] };
    node.right = { ids: R, depth: node.depth + 1, lo: mid, hi: node.hi, cons: [...node.cons, { n, d }] };
    grow(node.left);
    grow(node.right);
  };
  const root: TreeNode = { ids: pts.map((_, i) => i), depth: 0, lo: 0, hi: 1, cons: [] };
  grow(root);
  // per point: its node at each depth
  const nodeAt: TreeNode[][] = pts.map(() => []);
  for (const lvl of levels) for (const nd of lvl) for (const i of nd.ids) nodeAt[i][nd.depth] = nd;
  const at = (i: number, d: number) => nodeAt[i][Math.min(d, nodeAt[i].length - 1)];

  const q = centers[3].clone().add(V(0.25, 0.3, 0.1));
  const qPath: TreeNode[] = [root];
  while (qPath[qPath.length - 1].split) {
    const nd = qPath[qPath.length - 1];
    qPath.push(nd.split!.n.dot(q) >= nd.split!.d ? nd.right! : nd.left!);
  }
  const leaf = qPath[qPath.length - 1];
  const found = nearest(pts, q, 6, leaf.ids);
  const foundSet = new Set(found);
  const missed = nearest(pts, q, 6).filter((i) => !foundSet.has(i));
  const pathSet = new Set(qPath);
  const dmax = Math.max(...pts.map((p) => p.distanceTo(q)));

  const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(H.x * 2, H.y * 2, H.z * 2)), new THREE.LineBasicMaterial({ color: '#4a5070', transparent: true, opacity: 0.5 }));
  const dots = new Dots(N);
  const lines = new Lines(N + 60);
  const qm = new QueryMarker();
  g.add(frame, dots.mesh, lines.obj, qm.group);
  const LV = 0.8;
  const descend = qPath.length - 1;

  return {
    group: g,
    update(t, { camera, now }) {
      lines.reset();
      const lvlT = (t - T[0]) / LV;
      const done = clamp01(Math.floor(lvlT) / 4) * 4; // levels completed during build
      const leafF = ease(seg(t, T[1], T[1] + 0.8));
      const qa = ease(seg(t, T[2], T[2] + 0.4));
      const dl = seg(t, T[2] + 0.3, T[3]) * descend; // descent progress in levels
      const scan = seg(t, T[3], T[3] + 1.2) * leaf.ids.length;
      const pick = ease(seg(t, T[3] + 1.2, T[4]));
      // split planes
      for (const lvl of levels) for (const nd of lvl) {
        if (!nd.split) continue;
        const d = nd.depth;
        const a = ease(seg(lvlT, d + 0.35, d + 1));
        let o = a * (1 - 0.45 * leafF);
        let boost = 1;
        if (t >= T[2]) {
          const k = qPath.indexOf(nd);
          if (pathSet.has(nd) && dl >= k) { boost = 1.5; o = 1; }
          else o *= 0.3;
        }
        setPanel(nd.split.panel, o, boost);
        // the two random points that define this split
        const pickF = seg(lvlT, d, d + 0.35) * (1 - seg(lvlT, d + 0.7, d + 1));
        if (pickF > 0 && t < T[1]) lines.add(pts[nd.split.a], pts[nd.split.b], COL.white, 1.2 * pickF);
      }
      const curNode = qPath[Math.min(Math.floor(dl), descend)];
      const inCur = t >= T[2] ? new Set(curNode.ids) : null;
      for (let i = 0; i < N; i++) {
        const a = ease(seg(t, T[0] + (i / N) * 0.3, T[0] + (i / N) * 0.3 + 0.3));
        const nd = at(i, t >= T[1] ? 4 : done);
        const colorAmt = t >= T[1] ? 0.9 : done > 0 ? 0.8 : 0;
        ramp((nd.lo + nd.hi) / 2, tmp);
        tmp.lerp(COL.idle, 1 - colorAmt);
        let gain = 1;
        let s = 0.07 * a;
        for (const lvl of levels) for (const n2 of lvl) {
          if (!n2.split || t >= T[1]) continue;
          const f = seg(lvlT, n2.depth, n2.depth + 0.35) * (1 - seg(lvlT, n2.depth + 0.7, n2.depth + 1));
          if (f > 0 && (n2.split.a === i || n2.split.b === i)) { tmp.lerp(COL.white, f); gain = 1 + 1.5 * f; s *= 1 + f; }
        }
        if (inCur && !inCur.has(i)) tmp.lerp(COL.dim, 0.85 * qa);
        else if (inCur) gain = 1 + 0.6 * qa;
        if (pick > 0) {
          if (foundSet.has(i)) { tmp.lerp(COL.final, pick); gain = 1 + 1.8 * pick; s *= 1 + pick; }
          else if (missed.includes(i)) { tmp.lerp(COL.danger, pick); gain = 1 + pick * (1 + pulse(now, 6)); s *= 1 + 0.6 * pick; }
        }
        dots.set(i, pts[i], s, tmp, gain);
      }
      leaf.ids.forEach((i, j) => {
        if (j >= scan) return;
        const close = 1 - pts[i].distanceTo(q) / dmax;
        const hit = foundSet.has(i);
        tmp2.copy(COL.dense).lerp(COL.final, hit ? pick : 0);
        lines.add(q, pts[i], tmp2, (0.15 + 0.5 * close * close) * (1 - pick) + (hit ? 1.3 * pick : 0));
      });
      for (const i of missed) lines.add(q, pts[i], COL.danger, pick * (0.3 + 0.5 * pulse(now, 6)));
      dots.flush();
      lines.flush();
      qm.update(q, qa, camera, now);
      if (t < T[2]) return `Tree depth: ${Math.min(4, Math.max(0, Math.floor(lvlT)))} · leaves: ${t >= T[1] ? leaves.length : 2 ** Math.min(4, Math.max(0, Math.floor(lvlT)))}`;
      const tests = Math.min(descend, Math.floor(dl));
      return `${tests} plane tests + ${Math.floor(scan)} / ${N} distance computations` + (pick > 0.5 && missed.length ? ` · missed ${missed.length} (red) — a forest of trees fixes this` : '');
    },
  };
}

/* ═════════════════════════ 7 · DiskANN (Vamana) ═════════════════════════ */

function buildDiskANN(T: number[]): Built {
  const g = new THREE.Group();
  const N = 170;
  const R = 5;
  const ALPHA = 1.1;
  const { pts } = clusters(N, 6, 89, [4.3, 1.4, 2.8]);
  pts.forEach((p) => (p.y += 1.0));
  // Robust prune over every candidate: keeps a few long-range edges because of α > 1.
  const out: number[][] = pts.map((p, i) => {
    let cands = pts.map((_, j) => j).filter((j) => j !== i).sort((a, b) => pts[a].distanceToSquared(p) - pts[b].distanceToSquared(p));
    const keep: number[] = [];
    while (cands.length && keep.length < R) {
      const s = cands.shift()!;
      keep.push(s);
      cands = cands.filter((c) => ALPHA * pts[s].distanceTo(pts[c]) > p.distanceTo(pts[c]));
    }
    return keep;
  });
  const edges = out.flatMap((nb, i) => nb.map((j) => [i, j] as [number, number]));
  const mean = pts.reduce((a, p) => a.add(p), V()).divideScalar(N);
  const medoid = argmin(pts, mean);
  const far = [...pts.keys()].sort((a, b) => pts[b].distanceToSquared(pts[medoid]) - pts[a].distanceToSquared(pts[medoid]))[0];
  const q = pts[far].clone().lerp(pts[medoid], 0.12).add(V(0.2, 0.25, 0));
  const path = [medoid];
  const evalsAt = [0];
  let cur = medoid;
  let evals = 0;
  for (;;) {
    let best = cur;
    for (const nb of out[cur]) { evals++; if (pts[nb].distanceToSquared(q) < pts[best].distanceToSquared(q)) best = nb; }
    if (best === cur) break;
    cur = best;
    path.push(cur);
    evalsAt.push(evals);
  }
  const top = nearest(pts, q, 6);
  const topSet = new Set(top);
  const COLS = 17;
  const block = (i: number) => V(-4.4 + (i % COLS) * 0.55, -2.9, -2.5 + Math.floor(i / COLS) * 0.55);

  const slab = panel(rect(V(0, -2.98, 0), V(1, 0, 0), V(0, 0, 1), 4.9, 3.0), PALETTE[1], 0.05);
  const ssd = label('SSD · 1 block per node', '#a6a8b3', 0.3);
  ssd.position.set(-3.3, -2.55, 3.2);
  const ram = label('RAM · compressed PQ codes only', '#a6a8b3', 0.28);
  ram.position.set(0, 3.45, 0);
  const dots = new Dots(N);
  const blocks = new Dots(N, new THREE.BoxGeometry(1, 1, 1));
  const lines = new Lines(edges.length + N + 60);
  const qm = new QueryMarker();
  g.add(slab.group, ssd, ram, dots.mesh, blocks.mesh, lines.obj, qm.group);
  const S = path.length - 1;

  return {
    group: g,
    update(t, { camera, now }) {
      lines.reset();
      const appear = seg(t, T[0], T[0] + 0.6);
      const eg = seg(t, T[0] + 0.3, T[1]);
      const disk = seg(t, T[1], T[2] - 0.1);
      const qa = ease(seg(t, T[2], T[2] + 0.5));
      const u = seg(t, T[3], T[4] - 0.3) * S;
      const hop = Math.min(Math.floor(u), S);
      const searching = t >= T[3];
      const curNode = path[hop];
      const visited = new Set(searching ? path.slice(0, hop + 1) : []);
      const nbrs = new Set(searching && t < T[4] ? out[curNode] : []);
      const pick = ease(seg(t, T[4], T[4] + 0.8));
      setPanel(slab, ease(disk));
      fade(ssd, ease(disk), 0.9);
      fade(ram, ease(disk), 0.9);
      for (let i = 0; i < N; i++) {
        const a = ease(seg(appear, (i / N) * 0.5, (i / N) * 0.5 + 0.5));
        tmp.copy(COL.idle).lerp(COL.fuse, 0.35);
        let gain = 1;
        let s = 0.07 * a;
        if (i === medoid && t >= T[2]) { tmp.copy(COL.sparse); gain = 1.5 + (searching ? 0 : pulse(now) * 1.2); s *= 1.5; }
        if (nbrs.has(i)) { tmp.copy(COL.sparse); gain = 1.6; s *= 1.25; }
        if (visited.has(i)) { tmp.copy(COL.final); gain = 2.2; s *= 1.5; }
        if (pick > 0) {
          if (topSet.has(i)) { tmp.lerp(COL.final, pick); gain = 1 + 1.8 * pick; s *= 1 + 0.6 * pick; }
          else if (!visited.has(i)) tmp.lerp(COL.dim, 0.5 * pick);
        }
        dots.set(i, pts[i], s, tmp, gain);
        // disk blocks
        const bf = ease(seg(disk, (i / N) * 0.7, (i / N) * 0.7 + 0.3));
        const read = visited.has(i);
        tmp2.copy(COL.idle).lerp(COL.final, read ? 1 : 0);
        blocks.set(i, block(i), 1, tmp2, read ? 2.2 : 0.9);
        blocks.scl[i].set(0.44 * bf, 0.1 * bf, 0.44 * bf);
        // write-out sweep: node → its block
        const sweep = clamp01(1 - Math.abs(disk * 1.15 - (i / N) - 0.08) * 12);
        if (disk > 0 && disk < 1) lines.add(pts[i], block(i), COL.white, 0.35 * sweep);
      }
      edges.forEach(([i, j], e) => {
        const f = ease(seg(eg, (e / edges.length) * 0.6, (e / edges.length) * 0.6 + 0.4));
        const active = i === curNode && nbrs.size > 0;
        const long = pts[i].distanceTo(pts[j]) > 2.4;
        lines.grow(pts[i], pts[j], f, active ? COL.sparse : long ? COL.rerank : COL.fuse, active ? 1.1 : searching ? 0.1 : long ? 0.55 : 0.22);
      });
      if (searching) {
        for (let s = 0; s < S; s++) {
          if (s < Math.floor(u)) lines.add(pts[path[s]], pts[path[s + 1]], COL.final, 1.4);
          else if (s === Math.floor(u)) lines.grow(pts[path[s]], pts[path[s + 1]], u - s, COL.final, 1.4);
        }
        // one SSD read per visited node
        path.slice(0, hop + 1).forEach((id, h) => lines.add(block(id), pts[id], COL.final, h === hop && t < T[4] ? 0.9 : 0.25));
      }
      for (const id of top) lines.add(q, pts[id], COL.final, 1.2 * pick);
      dots.flush();
      blocks.flush();
      lines.flush();
      qm.update(q, qa, camera, now);
      if (!searching) return `Out-degree R = ${R} · α = ${ALPHA} · edges: ${edges.length}`;
      return `SSD reads: ${hop + 1} · PQ distance estimates in RAM: ${evalsAt[hop] + (t < T[4] ? out[curNode].length : 0)}`;
    },
  };
}

/* ═════════════════════════ Catalogue ═════════════════════════ */

export const ALGOS: AlgoDef[] = [
  {
    id: 'flat', name: 'Flat', full: 'Flat index (brute-force k-NN)', family: 'Exact',
    tagline: 'Compare the query with every single vector. Slow at scale, but always exactly right.',
    steps: [
      { title: 'Store raw vectors', detail: 'Every embedding is kept at full precision in one contiguous array. There is nothing to train and no extra structure.', dur: 1.6 },
      { title: 'Query arrives', detail: 'The question is embedded into the same space as the documents.', dur: 1.0 },
      { title: 'Scan every vector', detail: 'Compute the distance from the query to all N vectors. The cost grows linearly: O(N·d) per query.', dur: 3.2 },
      { title: 'Keep the exact top-k', detail: 'A small heap keeps the k closest. Recall is 100%, which is why other indexes are measured against this one.', dur: 2.2 },
    ],
    camera: [6.5, 3.6, 10.5], spin: true,
    rating: { speed: 1, recall: 5, memory: 3, build: 5 },
    complexity: 'O(N·d) per query', params: ['metric (L2 / cosine / dot)'],
    usedIn: 'FAISS IndexFlatL2, pgvector without an index, most vector DBs for small collections',
    bestFor: 'Under ~100k vectors, ground-truth evaluation, or re-ranking a shortlist.',
    watchOut: 'Latency grows with every vector you add.',
    build: buildFlat,
  },
  {
    id: 'ivf', name: 'IVF', full: 'Inverted File index (IVF-Flat)', family: 'Clustering',
    tagline: 'Cluster the space with k-means, then only search the few clusters closest to the query.',
    steps: [
      { title: 'Train k-means centroids', detail: 'Run k-means on a sample. Each of the nlist centroids becomes the head of an inverted list, called a cell.', dur: 2.6 },
      { title: 'Assign vectors to cells', detail: 'Every vector is filed under its nearest centroid. Together the cells form a Voronoi partition of the space.', dur: 1.8 },
      { title: 'Probe the nearest cells', detail: 'The query is compared with the centroids only, then the nprobe closest cells are opened (here nprobe = 2).', dur: 1.9 },
      { title: 'Scan inside probed cells', detail: 'Exact distances are computed only for vectors inside the opened cells, a small fraction of the data.', dur: 2.4 },
      { title: 'Return top-k', detail: 'Fast and light on memory, but a true neighbour just across a cell border can be missed (shown in red). Raising nprobe trades speed for recall.', dur: 2.2 },
    ],
    camera: [5.2, 3, 8.4], spin: true,
    rating: { speed: 4, recall: 3, memory: 3, build: 3 },
    complexity: 'O(nlist·d + nprobe·N/nlist·d)', params: ['nlist (≈ √N)', 'nprobe'],
    usedIn: 'FAISS IndexIVFFlat, pgvector ivfflat, Milvus IVF_FLAT',
    bestFor: 'Large, fairly static datasets where you can afford a training pass.',
    watchOut: 'Needs retraining when the data distribution drifts; recall depends on nprobe.',
    build: buildIVF,
  },
  {
    id: 'hnsw', name: 'HNSW', full: 'Hierarchical Navigable Small World graph', family: 'Graph',
    tagline: 'A stack of proximity graphs: sparse express lanes on top, every vector on the bottom.',
    steps: [
      { title: 'Build a layered graph', detail: 'Each vector draws a random top level, with higher levels exponentially rarer. Every layer is a proximity graph; the upper layers act as sparse express lanes.', dur: 2.8 },
      { title: 'Enter at the top', detail: 'Search starts from a fixed entry point on the highest, sparsest layer.', dur: 1.2 },
      { title: 'Greedy hops, then drop down', detail: 'Move to whichever neighbour is closer to the query. When no neighbour is closer, drop one layer and repeat.', dur: 3.0 },
      { title: 'Refine on the base layer', detail: 'Layer 0 holds every vector. A beam search of width efSearch explores the local neighbourhood.', dur: 2.4 },
      { title: 'Return nearest neighbours', detail: 'Search takes about log N hops with excellent recall. The cost is the RAM needed for the graph links. Most vector databases use HNSW by default.', dur: 2.0 },
    ],
    camera: [7.5, 5.2, 10], spin: true,
    rating: { speed: 5, recall: 5, memory: 2, build: 2 },
    complexity: '≈ O(log N) hops per query', params: ['M (links per node)', 'efConstruction', 'efSearch'],
    usedIn: 'pgvector hnsw, Qdrant, Weaviate, Milvus, Elasticsearch / OpenSearch, Redis, Chroma (hnswlib)',
    bestFor: 'Low-latency, high-recall search that fits in RAM, with online inserts.',
    watchOut: 'Memory-hungry (vectors + links); deletes and rebuilds are costly.',
    build: buildHNSW,
  },
  {
    id: 'pq', name: 'PQ', full: 'Product Quantization', family: 'Compression',
    tagline: 'Chop each vector into pieces and replace every piece with a 1-byte codebook ID.',
    steps: [
      { title: 'Split into sub-vectors', detail: 'A d-dimensional vector is cut into M equal sub-vectors. Here 8 dimensions become 4 pairs, and each pair lives in its own small sub-space.', dur: 2.0 },
      { title: 'Learn a codebook per sub-space', detail: 'k-means runs separately in each sub-space and produces 256 centroids in practice (8 here). Each centroid gets an ID.', dur: 2.4 },
      { title: 'Encode: snap to centroid IDs', detail: 'Each sub-vector is replaced by the ID of its nearest centroid. The whole vector becomes M small integers.', dur: 2.2 },
      { title: 'Build query lookup tables', detail: 'For a query, compute the distance from each query sub-vector to every centroid once. This gives an M × K table.', dur: 2.2 },
      { title: 'Sum lookups = approx. distance', detail: 'The distance to any stored vector is M table lookups added together, with no floating-point vector maths. It is lossy, so results are usually re-ranked.', dur: 2.2 },
    ],
    camera: [2.6, 1.4, 12.5], target: [2, 0.75, 0], spin: false,
    rating: { speed: 4, recall: 2, memory: 5, build: 3 },
    complexity: 'O(N·M) table lookups', params: ['M (sub-quantizers)', 'nbits per code (8 → 256 centroids)'],
    usedIn: 'FAISS IndexPQ / IVFPQ, Milvus IVF_PQ, ScaNN, DiskANN (in-RAM codes)',
    bestFor: 'Billion-scale corpora where raw vectors will not fit in memory. Usually combined with IVF.',
    watchOut: 'Approximate distances lower recall; keep full vectors to re-rank.',
    build: buildPQ,
  },
  {
    id: 'lsh', name: 'LSH', full: 'Locality-Sensitive Hashing', family: 'Hashing',
    tagline: 'Random hyperplanes turn each vector into a short binary hash; similar vectors collide.',
    steps: [
      { title: 'Store vectors', detail: 'Start from the raw embeddings.', dur: 1.2 },
      { title: 'Draw random hyperplanes', detail: 'Pick k random hyperplanes. The arrows show each plane\'s positive side. No training data is needed.', dur: 2.6 },
      { title: 'Hash = side of each plane', detail: 'Each plane contributes one bit, 1 if the vector is on its positive side. Three planes give 2³ = 8 buckets.', dur: 2.0 },
      { title: 'Query lands in a bucket', detail: 'Hashing the query costs only k dot products, and points straight to one bucket.', dur: 1.5 },
      { title: 'Compare within the bucket', detail: 'Only vectors in the same bucket are scored. Neighbours across a plane are missed (red), so real systems use L independent hash tables.', dur: 2.6 },
    ],
    camera: [7, 4, 10], spin: true,
    rating: { speed: 4, recall: 2, memory: 3, build: 5 },
    complexity: 'O(L·k·d) hash + bucket scan', params: ['k (bits per hash)', 'L (hash tables)'],
    usedIn: 'FAISS IndexLSH, SimHash / MinHash for near-duplicate detection, streaming systems',
    bestFor: 'Streaming or fast-changing data and near-duplicate detection, where indexing must be instant.',
    watchOut: 'Needs many tables for good recall in high dimensions, which uses a lot of memory.',
    build: buildLSH,
  },
  {
    id: 'annoy', name: 'Annoy', full: 'Annoy: forest of random projection trees', family: 'Tree',
    tagline: 'Recursively split space with random hyperplanes; the query walks down the tree to a small leaf.',
    steps: [
      { title: 'Pick two points, split halfway', detail: 'At each node, pick two random points and cut the space with the hyperplane halfway between them. Repeat in each half.', dur: 3.4 },
      { title: 'Leaves become buckets', detail: 'Recursion stops when a cell is small enough. Each leaf stores a short list of vector IDs.', dur: 1.3 },
      { title: 'Query descends the tree', detail: 'At each level a single dot product decides which side of the plane the query falls on. That takes log N tests to reach a leaf.', dur: 2.4 },
      { title: 'Search the leaf', detail: 'Score the vectors in the leaf. Annoy builds a forest of such trees and merges their leaves (search_k), which recovers neighbours a single tree misses.', dur: 2.4 },
    ],
    camera: [7, 4.2, 10], spin: true,
    rating: { speed: 3, recall: 3, memory: 3, build: 4 },
    complexity: 'O(trees · log N + search_k)', params: ['n_trees', 'search_k'],
    usedIn: 'Spotify Annoy (mmap-friendly, read-only files); KD-trees / ball trees in scikit-learn',
    bestFor: 'Static datasets shared across processes through memory-mapped files; low-dimensional data.',
    watchOut: 'Index is immutable (rebuild to add data); weaker than graphs in high dimensions.',
    build: buildAnnoy,
  },
  {
    id: 'diskann', name: 'DiskANN', full: 'DiskANN (Vamana graph on SSD)', family: 'Graph + disk',
    tagline: 'A flat graph with long-range shortcuts, stored on SSD, so billion-scale search fits on one machine.',
    steps: [
      { title: 'Build a Vamana graph', detail: 'Each node keeps at most R out-edges chosen by robust pruning with α > 1, which deliberately keeps long-range links (green). Search can then reach anywhere in a few hops.', dur: 2.8 },
      { title: 'Store the graph on SSD', detail: 'Full vectors and neighbour lists are written to fixed-size disk blocks. Only compressed PQ codes stay in RAM.', dur: 1.8 },
      { title: 'Start at the medoid', detail: 'Search always begins at the medoid, the point closest to the centre of the data.', dur: 1.1 },
      { title: 'Beam search: 1 SSD read per hop', detail: 'Neighbours are scored cheaply with in-RAM PQ codes (orange). The best one is fetched from SSD, giving one block read per hop.', dur: 3.4 },
      { title: 'Re-rank with full vectors', detail: 'The full-precision vectors read along the way re-rank the candidates, giving HNSW-level recall at a fraction of the RAM.', dur: 2.0 },
    ],
    camera: [7.5, 3.8, 10.5], target: [0, -0.6, 0], spin: true,
    rating: { speed: 4, recall: 5, memory: 5, build: 2 },
    complexity: '≈ O(log N) hops, ~1 SSD read each', params: ['R (max degree)', 'L (search list)', 'α (pruning)', 'PQ bytes'],
    usedIn: 'Microsoft DiskANN (Bing), Milvus DISKANN, Azure Cosmos DB, pgvectorscale (StreamingDiskANN)',
    bestFor: 'Hundreds of millions to billions of vectors on commodity hardware with NVMe SSDs.',
    watchOut: 'Latency depends on SSD speed, and building the graph is slow.',
    build: buildDiskANN,
  },
];
