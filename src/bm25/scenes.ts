import * as THREE from 'three';
import type { Built, SceneDef } from '../indexing/algorithms';
import { COL, PALETTE, Dots, Lines, clamp01, ease, fade, label, panel, ramp, rect, seg, setPanel } from '../indexing/kit';
import { BM25, tokenize } from '../lib/bm25';
import { B, DOCS, K1, QUERY, SHOWCASE_TERMS, corpusStats, tfPart } from './corpus';

export type FormulaPart = 'count' | 'sum' | 'idf' | 'tf' | 'len' | 'all';

export interface Chapter extends SceneDef {
  name: string;
  full: string;
  tagline: string;
  part: FormulaPart;
  insight: string;
}

const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
const tmp = new THREE.Color();
const NEUTRAL = new THREE.Color('#6b7390');
const QTERMS = [...new Set(tokenize(QUERY))];
const TERM_COL = [PALETTE[1], PALETTE[3], PALETTE[4]];
const qColor = (term: string) => TERM_COL[QTERMS.indexOf(term)] ?? NEUTRAL;
const byDf = (st: ReturnType<typeof corpusStats>) => [...SHOWCASE_TERMS].sort((a, b) => st.df(a) - st.df(b));
const fmt = (x: number) => x.toFixed(2);

/* ───────────── A 3D word tile: a slab with text on its front face ───────────── */

function textTexture(text: string) {
  const font = '600 64px -apple-system, system-ui, "Segoe UI", Roboto, sans-serif';
  const cv = document.createElement('canvas');
  const ctx = cv.getContext('2d')!;
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 36;
  cv.width = w;
  cv.height = 96;
  ctx.font = font;
  ctx.fillStyle = '#ffffff';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, w / 2, 50);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return { tex, aspect: w / 96 };
}

class Tile {
  group = new THREE.Group();
  w: number;
  private boxMat: THREE.MeshStandardMaterial;
  private faces: THREE.Mesh[];
  constructor(texts: string[], color: THREE.Color = NEUTRAL, h = 0.46) {
    const tx = texts.map(textTexture);
    const th = h * 0.74;
    this.w = th * Math.max(...tx.map((t) => t.aspect)) + 0.1;
    this.boxMat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.1, transparent: true });
    this.group.add(new THREE.Mesh(new THREE.BoxGeometry(this.w, h, 0.16), this.boxMat));
    this.faces = tx.map((t) => {
      const m = new THREE.Mesh(
        new THREE.PlaneGeometry(th * t.aspect, th),
        new THREE.MeshBasicMaterial({ map: t.tex, transparent: true, depthWrite: false, toneMapped: false }),
      );
      m.position.z = 0.082;
      this.group.add(m);
      return m;
    });
    this.style(1, color);
    this.show(0);
  }
  show(i: number) { this.faces.forEach((f, k) => (f.visible = k === i)); }
  style(opacity: number, color: THREE.Color, gain = 1) {
    this.group.visible = opacity > 0.01;
    this.boxMat.opacity = opacity;
    this.boxMat.color.copy(color).multiplyScalar(0.42 * gain);
    this.boxMat.emissive.copy(color).multiplyScalar(0.22 * gain);
    for (const f of this.faces) (f.material as THREE.MeshBasicMaterial).opacity = opacity;
  }
}

/* ═════════════ 1 · Tokenize ═════════════ */

function buildTokenize(T: number[]): Built {
  const g = new THREE.Group();
  const RAW = DOCS[5];
  const words = RAW.split(' ');
  const info = words.map((raw) => {
    const lower = raw.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
    const tok = tokenize(raw);
    return { raw, lower, term: tok[0] ?? lower, kept: tok.length > 0 };
  });
  const terms = [...new Set(info.filter((x) => x.kept).map((x) => x.term))];
  const termCol = (term: string) => (QTERMS.includes(term) ? qColor(term) : [PALETTE[2], PALETTE[5], PALETTE[0], PALETTE[6]][terms.indexOf(term) % 4]);
  const tiles = info.map((x) => new Tile(x.term !== x.lower ? [x.raw, x.lower, x.term] : [x.raw, x.lower]));
  g.add(...tiles.map((t) => t.group));

  // sentence layout, wrapped into centred rows
  const rowPos: THREE.Vector3[] = [];
  const rows: number[][] = [[]];
  let width = 0;
  tiles.forEach((t, i) => {
    if (width + t.w > 8.4 && rows[rows.length - 1].length) { rows.push([]); width = 0; }
    rows[rows.length - 1].push(i);
    width += t.w + 0.14;
  });
  rows.forEach((row, r) => {
    const total = row.reduce((a, i) => a + tiles[i].w + 0.14, -0.14);
    let x = -total / 2;
    for (const i of row) { rowPos[i] = V(x + tiles[i].w / 2, 1.5 - r * 0.62, 0); x += tiles[i].w + 0.14; }
  });
  // bag of words: one stack per term
  const counts = new Map<string, number>();
  const bagPos: THREE.Vector3[] = [];
  const keptOrder: number[] = [];
  let k = 0;
  info.forEach((x, i) => {
    if (!x.kept) return;
    const s = counts.get(x.term) ?? 0;
    counts.set(x.term, s + 1);
    bagPos[i] = V((terms.indexOf(x.term) - (terms.length - 1) / 2) * 1.6, -1.55 + s * 0.5, 0);
    keptOrder[i] = k++;
  });
  const tfLabels = terms.map((term, j) => {
    const n = counts.get(term)!;
    const s = label(`tf ${n}`, n > 1 ? '#ffd95e' : '#a6a8b3', 0.3);
    s.position.set((j - (terms.length - 1) / 2) * 1.6, -1.55 + n * 0.5 + 0.05, 0);
    return s;
  });
  g.add(...tfLabels);
  const kept = info.filter((x) => x.kept).length;

  return {
    group: g,
    update(t) {
      info.forEach((x, i) => {
        const tile = tiles[i];
        const a = ease(seg(t, T[0] + i * 0.07, T[0] + i * 0.07 + 0.45));
        const flip = ease(seg(t, T[1] + i * 0.05, T[1] + i * 0.05 + 0.7));
        let face = flip > 0.5 ? 1 : 0;
        const pos = rowPos[i].clone();
        let op = a;
        let rotZ = 0;
        let scale = 0.6 + 0.4 * a;
        let gain = 1;
        tmp.copy(NEUTRAL);
        if (!x.kept) {
          const f = ease(seg(t, T[2] + 0.2, T[2] + 1.3));
          tmp.lerp(COL.danger, clamp01(f * 3));
          pos.y -= 3.2 * f * f;
          rotZ = f * 0.8;
          op *= 1 - f;
        } else {
          if (x.term !== x.lower) {
            const f = seg(t, T[3] + 0.3, T[3] + 1.1);
            if (f >= 0.5) face = 2;
            const p = Math.sin(Math.PI * f);
            scale *= 1 + 0.35 * p;
            gain = 1 + 2 * p;
            tmp.lerp(COL.final, p);
          }
          const f = ease(seg(t, T[4] + keptOrder[i] * 0.1, T[4] + keptOrder[i] * 0.1 + 0.9));
          if (f > 0) {
            pos.lerp(bagPos[i], f);
            pos.y += Math.sin(Math.PI * f) * 1.0;
            tmp.copy(NEUTRAL).lerp(termCol(x.term), f);
          }
        }
        tile.group.position.copy(pos);
        tile.group.rotation.set(0, flip * Math.PI * 2, rotZ);
        tile.group.scale.setScalar(scale);
        tile.show(face);
        tile.style(op, tmp, gain);
      });
      tfLabels.forEach((s) => fade(s, ease(seg(t, T[4] + 1.5, T[4] + 2.2))));
      if (t < T[2]) return `${words.length} raw words`;
      if (t < T[4]) return `${kept} tokens kept · ${words.length - kept} stopwords dropped · "powers" → "power"`;
      return `${terms.length} unique terms · tf(war) = ${counts.get('war')} in this document`;
    },
  };
}

/* ═════════════ 2 · Inverted index ═════════════ */

function buildIndex(T: number[]): Built {
  const g = new THREE.Group();
  const st = corpusStats();
  const terms = byDf(st);
  const X = (d: number) => -3.85 + d * 1.1;
  const Z = (r: number) => -2.5 + r * 0.72;
  const CARD_Z = -3.9;
  const postings: { d: number; r: number; tf: number }[] = [];
  terms.forEach((term, r) => { for (let d = 0; d < st.n; d++) { const tf = st.tf(term, d); if (tf) postings.push({ d, r, tf }); } });
  const rWar = terms.indexOf('war');
  const rowCol = terms.map((term, r) => (QTERMS.includes(term) ? qColor(term) : PALETTE[[0, 2, 5, 6, 7][r % 5]]));
  const warDocs = new Set(postings.filter((p) => p.r === rWar).map((p) => p.d));

  const cards = new Dots(st.n, new THREE.BoxGeometry(1, 1, 1));
  const cubes = new Dots(postings.length, new THREE.BoxGeometry(1, 1, 1));
  const cardLabels = DOCS.map((_, d) => { const s = label(`D${d + 1}`, '#d6d8e2', 0.3); s.position.set(X(d), 1.5, CARD_Z); return s; });
  const rowLabels = terms.map((term, r) => { const s = label(term, '#d6d8e2', 0.3); s.position.set(-4.65, 0.08, Z(r)); s.center.set(1, 0.5); return s; });
  const strips = terms.map((_, r) => panel(rect(V(0, 0.005, Z(r)), V(1, 0, 0), V(0, 0, 1), 4.55, 0.3), rowCol[r], 0.06));
  const query = new Tile(['war'], qColor('war'));
  const lines = new Lines(40);
  g.add(cards.mesh, cubes.mesh, ...cardLabels, ...rowLabels, ...strips.map((s) => s.group), query.group, lines.obj);
  const top = (p: { d: number; r: number; tf: number }) => V(X(p.d), p.tf * 0.3, Z(p.r));

  return {
    group: g,
    update(t) {
      lines.reset();
      const look = ease(seg(t, T[3], T[3] + 0.7));
      const touch = ease(seg(t, T[4], T[4] + 0.8));
      for (let d = 0; d < st.n; d++) {
        const a = ease(seg(t, T[0] + d * 0.08, T[0] + d * 0.08 + 0.5));
        tmp.copy(NEUTRAL);
        let gain = 1;
        if (touch > 0) { if (warDocs.has(d)) { tmp.lerp(COL.final, touch); gain = 1 + touch; } else tmp.lerp(COL.dim, touch); }
        cards.set(d, V(X(d), 0.6, CARD_Z), 1, tmp, gain);
        cards.scl[d].set(0.8 * a, 1.2 * a, 0.06);
        fade(cardLabels[d], a * (touch > 0 && !warDocs.has(d) ? 1 - 0.6 * touch : 1));
      }
      const sweep = seg(t, T[2], T[3]) * terms.length;
      postings.forEach((p, i) => {
        const delay = p.d * 0.2 + p.r * 0.04;
        const f = ease(seg(t, T[1] + delay, T[1] + delay + 0.7));
        const h = p.tf * 0.3;
        const pos = V(X(p.d), 0.6, CARD_Z).lerp(V(X(p.d), h / 2, Z(p.r)), f);
        pos.y += Math.sin(Math.PI * f) * 1.2;
        tmp.copy(NEUTRAL).lerp(rowCol[p.r], 0.8);
        let gain = 1 + (t >= T[2] && t < T[3] && Math.abs(p.r + 0.5 - sweep) < 0.8 ? 1.2 : 0);
        if (look > 0) { if (p.r === rWar) { tmp.lerp(COL.final, look); gain = 1 + 1.2 * look; } else tmp.lerp(COL.dim, 0.6 * look); }
        cubes.set(i, pos, 1, tmp, gain);
        cubes.scl[i].set(0.5 * Math.max(f, 0.001), Math.max(0.06, h * f), 0.5 * Math.max(f, 0.001));
        if (p.r === rWar) {
          lines.add(V(0, 2.2, Z(rWar)), top(p), COL.final, 0.9 * seg(t, T[3] + 0.6, T[3] + 1.2) * (1 - 0.6 * touch));
          lines.grow(top(p), V(X(p.d), 0.1, CARD_Z), touch, COL.final, 0.8);
        }
      });
      terms.forEach((_, r) => {
        const a = ease(seg(t, T[2], T[2] + 0.6));
        setPanel(strips[r], a * (look > 0 && r !== rWar ? 1 - 0.6 * look : 1), r === rWar ? 1 + 1.2 * look : 1);
        fade(rowLabels[r], a * (look > 0 && r !== rWar ? 1 - 0.55 * look : 1));
      });
      cards.flush();
      cubes.flush();
      lines.flush();
      query.group.position.set(0, 4.6 - 2.3 * look, Z(rWar));
      query.style(look, qColor('war'), 1.3);
      const list = postings.filter((p) => p.r === rWar).map((p) => `D${p.d + 1}(${p.tf})`).join(' ');
      if (t < T[3]) return `${postings.length} postings for ${terms.length} terms × ${st.n} documents`;
      if (t < T[4]) return `posting list · war → ${list}`;
      return `Scored ${warDocs.size} of ${st.n} documents; the other ${st.n - warDocs.size} are never read`;
    },
  };
}

/* ═════════════ 3 · IDF ═════════════ */

function buildIdf(T: number[]): Built {
  const g = new THREE.Group();
  const st = corpusStats();
  const terms = byDf(st);
  const X = (i: number) => -3.85 + i * 1.1;
  const H = 2.0;
  const BACK = -2.3;
  const dfX = (df: number) => -4 + ((df - 0.5) / 7.5) * 8;
  const idfOf = (df: number) => Math.log(1 + (st.n - df + 0.5) / (df + 0.5));
  const maxIdf = idfOf(1);
  const chipsOf = terms.map((term) => st.df(term));
  const chips = new Dots(chipsOf.reduce((a, b) => a + b, 0), new THREE.BoxGeometry(1, 1, 1));
  const bars = new Dots(terms.length, new THREE.BoxGeometry(1, 1, 1));
  const balls = new Dots(terms.length);
  const lines = new Lines(200);
  const back = panel(rect(V(0, 1.85, BACK - 0.02), V(1, 0, 0), V(0, 1, 0), 4.4, 2.15), PALETTE[1], 0.04);
  const nameL = terms.map((term, i) => { const s = label(term, QTERMS.includes(term) ? '#ffffff' : '#c9cde0', 0.28); s.position.set(X(i), -0.35, 1.45); return s; });
  const dfL = terms.map((term, i) => { const s = label(`df ${st.df(term)}`, '#a6a8b3', 0.24); s.position.set(X(i), 0.1 + st.df(term) * 0.2 + 0.18, 0); return s; });
  const idfL = terms.map((term, i) => { const s = label(fmt(st.idf(term)), '#ffd95e', 0.26); s.position.set(X(i), st.idf(term) * H + 0.25, 1.0); return s; });
  const axisX = label('document frequency →', '#a6a8b3', 0.24);
  axisX.position.set(2.7, -0.05, BACK);
  const axisY = label('IDF', '#a6a8b3', 0.26);
  axisY.position.set(-4.6, 3.6, BACK);
  g.add(back.group, chips.mesh, bars.mesh, balls.mesh, lines.obj, ...nameL, ...dfL, ...idfL, axisX, axisY);

  return {
    group: g,
    update(t) {
      lines.reset();
      const curve = seg(t, T[2], T[2] + 1.3);
      const q = ease(seg(t, T[3], T[3] + 0.6));
      let c = 0;
      terms.forEach((term, i) => {
        const df = chipsOf[i];
        const isQ = QTERMS.includes(term);
        const f1 = ease(seg(t, T[1] + i * 0.1, T[1] + i * 0.1 + 0.9));
        for (let j = 0; j < df; j++, c++) {
          const a = ease(seg(t, T[0] + i * 0.12 + j * 0.06, T[0] + i * 0.12 + j * 0.06 + 0.35));
          tmp.copy(NEUTRAL).lerp(PALETTE[i], 0.7).lerp(COL.dim, 0.65 * f1);
          chips.set(c, V(X(i), 0.1 + j * 0.2, 0), 1, tmp);
          chips.scl[c].set(0.6 * a, 0.16 * a, 0.6 * a);
        }
        const idf = st.idf(term);
        const h = idf * H * f1;
        tmp.copy(COL.dense).lerp(COL.final, idf / maxIdf);
        let gain = 0.95;
        if (q > 0) { if (isQ) gain += 0.45 * q; else tmp.lerp(COL.dim, 0.7 * q); }
        bars.set(i, V(X(i), h / 2, 1.0), 1, tmp, gain);
        bars.scl[i].set(0.55, Math.max(h, 0.001), 0.55);
        fade(nameL[i], ease(seg(t, T[0] + i * 0.12, T[0] + i * 0.12 + 0.4)) * (q > 0 && !isQ ? 1 - 0.6 * q : 1));
        fade(dfL[i], ease(seg(t, T[0] + i * 0.12 + 0.4, T[0] + i * 0.12 + 0.8)) * (1 - 0.7 * f1));
        fade(idfL[i], f1 * (q > 0 && !isQ ? 1 - 0.7 * q : 1));
        // ball flies from the bar top onto the IDF curve
        const f2 = ease(seg(t, T[2] + 0.3 + i * 0.12, T[2] + 1.2 + i * 0.12));
        const from = V(X(i), idf * H, 1.0);
        const to = V(dfX(df), idf * H, BACK);
        balls.set(i, from.clone().lerp(to, f2), f2 > 0 ? 0.11 : 0, tmp, 1.8);
        if (f2 > 0) lines.add(from, from.clone().lerp(to, f2), tmp, 0.25);
      });
      setPanel(back, ease(seg(t, T[2], T[2] + 0.6)));
      fade(axisX, ease(seg(t, T[2], T[2] + 0.6)));
      fade(axisY, ease(seg(t, T[2], T[2] + 0.6)));
      const SEG = 60;
      for (let s = 0; s < SEG; s++) {
        if (s / SEG >= curve) break;
        const d0 = 0.5 + (7.5 * s) / SEG;
        const d1 = 0.5 + (7.5 * (s + 1)) / SEG;
        lines.add(V(dfX(d0), idfOf(d0) * H, BACK), V(dfX(d1), idfOf(d1) * H, BACK), ramp(1 - s / SEG), 1.2);
      }
      lines.add(V(-4.1, 0, BACK), V(4.1, 0, BACK), COL.white, 0.25 * curve);
      bars.flush();
      chips.flush();
      balls.flush();
      lines.flush();
      if (t < T[1]) return `N = ${st.n} documents · df = how many contain the term`;
      if (t < T[3]) return 'idf = ln(1 + (N − df + 0.5) / (df + 0.5))';
      return `query weights · ${QTERMS.map((x) => `${x} ${fmt(st.idf(x))}`).join(' · ')}`;
    },
  };
}

/* ═════════════ 4 · TF saturation ═════════════ */

function buildTf(T: number[]): Built {
  const g = new THREE.Group();
  const Y0 = -1.6;
  const S = 1.05;
  const X = (tf: number) => -4.5 + tf * 0.9;
  const KS = [0.3, 0.8, 1.5, 2.2, 3.0];
  const MAIN = 2;
  const Zk = (k: number) => 2.0 - k * 1.0;
  const f = (tf: number, k1: number) => (tf * (k1 + 1)) / (tf + k1);
  const raw = (tf: number) => tf * 0.42;
  const bars = new Dots(10, new THREE.BoxGeometry(1, 1, 1));
  const ball = new Dots(1);
  const lines = new Lines(700);
  const curtains = KS.map((k1, ki) => {
    const pos: number[] = [];
    const N = 50;
    for (let s = 0; s < N; s++) {
      const a = (10 * s) / N;
      const b = (10 * (s + 1)) / N;
      const z = Zk(ki);
      const p = [V(X(a), Y0, z), V(X(b), Y0, z), V(X(b), Y0 + f(b, k1) * S, z), V(X(a), Y0 + f(a, k1) * S, z)];
      pos.push(...p[0].toArray(), ...p[1].toArray(), ...p[2].toArray(), ...p[0].toArray(), ...p[2].toArray(), ...p[3].toArray());
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    const col = ki === MAIN ? COL.final : ramp(ki / (KS.length - 1));
    const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false, toneMapped: false }));
    return { m, col };
  });
  const kLabels = KS.map((k1, ki) => { const s = label(`k1 = ${k1}`, ki === MAIN ? '#ffd95e' : '#c9cde0', 0.26); s.position.set(X(10) + 0.25, Y0 + f(10, k1) * S, Zk(ki)); s.center.set(0, 0.5); return s; });
  const axisX = label('term frequency (tf) →', '#a6a8b3', 0.26);
  axisX.position.set(X(5), Y0 - 0.4, 2.6);
  const axisY = label('score contribution', '#a6a8b3', 0.26);
  axisY.position.set(X(0) + 0.2, Y0 + 4.7, 0);
  axisY.center.set(0, 0.5);
  g.add(...curtains.map((c) => c.m), bars.mesh, ball.mesh, lines.obj, ...kLabels, axisX, axisY);

  return {
    group: g,
    update(t) {
      lines.reset();
      const morph = ease(seg(t, T[1] + 0.2, T[1] + 1.4));
      const rawLine = seg(t, T[0] + 0.2, T[0] + 1.8) * (1 - 0.7 * morph);
      for (let n = 1; n <= 10; n++) {
        const a = ease(seg(t, T[0] + n * 0.14, T[0] + n * 0.14 + 0.3));
        const h = THREE.MathUtils.lerp(raw(n), f(n, KS[MAIN]) * S, morph) * a;
        tmp.copy(COL.danger).lerp(COL.final, morph).lerp(COL.dim, 0.5 * seg(t, T[2], T[2] + 0.8));
        bars.set(n - 1, V(X(n), Y0 + h / 2, 0), 1, tmp, 1.2);
        bars.scl[n - 1].set(0.34, Math.max(h, 0.001), 0.34);
      }
      lines.grow(V(X(0), Y0, 0), V(X(10), Y0 + raw(10), 0), rawLine > 0 ? 1 : 0, COL.danger, 0.9 * rawLine);
      // axes
      lines.add(V(X(0), Y0, 2.4), V(X(10.4), Y0, 2.4), COL.white, 0.3);
      lines.add(V(X(0), Y0, 0), V(X(0), Y0 + 4.5, 0), COL.white, 0.3);
      const SEG = 60;
      KS.forEach((k1, ki) => {
        const show = ki === MAIN ? seg(t, T[1], T[1] + 1.3) : seg(t, T[2] + 0.35 * Math.abs(ki - MAIN), T[2] + 0.35 * Math.abs(ki - MAIN) + 1);
        const op = ki === MAIN ? Math.max(show, 0) : ease(show);
        (curtains[ki].m.material as THREE.MeshBasicMaterial).opacity = (ki === MAIN ? 0.12 : 0.06) * ease(ki === MAIN ? seg(t, T[2], T[2] + 0.8) : show);
        curtains[ki].m.visible = (curtains[ki].m.material as THREE.MeshBasicMaterial).opacity > 0.005;
        for (let s = 0; s < SEG; s++) {
          if (s / SEG >= show) break;
          const a = (10 * s) / SEG;
          const b = (10 * (s + 1)) / SEG;
          lines.add(V(X(a), Y0 + f(a, k1) * S, Zk(ki)), V(X(b), Y0 + f(b, k1) * S, Zk(ki)), curtains[ki].col, ki === MAIN ? 1.4 : 1);
        }
        // ceiling k1 + 1 (dashed)
        const ceil = ease(seg(t, T[3] + 0.15 * ki, T[3] + 0.15 * ki + 0.6));
        for (let x = 0; x < 10; x += 0.5) lines.add(V(X(x), Y0 + (k1 + 1) * S, Zk(ki)), V(X(x + 0.25), Y0 + (k1 + 1) * S, Zk(ki)), curtains[ki].col, 0.6 * ceil);
        fade(kLabels[ki], ki === MAIN ? ease(seg(t, T[1] + 1, T[1] + 1.5)) : op);
      });
      const tfNow = seg(t, T[1] + 0.3, T[2]) * 10;
      ball.set(0, V(X(tfNow), Y0 + f(tfNow, KS[MAIN]) * S, 0), t >= T[1] + 0.3 && t < T[2] + 0.5 ? 0.13 : 0, COL.white, 2.5);
      fade(axisX, ease(seg(t, T[0], T[0] + 0.5)));
      fade(axisY, ease(seg(t, T[0], T[0] + 0.5)));
      bars.flush();
      ball.flush();
      lines.flush();
      if (t < T[1]) return 'Raw count: 10 mentions would count 10× as much as one';
      if (t < T[2]) {
        const n = Math.max(1, Math.round(tfNow));
        return `k1 = 1.5 · tf ${n} → ${fmt(f(n, 1.5))}` + (n > 1 ? ` (+${fmt(f(n, 1.5) - f(n - 1, 1.5))} for the extra mention)` : '');
      }
      if (t < T[3]) return 'Small k1 ≈ "does the term appear at all?" · large k1 ≈ closer to the raw count';
      return `Each curve flattens toward k1 + 1 · k1 = 1.5 caps at ${fmt(2.5)}`;
    },
  };
}

/* ═════════════ 5 · Length normalisation ═════════════ */

function buildLength(T: number[]): Built {
  const g = new THREE.Group();
  const st = corpusStats();
  const Y0 = -1.6;
  const DOCS2 = [0, 6];
  const cols = [PALETTE[0], PALETTE[3]];
  const TX = [-5.0, -3.8];
  const CUBE = 0.105;
  const dls = DOCS2.map((d) => st.lengths[d]);
  const warAt = DOCS2.map((d) => st.tokens[d].indexOf('war'));
  const tokens = new Dots(dls[0] + dls[1], new THREE.BoxGeometry(1, 1, 1));
  const bars = new Dots(2, new THREE.BoxGeometry(1, 1, 1));
  const marks = new Dots(2);
  const lines = new Lines(600);
  const avg = panel(rect(V(-4.4, Y0 + st.avgdl * CUBE, 0), V(1, 0, 0), V(0, 0, 1), 1.25, 0.65), COL.white, 0.1);
  const avgL = label(`avgdl = ${st.avgdl.toFixed(1)}`, '#d6d8e2', 0.24);
  avgL.position.set(-4.4, Y0 + st.avgdl * CUBE + 0.22, -0.7);
  const docL = DOCS2.map((d, i) => { const s = label(`D${d + 1} · ${dls[i]} tokens`, '#d6d8e2', 0.24); s.position.set(TX[i], Y0 - 0.35, 1.0); return s; });

  // surface: x = |D| / avgdl, z = b, y = tf-part for tf = 1
  const Lx = (L: number) => -1.6 + ((L - 0.2) / 2.8) * 6.2;
  const Bz = (b: number) => 1.6 - b * 3.6;
  const sOf = (L: number, b: number) => tfPart(1, L, 1, K1, b);
  const SH = 1.6;
  const NX = 32;
  const NZ = 14;
  const pos: number[] = [];
  const col: number[] = [];
  const idx: number[] = [];
  for (let j = 0; j <= NZ; j++) for (let i = 0; i <= NX; i++) {
    const L = 0.2 + (2.8 * i) / NX;
    const b = j / NZ;
    const s = sOf(L, b);
    pos.push(Lx(L), Y0 + s * SH, Bz(b));
    ramp(clamp01((s - 0.4) / 1.6), tmp);
    col.push(tmp.r, tmp.g, tmp.b);
  }
  for (let j = 0; j < NZ; j++) for (let i = 0; i < NX; i++) {
    const a = j * (NX + 1) + i;
    idx.push(a, a + 1, a + NX + 1, a + 1, a + NX + 2, a + NX + 1);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  geo.setIndex(idx);
  const surface = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false, toneMapped: false }));
  const axisL = label('|D| / avgdl →', '#a6a8b3', 0.24);
  axisL.position.set(Lx(1.6), Y0 - 0.45, Bz(0) + 0.3);
  const b0 = label('b = 0', '#a6a8b3', 0.24);
  b0.position.set(Lx(3) + 0.55, Y0 + sOf(3, 0) * SH, Bz(0));
  const b1 = label('b = 1', '#a6a8b3', 0.24);
  b1.position.set(Lx(3) + 0.55, Y0 + sOf(3, 1) * SH, Bz(1));
  g.add(surface, avg.group, avgL, ...docL, tokens.mesh, bars.mesh, marks.mesh, lines.obj, axisL, b0, b1);
  const Ls = dls.map((dl) => dl / st.avgdl);

  return {
    group: g,
    update(t, { now }) {
      lines.reset();
      const b = t < T[2] ? 0 : t < T[3] ? ease(seg(t, T[2] + 0.2, T[3] - 0.2)) : 1 - 0.25 * ease(seg(t, T[3] + 0.9, T[4] - 0.2));
      const glow = ease(seg(t, T[1], T[1] + 0.5));
      let c = 0;
      DOCS2.forEach((_, i) => {
        for (let j = 0; j < dls[i]; j++, c++) {
          const a = ease(seg(t, T[0] + j * 0.04 + i * 0.2, T[0] + j * 0.04 + i * 0.2 + 0.3));
          const war = j === warAt[i];
          tmp.copy(cols[i]).multiplyScalar(0.6);
          let gain = 1;
          if (war) { tmp.lerp(COL.final, glow); gain = 1 + glow * (1.4 + 0.6 * Math.sin(now * 4)); }
          tokens.set(c, V(TX[i], Y0 + 0.05 + j * CUBE, 0), 1, tmp, gain);
          tokens.scl[c].set(0.6 * a, 0.085 * a, 0.6 * a);
        }
        fade(docL[i], ease(seg(t, T[0], T[0] + 0.6)));
        const s = sOf(Ls[i], b);
        const h = s * SH * glow;
        bars.set(i, V(TX[i], Y0 + h / 2, 1.1), 1, cols[i], 1.6);
        bars.scl[i].set(0.4, Math.max(h, 0.001), 0.4);
      });
      setPanel(avg, glow);
      fade(avgL, glow);
      const surf = ease(seg(t, T[3], T[3] + 0.9));
      (surface.material as THREE.MeshBasicMaterial).opacity = 0.32 * surf;
      surface.visible = surf > 0.01;
      fade(axisL, surf);
      fade(b0, surf);
      fade(b1, surf);
      // wire grid over the surface
      for (let j = 0; j <= NZ; j += 2) for (let i = 0; i < NX; i++) {
        const L0 = 0.2 + (2.8 * i) / NX;
        const L1 = 0.2 + (2.8 * (i + 1)) / NX;
        const bb = j / NZ;
        lines.add(V(Lx(L0), Y0 + sOf(L0, bb) * SH, Bz(bb)), V(Lx(L1), Y0 + sOf(L1, bb) * SH, Bz(bb)), COL.white, 0.12 * surf);
      }
      // each document's path across b, plus its live marker
      DOCS2.forEach((_, i) => {
        const L = Ls[i];
        const draw = seg(t, T[3] + 0.3, T[3] + 1.3);
        for (let s = 0; s < 20; s++) {
          if (s / 20 >= draw) break;
          const a = s / 20;
          const z = (s + 1) / 20;
          lines.add(V(Lx(L), Y0 + sOf(L, a) * SH, Bz(a)), V(Lx(L), Y0 + sOf(L, z) * SH, Bz(z)), cols[i], 1.4);
        }
        marks.set(i, V(Lx(L), Y0 + sOf(L, b) * SH, Bz(b)), 0.12 * surf, cols[i], 2.4);
        // link tower bar to its point on the surface
        lines.add(V(TX[i], Y0 + sOf(L, b) * SH * glow, 1.1), V(Lx(L), Y0 + sOf(L, b) * SH, Bz(b)), cols[i], 0.3 * surf);
      });
      tokens.flush();
      bars.flush();
      marks.flush();
      lines.flush();
      if (t < T[1]) return `D1: ${dls[0]} tokens · D7: ${dls[1]} tokens · average ${st.avgdl.toFixed(1)}`;
      return `b = ${fmt(b)} → D1 ${fmt(sOf(Ls[0], b))} · D7 ${fmt(sOf(Ls[1], b))}   (both mention "war" once)`;
    },
  };
}

/* ═════════════ 6 · Score & rank ═════════════ */

function buildRank(T: number[]): Built {
  const g = new THREE.Group();
  const st = corpusStats();
  const bm = new BM25(DOCS, K1, B);
  const hits = bm.search(QUERY, st.n);
  const contrib = DOCS.map(() => QTERMS.map(() => 0));
  for (const h of hits) for (const term of h.terms) contrib[h.id][QTERMS.indexOf(term.term)] = term.score;
  const total = contrib.map((c) => c.reduce((a, b) => a + b, 0));
  const order = DOCS.map((_, d) => d).sort((a, b) => total[b] - total[a] || a - b);
  const slot = DOCS.map((_, d) => order.indexOf(d));
  const X = (s: number) => -3.85 + s * 1.1;
  const TX = (k: number) => -3.3 + k * 2.4;
  const Y0 = -1.6;
  const H = 1.05;
  const segs = new Dots(st.n * QTERMS.length, new THREE.BoxGeometry(1, 1, 1));
  const lines = new Lines(60);
  const tiles = QTERMS.map((term) => new Tile([term], qColor(term)));
  const idfL = QTERMS.map((term, k) => { const s = label(`idf ${fmt(st.idf(term))}`, '#c9cde0', 0.24); s.position.set(TX(k), 2.45, 0); return s; });
  const docL = DOCS.map((_, d) => label(`D${d + 1}`, '#d6d8e2', 0.28));
  const scoreL = DOCS.map((_, d) => label(slot[d] < 3 ? `#${slot[d] + 1} · ${fmt(total[d])}` : fmt(total[d]), slot[d] < 3 ? '#ffd95e' : '#a6a8b3', slot[d] < 3 ? 0.28 : 0.22));
  g.add(segs.mesh, lines.obj, ...tiles.map((x) => x.group), ...idfL, ...docL, ...scoreL);

  return {
    group: g,
    update(t) {
      lines.reset();
      const sortF = ease(seg(t, T[2], T[2] + 1.4));
      const pick = ease(seg(t, T[3], T[3] + 0.6));
      QTERMS.forEach((term, k) => {
        const a = ease(seg(t, T[0] + k * 0.25, T[0] + k * 0.25 + 0.5));
        tiles[k].group.position.set(TX(k), 3.0 + (1 - a) * 0.8, 0);
        tiles[k].style(a, qColor(term), 1.2);
        fade(idfL[k], a);
      });
      for (let d = 0; d < st.n; d++) {
        const x = THREE.MathUtils.lerp(X(d), X(slot[d]), sortF);
        const z = Math.sin(Math.PI * sortF) * (slot[d] < d ? 0.9 : slot[d] > d ? -0.6 : 0);
        let y = Y0;
        const zero = total[d] === 0;
        QTERMS.forEach((term, k) => {
          const w0 = T[1] + k * 1.1;
          const f = ease(seg(t, w0, w0 + 0.9));
          const h = contrib[d][k] * H * f;
          tmp.copy(qColor(term));
          let gain = 1.2;
          if (pick > 0) { if (slot[d] < 3) gain += 0.9 * pick; else tmp.lerp(COL.dim, 0.55 * pick); }
          segs.set(d * QTERMS.length + k, V(x, y + h / 2, z), 1, tmp, gain);
          segs.scl[d * QTERMS.length + k].setScalar(h > 0.001 ? 1 : 0.0001).multiply(V(0.6, Math.max(h, 0.001), 0.6));
          if (contrib[d][k] > 0) {
            const beam = seg(t, w0, w0 + 0.3) * (1 - seg(t, w0 + 0.7, w0 + 1.1));
            lines.add(V(TX(k), 2.75, 0), V(x, y + h, z), qColor(term), 0.9 * beam);
          }
          y += h;
        });
        docL[d].position.set(x, Y0 - 0.3, z + 0.4);
        fade(docL[d], ease(seg(t, T[0], T[0] + 0.6)) * (zero ? 1 - 0.6 * sortF : 1));
        scoreL[d].position.set(x, y + 0.28, z);
        fade(scoreL[d], zero ? 0 : slot[d] < 3 ? pick : 0.8 * pick);
      }
      segs.flush();
      lines.flush();
      if (t < T[1]) return `Query "${QUERY}" → ${QTERMS.join(', ')}`;
      if (t < T[2]) {
        const k = Math.min(QTERMS.length - 1, Math.floor((t - T[1]) / 1.1));
        return `Adding "${QTERMS[k]}": idf ${fmt(st.idf(QTERMS[k]))} × its tf part, for each document in its posting list`;
      }
      return order.slice(0, 3).map((d, i) => `#${i + 1} D${d + 1} ${fmt(total[d])}`).join(' · ');
    },
  };
}

/* ═════════════ Chapters ═════════════ */

export const CHAPTERS: Chapter[] = [
  {
    id: 'tokenize', name: 'Tokenize', full: 'From text to terms', part: 'count',
    tagline: 'BM25 never sees sentences. It sees a bag of normalised terms and how often each one occurs.',
    steps: [
      { title: 'Raw text', detail: 'Start from one document exactly as written, with capitals and punctuation.', dur: 1.6 },
      { title: 'Lowercase and split', detail: 'Split on anything that is not a letter or digit, and lowercase every word so "War" and "war" become the same term.', dur: 1.5 },
      { title: 'Drop stopwords', detail: 'Words like "and", "the" and "of" occur everywhere and carry no meaning for search, so they are removed.', dur: 1.6 },
      { title: 'Fold plurals', detail: 'A light stemmer turns "powers" into "power", so singular and plural forms match each other.', dur: 1.4 },
      { title: 'Bag of words', detail: 'Word order is thrown away. What is left is each term and its count, the term frequency tf. Here "war" has tf = 4.', dur: 2.6 },
    ],
    camera: [0, 0.6, 11], target: [0, 0.1, 0], spin: false,
    insight: 'Queries go through exactly the same steps, so "Declare WAR!" and "war declared" end up with matching terms.',
    build: buildTokenize,
  },
  {
    id: 'index', name: 'Inverted index', full: 'The inverted index', part: 'sum',
    tagline: 'Flip documents → terms into terms → documents, so a query only touches documents that contain its words.',
    steps: [
      { title: 'Documents', detail: 'Eight short documents, D1 to D8 (see the corpus below).', dur: 1.4 },
      { title: 'Postings fly into rows', detail: 'Every (term, document) pair becomes a posting. The bar height is how many times the term occurs in that document.', dur: 2.8 },
      { title: 'Posting lists', detail: 'Each row is a posting list: one term followed by every document that contains it, with its tf. Rare terms have short rows.', dur: 1.8 },
      { title: 'Look up a query term', detail: 'To score "war" the engine jumps straight to its row. It never scans the full text.', dur: 2.2 },
      { title: 'Only matching docs are touched', detail: 'Documents missing from the posting list are skipped entirely. That is why keyword search stays fast on millions of documents.', dur: 1.8 },
    ],
    camera: [0.5, 7.4, 10.8], target: [0, -0.9, -0.9], spin: false,
    insight: 'The sum Σ in the formula runs over query terms, and each term only visits its own posting list.',
    build: buildIndex,
  },
  {
    id: 'idf', name: 'IDF', full: 'Inverse document frequency', part: 'idf',
    tagline: 'A term found in every document says nothing. A term found in one says a lot.',
    steps: [
      { title: 'Count document frequency', detail: 'df is the number of documents that contain the term. "congress" is in 5 of the 8; "treason" is in just 1.', dur: 2.2 },
      { title: 'Flip it into a weight', detail: 'IDF falls as df rises. Rare terms get tall bars and common ones get short ones. Stopwords are the extreme case and are dropped altogether.', dur: 2.2 },
      { title: 'The IDF curve', detail: 'Every term sits on the same curve: idf = ln(1 + (N − df + 0.5) / (df + 0.5)). The +1 keeps it positive even for terms in every document.', dur: 2.4 },
      { title: 'Weighting the query', detail: 'For "Congress declare war", a match on "declare" is worth over twice a match on "congress".', dur: 2.0 },
    ],
    camera: [3.6, 3.2, 13.2], target: [0, 1.2, -0.6], spin: false,
    insight: 'IDF is computed once per term from the index, then reused for every query that contains it.',
    build: buildIdf,
  },
  {
    id: 'tf', name: 'TF saturation', full: 'Term frequency saturation (k1)', part: 'tf',
    tagline: 'The second mention of a word matters; the tenth barely does. k1 decides how fast it flattens.',
    steps: [
      { title: 'Raw count grows forever', detail: 'If the score were plain tf, a document repeating "war" ten times would beat everything, which invites keyword stuffing.', dur: 2.0 },
      { title: 'BM25 saturates', detail: 'With tf · (k1 + 1) / (tf + k1), each extra occurrence adds less than the last. Watch the gains shrink.', dur: 2.8 },
      { title: 'k1 shapes the curve', detail: 'A small k1 saturates almost at once: it only asks whether the term appears. A large k1 stays closer to the raw count. The usual range is 1.2–2.0.', dur: 2.6 },
      { title: 'Ceiling = k1 + 1', detail: 'However often a term repeats, its contribution never passes k1 + 1 (times its IDF).', dur: 1.8 },
    ],
    camera: [6.2, 3.0, 13.4], target: [0, 0.7, -0.3], spin: false,
    insight: 'This chart assumes an average-length document. The next chapter shows how length bends these curves.',
    build: buildTf,
  },
  {
    id: 'length', name: 'Length norm', full: 'Document length normalisation (b)', part: 'len',
    tagline: 'One mention in a short document is stronger evidence than one mention in a long one.',
    steps: [
      { title: 'Short vs long document', detail: 'D1 has 7 tokens and D7 has 28. Each contains "war" exactly once (the glowing block). The plane marks the average length.', dur: 2.0 },
      { title: 'Same tf, same score?', detail: 'With b = 0, length is ignored and both documents score the same for "war".', dur: 1.4 },
      { title: 'Turn b up to 1', detail: 'b scales by |D| / avgdl. Longer-than-average documents are penalised and shorter ones are boosted.', dur: 3.0 },
      { title: 'The full surface', detail: 'Score over every length and every b. The coloured paths are D1 and D7; they settle at the default b = 0.75.', dur: 2.8 },
    ],
    camera: [3.4, 3.0, 11], target: [-0.4, 0, -0.2], spin: false,
    insight: 'Set b lower for collections where length reflects real content (books). Set it higher where long means padded.',
    build: buildLength,
  },
  {
    id: 'rank', name: 'Score & rank', full: 'Putting it together', part: 'all',
    tagline: 'Each query term adds IDF × saturated, length-normalised tf. Sort the totals and you have a ranking.',
    steps: [
      { title: 'Query terms', detail: '"Congress declare war" becomes three terms, each with its own IDF weight.', dur: 1.6 },
      { title: 'Add each term\'s score', detail: 'Each term adds its contribution to every document in its posting list. The bars stack term by term.', dur: 3.4 },
      { title: 'Sort by total', detail: 'Documents move into rank order. D2 and D3 share no query term, so they score 0.', dur: 2.0 },
      { title: 'Return top-k', detail: 'D1 wins: short and it has all three terms. D6 repeats "war" four times but saturation caps it. D7 has every term but is long.', dur: 2.0 },
    ],
    camera: [0, 2.0, 11.8], target: [0, 0.5, 0], spin: false,
    insight: 'BM25 only rewards exact term matches. It misses synonyms that dense vectors catch, which is why RAG Lens fuses both.',
    build: buildRank,
  },
];
