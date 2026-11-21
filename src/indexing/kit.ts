import * as THREE from 'three';

/* Small immediate-mode toolkit: every scene recomputes its whole state from the timeline
   position `t` each frame, so any step can be jumped to (scrubbing) without replaying. */

export const COL = {
  idle: new THREE.Color('#39405a'),
  dim: new THREE.Color('#1b1f2c'),
  dense: new THREE.Color('#58a8ff'),
  sparse: new THREE.Color('#ffab3d'),
  fuse: new THREE.Color('#b98cff'),
  rerank: new THREE.Color('#4fd6a8'),
  final: new THREE.Color('#ffd95e'),
  danger: new THREE.Color('#ff6b6b'),
  white: new THREE.Color('#ffffff'),
};
export const PALETTE = ['#5fd8e8', '#58a8ff', '#8f7bff', '#f277b6', '#ffab3d', '#4fd6a8', '#ffd95e', '#ff8a80'].map((c) => new THREE.Color(c));

export const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
export const ease = (x: number) => { x = clamp01(x); return x * x * (3 - 2 * x); };
/** 0→1 progress of t through [a, b]. */
export const seg = (t: number, a: number, b: number) => clamp01((t - a) / (b - a));

export function rng(seed: number) {
  let s = seed % 2147483647 || 1;
  return () => (s = (s * 16807) % 2147483647) / 2147483647;
}
export const gauss = (r: () => number) => (r() + r() + r() + r() - 2) * 0.87;

export function ramp(t: number, out = new THREE.Color()) {
  return out.setHSL((188 + 150 * t) / 360, 0.78, 0.6);
}

export function clusters(n: number, k: number, seed: number, half: [number, number, number], sigma = 0.62) {
  const r = rng(seed);
  const centers = Array.from({ length: k }, () => new THREE.Vector3((r() * 2 - 1) * half[0] * 0.8, (r() * 2 - 1) * half[1] * 0.8, (r() * 2 - 1) * half[2] * 0.8));
  const pts: THREE.Vector3[] = [];
  const label: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = Math.floor(r() * k);
    const p = centers[c].clone().add(new THREE.Vector3(gauss(r) * sigma * 1.4, gauss(r) * sigma, gauss(r) * sigma * 1.2));
    p.set(THREE.MathUtils.clamp(p.x, -half[0], half[0]), THREE.MathUtils.clamp(p.y, -half[1], half[1]), THREE.MathUtils.clamp(p.z, -half[2], half[2]));
    pts.push(p);
    label.push(c);
  }
  return { pts, label, centers };
}

/** Indices of `pts` (optionally restricted to `among`) sorted by distance to q. */
export function nearest(pts: THREE.Vector3[], q: THREE.Vector3, k: number, among?: number[]) {
  const ids = among ? [...among] : pts.map((_, i) => i);
  ids.sort((a, b) => pts[a].distanceToSquared(q) - pts[b].distanceToSquared(q));
  return ids.slice(0, k);
}

/** Plain k-means that records centroid positions after each iteration. */
export function kmeans(pts: THREE.Vector3[], k: number, seed: number, iters = 7) {
  const r = rng(seed);
  let cent = Array.from({ length: k }, () => pts[Math.floor(r() * pts.length)].clone());
  const history = [cent.map((c) => c.clone())];
  let assign = new Array(pts.length).fill(0);
  for (let it = 0; it < iters; it++) {
    assign = pts.map((p) => argmin(cent, p));
    const sum = cent.map(() => new THREE.Vector3());
    const cnt = new Array(k).fill(0);
    pts.forEach((p, i) => { sum[assign[i]].add(p); cnt[assign[i]]++; });
    cent = cent.map((c, j) => (cnt[j] ? sum[j].divideScalar(cnt[j]) : c.clone()));
    history.push(cent.map((c) => c.clone()));
  }
  assign = pts.map((p) => argmin(cent, p));
  return { centroids: cent, history, assign };
}

export function argmin(cands: THREE.Vector3[], p: THREE.Vector3) {
  let best = 0;
  let bd = Infinity;
  cands.forEach((c, j) => { const d = c.distanceToSquared(p); if (d < bd) { bd = d; best = j; } });
  return best;
}

/* ───────────── Renderables ───────────── */

const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _c = new THREE.Color();

/** Instanced glyphs (spheres, boxes, octahedra) with per-frame position / scale / colour. */
export class Dots {
  mesh: THREE.InstancedMesh;
  pos: THREE.Vector3[];
  scl: THREE.Vector3[];
  col: THREE.Color[];
  constructor(n: number, geo: THREE.BufferGeometry = new THREE.IcosahedronGeometry(1, 1), mat?: THREE.Material) {
    this.mesh = new THREE.InstancedMesh(geo, mat ?? new THREE.MeshStandardMaterial({ roughness: 0.38, metalness: 0.08 }), n);
    this.mesh.frustumCulled = false;
    this.pos = Array.from({ length: n }, () => new THREE.Vector3());
    this.scl = Array.from({ length: n }, () => new THREE.Vector3());
    this.col = Array.from({ length: n }, () => new THREE.Color());
    for (let i = 0; i < n; i++) this.mesh.setColorAt(i, _c.set(0));
  }
  set(i: number, p: THREE.Vector3, s: number, c: THREE.Color, gain = 1) {
    this.pos[i].copy(p);
    this.scl[i].setScalar(Math.max(s, 1e-4));
    this.col[i].copy(c).multiplyScalar(gain);
  }
  flush() {
    for (let i = 0; i < this.pos.length; i++) {
      _m.compose(this.pos[i], _q, this.scl[i]);
      this.mesh.setMatrixAt(i, _m);
      this.mesh.setColorAt(i, this.col[i]);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }
}

/** A pool of line segments rebuilt every frame. Additive, so darker colour reads as fainter. */
export class Lines {
  obj: THREE.LineSegments;
  private p: Float32Array;
  private c: Float32Array;
  private n = 0;
  constructor(private max: number, opacity = 1) {
    this.p = new Float32Array(max * 6);
    this.c = new Float32Array(max * 6);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(this.p, 3));
    g.setAttribute('color', new THREE.BufferAttribute(this.c, 3));
    this.obj = new THREE.LineSegments(g, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false,
    }));
    this.obj.frustumCulled = false;
  }
  reset() { this.n = 0; }
  add(a: THREE.Vector3, b: THREE.Vector3, c: THREE.Color, k = 1, cb?: THREE.Color, kb = k) {
    if (this.n >= this.max || k <= 0.002) return;
    const o = this.n * 6;
    this.p[o] = a.x; this.p[o + 1] = a.y; this.p[o + 2] = a.z;
    this.p[o + 3] = b.x; this.p[o + 4] = b.y; this.p[o + 5] = b.z;
    const e = cb ?? c;
    this.c[o] = c.r * k; this.c[o + 1] = c.g * k; this.c[o + 2] = c.b * k;
    this.c[o + 3] = e.r * kb; this.c[o + 4] = e.g * kb; this.c[o + 5] = e.b * kb;
    this.n++;
  }
  /** Segment from a toward b, drawn only up to fraction f. */
  grow(a: THREE.Vector3, b: THREE.Vector3, f: number, c: THREE.Color, k = 1) {
    if (f <= 0) return;
    this.add(a, f >= 1 ? b : a.clone().lerp(b, f), c, k);
  }
  flush() {
    const g = this.obj.geometry;
    g.setDrawRange(0, this.n * 2);
    g.attributes.position.needsUpdate = true;
    g.attributes.color.needsUpdate = true;
  }
}

/** The glowing white query marker with a pulsing halo that always faces the camera. */
export class QueryMarker {
  group = new THREE.Group();
  private halo: THREE.Mesh;
  constructor(size = 0.13) {
    const core = new THREE.Mesh(new THREE.IcosahedronGeometry(size, 2), new THREE.MeshBasicMaterial({ color: new THREE.Color(1.8, 1.8, 1.8), toneMapped: false }));
    this.halo = new THREE.Mesh(
      new THREE.RingGeometry(size * 1.8, size * 1.95, 48),
      new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, side: THREE.DoubleSide, depthWrite: false }),
    );
    this.group.add(core, this.halo);
    this.group.visible = false;
  }
  update(p: THREE.Vector3, show: number, camera: THREE.Camera, now: number) {
    this.group.visible = show > 0.01;
    this.group.position.copy(p);
    this.group.scale.setScalar(Math.max(show, 1e-3));
    this.halo.quaternion.copy(camera.quaternion);
    const pulse = 1 + ((now * 0.8) % 1) * 1.4;
    this.halo.scale.setScalar(pulse);
    (this.halo.material as THREE.MeshBasicMaterial).opacity = 0.75 * (1 - (pulse - 1) / 1.4);
  }
}

/** Camera-facing text label. */
export function label(text: string, color = '#d6d8e2', height = 0.34) {
  const font = '500 64px -apple-system, system-ui, "Segoe UI", Roboto, sans-serif';
  const cv = document.createElement('canvas');
  const ctx = cv.getContext('2d')!;
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + 24;
  cv.width = w;
  cv.height = 88;
  ctx.font = font;
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 12, 46);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false, toneMapped: false }));
  sprite.scale.set((height * w) / 88, height, 1);
  return sprite;
}
export function fade(obj: THREE.Object3D & { material: THREE.Material | THREE.Material[] }, o: number, max = 1) {
  obj.visible = o > 0.01;
  const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
  for (const m of mats) m.opacity = o * max;
}

/** Translucent flat surface with a crisp outline, built from a convex polygon. */
export function panel(poly: THREE.Vector3[], color: THREE.Color, fill = 0.1) {
  const g = new THREE.Group();
  const verts: number[] = [];
  for (let i = 1; i < poly.length - 1; i++) verts.push(...poly[0].toArray(), ...poly[i].toArray(), ...poly[i + 1].toArray());
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  const face = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: fill, side: THREE.DoubleSide, depthWrite: false, toneMapped: false }));
  const edge = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(poly), new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7, toneMapped: false }));
  g.add(face, edge);
  return { group: g, face, edge, fill };
}
export function setPanel(p: ReturnType<typeof panel>, o: number, boost = 1) {
  p.group.visible = o > 0.01;
  (p.face.material as THREE.MeshBasicMaterial).opacity = p.fill * o * boost;
  (p.edge.material as THREE.LineBasicMaterial).opacity = 0.7 * o * Math.min(boost, 1.4);
}
export function rect(center: THREE.Vector3, u: THREE.Vector3, v: THREE.Vector3, hu: number, hv: number) {
  return [
    center.clone().addScaledVector(u, -hu).addScaledVector(v, -hv),
    center.clone().addScaledVector(u, hu).addScaledVector(v, -hv),
    center.clone().addScaledVector(u, hu).addScaledVector(v, hv),
    center.clone().addScaledVector(u, -hu).addScaledVector(v, hv),
  ];
}

/** Sutherland–Hodgman: keep the part of a convex polygon where n·p − d ≥ 0. */
export function clip(poly: THREE.Vector3[], n: THREE.Vector3, d: number) {
  const out: THREE.Vector3[] = [];
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    const da = n.dot(a) - d;
    const db = n.dot(b) - d;
    if (da >= 0) out.push(a);
    if (da >= 0 !== db >= 0) out.push(a.clone().lerp(b, da / (da - db)));
  }
  return out;
}

export function disposeTree(root: THREE.Object3D) {
  root.traverse((o) => {
    const m = o as THREE.Mesh;
    m.geometry?.dispose();
    const mats = m.material ? (Array.isArray(m.material) ? m.material : [m.material]) : [];
    for (const mat of mats) {
      (mat as THREE.MeshBasicMaterial).map?.dispose();
      mat.dispose();
    }
  });
}
