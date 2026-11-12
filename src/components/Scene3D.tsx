import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

export type SceneView = 'doc' | 'space';

export interface SceneProps {
  /** Number of chunks. 0 shows the idle scene. */
  n: number;
  revealed: number;
  embedded: number;
  embedding: boolean;
  /** PCA coordinates, 3 per chunk. Null until every chunk is embedded. */
  coords: Float32Array | null;
  view: SceneView;
  query: [number, number, number] | null;
  denseIds: number[];
  sparseIds: number[];
  finalIds: number[];
  hover: number | null;
  selected: number | null;
  onHover: (id: number | null) => void;
  onSelect: (id: number) => void;
}

const BG = '#07080c';
const C = {
  idle: new THREE.Color('#232737'),
  dense: new THREE.Color('#58a8ff'),
  sparse: new THREE.Color('#ffab3d'),
  both: new THREE.Color('#b98cff'),
  final: new THREE.Color('#ffd95e'),
  white: new THREE.Color('#ffffff'),
};

/** Document order mapped to a cyan → violet → rose ramp. */
function rampColor(t: number, out: THREE.Color) {
  return out.setHSL((188 + 150 * t) / 360, 0.78, 0.6);
}

/**
 * The 3D stage. Every chunk is one block.
 *  - "doc" view: blocks form a wall in reading order and light up as they are embedded.
 *  - "space" view: blocks fly to their position in embedding space (PCA to 3D), where the
 *    query and its retrieved neighbours are highlighted.
 */
export function Scene3D(props: SceneProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const latest = useRef(props);
  latest.current = props;
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const host = hostRef.current!;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    } catch {
      setFailed(true);
      return;
    }
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(BG);
    scene.fog = new THREE.Fog(BG, 16, 34);
    const camera = new THREE.PerspectiveCamera(40, 2, 0.1, 100);
    const DOC_POSE = new THREE.Vector3(-4.2, 2.2, 16.5);
    camera.position.copy(DOC_POSE);

    scene.add(new THREE.HemisphereLight('#dfe6ff', '#1a1626', 1.05));
    const sun = new THREE.DirectionalLight('#ffffff', 1.5);
    sun.position.set(4, 7, 9);
    scene.add(sun);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enableZoom = false;
    controls.enablePan = false;
    controls.autoRotateSpeed = 0.55;
    controls.minPolarAngle = Math.PI * 0.2;
    controls.maxPolarAngle = Math.PI * 0.8;

    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.42, 0.65, 0.85);
    composer.addPass(bloom);
    composer.addPass(new OutputPass());

    /* Ambient dust: gives depth, and is the whole scene before a document is loaded. */
    const DUST = 900;
    const dustPos = new Float32Array(DUST * 3);
    const dustCol = new Float32Array(DUST * 3);
    const tmp = new THREE.Color();
    for (let i = 0; i < DUST; i++) {
      const r = 5 + Math.random() * 9;
      const th = Math.random() * Math.PI * 2;
      const ph = Math.acos(2 * Math.random() - 1);
      dustPos[i * 3] = r * Math.sin(ph) * Math.cos(th) * 1.5;
      dustPos[i * 3 + 1] = r * Math.cos(ph) * 0.7;
      dustPos[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th);
      rampColor(Math.random(), tmp);
      dustCol.set([tmp.r, tmp.g, tmp.b], i * 3);
    }
    const dustGeo = new THREE.BufferGeometry();
    dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
    dustGeo.setAttribute('color', new THREE.BufferAttribute(dustCol, 3));
    const dustMat = new THREE.PointsMaterial({ size: 0.07, vertexColors: true, transparent: true, opacity: 0.8, depthWrite: false });
    const dust = new THREE.Points(dustGeo, dustMat);
    scene.add(dust);

    /* Idle demo: a slowly turning cloud of blocks hinting at what an embedded document looks like. */
    const DEMO = 340;
    const demoGeo = new THREE.BoxGeometry(1, 1, 1);
    const demoMat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.05 });
    const demo = new THREE.InstancedMesh(demoGeo, demoMat, DEMO);
    const demoGroup = new THREE.Group();
    demoGroup.add(demo);
    scene.add(demoGroup);
    {
      let seed = 7;
      const rnd = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
      const gauss = () => (rnd() + rnd() + rnd() + rnd() - 2) * 0.9;
      const centers = Array.from({ length: 7 }, () => [(rnd() - 0.5) * 7, (rnd() - 0.5) * 4.4, (rnd() - 0.5) * 6]);
      const m = new THREE.Matrix4();
      for (let i = 0; i < DEMO; i++) {
        const c = Math.floor(rnd() * centers.length);
        const size = 0.09 + rnd() * 0.09;
        m.makeScale(size, size, size);
        m.setPosition(centers[c][0] + gauss() * 1.1, centers[c][1] + gauss() * 0.9, centers[c][2] + gauss() * 1.1);
        demo.setMatrixAt(i, m);
        demo.setColorAt(i, rampColor(Math.min(1, Math.max(0, c / (centers.length - 1) + (rnd() - 0.5) * 0.12)), tmp).multiplyScalar(rnd() < 0.08 ? 2.2 : 0.95));
      }
    }
    let demoOffset = 5.4;

    /* Query marker and the beams from it to retrieved chunks. */
    const queryMesh = new THREE.Mesh(
      new THREE.IcosahedronGeometry(0.2, 2),
      new THREE.MeshBasicMaterial({ color: new THREE.Color(3, 3, 3), toneMapped: false }),
    );
    const halo = new THREE.Mesh(
      new THREE.RingGeometry(0.34, 0.37, 48),
      new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.6, side: THREE.DoubleSide, depthWrite: false }),
    );
    queryMesh.visible = halo.visible = false;
    // Everything document-related sits slightly low so it clears the title and stats overlay.
    const world = new THREE.Group();
    world.position.set(0.4, -1.15, 0);
    scene.add(world);
    world.add(queryMesh, halo);
    const MAX_BEAMS = 96;
    const beamPos = new Float32Array(MAX_BEAMS * 6);
    const beamCol = new Float32Array(MAX_BEAMS * 6);
    const beamGeo = new THREE.BufferGeometry();
    beamGeo.setAttribute('position', new THREE.BufferAttribute(beamPos, 3));
    beamGeo.setAttribute('color', new THREE.BufferAttribute(beamCol, 3));
    const beams = new THREE.LineSegments(
      beamGeo,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, toneMapped: false, depthWrite: false }),
    );
    beams.frustumCulled = false;
    world.add(beams);

    /* Chunk blocks. */
    const blockGeo = new THREE.BoxGeometry(1, 1, 1);
    const blockMat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.05 });
    let mesh: THREE.InstancedMesh | null = null;
    let n = 0;
    let cur = new Float32Array(0); // per block: x y z  sx sy sz  r g b
    let wall = new Float32Array(0);
    let cloud = new Float32Array(0);
    let inCloud = new Uint8Array(0);
    let tile = 1;
    let cube = 0.2;
    let center = [0, 0, 0];
    let scale = [1, 1, 1];
    let coordsSeen: Float32Array | null = null;
    let viewSeen: SceneView = 'doc';
    let viewChangedAt = -100;
    let idsSeen: unknown[] = [];
    const role = new Map<number, number>(); // 1 dense · 2 sparse · 3 both · 4 final

    function rebuild(count: number) {
      if (mesh) {
        world.remove(mesh);
        mesh.dispose();
        mesh = null;
      }
      n = count;
      coordsSeen = null;
      if (!n) return;
      const cols = Math.ceil(Math.sqrt(n * 2.6));
      const rows = Math.ceil(n / cols);
      tile = Math.min(13.5 / cols, 5.2 / rows, 0.9);
      cube = Math.max(0.075, Math.min(0.3, 0.95 / Math.cbrt(n)));
      wall = new Float32Array(n * 3);
      cloud = new Float32Array(n * 3);
      inCloud = new Uint8Array(n);
      cur = new Float32Array(n * 9);
      for (let i = 0; i < n; i++) {
        wall[i * 3] = ((i % cols) - (cols - 1) / 2) * tile;
        wall[i * 3 + 1] = ((rows - 1) / 2 - Math.floor(i / cols)) * tile;
        cur[i * 9] = wall[i * 3];
        cur[i * 9 + 1] = wall[i * 3 + 1];
        cur[i * 9 + 2] = -3; // blocks drop in from behind
      }
      mesh = new THREE.InstancedMesh(blockGeo, blockMat, n);
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
      mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false;
      mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 200);
      mesh.instanceMatrix.array.fill(0);
      world.add(mesh);
    }

    function fitCloud(coords: Float32Array) {
      const min = [Infinity, Infinity, Infinity];
      const max = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < n; i++)
        for (let a = 0; a < 3; a++) {
          min[a] = Math.min(min[a], coords[i * 3 + a]);
          max[a] = Math.max(max[a], coords[i * 3 + a]);
        }
      const half = [5.8, 2.9, 3.4];
      center = min.map((m, a) => (m + max[a]) / 2);
      scale = min.map((m, a) => half[a] / Math.max(1e-6, (max[a] - m) / 2));
      for (let i = 0; i < n; i++)
        for (let a = 0; a < 3; a++) cloud[i * 3 + a] = (coords[i * 3 + a] - center[a]) * scale[a];
    }

    /* Pointer: hover via raycast, click selects (unless it was a drag). */
    const raycaster = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let pointerInside = false;
    let downAt: [number, number] | null = null;
    const pick = (e: PointerEvent): number | null => {
      if (!mesh) return null;
      const rect = renderer.domElement.getBoundingClientRect();
      ndc.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      const hit = raycaster.intersectObject(mesh)[0];
      const id = hit?.instanceId;
      return id !== undefined && id < latest.current.revealed ? id : null;
    };
    const el = renderer.domElement;
    const onMove = (e: PointerEvent) => {
      pointerInside = true;
      if (e.buttons) return;
      const id = pick(e);
      el.style.cursor = id === null ? 'grab' : 'pointer';
      if (id !== latest.current.hover) latest.current.onHover(id);
    };
    const onLeave = () => {
      pointerInside = false;
      if (latest.current.hover !== null) latest.current.onHover(null);
    };
    const onDown = (e: PointerEvent) => (downAt = [e.clientX, e.clientY]);
    const onUp = (e: PointerEvent) => {
      if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
      const id = pick(e);
      if (id !== null) latest.current.onSelect(id);
    };
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerleave', onLeave);
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointerup', onUp);

    const resize = () => {
      const w = host.clientWidth || 1;
      const h = host.clientHeight || 1;
      renderer.setSize(w, h);
      composer.setSize(w, h);
      const aspect = w / h;
      camera.aspect = aspect;
      camera.fov = Math.min(78, aspect < 1.5 ? (40 * 1.5) / aspect : 40);
      camera.updateProjectionMatrix();
      demoOffset = aspect > 1.3 ? 5.4 : 0;
    };
    const ro = new ResizeObserver(resize);
    ro.observe(host);
    resize();

    const target = new THREE.Color();
    const qv = new THREE.Vector3();
    let last = performance.now() / 1000;
    let raf = 0;
    let tick = 0;
    let onScreen = true;
    const io = new IntersectionObserver(([entry]) => (onScreen = entry.isIntersecting));
    io.observe(host);

    const frame = () => {
      raf = requestAnimationFrame(frame);
      const p = latest.current;
      // Don't draw when nobody can see it, and draw at a third of the rate while the
      // embedding model is running so the scene doesn't compete with it for the CPU/GPU.
      if (!onScreen || document.hidden) return;
      if (p.embedding && tick++ % 3 !== 0) return;
      const now = performance.now() / 1000;
      const dt = Math.min(0.1, now - last);
      last = now;

      if (p.n !== n) rebuild(p.n);
      if (p.coords !== coordsSeen) {
        coordsSeen = p.coords;
        if (p.coords && n) fitCloud(p.coords);
      }
      if (p.view !== viewSeen) {
        viewSeen = p.view;
        viewChangedAt = now;
      }
      if (idsSeen[0] !== p.denseIds || idsSeen[1] !== p.sparseIds || idsSeen[2] !== p.finalIds) {
        idsSeen = [p.denseIds, p.sparseIds, p.finalIds];
        role.clear();
        for (const id of p.denseIds) role.set(id, 1);
        for (const id of p.sparseIds) role.set(id, role.has(id) ? 3 : 2);
        for (const id of p.finalIds) role.set(id, 4);
      }

      const space = p.view === 'space' && !!coordsSeen;
      const since = now - viewChangedAt;
      const hasQuery = role.size > 0;
      const k = reducedMotion ? 1 : 1 - Math.exp(-dt * 5.5);
      const kFast = reducedMotion ? 1 : 1 - Math.exp(-dt * 12);

      if (mesh) {
        const M = mesh.instanceMatrix.array as Float32Array;
        const K = mesh.instanceColor!.array as Float32Array;
        for (let i = 0; i < n; i++) {
          if (since > (i / n) * 0.9) inCloud[i] = space ? 1 : 0;
          const there = inCloud[i] === 1;
          const src = there ? cloud : wall;
          const visible = i < p.revealed;
          const isEmbedded = i < p.embedded;
          const r = role.get(i);
          const active = i === p.hover || i === p.selected;
          const front = p.embedding && isEmbedded && i >= p.embedded - 8;

          let grow = 1;
          if (!isEmbedded) target.copy(C.idle);
          else {
            rampColor(n > 1 ? i / (n - 1) : 0, target);
            if (hasQuery && !r) target.multiplyScalar(there ? 0.16 : 0.3);
          }
          if (r === 4) { target.copy(C.final).multiplyScalar(2.4); grow = there ? 2.3 : 1; }
          else if (r) { target.copy(r === 1 ? C.dense : r === 2 ? C.sparse : C.both).multiplyScalar(1.5); grow = there ? 1.6 : 1; }
          if (front) { target.lerp(C.white, 0.35).multiplyScalar(1.25); grow = 1.12; }
          if (active) { target.lerp(C.white, 0.7).multiplyScalar(2.2); grow = Math.max(grow, there ? 2.2 : 1.2); }

          const o = i * 9;
          const lift = !there && (r === 4 || active) ? 0.35 : front ? 0.25 : 0;
          cur[o] += (src[i * 3] - cur[o]) * k;
          cur[o + 1] += (src[i * 3 + 1] - cur[o + 1]) * k;
          cur[o + 2] += (src[i * 3 + 2] + lift - cur[o + 2]) * k;
          const sx = visible ? (there ? cube : tile * 0.86) * grow : 0;
          const sz = visible ? (there ? cube * grow : tile * 0.4) : 0;
          cur[o + 3] += (sx - cur[o + 3]) * kFast;
          cur[o + 4] += (sx - cur[o + 4]) * kFast;
          cur[o + 5] += (sz - cur[o + 5]) * kFast;
          cur[o + 6] += (target.r - cur[o + 6]) * kFast;
          cur[o + 7] += (target.g - cur[o + 7]) * kFast;
          cur[o + 8] += (target.b - cur[o + 8]) * kFast;

          const m = i * 16;
          M[m] = cur[o + 3]; M[m + 5] = cur[o + 4]; M[m + 10] = cur[o + 5];
          M[m + 12] = cur[o]; M[m + 13] = cur[o + 1]; M[m + 14] = cur[o + 2]; M[m + 15] = 1;
          K[i * 3] = cur[o + 6]; K[i * 3 + 1] = cur[o + 7]; K[i * 3 + 2] = cur[o + 8];
        }
        mesh.instanceMatrix.needsUpdate = true;
        mesh.instanceColor!.needsUpdate = true;
      }

      /* Query marker + beams (embedding space only). */
      let beamCount = 0;
      const showQuery = space && !!p.query && since > 0.9;
      queryMesh.visible = halo.visible = showQuery;
      if (showQuery && p.query) {
        qv.set(
          (p.query[0] - center[0]) * scale[0],
          (p.query[1] - center[1]) * scale[1],
          (p.query[2] - center[2]) * scale[2],
        );
        qv.clampLength(0, 8);
        queryMesh.position.copy(qv);
        halo.position.copy(qv);
        halo.quaternion.copy(camera.quaternion);
        const pulse = 1 + ((now * 0.8) % 1) * 1.6;
        halo.scale.setScalar(pulse);
        (halo.material as THREE.MeshBasicMaterial).opacity = 0.7 * (1 - (pulse - 1) / 1.6);
        for (const [id, r] of role) {
          if (beamCount >= MAX_BEAMS || id >= n || r === 2) continue;
          const c = r === 4 ? C.final : r === 3 ? C.both : C.dense;
          const gain = r === 4 ? 1.6 : 0.35;
          const b = beamCount * 6;
          beamPos.set([qv.x, qv.y, qv.z, cur[id * 9], cur[id * 9 + 1], cur[id * 9 + 2]], b);
          beamCol.set([c.r * gain, c.g * gain, c.b * gain, c.r * gain, c.g * gain, c.b * gain], b);
          beamCount++;
        }
        beamGeo.attributes.position.needsUpdate = true;
        beamGeo.attributes.color.needsUpdate = true;
      }
      beamGeo.setDrawRange(0, beamCount * 2);

      /* Camera: face the wall in document view, drift around the cloud in space view. */
      if (!space && since < 1.8 && !reducedMotion) camera.position.lerp(DOC_POSE, k * 0.8);
      controls.autoRotate = !reducedMotion && space && !pointerInside;
      const demoScale = demoGroup.scale.x + ((n === 0 ? 1 : 0) - demoGroup.scale.x) * k;
      demoGroup.scale.setScalar(demoScale);
      demoGroup.visible = demoScale > 0.01;
      demoGroup.position.x += (demoOffset - demoGroup.position.x) * k;
      if (!reducedMotion) demoGroup.rotation.y += dt * 0.16;
      controls.update();
      dust.rotation.y += dt * 0.012;
      dustMat.opacity += ((n === 0 ? 0.6 : 0.3) - dustMat.opacity) * k;

      composer.render();
    };
    frame();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerleave', onLeave);
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('pointerup', onUp);
      controls.dispose();
      mesh?.dispose();
      blockGeo.dispose();
      blockMat.dispose();
      dustGeo.dispose();
      demo.dispose();
      demoGeo.dispose();
      demoMat.dispose();
      beamGeo.dispose();
      composer.dispose();
      renderer.dispose();
      el.remove();
    };
  }, []);

  return (
    <div ref={hostRef} className="scene">
      {failed && <p className="scene-fallback">3D view unavailable (WebGL is disabled). The pipeline below still works.</p>}
    </div>
  );
}
