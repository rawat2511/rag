import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import type { Built, SceneDef } from './algorithms';
import { disposeTree } from './kit';

export interface SceneApi {
  seek(step: number): void;
}
export interface FrameInfo { t: number; starts: number[]; stat: string }

interface Props {
  algo: SceneDef;
  playing: boolean;
  apiRef: React.RefObject<SceneApi | null>;
  onFrame: (info: FrameInfo) => void;
  onLoop: () => void;
}

const BG = '#07080c';
const HOLD = 1.6; // seconds to linger on the final state before looping

export function IndexScene({ algo, playing, apiRef, onFrame, onLoop }: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const live = useRef({ algo, playing, onFrame, onLoop });
  live.current = { algo, playing, onFrame, onLoop };
  const stage = useRef<{ load(a: SceneDef): void } | null>(null);
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
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    host.appendChild(renderer.domElement);
    const el = renderer.domElement;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(BG);
    scene.fog = new THREE.Fog(BG, 14, 30);
    const camera = new THREE.PerspectiveCamera(42, 2, 0.1, 100);
    scene.add(new THREE.HemisphereLight('#dfe6ff', '#1a1626', 1.1));
    const sun = new THREE.DirectionalLight('#ffffff', 1.5);
    sun.position.set(4, 7, 9);
    scene.add(sun);

    const controls = new OrbitControls(camera, el);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.enableZoom = false; // keep the wheel for page scrolling
    controls.autoRotateSpeed = 0.5;

    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.5, 0.6, 0.82);
    composer.addPass(bloom);
    composer.addPass(new OutputPass());

    // Faint floor grid for spatial reference.
    const grid = new THREE.GridHelper(28, 28, '#1d2233', '#12151f');
    grid.position.y = -3.6;
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.5;
    scene.add(grid);

    let built: Built | null = null;
    let starts: number[] = [0];
    let t = 0;
    let camFrom = camera.position.clone();
    let camTo = camera.position.clone();
    let tgtFrom = new THREE.Vector3();
    let tgtTo = new THREE.Vector3();
    let fly = 1;

    let loaded: SceneDef | null = null;
    function load(a: SceneDef) {
      if (a === loaded) return;
      loaded = a;
      if (built) {
        scene.remove(built.group);
        disposeTree(built.group);
      }
      starts = [0];
      for (const s of a.steps) starts.push(starts[starts.length - 1] + s.dur);
      built = a.build(starts);
      scene.add(built.group);
      t = 0;
      camFrom = camera.position.clone();
      tgtFrom = controls.target.clone();
      camTo = new THREE.Vector3(...a.camera);
      tgtTo = new THREE.Vector3(...(a.target ?? [0, 0, 0]));
      fly = first ? 1 : 0;
      if (first) { camera.position.copy(camTo); controls.target.copy(tgtTo); }
      first = false;
    }
    let first = true;
    stage.current = { load };
    load(live.current.algo);
    apiRef.current = {
      seek(step: number) {
        t = starts[Math.max(0, Math.min(step, starts.length - 1))] + 0.001;
      },
    };

    let pointerInside = false;
    const onEnter = () => (pointerInside = true);
    const onLeave = () => (pointerInside = false);
    el.addEventListener('pointerenter', onEnter);
    el.addEventListener('pointerleave', onLeave);

    const ro = new ResizeObserver(() => {
      const w = host.clientWidth;
      const h = host.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h);
      composer.setSize(w, h);
      bloom.resolution.set(w / 2, h / 2);
      camera.aspect = w / h;
      // Pull back on narrow screens so the whole structure stays in frame.
      camera.fov = w / h < 1 ? 58 : w / h < 1.4 ? 50 : 42;
      camera.updateProjectionMatrix();
    });
    ro.observe(host);
    let visible = true;
    const io = new IntersectionObserver(([e]) => (visible = e.isIntersecting));
    io.observe(host);

    const clock = new THREE.Clock();
    let raf = 0;
    const frame = () => {
      raf = requestAnimationFrame(frame);
      const dt = Math.min(clock.getDelta(), 0.05);
      if (!visible || !built) return;
      const { playing: isPlaying, algo: a } = live.current;
      const end = starts[starts.length - 1];
      if (isPlaying) {
        t += dt;
        if (t > end + HOLD) { t = 0; live.current.onLoop(); }
      }
      if (fly < 1) {
        fly = Math.min(1, fly + dt / 1.1);
        const f = 1 - Math.pow(1 - fly, 3);
        camera.position.lerpVectors(camFrom, camTo, f);
        controls.target.lerpVectors(tgtFrom, tgtTo, f);
      }
      controls.autoRotate = !reducedMotion && !!a.spin && !pointerInside && fly >= 1;
      controls.update();
      const now = performance.now() / 1000;
      const stat = built.update(Math.min(t, end), { camera, now });
      live.current.onFrame({ t: Math.min(t, end), starts, stat });
      composer.render();
    };
    frame();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      el.removeEventListener('pointerenter', onEnter);
      el.removeEventListener('pointerleave', onLeave);
      if (built) disposeTree(built.group);
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      controls.dispose();
      composer.dispose();
      renderer.dispose();
      el.remove();
      stage.current = null;
      apiRef.current = null;
    };
  }, [apiRef]);

  useEffect(() => {
    stage.current?.load(algo);
  }, [algo]);

  return (
    <div ref={hostRef} className="ix-scene">
      {failed && <p className="scene-fallback">3D view unavailable (WebGL is disabled). The explanations below still apply.</p>}
    </div>
  );
}
