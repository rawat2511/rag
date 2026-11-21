import { useCallback, useEffect, useRef, useState } from 'react';
import { ALGOS, type AlgoDef, type Rating } from './algorithms';
import { IndexScene, type FrameInfo, type SceneApi } from './IndexScene';
import './indexing.css';

const RATING_LABELS: [keyof Rating, string][] = [
  ['speed', 'Query speed'],
  ['recall', 'Recall'],
  ['memory', 'Memory efficiency'],
  ['build', 'Build speed'],
];

function Pips({ n, tone }: { n: number; tone?: string }) {
  return (
    <span className="ix-pips" aria-label={`${n} out of 5`} style={tone ? ({ '--tone': tone } as React.CSSProperties) : undefined}>
      {[1, 2, 3, 4, 5].map((i) => <i key={i} className={i <= n ? 'on' : ''} />)}
    </span>
  );
}

/** Card with a cursor-following spotlight border. */
function Spotlight({ className = '', children, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const onMove = (e: React.PointerEvent<HTMLButtonElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    e.currentTarget.style.setProperty('--mx', `${e.clientX - r.left}px`);
    e.currentTarget.style.setProperty('--my', `${e.clientY - r.top}px`);
  };
  return <button type="button" className={`ix-spot ${className}`} onPointerMove={onMove} {...rest}>{children}</button>;
}

function initialAlgo() {
  const id = location.hash.split('/')[2];
  return Math.max(0, ALGOS.findIndex((a) => a.id === id));
}

export default function IndexingPage() {
  const [idx, setIdx] = useState(initialAlgo);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [autoNext, setAutoNext] = useState(false);
  const algo: AlgoDef = ALGOS[idx];
  const api = useRef<SceneApi | null>(null);
  const fills = useRef<(HTMLElement | null)[]>([]);
  const statRef = useRef<HTMLParagraphElement>(null);
  const stepRef = useRef(0);
  const autoNextRef = useRef(autoNext);
  autoNextRef.current = autoNext;

  useEffect(() => {
    const prev = document.title;
    document.title = 'Vector indexing algorithms · RAG Lens';
    window.scrollTo(0, 0);
    return () => { document.title = prev; };
  }, []);

  const choose = useCallback((i: number) => {
    setIdx(i);
    setStep(0);
    stepRef.current = 0;
    setPlaying(true);
    history.replaceState(null, '', `#/indexing/${ALGOS[i].id}`);
  }, []);

  const onFrame = useCallback(({ t, starts, stat }: FrameInfo) => {
    const n = starts.length - 1;
    let s = 0;
    while (s < n - 1 && t >= starts[s + 1]) s++;
    for (let i = 0; i < n; i++) {
      const el = fills.current[i];
      if (el) el.style.transform = `scaleX(${Math.min(1, Math.max(0, (t - starts[i]) / (starts[i + 1] - starts[i])))})`;
    }
    if (statRef.current && statRef.current.textContent !== stat) statRef.current.textContent = stat;
    if (s !== stepRef.current) { stepRef.current = s; setStep(s); }
  }, []);

  const onLoop = useCallback(() => {
    if (autoNextRef.current) setIdx((i) => {
      const next = (i + 1) % ALGOS.length;
      history.replaceState(null, '', `#/indexing/${ALGOS[next].id}`);
      return next;
    });
  }, []);

  const seek = (s: number) => {
    api.current?.seek(s);
    stepRef.current = s;
    setStep(s);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest('input, textarea, select, button')) return;
      if (e.key === ' ') { e.preventDefault(); setPlaying((p) => !p); }
      if (e.key === 'ArrowRight') seek(Math.min(stepRef.current + 1, algo.steps.length - 1));
      if (e.key === 'ArrowLeft') seek(Math.max(stepRef.current - 1, 0));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  return (
    <div className="app ix">
      <nav className="topbar">
        <a className="logo" href="#/"><span className="logo-mark" />RAG Lens</a>
        <div className="ix-nav">
          <a href="#/">Pipeline</a>
          <a href="#/indexing" className="on" aria-current="page">Indexing</a>
        </div>
        <span />
      </nav>

      <header className="ix-hero">
        <p className="eyebrow">Vector indexing · 7 algorithms</p>
        <h1>How a vector database finds a <em>nearest neighbour</em></h1>
        <p className="lede">
          Retrieval compares a query embedding with millions of document embeddings. Checking every one is exact but slow.
          Each index below skips most of that work in a different way: clustering, graphs, compression, hashing, trees or SSDs.
          Pick one to see it built and searched in 3D.
        </p>
      </header>

      <div className="ix-picker" role="tablist" aria-label="Indexing algorithm">
        {ALGOS.map((a, i) => (
          <Spotlight key={a.id} role="tab" aria-selected={i === idx} className={i === idx ? 'on' : ''} onClick={() => choose(i)}>
            <span className="ix-num">{String(i + 1).padStart(2, '0')}</span>
            <b>{a.name}</b>
            <small>{a.family}</small>
          </Spotlight>
        ))}
      </div>

      <section className="ix-stage" aria-label={`${algo.full} animation`}>
        <IndexScene algo={algo} playing={playing} apiRef={api} onFrame={onFrame} onLoop={onLoop} />
        <div className="ix-hud-tl" key={algo.id}>
          <p className="eyebrow">{String(idx + 1).padStart(2, '0')} · {algo.family}</p>
          <h2>{algo.full}</h2>
          <p className="ix-tagline">{algo.tagline}</p>
        </div>
        <div className="ix-caption" key={`${algo.id}-${step}`}>
          <span className="ix-step-no">Step {step + 1} / {algo.steps.length}</span>
          <b>{algo.steps[step].title}</b>
          <p>{algo.steps[step].detail}</p>
          <p className="ix-stat" ref={statRef} aria-live="off" />
        </div>
        <div className="ix-controls">
          <button type="button" className="btn ix-icon" onClick={() => setPlaying((p) => !p)} aria-label={playing ? 'Pause' : 'Play'}>
            {playing ? <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><rect x="3" y="2" width="3.5" height="12" rx="1" fill="currentColor" /><rect x="9.5" y="2" width="3.5" height="12" rx="1" fill="currentColor" /></svg>
              : <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M4 2.5v11l9.5-5.5z" fill="currentColor" /></svg>}
          </button>
          <button type="button" className="btn ix-icon" onClick={() => seek(0)} aria-label="Restart">
            <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path d="M3 8a5 5 0 1 0 1.6-3.7M3 2.5v3h3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </button>
          <ol className="ix-timeline">
            {algo.steps.map((s, i) => (
              <li key={`${algo.id}-${i}`} style={{ flexGrow: s.dur }}>
                <button type="button" className={i === step ? 'on' : i < step ? 'done' : ''} onClick={() => seek(i)} title={s.title}>
                  <span className="ix-track"><i ref={(el) => { fills.current[i] = el; }} /></span>
                  <span className="ix-tl-label">{s.title}</span>
                </button>
              </li>
            ))}
          </ol>
          <label className="ix-auto">
            <input type="checkbox" checked={autoNext} onChange={(e) => setAutoNext(e.target.checked)} /> Tour all
          </label>
        </div>
      </section>

      <section className="ix-detail" key={algo.id}>
        <div>
          <h3 className="col-title">How it works</h3>
          <ol className="ix-steps">
            {algo.steps.map((s, i) => (
              <li key={i} className={i === step ? 'on' : ''}>
                <button type="button" onClick={() => seek(i)}>
                  <b>{s.title}</b>
                  <span>{s.detail}</span>
                </button>
              </li>
            ))}
          </ol>
        </div>
        <aside className="ix-profile">
          <h3 className="col-title">Profile</h3>
          <dl className="ix-ratings">
            {RATING_LABELS.map(([k, l]) => (
              <div key={k}><dt>{l}</dt><dd><Pips n={algo.rating[k]} /></dd></div>
            ))}
          </dl>
          <dl className="ix-facts">
            <div><dt>Query cost</dt><dd>{algo.complexity}</dd></div>
            <div><dt>Knobs</dt><dd className="ix-knobs">{algo.params.map((p) => <span key={p} className="chip">{p}</span>)}</dd></div>
            <div><dt>Best for</dt><dd>{algo.bestFor}</dd></div>
            <div><dt>Watch out</dt><dd>{algo.watchOut}</dd></div>
            <div><dt>Found in</dt><dd>{algo.usedIn}</dd></div>
          </dl>
        </aside>
      </section>

      <section className="ix-compare">
        <h3>At a glance</h3>
        <div className="ix-table-wrap">
          <table>
            <thead>
              <tr><th scope="col">Index</th><th scope="col">Family</th>{RATING_LABELS.map(([k, l]) => <th key={k} scope="col">{l}</th>)}<th scope="col">Reach for it when</th></tr>
            </thead>
            <tbody>
              {ALGOS.map((a, i) => (
                <tr key={a.id} className={i === idx ? 'on' : ''} onClick={() => { choose(i); document.querySelector('.ix-stage')?.scrollIntoView({ behavior: 'smooth', block: 'center' }); }}>
                  <th scope="row">{a.name}</th>
                  <td>{a.family}</td>
                  {RATING_LABELS.map(([k]) => <td key={k}><Pips n={a.rating[k]} /></td>)}
                  <td className="ix-when">{a.bestFor}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="ix-note">
          Ratings are relative, for typical settings. Production systems mix these ideas: <b>IVF-PQ</b> (clusters + compression),
          <b> HNSW + PQ</b>, or <b>DiskANN</b> (graph + PQ + SSD). They usually re-rank the shortlist with exact distances, which is what the
          <a href="#/"> RAG Lens pipeline</a> does with its flat index.
        </p>
      </section>

      <footer className="ix-legend">
        <span><i style={{ background: '#fff', boxShadow: '0 0 8px #fff' }} />query</span>
        <span><i style={{ background: 'var(--dense)' }} />distance computed</span>
        <span><i style={{ background: 'var(--sparse)' }} />candidate / neighbour checked</span>
        <span><i style={{ background: 'var(--final)' }} />returned top-k</span>
        <span><i style={{ background: '#ff6b6b' }} />true neighbour the index missed</span>
        <span className="ix-keys">Space play/pause · ← → step · drag to orbit</span>
      </footer>
    </div>
  );
}
