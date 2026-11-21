import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { IndexScene, type FrameInfo, type SceneApi } from '../indexing/IndexScene';
import { Spotlight } from '../indexing/Spotlight';
import { BM25, tokenize } from '../lib/bm25';
import { B, DOCS, K1, QUERY } from './corpus';
import { CHAPTERS, type Chapter } from './scenes';
import '../indexing/indexing.css';
import './bm25.css';

const TERM_COLORS = ['#58a8ff', '#f277b6', '#ffab3d', '#4fd6a8', '#8f7bff', '#5fd8e8', '#ffd95e'];
const PRESETS = ['Congress declare war', 'war war war', 'President army navy', 'senator age thirty years', 'Treason against the United States'];

function initialChapter() {
  const id = location.hash.split('/')[2];
  return Math.max(0, CHAPTERS.findIndex((c) => c.id === id));
}

function Formula({ part }: { part: Chapter['part'] }) {
  return (
    <div className="bm-formula" data-part={part} aria-label="BM25 scoring formula">
      <span className="bm-lhs">score(D, Q) =</span>
      <span className="p-sum">Σ<sub>q ∈ Q</sub></span>
      <span className="p-idf">IDF(q)</span>
      <span className="bm-op">·</span>
      <span className="bm-frac p-tf">
        <span><span className="p-count">f(q, D)</span> · (k₁ + 1)</span>
        <span><span className="p-count">f(q, D)</span> + k₁ · <span className="p-len">(1 − b + b · |D| / avgdl)</span></span>
      </span>
    </div>
  );
}

function Highlight({ text, colors }: { text: string; colors: Map<string, string> }) {
  const parts = text.split(/([\p{L}\p{N}]+)/u);
  return (
    <>
      {parts.map((p, i) => {
        const term = i % 2 ? tokenize(p)[0] : undefined;
        const c = term ? colors.get(term) : undefined;
        return c ? <mark key={i} style={{ background: `color-mix(in srgb, ${c} 22%, transparent)`, color: '#fff', boxShadow: `inset 0 -1px 0 ${c}` }}>{p}</mark> : <Fragment key={i}>{p}</Fragment>;
      })}
    </>
  );
}

function Playground() {
  const [query, setQuery] = useState(QUERY);
  const [k1, setK1] = useState(K1);
  const [b, setB] = useState(B);
  const bm = useMemo(() => new BM25(DOCS, k1, b), [k1, b]);
  const analysis = useMemo(() => bm.analyze(query), [bm, query]);
  const terms = useMemo(() => [...new Set(analysis.filter((a) => a.kept).map((a) => a.term))], [analysis]);
  const colors = useMemo(() => new Map(terms.map((t, i) => [t, TERM_COLORS[i % TERM_COLORS.length]])), [terms]);
  const hits = useMemo(() => bm.search(query, DOCS.length), [bm, query]);
  const max = Math.max(1e-6, ...hits.map((h) => h.score));
  const lengths = useMemo(() => DOCS.map((d) => tokenize(d).length), []);
  const misses = DOCS.map((_, i) => i).filter((i) => !hits.some((h) => h.id === i));

  return (
    <section className="bm-play" aria-labelledby="bm-play-title">
      <h3 id="bm-play-title">Try it: live BM25 on the eight documents</h3>
      <p className="ix-note">This runs the same <code>BM25</code> class the RAG Lens pipeline uses. Change the query or the knobs and the ranking updates instantly.</p>

      <div className="bm-controls">
        <div className="ask">
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Type a keyword query…" aria-label="Query" />
        </div>
        <div className="suggestions">
          {PRESETS.map((p) => <button key={p} type="button" className="chip-btn" onClick={() => setQuery(p)}>{p}</button>)}
        </div>
        <div className="settings bm-knobs">
          <label>k₁ · term saturation <b>{k1.toFixed(1)}</b>
            <input type="range" min={0} max={3} step={0.1} value={k1} onChange={(e) => setK1(Number(e.target.value))} />
          </label>
          <label>b · length normalisation <b>{b.toFixed(2)}</b>
            <input type="range" min={0} max={1} step={0.05} value={b} onChange={(e) => setB(Number(e.target.value))} />
          </label>
          <button type="button" className="btn ghost" onClick={() => { setK1(K1); setB(B); }} disabled={k1 === K1 && b === B}>Reset to 1.5 / 0.75</button>
        </div>
      </div>

      <div className="terms bm-terms" aria-label="Query terms">
        {analysis.length === 0 && <span className="empty">Type a query to see its terms.</span>}
        {analysis.map((a, i) => a.kept
          ? <span key={i} className="term" style={{ borderColor: colors.get(a.term), background: `color-mix(in srgb, ${colors.get(a.term)} 14%, transparent)` }}>
              {a.term}<b style={{ color: colors.get(a.term) }}>df {a.df} · idf {a.idf.toFixed(2)}</b>
            </span>
          : <span key={i} className="term dropped" title="stopword">{a.raw}</span>)}
      </div>

      <ol className="bm-results">
        {hits.map((h) => (
          <li key={h.id} className="bm-hit">
            <div className="bm-hit-head">
              <span className="rank">{h.rank}</span>
              <b>D{h.id + 1}</b>
              <span className="card-id">{lengths[h.id]} tokens</span>
              <span className="score">{h.score.toFixed(3)}</span>
            </div>
            <div className="bm-stack" aria-hidden="true">
              {h.terms.map((t) => <i key={t.term} style={{ width: `${(t.score / max) * 100}%`, background: colors.get(t.term) }} title={`${t.term}: ${t.score.toFixed(3)}`} />)}
            </div>
            <div className="bm-contrib">
              {h.terms.map((t) => <span key={t.term} className="chip" style={{ color: colors.get(t.term), borderColor: `color-mix(in srgb, ${colors.get(t.term)} 45%, transparent)` }}>{t.term} +{t.score.toFixed(2)}</span>)}
            </div>
            <p className="snippet"><Highlight text={DOCS[h.id]} colors={colors} /></p>
          </li>
        ))}
        {misses.map((i) => (
          <li key={i} className="bm-hit miss">
            <div className="bm-hit-head"><span className="rank">–</span><b>D{i + 1}</b><span className="card-id">{lengths[i]} tokens</span><span className="score">0</span></div>
            <p className="snippet">{DOCS[i]}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

export default function Bm25Page() {
  const [idx, setIdx] = useState(initialChapter);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [autoNext, setAutoNext] = useState(false);
  const ch = CHAPTERS[idx];
  const api = useRef<SceneApi | null>(null);
  const fills = useRef<(HTMLElement | null)[]>([]);
  const statRef = useRef<HTMLParagraphElement>(null);
  const stepRef = useRef(0);
  const autoNextRef = useRef(autoNext);
  autoNextRef.current = autoNext;

  useEffect(() => {
    const prev = document.title;
    document.title = 'BM25 explained in 3D · RAG Lens';
    window.scrollTo(0, 0);
    return () => { document.title = prev; };
  }, []);

  const choose = useCallback((i: number) => {
    setIdx(i);
    setStep(0);
    stepRef.current = 0;
    setPlaying(true);
    history.replaceState(null, '', `#/bm25/${CHAPTERS[i].id}`);
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
      const next = (i + 1) % CHAPTERS.length;
      history.replaceState(null, '', `#/bm25/${CHAPTERS[next].id}`);
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
      if (e.key === 'ArrowRight') seek(Math.min(stepRef.current + 1, ch.steps.length - 1));
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
          <a href="#/indexing">Indexing</a>
          <a href="#/bm25" className="on" aria-current="page">BM25</a>
        </div>
        <span />
      </nav>

      <header className="ix-hero">
        <p className="eyebrow">Keyword search · Okapi BM25</p>
        <h1>How BM25 decides which document <em>matches best</em></h1>
        <p className="lede">
          BM25 is the ranking function behind Elasticsearch, OpenSearch, Lucene and the keyword half of RAG Lens.
          It is just one formula, built from three ideas: rare words matter more, repeating a word has diminishing returns,
          and long documents should not win just by being long. Six chapters build it up piece by piece in 3D.
        </p>
      </header>

      <div className="ix-picker bm-picker" role="tablist" aria-label="Chapter">
        {CHAPTERS.map((c, i) => (
          <Spotlight key={c.id} role="tab" aria-selected={i === idx} className={i === idx ? 'on' : ''} onClick={() => choose(i)}>
            <span className="ix-num">{String(i + 1).padStart(2, '0')}</span>
            <b>{c.name}</b>
            <small>{c.full}</small>
          </Spotlight>
        ))}
      </div>

      <section className="ix-stage" aria-label={`${ch.full} animation`}>
        <IndexScene algo={ch} playing={playing} apiRef={api} onFrame={onFrame} onLoop={onLoop} />
        <div className="ix-hud-tl" key={ch.id}>
          <p className="eyebrow">Chapter {idx + 1} of {CHAPTERS.length}</p>
          <h2>{ch.full}</h2>
          <p className="ix-tagline">{ch.tagline}</p>
        </div>
        <div className="ix-caption" key={`${ch.id}-${step}`}>
          <span className="ix-step-no">Step {step + 1} / {ch.steps.length}</span>
          <b>{ch.steps[step].title}</b>
          <p>{ch.steps[step].detail}</p>
          <p className="ix-stat" ref={statRef} />
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
            {ch.steps.map((s, i) => (
              <li key={`${ch.id}-${i}`} style={{ flexGrow: s.dur }}>
                <button type="button" className={i === step ? 'on' : i < step ? 'done' : ''} onClick={() => seek(i)} title={s.title}>
                  <span className="ix-track"><i ref={(el) => { fills.current[i] = el; }} /></span>
                  <span className="ix-tl-label">{s.title}</span>
                </button>
              </li>
            ))}
          </ol>
          <label className="ix-auto">
            <input type="checkbox" checked={autoNext} onChange={(e) => setAutoNext(e.target.checked)} /> Play all
          </label>
        </div>
      </section>

      <section className="ix-detail" key={ch.id}>
        <div>
          <h3 className="col-title">How it works</h3>
          <ol className="ix-steps">
            {ch.steps.map((s, i) => (
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
          <h3 className="col-title">Where this lives in the formula</h3>
          <Formula part={ch.part} />
          <p className="bm-insight">{ch.insight}</p>
          <dl className="ix-facts bm-glossary">
            <div><dt>f(q, D)</dt><dd>times term q occurs in document D (tf)</dd></div>
            <div><dt>IDF(q)</dt><dd>ln(1 + (N − df + 0.5) / (df + 0.5))</dd></div>
            <div><dt>k₁</dt><dd>saturation, default 1.5 (typical 1.2–2.0)</dd></div>
            <div><dt>b</dt><dd>length normalisation, default 0.75</dd></div>
            <div><dt>|D|, avgdl</dt><dd>document length and average length, in tokens</dd></div>
          </dl>
        </aside>
      </section>

      <Playground />

      <section className="bm-corpus">
        <h3 className="col-title">The corpus used in every chapter</h3>
        <ol>
          {DOCS.map((d, i) => <li key={i}><b>D{i + 1}</b><span>{d}</span></li>)}
        </ol>
      </section>

      <footer className="ix-legend">
        <span><i style={{ background: TERM_COLORS[0] }} />congress</span>
        <span><i style={{ background: TERM_COLORS[1] }} />declare</span>
        <span><i style={{ background: TERM_COLORS[2] }} />war</span>
        <span><i style={{ background: 'var(--final)' }} />highlighted / winning</span>
        <span><i style={{ background: '#ff6b6b' }} />removed</span>
        <span className="ix-keys">Space play/pause · ← → step · drag to orbit</span>
      </footer>
    </div>
  );
}
