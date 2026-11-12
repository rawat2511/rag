import { useEffect, useMemo, useRef, useState } from 'react';
import { VectorStrip } from './components/canvases';
import { Scene3D, type SceneView } from './components/Scene3D';
import { QueryView } from './components/QueryView';
import { EMBED_DIM, MAX_PAGES, REPO_URL } from './config';
import { BM25 } from './lib/bm25';
import { chunkPages, countWords } from './lib/chunker';
import { detectWebGPU, ml } from './lib/ml';
import { extractPdf, extractText } from './lib/pdf';
import type { Chunk, Device, FusedHit, PageText, QueryRun, RerankedHit, RetrievalMode } from './lib/types';
import { pca3d, project, rrf, topKDense, type Projection } from './lib/vector';

type Phase = 'idle' | 'parsing' | 'embedding' | 'ready';
type Compute = 'auto' | Device;

interface DocInfo {
  name: string;
  isSample: boolean;
  pages: number;
  words: number;
  chunks: Chunk[];
  vocabulary: number;
  size: number;
  overlap: number;
}

const SAMPLE_QUESTIONS = [
  'How old must someone be to become President?',
  'Who has the power to declare war?',
  'How can the Constitution be amended?',
  'What counts as treason?',
];

const NONE: number[] = [];
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const pagesOf = (c: Chunk) => (c.pageStart === c.pageEnd ? `page ${c.pageStart}` : `pages ${c.pageStart}–${c.pageEnd}`);

export default function App() {
  const [cfg, setCfg] = useState({ size: 160, overlap: 30, compute: 'auto' as Compute });
  const [phase, setPhase] = useState<Phase>('idle');
  const [error, setError] = useState('');
  const [doc, setDoc] = useState<DocInfo | null>(null);
  const [parse, setParse] = useState({ done: 0, total: 0 });
  const [revealed, setRevealed] = useState(0);
  const [embedded, setEmbedded] = useState(0);
  const [device, setDevice] = useState<Device | null>(null);
  const [download, setDownload] = useState<{ model: string; pct: number } | null>(null);
  const [embedSecs, setEmbedSecs] = useState(0);
  const [proj, setProj] = useState<Projection | null>(null);
  const [view, setView] = useState<SceneView>('doc');
  const [selected, setSelected] = useState<number | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [rerankerReady, setRerankerReady] = useState(false);
  const [gpu, setGpu] = useState(false);
  const gpuFast = useRef(false);
  const [dragging, setDragging] = useState(false);

  const [question, setQuestion] = useState('');
  const [mode, setMode] = useState<RetrievalMode>('hybrid');
  const [useRerank, setUseRerank] = useState(true);
  const [topK, setTopK] = useState(10);
  const [finalN, setFinalN] = useState(4);
  const [run, setRun] = useState<QueryRun | null>(null);
  const [querying, setQuerying] = useState(false);

  const pagesRef = useRef<{ pages: PageText[]; name: string; isSample: boolean } | null>(null);
  const embRef = useRef<Float32Array | null>(null);
  const bm25Ref = useRef<BM25 | null>(null);
  const deviceRef = useRef<Device>('wasm');
  const ingestId = useRef(0);
  const queryId = useRef(0);

  useEffect(() => {
    void detectWebGPU().then((g) => { setGpu(g.available); gpuFast.current = g.fast; });
    ml.onDownload = (model, loaded, total) => setDownload({ model, pct: Math.round((loaded / total) * 100) });
    return () => { ml.onDownload = null; };
  }, []);

  /* ───────────── Ingestion: parse → chunk → index → embed ───────────── */

  async function ingest(pages: PageText[], name: string, isSample: boolean) {
    const id = ++ingestId.current;
    queryId.current++;
    ml.cancelEmbedding();
    const stale = () => id !== ingestId.current;

    const chunks = chunkPages(pages, cfg.size, cfg.overlap);
    if (chunks.length === 0) throw new Error('No text could be extracted from this document.');
    const texts = chunks.map((c) => c.text);
    const bm25 = new BM25(texts);
    bm25Ref.current = bm25;
    const emb = new Float32Array(chunks.length * EMBED_DIM);
    embRef.current = emb;

    setDoc({
      name,
      isSample,
      pages: pages.length,
      words: pages.reduce((sum, p) => sum + countWords(p.text), 0),
      chunks,
      vocabulary: bm25.vocabularySize,
      size: cfg.size,
      overlap: cfg.overlap,
    });
    setRun(null);
    setProj(null);
    setView('doc');
    setSelected(null);
    setEmbedded(0);
    setEmbedSecs(0);
    setDownload(null);
    setQuestion('');
    setPhase('embedding');

    // Reveal chunk tiles progressively so the split is visible.
    const revealStart = performance.now();
    const reveal = () => {
      if (stale()) return;
      const f = Math.min(1, (performance.now() - revealStart) / 900);
      setRevealed(Math.ceil(chunks.length * f));
      if (f < 1) requestAnimationFrame(reveal);
    };
    setRevealed(0);
    requestAnimationFrame(reveal);

    const wanted: Device = cfg.compute === 'auto' ? (gpuFast.current && chunks.length > 400 ? 'webgpu' : 'wasm') : cfg.compute;
    let started = 0;
    try {
      const result = await ml.embed(
        texts,
        wanted,
        (dev) => {
          if (stale()) return;
          started = performance.now();
          setDevice(dev);
          setDownload(null);
        },
        (start, data) => {
          if (stale()) return;
          emb.set(data, start * EMBED_DIM);
          setEmbedded(start + data.length / EMBED_DIM);
          setEmbedSecs((performance.now() - started) / 1000);
        },
      );
      if (stale()) return;
      deviceRef.current = result.device;
      setProj(pca3d(emb, chunks.length, EMBED_DIM));
      // Let the fully lit wall register for a moment, then fly the blocks into embedding space.
      setTimeout(() => { if (!stale()) setView('space'); }, 900);
      setRevealed(chunks.length);
      setPhase('ready');
      ml.loadReranker().then(() => { setRerankerReady(true); setDownload(null); }, () => undefined);
    } catch (err) {
      if (stale()) return;
      throw err;
    }
  }

  async function loadFile(file: File) {
    setError('');
    const id = ++ingestId.current;
    try {
      let pages: PageText[];
      if (/\.pdf$/i.test(file.name) || file.type === 'application/pdf') {
        setPhase('parsing');
        setParse({ done: 0, total: 0 });
        setDoc(null);
        pages = await extractPdf(await file.arrayBuffer(), (done, total) => {
          if (id === ingestId.current) setParse({ done, total });
        });
      } else if (/\.(txt|md|markdown)$/i.test(file.name) || file.type.startsWith('text/')) {
        pages = extractText(await file.text());
      } else {
        throw new Error('Unsupported file type. Upload a PDF, TXT or Markdown file.');
      }
      if (id !== ingestId.current) return;
      pagesRef.current = { pages, name: file.name, isSample: false };
      await ingest(pages, file.name, false);
    } catch (err) {
      fail(err);
    }
  }

  async function loadSample() {
    setError('');
    try {
      const res = await fetch(`${import.meta.env.BASE_URL}sample.txt`);
      if (!res.ok) throw new Error('Could not load the sample document.');
      const pages = extractText(await res.text());
      const name = 'US Constitution (sample).txt';
      pagesRef.current = { pages, name, isSample: true };
      await ingest(pages, name, true);
    } catch (err) {
      fail(err);
    }
  }

  async function rechunk() {
    const src = pagesRef.current;
    if (!src) return;
    setError('');
    try {
      await ingest(src.pages, src.name, src.isSample);
    } catch (err) {
      fail(err);
    }
  }

  function fail(err: unknown) {
    setError(err instanceof Error ? err.message : String(err));
    setPhase((p) => (p === 'ready' ? p : 'idle'));
  }

  /* ───────────── Query: embed → dense + BM25 → fuse → re-rank ───────────── */

  async function ask(text: string) {
    const q = text.trim();
    const bm25 = bm25Ref.current;
    const emb = embRef.current;
    if (!q || !doc || !proj || !bm25 || !emb || phase !== 'ready') return;
    const id = ++queryId.current;
    const stale = () => id !== queryId.current;
    setQuestion(q);
    setQuerying(true);
    setView('space');
    setError('');

    let r: QueryRun = {
      id, query: q, stage: 0, mode, useRerank, terms: [], dense: [], sparse: [], fused: [], reranked: [], final: [], timings: {},
    };
    const push = (patch: Partial<QueryRun>) => {
      r = { ...r, ...patch };
      if (!stale()) setRun(r);
    };
    push({});

    try {
      // 1 · Query understanding
      let t0 = performance.now();
      const qvec = await ml.embedQuery(q, deviceRef.current);
      if (stale()) return;
      push({ stage: 1, qvec, qpos: project(proj, qvec), terms: bm25.analyze(q), timings: { embed: performance.now() - t0 } });
      await sleep(500);
      if (stale()) return;

      // 2 · Two retrievers, side by side
      t0 = performance.now();
      const dense = topKDense(emb, doc.chunks.length, EMBED_DIM, qvec, topK);
      const tDense = performance.now() - t0;
      t0 = performance.now();
      const sparse = bm25.search(q, topK);
      const tBm25 = performance.now() - t0;
      push({ stage: 2, dense, sparse, timings: { ...r.timings, dense: tDense, bm25: tBm25 } });
      await sleep(700);
      if (stale()) return;

      // 3 · Fusion
      t0 = performance.now();
      let fused: FusedHit[];
      if (mode === 'hybrid') fused = rrf(dense, sparse).slice(0, topK);
      else if (mode === 'dense') fused = dense.map((h) => ({ ...h, denseRank: h.rank }));
      else fused = sparse.map((h) => ({ id: h.id, score: h.score, rank: h.rank, sparseRank: h.rank }));
      push({ stage: 3, fused, timings: { ...r.timings, fuse: performance.now() - t0 } });
      await sleep(600);
      if (stale()) return;

      // 4 · Cross-encoder re-ranking
      if (useRerank && fused.length > 0) {
        const pending: RerankedHit[] = fused.map((h) => ({ id: h.id, rank: h.rank, prevRank: h.rank }));
        push({ stage: 4, reranked: pending });
        t0 = performance.now();
        const logits = await ml.rerank(q, fused.map((h) => doc.chunks[h.id].text), (start, scores) => {
          const next = r.reranked.slice();
          scores.forEach((s, i) => (next[start + i] = { ...next[start + i], score: sigmoid(s) }));
          push({ reranked: next });
        });
        if (stale()) return;
        setRerankerReady(true);
        setDownload(null);
        const reranked = fused
          .map((h, i) => ({ id: h.id, prevRank: h.rank, score: sigmoid(logits[i]), rank: 0 }))
          .sort((a, b) => b.score - a.score)
          .map((h, i) => ({ ...h, rank: i + 1 }));
        await sleep(350);
        push({ stage: 5, reranked, final: reranked.slice(0, finalN).map((h) => h.id), timings: { ...r.timings, rerank: performance.now() - t0 } });
      } else {
        push({ stage: 5, final: fused.slice(0, finalN).map((h) => h.id) });
      }
    } catch (err) {
      if (!stale()) setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (!stale()) setQuerying(false);
    }
  }

  /* ───────────── Derived view state ───────────── */

  const n = doc?.chunks.length ?? 0;
  const inspectId = doc ? (selected ?? hover ?? (phase === 'embedding' ? Math.max(0, embedded - 1) : null)) : null;
  const inspect = doc && inspectId !== null ? doc.chunks[Math.min(inspectId, n - 1)] : null;
  const inspectVec =
    inspect && embRef.current && inspect.id < embedded
      ? embRef.current.subarray(inspect.id * EMBED_DIM, (inspect.id + 1) * EMBED_DIM)
      : null;
  const rate = embedSecs > 0 ? embedded / embedSecs : 0;
  const eta = rate > 0 ? Math.ceil((n - embedded) / rate) : 0;
  const settingsChanged = !!doc && (doc.size !== cfg.size || doc.overlap !== cfg.overlap);
  const stageNow =
    phase === 'idle' ? -1 : phase === 'parsing' ? 0 : phase === 'embedding' ? (embedded > 0 ? 2 : 1)
    : !run ? 3 : run.stage < 2 ? 4 : run.stage < 3 ? 5 : run.stage < 5 ? 6 : 7;
  const pipeline = ['Parse', 'Chunk', 'Embed', 'Index', 'Retrieve', 'Fuse', 'Re-rank', 'Answer'];
  const retrieved = !!run && run.stage >= 2;
  const denseIds = useMemo(() => (retrieved ? run.dense.map((h) => h.id) : NONE), [retrieved, run?.dense]);
  const sparseIds = useMemo(() => (retrieved ? run.sparse.map((h) => h.id) : NONE), [retrieved, run?.sparse]);
  const status =
    phase === 'ready'
      ? `Embedded in ${embedSecs.toFixed(1)}s on ${device === 'webgpu' ? 'WebGPU' : 'WebAssembly'}`
      : download?.model === 'embedder'
        ? `Downloading embedding model · ${download.pct}%`
        : embedded === 0
          ? 'Loading embedding model…'
          : `Embedding ${embedded.toLocaleString()} / ${n.toLocaleString()} · ${rate.toFixed(0)} chunks/s · ~${eta}s left`;

  const filePicker = (
    <input type="file" accept=".pdf,.txt,.md,.markdown,application/pdf,text/plain"
      onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ''; if (f) void loadFile(f); }} />
  );
  const [showSettings, setShowSettings] = useState(false);
  const settings = (
    <div className="settings">
      <label>Chunk size <b>{cfg.size} words</b>
        <input type="range" min={60} max={300} step={20} value={cfg.size}
          onChange={(e) => setCfg({ ...cfg, size: +e.target.value, overlap: Math.min(cfg.overlap, +e.target.value / 2) })} />
      </label>
      <label>Overlap <b>{cfg.overlap} words</b>
        <input type="range" min={0} max={Math.min(80, cfg.size / 2)} step={10} value={cfg.overlap}
          onChange={(e) => setCfg({ ...cfg, overlap: +e.target.value })} />
      </label>
      <label>Compute
        <select value={cfg.compute} onChange={(e) => setCfg({ ...cfg, compute: e.target.value as Compute })}>
          <option value="auto">Auto</option>
          <option value="wasm">CPU (WebAssembly)</option>
          <option value="webgpu" disabled={!gpu}>GPU (WebGPU){gpu ? '' : ' · unavailable'}</option>
        </select>
      </label>
      {settingsChanged && phase !== 'parsing' && (
        <button className="btn primary" onClick={() => void rechunk()}>Re-chunk document</button>
      )}
    </div>
  );

  return (
    <div className="app">
      <nav className="topbar">
        <div className="logo"><span className="logo-mark" />RAG Lens</div>
        <ol className="pipeline" aria-label="Pipeline progress">
          {pipeline.map((label, i) => (
            <li key={label} className={i < stageNow ? 'done' : i === stageNow ? 'active' : ''}>{label}</li>
          ))}
        </ol>
        {REPO_URL ? <a className="btn ghost" href={REPO_URL} target="_blank" rel="noreferrer">Source ↗</a> : <span />}
      </nav>

      <section className={`stage${doc ? ' has-doc' : ''}`}>
        <Scene3D
          n={n} revealed={revealed} embedded={embedded} embedding={phase === 'embedding'}
          coords={proj?.coords ?? null} view={view} query={run?.qpos ?? null}
          denseIds={denseIds} sparseIds={sparseIds} finalIds={run?.final ?? NONE}
          hover={hover} selected={selected} onHover={setHover}
          onSelect={(id) => setSelected((s) => (s === id ? null : id))}
        />

        {error && <div className="toast" role="alert">{error}<button onClick={() => setError('')} aria-label="Dismiss">✕</button></div>}

        {!doc && (
          <div className="hero">
            <p className="eyebrow">Retrieval-augmented generation, made visible</p>
            <h1>Watch a document <em>become searchable.</em></h1>
            <p className="lede">
              Drop in up to {MAX_PAGES} pages. See it split into chunks, turned into vectors, and scattered through
              embedding space. Then ask a question and follow it through vector search, BM25, rank fusion and
              re-ranking. It all runs in your browser; the file never leaves this tab.
            </p>
            {phase === 'parsing' ? (
              <div className="parse">
                <span>Extracting text{parse.total ? ` · page ${parse.done} of ${parse.total}` : '…'}</span>
                <div className="meter"><i style={{ width: `${parse.total ? (parse.done / parse.total) * 100 : 4}%` }} /></div>
              </div>
            ) : (
              <>
                <label
                  className={`dropzone${dragging ? ' over' : ''}`}
                  onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragging(false);
                    const f = e.dataTransfer.files[0];
                    if (f) void loadFile(f);
                  }}
                >
                  {filePicker}
                  <span className="drop-icon" aria-hidden>↑</span>
                  <span><strong>Drop a PDF, TXT or Markdown file</strong><small>or click to browse</small></span>
                </label>
                <div className="hero-actions">
                  <button className="btn primary" onClick={() => void loadSample()}>Try the sample document</button>
                  <button className={`btn ghost${showSettings ? ' on' : ''}`} aria-expanded={showSettings}
                    aria-controls="hero-settings" onClick={() => setShowSettings((v) => !v)}>
                    Chunking settings
                  </button>
                </div>
                {showSettings && <div id="hero-settings" className="settings-panel fade">{settings}</div>}
              </>
            )}
          </div>
        )}

        {doc && (
          <>
            <div className="hud hud-tl">
              <p className="eyebrow">{view === 'doc' ? 'Document · one block per chunk' : 'Embedding space · 384 dimensions shown in 3'}</p>
              <h2 title={doc.name}>{doc.name}</h2>
              <dl className="stats">
                <div><dt>pages</dt><dd>{doc.pages.toLocaleString()}</dd></div>
                <div><dt>words</dt><dd>{doc.words.toLocaleString()}</dd></div>
                <div><dt>chunks</dt><dd>{n.toLocaleString()}</dd></div>
                <div><dt>BM25 terms</dt><dd>{doc.vocabulary.toLocaleString()}</dd></div>
                <div><dt>index</dt><dd>{((n * EMBED_DIM * 4) / 1048576).toFixed(1)} MB</dd></div>
              </dl>
              <div className={`status${phase === 'ready' ? ' ok' : ''}`}>
                <span>{status}</span>
                {phase !== 'ready' && <div className="meter"><i style={{ width: `${n ? (embedded / n) * 100 : 0}%` }} /></div>}
              </div>
            </div>

            <div className="hud hud-tr">
              <div className="seg" role="group" aria-label="3D view">
                <button className={view === 'doc' ? 'on' : ''} onClick={() => setView('doc')}>Document</button>
                <button className={view === 'space' ? 'on' : ''} disabled={!proj} onClick={() => setView('space')}
                  title={proj ? '' : 'Available once every chunk is embedded'}>Embedding space</button>
              </div>
              <ul className="legend">
                {view === 'doc' && <li><i style={{ background: '#2c3145' }} />waiting</li>}
                <li><i className="ramp" />{view === 'doc' ? 'embedded' : 'chunk'} · colour = position in document</li>
                {retrieved && view === 'space' && <li><i className="dot" style={{ background: '#fff' }} />your query</li>}
                {retrieved && <li><i style={{ background: 'var(--dense)' }} />dense hit</li>}
                {retrieved && <li><i style={{ background: 'var(--sparse)' }} />BM25 hit</li>}
                {retrieved && <li><i style={{ background: 'var(--fuse)' }} />found by both</li>}
                {run && run.final.length > 0 && <li><i style={{ background: 'var(--final)' }} />sent to the LLM</li>}
              </ul>
            </div>

            <div className="hud hud-bl">
              <label className="btn ghost file-btn">{filePicker}New document</label>
              <details className="pop up">
                <summary className="btn ghost">Settings</summary>
                <div className="pop-body">{settings}</div>
              </details>
              <span className="hint">Drag to rotate · click a block to pin it</span>
            </div>

            {inspect && (
              <aside className="hud hud-br inspector">
                <header>
                  <b>Chunk {inspect.id + 1}</b>
                  <span>{pagesOf(inspect)} · {inspect.words} words</span>
                  {selected !== null && <button className="link" onClick={() => setSelected(null)}>unpin</button>}
                </header>
                <p className="chunk-text">
                  {inspect.overlapChars > 0 && <span className="overlap" title="Overlap repeated from the previous chunk">{inspect.text.slice(0, inspect.overlapChars)}</span>}
                  {inspect.text.slice(inspect.overlapChars)}
                </p>
                <footer>
                  <span>
                    {inspectVec ? `its embedding · ${EMBED_DIM} numbers` : 'waiting to be embedded'}
                    {inspect.overlapChars > 0 && <em> · highlighted text overlaps the previous chunk</em>}
                  </span>
                  <VectorStrip vec={inspectVec} cell={5} />
                </footer>
              </aside>
            )}
          </>
        )}
      </section>

      {doc && (
        <section className={`ask-section${phase !== 'ready' ? ' disabled' : ''}`}>
          <form className="ask" onSubmit={(e) => { e.preventDefault(); void ask(question); }}>
            <input value={question} onChange={(e) => setQuestion(e.target.value)} disabled={phase !== 'ready'}
              aria-label="Question"
              placeholder={phase === 'ready' ? 'Ask something the document can answer…' : 'You can ask a question once embedding finishes'} />
            <button className="btn primary" disabled={phase !== 'ready' || querying || !question.trim()}>
              {querying ? 'Running…' : 'Run retrieval'}
            </button>
          </form>
          {doc.isSample && phase === 'ready' && (
            <div className="suggestions">
              {SAMPLE_QUESTIONS.map((q) => (
                <button key={q} className="chip-btn" disabled={querying} onClick={() => void ask(q)}>{q}</button>
              ))}
            </div>
          )}
          <div className="options">
            <div className="seg" role="group" aria-label="Retrieval mode">
              {(['hybrid', 'dense', 'bm25'] as const).map((m) => (
                <button key={m} className={mode === m ? 'on' : ''} onClick={() => setMode(m)}>
                  {m === 'hybrid' ? 'Hybrid' : m === 'dense' ? 'Dense only' : 'BM25 only'}
                </button>
              ))}
            </div>
            <label className="check"><input type="checkbox" checked={useRerank} onChange={(e) => setUseRerank(e.target.checked)} />Re-rank</label>
            <label>Top-K
              <select value={topK} onChange={(e) => setTopK(+e.target.value)}>{[5, 10, 20, 30].map((k) => <option key={k}>{k}</option>)}</select>
            </label>
            <label>Chunks to LLM
              <select value={finalN} onChange={(e) => setFinalN(+e.target.value)}>{[3, 4, 5, 8].map((k) => <option key={k}>{k}</option>)}</select>
            </label>
            {phase === 'ready' && useRerank && !rerankerReady && (
              <span className="loading">
                {download?.model === 'reranker' ? `Downloading re-ranker · ${download.pct}%` : 'Loading re-ranker…'}
              </span>
            )}
          </div>
          <QueryView chunks={doc.chunks} run={run} hover={hover} onHover={setHover} />
        </section>
      )}

      <footer className="how">
        <h3>What happens at each step</h3>
        <ol className="how-grid">
          <li><b>Parse</b>pdf.js extracts text page by page, so every chunk remembers its page numbers.</li>
          <li><b>Chunk</b>Sentences are packed up to a target size; the tail of each chunk is repeated at the start of the next.</li>
          <li><b>Embed</b>A MiniLM sentence-transformer runs in a Web Worker via ONNX Runtime (WebGPU or WebAssembly).</li>
          <li><b>Dense search</b>Cosine similarity between the query vector and every chunk vector finds matches by meaning.</li>
          <li><b>BM25</b>A hand-written inverted index scores exact keyword matches, weighted by term rarity and chunk length.</li>
          <li><b>Fuse</b>Reciprocal Rank Fusion merges both rankings without needing their scores to be comparable.</li>
          <li><b>Re-rank</b>A cross-encoder reads the query and each candidate together and re-scores them more precisely.</li>
          <li><b>Generate</b>The top chunks become the context for an LLM, which answers with citations.</li>
        </ol>
        <p>Built with React, TypeScript, three.js, Transformers.js and pdf.js. No backend.</p>
      </footer>
    </div>
  );
}
