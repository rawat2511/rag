import { useEffect, useRef, useState } from 'react';
import { ChunkGrid, EmbeddingMap, VectorStrip } from './components/canvases';
import { QueryView } from './components/QueryView';
import { EMBED_DIM, EMBED_MODEL, MAX_PAGES, REPO_URL } from './config';
import { BM25 } from './lib/bm25';
import { chunkPages, countWords } from './lib/chunker';
import { hasWebGPU, ml } from './lib/ml';
import { extractPdf, extractText } from './lib/pdf';
import type { Chunk, Device, FusedHit, PageText, QueryRun, RerankedHit, RetrievalMode } from './lib/types';
import { pca2d, project, rrf, topKDense, type Projection } from './lib/vector';

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
  const [selected, setSelected] = useState<number | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [rerankerReady, setRerankerReady] = useState(false);
  const [gpu, setGpu] = useState(false);
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
    void hasWebGPU().then(setGpu);
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

    const wanted: Device = cfg.compute === 'auto' ? (gpu && chunks.length > 400 ? 'webgpu' : 'wasm') : cfg.compute;
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
      setProj(pca2d(emb, chunks.length, EMBED_DIM));
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
      push({ stage: 1, qvec, qxy: project(proj, qvec), terms: bm25.analyze(q), timings: { embed: performance.now() - t0 } });
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
  const inspectId = doc ? (selected ?? hover ?? (phase === 'embedding' ? Math.max(0, embedded - 1) : 0)) : null;
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

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-top">
          <div className="logo"><span className="logo-mark" />RAG Lens</div>
          {REPO_URL && <a className="btn ghost" href={REPO_URL} target="_blank" rel="noreferrer">Source code ↗</a>}
        </div>
        <h1>See how retrieval-augmented generation actually works</h1>
        <p className="lede">
          Upload a document of up to {MAX_PAGES} pages and watch it get chunked, embedded and indexed. Then ask a
          question and follow it through dense vector search, BM25 keyword search, rank fusion and cross-encoder
          re-ranking. Everything runs in your browser: your document never leaves this tab.
        </p>
        <div className="pipeline">
          {pipeline.map((label, i) => (
            <span key={label} className={`pipe${i < stageNow ? ' done' : i === stageNow ? ' active' : ''}`}>{label}</span>
          ))}
        </div>
      </header>

      {error && <div className="error-banner" role="alert">{error}<button onClick={() => setError('')}>✕</button></div>}

      {/* ── 1 · Document ── */}
      <section className="panel">
        <h2><span className="num">1</span>Document</h2>
        <div className="doc-row">
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
            <input type="file" accept=".pdf,.txt,.md,.markdown,application/pdf,text/plain"
              onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ''; if (f) void loadFile(f); }} />
            <strong>Drop a PDF, TXT or Markdown file</strong>
            <span>or click to browse · up to {MAX_PAGES} pages · processed locally</span>
            <button type="button" className="btn" onClick={(e) => { e.preventDefault(); void loadSample(); }}>
              No file handy? Try the sample document
            </button>
          </label>

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
              <button className="btn primary" onClick={() => void rechunk()}>Re-chunk with these settings</button>
            )}
          </div>
        </div>

        {phase === 'parsing' && (
          <div className="progress-line">
            <span>Extracting text{parse.total ? ` · page ${parse.done} of ${parse.total}` : '…'}</span>
            <div className="meter"><i style={{ width: `${parse.total ? (parse.done / parse.total) * 100 : 4}%` }} /></div>
          </div>
        )}
        {doc && (
          <div className="stats">
            <div><b>{doc.name}</b><span>document</span></div>
            <div><b>{doc.pages.toLocaleString()}</b><span>pages</span></div>
            <div><b>{doc.words.toLocaleString()}</b><span>words</span></div>
            <div><b>{n.toLocaleString()}</b><span>chunks</span></div>
            <div><b>{doc.vocabulary.toLocaleString()}</b><span>BM25 terms</span></div>
            <div><b>{((n * EMBED_DIM * 4) / 1048576).toFixed(1)} MB</b><span>vector index</span></div>
          </div>
        )}
      </section>

      {/* ── 2 · Chunking & embedding ── */}
      {doc && (
        <section className="panel">
          <h2>
            <span className="num">2</span>Chunking and embedding
            <span className={`status ${phase === 'ready' ? 'ok' : 'run'}`}>
              {phase === 'ready'
                ? `${n.toLocaleString()} chunks embedded in ${embedSecs.toFixed(1)}s · ${device === 'webgpu' ? 'WebGPU' : 'WebAssembly'}`
                : download?.model === 'embedder'
                  ? `Downloading embedding model · ${download.pct}% (cached after first visit)`
                  : embedded === 0
                    ? 'Loading embedding model…'
                    : `Embedding ${embedded.toLocaleString()} / ${n.toLocaleString()} · ${rate.toFixed(0)} chunks/s · ~${eta}s left`}
            </span>
          </h2>
          <p className="explain">
            The text is split into overlapping, sentence-aligned chunks. Each tile is one chunk, in document order. A
            tile lights up when <code>{EMBED_MODEL.split('/')[1]}</code> has turned it into a {EMBED_DIM}-dimensional vector.
          </p>
          <div className="meter big"><i style={{ width: `${n ? (embedded / n) * 100 : 0}%` }} /></div>
          <div className="chunk-row">
            <ChunkGrid n={n} revealed={revealed} embedded={embedded} selected={selected} hover={hover}
              finalIds={run?.final ?? []} onSelect={(id) => setSelected((s) => (s === id ? null : id))} onHover={setHover} />
            {inspect && (
              <div className="inspector">
                <div className="inspector-head">
                  <b>Chunk {inspect.id + 1}</b>
                  <span>{pagesOf(inspect)} · {inspect.words} words</span>
                  {selected !== null && <button className="link" onClick={() => setSelected(null)}>unpin</button>}
                </div>
                <p className="chunk-text">
                  {inspect.overlapChars > 0 && <span className="overlap" title="Overlap repeated from the previous chunk">{inspect.text.slice(0, inspect.overlapChars)}</span>}
                  {inspect.text.slice(inspect.overlapChars)}
                </p>
                <div className="inspector-vec">
                  <span>{inspectVec ? `Embedding · ${EMBED_DIM} dimensions` : 'Waiting to be embedded…'}</span>
                  <VectorStrip vec={inspectVec} />
                </div>
              </div>
            )}
          </div>
          <div className="legend">
            <span><i className="sw" style={{ background: '#2a3350' }} />chunked</span>
            <span><i className="sw grad" />embedded (colour = position in document)</span>
            <span><i className="sw" style={{ background: 'rgba(255,215,106,.35)' }} />overlap with previous chunk</span>
            <span><i className="sw ring" />sent to the LLM for the current query</span>
            <span className="muted">Hover or click a tile to inspect it</span>
          </div>
        </section>
      )}

      {/* ── 3 · Embedding space ── */}
      {doc && (
        <section className="panel">
          <h2><span className="num">3</span>Embedding space</h2>
          <p className="explain">
            Every chunk vector projected from {EMBED_DIM} dimensions down to 2 with PCA. Chunks about similar things land
            near each other. When you ask a question it is embedded into the same space and plotted here.
          </p>
          {proj ? (
            <>
              <EmbeddingMap proj={proj} chunks={doc.chunks} selected={selected} hover={hover}
                queryXY={run?.qxy} denseIds={run && run.stage >= 2 ? run.dense.map((h) => h.id) : []}
                sparseIds={run && run.stage >= 2 ? run.sparse.map((h) => h.id) : []} finalIds={run?.final ?? []}
                onSelect={(id) => setSelected((s) => (s === id ? null : id))} onHover={setHover} />
              <div className="legend">
                <span><i className="sw diamond" />query</span>
                <span><i className="sw ring dense" />dense hit</span>
                <span><i className="sw ring sparse" />BM25 hit</span>
                <span><i className="sw" style={{ background: '#ffd76a', borderRadius: '50%' }} />sent to the LLM</span>
                <span className="muted">2D is a lossy view: neighbours here are a hint, the ranking uses all {EMBED_DIM} dimensions</span>
              </div>
            </>
          ) : (
            <div className="placeholder">The map appears once all chunks are embedded.</div>
          )}
        </section>
      )}

      {/* ── 4 · Ask ── */}
      {doc && (
        <section className={`panel${phase !== 'ready' ? ' disabled' : ''}`}>
          <h2>
            <span className="num">4</span>Ask a question
            {phase === 'ready' && useRerank && !rerankerReady && (
              <span className="status run">
                {download?.model === 'reranker' ? `Downloading re-ranker · ${download.pct}%` : 'Loading re-ranker model…'}
              </span>
            )}
          </h2>
          <form className="ask" onSubmit={(e) => { e.preventDefault(); void ask(question); }}>
            <input value={question} onChange={(e) => setQuestion(e.target.value)} disabled={phase !== 'ready'}
              placeholder={phase === 'ready' ? 'Ask something the document can answer…' : 'Available once embedding finishes'} />
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
            <label>Top-K per retriever
              <select value={topK} onChange={(e) => setTopK(+e.target.value)}>{[5, 10, 20, 30].map((k) => <option key={k}>{k}</option>)}</select>
            </label>
            <label>Chunks sent to LLM
              <select value={finalN} onChange={(e) => setFinalN(+e.target.value)}>{[3, 4, 5, 8].map((k) => <option key={k}>{k}</option>)}</select>
            </label>
            <span className="muted">Change a setting and run again to compare</span>
          </div>
          <QueryView chunks={doc.chunks} run={run} hover={hover} onHover={setHover} />
        </section>
      )}

      <footer className="how">
        <h3>What is happening under the hood</h3>
        <div className="how-grid">
          <div><b>Parse</b>pdf.js extracts text page by page, so every chunk remembers its page numbers.</div>
          <div><b>Chunk</b>Sentences are packed up to a target size; the tail of each chunk is repeated at the start of the next.</div>
          <div><b>Embed</b>A MiniLM sentence-transformer runs in a Web Worker via ONNX Runtime (WebGPU or WebAssembly).</div>
          <div><b>Dense search</b>Cosine similarity between the query vector and every chunk vector finds matches by meaning.</div>
          <div><b>BM25</b>A hand-written inverted index scores exact keyword matches, weighted by term rarity and chunk length.</div>
          <div><b>Fusion</b>Reciprocal Rank Fusion merges both rankings without needing their scores to be comparable.</div>
          <div><b>Re-rank</b>A cross-encoder reads the query and each candidate together and re-scores them more precisely.</div>
          <div><b>Generate</b>The top chunks become the context for an LLM, which answers with citations.</div>
        </div>
        <p className="muted">Built with React, TypeScript, Transformers.js and pdf.js. No backend.</p>
      </footer>
    </div>
  );
}
