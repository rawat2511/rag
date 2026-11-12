import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { RERANK_MODEL } from '../config';
import { PROVIDERS, streamAnswer, type ProviderId } from '../lib/llm';
import type { Chunk, QueryRun } from '../lib/types';
import { VectorStrip } from './canvases';

const pagesOf = (c: Chunk) => (c.pageStart === c.pageEnd ? `p.${c.pageStart}` : `p.${c.pageStart}–${c.pageEnd}`);
const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function Highlight({ text, terms }: { text: string; terms: string[] }) {
  const parts = useMemo(() => {
    if (!terms.length) return [text];
    const re = new RegExp(`(?<![\\p{L}\\p{N}])((?:${terms.map(escapeRe).join('|')})[\\p{L}\\p{N}]*)`, 'giu');
    return text.split(re);
  }, [text, terms]);
  return <>{parts.map((p, i) => (i % 2 ? <mark key={i}>{p}</mark> : p))}</>;
}

interface CardProps {
  chunk: Chunk;
  rank: number;
  tone: 'dense' | 'sparse' | 'fuse' | 'rerank' | 'final';
  scoreLabel: string;
  bar: number;
  terms: string[];
  hover: number | null;
  onHover: (id: number | null) => void;
  delay?: number;
  children?: ReactNode;
}

function ResultCard({ chunk, rank, tone, scoreLabel, bar, terms, hover, onHover, delay = 0, children }: CardProps) {
  const [open, setOpen] = useState(false);
  return (
    <div
      className={`card tone-${tone}${hover === chunk.id ? ' linked' : ''}${open ? ' open' : ''}`}
      style={{ animationDelay: `${delay}ms` }}
      onMouseEnter={() => onHover(chunk.id)}
      onMouseLeave={() => onHover(null)}
      onClick={() => setOpen((o) => !o)}
      title="Click to expand"
    >
      <div className="card-head">
        <span className="rank">{rank}</span>
        <span className="card-id">Chunk {chunk.id + 1} · {pagesOf(chunk)}</span>
        <span className="score">{scoreLabel}</span>
      </div>
      <div className="bar"><i style={{ width: `${Math.max(2, Math.min(100, bar * 100))}%` }} /></div>
      {children && <div className="card-meta">{children}</div>}
      <p className="snippet"><Highlight text={chunk.text} terms={terms} /></p>
    </div>
  );
}

interface Props {
  chunks: Chunk[];
  run: QueryRun | null;
  hover: number | null;
  onHover: (id: number | null) => void;
}

export function QueryView({ chunks, run, hover, onHover }: Props) {
  if (!run) return null;
  const terms = [...new Set(run.terms.filter((t) => t.kept).map((t) => t.term))];
  const t = run.timings;
  const ms = (v?: number) => (v === undefined ? '…' : v < 1 ? '<1 ms' : `${Math.round(v)} ms`);
  const steps: [string, string, boolean][] = [
    ['Embed query', ms(t.embed), run.stage >= 1],
    ['Dense search', ms(t.dense), run.stage >= 2],
    ['BM25 search', ms(t.bm25), run.stage >= 2],
    ['Fuse', ms(t.fuse), run.stage >= 3],
    ...(run.useRerank ? [['Re-rank', ms(t.rerank), run.stage >= 5] as [string, string, boolean]] : []),
  ];
  const fusedTitle =
    run.mode === 'hybrid' ? 'Fused · Reciprocal Rank Fusion' : run.mode === 'dense' ? 'Candidates · dense only' : 'Candidates · BM25 only';
  const denseTop = run.dense[0]?.score || 1;
  const sparseTop = run.sparse[0]?.score || 1;
  const fusedTop = run.fused[0]?.score || 1;
  const scored = run.reranked.filter((r) => r.score !== undefined).length;

  return (
    <div className="query-view">
      <div className="timing">
        {steps.map(([label, value, done], i) => (
          <span key={label} className={`step${done ? ' done' : run.stage === 0 && i === 0 ? ' active' : ''}`}>
            {label} <b>{done ? value : '…'}</b>
          </span>
        ))}
      </div>

      {run.stage >= 1 && (
        <div className="analysis fade">
          <div className="box tone-dense">
            <h4>Dense path · the query becomes a vector</h4>
            <VectorStrip vec={run.qvec ?? null} />
            <p className="hint">384 numbers that capture meaning. Compared against every chunk vector by cosine similarity.</p>
          </div>
          <div className="box tone-sparse">
            <h4>Keyword path · the query becomes terms</h4>
            <div className="terms">
              {run.terms.map((q, i) =>
                q.kept ? (
                  <span key={i} className={`term${q.df === 0 ? ' miss' : ''}`} title={`appears in ${q.df} chunk(s)`}>
                    {q.term} <b>{q.df === 0 ? 'not in doc' : `idf ${q.idf.toFixed(2)}`}</b>
                  </span>
                ) : (
                  <span key={i} className="term dropped" title="stopword, ignored">{q.raw}</span>
                ),
              )}
            </div>
            <p className="hint">Stopwords are dropped. Rare terms get a higher IDF weight, so they count for more in BM25.</p>
          </div>
        </div>
      )}

      {run.stage >= 2 && (
        <div className="columns fade">
          <div className="col">
            <h4 className="col-title tone-dense">Dense retrieval <small>cosine similarity · top {run.dense.length}</small></h4>
            <div className="col-list">
              {run.dense.map((h, i) => (
                <ResultCard key={h.id} chunk={chunks[h.id]} rank={h.rank} tone="dense" scoreLabel={h.score.toFixed(3)}
                  bar={h.score / denseTop} terms={terms} hover={hover} onHover={onHover} delay={i * 35} />
              ))}
            </div>
          </div>
          <div className="col">
            <h4 className="col-title tone-sparse">BM25 keyword search <small>top {run.sparse.length}</small></h4>
            <div className="col-list">
              {run.sparse.length === 0 && <p className="empty">No chunk contains any of the query terms.</p>}
              {run.sparse.map((h, i) => (
                <ResultCard key={h.id} chunk={chunks[h.id]} rank={h.rank} tone="sparse" scoreLabel={h.score.toFixed(2)}
                  bar={h.score / sparseTop} terms={terms} hover={hover} onHover={onHover} delay={i * 35}>
                  {h.terms.slice(0, 4).map((x) => (
                    <span key={x.term} className="chip sparse">{x.term} +{x.score.toFixed(1)}</span>
                  ))}
                </ResultCard>
              ))}
            </div>
          </div>
          <div className="col">
            <h4 className="col-title tone-fuse">{fusedTitle}</h4>
            <div className="col-list">
              {run.stage < 3 && <p className="empty">Merging the two rankings…</p>}
              {run.stage >= 3 && run.fused.map((h, i) => (
                <ResultCard key={h.id} chunk={chunks[h.id]} rank={h.rank} tone="fuse"
                  scoreLabel={run.mode === 'hybrid' ? h.score.toFixed(4) : ''} bar={h.score / fusedTop}
                  terms={terms} hover={hover} onHover={onHover} delay={i * 35}>
                  <span className={`chip ${h.denseRank ? 'dense' : 'off'}`}>dense {h.denseRank ? `#${h.denseRank}` : '—'}</span>
                  <span className={`chip ${h.sparseRank ? 'sparse' : 'off'}`}>BM25 {h.sparseRank ? `#${h.sparseRank}` : '—'}</span>
                </ResultCard>
              ))}
            </div>
          </div>
        </div>
      )}

      {run.useRerank && run.stage >= 4 && (
        <div className="rerank fade">
          <h4 className="col-title tone-rerank">
            Cross-encoder re-ranking
            <small>
              {run.stage === 4
                ? `reading query + chunk together… ${scored}/${run.reranked.length}`
                : `${RERANK_MODEL.split('/')[1]} · ${run.reranked.length} candidates re-scored`}
            </small>
          </h4>
          <div className="rerank-list">
            {run.reranked.map((h, i) => {
              const move = h.prevRank - h.rank;
              return (
                <ResultCard key={h.id} chunk={chunks[h.id]} rank={h.rank} tone={run.final.includes(h.id) ? 'final' : 'rerank'}
                  scoreLabel={h.score === undefined ? 'scoring…' : h.score.toFixed(3)} bar={h.score ?? 0}
                  terms={terms} hover={hover} onHover={onHover} delay={i * 25}>
                  {run.stage >= 5 && (
                    <span className={`chip ${move > 0 ? 'up' : move < 0 ? 'down' : 'off'}`}>
                      {move > 0 ? `▲ ${move}` : move < 0 ? `▼ ${-move}` : '＝'} was #{h.prevRank}
                    </span>
                  )}
                  {run.final.includes(h.id) && <span className="chip final">sent to LLM</span>}
                </ResultCard>
              );
            })}
          </div>
        </div>
      )}

      {run.stage >= 5 && <AnswerPanel key={run.id} run={run} chunks={chunks} hover={hover} onHover={onHover} />}
    </div>
  );
}

/* ───────────────────────── Generation ───────────────────────── */

const read = (k: string) => {
  try { return sessionStorage.getItem(k) ?? ''; } catch { return ''; }
};
const write = (k: string, v: string) => {
  try { sessionStorage.setItem(k, v); } catch { /* storage unavailable */ }
};
const keyStore = (id: ProviderId) => `raglens.key.${id}`;
const modelStore = (id: ProviderId) => `raglens.model.${id}`;
const fromStore = (store: (id: ProviderId) => string) =>
  Object.fromEntries(PROVIDERS.map((p) => [p.id, read(store(p.id))])) as Record<ProviderId, string>;

function AnswerPanel({ run, chunks, hover, onHover }: { run: QueryRun; chunks: Chunk[]; hover: number | null; onHover: (id: number | null) => void }) {
  const [providerId, setProviderId] = useState<ProviderId>(() => {
    const saved = read('raglens.provider');
    return PROVIDERS.some((p) => p.id === saved) ? (saved as ProviderId) : 'claude';
  });
  // One key and one model name per provider, so switching never loses what was typed.
  const [keys, setKeys] = useState(() => fromStore(keyStore));
  const [models, setModels] = useState(() => fromStore(modelStore));
  const [answer, setAnswer] = useState('');
  const [answeredBy, setAnsweredBy] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortRef.current?.abort(), []);

  const provider = PROVIDERS.find((p) => p.id === providerId)!;
  const apiKey = keys[providerId];
  const model = models[providerId].trim() || provider.model;

  const pick = (id: ProviderId) => {
    setProviderId(id);
    setError('');
    write('raglens.provider', id);
  };

  const generate = async () => {
    const key = apiKey.trim();
    if (!key || busy) return;
    write(keyStore(providerId), key);
    write(modelStore(providerId), models[providerId].trim());
    abortRef.current?.abort();
    const controller = (abortRef.current = new AbortController());
    setAnswer('');
    setError('');
    setAnsweredBy(`${provider.label} · ${model}`);
    setBusy(true);
    try {
      await streamAnswer(
        provider,
        model,
        key,
        run.query,
        run.final.map((id, i) => ({ index: i + 1, pages: pagesOf(chunks[id]).slice(2), text: chunks[id].text })),
        (delta) => setAnswer((a) => a + delta),
        controller.signal,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      if (abortRef.current === controller) setBusy(false);
    }
  };

  const rendered = answer.split(/(\[\d+\])/g).map((part, i) => {
    const m = /^\[(\d+)\]$/.exec(part);
    const id = m ? run.final[Number(m[1]) - 1] : undefined;
    return id !== undefined ? (
      <span key={i} className="cite" onMouseEnter={() => onHover(id)} onMouseLeave={() => onHover(null)}>{m![1]}</span>
    ) : (
      part
    );
  });

  return (
    <div className="answer fade">
      <h4 className="col-title tone-final">Context sent to the LLM <small>top {run.final.length} chunks</small></h4>
      <div className="context-list">
        {run.final.map((id, i) => (
          <div key={id} className={`context${hover === id ? ' linked' : ''}`}
            onMouseEnter={() => onHover(id)} onMouseLeave={() => onHover(null)}>
            <span className="cite">{i + 1}</span>
            <div>
              <div className="card-id">Chunk {id + 1} · {pagesOf(chunks[id])}</div>
              <p>{chunks[id].text}</p>
            </div>
          </div>
        ))}
      </div>

      <div className="generate">
        <h4>Generate a grounded answer <small>optional</small></h4>
        <p className="hint">
          Everything above ran in your browser with no API key. To also see the generation step, pick a model
          provider and paste your own key for it.
        </p>
        <div className="seg providers" role="tablist" aria-label="Model provider">
          {PROVIDERS.map((p) => (
            <button key={p.id} role="tab" aria-selected={p.id === providerId} disabled={busy}
              className={p.id === providerId ? 'on' : ''} onClick={() => pick(p.id)}>
              {p.label}{keys[p.id].trim() && <i className="has-key" title="Key entered" />}
            </button>
          ))}
        </div>
        <label className="field">
          <span>Model</span>
          <input type="text" spellCheck={false} placeholder={provider.model} value={models[providerId]} autoComplete="off"
            onChange={(e) => setModels((m) => ({ ...m, [providerId]: e.target.value }))} />
        </label>
        <div className="key-row">
          <input type="password" aria-label={`${provider.vendor} API key`} placeholder={`${provider.vendor} API key (${provider.keyHint})`}
            value={apiKey} autoComplete="off"
            onChange={(e) => setKeys((k) => ({ ...k, [providerId]: e.target.value }))}
            onKeyDown={(e) => e.key === 'Enter' && generate()} />
          <button className="btn primary" onClick={generate} disabled={!apiKey.trim() || busy}>
            {busy ? 'Generating…' : 'Generate answer'}
          </button>
        </div>
        <p className="key-note">
          The key is sent only from this tab to {provider.host} and is forgotten when the tab closes.
        </p>
        {error && <p className="error-text">{error}</p>}
        {(answer || busy) && (
          <div className="answer-text">
            {rendered}
            {busy && <span className="caret" />}
            <div className="answer-foot">{answeredBy} · citations refer to the numbered chunks above</div>
          </div>
        )}
      </div>
    </div>
  );
}
