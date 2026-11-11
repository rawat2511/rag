export interface PageText {
  page: number;
  text: string;
}

export interface Chunk {
  id: number;
  text: string;
  pageStart: number;
  pageEnd: number;
  words: number;
  /** Leading characters repeated from the end of the previous chunk. */
  overlapChars: number;
}

export interface Hit {
  id: number;
  score: number;
  rank: number;
}

export interface SparseHit extends Hit {
  terms: { term: string; score: number }[];
}

export interface FusedHit extends Hit {
  denseRank?: number;
  sparseRank?: number;
}

export interface RerankedHit {
  id: number;
  /** Cross-encoder relevance (sigmoid of the logit); undefined while still being scored. */
  score?: number;
  rank: number;
  prevRank: number;
}

export interface QueryTerm {
  raw: string;
  term: string;
  kept: boolean;
  df: number;
  idf: number;
}

export type RetrievalMode = 'hybrid' | 'dense' | 'bm25';

export interface QueryRun {
  id: number;
  query: string;
  /** 0 embedding query · 1 query analysed · 2 retrieved · 3 fused · 4 re-ranking · 5 done */
  stage: number;
  mode: RetrievalMode;
  useRerank: boolean;
  terms: QueryTerm[];
  qvec?: Float32Array;
  qxy?: [number, number];
  dense: Hit[];
  sparse: SparseHit[];
  fused: FusedHit[];
  reranked: RerankedHit[];
  final: number[];
  timings: Record<string, number>;
}

export type Device = 'webgpu' | 'wasm';
