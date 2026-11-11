import type { QueryTerm, SparseHit } from './types';

const STOPWORDS = new Set(
  (
    'a an and are as at be but by for if in into is it its no not of on or such that the their then there these ' +
    'they this to was will with from have has had were been which who whom what when where how why do does did ' +
    'can could should would may might shall must than so we you your i he she him her his our us am about any all'
  ).split(' '),
);

/** Lowercase, split on non-alphanumerics, and fold simple plurals ("senators" -> "senator"). */
function normalise(word: string): string {
  const w = word.toLowerCase();
  return w.length > 3 && w.endsWith('s') && !w.endsWith('ss') ? w.slice(0, -1) : w;
}

const splitWords = (text: string) => text.match(/[\p{L}\p{N}]+/gu) ?? [];

export function tokenize(text: string): string[] {
  const out: string[] = [];
  for (const raw of splitWords(text)) {
    const lower = raw.toLowerCase();
    if (lower.length < 2 || STOPWORDS.has(lower)) continue;
    out.push(normalise(raw));
  }
  return out;
}

interface Posting {
  ids: number[];
  tfs: number[];
}

/** Okapi BM25 over an in-memory inverted index. */
export class BM25 {
  readonly n: number;
  readonly avgdl: number;
  private readonly lengths: Uint32Array;
  private readonly index = new Map<string, Posting>();

  constructor(
    docs: string[],
    private readonly k1 = 1.5,
    private readonly b = 0.75,
  ) {
    this.n = docs.length;
    this.lengths = new Uint32Array(docs.length);
    let total = 0;
    docs.forEach((doc, id) => {
      const tokens = tokenize(doc);
      this.lengths[id] = tokens.length;
      total += tokens.length;
      const counts = new Map<string, number>();
      for (const t of tokens) counts.set(t, (counts.get(t) ?? 0) + 1);
      for (const [term, tf] of counts) {
        let posting = this.index.get(term);
        if (!posting) this.index.set(term, (posting = { ids: [], tfs: [] }));
        posting.ids.push(id);
        posting.tfs.push(tf);
      }
    });
    this.avgdl = total / Math.max(1, docs.length);
  }

  get vocabularySize(): number {
    return this.index.size;
  }

  df(term: string): number {
    return this.index.get(term)?.ids.length ?? 0;
  }

  idf(term: string): number {
    const df = this.df(term);
    return Math.log(1 + (this.n - df + 0.5) / (df + 0.5));
  }

  /** Show how the query is tokenised, including the words that were dropped. */
  analyze(query: string): QueryTerm[] {
    return splitWords(query).map((raw) => {
      const lower = raw.toLowerCase();
      const kept = lower.length >= 2 && !STOPWORDS.has(lower);
      const term = normalise(raw);
      return { raw, term, kept, df: kept ? this.df(term) : 0, idf: kept ? this.idf(term) : 0 };
    });
  }

  search(query: string, k: number): SparseHit[] {
    const terms = [...new Set(tokenize(query))];
    const scores = new Float32Array(this.n);
    const breakdown = new Map<number, { term: string; score: number }[]>();
    for (const term of terms) {
      const posting = this.index.get(term);
      if (!posting) continue;
      const idf = this.idf(term);
      for (let i = 0; i < posting.ids.length; i++) {
        const id = posting.ids[i];
        const tf = posting.tfs[i];
        const norm = 1 - this.b + this.b * (this.lengths[id] / this.avgdl);
        const s = idf * ((tf * (this.k1 + 1)) / (tf + this.k1 * norm));
        scores[id] += s;
        let list = breakdown.get(id);
        if (!list) breakdown.set(id, (list = []));
        list.push({ term, score: s });
      }
    }
    return [...breakdown.keys()]
      .sort((a, b) => scores[b] - scores[a])
      .slice(0, k)
      .map((id, i) => ({
        id,
        score: scores[id],
        rank: i + 1,
        terms: breakdown.get(id)!.sort((a, b) => b.score - a.score),
      }));
  }
}
