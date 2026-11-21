import { tokenize } from '../lib/bm25';

/** A tiny corpus, small enough to see every number. Lines echo the app's sample (the US Constitution). */
export const DOCS = [
  'The Congress shall have power to declare war and to raise and support armies.',
  'The President shall be commander in chief of the army and navy of the United States.',
  'No person shall be a senator who shall not have attained the age of thirty years.',
  'Treason against the United States shall consist only in levying war against them.',
  'The Congress shall assemble at least once in every year.',
  'Congress debated war, war funding, war powers and the cost of war.',
  'The Congress shall have power to lay and collect taxes, duties, imposts and excises, to pay the debts and provide for the common defence and general welfare of the United States, to borrow money, to regulate commerce with foreign nations and among the several states, and to declare war.',
  'Each state shall appoint electors equal to the number of senators and representatives in Congress.',
];

export const QUERY = 'Congress declare war';
export const K1 = 1.5;
export const B = 0.75;

/** Terms used by the index and IDF scenes, from rare to common once sorted by df. */
export const SHOWCASE_TERMS = ['treason', 'declare', 'senator', 'power', 'united', 'war', 'state', 'congress'];

export function corpusStats(docs = DOCS) {
  const tokens = docs.map(tokenize);
  const lengths = tokens.map((t) => t.length);
  const avgdl = lengths.reduce((a, b) => a + b, 0) / docs.length;
  const counts = tokens.map((ts) => {
    const m = new Map<string, number>();
    for (const t of ts) m.set(t, (m.get(t) ?? 0) + 1);
    return m;
  });
  const tf = (term: string, d: number) => counts[d].get(term) ?? 0;
  const df = (term: string) => counts.filter((m) => m.has(term)).length;
  const idf = (term: string) => Math.log(1 + (docs.length - df(term) + 0.5) / (df(term) + 0.5));
  return { n: docs.length, tokens, lengths, avgdl, tf, df, idf };
}

/** The term-frequency part of BM25 (everything except IDF). */
export const tfPart = (tf: number, dl: number, avgdl: number, k1 = K1, b = B) => (tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * dl) / avgdl));
