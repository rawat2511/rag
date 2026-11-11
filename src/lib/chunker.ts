import type { Chunk, PageText } from './types';

interface Unit {
  text: string;
  page: number;
  words: number;
}

const countWords = (s: string) => (s.match(/\S+/g) ?? []).length;

/** Split pages into sentence-sized units; oversized "sentences" are cut into word windows. */
function toUnits(pages: PageText[], maxWords: number): Unit[] {
  const units: Unit[] = [];
  for (const { page, text } of pages) {
    const sentences = text.match(/[^.!?]+(?:[.!?]+["')\]]*|$)/g) ?? [];
    for (const raw of sentences) {
      const s = raw.trim();
      if (!s) continue;
      const words = s.split(/\s+/);
      if (words.length <= maxWords) {
        units.push({ text: s, page, words: words.length });
      } else {
        for (let i = 0; i < words.length; i += maxWords) {
          const part = words.slice(i, i + maxWords);
          units.push({ text: part.join(' '), page, words: part.length });
        }
      }
    }
  }
  return units;
}

/**
 * Sentence-aware chunking: pack whole sentences until the target size is reached,
 * then start the next chunk with the trailing sentences of the previous one (overlap)
 * so that ideas spanning a chunk boundary stay retrievable.
 */
export function chunkPages(pages: PageText[], size: number, overlap: number): Chunk[] {
  const units = toUnits(pages, size);
  const chunks: Chunk[] = [];
  let current: Unit[] = [];
  let currentWords = 0;
  let carried = 0; // how many units at the start of `current` came from the previous chunk

  const emit = () => {
    const overlapText = current.slice(0, carried).map((u) => u.text).join(' ');
    const text = current.map((u) => u.text).join(' ');
    chunks.push({
      id: chunks.length,
      text,
      pageStart: current[0].page,
      pageEnd: current[current.length - 1].page,
      words: currentWords,
      overlapChars: overlapText.length,
    });
  };

  for (const unit of units) {
    const hasFresh = current.length > carried;
    if (hasFresh && currentWords + unit.words > size) {
      emit();
      const tail: Unit[] = [];
      let tailWords = 0;
      for (let i = current.length - 1; i >= 0; i--) {
        if (tailWords + current[i].words > overlap) break;
        tail.unshift(current[i]);
        tailWords += current[i].words;
      }
      current = tail;
      currentWords = tailWords;
      carried = tail.length;
    }
    current.push(unit);
    currentWords += unit.words;
  }
  if (current.length > carried) emit();
  return chunks;
}

export { countWords };
