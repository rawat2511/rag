import * as pdfjs from 'pdfjs-dist';
import workerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import { MAX_PAGES } from '../config';
import type { PageText } from './types';

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

const WORDS_PER_TEXT_PAGE = 400;

function clean(text: string): string {
  return text
    .replace(/(\p{L})-\n(\p{Ll})/gu, '$1$2') // re-join words hyphenated across lines
    .replace(/\s+/g, ' ')
    .trim();
}

/** Extract text page by page from a PDF. Rejects documents over MAX_PAGES. */
export async function extractPdf(
  data: ArrayBuffer,
  onProgress: (done: number, total: number) => void,
): Promise<PageText[]> {
  const task = pdfjs.getDocument({ data });
  const doc = await task.promise;
  try {
    if (doc.numPages > MAX_PAGES) {
      throw new Error(`This PDF has ${doc.numPages} pages. The limit is ${MAX_PAGES} pages.`);
    }
    const pages: PageText[] = [];
    for (let i = 1; i <= doc.numPages; i++) {
      const page = await doc.getPage(i);
      const content = await page.getTextContent();
      let text = '';
      for (const item of content.items) {
        if ('str' in item) text += item.str + (item.hasEOL ? '\n' : '');
      }
      page.cleanup();
      pages.push({ page: i, text: clean(text) });
      onProgress(i, doc.numPages);
    }
    if (!pages.some((p) => p.text.length > 0)) {
      throw new Error('No selectable text found. This looks like a scanned PDF, which needs OCR first.');
    }
    return pages;
  } finally {
    void task.destroy();
  }
}

/** Plain text / Markdown has no pages, so split it into ~400-word pseudo-pages. */
export function extractText(raw: string): PageText[] {
  const words = raw.split(/\s+/).filter(Boolean);
  if (words.length === 0) throw new Error('The file is empty.');
  const count = Math.ceil(words.length / WORDS_PER_TEXT_PAGE);
  if (count > MAX_PAGES) {
    throw new Error(
      `This file is about ${count} pages long (${words.length.toLocaleString()} words). The limit is ${MAX_PAGES} pages.`,
    );
  }
  const pages: PageText[] = [];
  for (let i = 0; i < count; i++) {
    pages.push({
      page: i + 1,
      text: words.slice(i * WORDS_PER_TEXT_PAGE, (i + 1) * WORDS_PER_TEXT_PAGE).join(' '),
    });
  }
  return pages;
}
