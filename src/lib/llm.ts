import Anthropic from '@anthropic-ai/sdk';
import { ANSWER_MODEL } from '../config';

export interface Excerpt {
  index: number;
  pages: string;
  text: string;
}

const SYSTEM = `You answer questions about a document the user uploaded, using only the numbered excerpts that a retrieval system selected from it.

Cite the excerpts you rely on inline, like [1] or [2][3]. If the excerpts do not contain the answer, say so plainly rather than guessing or drawing on outside knowledge. Keep the answer concise: a short paragraph, or a few bullet points when the answer is a list.`;

/**
 * Generation step of RAG: stream a grounded answer from Claude.
 * The key is supplied by the visitor and sent straight from their browser to the Anthropic API.
 */
export async function streamAnswer(
  apiKey: string,
  question: string,
  excerpts: Excerpt[],
  onText: (delta: string) => void,
  signal: AbortSignal,
): Promise<void> {
  const client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
  const context = excerpts
    .map((e) => `<excerpt index="${e.index}" pages="${e.pages}">\n${e.text}\n</excerpt>`)
    .join('\n\n');

  try {
    const stream = client.beta.messages.stream(
      {
        model: ANSWER_MODEL,
        max_tokens: 16000,
        output_config: { effort: 'low' },
        // If a safety classifier declines the request, the API retries it on a fallback model.
        betas: ['server-side-fallback-2026-07-01'],
        fallbacks: 'default',
        system: SYSTEM,
        messages: [{ role: 'user', content: `${context}\n\nQuestion: ${question}` }],
      },
      { signal },
    );
    for await (const event of stream) {
      if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
        onText(event.delta.text);
      }
    }
    const final = await stream.finalMessage();
    if (final.stop_reason === 'refusal') {
      throw new Error('The model declined to answer this question.');
    }
  } catch (err) {
    if (signal.aborted) return;
    if (err instanceof Anthropic.AuthenticationError) {
      throw new Error('That API key was rejected. Check it and try again.');
    }
    if (err instanceof Anthropic.PermissionDeniedError) {
      throw new Error('This API key does not have access to the model.');
    }
    if (err instanceof Anthropic.RateLimitError) {
      throw new Error('Rate limited by the API. Wait a moment and try again.');
    }
    if (err instanceof Anthropic.APIConnectionError) {
      throw new Error('Could not reach the Anthropic API. Check your connection.');
    }
    if (err instanceof Anthropic.APIError) {
      throw new Error(`API error ${err.status ?? ''}: ${err.message}`);
    }
    throw err;
  }
}
