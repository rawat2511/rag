import Anthropic from '@anthropic-ai/sdk';
import { ANSWER_MODEL } from '../config';

export interface Excerpt {
  index: number;
  pages: string;
  text: string;
}

export type ProviderId = 'claude' | 'openai' | 'gemini' | 'deepseek';

export interface Provider {
  id: ProviderId;
  /** Name on the picker. */
  label: string;
  /** Company whose API the key belongs to. */
  vendor: string;
  host: string;
  keyHint: string;
  /** Default model; the visitor can change it in the UI. */
  model: string;
}

export const PROVIDERS: Provider[] = [
  { id: 'claude', label: 'Claude', vendor: 'Anthropic', host: 'api.anthropic.com', keyHint: 'sk-ant-…', model: ANSWER_MODEL },
  { id: 'openai', label: 'ChatGPT', vendor: 'OpenAI', host: 'api.openai.com', keyHint: 'sk-…', model: 'gpt-5.6-terra' },
  { id: 'gemini', label: 'Gemini', vendor: 'Google', host: 'generativelanguage.googleapis.com', keyHint: 'AIza…', model: 'gemini-flash-latest' },
  { id: 'deepseek', label: 'DeepSeek', vendor: 'DeepSeek', host: 'api.deepseek.com', keyHint: 'sk-…', model: 'deepseek-flash' },
];

const SYSTEM = `You answer questions about a document the user uploaded, using only the numbered excerpts that a retrieval system selected from it.

Cite the excerpts you rely on inline, like [1] or [2][3]. If the excerpts do not contain the answer, say so plainly rather than guessing or drawing on outside knowledge. Keep the answer concise: a short paragraph, or a few bullet points when the answer is a list.`;

/**
 * Generation step of RAG: stream a grounded answer from the chosen provider.
 * The key is supplied by the visitor and sent straight from their browser to that provider's API.
 */
export async function streamAnswer(
  provider: Provider,
  model: string,
  apiKey: string,
  question: string,
  excerpts: Excerpt[],
  onText: (delta: string) => void,
  signal: AbortSignal,
): Promise<void> {
  const context = excerpts
    .map((e) => `<excerpt index="${e.index}" pages="${e.pages}">\n${e.text}\n</excerpt>`)
    .join('\n\n');
  const user = `${context}\n\nQuestion: ${question}`;
  try {
    if (provider.id === 'claude') await streamClaude(model, apiKey, user, onText, signal);
    else if (provider.id === 'gemini') await streamGemini(provider, model, apiKey, user, onText, signal);
    else await streamOpenAICompatible(provider, model, apiKey, user, onText, signal);
  } catch (err) {
    if (signal.aborted) return;
    throw err;
  }
}

/* ───────────────────────── Claude (Anthropic SDK) ───────────────────────── */

async function streamClaude(model: string, apiKey: string, user: string, onText: (d: string) => void, signal: AbortSignal) {
  const client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
  try {
    const stream = client.beta.messages.stream(
      {
        model,
        max_tokens: 16000,
        // Options tied to the default model are left off when the visitor picks another one.
        ...(model === ANSWER_MODEL
          ? {
              output_config: { effort: 'low' as const },
              // If a safety classifier declines the request, the API retries it on a fallback model.
              betas: ['server-side-fallback-2026-07-01'],
              fallbacks: 'default' as const,
            }
          : {}),
        system: SYSTEM,
        messages: [{ role: 'user', content: user }],
      },
      { signal },
    );
    for await (const event of stream) {
      if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') onText(event.delta.text);
    }
    const final = await stream.finalMessage();
    if (final.stop_reason === 'refusal') throw new Error('The model declined to answer this question.');
  } catch (err) {
    if (signal.aborted) throw err;
    if (err instanceof Anthropic.AuthenticationError) throw new Error('That API key was rejected. Check it and try again.');
    if (err instanceof Anthropic.PermissionDeniedError) throw new Error('This API key does not have access to the model.');
    if (err instanceof Anthropic.NotFoundError) throw new Error(`Anthropic does not recognise the model "${model}". Edit the model name and try again.`);
    if (err instanceof Anthropic.RateLimitError) throw new Error('Rate limited by the API. Wait a moment and try again.');
    if (err instanceof Anthropic.APIConnectionError) throw new Error('Could not reach the Anthropic API. Check your connection.');
    if (err instanceof Anthropic.APIError) throw new Error(`API error ${err.status ?? ''}: ${err.message}`);
    throw err;
  }
}

/* ───────────────────────── ChatGPT and DeepSeek (chat completions) ───────────────────────── */

const CHAT_URL: Partial<Record<ProviderId, string>> = {
  openai: 'https://api.openai.com/v1/chat/completions',
  deepseek: 'https://api.deepseek.com/chat/completions',
};

async function streamOpenAICompatible(p: Provider, model: string, apiKey: string, user: string, onText: (d: string) => void, signal: AbortSignal) {
  const res = await post(p, CHAT_URL[p.id]!, { Authorization: `Bearer ${apiKey}` }, {
    model,
    stream: true,
    messages: [
      { role: 'system', content: SYSTEM },
      { role: 'user', content: user },
    ],
  }, signal);
  await ensureOk(p, model, res);
  for await (const data of sse(res)) {
    if (data === '[DONE]') break;
    const json = parse(data);
    if (json?.error) throw new Error(`${p.vendor} API error: ${json.error.message ?? 'unknown error'}`);
    // Reasoning models also stream `reasoning_content`; only the answer text is shown.
    const delta = json?.choices?.[0]?.delta?.content;
    if (typeof delta === 'string' && delta) onText(delta);
  }
}

/* ───────────────────────── Gemini (generateContent) ───────────────────────── */

async function streamGemini(p: Provider, model: string, apiKey: string, user: string, onText: (d: string) => void, signal: AbortSignal) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`;
  const res = await post(p, url, { 'x-goog-api-key': apiKey }, {
    systemInstruction: { parts: [{ text: SYSTEM }] },
    contents: [{ role: 'user', parts: [{ text: user }] }],
  }, signal);
  await ensureOk(p, model, res);
  let any = false;
  let blocked = '';
  for await (const data of sse(res)) {
    const json = parse(data);
    if (json?.error) throw new Error(`Google API error: ${json.error.message ?? 'unknown error'}`);
    if (json?.promptFeedback?.blockReason) blocked = json.promptFeedback.blockReason;
    for (const part of json?.candidates?.[0]?.content?.parts ?? []) {
      if (typeof part.text === 'string' && part.text && !part.thought) {
        any = true;
        onText(part.text);
      }
    }
  }
  if (!any) throw new Error(blocked ? `Gemini blocked this request (${blocked}).` : 'Gemini returned an empty answer.');
}

/* ───────────────────────── Shared HTTP helpers ───────────────────────── */

/* eslint-disable @typescript-eslint/no-explicit-any */
function parse(data: string): any {
  try { return JSON.parse(data); } catch { return null; }
}

async function post(p: Provider, url: string, headers: Record<string, string>, body: unknown, signal: AbortSignal): Promise<Response> {
  try {
    return await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify(body),
      signal,
    });
  } catch (err) {
    if (signal.aborted) throw err;
    // OpenAI omits CORS headers on a 401, so the browser reports a rejected key as a network failure.
    if (p.id === 'openai') {
      throw new Error('No readable response from the OpenAI API. This usually means the API key was rejected; otherwise check your connection.');
    }
    throw new Error(`Could not reach the ${p.vendor} API. Check your connection.`);
  }
}

async function ensureOk(p: Provider, model: string, res: Response): Promise<void> {
  if (res.ok) return;
  const raw = await res.text().catch(() => '');
  const json = parse(raw);
  // OpenAI and DeepSeek return {error}; Google returns {error} or [{error}].
  const message: string = (Array.isArray(json) ? json[0] : json)?.error?.message ?? '';
  if (res.status === 401 || res.status === 403 || /api key not valid|invalid api key|incorrect api key/i.test(message)) {
    throw new Error(`That ${p.vendor} API key was rejected. Check it and try again.`);
  }
  if (res.status === 404 || /model.*(not found|not exist|not supported)/i.test(message)) {
    throw new Error(`${p.vendor} does not recognise the model "${model}". Edit the model name and try again.`);
  }
  if (res.status === 402) throw new Error(`The ${p.vendor} account has no remaining balance.`);
  if (res.status === 429) throw new Error(`Rate limited or out of quota at ${p.vendor}. ${message}`.trim());
  throw new Error(`${p.vendor} API error ${res.status}${message ? `: ${message}` : ''}`);
}

/** Yield the `data:` payload of each server-sent event in a streaming response. */
async function* sse(res: Response): AsyncGenerator<string> {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, nl).replace(/\r$/, '');
        buffer = buffer.slice(nl + 1);
        if (line.startsWith('data:')) yield line.slice(5).trim();
      }
    }
    if (buffer.startsWith('data:')) yield buffer.slice(5).trim();
  } finally {
    reader.cancel().catch(() => {});
  }
}
