import { parseSSEStream } from "./streaming";
import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
  Model,
  ModelList,
  NodeStatus,
  Peer,
  Stats,
  SwarmLLMClientOptions,
} from "./types";
import { SwarmLLMError } from "./types";

/**
 * The error a failed response stands for.
 *
 * The server answers every failure in OpenAI's envelope,
 * `{"error": {"message", "type", "param", "code"}}`. `String(body.error)` made
 * that "[object Object]", so the message is read from inside it, as the OpenAI
 * SDKs do; a bare string under `error` is still accepted. The body is read once
 * as text: after a failed `response.json()` the stream is spent, and the
 * `response.text()` fallback threw instead of reporting the error.
 */
async function errorFrom(response: Response): Promise<SwarmLLMError> {
  const text = await response.text().catch(() => "");
  let body: unknown = text;
  try {
    body = JSON.parse(text);
  } catch {
    // Not JSON: the text is the message.
  }
  let msg = text || response.statusText;
  if (typeof body === "object" && body !== null && "error" in body) {
    const err = (body as Record<string, unknown>).error;
    if (typeof err === "string" && err) {
      msg = err;
    } else if (typeof err === "object" && err !== null) {
      const inner = (err as Record<string, unknown>).message;
      if (typeof inner === "string" && inner) msg = inner;
    }
  }
  return new SwarmLLMError(response.status, msg, body);
}

export class SwarmLLMClient {
  private baseUrl: string;
  private apiKey?: string;
  private timeout: number;

  /** Admin endpoints for node management. */
  admin: AdminClient;

  constructor(options: SwarmLLMClientOptions = {}) {
    this.baseUrl = (options.baseUrl || "http://localhost:8800").replace(
      /\/$/,
      ""
    );
    this.apiKey = options.apiKey;
    this.timeout = options.timeout ?? 120_000;
    this.admin = new AdminClient(this);
  }

  // ---- Internal helpers ----

  /** @internal */
  async _fetch(path: string, init: RequestInit = {}): Promise<Response> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(init.headers as Record<string, string>),
    };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        headers,
        signal: controller.signal,
      });
      return response;
    } finally {
      clearTimeout(timer);
    }
  }

  /** @internal */
  async _request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await this._fetch(path, init);
    if (!response.ok) {
      throw await errorFrom(response);
    }
    return response.json() as Promise<T>;
  }

  // ---- Chat Completions ----

  /**
   * Create a chat completion (non-streaming).
   *
   * ```ts
   * const res = await client.chat({
   *   model: 'TinyLlama-1.1B',
   *   messages: [{ role: 'user', content: 'Hello!' }],
   * });
   * console.log(res.choices[0].message.content);
   * ```
   */
  async chat(
    params: ChatCompletionRequest
  ): Promise<ChatCompletionResponse> {
    return this._request<ChatCompletionResponse>("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify({ ...params, stream: false }),
    });
  }

  /**
   * Create a streaming chat completion. Returns an async iterable of chunks.
   *
   * ```ts
   * for await (const chunk of client.chatStream({
   *   model: 'TinyLlama-1.1B',
   *   messages: [{ role: 'user', content: 'Hello!' }],
   * })) {
   *   process.stdout.write(chunk.choices[0]?.delta?.content || '');
   * }
   * ```
   */
  async *chatStream(
    params: ChatCompletionRequest
  ): AsyncIterable<ChatCompletionChunk> {
    const response = await this._fetch("/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify({ ...params, stream: true }),
    });

    if (!response.ok) {
      throw await errorFrom(response);
    }

    yield* parseSSEStream(response);
  }

  // ---- Models ----

  /** List available models. */
  async listModels(): Promise<Model[]> {
    const data = await this._request<ModelList>("/v1/models");
    return data.data;
  }

  // ---- Status ----

  /** Get node status. */
  async status(): Promise<NodeStatus> {
    return this._request<NodeStatus>("/v1/status");
  }

  /** Health check — returns "ok" if node is running. */
  async health(): Promise<string> {
    const response = await this._fetch("/health");
    return response.text();
  }
}

class AdminClient {
  constructor(private client: SwarmLLMClient) {}

  /** Get node statistics. */
  async stats(): Promise<Stats> {
    return this.client._request<Stats>("/api/admin/stats");
  }

  /** List connected peers. */
  async peers(): Promise<Peer[]> {
    const data = await this.client._request<{ peers: Peer[] }>(
      "/api/admin/peers"
    );
    return data.peers;
  }
}
