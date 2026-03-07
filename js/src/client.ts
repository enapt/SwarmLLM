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
      let body: unknown;
      try {
        body = await response.json();
      } catch {
        body = await response.text();
      }
      const msg =
        typeof body === "object" && body !== null && "error" in body
          ? String((body as Record<string, unknown>).error)
          : String(body);
      throw new SwarmLLMError(response.status, msg, body);
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
      let body: unknown;
      try {
        body = await response.json();
      } catch {
        body = await response.text();
      }
      const msg =
        typeof body === "object" && body !== null && "error" in body
          ? String((body as Record<string, unknown>).error)
          : String(body);
      throw new SwarmLLMError(response.status, msg, body);
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
