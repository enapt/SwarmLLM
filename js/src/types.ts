// ---- Chat Completion Request ----

export interface ChatCompletionRequest {
  model: string;
  messages: Message[];
  stream?: boolean;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  stop?: string | string[];
  frequency_penalty?: number;
  presence_penalty?: number;
  session_id?: string;
  tools?: Tool[];
  response_format?: ResponseFormat;
}

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string | ContentPart[];
  tool_call_id?: string;
  tool_calls?: ToolCall[];
}

export interface ContentPart {
  type: "text" | "image_url";
  text?: string;
  image_url?: { url: string; detail?: "auto" | "low" | "high" };
}

export interface Tool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface ResponseFormat {
  type: "text" | "json_object";
}

// ---- Chat Completion Response ----

export interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: Choice[];
  usage: Usage;
  session_id?: string;
}

export interface Choice {
  index: number;
  message: Message;
  finish_reason: "stop" | "length" | "tool_calls" | null;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

// ---- Streaming ----

export interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: ChunkChoice[];
}

export interface ChunkChoice {
  index: number;
  delta: Delta;
  finish_reason: "stop" | "length" | "tool_calls" | null;
}

export interface Delta {
  role?: "assistant";
  content?: string;
  tool_calls?: ToolCall[];
}

// ---- Models ----

export interface Model {
  id: string;
  object: "model";
  owned_by: string;
}

export interface ModelList {
  object: "list";
  data: Model[];
}

// ---- Node Status ----

export interface NodeStatus {
  node_id: string;
  version: string;
  uptime_seconds: number;
  peers_connected: number;
  credits_balance: number;
  credit_tier: string;
  hosted_shards: number;
  [key: string]: unknown;
}

// ---- Admin ----

export interface Stats {
  node_id: string;
  version: string;
  uptime_seconds: number;
  peers_connected: number;
  credits_balance: number;
  credit_tier: string;
  hosted_shards: number;
  [key: string]: unknown;
}

export interface Peer {
  node_id: string;
  healthy: boolean;
  latency_ms?: number;
  trust_score?: number;
  gpu?: string;
  hosted_models?: string[];
  [key: string]: unknown;
}

// ---- Client Options ----

export interface SwarmLLMClientOptions {
  baseUrl?: string;
  apiKey?: string;
  timeout?: number;
}

// ---- Error ----

export class SwarmLLMError extends Error {
  status: number;
  body: unknown;

  constructor(status: number, message: string, body?: unknown) {
    super(`SwarmLLM API error (${status}): ${message}`);
    this.name = "SwarmLLMError";
    this.status = status;
    this.body = body;
  }
}
