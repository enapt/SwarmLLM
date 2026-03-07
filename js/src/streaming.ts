import type { ChatCompletionChunk } from "./types";

/**
 * Parse an SSE stream from a fetch Response into an async iterable of ChatCompletionChunk.
 * Works in both browser (ReadableStream) and Node.js 18+ (also ReadableStream).
 */
export async function* parseSSEStream(
  response: Response
): AsyncIterable<ChatCompletionChunk> {
  const body = response.body;
  if (!body) {
    return;
  }

  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      // Keep incomplete last line in buffer
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith(":")) continue;

        if (trimmed.startsWith("data: ")) {
          const data = trimmed.slice(6);
          if (data === "[DONE]") return;

          try {
            yield JSON.parse(data) as ChatCompletionChunk;
          } catch {
            // Skip malformed JSON lines
          }
        }
      }
    }

    // Process any remaining data in buffer
    if (buffer.trim()) {
      const trimmed = buffer.trim();
      if (trimmed.startsWith("data: ")) {
        const data = trimmed.slice(6);
        if (data !== "[DONE]") {
          try {
            yield JSON.parse(data) as ChatCompletionChunk;
          } catch {
            // Skip malformed JSON
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
