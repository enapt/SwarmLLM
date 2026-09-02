import { createServer, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { OpenClawPluginApi, ProviderPlugin } from "openclaw/plugin-sdk/plugin-entry";
import { discoverOpenAICompatibleLocalModels } from "openclaw/plugin-sdk/provider-setup";
import {
  SWARMLLM_DEFAULT_API_KEY_ENV_VAR,
  SWARMLLM_DEFAULT_BASE_URL,
  SWARMLLM_DOCS_URL,
} from "./defaults.js";
import entry, { buildSwarmLlmUnknownModelHint } from "./index.js";

/**
 * A real node's `GET /v1/models` body. `context_length` is what OpenClaw's
 * discovery reads; `max_model_len` is the same figure under the name vLLM
 * clients read. The third entry has neither — a model whose declared context
 * the node could not read — which is the case the field exists to make rare.
 */
const MODELS_BODY = {
  object: "list",
  data: [
    {
      id: "llama-3.2-3b-instruct-q4-k-m",
      object: "model",
      created: 1785290835,
      owned_by: "local",
      max_model_len: 8192,
      context_length: 8192,
    },
    {
      id: "tinyllama-1.1b-chat-v1.0.q4-k-m",
      object: "model",
      created: 1784980707,
      owned_by: "network",
      max_model_len: 2048,
      context_length: 2048,
    },
    { id: "qwen2.5-14b-instruct-q4-k-m", object: "model", created: 1780000000, owned_by: "hybrid" },
  ],
};

const FIXTURE_KEY = "fixture-swarmllm-key";

function registerProvider(): ProviderPlugin {
  const providers: ProviderPlugin[] = [];
  const api = {
    registerProvider(provider: ProviderPlugin) {
      providers.push(provider);
    },
  } as Partial<OpenClawPluginApi>;
  entry.register(api as OpenClawPluginApi);
  expect(providers).toHaveLength(1);
  return providers[0]!;
}

describe("swarmllm provider registration", () => {
  it("registers under the id OpenClaw model refs use, with SwarmLLM's defaults", () => {
    const provider = registerProvider();
    expect(provider.id).toBe("swarmllm");
    expect(provider.label).toBe("SwarmLLM");
    expect(provider.envVars).toEqual([SWARMLLM_DEFAULT_API_KEY_ENV_VAR]);
    expect(provider.auth?.map((method) => method.id)).toEqual(["custom"]);
    expect(provider.wizard?.setup?.choiceId).toBe("swarmllm");
    expect(provider.catalog).toBeDefined();
    expect(provider.buildUnknownModelHint).toBe(buildSwarmLlmUnknownModelHint);
  });

  it("tells a user with an unresolvable model ref where the key and the model list are", () => {
    const hint = buildSwarmLlmUnknownModelHint();
    expect(hint).toContain(SWARMLLM_DEFAULT_API_KEY_ENV_VAR);
    expect(hint).toContain(`${SWARMLLM_DEFAULT_BASE_URL}/models`);
    expect(hint).toContain(SWARMLLM_DOCS_URL);
  });

  it("returns no catalog without a key, and the configured base URL with one", async () => {
    const provider = registerProvider();
    const run = (apiKey: string | undefined, config: Record<string, unknown>) =>
      provider.catalog!.run({
        config,
        env: {},
        resolveProviderApiKey: () => ({ apiKey, discoveryApiKey: apiKey }),
        resolveProviderAuth: () => ({ apiKey, mode: "api_key", source: "env" }),
      } as never);

    expect(await run(undefined, {})).toBeNull();

    const fresh = (await run(FIXTURE_KEY, {})) as { provider: Record<string, unknown> };
    expect(fresh.provider.baseUrl).toBe(SWARMLLM_DEFAULT_BASE_URL);
    expect(fresh.provider.api).toBe("openai-completions");
    expect(fresh.provider.apiKey).toBe(FIXTURE_KEY);

    // A hand-written provider block keeps discovery only when the config also
    // opts the provider's models in with a wildcard — otherwise its own
    // `models:` list is the catalog. Pinned because the README documents it.
    const configured = {
      models: { providers: { swarmllm: { baseUrl: "http://10.0.0.7:8800/v1/", models: [] } } },
      agents: { defaults: { models: { "swarmllm/*": {} } } },
    };
    const remote = (await run(FIXTURE_KEY, configured)) as { provider: Record<string, unknown> };
    expect(remote.provider.baseUrl).toBe("http://10.0.0.7:8800/v1");
    expect(
      await run(FIXTURE_KEY, { models: { providers: { swarmllm: { baseUrl: "http://10.0.0.7:8800/v1", models: [] } } } }),
    ).toBeNull();
  });
});

describe("model discovery against a SwarmLLM node", () => {
  let server: Server;
  let baseUrl: string;
  const seenAuthorization: Array<string | undefined> = [];

  beforeAll(async () => {
    server = createServer((req: IncomingMessage, res) => {
      if (req.method === "GET" && req.url === "/v1/models") {
        seenAuthorization.push(req.headers.authorization);
        // Every /v1 route on a node requires a Bearer key.
        if (req.headers.authorization !== `Bearer ${FIXTURE_KEY}`) {
          res.writeHead(401, { "content-type": "application/json" });
          res.end(JSON.stringify({ error: { message: "Missing or invalid API key", type: "authentication_error" } }));
          return;
        }
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(MODELS_BODY));
        return;
      }
      res.writeHead(404, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: "Unknown route" } }));
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1`;
  });

  afterAll(async () => {
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  });

  it("reads each model's real context window from context_length", async () => {
    const models = await discoverOpenAICompatibleLocalModels({
      baseUrl,
      apiKey: FIXTURE_KEY,
      label: "SwarmLLM",
      discoverRuntimeContext: false,
      env: {},
    });
    expect(Array.isArray(models)).toBe(true);
    const byId = new Map((models as Array<{ id: string; contextWindow?: number; maxTokens?: number; cost?: unknown }>).map((m) => [m.id, m]));
    expect([...byId.keys()]).toEqual([
      "llama-3.2-3b-instruct-q4-k-m",
      "tinyllama-1.1b-chat-v1.0.q4-k-m",
      "qwen2.5-14b-instruct-q4-k-m",
    ]);
    expect(byId.get("llama-3.2-3b-instruct-q4-k-m")?.contextWindow).toBe(8192);
    expect(byId.get("tinyllama-1.1b-chat-v1.0.q4-k-m")?.contextWindow).toBe(2048);
    // No context_length → OpenClaw's self-hosted default, which overstates a
    // node serving 8192 by 16x. That is why the node reports the field.
    expect(byId.get("qwen2.5-14b-instruct-q4-k-m")?.contextWindow).toBe(128000);
    expect(byId.get("llama-3.2-3b-instruct-q4-k-m")?.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
    expect(seenAuthorization.at(-1)).toBe(`Bearer ${FIXTURE_KEY}`);
  });

  it("sends no header for one of OpenClaw's non-secret markers, so the node's 401 empties the list", async () => {
    const models = await discoverOpenAICompatibleLocalModels({
      baseUrl,
      apiKey: "ollama-local",
      label: "SwarmLLM",
      discoverRuntimeContext: false,
      env: {},
    });
    expect(seenAuthorization.at(-1)).toBeUndefined();
    expect(models).toEqual([]);
  });
});
