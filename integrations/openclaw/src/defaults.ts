// What a fresh SwarmLLM node looks like from the outside. Kept in one place so
// the plugin entry, its tests and the README cannot drift apart.

/** Where a node serves its OpenAI-compatible API by default (`swarmllm run -p 8800`). */
export const SWARMLLM_DEFAULT_BASE_URL = "http://127.0.0.1:8800/v1";

export const SWARMLLM_PROVIDER_LABEL = "SwarmLLM";

/**
 * The same variable the daemon itself reads as its API-key override, so one
 * `export SWARMLLM_API_KEY=…` serves both sides. A real key is required:
 * OpenClaw's non-secret local markers send no `Authorization` header at all,
 * and every `/v1` route on a node refuses a request without one.
 */
export const SWARMLLM_DEFAULT_API_KEY_ENV_VAR = "SWARMLLM_API_KEY";

/** A model most swarms carry; only a placeholder — discovery lists the real ones. */
export const SWARMLLM_MODEL_PLACEHOLDER = "meta-llama-3.1-8b-instruct-q4-k-m";

/** The README section that explains the two things a node needs before an agent can use it. */
export const SWARMLLM_DOCS_URL = "https://github.com/enapt/SwarmLLM#use-it-as-an-api";
