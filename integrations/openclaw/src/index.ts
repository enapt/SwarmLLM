// OpenClaw provider plugin for SwarmLLM — a peer-to-peer inference network
// exposed as an OpenAI-compatible server on the user's own machine.
//
// Built on the same SDK helper OpenClaw's bundled vLLM and SGLang providers
// use, so it inherits their onboarding wizard entry, non-interactive setup
// flags, and live model discovery from `GET <baseUrl>/models`. A SwarmLLM
// node reports each model's usable context under `context_length`, which is
// the field that discovery reads; without it OpenClaw would assume a 128k
// window and every agent turn would be refused as too long.
import {
  buildProviderReplayFamilyHooks,
  defineSelfHostedOpenAICompatibleProvider,
} from "openclaw/plugin-sdk/provider-model-shared";
import {
  SWARMLLM_DEFAULT_API_KEY_ENV_VAR,
  SWARMLLM_DEFAULT_BASE_URL,
  SWARMLLM_DOCS_URL,
  SWARMLLM_MODEL_PLACEHOLDER,
  SWARMLLM_PROVIDER_LABEL,
} from "./defaults.js";

/**
 * Shown when a `swarmllm/<model>` ref cannot be resolved. The two causes in
 * practice are a missing key (discovery never ran) and a model the swarm does
 * not currently cover — both are explained at the README anchor.
 */
export function buildSwarmLlmUnknownModelHint(): string {
  return (
    `${SWARMLLM_PROVIDER_LABEL} lists the models it can serve at ${SWARMLLM_DEFAULT_BASE_URL}/models. ` +
    `Set ${SWARMLLM_DEFAULT_API_KEY_ENV_VAR} to the key in the node's data directory (Settings → Access Token), ` +
    `or run "openclaw configure", then pick a listed model. See ${SWARMLLM_DOCS_URL}`
  );
}

export default defineSelfHostedOpenAICompatibleProvider({
  id: "swarmllm",
  label: SWARMLLM_PROVIDER_LABEL,
  hint: "Free models from a peer-to-peer swarm on your own machine",
  groupHint: "Peer-to-peer inference network",
  defaultBaseUrl: SWARMLLM_DEFAULT_BASE_URL,
  apiKeyEnvVar: SWARMLLM_DEFAULT_API_KEY_ENV_VAR,
  modelPlaceholder: SWARMLLM_MODEL_PLACEHOLDER,
  overrides: {
    ...buildProviderReplayFamilyHooks({
      family: "openai-compatible",
      dropReasoningFromHistory: false,
    }),
    buildUnknownModelHint: buildSwarmLlmUnknownModelHint,
  },
});
