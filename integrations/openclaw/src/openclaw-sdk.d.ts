// Type declarations for the two plugin-sdk subpaths this plugin uses.
//
// The published `openclaw` package ships `.d.ts` files for
// `plugin-sdk/plugin-entry` (all the plugin and provider types) but not for
// `plugin-sdk/provider-model-shared` or `plugin-sdk/provider-setup`, so without
// these tsc sees an implicit `any` and refuses under `strict`. The shapes are
// transcribed from OpenClaw's source at the version pinned in package.json;
// they add nothing the runtime does not already have.

declare module "openclaw/plugin-sdk/provider-model-shared" {
  import type { definePluginEntry, ProviderPlugin } from "openclaw/plugin-sdk/plugin-entry";

  export type SelfHostedOpenAICompatibleProviderOverrides = Partial<
    Omit<ProviderPlugin, "id" | "label" | "docsPath" | "envVars" | "auth" | "catalog" | "wizard">
  >;

  export type SelfHostedOpenAICompatibleProviderOptions = {
    id: string;
    label: string;
    hint: string;
    groupHint: string;
    defaultBaseUrl: string;
    apiKeyEnvVar: string;
    modelPlaceholder: string;
    overrides?: SelfHostedOpenAICompatibleProviderOverrides;
  };

  /** One self-hosted OpenAI-compatible endpoint: auth prompt, discovery from `/models`, wizard entry. */
  export function defineSelfHostedOpenAICompatibleProvider(
    options: SelfHostedOpenAICompatibleProviderOptions,
  ): ReturnType<typeof definePluginEntry>;

  export type BuildProviderReplayFamilyHooksOptions = {
    family: "openai-compatible" | "anthropic" | "google-gemini" | (string & {});
    dropReasoningFromHistory?: boolean;
    sanitizeToolCallIds?: boolean;
    duplicateToolCallIdStyle?: string;
  };

  /** How a provider family replays conversation history; returns the matching ProviderPlugin hooks. */
  export function buildProviderReplayFamilyHooks(
    options: BuildProviderReplayFamilyHooksOptions,
  ): SelfHostedOpenAICompatibleProviderOverrides;
}

declare module "openclaw/plugin-sdk/provider-setup" {
  export type OpenAICompatibleLocalModelsParams = {
    baseUrl: string;
    serverBaseUrl?: string;
    apiKey?: string;
    headers?: Record<string, string>;
    label: string;
    healthPath?: string;
    modelsPathOrder?: "inference" | "server-first";
    routerModelProps?: boolean;
    contextWindow?: number;
    maxTokens?: number;
    discoverRuntimeContext?: boolean;
    timeoutMs?: number;
    propsTimeoutMs?: number;
    signal?: AbortSignal;
    env?: Record<string, string | undefined>;
    rawResult?: boolean;
  };

  export type DiscoveredModel = {
    id: string;
    name: string;
    reasoning: boolean;
    input: Array<"text" | "image">;
    cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
    contextWindow: number;
    maxTokens: number;
    contextTokens?: number;
  };

  /** Probe `<baseUrl>/models` and turn each row into a model entry; `[]` when unreachable or unauthorised. */
  export function discoverOpenAICompatibleLocalModels(
    params: OpenAICompatibleLocalModelsParams,
  ): Promise<DiscoveredModel[] | { kind: string }>;
}
