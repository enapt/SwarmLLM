'use strict';

// ============================================================================
// SwarmLLM Providers — single source of truth for provider/model metadata
// ============================================================================

var _ICON_BASE = '/static/icons/';

// Maps provider keys and local model families to icon file base names.
var _ICON_MAP = {
  // Cloud providers
  openai:     'openai',
  anthropic:  'anthropic',
  claude_subscription: 'claude-code',
  deepseek:   'deepseek-color',
  mistral:    'mistral-color',
  groq:       'groq',
  nvidia_nim: 'nvidia-color',
  cerebras:   'cerebras-color',
  sambanova:  'sambanova-color',
  fireworks:  'fireworks-color',
  together:   'together-color',
  deepinfra:  'deepinfra-color',
  moonshot:   'moonshot',
  // Local / swarm model families
  llama:      'meta-color',
  gemma:      'gemma-color',
  gemini:     'gemini-color',
  qwen:       'qwen-color',
  phi:        'microsoft-color',
  claude:     'claude-color',
};

// Canonical display names — used in dropdowns, cards, and badges everywhere.
// Previously duplicated as providerLabels (×3) and providerDisplayNames (×1).
var PROVIDER_NAMES = {
  anthropic:  'Anthropic',
  claude_subscription: 'Claude Code',
  openai:     'OpenAI',
  deepseek:   'DeepSeek',
  mistral:    'Mistral',
  groq:       'Groq',
  nvidia_nim: 'NVIDIA NIM',
  cerebras:   'Cerebras',
  sambanova:  'SambaNova',
  fireworks:  'Fireworks AI',
  together:   'Together AI',
  deepinfra:  'DeepInfra',
  moonshot:   'Moonshot (Kimi)',
};

// Default test models per provider. Used by setup wizard + settings "Test"
// button to validate an API key with the cheapest credible model.
var PROVIDER_TEST_MODELS = {
  openai:     'gpt-4o-mini',
  deepseek:   'deepseek-chat',
  mistral:    'mistral-small-latest',
  groq:       'llama-3.1-8b-instant',
  nvidia_nim: 'meta/llama-3.1-8b-instruct',
  cerebras:   'cerebras:llama-3.1-8b',
  sambanova:  'sambanova:Meta-Llama-3.3-70B-Instruct',
  fireworks:  'accounts/fireworks/models/llama-v3p3-70b-instruct',
  together:   'together:meta-llama/Llama-3.3-70B-Instruct-Turbo',
  deepinfra:  'deepinfra:meta-llama/Llama-3.3-70B-Instruct',
  moonshot:   'moonshot-v1-8k',
};

// Signup / API-key URLs for each provider. Single source of truth — consumed
// by both the setup wizard (`init.js`) and the settings panel (populated into
// `.provider-signup-link[data-provider=X]` anchors at init time).
var PROVIDER_SIGNUP_URLS = {
  anthropic:  'https://console.anthropic.com/settings/keys',
  openai:     'https://platform.openai.com/api-keys',
  deepseek:   'https://platform.deepseek.com/api_keys',
  mistral:    'https://console.mistral.ai/api-keys',
  groq:       'https://console.groq.com/keys',
  nvidia_nim: 'https://build.nvidia.com/',
  cerebras:   'https://cloud.cerebras.ai/',
  sambanova:  'https://cloud.sambanova.ai/',
  fireworks:  'https://fireworks.ai/account/api-keys',
  together:   'https://api.together.xyz/settings/api-keys',
  deepinfra:  'https://deepinfra.com/dash/api_keys',
  moonshot:   'https://platform.moonshot.cn/console/api-keys',
};

// Ordered list of all supported cloud provider keys.

// Prebuilt <img> HTML strings for each provider (16px, avoids repeated DOM creation).
var _providerIconCache = {};

function providerIconUrl(key) {
  var id = _ICON_MAP[key];
  return id ? _ICON_BASE + id + '.svg' : null;
}

// Returns an <img> tag string for a provider/model icon, or '' if unknown.
function providerIconHtml(key, size) {
  var url = providerIconUrl(key);
  if (!url) return '';
  size = size || 16;
  var cacheKey = key + ':' + size;
  if (!_providerIconCache[cacheKey]) {
    _providerIconCache[cacheKey] = '<img src="' + url + '" width="' + size + '" height="' + size +
      '" alt="" aria-hidden="true" class="provider-icon" style="display:inline-block;vertical-align:middle;flex-shrink:0">';
  }
  return _providerIconCache[cacheKey];
}

// Subscription providers — models accessed via CLI/subscription, not API key billing.
// Add new subscription providers here as they are introduced.
var SUBSCRIPTION_PROVIDERS = {
  claude_subscription: true,
};

function isSubscriptionProvider(key) {
  return !!SUBSCRIPTION_PROVIDERS[key];
}

// Infer a provider/family key from a model ID string.
function modelIconKey(modelId) {
  if (!modelId) return null;
  var m = modelId.toLowerCase();
  if (m.startsWith('gpt') || m.startsWith('o1') || m.startsWith('o3') || m.startsWith('o4') || m.includes('-openai')) return 'openai';
  if (m.startsWith('claude')) return 'claude';
  if (m.startsWith('deepseek')) return 'deepseek';
  if (m.startsWith('mistral') || m.startsWith('mixtral') || m.startsWith('codestral')) return 'mistral';
  if (m.startsWith('llama') || m.startsWith('meta-llama') || m.startsWith('tinyllama')) return 'llama';
  if (m.startsWith('llava')) return 'llama';
  if (m.startsWith('gemma')) return 'gemma';
  if (m.startsWith('gemini')) return 'gemini';
  if (m.startsWith('qwen')) return 'qwen';
  if (m.startsWith('phi')) return 'phi';
  if (m.startsWith('starcoder') || m.startsWith('codegen')) return 'code';
  if (m.startsWith('yi')) return 'yi';
  if (m.startsWith('vicuna') || m.startsWith('wizardlm')) return 'llama';
  return null;
}

// Save a provider API key, then verify it with a 1-token request.
//
// Shared by the Settings provider cards and the first-run setup wizard, which
// previously carried ~40 identical lines each: the same PUT, the same
// anthropic-vs-OpenAI-shape branch, and the same error-body unwrapping. Only
// their DOM handling genuinely differed, so that stays at the call sites.
//
// Returns { ok, stage, message }:
//   ok:true                      — key saved and verified
//   ok:false, stage:'save'       — the key could not be stored
//   ok:false, stage:'test'       — stored, but the provider rejected it
//   ok:false, stage:'network'    — the request itself threw
// `message` is already unwrapped from the provider's JSON error envelope and
// truncated, so callers can display it directly.
async function saveAndVerifyProviderKey(provider, key) {
  var MAX_ERR = 200;
  try {
    var saveBody = {};
    saveBody[provider + '_key'] = key;
    var saveResp = await App.authFetch('/api/admin/providers', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(saveBody),
    });
    // Settings used to skip this check, so a failed save fell through to a
    // test that failed for a completely unrelated-looking reason.
    if (!saveResp.ok) return { ok: false, stage: 'save', message: '' };

    var testResp;
    if (provider === 'anthropic') {
      testResp = await App.authFetch('/v1/messages', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'claude-haiku-4-5-20251001', max_tokens: 1, messages: [{ role: 'user', content: 'hi' }] }),
      });
    } else {
      testResp = await App.authFetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: PROVIDER_TEST_MODELS[provider] || provider + '-test', max_tokens: 1, messages: [{ role: 'user', content: 'hi' }] }),
      });
    }
    if (testResp.ok) return { ok: true, stage: 'test', message: '' };

    var raw = await testResp.text();
    var friendly = raw;
    try { var ej = JSON.parse(raw); friendly = (ej.error && ej.error.message) || raw; } catch (pe) {}
    if (friendly.length > MAX_ERR) friendly = friendly.substring(0, MAX_ERR) + '…';
    return { ok: false, stage: 'test', message: friendly };
  } catch (e) {
    return { ok: false, stage: 'network', message: e.message || '' };
  }
}
