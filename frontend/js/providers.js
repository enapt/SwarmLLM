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
