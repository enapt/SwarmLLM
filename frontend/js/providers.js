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

// Infer a provider/family key from a model ID string.
function modelIconKey(modelId) {
  if (!modelId) return null;
  var m = modelId.toLowerCase();
  if (m.startsWith('gpt') || m.startsWith('o1') || m.startsWith('o3') || m.startsWith('o4') || m.includes('-openai')) return 'openai';
  if (m.startsWith('claude')) return 'claude';
  if (m.startsWith('deepseek')) return 'deepseek';
  if (m.startsWith('mistral') || m.startsWith('mixtral') || m.startsWith('codestral')) return 'mistral';
  if (m.startsWith('llama') || m.startsWith('meta-llama')) return 'llama';
  if (m.startsWith('gemma')) return 'gemma';
  if (m.startsWith('gemini')) return 'gemini';
  if (m.startsWith('qwen')) return 'qwen';
  if (m.startsWith('phi')) return 'phi';
  return null;
}
