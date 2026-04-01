'use strict';

// ============================================================================
// SwarmLLM — Application State & Configuration
// Global namespace + shared mutable state + constants + theme
// ============================================================================

window.App = {
  // --- Shared mutable state ---
  state: {
    ws: null,
    wsWasConnected: false,
    wsBannerTimer: null,
    pollTimers: [],
    creditHistory: [],
    activeAcquisitions: {},
    _swarmModelSort: 'az', // initialized below after App is defined, using App.MODEL_SORT_KEY
    isStreaming: false,
    currentModel: '',
    currentSessionId: null,
    sessions: {},
    activeTab: (function() {
      var p = window.location.pathname;
      if (p === '/chat' || p.startsWith('/chat/')) return 'chat';
      if (p === '/admin/leaderboard') return 'leaderboard';
      if (p === '/admin/network') return 'network-map';
      if (p === '/admin/compare') return 'compare';
      if (p === '/admin/devices') return 'devices';
      return 'dashboard';
    })(),
    providerHealth: {},
    healthTimer: null,
    modelStatus: {},
    _modelStatusPending: {},
    pendingImages: [],
    _modelDropdownData: [],
    metadataCache: {},
    _cachedProviderData: null,
    inputEl: null,
  },

  // --- Storage keys (single source of truth for all storage key strings) ---
  // localStorage keys
  SESSIONS_KEY: 'swarmllm_sessions',
  ACTIVE_SESSION_KEY: 'swarmllm_active_session',
  SETUP_DONE_KEY: 'swarmllm_setup_done',
  CHAT_LAYOUT_KEY: 'swarmllm_chat_layout',
  HEALTH_INTERVAL_KEY: 'swarmllm_health_interval',
  THEME_KEY: 'swarmllm_theme',
  MODEL_SORT_KEY: 'swarmllm_model_sort',
  CURRENT_MODEL_KEY: 'swarmllm_current_model',
  COMPARE_HISTORY_KEY: 'swarmllm_compare_history',
  CHAT_HISTORY_KEY: 'swarmllm_chat_history', // legacy migration key
  // sessionStorage keys
  MODEL_STATUS_KEY: 'swarmllm_model_status',
  ACTIVITY_KEY: 'swarmllm_activity',
  NETWORK_LOG_KEY: 'swarmllm_network_log',
  MODEL_EVENTS_KEY: 'swarmllm_model_events',
  MODEL_NET_EVENTS_KEY: 'swarmllm_model_net_events',

  // --- Component namespaces (populated by component files) ---
  // ui, chat, dashboard, hf, settings, setup, identity, networkMap,
  // compare, data, notifications, models, downloads, shardMenu, providerHealth
};

// Initialize _swarmModelSort using the constant now that App is defined
try {
  App.state._swarmModelSort = localStorage.getItem(App.MODEL_SORT_KEY) || 'az';
} catch (e) {}

// Initialize modelStatus from sessionStorage cache
try {
  var _cached = sessionStorage.getItem(App.MODEL_STATUS_KEY);
  if (_cached) App.state.modelStatus = JSON.parse(_cached);
} catch (e) {}

// --- Theme ---
App.applyTheme = function(theme) {
  var resolved = theme;
  if (theme === 'system') {
    resolved = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  }
  document.documentElement.setAttribute('data-theme', resolved);
  var btn = document.getElementById('btn-theme-toggle');
  var icons = { dark: '\u263E', light: '\u2600', system: '\u25D1' };
  if (btn) btn.textContent = icons[theme] || '\u263E';
};

// Listen for system theme changes when in 'system' mode
try {
  window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', function() {
    if ((localStorage.getItem(App.THEME_KEY) || 'dark') === 'system') App.applyTheme('system');
  });
} catch(e) {}
