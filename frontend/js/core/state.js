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
    statHistory: { peers: [], credits: [], requests: [], served: [], forwards: [], active: [] },
    _expandedModels: {},
    activeAcquisitions: {},
    _swarmModelSort: 'problems', // initialized below after App is defined, using App.MODEL_SORT_KEY
    _shardView: 'list', // initialized below after App is defined, using App.SHARD_VIEW_KEY
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
      if (p === '/admin/responses') return 'responses';
      if (p === '/admin/devices') return 'devices';
      if (p === '/admin/swarm') return 'swarm';
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
  // API key the user pasted in by hand, for dashboards the daemon won't hand a
  // key to automatically (see App.utils.clientTrust). Per-origin so a browser
  // used against several nodes doesn't send one node's key to another.
  MANUAL_KEY_KEY: 'swarmllm_manual_api_key',
  SETUP_SKIPPED_KEY: 'swarmllm_setup_skipped',
  SETUP_CHIP_DISMISSED_KEY: 'swarmllm_setup_chip_dismissed',
  WELCOME_SEEN_KEY: 'swarmllm_welcome_seen',
  HEALTH_INTERVAL_KEY: 'swarmllm_health_interval',
  THEME_KEY: 'swarmllm_theme',
  MODEL_SORT_KEY: 'swarmllm_model_sort',
  CURRENT_MODEL_KEY: 'swarmllm_current_model',
  COMPARE_HISTORY_KEY: 'swarmllm_compare_history',
  CHAT_HISTORY_KEY: 'swarmllm_chat_history',
  SHARD_VIEW_KEY: 'swarmllm_shard_view',
  // sessionStorage keys
  MODEL_STATUS_KEY: 'swarmllm_model_status',
  ACTIVITY_KEY: 'swarmllm_activity',
  NETWORK_LOG_KEY: 'swarmllm_network_log',

  // --- Constants ---
  MMPROJ_SHARD_INDEX: 4294967295, // u32::MAX — sentinel for vision encoder shards

  // --- Component namespaces (populated by component files) ---
  // ui, chat, dashboard, hf, settings, setup, identity, networkMap,
  // compare, data, notifications, models, downloads, providerHealth,
  // pruneSchedule, networkCode, networkStatus, pool, swarmTab
};

// Initialize _swarmModelSort using the constant now that App is defined
try {
  App.state._swarmModelSort = localStorage.getItem(App.MODEL_SORT_KEY) || 'problems';
  App.state._shardView = localStorage.getItem(App.SHARD_VIEW_KEY) || 'list';
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
