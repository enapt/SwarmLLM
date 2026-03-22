'use strict';

// ============================================================================
// SwarmLLM — Application State & Configuration
// Global namespace + shared mutable state + constants + theme
// ============================================================================

window.App = {
  // --- Shared mutable state ---
  state: {
    ws: null,
    wsHealthy: false,
    wsWasConnected: false,
    wsBannerTimer: null,
    pollTimers: [],
    creditHistory: [],
    activeAcquisitions: {},
    _swarmModelSort: (function() { try { return localStorage.getItem('swarmllm_model_sort') || 'az'; } catch(e) { return 'az'; } })(),
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

  // --- Storage keys ---
  SESSIONS_KEY: 'swarmllm_sessions',
  ACTIVE_SESSION_KEY: 'swarmllm_active_session',
  SETUP_DONE_KEY: 'swarmllm_setup_done',
  CHAT_LAYOUT_KEY: 'swarmllm_chat_layout',
  HEALTH_INTERVAL_KEY: 'swarmllm_health_interval',
  THEME_KEY: 'swarmllm_theme',

  // --- Component namespaces (populated by component files) ---
  // ui, chat, dashboard, hf, settings, setup, identity, networkMap,
  // compare, data, notifications, models, downloads, shardMenu, providerHealth
};

// Initialize modelStatus from sessionStorage cache
try {
  var _cached = sessionStorage.getItem('swarmllm_model_status');
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
