'use strict';

// ============================================================================
// SwarmLLM — Data Store & Auth
// Single fetch point, in-flight deduplication, shared cache, authFetch
// ============================================================================

(function() {
  // --- Authenticated fetch with timeout ---
  var DEFAULT_TIMEOUT_MS = 30000;

  async function authFetch(url, opts) {
    // Ensure the key bootstrap has run — do NOT merely check whether something
    // else already started it. See App.settings.ensureApiKey.
    if (!App.settings._apiKeyFull && App.settings.ensureApiKey) {
      try { await App.settings.ensureApiKey(); } catch (e) { /* fall through unauthenticated */ }
    }
    opts = opts || {};
    opts.headers = opts.headers || {};
    if (App.settings._apiKeyFull && !opts.headers['Authorization']) {
      opts.headers['Authorization'] = 'Bearer ' + App.settings._apiKeyFull;
    }
    var timeoutMs = opts._timeout !== undefined ? opts._timeout : DEFAULT_TIMEOUT_MS;
    delete opts._timeout;
    if (timeoutMs > 0 && typeof AbortController !== 'undefined' && !opts.signal) {
      var controller = new AbortController();
      opts.signal = controller.signal;
      var timer = setTimeout(function() { controller.abort(); }, timeoutMs);
      try {
        var resp = await fetch(url, opts);
        clearTimeout(timer);
        return resp;
      } catch (e) {
        clearTimeout(timer);
        if (e.name === 'AbortError') {
          throw new Error(typeof I18n !== 'undefined' ? I18n.t('errors.request_timeout') : 'Request timed out');
        }
        throw e;
      }
    }
    return fetch(url, opts);
  }

  // --- Data Store ---
  var _inFlight = {};
  var cache = {
    models: [],
    cloudModels: [],
    stats: null,
    config: null,
    peers: [],
    providers: null,
  };

  function dedupe(key, fn) {
    if (_inFlight[key]) return _inFlight[key];
    _inFlight[key] = Promise.resolve().then(fn).finally(function() {
      delete _inFlight[key];
    });
    return _inFlight[key];
  }

  function invalidateDedup(key) {
    delete _inFlight[key];
  }

  function loadModels() {
    return dedupe('models', async function() {
      var models = [];
      var cloudModels = [];
      try {
        var r = await authFetch('/api/admin/models');
        if (r.ok) models = await r.json();
      } catch (e) {}
      try {
        var r2 = await authFetch('/api/admin/provider-models');
        if (r2.ok) { var d = await r2.json(); cloudModels = d.models || []; }
      } catch (e) {}
      cache.models = models;
      cache.cloudModels = cloudModels;
      // models cached in App.data.cache.models
      return { models: models, cloudModels: cloudModels };
    });
  }

  function loadStats() {
    return dedupe('stats', async function() {
      var stats = null;
      try {
        var r = await authFetch('/api/admin/stats');
        if (r.ok) stats = await r.json();
      } catch (e) {}
      var config = await loadConfig();
      cache.stats = stats;
      return { stats: stats, config: config };
    });
  }

  function loadPeers() {
    return dedupe('peers', async function() {
      var peers = [];
      try {
        var r = await authFetch('/api/admin/peers');
        if (r.ok) peers = await r.json();
      } catch (e) {}
      cache.peers = peers;
      return peers;
    });
  }

  function loadConfig() {
    return dedupe('config', async function() {
      var config = null;
      try {
        var r = await authFetch('/api/admin/config');
        if (r.ok) config = await r.json();
      } catch (e) {}
      cache.config = config;
      return config;
    });
  }

  function loadProviders() {
    return dedupe('providers', async function() {
      var providers = null;
      try {
        var r = await authFetch('/api/admin/providers');
        if (r.ok) providers = await r.json();
      } catch (e) {}
      cache.providers = providers;
      return providers;
    });
  }

  // Dedup pipeline-plan fetches across the dashboard pipeline overlay and
  // network-map's region path renderer — both expand on initial load and
  // would otherwise issue duplicate /pipeline-plan requests for the same
  // model. No long-lived cache (the plan changes when peers join/leave).
  function loadPipelinePlan(modelId) {
    if (!modelId) return Promise.resolve(null);
    return dedupe('pipelinePlan:' + modelId, async function() {
      try {
        var r = await authFetch('/api/admin/models/' + encodeURIComponent(modelId) + '/pipeline-plan');
        if (r.ok) return await r.json();
      } catch (e) {}
      return null;
    });
  }

  // In-flight dedup only (no long-lived cache): three components can request
  // this concurrently on page load — the header strip, the dashboard panel,
  // and settings — but we want a single `claude --version` subprocess call.
  function loadClaudeSubStatus() {
    return dedupe('claudeSubStatus', async function() {
      try {
        var r = await authFetch('/api/admin/claude-subscription/status');
        if (r && r.ok) return await r.json();
      } catch (e) {}
      return null;
    });
  }

  App.authFetch = authFetch;
  App.data = {
    loadModels: loadModels,
    loadStats: loadStats,
    loadConfig: loadConfig,
    loadPeers: loadPeers,
    loadProviders: loadProviders,
    loadPipelinePlan: loadPipelinePlan,
    loadClaudeSubStatus: loadClaudeSubStatus,
    invalidateDedup: invalidateDedup,
    cache: cache,
  };
})();
