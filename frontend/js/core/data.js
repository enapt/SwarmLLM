'use strict';

// ============================================================================
// SwarmLLM — Data Store & Auth
// Single fetch point, in-flight deduplication, shared cache, authFetch
// ============================================================================

(function() {
  // --- Authenticated fetch with timeout ---
  var DEFAULT_TIMEOUT_MS = 30000;

  async function authFetch(url, opts) {
    if (!App.settings._apiKeyFull && App.settings._apiKeyPromise) {
      await App.settings._apiKeyPromise;
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
          throw new Error('Request timed out — server may be busy or unreachable');
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
  };

  function dedupe(key, fn) {
    if (_inFlight[key]) return _inFlight[key];
    _inFlight[key] = Promise.resolve().then(fn).finally(function() {
      delete _inFlight[key];
    });
    return _inFlight[key];
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
      window._lastModelsData = models;
      return { models: models, cloudModels: cloudModels };
    });
  }

  function loadStats() {
    return dedupe('stats', async function() {
      var stats = null;
      var config = null;
      try {
        var r = await authFetch('/api/admin/stats');
        if (r.ok) stats = await r.json();
      } catch (e) {}
      try {
        var r2 = await authFetch('/api/admin/config');
        if (r2.ok) config = await r2.json();
      } catch (e) {}
      cache.stats = stats;
      cache.config = config;
      return { stats: stats, config: config };
    });
  }

  App.authFetch = authFetch;
  App.data = {
    loadModels: loadModels,
    loadStats: loadStats,
    cache: cache,
  };
})();
