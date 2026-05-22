'use strict';

// SwarmLLM i18n — Lightweight internationalization
// Translation files: frontend/i18n/{lang}.json
// Usage: I18n.t('key') or I18n.t('key', {count: 3, name: 'foo'})

var I18n = (function() {
  var strings = {};
  var fallback = {};
  var currentLang = '';
  var STORAGE_KEY = 'swarmllm_lang'; // raw string — i18n.js loads before state.js

  function detectLang(available) {
    var stored;
    try { stored = localStorage.getItem(STORAGE_KEY); } catch(e) {}
    if (stored && available.indexOf(stored) !== -1) return stored;
    var nav = (navigator.language || 'en').split('-')[0];
    if (available.indexOf(nav) !== -1) return nav;
    return 'en';
  }

  function loadLang(lang, cb) {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/static/i18n/' + lang + '.json', true);
    xhr.onload = function() {
      if (xhr.status === 200) {
        try { cb(null, JSON.parse(xhr.responseText)); }
        catch(e) { cb(e, null); }
      } else { cb(new Error('HTTP ' + xhr.status), null); }
    };
    xhr.onerror = function() { cb(new Error('Network error'), null); };
    xhr.send();
  }

  function interpolate(str, params) {
    if (!params) return str;
    return str.replace(/\{(\w+)\}/g, function(_, key) {
      return params[key] !== undefined ? String(params[key]) : '{' + key + '}';
    });
  }

  function pluralKey(key, count) {
    if (count === undefined || count === null) return key;
    var n = typeof count === 'number' ? count : parseInt(count, 10);
    var suffix = (n === 1) ? '_one' : '_other';
    var candidate = key + suffix;
    if (strings[candidate] || fallback[candidate]) return candidate;
    return key;
  }

  function t(key, params) {
    var resolvedKey = (params && params.count !== undefined)
      ? pluralKey(key, params.count) : key;
    var str = strings[resolvedKey] || fallback[resolvedKey] || resolvedKey;
    return interpolate(str, params);
  }

  function translatePage() {
    document.querySelectorAll('[data-i18n]').forEach(function(el) {
      el.textContent = t(el.getAttribute('data-i18n'));
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(function(el) {
      el.placeholder = t(el.getAttribute('data-i18n-placeholder'));
    });
    document.querySelectorAll('[data-i18n-title]').forEach(function(el) {
      el.title = t(el.getAttribute('data-i18n-title'));
    });
    document.querySelectorAll('[data-i18n-aria-label]').forEach(function(el) {
      el.setAttribute('aria-label', t(el.getAttribute('data-i18n-aria-label')));
    });
    document.documentElement.lang = currentLang;
    if (strings._dir) document.documentElement.dir = strings._dir;
  }

  function init(available, cb) {
    loadLang('en', function(err, en) {
      fallback = en || {};
      var lang = detectLang(available);
      if (lang === 'en') {
        strings = fallback;
        currentLang = 'en';
        translatePage();
        if (cb) cb();
        return;
      }
      setLang(lang, cb);
    });
  }

  function setLang(lang, cb) {
    if (lang === 'en') {
      strings = fallback;
      currentLang = 'en';
      try { localStorage.setItem(STORAGE_KEY, lang); } catch(e) {}
      translatePage();
      if (cb) cb();
      return;
    }
    loadLang(lang, function(err, data) {
      if (err) {
        // Fall back to English — don't persist the broken language
        strings = fallback;
        currentLang = 'en';
      } else {
        strings = data || {};
        currentLang = lang;
        try { localStorage.setItem(STORAGE_KEY, lang); } catch(e) {}
      }
      translatePage();
      if (cb) cb();
    });
  }

  function getLang() { return currentLang; }

  return {
    t: t,
    init: init,
    setLang: setLang,
    getLang: getLang,
    translatePage: translatePage
  };
})();
