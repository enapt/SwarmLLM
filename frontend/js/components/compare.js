'use strict';

// ============================================================================
// SwarmLLM — Compare Component
// Side-by-side multi-model comparison
// ============================================================================

(function() {
  var U = App.utils;

  App.compare = {
    models: [],
    selected: [],
    running: false,

    loadModels: async function() {
      try {
        var container = document.getElementById('compare-model-list');
        if (!container) return;

        // Use the shared data store cache (deduped fetching)
        var result = await App.data.loadModels();
        var localModels = result.models || [];
        var cloudModels = result.cloudModels || [];

        App.compare.models = [];
        localModels.forEach(function(m) {
          App.compare.models.push({ id: m.id, type: 'local' });
        });
        cloudModels.forEach(function(m) {
          var mid = m.id;
          if (!App.compare.models.some(function(x) { return x.id === mid; })) {
            var ctx = m.context_length || m.context_window || m.max_model_len || 0;
            App.compare.models.push({ id: mid, type: 'cloud', context: ctx });
          }
        });

        if (App.compare.models.length === 0) {
          container.innerHTML = '<span class="text-muted" style="font-size:0.8rem">No models available yet. Download a model or add a cloud provider in Settings first.</span>';
          return;
        }

        container.innerHTML = '';
        var chipTmpl = document.getElementById('tmpl-compare-model-chip');
        App.compare.models.forEach(function(m, idx) {
          var chip = chipTmpl.content.cloneNode(true).firstElementChild;
          chip.className = 'compare-model-chip type-' + m.type;
          chip.style.animationDelay = (idx * 30) + 'ms';
          var displayName = m.id.length > 35 ? m.id.substring(0, 35) + '...' : m.id;
          var ctxLabel = m.context && m.context > 0 ? ' \u00B7 ' + Math.round(m.context / 1000) + 'k ctx' : '';
          chip.querySelector('input').value = m.id;
          chip.querySelector('.chip-name').textContent = displayName;
          chip.querySelector('.chip-type').textContent = m.type + ctxLabel;
          chip.title = m.id + (ctxLabel ? ' (' + m.context + ' tokens)' : '');
          chip.querySelector('input').addEventListener('change', function() {
            chip.classList.toggle('selected', this.checked);
            App.compare.updateSelected();
          });
          container.appendChild(chip);
        });

        var filters = document.getElementById('compare-filters');
        if (filters) {
          filters.querySelectorAll('.compare-filter').forEach(function(btn) {
            var fresh = btn.cloneNode(true);
            btn.parentNode.replaceChild(fresh, btn);
            fresh.addEventListener('click', function() {
              filters.querySelectorAll('.compare-filter').forEach(function(b) { b.classList.remove('active'); });
              fresh.classList.add('active');
              var f = fresh.getAttribute('data-filter');
              container.querySelectorAll('.compare-model-chip').forEach(function(chip) {
                if (f === 'all') { chip.style.display = ''; }
                else { chip.style.display = chip.classList.contains('type-' + f) ? '' : 'none'; }
              });
            });
          });
        }
      } catch(e) {}
    },

    updateSelected: function() {
      App.compare.selected = [];
      var checks = document.querySelectorAll('#compare-model-list input[type="checkbox"]:checked');
      checks.forEach(function(cb) { App.compare.selected.push(cb.value); });
    },

    run: async function() {
      if (App.compare.running) return;
      var prompt = (document.getElementById('compare-prompt') || {}).value;
      if (!prompt || !prompt.trim()) {
        App.notifications.showToast(I18n.t('compare.enter_prompt'), 'error');
        return;
      }
      if (App.compare.selected.length < 2) {
        App.notifications.showToast(I18n.t('compare.select_min'), 'error');
        return;
      }
      if (App.compare.selected.length > 10) {
        App.notifications.showToast(I18n.t('compare.select_max'), 'error');
        return;
      }

      var system = (document.getElementById('compare-system') || {}).value || '';
      var temperature = parseFloat((document.getElementById('compare-temp') || {}).value) || 0.7;
      var maxTokens = parseInt((document.getElementById('compare-max-tokens') || {}).value) || 1024;

      App.compare.running = true;
      var btn = document.getElementById('btn-compare-run');
      if (btn) { btn.disabled = true; btn.textContent = I18n.t('compare.running'); }

      var resultsDiv = document.getElementById('compare-results');
      var n = App.compare.selected.length;
      var colClass = n <= 2 ? 'cols-2' : n <= 3 ? 'cols-3' : n <= 4 ? 'cols-4' : 'cols-many';
      resultsDiv.className = 'compare-results ' + colClass;

      resultsDiv.innerHTML = '';
      var cardTmpl = document.getElementById('tmpl-compare-card');
      App.compare.selected.forEach(function(modelId) {
        var card = cardTmpl.content.cloneNode(true).firstElementChild;
        card.id = 'compare-card-' + modelId.replace(/[^a-zA-Z0-9_-]/g, '_');
        card.querySelector('.compare-card-model').textContent = modelId;
        card.querySelector('.compare-card-model').title = modelId;
        card.querySelector('.compare-card-status').innerHTML = '<span class="spinner" style="width:14px;height:14px"></span>';
        card.querySelector('.compare-card-body').innerHTML = '<div class="compare-spinner"><div class="spinner"></div> ' + I18n.t('compare.waiting') + '</div>';
        card.querySelector('.compare-card-actions').style.display = 'none';
        resultsDiv.appendChild(card);
      });

      var statusDiv = document.getElementById('compare-status');
      if (statusDiv) { statusDiv.style.display = ''; statusDiv.innerHTML = '<span class="text-muted">' + I18n.t('compare.sending', { n: n }) + '</span>'; }

      var promises = App.compare.selected.map(function(modelId) {
        var body = {
          model: modelId,
          max_tokens: maxTokens,
          temperature: temperature,
          messages: [{ role: 'user', content: prompt.trim() }],
          stream: false,
        };
        if (system.trim()) body.system = system.trim();

        var start = performance.now();
        var controller = new AbortController();
        var timeoutId = setTimeout(function() { controller.abort(); }, 45000);
        return App.authFetch('/v1/messages', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: controller.signal,
        }).then(function(resp) {
          clearTimeout(timeoutId);
          var elapsed = Math.round(performance.now() - start);
          return resp.json().then(function(data) {
            return { model: modelId, data: data, ok: resp.ok, latency_ms: elapsed };
          });
        }).catch(function(err) {
          clearTimeout(timeoutId);
          var msg = err.name === 'AbortError' ? 'Timed out after 45s' : err.message;
          return { model: modelId, error: msg, ok: false, latency_ms: Math.round(performance.now() - start) };
        });
      });

      var completed = 0;
      promises.forEach(function(p) {
        p.then(function(result) {
          completed++;
          App.compare.renderCard(result);
          if (statusDiv) {
            statusDiv.innerHTML = '<span class="text-muted">' + I18n.t('compare.progress', { done: completed, total: n }) + '</span>';
            if (completed === n) {
              statusDiv.innerHTML = '<span style="color:var(--green)">' + I18n.t('compare.all_complete', { n: n }) + '</span>';
              setTimeout(function() { statusDiv.style.display = 'none'; }, 3000);
            }
          }
        });
      });

      Promise.all(promises).then(function(results) {
        App.compare.running = false;
        if (btn) { btn.disabled = false; btn.textContent = I18n.t('compare.run_compare'); }
        try {
          var history = JSON.parse(localStorage.getItem(App.COMPARE_HISTORY_KEY) || '[]');
          history.unshift({
            prompt: prompt.trim().substring(0, 200),
            models: App.compare.selected.slice(),
            timestamp: Date.now(),
            results: results.map(function(r) {
              var content = '';
              if (!r.error && r.ok) {
                (r.data.content || []).forEach(function(b) { if (b.type === 'text') content += b.text; });
              }
              return {
                model: r.model, ok: r.ok, error: r.error || null,
                latency_ms: r.latency_ms, content: content,
                input_tokens: r.ok ? ((r.data.usage || {}).input_tokens || 0) : 0,
                output_tokens: r.ok ? ((r.data.usage || {}).output_tokens || 0) : 0,
              };
            }),
          });
          if (history.length > 20) history = history.slice(0, 20);
          localStorage.setItem(App.COMPARE_HISTORY_KEY, JSON.stringify(history));
          App.compare.renderHistory();
        } catch (e) {}
      });
    },

    renderHistory: function() {
      var container = document.getElementById('compare-history');
      if (!container) return;
      try {
        var history = JSON.parse(localStorage.getItem(App.COMPARE_HISTORY_KEY) || '[]');
        if (history.length === 0) { container.style.display = 'none'; return; }
        container.style.display = '';
        var html = '<div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em">' + I18n.t('compare.history_title') + '</div>';
        history.slice(0, 10).forEach(function(item, idx) {
          var ago = U.timeAgo(item.timestamp);
          var modelList = (item.models || []).map(function(m) {
            return m.split('/').pop().replace(/-\d{4}-\d{2}-\d{2}$/, '');
          }).join(', ');
          html += '<div class="compare-history-item" data-compare-idx="' + idx + '">' +
            '<span class="compare-history-prompt">' + U.escapeHtml(item.prompt) + '</span>' +
            '<span class="compare-history-meta">' + U.escapeHtml(modelList) + ' &middot; ' + ago + '</span>' +
          '</div>';
        });
        container.innerHTML = html;
      } catch (e) { container.style.display = 'none'; }
    },

    restoreFromHistory: function(item) {
      var promptEl = document.getElementById('compare-prompt');
      if (promptEl) promptEl.value = item.prompt;

      var resultsDiv = document.getElementById('compare-results');
      if (!resultsDiv || !item.results || !item.results.length) return;

      resultsDiv.innerHTML = '';
      var rCardTmpl = document.getElementById('tmpl-compare-card');
      item.results.forEach(function(r) {
        var card = rCardTmpl.content.cloneNode(true).firstElementChild;
        card.id = 'compare-card-' + r.model.replace(/[^a-zA-Z0-9_-]/g, '_');
        resultsDiv.appendChild(card);
        App.compare.renderCard({
          model: r.model, ok: r.ok, error: r.error,
          latency_ms: r.latency_ms,
          data: {
            content: [{ type: 'text', text: r.content || '' }],
            usage: { input_tokens: r.input_tokens, output_tokens: r.output_tokens },
          },
        });
      });

      var statusDiv = document.getElementById('compare-status');
      if (statusDiv) { statusDiv.style.display = ''; statusDiv.innerHTML = '<span class="text-muted">' + I18n.t('compare.restored', { ago: U.timeAgo(item.timestamp) }) + '</span>'; }
    },

    renderCard: function(result) {
      var cardId = 'compare-card-' + result.model.replace(/[^a-zA-Z0-9_-]/g, '_');
      var card = document.getElementById(cardId);
      if (!card) return;

      var content = '';
      var isError = false;
      var inputTokens = 0;
      var outputTokens = 0;

      if (result.error) {
        content = result.error;
        isError = true;
      } else if (!result.ok) {
        content = result.data.error && result.data.error.message
          ? result.data.error.message
          : JSON.stringify(result.data.error || result.data, null, 2);
        isError = true;
      } else {
        var blocks = result.data.content || [];
        blocks.forEach(function(b) {
          if (b.type === 'text' && b.text) content += b.text;
        });
        if (!content) content = '(empty response)';
        inputTokens = (result.data.usage || {}).input_tokens || 0;
        outputTokens = (result.data.usage || {}).output_tokens || 0;
      }

      var cardContentId = 'compare-content-' + result.model.replace(/[^a-zA-Z0-9_-]/g, '_');

      var modelEl = card.querySelector('.compare-card-model');
      modelEl.textContent = result.model;
      modelEl.title = result.model;

      var statusEl = card.querySelector('.compare-card-status');
      if (isError) {
        statusEl.style.color = 'var(--red)';
        statusEl.style.fontSize = '0.7rem';
        statusEl.textContent = 'error';
      } else {
        statusEl.style.color = 'var(--green)';
        statusEl.style.fontSize = '0.7rem';
        statusEl.textContent = result.latency_ms + 'ms';
      }

      var actionsEl = card.querySelector('.compare-card-actions');
      actionsEl.style.display = '';
      var copyBtn = card.querySelector('.compare-card-copy-btn');
      if (copyBtn) copyBtn.setAttribute('data-copy-compare', cardContentId);

      var bodyEl = card.querySelector('.compare-card-body');
      bodyEl.id = cardContentId;
      bodyEl.textContent = content;
      if (isError) bodyEl.classList.add('error');

      if (!isError) {
        var footerEl = card.querySelector('.compare-card-footer');
        footerEl.removeAttribute('hidden');
        footerEl.querySelector('.ccf-in').textContent = 'In: ' + inputTokens;
        footerEl.querySelector('.ccf-out').textContent = 'Out: ' + outputTokens;
        footerEl.querySelector('.ccf-latency').textContent = result.latency_ms + 'ms';
        if (outputTokens > 0) {
          var tpsEl = footerEl.querySelector('.ccf-tps');
          tpsEl.removeAttribute('hidden');
          var t = outputTokens / (result.latency_ms / 1000);
          tpsEl.textContent = (t >= 1 ? Math.round(t) : t.toFixed(1)) + ' tok/s';
        }
      }
    },
  };
})();
