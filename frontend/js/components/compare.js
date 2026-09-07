'use strict';

// ============================================================================
// SwarmLLM — Compare Component
// Side-by-side multi-model comparison
// ============================================================================

(function() {
  var U = App.utils;

  // Last-resort backstop for a compare request, in ms. Deliberately as long as
  // the longest single wait the daemon permits for a generation, so it can only
  // fire after the node has itself given up — see the comment at its use.
  var COMPARE_BACKSTOP_MS = 600000;

  // Live elapsed seconds on each pending card, so a slow processor node looks
  // like it is working rather than frozen. Report #009: "The user has no way to
  // know from the UI that the model actually succeeded server-side."
  var elapsedTimer = null;
  function startElapsedTicker() {
    if (elapsedTimer) return;
    elapsedTimer = setInterval(function() {
      var nodes = document.querySelectorAll('#compare-results .compare-elapsed');
      if (!nodes.length) { stopElapsedTicker(); return; }
      nodes.forEach(function(el) {
        var since = parseInt(el.getAttribute('data-since'), 10);
        if (!since) return;
        el.textContent = Math.round((Date.now() - since) / 1000) + 's';
      });
    }, 1000);
  }
  function stopElapsedTicker() {
    if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
  }

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
          // The backend already says whether this node holds any of the model
          // (`local`) or is merely aware of it from the swarm. Labelling every
          // non-cloud entry "local" told the user a model served entirely by
          // peers was on their own machine (gotcha #484, second half) —
          // including entries the backend itself marks `source: "network"`
          // with zero hosted shards.
          App.compare.models.push({ id: m.id, type: m.local ? 'local' : 'network' });
        });
        cloudModels.forEach(function(m) {
          var mid = m.id;
          if (!App.compare.models.some(function(x) { return x.id === mid; })) {
            var ctx = m.context_length || m.context_window || m.max_model_len || 0;
            App.compare.models.push({ id: mid, type: 'cloud', context: ctx });
          }
        });

        if (App.compare.models.length === 0) {
          container.innerHTML = '<span class="text-muted" style="font-size:0.8rem">' + U.escapeHtml(I18n.t('compare.no_models')) + '</span>';
          return;
        }

        container.innerHTML = '';
        var chipTmpl = document.getElementById('tmpl-compare-model-chip');
        App.compare.models.forEach(function(m, idx) {
          var chip = chipTmpl.content.cloneNode(true).firstElementChild;
          chip.className = 'compare-model-chip type-' + m.type;
          chip.style.animationDelay = (idx * 30) + 'ms';
          var displayName = m.id.length > 35 ? m.id.substring(0, 35) + '...' : m.id;
          var ctxLabel = m.context && m.context > 0 ? ' \u00B7 ' + I18n.t('models.context_k_abbr', { n: Math.round(m.context / 1000) }) : '';
          chip.querySelector('input').value = m.id;
          chip.querySelector('.chip-name').textContent = displayName;
          var typeLabel = m.type === 'local'
            ? I18n.t('compare.filter_local')
            : (m.type === 'network' ? I18n.t('chat.source_network') : I18n.t('dashboard.chip_cloud'));
          chip.querySelector('.chip-type').textContent = typeLabel + ctxLabel;
          chip.title = m.id + (ctxLabel ? ' (' + I18n.t('compare.context_tokens', { n: m.context }) + ')' : '');
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
        card.id = 'compare-card-' + U.safeId(modelId);
        card.querySelector('.compare-card-model').textContent = modelId;
        card.querySelector('.compare-card-model').title = modelId;
        card.querySelector('.compare-card-status').innerHTML = '<span class="spinner" style="width:14px;height:14px"></span>';
        card.querySelector('.compare-card-body').innerHTML = '<div class="compare-spinner"><div class="spinner"></div> ' + U.escapeHtml(I18n.t('compare.waiting')) + ' <span class="compare-elapsed" data-since="' + Date.now() + '">0s</span></div>';
        card.querySelector('.compare-card-actions').style.display = 'none';
        resultsDiv.appendChild(card);
      });

      startElapsedTicker();

      var statusDiv = document.getElementById('compare-status');
      if (statusDiv) { statusDiv.style.display = ''; statusDiv.innerHTML = '<span class="text-muted">' + U.escapeHtml(I18n.t('compare.sending', { n: n })) + '</span>'; }

      var promises = App.compare.selected.map(function(modelId) {
        var body = {
          model: modelId,
          max_tokens: maxTokens,
          temperature: temperature,
          messages: [{ role: 'user', content: prompt.trim() }],
          // Streamed, like the chat tab, and for a reason specific to THIS
          // screen: a comparison exists to tell models apart, and the one
          // running on a processor-only peer — exactly the one a user is here
          // to find — is the one that shows nothing for 30-60s. Without a
          // partial reply that is indistinguishable from a stall, so the
          // screen built for judging replies was the screen that showed the
          // least while they were produced (report #026).
          stream: true,
        };
        if (system.trim()) body.system = system.trim();

        var start = performance.now();
        var controller = new AbortController();
        // Generation gets no deadline of the CLIENT's invention.
        //
        // This was a flat 45 s. A distributed forward for an 8B-14B on a
        // processor-only node routinely takes tens of seconds, and those nodes
        // are explicitly supported — so the ceiling was a minimum-capability
        // requirement nobody chose, and it DISCARDED work the daemon had
        // finished: report #009 caught a reply completing at execute_ms=44886
        // with finish_reason=stop, thrown away because the browser had aborted
        // a fraction of a second earlier.
        //
        // The daemon deliberately serves generation OUTSIDE its own timeout
        // layer for exactly this reason; the browser then reimposed one, which
        // is `.claude/rules/architecture.md` § "Timeouts: bound what actually
        // varies" rule 3 on the other side of the wire — and rule 4, since the
        // tightest ceiling was in another file nobody grepped.
        //
        // What remains is a last-resort backstop, not a judgement about how
        // long an answer may take: it matches the longest single wait the
        // daemon itself permits (`remote_generate`'s 600 s first-token
        // deadline), so it can only fire once the node has already given up.
        var timeoutId = setTimeout(function() { controller.abort(); }, COMPARE_BACKSTOP_MS);

        // The stream is re-assembled into the same non-streaming shape the
        // card renderer and the history entry already read — which is what the
        // official SDKs' `.accumulate()` does, and what `renderHistory` was
        // already building by hand. One result shape, whichever way the text
        // arrived.
        var text = '';
        // `null`, not 0. The Anthropic surface omits `input_tokens` rather
        // than sending a confident zero when it does not know it — which is
        // the case on the router path, i.e. exactly a peer-served model — and
        // the card must not turn that silence into a figure.
        var usage = { input_tokens: null, output_tokens: 0 };
        var streamError = null;
        var streamBody = null;
        var onEvent = function(evt) {
          if (!evt || !evt.type) return;
          if (evt.type === 'content_block_delta') {
            var d = evt.delta || {};
            // `input_json_delta` is a tool call's arguments. Compare sends no
            // tools, but ignoring anything that is not text keeps a model that
            // emits one from writing raw JSON into the card.
            if (d.type !== 'text_delta' || !d.text) return;
            text += d.text;
            if (!streamBody) streamBody = App.compare._beginStreaming(modelId);
            if (streamBody) U.renderReplyInto(streamBody, text);
            return;
          }
          if (evt.type === 'message_delta' && evt.usage) {
            if (typeof evt.usage.output_tokens === 'number') usage.output_tokens = evt.usage.output_tokens;
            if (typeof evt.usage.input_tokens === 'number') usage.input_tokens = evt.usage.input_tokens;
            return;
          }
          // The Anthropic surface's own failure frame. It is terminal, so the
          // reason must be kept rather than left to look like an empty reply.
          if (evt.type === 'error' && evt.error) streamError = evt.error.message || I18n.t('compare.status_error');
        };

        return App.authFetch('/v1/messages', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: controller.signal,
        }).then(function(resp) {
          // A request refused before the stream opens answers with the ordinary
          // JSON error envelope, not SSE.
          if (!resp.ok || !resp.body) {
            return resp.json().catch(function() { return {}; }).then(function(data) {
              clearTimeout(timeoutId);
              return {
                model: modelId, data: data, ok: false,
                latency_ms: Math.round(performance.now() - start),
              };
            });
          }
          return U.readSseStream(resp.body.getReader(), onEvent).then(function() {
            clearTimeout(timeoutId);
            var elapsed = Math.round(performance.now() - start);
            if (streamError) {
              return { model: modelId, error: streamError, ok: false, latency_ms: elapsed };
            }
            return {
              model: modelId,
              ok: true,
              latency_ms: elapsed,
              data: { content: [{ type: 'text', text: text }], usage: usage },
            };
          });
        }).catch(function(err) {
          clearTimeout(timeoutId);
          // Whatever already streamed is the model's real answer so far and is
          // kept — the same rule the chat tab applies to a stopped reply. Only
          // when nothing arrived at all is this reported as a failure.
          var elapsed = Math.round(performance.now() - start);
          if (text) {
            return {
              model: modelId,
              ok: true,
              latency_ms: elapsed,
              data: { content: [{ type: 'text', text: text }], usage: usage },
            };
          }
          // NOT "this model failed": the daemon may well still be working, and
          // on the reported case it had already produced a complete reply.
          var msg = err.name === 'AbortError' ? I18n.t('compare.no_reply_yet') : err.message;
          return { model: modelId, error: msg, ok: false, latency_ms: elapsed };
        });
      });

      var completed = 0;
      promises.forEach(function(p) {
        p.then(function(result) {
          completed++;
          App.compare.renderCard(result);
          if (statusDiv) {
            statusDiv.innerHTML = '<span class="text-muted">' + U.escapeHtml(I18n.t('compare.progress', { done: completed, total: n })) + '</span>';
            if (completed === n) {
              statusDiv.innerHTML = '<span style="color:var(--green)">' + U.escapeHtml(I18n.t('compare.all_complete', { n: n })) + '</span>';
              setTimeout(function() { statusDiv.style.display = 'none'; }, 3000);
            }
          }
        });
      });

      Promise.all(promises).then(function(results) {
        App.compare.running = false;
        stopElapsedTicker();
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
              // `null` survives into the stored entry so a restored card says
              // the same thing the live one did — an unreported prompt count
              // must not become a zero on the way through localStorage.
              var ru = (r.ok && r.data && r.data.usage) || {};
              return {
                model: r.model, ok: r.ok, error: r.error || null,
                latency_ms: r.latency_ms, content: content,
                input_tokens: typeof ru.input_tokens === 'number' ? ru.input_tokens : null,
                output_tokens: ru.output_tokens || 0,
              };
            }),
          });
          if (history.length > 20) history = history.slice(0, 20);
          localStorage.setItem(App.COMPARE_HISTORY_KEY, JSON.stringify(history));
          App.compare.renderHistory();
        } catch (e) {}
      }).catch(function() {
        App.compare.running = false;
        if (btn) { btn.disabled = false; btn.textContent = I18n.t('compare.run_compare'); }
      });
    },

    renderHistory: function() {
      var container = document.getElementById('compare-history');
      if (!container) return;
      try {
        var history = JSON.parse(localStorage.getItem(App.COMPARE_HISTORY_KEY) || '[]');
        if (history.length === 0) { container.style.display = 'none'; return; }
        container.style.display = '';
        var html = '<div style="font-size:0.75rem;color:var(--text-muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:0.06em">' + U.escapeHtml(I18n.t('compare.history_title')) + '</div>';
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
        card.id = 'compare-card-' + U.safeId(r.model);
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

    // Swap a card from "waiting" to "streaming" on its first token, and hand
    // back the element the reply is rendered into.
    //
    // The elapsed counter moves into the status slot rather than being lost
    // with the placeholder: a card that is producing text still wants to say
    // how long it has been at it, which is the number this screen is for.
    _beginStreaming: function(modelId) {
      var card = document.getElementById('compare-card-' + U.safeId(modelId));
      if (!card) return null;
      var bodyEl = card.querySelector('.compare-card-body');
      if (!bodyEl) return null;
      // Carry the placeholder's own start time across, rather than restarting
      // the count from the first token — the ticker reads `Date.now()`, and
      // the number the reader wants is how long this card has been running.
      var existing = bodyEl.querySelector('.compare-elapsed');
      var since = (existing && existing.getAttribute('data-since')) || String(Date.now());
      var statusEl = card.querySelector('.compare-card-status');
      if (statusEl) {
        statusEl.innerHTML = '<span class="compare-elapsed" data-since="' + U.escapeHtml(since) + '">0s</span>';
      }
      bodyEl.innerHTML = '';
      bodyEl.classList.remove('error');
      return bodyEl;
    },

    renderCard: function(result) {
      var cardId = 'compare-card-' + U.safeId(result.model);
      var card = document.getElementById(cardId);
      if (!card) return;

      var content = '';
      var isError = false;
      var inputTokens = null;
      var outputTokens = 0;

      if (result.error) {
        content = result.error;
        isError = true;
      } else if (!result.ok) {
        content = U.extractErrorMessage(
          result.data,
          JSON.stringify(result.data.error || result.data, null, 2)
        );
        isError = true;
      } else {
        var blocks = result.data.content || [];
        blocks.forEach(function(b) {
          if (b.type === 'text' && b.text) content += b.text;
        });
        if (!content) content = I18n.t('compare.empty_response');
        var u = result.data.usage || {};
        inputTokens = typeof u.input_tokens === 'number' ? u.input_tokens : null;
        outputTokens = u.output_tokens || 0;
      }

      var cardContentId = 'compare-content-' + U.safeId(result.model);

      var modelEl = card.querySelector('.compare-card-model');
      modelEl.textContent = result.model;
      modelEl.title = result.model;

      var statusEl = card.querySelector('.compare-card-status');
      if (isError) {
        statusEl.style.color = 'var(--red)';
        statusEl.style.fontSize = '0.7rem';
        statusEl.textContent = I18n.t('compare.status_error');
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
      if (isError) {
        // An error is a message from this node, not a model's reply: it is not
        // markdown and must not be rendered as any.
        bodyEl.classList.remove('md-body');
        bodyEl.textContent = content;
        bodyEl._rawText = content;
        bodyEl.classList.add('error');
      } else {
        // `flush` because this is the final render: a tab in the background
        // suspends rAF, so a comparison that finished while the user was
        // elsewhere would otherwise sit on its last streamed frame
        // (gotcha #471).
        bodyEl.classList.remove('error');
        U.renderReplyInto(bodyEl, content, { flush: true });
      }

      if (!isError) {
        var footerEl = card.querySelector('.compare-card-footer');
        footerEl.removeAttribute('hidden');
        var inEl = footerEl.querySelector('.ccf-in');
        // Shown only when the surface actually told us. A peer-served reply
        // streams without a prompt count (the router arm returns before the
        // final result carrying it), and "in 0" would be a claim rather than
        // a gap.
        if (inputTokens === null) {
          inEl.hidden = true;
        } else {
          inEl.hidden = false;
          inEl.textContent = I18n.t('compare.label_in') + inputTokens;
        }
        footerEl.querySelector('.ccf-out').textContent = I18n.t('compare.label_out') + outputTokens;
        footerEl.querySelector('.ccf-latency').textContent = result.latency_ms + 'ms';
        if (outputTokens > 0) {
          var tpsEl = footerEl.querySelector('.ccf-tps');
          tpsEl.removeAttribute('hidden');
          var t = outputTokens / (result.latency_ms / 1000);
          tpsEl.textContent = (t >= 1 ? Math.round(t) : t.toFixed(1)) + ' ' + I18n.t('compare.tok_per_sec');
        }
      }
    },
  };
})();
