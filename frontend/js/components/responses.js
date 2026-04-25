'use strict';

// ============================================================================
// SwarmLLM — Responses Component (V6 of responses_api_v2)
// Dashboard surface for the OpenAI /v1/responses endpoint:
//  - retrieve a stored response by id
//  - browse the recent responses list (filtered by status)
//  - cancel an active background response
// Talks to /api/admin/responses (list) and /v1/responses/:id (retrieve, delete,
// cancel). Bearer auth piggybacks on the same admin key the rest of the
// dashboard uses via authFetch.
// ============================================================================

(function() {
  var U = App.utils;

  App.responses = {
    _records: [],
    _statusFilter: 'all',
    _refreshTimer: null,

    init: function() {
      var self = this;
      var refresh = document.getElementById('btn-responses-refresh');
      if (refresh) refresh.addEventListener('click', function() { self.load(); });

      var retrieveBtn = document.getElementById('btn-responses-retrieve');
      var retrieveInput = document.getElementById('responses-retrieve-id');
      if (retrieveBtn && retrieveInput) {
        retrieveBtn.addEventListener('click', function() { self.retrieve(retrieveInput.value.trim()); });
        retrieveInput.addEventListener('keydown', function(e) {
          if (e.key === 'Enter') { e.preventDefault(); self.retrieve(retrieveInput.value.trim()); }
        });
      }

      // Status filter chips.
      document.querySelectorAll('#responses-filter [data-resp-filter]').forEach(function(btn) {
        btn.addEventListener('click', function() {
          self._statusFilter = btn.dataset.respFilter;
          document.querySelectorAll('#responses-filter [data-resp-filter]').forEach(function(b) {
            b.classList.toggle('active', b === btn);
          });
          self.load();
        });
      });
    },

    enter: function() {
      this.load();
      // Light auto-refresh while the panel is visible — picks up status
      // changes for active background jobs.
      var self = this;
      this.leave();
      this._refreshTimer = setInterval(function() {
        if (App.state && App.state.activeTab === 'responses') {
          self.load();
        } else {
          self.leave();
        }
      }, 5000);
    },

    leave: function() {
      if (this._refreshTimer) {
        clearInterval(this._refreshTimer);
        this._refreshTimer = null;
      }
    },

    load: async function() {
      var body = document.getElementById('responses-body');
      if (!body) return;
      try {
        var url = '/api/admin/responses?limit=200';
        if (this._statusFilter && this._statusFilter !== 'all') {
          url += '&status=' + encodeURIComponent(this._statusFilter);
        }
        var resp = await App.data.authFetch(url);
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        var data = await resp.json();
        this._records = data.data || [];
        this.render();
      } catch (e) {
        body.innerHTML = '<tr><td colspan="6" class="text-center text-muted">' +
          U.escapeHtml(I18n.t('responses.load_error', { error: String(e) })) + '</td></tr>';
      }
    },

    render: function() {
      var body = document.getElementById('responses-body');
      if (!body) return;
      var rows = this._records;
      if (!rows.length) {
        body.innerHTML = '<tr><td colspan="6" class="text-center text-muted">' +
          U.escapeHtml(I18n.t('responses.empty')) + '</td></tr>';
        return;
      }
      var self = this;
      body.innerHTML = '';
      rows.forEach(function(r) {
        var tr = document.createElement('tr');
        tr.appendChild(self._cell(r.id, 'responses-id-cell'));
        tr.appendChild(self._cell(r.model, 'text-sm'));
        tr.appendChild(self._statusCell(r));
        tr.appendChild(self._cell(self._formatDate(r.created_at), 'text-sm text-muted'));
        tr.appendChild(self._cell(r.input_preview || '', 'text-sm responses-preview'));
        tr.appendChild(self._actionsCell(r));
        body.appendChild(tr);
      });
    },

    _cell: function(text, cls) {
      var td = document.createElement('td');
      if (cls) td.className = cls;
      td.textContent = text == null ? '' : String(text);
      return td;
    },

    _statusCell: function(r) {
      var td = document.createElement('td');
      var span = document.createElement('span');
      var status = r.status || 'unknown';
      span.className = 'status-badge status-' + status;
      span.textContent = I18n.t('responses.status_' + status, status);
      td.appendChild(span);
      if (r.live) {
        var live = document.createElement('span');
        live.className = 'text-xs text-muted';
        live.style.marginLeft = '6px';
        live.textContent = I18n.t('responses.live_tag');
        td.appendChild(live);
      }
      return td;
    },

    _actionsCell: function(r) {
      var td = document.createElement('td');
      td.className = 'text-right';
      var self = this;

      var viewBtn = document.createElement('button');
      viewBtn.className = 'btn btn-xs';
      viewBtn.textContent = I18n.t('responses.view_btn');
      viewBtn.addEventListener('click', function() { self.retrieve(r.id); });
      td.appendChild(viewBtn);

      // Cancel only meaningful when the response is in flight.
      if (r.status === 'queued' || r.status === 'in_progress' || r.live) {
        var cancelBtn = document.createElement('button');
        cancelBtn.className = 'btn btn-xs btn-danger';
        cancelBtn.style.marginLeft = '4px';
        cancelBtn.textContent = I18n.t('responses.cancel_btn');
        cancelBtn.addEventListener('click', function() { self.cancel(r.id); });
        td.appendChild(cancelBtn);
      }

      var delBtn = document.createElement('button');
      delBtn.className = 'btn btn-xs';
      delBtn.style.marginLeft = '4px';
      delBtn.textContent = I18n.t('responses.delete_btn');
      delBtn.addEventListener('click', function() { self.del(r.id); });
      td.appendChild(delBtn);

      return td;
    },

    _formatDate: function(epochSeconds) {
      if (!epochSeconds) return '';
      var d = new Date(epochSeconds * 1000);
      return d.toLocaleString();
    },

    retrieve: async function(id) {
      if (!id) return;
      var detail = document.getElementById('responses-detail');
      if (!detail) return;
      detail.style.display = 'block';
      detail.textContent = I18n.t('responses.loading');
      try {
        var resp = await App.data.authFetch('/v1/responses/' + encodeURIComponent(id));
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        var json = await resp.json();
        detail.innerHTML = '<pre class="responses-detail-pre">' + U.escapeHtml(JSON.stringify(json, null, 2)) + '</pre>';
      } catch (e) {
        detail.textContent = I18n.t('responses.retrieve_error', { error: String(e) });
      }
    },

    cancel: async function(id) {
      try {
        var resp = await App.data.authFetch('/v1/responses/' + encodeURIComponent(id) + '/cancel', { method: 'POST' });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        App.notifications.toast(I18n.t('responses.cancelled_toast', { id: id }), 'info', 3000);
        this.load();
      } catch (e) {
        App.notifications.toast(I18n.t('responses.cancel_error', { error: String(e) }), 'error', 4000);
      }
    },

    del: async function(id) {
      if (!confirm(I18n.t('responses.delete_confirm', { id: id }))) return;
      try {
        var resp = await App.data.authFetch('/v1/responses/' + encodeURIComponent(id), { method: 'DELETE' });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        App.notifications.toast(I18n.t('responses.deleted_toast', { id: id }), 'info', 3000);
        this.load();
      } catch (e) {
        App.notifications.toast(I18n.t('responses.delete_error', { error: String(e) }), 'error', 4000);
      }
    },
  };
})();
