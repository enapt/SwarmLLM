'use strict';

// ============================================================================
// SwarmLLM — Claude Code Integration Component
// Handles: session creation, project picker, tool rendering, permissions,
// SSE event parsing for Claude Code bidirectional sessions.
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

  App.claudeCode = {
    // Check if a model ID is a Claude subscription model
    isClaudeCodeModel: function(modelId) {
      if (!modelId) return false;
      var match = S._modelDropdownData.find(function(m) { return m.id === modelId; });
      return match && match.group === 'claude_subscription';
    },

    // Get or initialize claude_code metadata for a session
    getSessionCC: function(session) {
      if (!session.claude_code) {
        session.claude_code = {
          active: false,
          working_dir: null,
          claude_session_id: null,
          tools_available: [],
          permission_mode: 'acceptEdits',
          state: null,
        };
      }
      return session.claude_code;
    },

    // Show the project picker bar
    showProjectPicker: function() {
      var bar = document.getElementById('cc-project-bar');
      if (bar) bar.style.display = '';
    },

    // Hide the project picker bar
    hideProjectPicker: function() {
      var bar = document.getElementById('cc-project-bar');
      if (bar) bar.style.display = 'none';
    },

    // Create a Claude Code backend session
    createSession: async function(sessionId, model, workingDir, permissionMode) {
      var body = {
        session_id: sessionId,
        model: model,
        working_dir: workingDir || '',
        permission_mode: permissionMode || 'acceptEdits',
      };

      // Check if we have a stored claude_session_id for resume
      var session = S.sessions[sessionId];
      if (session && session.claude_code && session.claude_code.claude_session_id) {
        body.resume_claude_session_id = session.claude_code.claude_session_id;
      }

      try {
        var resp = await App.authFetch('/api/claude-code/session', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          _timeout: 130000, // 130s — hooks (SessionStart) can take 30-60s before init
        });

        if (!resp.ok) {
          var errText = await resp.text();
          throw new Error(errText);
        }

        var data = await resp.json();

        // Update session metadata
        if (session) {
          var cc = App.claudeCode.getSessionCC(session);
          cc.active = true;
          cc.working_dir = workingDir || data.working_dir || null;
          cc.claude_session_id = data.claude_session_id || null;
          cc.tools_available = data.tools || [];
          cc.state = data.status || 'active';
          App.chat.saveSessions();
        }

        return data;
      } catch (e) {
        App.notifications.showToast(
          I18n.t('claude_code.session_failed') + ': ' + (e.message || ''),
          'error', 5000
        );
        throw e;
      }
    },

    // Send a message through the Claude Code session and handle the SSE stream
    sendMessage: async function(sessionId, content, contentEl, assistantEl) {
      var resp = await App.authFetch('/api/claude-code/session/' + encodeURIComponent(sessionId) + '/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: content }),
        _timeout: 0, // no timeout — long-running tool calls
      });

      if (!resp.ok) {
        var errText = await resp.text();
        throw new Error(errText);
      }

      var fullContent = '';
      var cleared = false;
      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buffer = '';
      var pendingPermission = null;
      var toolPanels = {};
      var startTime = performance.now();
      var timerInterval = null;

      // Live elapsed timer
      var timerEl = document.createElement('div');
      timerEl.className = 'msg-timer cc-live-timer';
      var timerTarget = assistantEl.querySelector('.msg-bubble') || assistantEl;
      timerTarget.appendChild(timerEl);
      timerInterval = setInterval(function() {
        var elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
        timerEl.textContent = elapsed + 's';
      }, 100);

      try {
        while (true) {
          var result = await reader.read();
          if (result.done) break;

          buffer += decoder.decode(result.value, { stream: true });
          var lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (var i = 0; i < lines.length; i++) {
            var line = lines[i].trim();
            if (!line.startsWith('data:')) continue;
            var payload = line.substring(5).trim();
            if (payload === '[DONE]') continue;

            try {
              var evt = JSON.parse(payload);
              App.claudeCode._handleEvent(evt, contentEl, assistantEl, toolPanels, {
                cleared: cleared,
                setClear: function() { cleared = true; },
                appendText: function(text) { fullContent += text; },
                getFullContent: function() { return fullContent; },
                setPendingPermission: function(p) { pendingPermission = p; },
              });
              // Update cleared flag from handler
              if (contentEl.querySelector('.response-text') || contentEl.querySelector('.cc-tool-call')) {
                cleared = true;
              }
            } catch (e) {}
          }
        }
      } finally {
        clearInterval(timerInterval);
        // Final timer
        var elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
        timerEl.textContent = I18n.t('chat.response_time', { seconds: elapsed });
        timerEl.classList.remove('cc-live-timer');
      }

      return { content: fullContent, pendingPermission: pendingPermission };
    },

    // Handle a single NDJSON event from the Claude Code stream
    _handleEvent: function(evt, contentEl, assistantEl, toolPanels, ctx) {
      var evtType = evt.type || '';

      switch (evtType) {
        case 'stream_event':
          App.claudeCode._handleStreamEvent(evt, contentEl, ctx);
          break;

        case 'assistant':
          // Complete assistant turn — may contain tool_use blocks
          App.claudeCode._handleAssistantMessage(evt, contentEl, toolPanels, ctx);
          break;

        case 'user':
          // Tool results
          if (evt.message && evt.message.content) {
            App.claudeCode._handleToolResult(evt, contentEl, toolPanels);
          }
          break;

        case 'control_request':
          App.claudeCode._handlePermissionRequest(evt, contentEl, ctx);
          break;

        case 'result':
          App.claudeCode._handleResult(evt, contentEl);
          break;

        case 'system':
          // api_retry, compact_boundary — show status
          if (evt.subtype === 'api_retry') {
            App.claudeCode._showStatus(contentEl, I18n.t('claude_code.retrying', { attempt: evt.attempt || 1 }));
          }
          break;

        case 'error':
          App.claudeCode._showStatus(contentEl, evt.message || 'Unknown error', true);
          break;
      }
    },

    // Handle streaming text deltas
    _handleStreamEvent: function(evt, contentEl, ctx) {
      var inner = evt.event || {};
      var innerType = inner.type || '';

      if (innerType === 'content_block_delta') {
        var deltaType = (inner.delta || {}).type || '';
        if (deltaType === 'text_delta') {
          var text = inner.delta.text || '';
          if (!ctx.cleared) {
            contentEl.textContent = '';
            ctx.setClear();
          }
          ctx.appendText(text);
          var textNode = contentEl.querySelector('.response-text');
          if (!textNode) {
            textNode = document.createElement('div');
            textNode.className = 'response-text';
            contentEl.appendChild(textNode);
          }
          textNode.textContent = ctx.getFullContent();
          App.chat.scrollToBottom();
        } else if (deltaType === 'thinking_delta') {
          // Extended thinking
          var thinkText = inner.delta.thinking || '';
          if (!ctx.cleared) { contentEl.textContent = ''; ctx.setClear(); }
          var thinkingEl = contentEl.querySelector('.reasoning-block');
          if (!thinkingEl) {
            thinkingEl = document.createElement('details');
            thinkingEl.className = 'reasoning-block';
            thinkingEl.innerHTML = '<summary>' + U.escapeHtml(I18n.t('chat.reasoning_label')) + '</summary><pre class="reasoning-content"></pre>';
            thinkingEl.open = true;
            contentEl.appendChild(thinkingEl);
          }
          var preEl = thinkingEl.querySelector('.reasoning-content');
          if (preEl) preEl.textContent += thinkText;
          App.chat.scrollToBottom();
        }
      }
    },

    // Handle complete assistant message (may contain tool_use)
    _handleAssistantMessage: function(evt, contentEl, toolPanels, ctx) {
      var msg = evt.message || {};
      var content = msg.content || [];
      if (!Array.isArray(content)) return;

      content.forEach(function(block) {
        if (block.type === 'text' && block.text) {
          // Text already streamed via stream_event — skip unless not streamed
          if (!ctx.getFullContent()) {
            if (!ctx.cleared) { contentEl.textContent = ''; ctx.setClear(); }
            ctx.appendText(block.text);
            var textNode = contentEl.querySelector('.response-text');
            if (!textNode) {
              textNode = document.createElement('div');
              textNode.className = 'response-text';
              contentEl.appendChild(textNode);
            }
            textNode.textContent = block.text;
          }
        } else if (block.type === 'tool_use') {
          if (!ctx.cleared) { contentEl.textContent = ''; ctx.setClear(); }
          App.claudeCode._renderToolCall(contentEl, block, toolPanels);
        }
      });
      App.chat.scrollToBottom();
    },

    // Render a tool call block
    _renderToolCall: function(contentEl, block, toolPanels) {
      var toolId = block.id || '';
      var toolName = block.name || 'Unknown';
      var input = block.input || {};

      var panel = document.createElement('div');
      panel.className = 'cc-tool-call';
      panel.setAttribute('data-tool-id', toolId);

      var fileHint = input.file_path || input.command || input.pattern || input.url || '';
      if (fileHint.length > 80) fileHint = '...' + fileHint.slice(-77);

      var icon = App.claudeCode._toolIcon(toolName);
      panel.innerHTML =
        '<div class="cc-tool-header">' +
          '<span class="cc-tool-icon">' + icon + '</span>' +
          '<span class="cc-tool-name">' + U.escapeHtml(toolName) + '</span>' +
          (fileHint ? '<span class="cc-tool-file">' + U.escapeHtml(fileHint) + '</span>' : '') +
          '<span class="cc-tool-status pending">' + U.escapeHtml(I18n.t('claude_code.running')) + '</span>' +
        '</div>';

      // Show input details for certain tools
      if (toolName === 'Bash' && input.command) {
        var pre = document.createElement('pre');
        pre.className = 'cc-tool-input cc-bash-cmd';
        pre.textContent = '$ ' + input.command;
        panel.appendChild(pre);
      } else if (toolName === 'Edit' && input.old_string) {
        var pre = document.createElement('pre');
        pre.className = 'cc-tool-input';
        pre.textContent = input.old_string + ' → ' + (input.new_string || '');
        if (pre.textContent.length > 200) pre.textContent = pre.textContent.substring(0, 200) + '...';
        panel.appendChild(pre);
      }

      contentEl.appendChild(panel);
      toolPanels[toolId] = panel;
    },

    // Handle tool result
    _handleToolResult: function(evt, contentEl, toolPanels) {
      var msg = evt.message || {};
      var content = msg.content || [];
      if (!Array.isArray(content)) {
        content = [{ type: 'tool_result', tool_use_id: '', content: String(msg.content || '') }];
      }

      var toolResult = evt.tool_use_result || {};

      content.forEach(function(block) {
        if (block.type !== 'tool_result') return;
        var toolId = block.tool_use_id || '';
        var panel = toolPanels[toolId];

        // Update status to done
        if (panel) {
          var statusEl = panel.querySelector('.cc-tool-status');
          if (statusEl) {
            statusEl.textContent = I18n.t('claude_code.done');
            statusEl.className = 'cc-tool-status done';
          }
        }

        // Render result
        var resultEl = document.createElement('div');
        resultEl.className = 'cc-tool-result';

        // Check for git diff (Edit/Write results)
        if (toolResult.gitDiff && toolResult.gitDiff.patch) {
          resultEl.className += ' cc-diff';
          var diffHtml = '<div class="cc-diff-header">' +
            '<span class="cc-diff-file">' + U.escapeHtml(toolResult.gitDiff.filename || '') + '</span>' +
            '<span class="cc-diff-stats">+' + (toolResult.gitDiff.additions || 0) + ' -' + (toolResult.gitDiff.deletions || 0) + '</span>' +
            '</div>';
          diffHtml += '<pre class="cc-diff-content">' + App.claudeCode._formatDiff(toolResult.gitDiff.patch) + '</pre>';
          resultEl.innerHTML = diffHtml;
        } else if (toolResult.stdout !== undefined || toolResult.stderr !== undefined) {
          // Bash result
          resultEl.className += ' cc-bash';
          var output = (toolResult.stdout || '') + (toolResult.stderr ? '\n' + toolResult.stderr : '');
          if (output.length > 2000) output = output.substring(0, 2000) + '\n... (truncated)';
          resultEl.innerHTML = '<pre class="cc-bash-output">' + U.escapeHtml(output) + '</pre>';
        } else if (toolResult.filenames) {
          // Glob/Grep result
          var count = toolResult.numFiles || (toolResult.filenames || []).length || 0;
          resultEl.innerHTML = '<div class="cc-file-summary">' + count + ' file(s) found</div>';
        } else if (typeof block.content === 'string' && block.content.length > 0) {
          // Generic text result
          var text = block.content;
          if (text.length > 1500) text = text.substring(0, 1500) + '\n... (truncated)';
          resultEl.innerHTML = '<pre class="cc-tool-output">' + U.escapeHtml(text) + '</pre>';
        }

        if (panel) {
          panel.appendChild(resultEl);
        } else {
          contentEl.appendChild(resultEl);
        }
      });

      App.chat.scrollToBottom();
    },

    // Handle permission request — show approve/deny UI
    _handlePermissionRequest: function(evt, contentEl, ctx) {
      var req = evt.request || {};
      var toolName = req.tool_name || 'Unknown';
      var input = req.input || {};
      var requestId = evt.request_id || '';
      var sessionId = S.currentSessionId || '';

      var panel = document.createElement('div');
      panel.className = 'cc-permission-prompt';

      var desc = '';
      if (toolName === 'Bash') desc = input.command || '';
      else if (toolName === 'Edit') desc = (input.file_path || '') + ': ' + (input.old_string || '').substring(0, 50) + ' → ...';
      else if (toolName === 'Write') desc = input.file_path || '';
      else desc = JSON.stringify(input).substring(0, 100);

      panel.innerHTML =
        '<div class="cc-perm-header">' + U.escapeHtml(I18n.t('claude_code.permission_prompt')) + '</div>' +
        '<div class="cc-perm-tool">' +
          '<span class="cc-tool-icon">' + App.claudeCode._toolIcon(toolName) + '</span> ' +
          '<strong>' + U.escapeHtml(toolName) + '</strong>: ' + U.escapeHtml(desc) +
        '</div>' +
        '<div class="cc-perm-actions">' +
          '<button class="btn btn-sm cc-perm-allow" data-cc-perm-id="' + U.escapeHtml(requestId) + '">' + U.escapeHtml(I18n.t('claude_code.allow')) + '</button>' +
          '<button class="btn btn-sm cc-perm-deny" data-cc-perm-id="' + U.escapeHtml(requestId) + '">' + U.escapeHtml(I18n.t('claude_code.deny')) + '</button>' +
        '</div>';

      contentEl.appendChild(panel);
      ctx.setPendingPermission({ requestId: requestId, element: panel });

      // Bind click handlers
      panel.querySelector('.cc-perm-allow').addEventListener('click', function() {
        App.claudeCode._respondPermission(sessionId, requestId, true, panel);
      });
      panel.querySelector('.cc-perm-deny').addEventListener('click', function() {
        App.claudeCode._respondPermission(sessionId, requestId, false, panel);
      });

      App.chat.scrollToBottom();
    },

    // Send permission response
    _respondPermission: async function(sessionId, requestId, allow, panelEl) {
      try {
        await App.authFetch('/api/claude-code/session/' + encodeURIComponent(sessionId) + '/permission', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ request_id: requestId, allow: allow }),
        });
        panelEl.querySelector('.cc-perm-actions').innerHTML =
          '<span class="cc-perm-resolved">' + (allow ? '✓ ' + I18n.t('claude_code.allowed') : '✗ ' + I18n.t('claude_code.denied')) + '</span>';
        panelEl.classList.add(allow ? 'cc-perm-allowed' : 'cc-perm-denied');
      } catch (e) {
        App.notifications.showToast(I18n.t('common.request_failed'), 'error', 3000);
      }
    },

    // Handle final result event
    _handleResult: function(evt, contentEl) {
      if (evt.is_error) {
        App.claudeCode._showStatus(contentEl, evt.result || 'Error', true);
      }
      // Cost info
      if (evt.total_cost_usd) {
        var costEl = document.createElement('div');
        costEl.className = 'cc-cost-info';
        costEl.textContent = I18n.t('claude_code.cost', {
          cost: '$' + evt.total_cost_usd.toFixed(4),
          turns: evt.num_turns || 1,
        });
        contentEl.appendChild(costEl);
      }
    },

    // Show a status message in the content area
    _showStatus: function(contentEl, message, isError) {
      var el = document.createElement('div');
      el.className = 'cc-status' + (isError ? ' cc-status-error' : '');
      el.textContent = message;
      contentEl.appendChild(el);
    },

    // Format a unified diff patch into HTML with color highlighting
    _formatDiff: function(patch) {
      if (!patch) return '';
      return patch.split('\n').map(function(line) {
        if (line.startsWith('+') && !line.startsWith('+++')) {
          return '<span class="diff-add">' + U.escapeHtml(line) + '</span>';
        } else if (line.startsWith('-') && !line.startsWith('---')) {
          return '<span class="diff-del">' + U.escapeHtml(line) + '</span>';
        } else if (line.startsWith('@@')) {
          return '<span class="diff-hunk">' + U.escapeHtml(line) + '</span>';
        }
        return U.escapeHtml(line);
      }).join('\n');
    },

    // Get an icon for a tool name
    _toolIcon: function(name) {
      var icons = {
        Bash: '⌨',
        Read: '📄',
        Edit: '✏',
        Write: '📝',
        Glob: '🔍',
        Grep: '🔎',
        WebSearch: '🌐',
        WebFetch: '🌐',
        Agent: '🤖',
        TodoWrite: '📋',
      };
      return icons[name] || '⚡';
    },

    // Update the project picker bar visibility based on current model
    updateProjectBar: function() {
      var session = S.currentSessionId ? S.sessions[S.currentSessionId] : null;
      var model = session ? session.model : S.currentModel;
      var bar = document.getElementById('cc-project-bar');
      if (!bar) return;

      if (App.claudeCode.isClaudeCodeModel(model)) {
        bar.style.display = '';
        // If session already has CC active, show working dir
        if (session && session.claude_code && session.claude_code.active) {
          var dirLabel = document.getElementById('cc-working-dir-label');
          if (dirLabel) {
            dirLabel.textContent = session.claude_code.working_dir || I18n.t('claude_code.quick_chat');
          }
          var picker = document.getElementById('cc-dir-picker');
          if (picker) picker.style.display = 'none';
          var info = document.getElementById('cc-session-info');
          if (info) info.style.display = '';
        } else {
          var picker = document.getElementById('cc-dir-picker');
          if (picker) picker.style.display = '';
          var info = document.getElementById('cc-session-info');
          if (info) info.style.display = 'none';
        }
      } else {
        bar.style.display = 'none';
      }
    },

    // Initialize event bindings
    init: function() {
      // Quick chat button
      var quickBtn = document.getElementById('cc-quick-chat-btn');
      if (quickBtn) {
        quickBtn.addEventListener('click', function() {
          var dirInput = document.getElementById('cc-dir-input');
          if (dirInput) dirInput.value = '';
          // Will be created on first send()
          App.claudeCode.updateProjectBar();
        });
      }

      // Permission mode selector
      var permSelect = document.getElementById('cc-permission-mode');
      if (permSelect) {
        permSelect.addEventListener('change', function() {
          var session = S.currentSessionId ? S.sessions[S.currentSessionId] : null;
          if (session && session.claude_code) {
            session.claude_code.permission_mode = permSelect.value;
            App.chat.saveSessions();
          }
        });
      }
    },
  };
})();
