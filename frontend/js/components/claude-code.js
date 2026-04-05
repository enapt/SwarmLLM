'use strict';

// ============================================================================
// SwarmLLM — Claude Code Integration Component
// Handles: session creation, project picker, tool rendering, permissions,
// SSE event parsing for Claude Code bidirectional sessions.
// ============================================================================

(function() {
  var S = App.state;
  var U = App.utils;

  // ── Agent/task tool detection sets ──
  var AGENT_TOOLS = { Agent: 1, SendMessage: 1, TeamCreate: 1 };
  var TASK_TOOLS = { TaskCreate: 1, TaskUpdate: 1, TaskGet: 1, TaskList: 1, TaskStop: 1, TaskOutput: 1 };

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
          permission_mode: 'bypassPermissions',
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
        permission_mode: permissionMode || 'bypassPermissions',
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
          cc.slash_commands = data.slash_commands || [];
          cc.state = data.status || 'active';
          cc.mcp_connected = data.mcp_connected || false;
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
      var agentPanels = {};  // toolId → { panel, contentArea, taskList }
      var taskItems = {};    // taskId → { element, status }
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
              App.claudeCode._handleEvent(evt, contentEl, assistantEl, toolPanels, agentPanels, taskItems, {
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

    // Resolve the target container — if event has parent_tool_use_id pointing
    // to an agent panel, render inside that agent's content area instead.
    _resolveTarget: function(evt, contentEl, agentPanels) {
      var parentId = evt.parent_tool_use_id;
      if (parentId && agentPanels[parentId]) {
        return agentPanels[parentId].contentArea;
      }
      return contentEl;
    },

    // Handle a single NDJSON event from the Claude Code stream
    _handleEvent: function(evt, contentEl, assistantEl, toolPanels, agentPanels, taskItems, ctx) {
      var evtType = evt.type || '';
      var target = App.claudeCode._resolveTarget(evt, contentEl, agentPanels);


      switch (evtType) {
        case 'stream_event':
          App.claudeCode._handleStreamEvent(evt, target, ctx);
          break;

        case 'assistant':
          // Complete assistant turn — may contain tool_use blocks
          App.claudeCode._handleAssistantMessage(evt, target, toolPanels, agentPanels, taskItems, ctx);
          break;

        case 'user':
          // Tool results
          if (evt.message && evt.message.content) {
            App.claudeCode._handleToolResult(evt, target, toolPanels, agentPanels, taskItems);
          }
          break;

        case 'control_request':
          App.claudeCode._handlePermissionRequest(evt, target, ctx);
          break;

        case 'result':
          App.claudeCode._handleResult(evt, contentEl);
          break;

        case 'system':
          // api_retry, compact_boundary — show status
          if (evt.subtype === 'api_retry') {
            App.claudeCode._showStatus(target, I18n.t('claude_code.retrying', { attempt: evt.attempt || 1 }));
          }
          break;

        case 'error':
          App.claudeCode._showStatus(target, evt.message || 'Unknown error', true);
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
    _handleAssistantMessage: function(evt, contentEl, toolPanels, agentPanels, taskItems, ctx) {
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
          var toolName = block.name || '';
          if (AGENT_TOOLS[toolName]) {
            App.claudeCode._renderAgentCall(contentEl, block, agentPanels);
          } else if (TASK_TOOLS[toolName]) {
            App.claudeCode._renderTaskCall(contentEl, block, toolPanels, taskItems);
          } else {
            App.claudeCode._renderToolCall(contentEl, block, toolPanels);
          }
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

    // Update the running agents summary bar
    _updateAgentSummary: function(contentEl, agentPanels) {
      var running = 0;
      var bg = 0;
      for (var id in agentPanels) {
        var p = agentPanels[id];
        var statusEl = p.panel.querySelector('.cc-tool-status');
        if (statusEl && statusEl.classList.contains('pending')) {
          if (p.background) bg++; else running++;
        }
      }
      var bar = contentEl.querySelector('.cc-agents-summary');
      if (running + bg > 1) {
        if (!bar) {
          bar = document.createElement('div');
          bar.className = 'cc-agents-summary';
          // Insert before first agent panel
          var firstAgent = contentEl.querySelector('.cc-agent-panel');
          if (firstAgent) contentEl.insertBefore(bar, firstAgent);
          else contentEl.appendChild(bar);
        }
        var parts = [];
        if (running > 0) parts.push(running + ' ' + I18n.t('claude_code.agents_running'));
        if (bg > 0) parts.push(bg + ' ' + I18n.t('claude_code.agents_background'));
        bar.textContent = parts.join(' · ');
        bar.style.display = '';
      } else if (bar) {
        bar.style.display = 'none';
      }
    },

    // Render an Agent/SendMessage/TeamCreate tool call as a collapsible sub-agent panel
    _renderAgentCall: function(contentEl, block, agentPanels) {
      var toolId = block.id || '';
      var toolName = block.name || 'Agent';
      var input = block.input || {};

      var panel = document.createElement('details');
      panel.className = 'cc-agent-panel';
      panel.setAttribute('data-tool-id', toolId);
      panel.open = true;

      var desc = input.description || input.prompt || '';
      if (desc.length > 120) desc = desc.substring(0, 117) + '...';

      var agentType = input.subagent_type || '';
      var model = input.model || '';
      var targetName = input.to || input.name || '';

      var icon = '🤖';
      if (toolName === 'SendMessage') icon = '💬';
      else if (toolName === 'TeamCreate') icon = '👥';

      var metaParts = [];
      if (agentType) metaParts.push(U.escapeHtml(agentType));
      if (model) metaParts.push(U.escapeHtml(model));
      if (targetName) metaParts.push(U.escapeHtml(targetName));
      var metaHtml = metaParts.length ? '<span class="cc-agent-meta">' + metaParts.join(' · ') + '</span>' : '';

      var summary = document.createElement('summary');
      summary.className = 'cc-agent-header';
      summary.innerHTML =
        '<span class="cc-agent-icon">' + icon + '</span>' +
        '<span class="cc-agent-label">' + U.escapeHtml(I18n.t('claude_code.agent_spawned')) + '</span>' +
        metaHtml +
        '<span class="cc-tool-status pending">' + U.escapeHtml(I18n.t('claude_code.running')) + '</span>';
      panel.appendChild(summary);

      if (desc) {
        var descEl = document.createElement('div');
        descEl.className = 'cc-agent-desc';
        descEl.textContent = desc;
        panel.appendChild(descEl);
      }

      // Content area where nested sub-agent events will render
      var contentArea = document.createElement('div');
      contentArea.className = 'cc-agent-content';
      panel.appendChild(contentArea);

      var isBg = input.run_in_background === true;
      if (isBg) {
        panel.classList.add('cc-agent-bg');
        panel.open = false; // collapsed by default for background agents
        // Add to background tray instead of main content
        var tray = contentEl.querySelector('.cc-bg-tray');
        if (!tray) {
          tray = document.createElement('div');
          tray.className = 'cc-bg-tray';
          tray.innerHTML = '<div class="cc-bg-tray-label">' + U.escapeHtml(I18n.t('claude_code.background_agents')) + '</div>';
          contentEl.appendChild(tray);
        }
        tray.appendChild(panel);
      } else {
        contentEl.appendChild(panel);
      }
      agentPanels[toolId] = { panel: panel, contentArea: contentArea, background: isBg };
      App.claudeCode._updateAgentSummary(contentEl, agentPanels);
    },

    // Render TaskCreate/TaskUpdate/TaskGet as compact task items
    _renderTaskCall: function(contentEl, block, toolPanels, taskItems) {
      var toolId = block.id || '';
      var toolName = block.name || '';
      var input = block.input || {};

      // Find or create task list container
      var taskList = contentEl.querySelector('.cc-task-list');
      if (!taskList) {
        taskList = document.createElement('div');
        taskList.className = 'cc-task-list';
        taskList.innerHTML = '<div class="cc-task-list-header">' +
          '<span class="cc-task-list-icon">📋</span>' +
          '<span class="cc-task-list-label">' + U.escapeHtml(I18n.t('claude_code.task_list')) + '</span>' +
          '</div>';
        contentEl.appendChild(taskList);
      }

      if (toolName === 'TaskCreate') {
        var item = document.createElement('div');
        item.className = 'cc-task-item cc-task-pending';
        item.setAttribute('data-tool-id', toolId);
        var subject = input.subject || input.description || '';
        item.innerHTML =
          '<span class="cc-task-check">○</span>' +
          '<span class="cc-task-subject">' + U.escapeHtml(subject) + '</span>';
        taskList.appendChild(item);
        // We'll match by tool_result to get the actual taskId later
        toolPanels[toolId] = item;
      } else if (toolName === 'TaskUpdate') {
        var taskId = input.taskId || '';
        var status = input.status || '';
        // Update existing task item if we have it
        if (taskItems[taskId]) {
          var el = taskItems[taskId].element;
          if (status === 'completed') {
            el.className = 'cc-task-item cc-task-completed';
            el.querySelector('.cc-task-check').textContent = '✓';
          } else if (status === 'in_progress') {
            el.className = 'cc-task-item cc-task-in-progress';
            el.querySelector('.cc-task-check').textContent = '◉';
          } else if (status === 'deleted') {
            el.className = 'cc-task-item cc-task-deleted';
            el.querySelector('.cc-task-check').textContent = '✗';
          }
          taskItems[taskId].status = status;
        }
        // Also track tool panel for the result
        toolPanels[toolId] = taskList;
      } else {
        // TaskGet, TaskList, TaskStop, TaskOutput — just track for result
        toolPanels[toolId] = taskList;
      }
    },

    // Handle tool result
    _handleToolResult: function(evt, contentEl, toolPanels, agentPanels, taskItems) {
      var msg = evt.message || {};
      var content = msg.content || [];
      if (!Array.isArray(content)) {
        content = [{ type: 'tool_result', tool_use_id: '', content: String(msg.content || '') }];
      }

      content.forEach(function(block) {
        if (block.type !== 'tool_result') return;
        var toolId = block.tool_use_id || '';

        // Extract text from block.content (string or array of content blocks)
        var blockText = App.claudeCode._extractResultText(block.content);

        // ── Agent result — mark panel done, collapse/expand ──
        if (agentPanels[toolId]) {
          var agentInfo = agentPanels[toolId];
          var statusEl = agentInfo.panel.querySelector('.cc-tool-status');
          if (statusEl) {
            statusEl.textContent = I18n.t('claude_code.done');
            statusEl.className = 'cc-tool-status done';
          }
          if (blockText) {
            var summaryEl = document.createElement('div');
            summaryEl.className = 'cc-agent-summary';
            var summaryText = blockText.length > 500 ? blockText.substring(0, 497) + '...' : blockText;
            summaryEl.textContent = summaryText;
            agentInfo.contentArea.appendChild(summaryEl);
          }
          // Background agents: expand briefly to show completion
          if (agentInfo.background) {
            agentInfo.panel.open = true;
            agentInfo.panel.classList.add('cc-agent-bg-done');
          } else {
            agentInfo.panel.open = false;
          }
          App.claudeCode._updateAgentSummary(contentEl, agentPanels);
          App.chat.scrollToBottom();
          return;
        }

        // ── TaskCreate result — capture the real taskId ──
        var panel = toolPanels[toolId];
        if (panel && panel.classList && panel.classList.contains('cc-task-item')) {
          var idMatch = blockText.match(/#(\d+)/);
          if (idMatch) {
            taskItems[idMatch[1]] = { element: panel, status: 'pending' };
          }
          App.chat.scrollToBottom();
          return;
        }

        // ── Standard tool result ──
        if (panel) {
          var statusEl2 = panel.querySelector('.cc-tool-status');
          if (statusEl2) {
            statusEl2.textContent = I18n.t('claude_code.done');
            statusEl2.className = 'cc-tool-status done';
          }
        }

        // Render result as text (Claude CLI returns tool results as plain text)
        var resultEl = document.createElement('div');
        resultEl.className = 'cc-tool-result';

        if (blockText.length > 0) {
          var displayText = blockText.length > 2000 ? blockText.substring(0, 2000) + '\n... (truncated)' : blockText;
          resultEl.innerHTML = '<pre class="cc-tool-output">' + U.escapeHtml(displayText) + '</pre>';
        }

        if (panel && panel.appendChild) {
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
      // If no text was streamed, show the result text as the response
      // (happens when Claude only uses thinking/tools with no visible text output)
      var resultText = evt.result || '';
      if (resultText && !contentEl.querySelector('.response-text') && !contentEl.querySelector('.cc-tool-call')) {
        // Clear the "Thinking..." indicator
        var typing = contentEl.querySelector('.typing-indicator');
        if (typing) typing.remove();
        contentEl.textContent = '';
        var textNode = document.createElement('div');
        textNode.className = 'response-text';
        textNode.textContent = resultText;
        contentEl.appendChild(textNode);
      }

      if (evt.is_error) {
        App.claudeCode._showStatus(contentEl, resultText || 'Error', true);
      }

      // Show turns + duration (skip cost for subscription — it's not real API billing)
      var turns = evt.num_turns || 1;
      var duration = evt.duration_ms ? (evt.duration_ms / 1000).toFixed(1) + 's' : '';
      if (turns > 0 || duration) {
        var infoEl = document.createElement('div');
        infoEl.className = 'cc-cost-info';
        var parts = [];
        if (turns > 1) parts.push(turns + ' turns');
        if (duration) parts.push(duration);
        if (parts.length) infoEl.textContent = parts.join(' · ');
        if (infoEl.textContent) contentEl.appendChild(infoEl);
      }
    },

    // Show a status message in the content area
    _showStatus: function(contentEl, message, isError) {
      var el = document.createElement('div');
      el.className = 'cc-status' + (isError ? ' cc-status-error' : '');
      el.textContent = message;
      contentEl.appendChild(el);
    },

    // Extract text from a tool_result content field (string or array of blocks).
    _extractResultText: function(content) {
      if (typeof content === 'string') return content;
      if (Array.isArray(content)) {
        return content.map(function(part) {
          if (typeof part === 'string') return part;
          if (part && part.type === 'text' && part.text) return part.text;
          return '';
        }).filter(Boolean).join('\n');
      }
      return String(content || '');
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
        SendMessage: '💬',
        TeamCreate: '👥',
        TaskCreate: '📋',
        TaskUpdate: '📋',
        TaskGet: '📋',
        TaskList: '📋',
        TaskStop: '🛑',
        TaskOutput: '📋',
        TodoWrite: '📋',
        LSP: '🔗',
        NotebookEdit: '📓',
      };
      return icons[name] || '⚡';
    },

    // Pass slash commands directly to Claude Code CLI — it has its own
    // built-in commands (/compact, /cost, /review, etc.) plus user-configured
    // skills. The system/init event lists all available slash_commands.
    // Returns the text unchanged if it's a slash command, null otherwise.
    translateSlashCommand: function(text) {
      if (!text || text.charAt(0) !== '/') return null;
      return text; // pass through as-is to the CLI
    },

    // Get autocomplete suggestions for slash commands
    getSlashSuggestions: function(partial) {
      if (!partial || partial.charAt(0) !== '/') return [];
      var lower = partial.toLowerCase().substring(1); // strip leading /
      // Use slash commands from the CLI's system/init event if available
      var session = S.currentSessionId ? S.sessions[S.currentSessionId] : null;
      var cc = session && session.claude_code;
      var commands = (cc && cc.slash_commands) || [];
      return commands.filter(function(cmd) {
        return cmd.toLowerCase().indexOf(lower) === 0;
      }).map(function(cmd) { return '/' + cmd; });
    },

    // Check backend session status. Returns 'active', 'suspended', 'expired', or null.
    checkSessionStatus: async function(sessionId) {
      try {
        var resp = await App.authFetch('/api/claude-code/session/' + encodeURIComponent(sessionId), {
          method: 'GET', _timeout: 5000,
        });
        if (!resp.ok) return null;
        var data = await resp.json();
        return data.state || null;
      } catch (_e) { return null; }
    },

    // Update the project picker bar visibility based on current model
    updateProjectBar: function() {
      var session = S.currentSessionId ? S.sessions[S.currentSessionId] : null;
      var model = session ? session.model : S.currentModel;
      var bar = document.getElementById('cc-project-bar');
      if (!bar) return;

      if (App.claudeCode.isClaudeCodeModel(model)) {
        bar.style.display = '';
        var cc = session ? (session.claude_code || null) : null;
        var picker = document.getElementById('cc-dir-picker');
        var info = document.getElementById('cc-session-info');
        var dirLabel = document.getElementById('cc-working-dir-label');
        var dot = info ? info.querySelector('.cc-state-dot') : null;
        var mcpBadge = document.getElementById('cc-mcp-badge');

        if (cc && cc.active) {
          // Active or resumed session
          if (dirLabel) dirLabel.textContent = cc.working_dir || I18n.t('claude_code.quick_chat');
          if (picker) picker.style.display = 'none';
          if (info) info.style.display = '';
          // State dot color
          if (dot) {
            var state = cc.state || 'active';
            dot.style.background = state === 'active' ? '#3ddc84' : state === 'suspended' ? 'var(--orange)' : 'var(--red)';
            dot.title = state;
          }
          // MCP badge
          if (mcpBadge) mcpBadge.style.display = cc.mcp_connected ? '' : 'none';
        } else if (cc && cc.claude_session_id && !cc.active) {
          // Suspended — show resume prompt
          if (dirLabel) dirLabel.textContent = cc.working_dir || I18n.t('claude_code.quick_chat');
          if (picker) picker.style.display = 'none';
          if (info) info.style.display = '';
          if (dot) { dot.style.background = 'var(--orange)'; dot.title = 'suspended'; }
          if (mcpBadge) mcpBadge.style.display = 'none';
        } else {
          // New session — show picker
          if (picker) picker.style.display = '';
          if (info) info.style.display = 'none';
        }
      } else {
        bar.style.display = 'none';
      }
    },

    // Get session list badge info for a CC session
    getSessionBadge: function(session) {
      if (!session.claude_code) return null;
      var cc = session.claude_code;
      var dir = cc.working_dir;
      var short = dir ? dir.split('/').pop() || dir : null;
      return {
        isCC: true,
        dir: short || I18n.t('claude_code.quick_chat'),
        state: cc.state || (cc.active ? 'active' : cc.claude_session_id ? 'suspended' : 'new'),
      };
    },

    // Initialize event bindings
    init: function() {
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

      // Slash command autocomplete
      var chatInput = document.getElementById('chat-input');
      var acContainer = document.getElementById('cc-slash-autocomplete');
      if (chatInput && acContainer) {
        chatInput.addEventListener('input', function() {
          var val = chatInput.value;
          var session = S.currentSessionId ? S.sessions[S.currentSessionId] : null;
          if (!session || !App.claudeCode.isClaudeCodeModel(session.model)) {
            acContainer.style.display = 'none';
            return;
          }
          var word = val.split(/\s/)[0];
          var suggestions = App.claudeCode.getSlashSuggestions(word);
          if (suggestions.length > 0 && val === word) {
            acContainer.innerHTML = suggestions.map(function(cmd) {
              return '<div class="cc-slash-item" data-cmd="' + U.escapeHtml(cmd) + '">' +
                '<span class="cc-slash-cmd">' + U.escapeHtml(cmd) + '</span>' +
                '</div>';
            }).join('');
            acContainer.style.display = '';
          } else {
            acContainer.style.display = 'none';
          }
        });
        acContainer.addEventListener('click', function(e) {
          var item = e.target.closest('.cc-slash-item');
          if (item) {
            var cmd = item.getAttribute('data-cmd');
            chatInput.value = cmd + ' ';
            chatInput.focus();
            acContainer.style.display = 'none';
          }
        });
      }
    },
  };
})();
