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

        // If resume failed (stale session from previous daemon run), retry without resume
        if (!resp.ok && body.resume_claude_session_id) {
          var errText = await resp.text();
          if (errText.indexOf('No Claude Code session') !== -1) {
            delete body.resume_claude_session_id;
            if (session && session.claude_code) session.claude_code.claude_session_id = null;
            resp = await App.authFetch('/api/claude-code/session', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(body),
              _timeout: 130000,
            });
          } else {
            throw new Error(errText);
          }
        }

        if (!resp.ok) {
          var errText2 = await resp.text();
          throw new Error(errText2);
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

      // Live elapsed timer — replaced by result info (turns + duration) when done
      var timerEl = document.createElement('div');
      timerEl.className = 'msg-timer cc-live-timer';
      var timerTarget = assistantEl.querySelector('.msg-bubble') || assistantEl;
      timerTarget.appendChild(timerEl);
      timerInterval = setInterval(function() {
        var elapsed = ((performance.now() - startTime) / 1000).toFixed(0);
        timerEl.textContent = elapsed + 's';
      }, 1000);

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
                sessionId: sessionId,
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
        timerEl.classList.remove('cc-live-timer');
        // If _handleResult didn't set final text, use elapsed time
        if (!timerEl.dataset.final) {
          var elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
          timerEl.textContent = elapsed + 's';
        }
      }

      var elapsedSec = ((performance.now() - startTime) / 1000).toFixed(2);
      return { content: fullContent, pendingPermission: pendingPermission, duration: elapsedSec };
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
          // Extended thinking — render as faded italic preface, not a box
          var thinkText = inner.delta.thinking || '';
          if (!ctx.cleared) { contentEl.textContent = ''; ctx.setClear(); }
          var thinkingEl = contentEl.querySelector('.cc-thinking');
          if (!thinkingEl) {
            thinkingEl = document.createElement('div');
            thinkingEl.className = 'cc-thinking';
            contentEl.appendChild(thinkingEl);
          }
          thinkingEl.textContent += thinkText;
          App.chat.scrollToBottom();
        }
      }
    },

    // Get or create a tool group container for batching consecutive tool calls.
    // A new group is created when the last child of contentEl is not a group,
    // or when forceNew is true (e.g., after a text block).
    _getOrCreateToolGroup: function(contentEl, ctx) {
      var last = contentEl.lastElementChild;
      if (last && last.classList && last.classList.contains('cc-tool-group') && !last.classList.contains('cc-group-closed')) {
        return last;
      }
      var group = document.createElement('details');
      group.className = 'cc-tool-group';
      group.open = true;
      group.innerHTML = '<summary class="cc-group-header">' +
        '<span class="cc-group-label"></span>' +
        '<span class="cc-group-count"></span>' +
        '</summary>';
      contentEl.appendChild(group);
      return group;
    },

    // Update the group summary label with tool count, and surface pending
    // permission prompts' action buttons onto the group header so they're
    // accessible even when the group is collapsed.
    _updateGroupSummary: function(group) {
      var items = group.querySelectorAll('.cc-tool-call, .cc-permission-prompt, .cc-perm-collapsed');
      var headerEl = group.querySelector('.cc-group-header');
      var countEl = group.querySelector('.cc-group-count');
      var labelEl = group.querySelector('.cc-group-label');
      if (!countEl) return;
      var total = items.length;
      var done = group.querySelectorAll('.cc-tool-done, .cc-perm-allowed, .cc-perm-denied, .cc-perm-collapsed').length;
      countEl.textContent = total > 1 ? (done + '/' + total) : '';
      // Build a label from tool names
      var names = [];
      for (var i = 0; i < items.length && names.length < 4; i++) {
        var n = items[i].querySelector('.cc-tool-name, .cc-perm-tool-name');
        if (n && names.indexOf(n.textContent) === -1) names.push(n.textContent);
      }
      if (names.length > 0) {
        var suffix = total > items.length ? ' +' + (total - items.length) : '';
        labelEl.textContent = names.join(', ') + suffix;
      }

      // Surface pending permission actions onto the group header
      var oldActions = headerEl.querySelector('.cc-group-actions');
      if (oldActions) oldActions.remove();

      var pending = group.querySelectorAll('.cc-perm-waiting');
      if (pending.length > 0) {
        // Show the first pending prompt's tool name + Allow/Deny on the header
        var first = pending[0];
        var pToolName = first._ccToolName || first.querySelector('.cc-perm-tool-name');
        var pName = pToolName ? (typeof pToolName === 'string' ? pToolName : pToolName.textContent) : '';
        var pIcon = first._ccIcon || '';

        var actions = document.createElement('span');
        actions.className = 'cc-group-actions';
        actions.innerHTML =
          (pIcon ? '<span class="cc-tool-icon">' + pIcon + '</span>' : '') +
          '<span class="cc-group-action-label">' + U.escapeHtml(pName) + '</span>' +
          (pending.length > 1 ? '<span class="cc-group-action-count">+' + (pending.length - 1) + '</span>' : '') +
          '<button class="btn btn-sm cc-perm-allow cc-group-allow">' + U.escapeHtml(I18n.t('claude_code.allow')) + '</button>' +
          '<button class="btn btn-sm cc-perm-deny cc-group-deny">' + U.escapeHtml(I18n.t('claude_code.deny')) + '</button>';

        // Wire up to the first pending prompt's actual handlers
        var allowBtn = actions.querySelector('.cc-group-allow');
        var denyBtn = actions.querySelector('.cc-group-deny');
        var origAllow = first.querySelector('.cc-perm-allow');
        var origDeny = first.querySelector('.cc-perm-deny');
        allowBtn.addEventListener('click', function(e) {
          e.preventDefault(); e.stopPropagation();
          if (origAllow) origAllow.click();
        });
        denyBtn.addEventListener('click', function(e) {
          e.preventDefault(); e.stopPropagation();
          if (origDeny) origDeny.click();
        });

        headerEl.appendChild(actions);
        // Mark the group as needing attention
        group.classList.add('cc-group-pending');
      } else {
        group.classList.remove('cc-group-pending');
      }
    },

    // Close the current tool group (called when text content appears)
    _closeCurrentGroup: function(contentEl) {
      var last = contentEl.lastElementChild;
      if (last && last.classList && last.classList.contains('cc-tool-group')) {
        last.classList.add('cc-group-closed');
        // Auto-collapse groups where all items are done
        App.claudeCode._maybeCollapseGroup(last);
      }
    },

    // Collapse a group if all its items are resolved
    _maybeCollapseGroup: function(group) {
      // Never collapse if there are pending permission prompts
      if (group.querySelector('.cc-perm-waiting')) return;
      var items = group.querySelectorAll('.cc-tool-call, .cc-permission-prompt, .cc-perm-collapsed');
      var done = group.querySelectorAll('.cc-tool-done, .cc-perm-allowed, .cc-perm-denied, .cc-perm-collapsed').length;
      if (items.length > 0 && done === items.length) {
        group.open = false;
        group.classList.add('cc-group-done');
        App.claudeCode._updateGroupSummary(group);
      }
    },

    // Handle complete assistant message (may contain tool_use)
    _handleAssistantMessage: function(evt, contentEl, toolPanels, agentPanels, taskItems, ctx) {
      var msg = evt.message || {};
      var content = msg.content || [];
      if (!Array.isArray(content)) return;

      content.forEach(function(block) {
        if (block.type === 'thinking' && block.thinking) {
          // Thinking block — render as faded preface text
          if (!ctx.cleared) { contentEl.textContent = ''; ctx.setClear(); }
          var thinkingEl = contentEl.querySelector('.cc-thinking');
          if (!thinkingEl) {
            thinkingEl = document.createElement('div');
            thinkingEl.className = 'cc-thinking';
            contentEl.appendChild(thinkingEl);
          }
          thinkingEl.textContent += block.thinking;
        } else if (block.type === 'text' && block.text) {
          // Close any open tool group before text
          App.claudeCode._closeCurrentGroup(contentEl);
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
            App.claudeCode._closeCurrentGroup(contentEl);
            App.claudeCode._renderAgentCall(contentEl, block, agentPanels);
          } else if (TASK_TOOLS[toolName]) {
            App.claudeCode._renderTaskCall(contentEl, block, toolPanels, taskItems);
          } else {
            // Render inside current tool group
            var group = App.claudeCode._getOrCreateToolGroup(contentEl, ctx);
            App.claudeCode._renderToolCall(group, block, toolPanels);
            App.claudeCode._updateGroupSummary(group);
          }
        }
      });
      App.chat.scrollToBottom();
    },

    // Get a short description for a tool's file/target hint
    _toolHint: function(toolName, input) {
      if (toolName === 'Bash') return input.description || '';
      if (toolName === 'Read' || toolName === 'Write' || toolName === 'Edit') {
        var fp = input.file_path || '';
        return fp.split('/').pop() || fp;
      }
      if (toolName === 'Glob') return input.pattern || '';
      if (toolName === 'Grep') return input.pattern || '';
      if (toolName === 'WebSearch') return input.query || '';
      if (toolName === 'WebFetch') {
        var url = input.url || '';
        try { return new URL(url).hostname; } catch (_e) { return url.substring(0, 40); }
      }
      if (toolName === 'ToolSearch') return input.query || '';
      if (toolName === 'AskUserQuestion') return input.question ? input.question.substring(0, 50) : '';
      return '';
    },

    // Render a tool call block as a collapsible <details> element
    _renderToolCall: function(contentEl, block, toolPanels) {
      var toolId = block.id || '';
      var toolName = block.name || 'Unknown';
      var input = block.input || {};

      var panel = document.createElement('details');
      panel.className = 'cc-tool-call';
      panel.setAttribute('data-tool-id', toolId);
      panel.open = false; // collapsed by default, shows summary

      var hint = App.claudeCode._toolHint(toolName, input);
      if (hint.length > 60) hint = hint.substring(0, 57) + '...';

      var icon = App.claudeCode._toolIcon(toolName);
      var summary = document.createElement('summary');
      summary.className = 'cc-tool-header';
      summary.innerHTML =
        '<span class="cc-tool-icon">' + icon + '</span>' +
        '<span class="cc-tool-name">' + U.escapeHtml(toolName) + '</span>' +
        (hint ? '<span class="cc-tool-file">' + U.escapeHtml(hint) + '</span>' : '') +
        '<span class="cc-tool-status pending">' + U.escapeHtml(I18n.t('claude_code.running')) + '</span>';
      panel.appendChild(summary);

      // Build expandable detail content
      var detail = document.createElement('div');
      detail.className = 'cc-tool-body';
      var detailHtml = App.claudeCode._buildToolDetail(toolName, input);
      if (detailHtml) {
        detail.innerHTML = detailHtml;
      }
      panel.appendChild(detail);

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

    // Render tool output with smart formatting (diffs, file lists, errors)
    _renderToolOutput: function(toolName, blockText) {
      var el = document.createElement('div');
      el.className = 'cc-tool-result';
      if (!blockText || blockText.length === 0) return el;

      // Detect diff output
      if (blockText.indexOf('@@') !== -1 && (blockText.indexOf('+') !== -1 || blockText.indexOf('-') !== -1) &&
          (blockText.indexOf('---') !== -1 || blockText.indexOf('+++') !== -1)) {
        el.innerHTML = App.claudeCode._renderDiffOutput(blockText);
        return el;
      }

      // Detect error output
      var isError = /^(error|Error|ERROR|FAIL|panic)/.test(blockText);

      // Truncate with expandable toggle
      var truncated = blockText.length > 800;
      var displayText = truncated ? blockText.substring(0, 800) : blockText;

      var pre = document.createElement('pre');
      pre.className = 'cc-tool-output' + (isError ? ' cc-tool-output-error' : '');
      if (toolName === 'Bash') pre.classList.add('cc-bash-output-dark');
      pre.textContent = displayText;
      el.appendChild(pre);

      if (truncated) {
        var toggle = document.createElement('button');
        toggle.className = 'cc-output-toggle';
        toggle.textContent = I18n.t('claude_code.show_more');
        toggle.addEventListener('click', function() {
          if (pre.textContent === displayText) {
            pre.textContent = blockText;
            toggle.textContent = I18n.t('claude_code.show_less');
          } else {
            pre.textContent = displayText;
            toggle.textContent = I18n.t('claude_code.show_more');
          }
        });
        el.appendChild(toggle);
      }

      return el;
    },

    // Render diff-formatted output with syntax highlighting
    _renderDiffOutput: function(text) {
      var lines = text.split('\n');
      var html = '<pre class="cc-diff-output">';
      for (var i = 0; i < lines.length && i < 200; i++) {
        var line = lines[i];
        var cls = '';
        if (line.charAt(0) === '+' && line.charAt(1) !== '+') cls = 'diff-add';
        else if (line.charAt(0) === '-' && line.charAt(1) !== '-') cls = 'diff-del';
        else if (line.indexOf('@@') === 0) cls = 'diff-hunk';
        if (cls) html += '<span class="' + cls + '">' + U.escapeHtml(line) + '</span>\n';
        else html += U.escapeHtml(line) + '\n';
      }
      if (lines.length > 200) html += '<span class="diff-hunk">... (' + (lines.length - 200) + ' more lines)</span>\n';
      html += '</pre>';
      return html;
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
          // Collapse the details panel now that it's done
          if (panel.tagName === 'DETAILS') {
            panel.classList.add('cc-tool-done');
          }
        }

        // Determine tool name from panel data
        var toolNameEl = panel && panel.querySelector('.cc-tool-name');
        var toolName = toolNameEl ? toolNameEl.textContent : '';

        // Render result with smart formatting
        var resultEl = App.claudeCode._renderToolOutput(toolName, blockText);

        if (panel) {
          var body = panel.querySelector('.cc-tool-body');
          if (body) {
            body.appendChild(resultEl);
          } else {
            panel.appendChild(resultEl);
          }
          // Check if parent group can collapse
          var group = panel.closest('.cc-tool-group');
          if (group) {
            App.claudeCode._updateGroupSummary(group);
            App.claudeCode._maybeCollapseGroup(group);
          }
        } else {
          contentEl.appendChild(resultEl);
        }
      });

      App.chat.scrollToBottom();
    },

    // Build rich detail view for a tool's input
    _buildToolDetail: function(toolName, input) {
      var html = '';
      if (toolName === 'Bash') {
        var cmd = input.command || '';
        html += '<pre class="cc-perm-code cc-bash-cmd">$ ' + U.escapeHtml(cmd) + '</pre>';
        if (input.description) {
          html += '<div class="cc-perm-desc">' + U.escapeHtml(input.description) + '</div>';
        }
      } else if (toolName === 'Edit') {
        html += '<div class="cc-perm-file">' + U.escapeHtml(input.file_path || '') + '</div>';
        if (input.old_string || input.new_string) {
          var old = (input.old_string || '').substring(0, 300);
          var nw = (input.new_string || '').substring(0, 300);
          html += '<div class="cc-perm-diff">';
          if (old) html += '<div class="cc-perm-diff-del">' + U.escapeHtml(old) + '</div>';
          if (nw) html += '<div class="cc-perm-diff-add">' + U.escapeHtml(nw) + '</div>';
          html += '</div>';
        }
      } else if (toolName === 'Write') {
        html += '<div class="cc-perm-file">' + U.escapeHtml(input.file_path || '') + '</div>';
        if (input.content) {
          var preview = input.content.substring(0, 300);
          if (input.content.length > 300) preview += '\n...';
          html += '<pre class="cc-perm-code">' + U.escapeHtml(preview) + '</pre>';
        }
      } else if (toolName === 'Read') {
        html += '<div class="cc-perm-file">' + U.escapeHtml(input.file_path || '') + '</div>';
      } else if (toolName === 'Glob' || toolName === 'Grep') {
        if (input.pattern) html += '<div class="cc-perm-file">' + U.escapeHtml(input.pattern) + '</div>';
        if (input.path) html += '<div class="cc-perm-desc">' + I18n.t('claude_code.perm_in') + ' ' + U.escapeHtml(input.path) + '</div>';
      } else if (toolName === 'WebFetch') {
        html += '<div class="cc-perm-file">' + U.escapeHtml(input.url || '') + '</div>';
      } else if (toolName === 'WebSearch') {
        html += '<div class="cc-perm-file">' + U.escapeHtml(input.query || '') + '</div>';
      } else if (toolName === 'AskUserQuestion') {
        // AskUserQuestion — render the question and options
        if (input.question) {
          html += '<div class="cc-perm-desc">' + U.escapeHtml(input.question) + '</div>';
        }
      } else {
        // Generic: show JSON preview (skip if empty)
        var keys = Object.keys(input);
        if (keys.length > 0) {
          var jsonStr = JSON.stringify(input, null, 2);
          if (jsonStr.length > 300) jsonStr = jsonStr.substring(0, 300) + '\n...';
          html += '<pre class="cc-perm-code">' + U.escapeHtml(jsonStr) + '</pre>';
        }
      }
      return html;
    },

    // Handle permission request — compact inline bar with quick actions,
    // expandable detail if the user wants to inspect.
    _handlePermissionRequest: function(evt, contentEl, ctx) {
      var req = evt.request || {};
      var toolName = req.tool_name || 'Unknown';
      var input = req.input || {};
      var requestId = evt.request_id || '';
      var sessionId = ctx.sessionId || S.currentSessionId || '';
      var reason = req.decision_reason || '';

      // Render inside the current tool group
      var group = App.claudeCode._getOrCreateToolGroup(contentEl, ctx);

      var panel = document.createElement('details');
      panel.className = 'cc-permission-prompt cc-perm-waiting';

      var icon = App.claudeCode._toolIcon(toolName);
      var hint = App.claudeCode._toolHint(toolName, input);
      var detailHtml = App.claudeCode._buildToolDetail(toolName, input);
      var hasDetail = detailHtml && Object.keys(input).length > 0;

      // Summary line: icon + tool + hint + Allow/Deny buttons — always visible
      var summary = document.createElement('summary');
      summary.className = 'cc-perm-bar';
      summary.innerHTML =
        '<span class="cc-perm-pulse"></span>' +
        '<span class="cc-tool-icon">' + icon + '</span>' +
        '<strong class="cc-perm-tool-name">' + U.escapeHtml(toolName) + '</strong>' +
        (hint ? '<span class="cc-tool-file">' + U.escapeHtml(hint) + '</span>' : '') +
        '<span class="cc-perm-actions">' +
          '<button class="btn btn-sm cc-perm-allow">' + U.escapeHtml(I18n.t('claude_code.allow')) + '</button>' +
          '<button class="btn btn-sm cc-perm-deny">' + U.escapeHtml(I18n.t('claude_code.deny')) + '</button>' +
        '</span>';
      panel.appendChild(summary);

      // Expandable detail body
      if (hasDetail || reason) {
        var body = document.createElement('div');
        body.className = 'cc-perm-body';
        if (hasDetail) body.innerHTML += '<div class="cc-perm-detail">' + detailHtml + '</div>';
        if (reason) body.innerHTML += '<div class="cc-perm-reason">' + U.escapeHtml(reason) + '</div>';
        panel.appendChild(body);
      }

      // Store tool metadata for the collapsed view
      panel._ccToolName = toolName;
      panel._ccIcon = icon;
      panel._ccHint = hint;

      group.appendChild(panel);
      App.claudeCode._updateGroupSummary(group);
      ctx.setPendingPermission({ requestId: requestId, element: panel });

      // Quick action buttons — stop click from toggling <details>
      var allowBtn = summary.querySelector('.cc-perm-allow');
      var denyBtn = summary.querySelector('.cc-perm-deny');

      allowBtn.addEventListener('click', function(e) {
        e.preventDefault(); e.stopPropagation();
        App.claudeCode._respondPermission(sessionId, requestId, true, input, panel);
      });
      denyBtn.addEventListener('click', function(e) {
        e.preventDefault(); e.stopPropagation();
        App.claudeCode._respondPermission(sessionId, requestId, false, input, panel);
      });

      // Keyboard: Enter = allow, Escape = deny
      summary.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !panel.classList.contains('cc-perm-allowed') && !panel.classList.contains('cc-perm-denied')) {
          e.preventDefault(); e.stopPropagation();
          App.claudeCode._respondPermission(sessionId, requestId, true, input, panel);
        } else if (e.key === 'Escape') {
          e.preventDefault();
          App.claudeCode._respondPermission(sessionId, requestId, false, input, panel);
        }
      });

      setTimeout(function() { allowBtn.focus(); }, 50);
      App.claudeCode._notifyPermissionNeeded();
      App.chat.scrollToBottom();
    },

    // Notify user that a permission decision is needed
    _notifyPermissionNeeded: function() {
      // Flash the document title to draw attention
      if (document.hidden) {
        var original = document.title;
        var flash = function() {
          document.title = document.title === original ? '⚠ Permission Required' : original;
        };
        var interval = setInterval(flash, 800);
        var restore = function() {
          clearInterval(interval);
          document.title = original;
          document.removeEventListener('visibilitychange', restore);
        };
        document.addEventListener('visibilitychange', restore);
        // Auto-stop after 30s
        setTimeout(restore, 30000);
      }
    },

    // Send permission response — then collapse to one-liner
    _respondPermission: async function(sessionId, requestId, allow, input, panelEl) {
      // Prevent double-click
      if (panelEl.classList.contains('cc-perm-allowed') || panelEl.classList.contains('cc-perm-denied')) return;

      var actionsEl = panelEl.querySelector('.cc-perm-actions');
      actionsEl.innerHTML = '<span class="cc-perm-resolving">' + U.escapeHtml(I18n.t('claude_code.sending')) + '...</span>';

      try {
        var body = { request_id: requestId, allow: allow };
        if (allow && input) body.updated_input = input;
        if (!allow) body.message = 'User denied this action';

        await App.authFetch('/api/claude-code/session/' + encodeURIComponent(sessionId) + '/permission', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });

        // Collapse to a slim one-liner
        var toolName = panelEl._ccToolName || 'Tool';
        var icon = panelEl._ccIcon || '⚡';
        var hint = panelEl._ccHint || '';
        var statusClass = allow ? 'cc-perm-allowed' : 'cc-perm-denied';
        var statusIcon = allow ? '✓' : '✗';
        var statusText = allow ? I18n.t('claude_code.allowed') : I18n.t('claude_code.denied');

        panelEl.classList.remove('cc-perm-waiting');
        panelEl.classList.add(statusClass, 'cc-perm-collapsed');
        // Replace with a non-details div so it renders as a simple one-liner
        var oneliner = document.createElement('div');
        oneliner.className = 'cc-permission-prompt cc-perm-collapsed ' + statusClass;
        oneliner.innerHTML =
          '<div class="cc-perm-oneliner">' +
            '<span class="cc-perm-status-badge ' + (allow ? 'cc-badge-allow' : 'cc-badge-deny') + '">' + statusIcon + ' ' + statusText + '</span>' +
            '<span class="cc-tool-icon">' + icon + '</span>' +
            '<span class="cc-perm-tool-name">' + U.escapeHtml(toolName) + '</span>' +
            (hint ? '<span class="cc-tool-file">' + U.escapeHtml(hint) + '</span>' : '') +
          '</div>';
        panelEl.replaceWith(oneliner);

        // Check if the parent group can now collapse
        var group = oneliner.closest('.cc-tool-group');
        if (group) {
          App.claudeCode._updateGroupSummary(group);
          App.claudeCode._maybeCollapseGroup(group);
        }
      } catch (e) {
        actionsEl.innerHTML =
          '<span class="cc-perm-resolved cc-perm-error">' + U.escapeHtml(I18n.t('common.request_failed')) + '</span>';
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

      // Update the live timer with final info (turns + duration)
      var timerEl = contentEl.closest('.msg-row') && contentEl.closest('.msg-row').querySelector('.msg-timer');
      if (timerEl) {
        var parts = [];
        var turns = evt.num_turns || 1;
        if (turns > 1) parts.push(I18n.t('claude_code.turns_count', { count: turns }));
        var duration = evt.duration_ms ? (evt.duration_ms / 1000).toFixed(1) + 's' : '';
        if (duration) parts.push(duration);
        timerEl.textContent = parts.join(' \u00b7 ') || '';
        timerEl.dataset.final = '1';
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
        AskUserQuestion: '❓',
        EnterPlanMode: '📐',
        ExitPlanMode: '📐',
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
