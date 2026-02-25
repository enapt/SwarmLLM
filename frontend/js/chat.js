'use strict';

var STORAGE_KEY = 'swarmllm_chat_history';
var messages = [];
var isStreaming = false;
var currentModel = 'local';

// --- Load conversation from localStorage ---

function loadHistory() {
  try {
    var saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      messages = JSON.parse(saved);
      messages.forEach(function (msg) {
        appendMessageToDOM(msg.role, msg.content);
      });
    }
  } catch (e) {
    messages = [];
  }
}

function saveHistory() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  } catch (e) { /* quota exceeded, ignore */ }
}

function clearChat() {
  messages = [];
  saveHistory();
  document.getElementById('chat-messages').innerHTML = '';
  // Restore empty state
  var container = document.getElementById('chat-messages');
  container.innerHTML = '<div class="chat-empty" id="chat-empty">' +
    '<div class="chat-empty-icon">&#11088;</div>' +
    '<div style="font-size:1.2rem;font-weight:600;color:var(--text-primary)">SwarmLLM Chat</div>' +
    '<div>Send a message to start a conversation</div></div>';
}

// --- Model selector (hidden, auto-detect) ---

async function loadModels() {
  try {
    var resp = await fetch('/v1/models');
    var data = await resp.json();
    var sel = document.getElementById('model-select');
    sel.innerHTML = '';
    if (data.data && data.data.length > 0) {
      currentModel = data.data[0].id;
      data.data.forEach(function (m) {
        var opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.id;
        sel.appendChild(opt);
      });
    }
  } catch (e) {
    console.error('Failed to load models:', e);
  }
}

// --- Send message ---

function handleInputKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

async function sendMessage() {
  if (isStreaming) return;

  var input = document.getElementById('chat-input');
  var text = input.value.trim();
  if (!text) return;

  input.value = '';
  autoResizeInput();

  // Add user message
  messages.push({ role: 'user', content: text });
  saveHistory();
  appendMessageToDOM('user', text);

  // Prepare assistant message element for streaming
  var assistantEl = appendMessageToDOM('assistant', '');
  var contentEl = assistantEl.querySelector('.msg-content');
  contentEl.innerHTML = '<span class="typing-indicator">Thinking...</span>';

  isStreaming = true;
  document.getElementById('send-btn').disabled = true;

  var body = {
    model: currentModel,
    messages: messages.map(function (m) {
      return { role: m.role, content: m.content };
    }),
    temperature: 0.7,
    max_tokens: 512,
    stream: true,
  };

  var fullContent = '';

  try {
    var resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      var errText = await resp.text();
      contentEl.textContent = 'Error: ' + errText;
      isStreaming = false;
      document.getElementById('send-btn').disabled = false;
      return;
    }

    // Clear typing indicator on first data
    var cleared = false;

    var reader = resp.body.getReader();
    var decoder = new TextDecoder();
    var buffer = '';

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
          var chunk = JSON.parse(payload);
          if (chunk.choices && chunk.choices[0] && chunk.choices[0].delta) {
            var delta = chunk.choices[0].delta;
            if (delta.content) {
              if (!cleared) {
                contentEl.textContent = '';
                cleared = true;
              }
              fullContent += delta.content;
              contentEl.textContent = fullContent;
              scrollToBottom();
            }
          }
        } catch (e) { /* skip malformed chunks */ }
      }
    }

    // If stream ended but nothing streamed, it was a non-streaming response wrapped in SSE
    if (!cleared && !fullContent) {
      contentEl.textContent = 'No response received. The model may still be loading.';
    }
  } catch (e) {
    if (!fullContent) {
      contentEl.textContent = 'Error: Connection failed.';
    }
  }

  if (fullContent) {
    messages.push({ role: 'assistant', content: fullContent });
    saveHistory();
  }

  isStreaming = false;
  document.getElementById('send-btn').disabled = false;
}

// --- DOM helpers ---

function appendMessageToDOM(role, content) {
  var container = document.getElementById('chat-messages');
  // Hide empty state on first message
  var empty = document.getElementById('chat-empty');
  if (empty) empty.style.display = 'none';

  var div = document.createElement('div');
  div.className = 'chat-msg ' + role;
  var label = role === 'user' ? 'You' : 'Assistant';
  div.innerHTML = '<div class="msg-role">' + label + '</div><div class="msg-content"></div>';
  div.querySelector('.msg-content').textContent = content;
  container.appendChild(div);
  scrollToBottom();
  return div;
}

function scrollToBottom() {
  var container = document.getElementById('chat-messages');
  container.scrollTop = container.scrollHeight;
}

// --- Auto-resize textarea ---

var inputEl = document.getElementById('chat-input');
inputEl.addEventListener('input', autoResizeInput);

function autoResizeInput() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
}

// --- Node ID display ---

async function loadNodeInfo() {
  try {
    var resp = await fetch('/api/admin/stats');
    var data = await resp.json();
    if (data.node_id) {
      document.getElementById('node-id').textContent = data.node_id;
    }
  } catch (e) { /* ignore */ }
}

// --- Init ---
loadHistory();
loadModels();
loadNodeInfo();
