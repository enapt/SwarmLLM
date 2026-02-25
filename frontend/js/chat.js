'use strict';

var STORAGE_KEY = 'swarmllm_chat_history';
var messages = [];
var isStreaming = false;

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
}

// --- Model selector ---

async function loadModels() {
  try {
    var resp = await fetch('/v1/models');
    var data = await resp.json();
    var sel = document.getElementById('model-select');
    sel.innerHTML = '';
    if (data.data && data.data.length > 0) {
      data.data.forEach(function (m) {
        var opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.id;
        sel.appendChild(opt);
      });
    } else {
      var opt = document.createElement('option');
      opt.value = 'local';
      opt.textContent = 'local';
      sel.appendChild(opt);
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

  isStreaming = true;
  document.getElementById('send-btn').disabled = true;

  var model = document.getElementById('model-select').value;
  var temperature = parseFloat(document.getElementById('chat-temp').value) || 0.7;
  var maxTokens = parseInt(document.getElementById('chat-max-tokens').value, 10) || 2048;

  var body = {
    model: model,
    messages: messages.map(function (m) {
      return { role: m.role, content: m.content };
    }),
    temperature: temperature,
    max_tokens: maxTokens,
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
              fullContent += delta.content;
              contentEl.textContent = fullContent;
              scrollToBottom();
            }
          }
        } catch (e) { /* skip malformed chunks */ }
      }
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
  div.innerHTML = '<div class="msg-role">' + role + '</div><div class="msg-content"></div>';
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
