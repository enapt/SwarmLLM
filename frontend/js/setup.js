'use strict';

let currentStep = 1;
const totalSteps = 4;
let hwData = null;

const contributionLevels = ['minimal', 'moderate', 'maximum'];
const contributionDescs = [
  'Low impact: uses minimal resources. Best for shared or low-spec machines.',
  'Balanced: uses ~50% of available resources. Good for most users.',
  'Full power: uses all available resources. Best for dedicated nodes.',
];

// --- Step navigation ---

function updateStepUI() {
  for (let i = 1; i <= totalSteps; i++) {
    const body = document.getElementById('step-' + i);
    const indicator = document.querySelector('[data-step="' + i + '"]');
    if (i === currentStep) {
      body.classList.remove('hidden');
      indicator.classList.add('active');
      indicator.classList.remove('done');
    } else if (i < currentStep) {
      body.classList.add('hidden');
      indicator.classList.remove('active');
      indicator.classList.add('done');
    } else {
      body.classList.add('hidden');
      indicator.classList.remove('active', 'done');
    }
  }
  // Connectors
  const connectors = document.querySelectorAll('.wizard-connector');
  connectors.forEach(function (c, idx) {
    if (idx + 1 < currentStep) c.classList.add('done');
    else c.classList.remove('done');
  });

  document.getElementById('btn-prev').classList.toggle('hidden', currentStep === 1);

  var nextBtn = document.getElementById('btn-next');
  if (currentStep === totalSteps) {
    nextBtn.textContent = 'Start SwarmLLM';
  } else {
    nextBtn.textContent = 'Continue';
  }
}

function nextStep() {
  if (currentStep === totalSteps) {
    submitSetup();
    return;
  }
  currentStep++;
  updateStepUI();
  if (currentStep === 4) populateSummary();
}

function prevStep() {
  if (currentStep > 1) {
    currentStep--;
    updateStepUI();
  }
}

// --- Hardware detection ---

async function detectHardware() {
  try {
    var resp = await fetch('/api/admin/stats');
    var data = await resp.json();
    hwData = data.hardware || {};
    document.getElementById('hw-gpu').textContent = hwData.gpu_name || 'No GPU detected (CPU mode)';
    document.getElementById('hw-vram').textContent = hwData.gpu_vram_mb ? hwData.gpu_vram_mb + ' MB' : 'N/A';
    document.getElementById('hw-ram').textContent = formatMB(hwData.total_ram_mb || 0);
    document.getElementById('hw-disk').textContent = formatMB(hwData.available_disk_mb || 0);
  } catch (e) {
    document.getElementById('hw-gpu').textContent = 'Detection failed';
    hwData = {};
  }
  document.getElementById('hw-loading').classList.add('hidden');
  document.getElementById('hw-results').classList.remove('hidden');
}

// --- Contribution slider ---

document.getElementById('contribution-slider').addEventListener('input', function () {
  var val = parseInt(this.value, 10);
  document.getElementById('contribution-label').textContent = capitalize(contributionLevels[val]);
  document.getElementById('contribution-desc').textContent = contributionDescs[val];
});

// --- Model selection (populated from API) ---

async function loadModels() {
  try {
    var resp = await fetch('/v1/models');
    var data = await resp.json();
    var list = document.getElementById('model-list');
    if (!data.data || data.data.length === 0) {
      list.innerHTML = '<p class="text-muted">No models available yet. You can add models later from the dashboard.</p>';
      return;
    }
    list.innerHTML = '';
    data.data.forEach(function (m) {
      var div = document.createElement('div');
      div.className = 'flex gap-1 mb-1';
      div.innerHTML = '<label style="display:flex;align-items:center;gap:8px;cursor:pointer">' +
        '<input type="checkbox" class="model-checkbox" value="' + m.id + '" checked> ' +
        '<span class="mono">' + m.id + '</span></label>';
      list.appendChild(div);
    });
  } catch (e) {
    document.getElementById('model-list').innerHTML = '<p class="text-muted">Could not load models. You can configure them later.</p>';
  }
}

// --- Summary ---

function populateSummary() {
  var slider = document.getElementById('contribution-slider');
  var level = contributionLevels[parseInt(slider.value, 10)];
  document.getElementById('summary-contribution').textContent = capitalize(level);
  document.getElementById('summary-gpu').textContent = hwData && hwData.gpu_name ? hwData.gpu_name : 'CPU only';
  document.getElementById('summary-ram').textContent = formatMB(hwData ? hwData.total_ram_mb || 0 : 0);
  document.getElementById('summary-disk').textContent = formatMB(hwData ? hwData.available_disk_mb || 0 : 0);

  var checked = document.querySelectorAll('.model-checkbox:checked');
  var models = [];
  checked.forEach(function (cb) { models.push(cb.value); });
  document.getElementById('summary-models').textContent = models.length > 0 ? models.join(', ') : 'None selected';
}

// --- Submit setup ---

async function submitSetup() {
  var slider = document.getElementById('contribution-slider');
  var level = contributionLevels[parseInt(slider.value, 10)];

  var checked = document.querySelectorAll('.model-checkbox:checked');
  var models = [];
  checked.forEach(function (cb) { models.push(cb.value); });

  try {
    await fetch('/api/admin/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contribution: level,
        models: models,
      }),
    });
    window.location.href = '/admin';
  } catch (e) {
    alert('Failed to save configuration. Check if SwarmLLM is running.');
  }
}

// --- Helpers ---

function formatMB(mb) {
  if (!mb || mb === 0) return '—';
  if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
  return mb + ' MB';
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// --- Init ---
detectHardware();
loadModels();
updateStepUI();
