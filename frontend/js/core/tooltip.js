'use strict';

// ============================================================================
// SwarmLLM — Unified tooltip
// Replaces native title= tooltips with a styled popover.
// Targets any element with `title` or `data-tooltip`. Leaves existing
// .hw-mode-popover / .modal / form-error etc. alone.
// ============================================================================

(function() {
  var tip = null;
  var arrow = null;
  var hideTimer = null;
  var currentTarget = null;

  function ensure() {
    if (tip) return;
    tip = document.createElement('div');
    tip.className = 'app-tooltip';
    tip.setAttribute('role', 'tooltip');
    tip.hidden = true;
    arrow = document.createElement('div');
    arrow.className = 'app-tooltip-arrow';
    tip.appendChild(arrow);
    var body = document.createElement('div');
    body.className = 'app-tooltip-body';
    tip.appendChild(body);
    document.body.appendChild(tip);
  }

  function getText(el) {
    // data-tooltip overrides title
    var dt = el.getAttribute('data-tooltip');
    if (dt) return dt;
    // Stash title in data-tip so native tooltip is suppressed while ours is up
    var t = el.getAttribute('title');
    if (t) {
      el.setAttribute('data-tip', t);
      el.removeAttribute('title');
      return t;
    }
    return el.getAttribute('data-tip') || '';
  }

  function restoreTitle(el) {
    if (!el) return;
    var saved = el.getAttribute('data-tip');
    if (saved != null && !el.hasAttribute('title')) el.setAttribute('title', saved);
  }

  function show(el) {
    var text = getText(el);
    if (!text) return;
    ensure();
    var body = tip.querySelector('.app-tooltip-body');
    body.textContent = text;
    tip.hidden = false;
    currentTarget = el;
    position(el);
  }

  function hide() {
    if (!tip) return;
    tip.hidden = true;
    if (currentTarget) restoreTitle(currentTarget);
    currentTarget = null;
  }

  function position(el) {
    var r = el.getBoundingClientRect();
    var tr = tip.getBoundingClientRect();
    var vw = window.innerWidth, vh = window.innerHeight;
    var margin = 8;
    // Default: below and centered
    var top = r.bottom + margin;
    var placement = 'bottom';
    if (top + tr.height > vh - 4 && r.top - margin - tr.height > 4) {
      top = r.top - margin - tr.height;
      placement = 'top';
    }
    var left = r.left + (r.width / 2) - (tr.width / 2);
    if (left < 4) left = 4;
    if (left + tr.width > vw - 4) left = vw - tr.width - 4;
    tip.style.top = Math.round(top + window.scrollY) + 'px';
    tip.style.left = Math.round(left + window.scrollX) + 'px';
    tip.setAttribute('data-placement', placement);
    // Arrow x = target center - tooltip left
    var ax = (r.left + r.width / 2) - left;
    arrow.style.left = Math.max(8, Math.min(tr.width - 16, ax)) + 'px';
  }

  // Elements to skip (their own tooltip systems or none wanted)
  function isSkip(el) {
    if (!el || el.nodeType !== 1) return true;
    // Skip inputs/selects — browser autofill + form validation prefer native title
    if (el.tagName === 'INPUT' || el.tagName === 'SELECT' || el.tagName === 'TEXTAREA') return true;
    // Skip elements inside our own popover system
    if (el.closest('.hw-mode-popover, .app-tooltip, .modal')) return true;
    // Honor explicit opt-out
    if (el.closest('[data-tooltip-off]')) return true;
    return false;
  }

  function findTarget(el) {
    while (el && el.nodeType === 1) {
      if (el.hasAttribute('data-tooltip') || el.hasAttribute('title') || el.hasAttribute('data-tip')) return el;
      el = el.parentElement;
    }
    return null;
  }

  document.addEventListener('mouseover', function(e) {
    var t = findTarget(e.target);
    if (!t || isSkip(t)) return;
    if (t === currentTarget) return;
    clearTimeout(hideTimer);
    if (currentTarget) restoreTitle(currentTarget);
    show(t);
  }, true);

  document.addEventListener('mouseout', function(e) {
    var t = findTarget(e.target);
    if (!t) return;
    // Delay hide so we don't flicker when moving between child nodes
    clearTimeout(hideTimer);
    hideTimer = setTimeout(hide, 80);
  }, true);

  document.addEventListener('focusin', function(e) {
    var t = findTarget(e.target);
    if (!t || isSkip(t)) return;
    clearTimeout(hideTimer);
    if (currentTarget && currentTarget !== t) restoreTitle(currentTarget);
    show(t);
  });

  document.addEventListener('focusout', function(e) {
    var t = findTarget(e.target);
    if (!t) return;
    clearTimeout(hideTimer);
    hideTimer = setTimeout(hide, 80);
  });

  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') hide();
  });

  window.addEventListener('scroll', hide, true);
  window.addEventListener('resize', hide);

  // Expose for manual control if needed
  App.tooltip = { hide: hide, show: show };
})();
