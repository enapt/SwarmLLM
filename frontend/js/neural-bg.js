'use strict';

// ============================================================================
// Neural Network Background — animated particle network behind dashboard
// Responds to mouse, colorizes based on system state
// ============================================================================

var NeuralBg = (function() {
  var canvas, ctx;
  var W, H;
  var nodes = [];
  var mouse = { x: -1000, y: -1000 };
  var raf = null;
  var paused = false;

  // State-driven colors (updated from WebSocket stats)
  var state = {
    peers: 0,
    active: 0,
    health: 1.0  // 0=bad, 1=good
  };

  // Tuning — keep it lightweight
  var NODE_COUNT = 60;
  var CONNECT_DIST = 160;
  var MOUSE_RADIUS = 200;
  var BASE_SPEED = 0.3;
  var NODE_RADIUS_MIN = 1.5;
  var NODE_RADIUS_MAX = 3;
  var EDGE_OPACITY = 0.12;
  var NODE_OPACITY = 0.5;
  var PULSE_SPEED = 0.002;
  var DPR = 1; // device pixel ratio (capped)

  function init() {
    canvas = document.getElementById('neural-bg');
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    DPR = Math.min(window.devicePixelRatio || 1, 2);

    resize();
    seed();

    // Events
    window.addEventListener('resize', debounceResize);
    document.addEventListener('mousemove', onMouse);
    document.addEventListener('mouseleave', function() {
      mouse.x = -1000; mouse.y = -1000;
    });

    // Visibility — pause when hidden
    document.addEventListener('visibilitychange', function() {
      if (document.hidden) {
        paused = true;
      } else {
        paused = false;
        if (!raf) tick();
      }
    });

    tick();
  }

  var _resizeTimer;
  function debounceResize() {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(function() {
      resize();
      seed();
    }, 200);
  }

  function resize() {
    var main = document.getElementById('view-dashboard');
    if (!main) main = document.body;
    W = main.offsetWidth;
    H = Math.max(main.scrollHeight, window.innerHeight);
    canvas.width = W * DPR;
    canvas.height = H * DPR;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }

  function seed() {
    nodes = [];
    // Scale count to viewport area (fewer on mobile)
    var area = W * H;
    var count = Math.min(NODE_COUNT, Math.round(area / 15000));
    count = Math.max(20, count);
    for (var i = 0; i < count; i++) {
      nodes.push({
        x: Math.random() * W,
        y: Math.random() * H,
        vx: (Math.random() - 0.5) * BASE_SPEED,
        vy: (Math.random() - 0.5) * BASE_SPEED,
        r: NODE_RADIUS_MIN + Math.random() * (NODE_RADIUS_MAX - NODE_RADIUS_MIN),
        phase: Math.random() * Math.PI * 2
      });
    }
  }

  function onMouse(e) {
    var rect = canvas.getBoundingClientRect();
    mouse.x = e.clientX - rect.left;
    mouse.y = e.clientY - rect.top;
  }

  // Color interpolation based on system state
  function getColor(alpha) {
    // Idle: blue (#3b82f6) → Active: cyan (#22d3ee) → Stressed: orange (#f59e0b)
    var r, g, b;
    var activity = Math.min(state.active / 5, 1); // 0-1 based on active requests
    var peerBoost = Math.min(state.peers / 10, 1); // more peers = more vibrant

    if (state.health < 0.5) {
      // Unhealthy → red-orange
      r = 239; g = 68 + Math.round(90 * state.health); b = 68;
    } else if (activity > 0.3) {
      // Active → blend from accent-blue toward cyan
      var t = activity;
      r = Math.round(59 * (1 - t) + 34 * t);
      g = Math.round(130 * (1 - t) + 211 * t);
      b = Math.round(246 * (1 - t) + 238 * t);
    } else {
      // Idle → accent blue
      r = 59; g = 130; b = 246;
    }

    // Peer vibrancy boost
    var vibrancy = 0.4 + 0.6 * peerBoost;
    alpha *= vibrancy;

    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha.toFixed(3) + ')';
  }

  function getGlowColor(alpha) {
    if (state.health < 0.5) return 'rgba(239,68,68,' + alpha.toFixed(3) + ')';
    if (state.active > 2) return 'rgba(34,211,238,' + alpha.toFixed(3) + ')';
    return 'rgba(59,130,246,' + alpha.toFixed(3) + ')';
  }

  var _now = 0;

  function tick() {
    if (paused) { raf = null; return; }
    raf = requestAnimationFrame(tick);
    _now += 16; // approximate dt

    ctx.clearRect(0, 0, W, H);

    var n = nodes.length;
    var connectDist2 = CONNECT_DIST * CONNECT_DIST;
    var mouseR2 = MOUSE_RADIUS * MOUSE_RADIUS;

    // Update positions
    for (var i = 0; i < n; i++) {
      var nd = nodes[i];

      // Mouse repulsion (gentle)
      var dx = nd.x - mouse.x;
      var dy = nd.y - mouse.y;
      var d2 = dx * dx + dy * dy;
      if (d2 < mouseR2 && d2 > 1) {
        var force = (1 - d2 / mouseR2) * 0.5;
        var dist = Math.sqrt(d2);
        nd.vx += (dx / dist) * force;
        nd.vy += (dy / dist) * force;
      }

      // Damping
      nd.vx *= 0.98;
      nd.vy *= 0.98;

      // Minimum drift
      var speed2 = nd.vx * nd.vx + nd.vy * nd.vy;
      if (speed2 < 0.01) {
        nd.vx += (Math.random() - 0.5) * 0.1;
        nd.vy += (Math.random() - 0.5) * 0.1;
      }

      nd.x += nd.vx;
      nd.y += nd.vy;

      // Wrap around
      if (nd.x < -20) nd.x = W + 20;
      if (nd.x > W + 20) nd.x = -20;
      if (nd.y < -20) nd.y = H + 20;
      if (nd.y > H + 20) nd.y = -20;
    }

    // Draw edges (spatial skip — only check i < j)
    ctx.lineWidth = 1;
    for (var i = 0; i < n; i++) {
      var a = nodes[i];
      for (var j = i + 1; j < n; j++) {
        var b = nodes[j];
        var ex = a.x - b.x;
        var ey = a.y - b.y;
        var ed2 = ex * ex + ey * ey;
        if (ed2 < connectDist2) {
          var alpha = EDGE_OPACITY * (1 - ed2 / connectDist2);
          ctx.strokeStyle = getColor(alpha);
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    var pulse = Math.sin(_now * PULSE_SPEED) * 0.3 + 0.7; // 0.4 — 1.0
    for (var i = 0; i < n; i++) {
      var nd = nodes[i];
      var nodePulse = Math.sin(_now * PULSE_SPEED + nd.phase) * 0.3 + 0.7;
      var r = nd.r * nodePulse;

      // Mouse proximity glow
      var mdx = nd.x - mouse.x;
      var mdy = nd.y - mouse.y;
      var md2 = mdx * mdx + mdy * mdy;
      var glow = 0;
      if (md2 < mouseR2) {
        glow = (1 - md2 / mouseR2);
      }

      // Activity pulse — certain nodes "fire" when active requests exist
      var fireAlpha = 0;
      if (state.active > 0 && ((i + Math.floor(_now * 0.003)) % Math.max(3, 8 - state.active)) === 0) {
        fireAlpha = 0.4 * pulse;
      }

      var baseAlpha = NODE_OPACITY + glow * 0.4 + fireAlpha;

      ctx.fillStyle = getColor(Math.min(baseAlpha, 1));
      ctx.beginPath();
      ctx.arc(nd.x, nd.y, r + glow * 2, 0, Math.PI * 2);
      ctx.fill();

      // Glow halo for mouse-proximate nodes
      if (glow > 0.3) {
        ctx.fillStyle = getGlowColor(glow * 0.15);
        ctx.beginPath();
        ctx.arc(nd.x, nd.y, r + glow * 6, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // Called by dashboard.updateStats to reflect live state
  function updateState(data) {
    if (data.peers !== undefined) state.peers = data.peers;
    if (data.active_requests !== undefined) state.active = data.active_requests;
    // Infer health from presence of data
    if (data.peers !== undefined) {
      state.health = data.peers > 0 ? 1.0 : 0.7;
    }
  }

  // Manual override for errors
  function setHealth(h) {
    state.health = Math.max(0, Math.min(1, h));
  }

  return {
    init: init,
    updateState: updateState,
    setHealth: setHealth,
    resize: function() { resize(); }
  };
})();
