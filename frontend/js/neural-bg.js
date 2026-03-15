'use strict';

// ============================================================================
// Swarm Background — murmuration-style flocking with neural links
// Boids form linked clusters, scatter like startled swallows, reform naturally
// ============================================================================

var NeuralBg = (function() {
  var canvas, ctx;
  var W, H;
  var boids = [];
  var mouse = { x: -1000, y: -1000, vx: 0, vy: 0, speed: 0, active: false };
  var raf = null;
  var paused = false;

  var state = { peers: 0, active: 0, health: 1.0 };

  // --- Tuning ---
  var BOID_COUNT = 90;
  var MAX_SPEED = 0.7;
  var MAX_FORCE = 0.018;

  // Flocking radii — wider spread than default
  var SEPARATION_DIST = 30;
  var NEIGHBOR_DIST = 120;

  // Link distance — links form/break at this threshold
  var LINK_DIST = 160;
  var LINK_STRONG = 55;
  var LINK_OPACITY = 0.18;

  // Force weights
  var SEPARATION_W = 2.5;     // nodes stay apart
  var ALIGNMENT_W = 0.9;      // clusters move together as a group
  var COHESION_W = 0.3;       // loose cohesion = don't blob up
  var WANDER_W = 0.4;         // gentle drift when calm
  var LINK_PULL_W = 0.08;     // very weak link pull

  // Feelers — long-range probing tendrils
  var FEELER_DIST = 380;      // max reach
  var FEELER_CHANCE = 0.008;  // probability per boid per frame of sending a feeler
  var FEELER_DURATION = 100;  // frames a feeler lives
  var FEELER_OPACITY = 0.12;
  var MAX_FEELERS = 12;       // max simultaneous feelers on screen

  // Mouse
  var MOUSE_ATTRACT_RADIUS = 250;
  var MOUSE_SCATTER_RADIUS = 180;
  var MOUSE_SPEED_THRESH = 6;
  var SCATTER_BURST = 5.0;
  var SCATTER_DECAY = 0.94;

  // Spontaneous startles — random boids spook their neighbors
  var STARTLE_CHANCE = 0.0006;  // per boid per frame (~1 startle every ~18s for 90 boids)
  var STARTLE_RADIUS = 140;     // how far the panic spreads
  var STARTLE_BURST = 3.5;      // impulse strength — noticeable burst in slow swarm
  var STARTLE_CONTAGION = 0.6;  // scattered boids can spook calm neighbors

  // Visual
  var TRAIL_ALPHA = 0.07;
  var PULSE_SPEED = 0.002;
  var DPR = 1;

  // Active feelers list
  var feelers = [];

  // Spatial grid
  var grid = {};
  var CELL_SIZE = 130;

  function init() {
    canvas = document.getElementById('neural-bg');
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    DPR = Math.min(window.devicePixelRatio || 1, 2);

    resize();
    seed();

    window.addEventListener('resize', debounceResize);
    document.addEventListener('mousemove', onMouse);
    document.addEventListener('mouseleave', function() {
      mouse.x = -1000; mouse.y = -1000; mouse.active = false;
      mouse.speed = 0;
    });
    document.addEventListener('visibilitychange', function() {
      if (document.hidden) { paused = true; }
      else { paused = false; if (!raf) tick(); }
    });

    tick();
  }

  var _resizeTimer;
  function debounceResize() {
    clearTimeout(_resizeTimer);
    _resizeTimer = setTimeout(function() {
      resize();
      var area = W * H;
      var target = Math.min(BOID_COUNT, Math.max(30, Math.round(area / 12000)));
      if (Math.abs(boids.length - target) > 10) seed();
    }, 200);
  }

  function resize() {
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width = W * DPR;
    canvas.height = H * DPR;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  }

  function seed() {
    boids = [];
    var area = W * H;
    var count = Math.min(BOID_COUNT, Math.max(30, Math.round(area / 12000)));

    // Spawn in loose clusters spread wide across the canvas
    var clusterCount = Math.floor(count / 6) + 1;
    var perCluster = Math.ceil(count / clusterCount);

    for (var c = 0; c < clusterCount; c++) {
      var cx = Math.random() * W * 0.9 + W * 0.05;
      var cy = Math.random() * H * 0.9 + H * 0.05;
      var clusterAngle = Math.random() * Math.PI * 2;
      var num = Math.min(perCluster, count - boids.length);

      for (var i = 0; i < num; i++) {
        var a = clusterAngle + (Math.random() - 0.5) * 1.5;
        var spd = 0.3 + Math.random() * 0.6;
        boids.push({
          x: cx + (Math.random() - 0.5) * 150,
          y: cy + (Math.random() - 0.5) * 150,
          vx: Math.cos(a) * spd,
          vy: Math.sin(a) * spd,
          phase: Math.random() * Math.PI * 2,
          wanderAngle: a + (Math.random() - 0.5) * 0.5,
          size: 1.5 + Math.random() * 2,
          energy: 0,
          speedMul: 0.3 + Math.random() * 1.4,
          scattered: 0
        });
      }
    }
  }

  function onMouse(e) {
    var rect = canvas.getBoundingClientRect();
    var nx = e.clientX - rect.left;
    var ny = e.clientY - rect.top;
    if (mouse.active) {
      mouse.vx = nx - mouse.x;
      mouse.vy = ny - mouse.y;
      mouse.speed = Math.sqrt(mouse.vx * mouse.vx + mouse.vy * mouse.vy);
    }
    mouse.x = nx;
    mouse.y = ny;
    mouse.active = true;
  }

  function buildGrid() {
    grid = {};
    for (var i = 0; i < boids.length; i++) {
      var b = boids[i];
      var cx = Math.floor(b.x / CELL_SIZE);
      var cy = Math.floor(b.y / CELL_SIZE);
      var key = cx + ',' + cy;
      if (!grid[key]) grid[key] = [];
      grid[key].push(i);
    }
  }

  function getNeighborIndices(boid) {
    var cx = Math.floor(boid.x / CELL_SIZE);
    var cy = Math.floor(boid.y / CELL_SIZE);
    var result = [];
    for (var dx = -1; dx <= 1; dx++) {
      for (var dy = -1; dy <= 1; dy++) {
        var key = (cx + dx) + ',' + (cy + dy);
        if (grid[key]) {
          for (var k = 0; k < grid[key].length; k++) {
            result.push(grid[key][k]);
          }
        }
      }
    }
    return result;
  }

  function getColor(alpha) {
    var r, g, b;
    var activity = Math.min(state.active / 5, 1);
    var peerBoost = Math.min(state.peers / 10, 1);
    if (state.health < 0.5) {
      r = 239; g = 68 + Math.round(90 * state.health); b = 68;
    } else if (activity > 0.3) {
      var t = activity;
      r = Math.round(59 * (1 - t) + 34 * t);
      g = Math.round(130 * (1 - t) + 211 * t);
      b = Math.round(246 * (1 - t) + 238 * t);
    } else {
      r = 59; g = 130; b = 246;
    }
    alpha *= (0.4 + 0.6 * peerBoost);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + Math.min(alpha, 1).toFixed(3) + ')';
  }

  function getGlowColor(alpha) {
    if (state.health < 0.5) return 'rgba(239,68,68,' + alpha.toFixed(3) + ')';
    if (state.active > 2) return 'rgba(34,211,238,' + alpha.toFixed(3) + ')';
    return 'rgba(59,130,246,' + alpha.toFixed(3) + ')';
  }

  function limit(vx, vy, max) {
    var mag2 = vx * vx + vy * vy;
    if (mag2 > max * max) {
      var mag = Math.sqrt(mag2);
      return [vx / mag * max, vy / mag * max];
    }
    return [vx, vy];
  }

  var _now = 0;

  function tick() {
    if (paused) { raf = null; return; }
    raf = requestAnimationFrame(tick);
    _now += 16;

    ctx.clearRect(0, 0, W, H);

    var n = boids.length;
    buildGrid();

    var mouseDisturbing = mouse.active && mouse.speed > MOUSE_SPEED_THRESH;
    mouse.speed *= 0.85;

    var activityMul = 1.0 + Math.min(state.active, 5) * 0.12;
    var curMaxSpeed = MAX_SPEED * activityMul;
    var curMaxForce = MAX_FORCE * activityMul;

    // --- Collect links for drawing (computed during physics) ---
    var links = [];

    // --- Physics ---
    for (var i = 0; i < n; i++) {
      var b = boids[i];
      var neighbors = getNeighborIndices(b);
      var calm = Math.max(0, 1 - b.scattered);

      var sepX = 0, sepY = 0, sepCount = 0;
      var aliX = 0, aliY = 0, aliCount = 0;
      var cohX = 0, cohY = 0, cohCount = 0;
      var pullX = 0, pullY = 0, pullCount = 0;

      for (var k = 0; k < neighbors.length; k++) {
        var j = neighbors[k];
        if (j === i) continue;
        var other = boids[j];
        var dx = b.x - other.x;
        var dy = b.y - other.y;
        var d2 = dx * dx + dy * dy;

        // Separation — always active
        if (d2 < SEPARATION_DIST * SEPARATION_DIST && d2 > 0) {
          var d = Math.sqrt(d2);
          sepX += dx / d / d;
          sepY += dy / d / d;
          sepCount++;
        }

        // Within neighbor radius — flocking
        if (d2 < NEIGHBOR_DIST * NEIGHBOR_DIST) {
          aliX += other.vx;
          aliY += other.vy;
          aliCount++;
          cohX += other.x;
          cohY += other.y;
          cohCount++;
        }

        // Within link distance — form bond + pull force
        if (d2 < LINK_DIST * LINK_DIST && j > i) {
          var dist = Math.sqrt(d2);
          // Link opacity
          var linkAlpha;
          if (d2 < LINK_STRONG * LINK_STRONG) {
            linkAlpha = LINK_OPACITY;
          } else {
            linkAlpha = LINK_OPACITY * (1 - (dist - LINK_STRONG) / (LINK_DIST - LINK_STRONG));
          }
          // Scattered boids = links fade/break
          linkAlpha *= calm * Math.max(0, 1 - other.scattered);
          // Energy brightens links
          linkAlpha += (b.energy + other.energy) * 0.05;

          if (linkAlpha > 0.005) {
            links.push(i, j, linkAlpha);
          }

          // Linked boids pull toward each other (the bond)
          // Stronger pull when further apart (spring-like, up to link dist)
          if (dist > LINK_STRONG * 0.8) {
            var pullStr = (dist - LINK_STRONG * 0.8) / (LINK_DIST - LINK_STRONG * 0.8);
            pullX -= (dx / dist) * pullStr;
            pullY -= (dy / dist) * pullStr;
            pullCount++;
          }
        }
      }

      var fx = 0, fy = 0;

      // Separation
      if (sepCount > 0) {
        var sv = limit(sepX, sepY, curMaxForce);
        fx += sv[0] * SEPARATION_W;
        fy += sv[1] * SEPARATION_W;
      }

      // Alignment — weakened during scatter
      if (aliCount > 0) {
        aliX /= aliCount; aliY /= aliCount;
        var desired = limit(aliX, aliY, curMaxSpeed);
        var av = limit(desired[0] - b.vx, desired[1] - b.vy, curMaxForce);
        fx += av[0] * ALIGNMENT_W * calm;
        fy += av[1] * ALIGNMENT_W * calm;
      }

      // Cohesion — the main swarming force, weakened during scatter
      if (cohCount > 0) {
        var tcx = cohX / cohCount - b.x;
        var tcy = cohY / cohCount - b.y;
        var cv = limit(tcx, tcy, curMaxForce);
        fx += cv[0] * COHESION_W * calm;
        fy += cv[1] * COHESION_W * calm;
      }

      // Link pull — bonded boids spring together
      if (pullCount > 0) {
        var pv = limit(pullX / pullCount, pullY / pullCount, curMaxForce);
        fx += pv[0] * LINK_PULL_W * calm;
        fy += pv[1] * LINK_PULL_W * calm;
      }

      // Wander — slow lazy drift when calm, erratic when startled
      var wanderMul = 1 + b.scattered * 4;
      b.wanderAngle += (Math.random() - 0.5) * (0.25 + b.scattered * 3.0);
      if (Math.random() < 0.003 + b.scattered * 0.08) {
        b.wanderAngle += (Math.random() - 0.5) * Math.PI * 0.5;
      }
      fx += Math.cos(b.wanderAngle) * curMaxForce * WANDER_W * wanderMul;
      fy += Math.sin(b.wanderAngle) * curMaxForce * WANDER_W * wanderMul;

      // --- Mouse ---
      if (mouse.active) {
        var mx = b.x - mouse.x;
        var my = b.y - mouse.y;
        var md2 = mx * mx + my * my;
        var md = Math.sqrt(md2 + 1);

        if (mouseDisturbing && md < MOUSE_SCATTER_RADIUS) {
          // Startled scatter — burst away with random spray
          var intensity = (1 - md / MOUSE_SCATTER_RADIUS);
          var burst = intensity * SCATTER_BURST;
          var fleeAngle = Math.atan2(my, mx) + (Math.random() - 0.5) * 1.5;
          b.vx += Math.cos(fleeAngle) * burst;
          b.vy += Math.sin(fleeAngle) * burst;
          b.scattered = Math.min(1, b.scattered + intensity * 0.7);
          b.energy = Math.min(1, b.energy + intensity * 0.4);
        } else if (!mouseDisturbing && md < MOUSE_ATTRACT_RADIUS && md > 25) {
          // Gentle attraction to calm cursor
          var attract = (1 - md / MOUSE_ATTRACT_RADIUS) * curMaxForce * 0.4 * calm;
          fx -= (mx / md) * attract;
          fy -= (my / md) * attract;
        }
      }

      // Scatter decay — gradually calm down and reform
      b.scattered *= SCATTER_DECAY;

      // Spontaneous startle — random boid gets spooked and startles neighbors
      if (b.scattered < 0.1 && Math.random() < STARTLE_CHANCE) {
        b.scattered = 0.8;
        b.energy = 0.6;
        var burstAngle = Math.random() * Math.PI * 2;
        b.vx += Math.cos(burstAngle) * STARTLE_BURST;
        b.vy += Math.sin(burstAngle) * STARTLE_BURST;
        // Ripple: spook nearby calm boids
        for (var si = 0; si < neighbors.length; si++) {
          var sj = neighbors[si];
          if (sj === i) continue;
          var sOther = boids[sj];
          var sdx = sOther.x - b.x;
          var sdy = sOther.y - b.y;
          var sd2 = sdx * sdx + sdy * sdy;
          if (sd2 < STARTLE_RADIUS * STARTLE_RADIUS && sd2 > 0) {
            var sd = Math.sqrt(sd2);
            var intensity = 1 - sd / STARTLE_RADIUS;
            // Contagion: already-scattered boids spread panic further
            var spread = intensity * STARTLE_CONTAGION;
            sOther.scattered = Math.min(1, sOther.scattered + spread);
            sOther.energy = Math.min(1, sOther.energy + spread * 0.3);
            var fleeAngle = Math.atan2(sdy, sdx) + (Math.random() - 0.5) * 1.2;
            sOther.vx += Math.cos(fleeAngle) * STARTLE_BURST * intensity * 0.6;
            sOther.vy += Math.sin(fleeAngle) * STARTLE_BURST * intensity * 0.6;
          }
        }
      }

      // Soft boundaries
      var margin = 50;
      if (b.x < margin) fx += curMaxForce * (margin - b.x) / margin;
      if (b.x > W - margin) fx -= curMaxForce * (b.x - (W - margin)) / margin;
      if (b.y < margin) fy += curMaxForce * (margin - b.y) / margin;
      if (b.y > H - margin) fy -= curMaxForce * (b.y - (H - margin)) / margin;

      // Apply
      b.vx += fx;
      b.vy += fy;

      // Speed limit — burst higher when scattered
      var boidMax = curMaxSpeed * b.speedMul * (1 + b.scattered * 2);
      var lim = limit(b.vx, b.vy, boidMax);
      b.vx = lim[0];
      b.vy = lim[1];

      // Damping — smooth glide
      b.vx *= 0.99;
      b.vy *= 0.99;

      // Minimum drift
      var spd2 = b.vx * b.vx + b.vy * b.vy;
      if (spd2 < 0.03) {
        b.vx += (Math.random() - 0.5) * 0.12;
        b.vy += (Math.random() - 0.5) * 0.12;
      }

      b.x += b.vx;
      b.y += b.vy;

      if (b.x < -50) b.x = W + 50;
      if (b.x > W + 50) b.x = -50;
      if (b.y < -50) b.y = H + 50;
      if (b.y > H + 50) b.y = -50;

      if (state.active > 0 && Math.random() < 0.02 * state.active) b.energy = 1.0;
      b.energy *= 0.95;

      // --- Spawn feelers ---
      // Random chance: a boid reaches out toward a distant boid
      if (feelers.length < MAX_FEELERS && b.scattered < 0.3 && Math.random() < FEELER_CHANCE) {
        // Find a distant boid (outside link range, within feeler range)
        var candidates = [];
        for (var fi = 0; fi < n; fi++) {
          if (fi === i) continue;
          var fdx = b.x - boids[fi].x;
          var fdy = b.y - boids[fi].y;
          var fd2 = fdx * fdx + fdy * fdy;
          if (fd2 > LINK_DIST * LINK_DIST && fd2 < FEELER_DIST * FEELER_DIST) {
            candidates.push(fi);
          }
        }
        if (candidates.length > 0) {
          var target = candidates[Math.floor(Math.random() * candidates.length)];
          feelers.push({
            from: i,
            to: target,
            age: 0,
            life: FEELER_DURATION + Math.floor(Math.random() * 40),
            progress: 0  // 0→1 how far the tendril has reached
          });
        }
      }
    }

    // --- Update feelers ---
    for (var fi = feelers.length - 1; fi >= 0; fi--) {
      var f = feelers[fi];
      f.age++;
      // Progress: grows out then holds then fades
      if (f.age < f.life * 0.3) {
        f.progress = f.age / (f.life * 0.3); // extend
      } else {
        f.progress = 1; // fully extended, then fading
      }
      // Remove expired or if source/target scattered
      if (f.age >= f.life || boids[f.from].scattered > 0.5 || boids[f.to].scattered > 0.5) {
        feelers.splice(fi, 1);
      }
    }

    // --- Draw links ---
    ctx.lineWidth = 0.8;
    for (var l = 0; l < links.length; l += 3) {
      var a = boids[links[l]];
      var ob = boids[links[l + 1]];
      var la = links[l + 2];
      ctx.strokeStyle = getColor(la);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(ob.x, ob.y);
      ctx.stroke();
    }

    // --- Draw feelers ---
    // Animated tendrils reaching from one boid toward a distant one
    for (var fi = 0; fi < feelers.length; fi++) {
      var f = feelers[fi];
      var fa = boids[f.from];
      var fb = boids[f.to];
      var fdx = fb.x - fa.x;
      var fdy = fb.y - fa.y;

      // Fade: ramp up, hold, ramp down
      var fadeIn = Math.min(f.age / (f.life * 0.15), 1);
      var fadeOut = Math.max(0, 1 - (f.age - f.life * 0.7) / (f.life * 0.3));
      var alpha = FEELER_OPACITY * fadeIn * fadeOut;
      if (alpha < 0.003) continue;

      // Draw as a dashed/dotted tendril extending from source toward target
      var endX = fa.x + fdx * f.progress;
      var endY = fa.y + fdy * f.progress;
      var dist = Math.sqrt(fdx * fdx + fdy * fdy);

      // Main tendril line
      ctx.strokeStyle = getColor(alpha);
      ctx.lineWidth = 0.6;
      ctx.setLineDash([4, 6]);
      ctx.beginPath();
      ctx.moveTo(fa.x, fa.y);
      ctx.lineTo(endX, endY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Small dot at the tip of the feeler
      if (f.progress > 0.1) {
        ctx.fillStyle = getColor(alpha * 1.5);
        ctx.beginPath();
        ctx.arc(endX, endY, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // If feeler reaches target (progress ~1 and boids close enough), flash the endpoint
      if (f.progress > 0.95) {
        var reachDx = endX - fb.x;
        var reachDy = endY - fb.y;
        var reachD = Math.sqrt(reachDx * reachDx + reachDy * reachDy);
        if (reachD < 15) {
          ctx.fillStyle = getGlowColor(alpha * 2);
          ctx.beginPath();
          ctx.arc(fb.x, fb.y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    // --- Draw boids ---
    for (var i = 0; i < n; i++) {
      var b = boids[i];
      var angle = Math.atan2(b.vy, b.vx);
      var nodePulse = Math.sin(_now * PULSE_SPEED + b.phase) * 0.2 + 0.8;
      var size = b.size * nodePulse;

      var energyGlow = b.energy * 0.5;
      var scatterGlow = b.scattered * 0.3;

      var mouseGlow = 0;
      if (mouse.active) {
        var mdx = b.x - mouse.x;
        var mdy = b.y - mouse.y;
        var md2 = mdx * mdx + mdy * mdy;
        if (md2 < MOUSE_ATTRACT_RADIUS * MOUSE_ATTRACT_RADIUS) {
          mouseGlow = (1 - md2 / (MOUSE_ATTRACT_RADIUS * MOUSE_ATTRACT_RADIUS)) * 0.4;
        }
      }

      var alpha = 0.55 + mouseGlow + energyGlow + scatterGlow;
      var s = size + mouseGlow * 2.5 + energyGlow * 2 + scatterGlow * 1.5;
      var headLen = s * 2;
      var tailWidth = s * 1.1;

      ctx.fillStyle = getColor(Math.min(alpha, 1));
      ctx.beginPath();
      ctx.moveTo(b.x + Math.cos(angle) * headLen, b.y + Math.sin(angle) * headLen);
      ctx.lineTo(b.x + Math.cos(angle + 2.5) * tailWidth, b.y + Math.sin(angle + 2.5) * tailWidth);
      ctx.lineTo(b.x + Math.cos(angle - 2.5) * tailWidth, b.y + Math.sin(angle - 2.5) * tailWidth);
      ctx.closePath();
      ctx.fill();

      if (mouseGlow > 0.15 || energyGlow > 0.25 || scatterGlow > 0.15) {
        var haloAlpha = Math.max(mouseGlow, energyGlow, scatterGlow) * 0.12;
        ctx.fillStyle = getGlowColor(haloAlpha);
        ctx.beginPath();
        ctx.arc(b.x, b.y, s * 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  function updateState(data) {
    if (data.peers !== undefined) state.peers = data.peers;
    if (data.active_requests !== undefined) state.active = data.active_requests;
    if (data.peers !== undefined) {
      state.health = data.peers > 0 ? 1.0 : 0.7;
    }
  }

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
