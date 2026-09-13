#!/usr/bin/env node
//
// Load every frontend module the way the browser does, and report what throws.
//
// **`node -c` is a syntax check and nothing more** (gotcha #568): a reference
// to a name that is out of scope inside a function body passes it, and so does
// a component whose IIFE throws the moment it runs. The pre-push check in
// `.claude/rules/completeness.md` runs `node -c` over every file, which is
// worth having and is not this.
//
// What it adds, each verified by planting the violation rather than assumed:
//
//   * a module-scope reference to a symbol that was deleted — the shape of
//     gotcha #585, where removing a renderer left its callers pointing at
//     nothing. Planted `App.dashboard._renderModelTicker()` at module scope:
//     `THREW  js/components/pool.js`;
//   * `index.html` pointing at a script that is not there. Planted: `MISSING`;
//   * an export other code depends on going away in a refactor — `MUST_EXIST`
//     below. Planted by deleting `filterByModel` from the notifications
//     export: caught.
//
// **What it does NOT catch, and this is the important half.** A missing
// `var U = App.utils` in a component's IIFE boilerplate — the R111 swarm-tab
// regression — passes cleanly, because `U` is referenced only inside functions
// that no load-time code runs. Planted and confirmed: `OK`. Anything reachable
// only by rendering a view needs the view rendered, and this stub is nowhere
// near a browser. Do not read a pass here as "the frontend works".
//
// The load ORDER is read out of `index.html` rather than hardcoded, so it
// cannot drift from what ships.
//
// Usage:  node examples/frontend_load_check.js
// Exits non-zero if any module fails to load or any expected export is absent.
//
// The DOM stub is deliberately shallow. This checks that modules LOAD, not that
// they render — two files (`core/utils.js`, `init.js`) legitimately touch real
// elements at load time and are reported as expected-fail rather than pretended
// away, because a stub deep enough to satisfy them would be a browser.

const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..', 'frontend');
const EXPECTED_DOM_DEPENDENT = new Set(['js/core/utils.js', 'js/init.js']);

// Exports that must survive any refactor of the components that own them.
const MUST_EXIST = [
  'notifications.filterByModel',
  'networkMap.renderRouteKey',
  'dashboardShards.buildShardRow',
  'chat.refreshEmptyState',
  'utils.renderReplyInto',
];

const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const order = [...html.matchAll(/<script src="\/static\/(js\/[^"]+)"><\/script>/g)].map((m) => m[1]);
if (order.length < 20) {
  console.error(`only ${order.length} scripts found in index.html — the selector has drifted`);
  process.exit(1);
}

const noop = () => {};
const el = new Proxy({}, {
  get: (_t, k) =>
    k === 'style' ? { setProperty: noop, removeProperty: noop }
    : k === 'classList' ? { add: noop, remove: noop, contains: () => false, toggle: noop }
    : k === 'dataset' ? {}
    : k === 'content' ? { cloneNode: () => ({ firstElementChild: null }) }
    : ['innerHTML', 'textContent', 'value', 'id', 'className'].includes(k) ? ''
    : noop,
});

// `window` IS the global scope in a browser, so `window.App = {...}` in
// state.js has to make a bare `App` visible to every later file. A plain
// object does not, and getting this wrong makes every module "fail".
global.window = globalThis;
Object.assign(global.window, {
  addEventListener: noop,
  location: { href: '', pathname: '/', search: '', origin: 'http://localhost:8800' },
  matchMedia: () => ({ matches: false, addEventListener: noop, addListener: noop }),
  localStorage: { getItem: () => null, setItem: noop, removeItem: noop },
  sessionStorage: { getItem: () => null, setItem: noop, removeItem: noop },
  navigator: { language: 'en', languages: ['en'] },
});
global.document = {
  getElementById: () => null, querySelector: () => null, querySelectorAll: () => [],
  createElement: () => el, createElementNS: () => el, addEventListener: noop,
  body: el, documentElement: el, head: el, cookie: '', readyState: 'complete', title: '',
};
global.fetch = () => Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
global.WebSocket = function () { return { addEventListener: noop, close: noop }; };
global.AbortController = function () { return { abort: noop, signal: {} }; };
global.requestAnimationFrame = (f) => setTimeout(f, 0);
global.self = global.window;
global.Notification = function () {};
global.Notification.permission = 'default';

let failed = 0;
let loaded = 0;
for (const rel of order) {
  const p = path.join(ROOT, rel);
  if (!fs.existsSync(p)) {
    console.log(`MISSING   ${rel}  — index.html points at a file that is not there`);
    failed++;
    continue;
  }
  try {
    (0, eval)(fs.readFileSync(p, 'utf8'));
    loaded++;
  } catch (e) {
    if (EXPECTED_DOM_DEPENDENT.has(rel)) {
      console.log(`dom-dep   ${rel}  — touches real elements at load (expected)`);
    } else {
      console.log(`THREW     ${rel}  → ${e.message}`);
      failed++;
    }
  }
}

console.log(`\nloaded ${loaded}/${order.length} modules; App has ${global.App ? Object.keys(global.App).length : 0} keys`);

for (const spec of MUST_EXIST) {
  const [ns, member] = spec.split('.');
  const found = global.App && global.App[ns] && global.App[ns][member];
  console.log(`${found ? 'ok       ' : 'MISSING  '} App.${spec}`);
  if (!found) failed++;
}

// A timer scheduled by a module that has since been unloaded will fire after
// this script's work is done and throw into an empty world; nothing here waits
// for one, so exit rather than let node keep the loop alive.
console.log(failed === 0 ? '\nOK' : `\n${failed} problem(s)`);
process.exit(failed === 0 ? 0 : 1);
