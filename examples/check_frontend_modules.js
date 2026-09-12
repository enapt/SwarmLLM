// Load the frontend's core modules the way a browser does, and fail if one
// throws while building its exports.
//
// **Why this exists.** v0.3.160 shipped a dashboard that never left
// "Connecting…" on every machine. One helper had been added to the
// `App.utils = { … }` export list while its definition sat inside another
// function's body, so the name was not in scope when the object literal was
// evaluated: `ReferenceError`, `App.utils` never assigned at all, and every
// component reading `App.utils.<anything>` died with it. One misplaced
// function took down the entire admin UI (gotcha #488).
//
// `node -c` — the only frontend check this project had — cannot see this: the
// file is syntactically perfect. Nor can a source scan that assumes module
// scope means a particular indentation; the misplaced function was indented
// exactly like a top-level one, and a first attempt at such a guard passed
// against the broken file. The only thing that reliably catches it is
// EVALUATING the module, which is what this does.
//
//   node examples/check_frontend_modules.js
//
// Exit 0 = every module built its exports. Non-zero = the name and the error.
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const root = path.resolve(__dirname, '..');
// Loaded in dependency order, as index.html loads them.
const MODULES = [
  ['frontend/js/core/state.js', null],
  ['frontend/js/core/utils.js', 'utils'],
  ['frontend/js/core/data.js', 'data'],
  ['frontend/js/core/tooltip.js', null],
  // Components load-check too: one that throws while building its exports
  // takes its whole namespace with it, exactly as `utils` did.
  //
  // Order matters and mirrors `index.html`: `dashboard.js` aliases
  // `App.dashboardShards.*` into locals at load time, so its dependency comes
  // first — the same reason the page loads them in that order.
  ['frontend/js/components/dashboard-shards.js', 'dashboardShards'],
  ['frontend/js/components/dashboard.js', 'dashboard'],
  ['frontend/js/components/chat.js', 'chat'],
];

// Render functions CALLED with stubs, because loading a module cannot see a
// name that is only missing inside a function body.
//
// `_renderHardware` read `data.network_traffic` while its parameter was `hw`
// (2026-09-12). Syntactically perfect, loads perfectly, and throws
// `ReferenceError: data is not defined` the first time the dashboard renders —
// which is every two seconds, on every node. That is the same failure as the
// one this file was written for (gotcha #488), one level deeper: there the
// name was missing when the module was built, here when the function ran.
//
// Only functions that are pure render-from-arguments belong here. Anything
// that starts a timer, fetches, or loops over live state does not: the point
// is to evaluate the BODY, not to simulate the app.
const SMOKE_CALLS = [
  ['App.dashboard._renderHardware(hw, traffic)', () => App.dashboard._renderHardware(
    {
      cpu_name: 'Test CPU', cpu_cores: 8,
      total_ram_mb: 16000, used_ram_mb: 8000, process_rss_mb: 4000,
      daemon_rss_mb: 300, worker_rss_mb: 3700, worker_count: 1,
      total_disk_mb: 500000, used_disk_mb: 100000,
      gpu_name: null, gpu_vram_mb: 0, gpu_vram_used_mb: 0, gpu_inference: false,
    },
    { in_bytes: 1024, out_bytes: 2048, in_bytes_per_sec: 12.5, out_bytes_per_sec: 30.0 },
  )],
  ['App.dashboard._renderHardware(hw, null)', () => App.dashboard._renderHardware(
    { cpu_name: 'Test CPU', cpu_cores: 8, total_ram_mb: 16000, total_disk_mb: 500000 },
    null,
  )],
];

// A permissive stand-in for anything a browser provides. Returns itself for
// every property and call, so module-load-time DOM poking does not throw and
// the check stays about SCOPE rather than about how good these stubs are.
const stub = new Proxy(function () {}, {
  get: (_t, k) => (k === Symbol.toPrimitive ? () => '' : stub),
  apply: () => stub,
  construct: () => stub,
  has: () => true,
});

const App = {};
const sandbox = {
  App,
  window: {
    App,
    addEventListener() {},
    matchMedia: () => ({ matches: false, addEventListener() {} }),
    location: { href: 'http://localhost/', host: 'localhost', hostname: 'localhost', protocol: 'http:', pathname: '/', search: '', origin: 'http://localhost' },
    localStorage: stub,
    sessionStorage: stub,
    navigator: stub,
  },
  document: stub,
  navigator: stub,
  localStorage: stub,
  sessionStorage: stub,
  location: { href: 'http://localhost/', host: 'localhost', hostname: 'localhost', protocol: 'http:', pathname: '/', search: '', origin: 'http://localhost' },
  I18n: stub,
  console,
  setTimeout, clearTimeout, setInterval, clearInterval,
  fetch: () => stub,
  WebSocket: stub,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

let failed = 0;
for (const [rel, exportName] of MODULES) {
  const file = path.join(root, rel);
  if (!fs.existsSync(file)) {
    console.error(`MISSING  ${rel}`);
    failed++;
    continue;
  }
  try {
    vm.runInContext(fs.readFileSync(file, 'utf8'), sandbox, { filename: rel });
  } catch (e) {
    console.error(`THREW    ${rel}: ${e.constructor.name}: ${e.message}`);
    failed++;
    continue;
  }
  // `state.js` creates the namespace with `window.App = { … }`, which REPLACES
  // the object every other module reaches through the bare global `App`. Left
  // alone, the sandbox ends up with two namespaces and components load against
  // an empty one — so the constants state.js defines read as `undefined` and
  // the check fails for a reason that has nothing to do with the code. Fold
  // them into one identity after each module.
  if (sandbox.window.App && sandbox.window.App !== App) {
    Object.assign(App, sandbox.window.App);
    sandbox.window.App = App;
  }
  if (exportName && (!App[exportName] || typeof App[exportName] !== 'object')) {
    console.error(`NO EXPORT ${rel}: App.${exportName} was never assigned`);
    failed++;
    continue;
  }
  const n = exportName ? Object.keys(App[exportName]).length : 0;
  console.log(`ok       ${rel}${exportName ? ` (App.${exportName}, ${n} exports)` : ''}`);
}

// The DOM these need is the permissive stub, so what survives this is the
// SCOPE of every name the body touches — which is the whole point.
for (const [label, call] of SMOKE_CALLS) {
  try {
    vm.runInContext('(' + call.toString() + ')()', sandbox, { filename: label });
    console.log(`ok       ${label}`);
  } catch (e) {
    console.error(`THREW    ${label}: ${e.constructor.name}: ${e.message}`);
    failed++;
  }
}
process.exit(failed ? 1 : 0);
