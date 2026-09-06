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
  if (exportName && (!App[exportName] || typeof App[exportName] !== 'object')) {
    console.error(`NO EXPORT ${rel}: App.${exportName} was never assigned`);
    failed++;
    continue;
  }
  const n = exportName ? Object.keys(App[exportName]).length : 0;
  console.log(`ok       ${rel}${exportName ? ` (App.${exportName}, ${n} exports)` : ''}`);
}
process.exit(failed ? 1 : 0);
