'use strict';

// ============================================================================
// SwarmLLM — Welcome Tour Component
// One-time first-run modal that explains the 4 most important header
// elements. Triggered by Setup.complete / Setup.finish; re-openable from
// Settings via #btn-show-welcome.
// ============================================================================

(function () {
  if (!window.App) return;

  function $(id) { return document.getElementById(id); }

  App.welcome = {
    // Show the modal if the user hasn't seen it yet AND they've completed or
    // skipped Setup (so it doesn't stack on top of the wizard).
    maybeShow: function () {
      if (localStorage.getItem(App.WELCOME_SEEN_KEY) === 'true') return;
      var done = localStorage.getItem(App.SETUP_DONE_KEY) === 'true';
      var skipped = localStorage.getItem(App.SETUP_SKIPPED_KEY) === 'true';
      if (!done && !skipped) return;
      App.welcome.show();
    },

    show: function () {
      var m = $('welcome-modal');
      if (!m) return;
      m.classList.remove('hidden');
    },

    dismiss: function () {
      var m = $('welcome-modal');
      if (m) m.classList.add('hidden');
      localStorage.setItem(App.WELCOME_SEEN_KEY, 'true');
    },

    // Re-open from Settings. Doesn't clear the seen flag — the user is
    // explicitly re-reading the tour, not being shown it for the first time.
    reopen: function () {
      App.welcome.show();
    },

    init: function () {
      var dismiss = function () { App.welcome.dismiss(); };
      var close = $('btn-welcome-close');
      if (close) close.addEventListener('click', dismiss);
      var got = $('btn-welcome-got-it');
      if (got) got.addEventListener('click', dismiss);
      // Backdrop click closes too.
      var overlay = $('welcome-modal');
      if (overlay) {
        overlay.addEventListener('click', function (e) {
          if (e.target === overlay) dismiss();
        });
      }
      var settingsBtn = $('btn-show-welcome');
      if (settingsBtn) settingsBtn.addEventListener('click', function () { App.welcome.reopen(); });
    },
  };
})();
