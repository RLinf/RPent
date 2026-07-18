(function () {
  "use strict";

  function setMode(mode) {
    var root = document.documentElement;
    root.dataset.mode = mode;
    root.dataset.theme = mode;
    try {
      localStorage.setItem("mode", mode);
      localStorage.setItem("theme", mode);
    } catch (error) {
      // Storage can be disabled without affecting the current page.
    }
  }

  function init() {
    var button = document.querySelector(".rpent-theme-toggle");
    if (!button) return;
    button.addEventListener("click", function () {
      var isDark = document.documentElement.dataset.theme === "dark";
      setMode(isDark ? "light" : "dark");
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
