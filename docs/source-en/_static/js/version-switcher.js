(function () {
  "use strict";

  var script = document.currentScript;
  if (!script || typeof DOCUMENTATION_OPTIONS === "undefined") return;

  var url = new URL(script.src, window.location.href);
  url.search = "";
  url.hash = "";
  url.pathname = url.pathname.replace(/\/js\/version-switcher\.js$/, "/versions.json");
  DOCUMENTATION_OPTIONS.theme_switcher_json_url = url.toString();
})();
