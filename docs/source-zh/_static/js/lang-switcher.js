(function () {
  "use strict";

  function init() {
    var match = window.location.pathname.match(/^\/(en|zh-cn)\/([^/]+)\/(.*)$/);
    if (!match) return;

    var version = match[2];
    var page = match[3];
    var english = document.querySelector(".rpent-lang-en");
    var chinese = document.querySelector(".rpent-lang-zh");

    if (english) english.setAttribute("href", "/en/" + version + "/" + page);
    if (chinese) chinese.setAttribute("href", "/zh-cn/" + version + "/" + page);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
