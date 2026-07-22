(function () {
  "use strict";

  function scrollableAncestor(element) {
    for (var parent = element.parentElement; parent; parent = parent.parentElement) {
      var overflow = getComputedStyle(parent).overflowY;
      if ((overflow === "auto" || overflow === "scroll") && parent.scrollHeight > parent.clientHeight) {
        return parent;
      }
    }
    return null;
  }

  function init() {
    var navigation = document.querySelector(".bd-docs-nav");
    if (!navigation) return;

    navigation.querySelectorAll("li.toctree-l1.has-children > details").forEach(
      function (section) {
        section.setAttribute("open", "");
      }
    );

    var current = navigation.querySelector("a.current");
    var scroller = current && scrollableAncestor(current);
    if (!current || !scroller) return;

    var box = scroller.getBoundingClientRect();
    var item = current.getBoundingClientRect();
    scroller.scrollTop += item.top - box.top - scroller.clientHeight / 2 + item.height / 2;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
