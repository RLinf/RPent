/* Progressive enhancement for the generated, accessible benchmark SVGs. */
(() => {
  "use strict";

  const root = document.getElementById("rpent-interactive-leaderboard");
  if (!root) return;
  const language = root.dataset.language === "zh" ? "zh" : "en";
  const labels = {
    en: {
      benchmark: "Benchmark", suite: "Task / suite", overview: "Overview · six panels",
      allSuites: "All evaluation items", configurations: "RPent model configurations",
      external: "Show external methods", sort: "Success rate", desc: "Highest first",
      asc: "Lowest first", reset: "Reset", csv: "Download CSV", loading: "Loading charts…",
      ready: "panels shown. Hover over or focus a bar for evaluation details.",
      failed: "Interactive charts could not load. The static overview and full result tables remain available below.",
      notReported: "Not reported", empty: "No reported results match these filters.",
      model: "Model / method", backend: "Backend", reasoning: "Reasoning", effort: "Effort",
      rate: "Success rate", count: "Successes / episodes", episodes: "Evaluation episodes",
      source: "Source", protocol: "Setting", on: "On", off: "Off", na: "Not applicable",
      details: "Results and evaluation details", result: "result", results: "results",
      hint: "Filters keep each evaluation setting separate. Axis limits stay fixed when models are hidden.",
    },
    zh: {
      benchmark: "基准", suite: "任务／套件", overview: "总览 · 六个面板",
      allSuites: "全部评测分项", configurations: "RPent 模型配置",
      external: "显示外部方法", sort: "成功率排序", desc: "从高到低",
      asc: "从低到高", reset: "重置", csv: "下载 CSV", loading: "正在加载图表…",
      ready: "个面板。悬停或用键盘聚焦柱子可查看评测详情。",
      failed: "交互图表加载失败，下方仍可查看静态总览和完整结果表。",
      notReported: "未报告", empty: "当前筛选没有已报告成绩。",
      model: "模型／方法", backend: "后端", reasoning: "推理模式", effort: "Effort",
      rate: "成功率", count: "成功数／回合数", episodes: "评测回合数",
      source: "来源", protocol: "评测设置", on: "开启", off: "关闭", na: "不适用",
      details: "结果与评测说明", result: "条结果", results: "条结果",
      hint: "各评测设置独立展示。隐藏模型时，纵轴范围保持不变。",
    },
  }[language];
  const localized = (value) => typeof value === "string" ? value : value?.[language] || value?.en || "";
  const create = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const mapById = (items) => new Map(items.map((item) => [item.id, item]));
  const fallback = root.querySelector(".rpent-static-leaderboard");
  const controls = create("div", "rpent-leaderboard-controls");
  controls.hidden = true;
  const grid = create("div", "rpent-leaderboard-grid");
  grid.id = "rpent-interactive-grid";
  grid.hidden = true;
  const status = create("p", "rpent-leaderboard-status");
  status.setAttribute("role", "status");
  status.setAttribute("aria-live", "polite");
  root.prepend(controls);
  root.append(grid, status);

  const tooltip = create("div", "rpent-leaderboard-tooltip");
  tooltip.id = "rpent-leaderboard-tooltip";
  tooltip.setAttribute("role", "tooltip");
  tooltip.hidden = true;
  document.body.append(tooltip);
  let activeBar = null;
  let tooltipPositionFrame = 0;
  let generation = 0;
  let data;
  let configurations;
  let records;
  let sources;
  let protocols;
  let benchmarkSelect;
  let suiteSelect;
  let sortSelect;
  let externalInput;
  let csvButton;
  let configurationInputs = [];
  let visibleRecords = [];
  const svgCache = new Map();
  const theme = () => document.documentElement.dataset.theme === "dark" ? "dark" : "light";

  function hideTooltip() {
    window.cancelAnimationFrame(tooltipPositionFrame);
    tooltipPositionFrame = 0;
    tooltip.hidden = true;
    if (activeBar) activeBar.removeAttribute("aria-describedby");
    activeBar = null;
  }

  function repositionFocusedTooltip() {
    if (!activeBar) return;
    if (document.activeElement !== activeBar) {
      hideTooltip();
      return;
    }
    // Focusing an offscreen SVG bar scrolls the page after its focus event.
    // Keep that keyboard tooltip visible without reviving an Escape dismissal.
    if (tooltipPositionFrame) return;
    const bar = activeBar;
    tooltipPositionFrame = window.requestAnimationFrame(() => {
      tooltipPositionFrame = 0;
      if (activeBar === bar && document.activeElement === bar && !tooltip.hidden) {
        showTooltip(bar, records.get(bar.dataset.recordId));
      }
    });
  }

  function details(record) {
    const configuration = configurations.get(record.configuration_id);
    const unspecified = configuration.metadata_status === "not_reported" ? labels.notReported : labels.na;
    const view = data.views.find((item) => item.id === record.view_id);
    const fields = [
      [labels.protocol, localized(view.title)],
      [labels.model, configuration.model],
      [labels.backend, configuration.backend || unspecified],
      [labels.reasoning, configuration.reasoning === null ? unspecified : configuration.reasoning ? labels.on : labels.off],
      [labels.effort, configuration.effort || unspecified],
      [labels.rate, record.status === "reported" ? `${record.rate}%` : labels.notReported],
    ];
    if (record.successes !== null && record.episodes !== null) {
      fields.push([labels.count, `${record.successes}/${record.episodes}`]);
    } else if (record.episodes !== null) {
      fields.push([labels.episodes, String(record.episodes)]);
    }
    fields.push([labels.source, record.source_ids.map((id) => localized(sources.get(id)?.label)).filter(Boolean).join("; ")]);
    return fields;
  }

  function showTooltip(bar, record, event) {
    if (activeBar !== bar) hideTooltip();
    activeBar = bar;
    tooltip.replaceChildren();
    const list = create("dl");
    for (const [label, value] of details(record)) {
      list.append(create("dt", "", label), create("dd", "", value));
    }
    tooltip.append(list);
    tooltip.hidden = false;
    bar.setAttribute("aria-describedby", tooltip.id);
    const bounds = bar.getBoundingClientRect();
    const x = event?.clientX ?? (bounds.left + bounds.width / 2);
    const y = event?.clientY ?? bounds.top;
    const padding = 12;
    const box = tooltip.getBoundingClientRect();
    const left = Math.max(padding, Math.min(x + padding, window.innerWidth - box.width - padding));
    let top = y - box.height - padding;
    if (top < padding) top = y + padding;
    top = Math.max(padding, Math.min(top, window.innerHeight - box.height - padding));
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }

  function addSelect(parent, id, title, options) {
    const field = create("label", "rpent-leaderboard-field");
    field.htmlFor = id;
    field.append(create("span", "", title));
    const select = create("select");
    select.id = id;
    for (const [value, text] of options) select.add(new Option(text, value));
    field.append(select);
    parent.append(field);
    select.addEventListener("change", () => render());
    return select;
  }

  function availableViews() {
    if (benchmarkSelect.value === "overview") {
      return data.overview_views.map((id) => data.views.find((view) => view.id === id));
    }
    return data.views.filter((view) => view.benchmark_id === benchmarkSelect.value);
  }

  function updateSuites() {
    suiteSelect.replaceChildren(new Option(labels.allSuites, "all"));
    for (const view of availableViews()) suiteSelect.add(new Option(localized(view.label), view.id));
    suiteSelect.disabled = benchmarkSelect.value === "overview";
  }

  function selectedRecords(view) {
    const overview = benchmarkSelect.value === "overview";
    const plottedIds = new Set(overview ? view.overview_record_ids : view.chart_record_ids);
    const chosenConfigurations = new Set(configurationInputs.filter((input) => input.checked).map((input) => input.value));
    return view.record_ids.map((id) => records.get(id)).filter((record) => {
      const configuration = configurations.get(record.configuration_id);
      if (configuration.kind === "external") return externalInput.checked && plottedIds.has(record.id);
      return chosenConfigurations.has(configuration.id) && (record.status !== "reported" || plottedIds.has(record.id));
    }).sort((left, right) => {
      if (left.status !== right.status) return left.status === "reported" ? -1 : 1;
      const direction = sortSelect.value === "asc" ? 1 : -1;
      return direction * (Number(left.rate) - Number(right.rate)) || left.id.localeCompare(right.id);
    });
  }

  async function loadSvg(view, currentTheme) {
    const url = new URL(`${view.asset_slug || view.id}-${language}-${currentTheme}.svg`, root.dataset.assetBase);
    if (!svgCache.has(url.href)) {
      svgCache.set(url.href, fetch(url).then((response) => {
        if (!response.ok) throw new Error(`SVG response ${response.status}`);
        return response.text();
      }).then((text) => {
        const documentSvg = new DOMParser().parseFromString(text, "image/svg+xml");
        const svg = documentSvg.documentElement;
        if (documentSvg.querySelector("parsererror") || svg.localName !== "svg") throw new Error("Invalid chart SVG");
        return svg;
      }).catch((error) => {
        svgCache.delete(url.href);
        throw error;
      }));
    }
    const svg = document.importNode(await svgCache.get(url.href), true);
    // Each inline chart must own its glyph and clipping IDs, including repeated fonts.
    const prefix = `rpent-${view.id}-${currentTheme}-`;
    const ids = new Map(Array.from(svg.querySelectorAll("[id]"), (node) => [node.id, prefix + node.id]));
    for (const node of svg.querySelectorAll("*")) {
      if (node.id) node.id = ids.get(node.id);
      for (const attribute of Array.from(node.attributes)) {
        let value = attribute.value.replace(/url\(#([^)]*)\)/g, (match, id) => ids.has(id) ? `url(#${ids.get(id)})` : match);
        if ((attribute.localName === "href") && value.startsWith("#") && ids.has(value.slice(1))) value = `#${ids.get(value.slice(1))}`;
        if (value !== attribute.value) node.setAttributeNS(attribute.namespaceURI, attribute.name, value);
      }
    }
    svg.removeAttribute("width");
    svg.removeAttribute("height");
    svg.classList.add("rpent-interactive-chart");
    svg.setAttribute("role", "group");
    svg.setAttribute("aria-label", `${localized(view.title)}. ${localized(view.scope || view.subtitle)}`);
    return svg;
  }

  async function renderPanel(view, currentTheme, selected) {
    const figure = create("figure", "rpent-leaderboard-panel");
    figure.dataset.viewId = view.id;
    const svg = await loadSvg(view, currentTheme);
    const plotted = selected.filter((record) => record.status === "reported");
    const visible = new Map(plotted.map((record, index) => [record.id, index]));
    const left = Number(svg.dataset.axisLeft);
    const width = Number(svg.dataset.axisWidth);
    if (!Number.isFinite(left) || !Number.isFinite(width) || width <= 0) throw new Error("Missing chart geometry");
    const bars = Array.from(svg.querySelectorAll("[data-record-id]"));
    if (plotted.some((record) => !bars.some((bar) => bar.dataset.recordId === record.id))) throw new Error("Missing generated record");
    for (const bar of bars) {
      const index = visible.get(bar.dataset.recordId);
      if (index === undefined) {
        bar.style.display = "none";
        bar.setAttribute("aria-hidden", "true");
        bar.removeAttribute("tabindex");
        continue;
      }
      const record = records.get(bar.dataset.recordId);
      const target = left + width * (index + 0.62) / (plotted.length + 0.24);
      bar.setAttribute("transform", `translate(${target - Number(bar.dataset.centerX)} 0)`);
      bar.classList.add("rpent-bar-record");
      bar.setAttribute("tabindex", "0");
      bar.setAttribute("role", "img");
      bar.setAttribute("aria-label", details(record).map(([label, value]) => `${label}: ${value}`).join(". "));
      bar.addEventListener("pointerenter", (event) => showTooltip(bar, record, event));
      bar.addEventListener("pointermove", (event) => showTooltip(bar, record, event));
      bar.addEventListener("pointerleave", () => { if (document.activeElement !== bar) hideTooltip(); });
      bar.addEventListener("focus", () => showTooltip(bar, record));
      bar.addEventListener("blur", hideTooltip);
      bar.addEventListener("keydown", (event) => { if (event.key === "Escape") hideTooltip(); });
    }
    // Preserve the visible sorting order for keyboard and assistive technology too.
    for (const record of plotted) {
      const bar = bars.find((candidate) => candidate.dataset.recordId === record.id);
      bar.parentNode.append(bar);
    }
    figure.append(svg);
    const caption = create("figcaption");
    const sourceIds = Array.from(new Set(selected.flatMap((record) => record.source_ids)));
    for (const [index, id] of sourceIds.entries()) {
      const source = sources.get(id);
      if (!source) continue;
      if (index) caption.append(document.createTextNode(" · "));
      const link = create("a", "", localized(source.label));
      link.href = source.anchor ? `#${source.anchor}` : source.url;
      caption.append(link);
    }
    const missing = selected.filter((record) => record.status !== "reported");
    if (missing.length) caption.append(create("p", "rpent-unreported", `${labels.notReported}: ${missing.map((record) => {
      const configuration = configurations.get(record.configuration_id);
      return configuration.effort ? `${configuration.model} · ${configuration.effort}` : configuration.model;
    }).join("; ")}`));
    if (!plotted.length) caption.append(create("p", "rpent-empty-results", labels.empty));
    figure.append(caption);
    return figure;
  }

  async function render() {
    const revision = ++generation;
    hideTooltip();
    status.textContent = labels.loading;
    grid.setAttribute("aria-busy", "true");
    csvButton.disabled = true;
    const views = availableViews().filter((view) => suiteSelect.value === "all" || suiteSelect.value === view.id);
    const currentTheme = theme();
    const selection = views.map((view) => selectedRecords(view));
    try {
      const figures = await Promise.all(views.map((view, index) => renderPanel(view, currentTheme, selection[index])));
      if (revision !== generation) return;
      grid.replaceChildren(...figures);
      grid.hidden = false;
      fallback.hidden = true;
      controls.hidden = false;
      visibleRecords = selection.flat();
      status.textContent = `${figures.length} ${labels.ready}`;
      root.dataset.interactive = "ready";
      csvButton.disabled = false;
    } catch (error) {
      if (revision !== generation) return;
      grid.hidden = true;
      fallback.hidden = false;
      controls.hidden = false;
      root.dataset.interactive = "failed";
      status.textContent = labels.failed;
      console.warn("RPent leaderboard:", error);
    } finally {
      if (revision === generation) grid.removeAttribute("aria-busy");
    }
  }

  function exportCsv() {
    const columns = ["benchmark", "evaluation_item", "method_kind", "model", "backend", "reasoning", "effort", "planner_metadata_status", "status", "success_rate_percent", "successes", "episodes", "protocol", "sources"];
    const rows = visibleRecords.map((record) => {
      const configuration = configurations.get(record.configuration_id);
      const view = data.views.find((item) => item.id === record.view_id);
      const benchmark = data.benchmarks.find((item) => item.id === view.benchmark_id);
      return [localized(benchmark.label), localized(view.label), configuration.kind, configuration.model, configuration.backend, configuration.reasoning, configuration.effort, configuration.metadata_status || "reported",
        record.status, record.rate, record.successes, record.episodes, localized(protocols.get(record.protocol_id)?.label),
        record.source_ids.map((id) => sources.get(id)?.url).filter(Boolean).join("; ")];
    });
    const quote = (value) => `"${String(value ?? "").replace(/"/g, '""')}"`;
    const csv = "\uFEFF" + [columns, ...rows].map((row) => row.map(quote).join(",")).join("\r\n") + "\r\n";
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = create("a");
    link.href = url;
    link.download = `rpent-benchmark-results-${language}.csv`;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function setupControls() {
    const fields = create("div", "rpent-leaderboard-fields");
    controls.append(fields);
    benchmarkSelect = addSelect(fields, "rpent-benchmark-select", labels.benchmark, [
      ["overview", labels.overview], ...data.benchmarks.map((benchmark) => [benchmark.id, localized(benchmark.label)]),
    ]);
    suiteSelect = addSelect(fields, "rpent-suite-select", labels.suite, []);
    sortSelect = addSelect(fields, "rpent-sort-select", labels.sort, [["desc", labels.desc], ["asc", labels.asc]]);
    // This runs before the shared change handler, so a benchmark change cannot keep a stale suite.
    benchmarkSelect.addEventListener("change", updateSuites, { capture: true });
    updateSuites();
    const configurationFieldset = create("fieldset", "rpent-configuration-filters");
    configurationFieldset.append(create("legend", "", labels.configurations));
    configurationInputs = data.configurations.filter((configuration) => configuration.kind === "rpent").map((configuration) => {
      const label = create("label");
      const input = create("input");
      input.type = "checkbox";
      input.name = "rpent-configuration";
      input.value = configuration.id;
      input.checked = true;
      input.addEventListener("change", () => render());
      const configurationLabel = configuration.effort ? `${configuration.model} · ${configuration.effort}` : configuration.model;
      label.append(input, create("span", "", configurationLabel));
      configurationFieldset.append(label);
      return input;
    });
    controls.append(configurationFieldset);
    const actions = create("div", "rpent-leaderboard-actions");
    const externalLabel = create("label", "rpent-external-filter");
    externalInput = create("input");
    externalInput.type = "checkbox";
    externalInput.id = "rpent-external-methods";
    externalInput.checked = true;
    externalInput.addEventListener("change", () => render());
    externalLabel.append(externalInput, create("span", "", labels.external));
    const resetButton = create("button", "", labels.reset);
    resetButton.type = "button";
    resetButton.id = "rpent-reset";
    resetButton.addEventListener("click", () => {
      benchmarkSelect.value = "overview";
      updateSuites();
      sortSelect.value = "desc";
      externalInput.checked = true;
      configurationInputs.forEach((input) => { input.checked = true; });
      render();
    });
    csvButton = create("button", "", labels.csv);
    csvButton.type = "button";
    csvButton.id = "rpent-export-csv";
    csvButton.addEventListener("click", exportCsv);
    actions.append(externalLabel, resetButton, csvButton);
    controls.append(actions, create("p", "rpent-filter-hint", labels.hint));
  }

  async function initialize() {
    try {
      root.dataset.assetBase = new URL(root.dataset.assetBase, document.baseURI).href;
      const response = await fetch(root.dataset.resultsUrl);
      if (!response.ok) throw new Error(`Data response ${response.status}`);
      data = await response.json();
      if (data.schema_version !== 1) throw new Error("Unsupported results schema");
      configurations = mapById(data.configurations);
      records = mapById(data.results);
      sources = mapById(data.sources);
      protocols = mapById(data.protocols);
      setupControls();
      await render();
      let previousTheme = theme();
      new MutationObserver(() => {
        const currentTheme = theme();
        if (currentTheme !== previousTheme) {
          previousTheme = currentTheme;
          render();
        }
      }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    } catch (error) {
      status.textContent = labels.failed;
      root.dataset.interactive = "failed";
      console.warn("RPent leaderboard:", error);
    }
  }

  window.addEventListener("resize", repositionFocusedTooltip);
  window.addEventListener("scroll", repositionFocusedTooltip, { passive: true, capture: true });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") hideTooltip(); });
  initialize();
})();
