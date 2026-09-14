/* A sortable benchmark table and progressively enhanced, generated SVG charts. */
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
      ranking: "Rankings", charts: "Charts", rank: "Rank", rankBy: "Rank by",
      filters: "Model filters", search: "Search models", simulation: "SIMULATION BENCHMARKS",
      introduction: "Compare models across robot manipulation benchmarks.", view: "View",
      tableHint: "Click a score heading to rank by that evaluation item. Scores are success rates; — means not reported.",
      rankHint: "Ranks use only the selected evaluation item and visible methods. Equal scores share a rank; missing scores remain unranked.",
      modelsShown: "models shown", sortedBy: "Ranked by", externalShort: "External",
      overviewTitle: "Benchmark overview", overviewHint: "Six evaluation settings, with representative reference methods.",
      updated: "Results checked", scroll: "Scroll horizontally to compare all evaluation items →",
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
      ranking: "排行榜", charts: "柱状图", rank: "排名", rankBy: "排名依据",
      filters: "模型筛选", search: "搜索模型", simulation: "仿真基准测试",
      introduction: "按基准查看排名，逐项对比模型成绩。", view: "视图",
      tableHint: "点击成绩列标题，按该评测分项排名。数值为成功率；— 表示未报告。",
      rankHint: "排名仅依据当前分项及可见方法。同分并列，缺失成绩不参与排名。",
      modelsShown: "个模型", sortedBy: "排名依据", externalShort: "外部方法",
      overviewTitle: "基准总览", overviewHint: "六个评测范围，展示 RPent 与代表性参考方法。",
      updated: "成绩核对日期", scroll: "左右滑动，查看全部评测分项 →",
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
  const rankings = create("section", "rpent-rankings");
  rankings.id = "rpent-rankings";
  rankings.hidden = true;
  root.append(rankings, grid, status);

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
  let modeSelect;
  let rankSelect;
  let searchInput;
  let tabs;
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
    rankSelect.replaceChildren();
    const views = availableViews();
    for (const view of views) rankSelect.add(new Option(localized(view.label), view.id));
    rankSelect.value = views.find((view) => view.id === benchmarkSelect.value)?.id
      || views.find((view) => view.aggregate_of)?.id || views[0].id;
  }

  function configurationSelected(configuration) {
    const query = searchInput.value.trim().toLocaleLowerCase();
    if (query && ![configuration.model, configuration.backend, configuration.effort,
      configuration.kind === "rpent" ? "RPent" : labels.externalShort].filter(Boolean).join(" ").toLocaleLowerCase().includes(query)) return false;
    return configuration.kind === "external" ? externalInput.checked
      : configurationInputs.some((input) => input.value === configuration.id && input.checked);
  }

  function selectedRecords(view) {
    const overview = benchmarkSelect.value === "overview";
    const plottedIds = new Set(overview ? view.overview_record_ids : view.chart_record_ids);
    const chosenConfigurations = new Set(configurationInputs.filter((input) => input.checked).map((input) => input.value));
    return view.record_ids.map((id) => records.get(id)).filter((record) => {
      const configuration = configurations.get(record.configuration_id);
      if (!configurationSelected(configuration)) return false;
      if (configuration.kind === "external") return externalInput.checked && plottedIds.has(record.id);
      return chosenConfigurations.has(configuration.id) && (record.status !== "reported" || plottedIds.has(record.id));
    }).sort((left, right) => {
      if (left.status !== right.status) return left.status === "reported" ? -1 : 1;
      const direction = sortSelect.value === "asc" ? 1 : -1;
      return direction * (Number(left.rate) - Number(right.rate)) || left.id.localeCompare(right.id);
    });
  }

  function bindDetails(node, record) {
    node.dataset.recordId = record.id;
    node.setAttribute("aria-label", details(record).map(([label, value]) => `${label}: ${value}`).join(". "));
    node.addEventListener("pointerenter", (event) => showTooltip(node, record, event));
    node.addEventListener("pointermove", (event) => showTooltip(node, record, event));
    node.addEventListener("pointerleave", () => { if (document.activeElement !== node) hideTooltip(); });
    node.addEventListener("focus", () => showTooltip(node, record));
    node.addEventListener("blur", hideTooltip);
    node.addEventListener("click", () => showTooltip(node, record));
  }

  function renderRankings(views) {
    const benchmark = data.benchmarks.find((item) => item.id === benchmarkSelect.value);
    const rankingView = views.find((view) => view.id === rankSelect.value) || views[0];
    rankSelect.value = rankingView.id;
    // Overall is an existing record, never an average of partially reported items.
    const isOverall = (view) => view.id === benchmarkSelect.value || Boolean(view.aggregate_of);
    const orderedViews = [...views].sort((a, b) => Number(isOverall(b)) - Number(isOverall(a)));
    const rows = new Map();
    for (const view of orderedViews) {
      for (const id of view.record_ids) {
        const record = records.get(id);
        const configuration = configurations.get(record.configuration_id);
        if (!configurationSelected(configuration)) continue;
        if (!rows.has(configuration.id)) rows.set(configuration.id, { configuration, values: new Map() });
        rows.get(configuration.id).values.set(view.id, record);
      }
    }
    const score = (row) => row.values.get(rankingView.id)?.status === "reported"
      ? Number(row.values.get(rankingView.id).rate) : null;
    const orderedRows = [...rows.values()].sort((a, b) => {
      const left = score(a), right = score(b);
      if (left === null || right === null) return Number(left === null) - Number(right === null)
        || a.configuration.model.localeCompare(b.configuration.model);
      return (sortSelect.value === "asc" ? 1 : -1) * (left - right)
        || a.configuration.model.localeCompare(b.configuration.model);
    });
    const heading = create("div", "rpent-ranking-heading");
    const headingText = create("div");
    const title = create("h3", "", localized(benchmark.title));
    title.id = "rpent-ranking-title";
    const scope = create("p", "rpent-ranking-scope", localized(rankingView.scope || rankingView.subtitle));
    headingText.append(title, scope);
    heading.append(headingText, create("span", "rpent-result-count", `${orderedRows.length} ${labels.modelsShown}`));
    const wrapper = create("div", "rpent-ranking-scroll");
    wrapper.tabIndex = 0;
    wrapper.setAttribute("role", "region");
    wrapper.setAttribute("aria-labelledby", title.id);
    const table = create("table", "rpent-ranking-table");
    const caption = create("caption", "rpent-visually-hidden", `${localized(benchmark.title)}. ${labels.tableHint}`);
    const head = create("thead");
    const header = create("tr");
    const rankHead = create("th", "rpent-rank-column", labels.rank);
    rankHead.scope = "col";
    const modelHead = create("th", "rpent-model-column", labels.model);
    modelHead.scope = "col";
    header.append(rankHead, modelHead);
    for (const view of orderedViews) {
      const selected = view.id === rankingView.id;
      const cell = create("th", selected ? "rpent-selected-score" : "");
      cell.scope = "col";
      cell.dataset.viewId = view.id;
      cell.setAttribute("aria-sort", selected ? sortSelect.value === "asc" ? "ascending" : "descending" : "none");
      const button = create("button", "rpent-sort-heading");
      button.type = "button";
      button.append(create("span", "", localized(view.label)), create("span", "rpent-sort-arrow", selected ? sortSelect.value === "asc" ? "↑" : "↓" : "↕"));
      button.setAttribute("aria-label", `${labels.rankBy}: ${localized(view.label)}`);
      button.addEventListener("click", () => {
        if (rankSelect.value === view.id) sortSelect.value = sortSelect.value === "desc" ? "asc" : "desc";
        else { rankSelect.value = view.id; sortSelect.value = "desc"; }
        render();
        rankings.querySelector(`th[data-view-id="${view.id}"] button`)?.focus();
      });
      cell.append(button);
      header.append(cell);
    }
    head.append(header);
    const body = create("tbody");
    let previousScore = null, rank = 0;
    for (const [index, row] of orderedRows.entries()) {
      const currentScore = score(row);
      if (currentScore !== null && (index === 0 || currentScore !== previousScore)) rank = index + 1;
      previousScore = currentScore;
      const tr = create("tr", row.configuration.kind === "rpent" ? "rpent-model-row" : "rpent-external-row");
      tr.dataset.configurationId = row.configuration.id;
      const rankCell = create("td", "rpent-rank-column");
      rankCell.append(create("span", "rpent-rank-number", currentScore === null ? "—" : String(rank).padStart(2, "0")));
      if (currentScore === null) rankCell.setAttribute("aria-label", labels.notReported);
      const nameCell = create("th", "rpent-model-column");
      nameCell.scope = "row";
      nameCell.append(create("span", "rpent-model-name", row.configuration.model));
      const metadata = create("span", "rpent-model-meta");
      metadata.append(create("span", "rpent-method-badge", row.configuration.kind === "rpent" ? "RPent" : labels.externalShort));
      if (row.configuration.backend) metadata.append(create("span", "", row.configuration.backend));
      if (row.configuration.effort) metadata.append(create("span", "rpent-effort", row.configuration.effort));
      nameCell.append(metadata);
      tr.append(rankCell, nameCell);
      for (const view of orderedViews) {
        const record = row.values.get(view.id);
        const cell = create("td", view.id === rankingView.id ? "rpent-selected-score" : "");
        cell.dataset.viewId = view.id;
        if (record?.status === "reported") {
          const button = create("button", "rpent-score");
          button.type = "button";
          button.append(create("span", "rpent-score-value", `${record.rate}%`));
          const track = create("span", "rpent-score-track");
          track.setAttribute("aria-hidden", "true");
          const fill = create("span", "rpent-score-fill");
          fill.style.width = `${record.rate}%`;
          track.append(fill);
          button.append(track);
          bindDetails(button, record);
          cell.append(button);
        } else {
          const missing = create("span", "rpent-score-missing", "—");
          missing.setAttribute("aria-label", labels.notReported);
          missing.title = labels.notReported;
          cell.append(missing);
        }
        tr.append(cell);
      }
      body.append(tr);
    }
    if (!orderedRows.length) {
      const row = create("tr"), cell = create("td", "rpent-empty-results", labels.empty);
      cell.colSpan = orderedViews.length + 2;
      row.append(cell); body.append(row);
    }
    table.append(caption, head, body);
    wrapper.append(table);
    const note = create("p", "rpent-ranking-note", labels.tableHint);
    const sourceLine = create("p", "rpent-ranking-sources");
    sourceLine.append(document.createTextNode(`${labels.source}: `));
    visibleRecords = orderedRows.flatMap((row) => orderedViews.map((view) => row.values.get(view.id)).filter(Boolean));
    for (const [index, id] of [...new Set(visibleRecords.flatMap((record) => record.source_ids))].entries()) {
      const source = sources.get(id);
      if (!source) continue;
      if (index) sourceLine.append(document.createTextNode(" · "));
      const link = create("a", "", localized(source.label));
      link.href = source.anchor ? `#${source.anchor}` : source.url;
      sourceLine.append(link);
    }
    rankings.replaceChildren(heading, note, wrapper, create("p", "rpent-scroll-hint", labels.scroll), sourceLine);
    status.textContent = `${labels.sortedBy}: ${localized(rankingView.label)}. ${labels.rankHint}`;
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
    let views = availableViews().filter((view) => suiteSelect.value === "all" || suiteSelect.value === view.id);
    const overview = benchmarkSelect.value === "overview";
    const tableMode = !overview && modeSelect.value === "ranking";
    // A reduced-coverage reference is a separate comparison, not another
    // component of an aggregate. It remains selectable as its own task/suite.
    const aggregate = views.find((view) => view.aggregate_of);
    if (tableMode && suiteSelect.value === "all" && aggregate) {
      views = views.filter((view) => view.id === aggregate.id || aggregate.aggregate_of.includes(view.id));
    }
    root.dataset.mode = tableMode ? "ranking" : "charts";
    for (const tab of tabs.querySelectorAll("button")) tab.setAttribute("aria-pressed", String(tab.dataset.benchmark === benchmarkSelect.value));
    modeSelect.disabled = overview;
    rankSelect.closest("label").hidden = !tableMode;
    for (const option of rankSelect.options) option.disabled = !views.some((view) => view.id === option.value);
    rankings.hidden = true;
    if (tableMode) {
      renderRankings(views);
      grid.hidden = true;
      rankings.hidden = false;
      updateScrollHint();
      fallback.hidden = true;
      controls.hidden = false;
      root.dataset.interactive = "ready";
      grid.removeAttribute("aria-busy");
      csvButton.disabled = false;
      return;
    }
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
    const masthead = create("div", "rpent-leaderboard-masthead");
    const intro = create("div");
    intro.append(create("p", "rpent-leaderboard-eyebrow", labels.simulation), create("p", "rpent-leaderboard-intro", labels.introduction));
    masthead.append(intro, create("span", "rpent-checked-date", `${labels.updated} · ${data.checked_at}`));
    controls.append(masthead);
    tabs = create("div", "rpent-benchmark-tabs");
    tabs.setAttribute("role", "group");
    tabs.setAttribute("aria-label", labels.benchmark);
    controls.append(tabs);
    const fields = create("div", "rpent-leaderboard-fields");
    controls.append(fields);
    benchmarkSelect = addSelect(fields, "rpent-benchmark-select", labels.benchmark, [
      ["overview", labels.overview], ...data.benchmarks.map((benchmark) => [benchmark.id, localized(benchmark.label)]),
    ]);
    benchmarkSelect.closest("label").hidden = true;
    const benchmarkIds = ["libero-pro", "standard-libero", "robocasa", "robotwin", "libero-pro-zero-shot"];
    const orderedBenchmarks = [...data.benchmarks].sort((a, b) => {
      const priority = (id) => benchmarkIds.includes(id) ? benchmarkIds.indexOf(id) : benchmarkIds.length;
      return priority(a.id) - priority(b.id);
    });
    for (const item of [...orderedBenchmarks, { id: "overview", label: labels.overview }]) {
      const tab = create("button", "rpent-benchmark-tab", localized(item.label));
      tab.type = "button";
      tab.dataset.benchmark = item.id;
      tab.addEventListener("click", () => {
        benchmarkSelect.value = item.id;
        updateSuites();
        render();
      });
      tabs.append(tab);
    }
    benchmarkSelect.value = data.benchmarks.some((item) => item.id === "libero-pro") ? "libero-pro" : data.benchmarks[0].id;
    modeSelect = addSelect(fields, "rpent-view-mode", labels.view, [["ranking", labels.ranking], ["charts", labels.charts]]);
    suiteSelect = addSelect(fields, "rpent-suite-select", labels.suite, []);
    rankSelect = addSelect(fields, "rpent-rank-select", labels.rankBy, []);
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
    const filterDetails = create("details", "rpent-model-filters");
    filterDetails.append(create("summary", "", labels.filters));
    const searchLabel = create("label", "rpent-leaderboard-field");
    searchLabel.htmlFor = "rpent-model-search";
    searchLabel.append(create("span", "", labels.search));
    searchInput = create("input");
    searchInput.type = "search";
    searchInput.id = "rpent-model-search";
    searchInput.placeholder = labels.search;
    searchInput.addEventListener("input", () => render());
    searchLabel.append(searchInput);
    filterDetails.append(searchLabel, configurationFieldset);
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
      benchmarkSelect.value = data.benchmarks.some((item) => item.id === "libero-pro") ? "libero-pro" : data.benchmarks[0].id;
      updateSuites();
      modeSelect.value = "ranking";
      searchInput.value = "";
      sortSelect.value = "desc";
      externalInput.checked = true;
      configurationInputs.forEach((input) => { input.checked = true; });
      render();
    });
    csvButton = create("button", "", labels.csv);
    csvButton.type = "button";
    csvButton.id = "rpent-export-csv";
    csvButton.addEventListener("click", exportCsv);
    filterDetails.append(externalLabel);
    actions.append(filterDetails, resetButton, csvButton);
    controls.append(actions);
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
  function updateScrollHint() {
    const wrapper = rankings.querySelector(".rpent-ranking-scroll");
    const hint = rankings.querySelector(".rpent-scroll-hint");
    if (wrapper && hint) hint.hidden = wrapper.scrollWidth <= wrapper.clientWidth + 1;
  }
  window.addEventListener("resize", updateScrollHint);
  window.addEventListener("scroll", repositionFocusedTooltip, { passive: true, capture: true });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") hideTooltip(); });
  initialize();
})();
