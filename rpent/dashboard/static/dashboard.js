import { formatValue as fmtArgs, requestJSON } from "./http.js";
import { createInteractionController } from "./interaction.js";
import { makeAssistantTextElement } from "./markdown_table.js";
import {
  analyzePrimitiveSchema,
  readPrimitiveArguments,
  renderPrimitiveFields,
} from "./primitive_controls.js";

function $(selector) {
  return document.querySelector(selector);
}

const LANGUAGE = document.documentElement.lang === "zh-cn" ? "zh-cn" : "en";

const COPY = {
  en: {
    pageTitle: "RPent · Live Monitor",
    liveMonitor: "Live Monitor",
    planner: "planner",
    model: "model",
    defaultModel: "configured default",
    runtimeStates: {
      pending: "waiting",
      starting: "starting",
      ready: "ready",
      failed: "failed",
    },
    reasoning: "Agent reasoning & tool calls",
    expandTools: "expand tool calls",
    autoScroll: "auto-scroll",
    resizeColumns: "Drag to resize columns",
    composerLabel: "Agent message composer",
    taskSuggestionsLabel: "Task value suggestions",
    resizeComposer: "Drag to resize composer height · double-click to reset",
    composerPlaceholder: "Message the agent…",
    composerKeys: "Enter to send · Shift+Enter for newline · Esc to interrupt",
    commandPlaceholder: (usage) => usage,
    commandKeys: (usage) => `Enter to submit · ${usage}`,
    sessionStarting: "Starting shared robot services…",
    commandReady: (usage) => `Ready for ${usage}.`,
    taskStarting: "Starting the selected TaskRun…",
    taskSwitchPending: (target) => `Task switch pending${target ? `: ${target}` : ""}.`,
    sessionFatal: "The Dashboard Session is unavailable.",
    interactionStarting: "Waiting for robot startup…",
    interactionReady: "The agent is ready for another message.",
    interactionBusy: "The agent is working; new messages will be queued.",
    interactionUnavailable: "The agent is not accepting messages yet.",
    interruptRequested: "Interrupt requested; any active tool will stop at its next safe boundary.",
    interruptSucceeded: "The agent was interrupted; queued messages are being submitted.",
    submittingMessage: "Submitting message…",
    pendingHeading: "Messages to be submitted after next tool call",
    inactiveMessagesHeading: "Messages not submitted",
    messageStates: {
      pending: "pending",
      sending: "sending",
      failed: "failed",
      unsent: "unsent",
    },
    withdraw: "Withdraw",
    withdrawMessage: (text) => `Withdraw queued message: ${text}`,
    submitFailed: (error) => `Message was not submitted: ${error}`,
    withdrawFailed: (error) => `Message was not withdrawn: ${error}`,
    interruptFailed: (error) => `Interrupt request failed: ${error}`,
    interactionError: (error) => `Agent interaction error: ${error}`,
    unknownRequestError: "request failed",
    waitingFrame: "waiting for first frame…",
    frameUnavailable: (label) => `${label} unavailable`,
    resizeFrame: "Drag to resize frame height",
    primitiveControls: "Primitive controls",
    primitiveWaiting: "waiting for TaskRun",
    primitiveReady: (count) => `${count} available`,
    primitiveUnavailable: "controls unavailable",
    primitiveUnsupported: (reason) => `Unavailable: ${reason}`,
    unsupportedSchema: "unsupported schema",
    advanced: "Advanced",
    executePrimitive: "Execute",
    executingPrimitive: "Executing…",
    primitiveSucceeded: "Executed — inspect the camera and timeline.",
    primitiveFailed: (error) => `Execution failed: ${error}`,
    notSet: "not set",
    invalidField: (field) => `${field} must be a valid value.`,
    actionTimeline: "Action timeline",
    noActions: "No actions yet.",
    stateLabels: {
      starting: "starting",
      ready: "ready",
      running: "running",
      succeeded: "succeeded",
      failed: "failed",
      cancelled: "cancelled",
      stale: "stale",
    },
    solved: "TASK SOLVED",
    notSolved: "not solved",
    full: "full",
    episodeVideo: "episode video",
    completeRunVideo: "complete run video",
    finished: "finished",
    noTranscript: "No transcript events yet.",
    you: "You",
    initialTaskSubmitted:
      "Initial task instructions were generated from the run configuration and submitted to the agent automatically.",
    loading: "Loading…",
    live: "● live",
    reconnecting: "○ reconnecting…",
    fieldRequired: (field) => `${field} is required.`,
    awaitingTask: (usage) => `Waiting for ${usage}`,
    actionReplayTitle: "Click to replay this action",
    episodeReplayTitle: "Click to replay the full episode",
    thinking: (count) => `thinking · ${count.toLocaleString()} chars`,
    toolCalls: "tool calls",
    eventCount: (count) => `${count} events`,
    actionCaption: (step, action) => `action #${step} ${action}`,
    frameCaption: (label, index) => `${label} · frame #${index}`,
    usage: (usage) =>
      `token in ${usage.in.toLocaleString()} · out ${usage.out.toLocaleString()} · ${usage.tool_calls} tools`,
  },
  "zh-cn": {
    pageTitle: "RPent · 实时监控",
    liveMonitor: "实时监控",
    planner: "planner",
    model: "model",
    defaultModel: "默认配置",
    runtimeStates: {
      pending: "等待中",
      starting: "启动中",
      ready: "就绪",
      failed: "启动失败",
    },
    reasoning: "智能体推理与工具调用",
    expandTools: "展开工具调用",
    autoScroll: "自动滚动",
    resizeColumns: "拖动调整左右宽度",
    composerLabel: "智能体消息输入区",
    taskSuggestionsLabel: "任务参数候选",
    resizeComposer: "拖动调整输入区高度 · 双击复位",
    composerPlaceholder: "向智能体发送消息…",
    composerKeys: "Enter 发送 · Shift+Enter 换行 · Esc 中断",
    commandPlaceholder: (usage) => usage,
    commandKeys: (usage) => `Enter 提交 · ${usage}`,
    sessionStarting: "正在启动共享环境服务…",
    commandReady: (usage) => `可提交 ${usage}。`,
    taskStarting: "正在启动已选 TaskRun…",
    taskSwitchPending: (target) => `任务切换等待中${target ? `：${target}` : ""}。`,
    sessionFatal: "Dashboard Session 已不可用。",
    interactionStarting: "正在等待环境启动…",
    interactionReady: "智能体已准备好接收新消息。",
    interactionBusy: "智能体正在工作；新消息将进入等待队列。",
    interactionUnavailable: "智能体暂未开始接收消息。",
    interruptRequested: "已请求中断；活动工具将在下一个安全边界停止。",
    interruptSucceeded: "智能体已中断；正在提交排队消息。",
    submittingMessage: "正在提交消息…",
    pendingHeading: "等待下次工具调用后提交的消息",
    inactiveMessagesHeading: "未提交的消息",
    messageStates: {
      pending: "等待中",
      sending: "发送中",
      failed: "发送失败",
      unsent: "未发送",
    },
    withdraw: "撤回",
    withdrawMessage: (text) => `撤回排队消息：${text}`,
    submitFailed: (error) => `消息未提交：${error}`,
    withdrawFailed: (error) => `消息未撤回：${error}`,
    interruptFailed: (error) => `中断请求失败：${error}`,
    interactionError: (error) => `智能体交互错误：${error}`,
    unknownRequestError: "请求失败",
    waitingFrame: "等待第一帧…",
    frameUnavailable: (label) => `${label}画面不可用`,
    resizeFrame: "拖动调整画面高度",
    primitiveControls: "Primitive 控制",
    primitiveWaiting: "等待 TaskRun",
    primitiveReady: (count) => `${count} 个可用`,
    primitiveUnavailable: "控件不可用",
    primitiveUnsupported: (reason) => `不可用：${reason}`,
    unsupportedSchema: "schema 不支持",
    advanced: "高级参数",
    executePrimitive: "执行",
    executingPrimitive: "执行中…",
    primitiveSucceeded: "已执行，请查看相机画面和动作时间线。",
    primitiveFailed: (error) => `执行失败：${error}`,
    notSet: "不设置",
    invalidField: (field) => `${field} 的值无效。`,
    actionTimeline: "动作时间线",
    noActions: "暂无动作。",
    stateLabels: {
      starting: "启动中",
      ready: "就绪",
      running: "运行中",
      succeeded: "执行成功",
      failed: "运行失败",
      cancelled: "已取消",
      stale: "已停止",
    },
    solved: "任务完成",
    notSolved: "未完成",
    full: "全",
    episodeVideo: "完整回放",
    completeRunVideo: "整段运行视频",
    finished: "已完成",
    noTranscript: "暂无推理记录。",
    you: "你",
    initialTaskSubmitted: "已根据当前任务配置，自动向智能体提交初始任务指令。",
    loading: "加载中…",
    live: "● 实时",
    reconnecting: "○ 正在重连…",
    fieldRequired: (field) => `请填写${field}。`,
    awaitingTask: (usage) => `等待 ${usage}`,
    actionReplayTitle: "点击回放该动作",
    episodeReplayTitle: "点击回放完整过程",
    thinking: (count) => `思考 · ${count.toLocaleString()} 字符`,
    toolCalls: "次工具调用",
    eventCount: (count) => `${count} 条事件`,
    actionCaption: (step, action) => `动作 #${step} ${action}`,
    frameCaption: (label, index) => `${label} · 第 ${index} 帧`,
    usage: (usage) =>
      `输入 ${usage.in.toLocaleString()} · 输出 ${usage.out.toLocaleString()} · ${usage.tool_calls} 次工具调用`,
  },
};

const copy = COPY[LANGUAGE];
const RUNTIME_STATES = ["pending", "starting", "ready", "failed"];
let runtimeComponents = [];
let frameChannels = [];
let taskCommandUsage = "";

function applyStaticCopy() {
  for (const element of document.querySelectorAll("[data-i18n]")) {
    element.textContent = copy[element.dataset.i18n];
  }
  for (const element of document.querySelectorAll("[data-i18n-placeholder]")) {
    element.placeholder = copy[element.dataset.i18nPlaceholder];
  }
  for (const element of document.querySelectorAll("[data-i18n-title]")) {
    element.title = copy[element.dataset.i18nTitle];
  }
  for (const element of document.querySelectorAll("[data-i18n-aria-label]")) {
    element.setAttribute("aria-label", copy[element.dataset.i18nAriaLabel]);
  }
}

function frameChannelLabel(kind) {
  return frameChannels.find(channel => channel.name === kind)?.label || kind;
}

function defaultFrameKind() {
  return frameChannels[0].name;
}

function renderFrameTabs() {
  const container = $(".frame-tabs");
  const buttons = frameChannels.map(channel => {
    const button = document.createElement("button");
    button.dataset.kind = channel.name;
    button.textContent = frameChannelLabel(channel.name);
    button.classList.toggle("active", channel.name === mediaState.kind);
    button.addEventListener("click", () => setFrameKind(channel.name));
    return button;
  });
  container.replaceChildren(...buttons);
}

function configureDashboardSpec(spec) {
  runtimeComponents = spec.runtime_components;
  frameChannels = spec.frame_channels;
  taskCommandUsage = spec.task.usage;
  const initialFrameKind = defaultFrameKind();
  mediaState.kind = initialFrameKind;
  mediaState.lastRealtimeKind = initialFrameKind;
  renderFrameTabs();
}

function renderPlannerConfig(config) {
  const planner = config.planner || copy.notSet;
  const model = config.model || copy.defaultModel;
  const element = $("#plannerMeta");
  element.textContent = `${copy.planner} ${planner} · ${copy.model} ${model}`;
  element.title = element.textContent;
}

const runState = {
  eventSource: null,
  lastStepCount: -1,
  lastEventCount: -1,
  taskGeneration: null,
};

function selectFrameTab(kind = null) {
  document.querySelectorAll(".frame-tabs button").forEach(button =>
    button.classList.toggle("active", button.dataset.kind === kind)
  );
}

const transcriptState = {
  shown: 0,
  toolGroup: null,
  inFlight: false,
  refreshAgain: false,
  initialized: false,
  epoch: 0,
};

const timelineState = {
  initialized: false,
  loaded: 0,
  episodeElement: null,
};

const mediaState = {
  kind: null,
  frameIndex: -1,
  frameAvailable: null,
  unavailableKind: null,
  actionVideo: null,
  episodeVideoAvailable: false,
  lastRealtimeKind: null,
  returnTimer: null,
  activeImage: null,
  activeVideo: null,
  swapQueue: [],
  swapInFlight: false,
  generation: 0,
};

const primitiveState = {
  available: false,
  loadedGeneration: null,
  loadingGeneration: null,
  primitives: [],
  selected: null,
  executing: false,
  epoch: 0,
  retryAt: 0,
  statusTimer: null,
};

function setPrimitiveStatus(message = "", kind = "", timeout = 0) {
  if (primitiveState.statusTimer) clearTimeout(primitiveState.statusTimer);
  primitiveState.statusTimer = null;
  const status = $("#primitiveStatus");
  status.textContent = message;
  status.className = `primitive-status${kind ? ` ${kind}` : ""}`;
  if (timeout) {
    primitiveState.statusTimer = setTimeout(() => {
      status.textContent = "";
      status.className = "primitive-status";
      primitiveState.statusTimer = null;
    }, timeout);
  }
}

function resetPrimitivePanel() {
  primitiveState.epoch++;
  primitiveState.available = false;
  primitiveState.loadedGeneration = null;
  primitiveState.loadingGeneration = null;
  primitiveState.primitives = [];
  primitiveState.selected = null;
  primitiveState.executing = false;
  primitiveState.retryAt = 0;
  $("#primitiveButtons").replaceChildren();
  $("#primitiveRequiredFields").replaceChildren();
  $("#primitiveOptionalFields").replaceChildren();
  $("#primitiveAdvanced").hidden = true;
  $("#primitiveAvailability").textContent = copy.primitiveWaiting;
  $("#primitiveAvailability").classList.remove("available");
  $("#executePrimitive").disabled = true;
  $("#executePrimitive").textContent = copy.executePrimitive;
  setPrimitiveStatus();
}

function selectPrimitive(name) {
  if (primitiveState.executing) return;
  const primitive = primitiveState.primitives.find(item => item.name === name);
  if (!primitive?.analysis.supported) return;
  primitiveState.selected = primitive;
  for (const button of $("#primitiveButtons").querySelectorAll("button")) {
    button.setAttribute("aria-selected", String(button.dataset.name === name));
  }
  const required = primitive.analysis.fields.filter(field => field.required);
  const optional = primitive.analysis.fields.filter(field => !field.required);
  renderPrimitiveFields($("#primitiveRequiredFields"), required, copy);
  renderPrimitiveFields($("#primitiveOptionalFields"), optional, copy);
  $("#primitiveAdvanced").hidden = optional.length === 0;
  $("#primitiveAdvanced").open = false;
  $("#executePrimitive").disabled = false;
  setPrimitiveStatus();
}

function renderPrimitiveChoices(primitives) {
  primitiveState.primitives = primitives.map(primitive => ({
    ...primitive,
    analysis: analyzePrimitiveSchema(primitive),
  }));
  const buttons = primitiveState.primitives.map(primitive => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.name = primitive.name;
    button.textContent = primitive.analysis.supported
      ? primitive.name
      : `${primitive.name} · ${copy.unsupportedSchema}`;
    button.disabled = !primitive.analysis.supported;
    button.setAttribute("aria-selected", "false");
    if (primitive.analysis.supported) {
      button.addEventListener("click", () => selectPrimitive(primitive.name));
    } else {
      button.title = copy.primitiveUnsupported(primitive.analysis.reason);
      button.setAttribute("aria-label", `${primitive.name}. ${button.title}`);
    }
    return button;
  });
  $("#primitiveButtons").replaceChildren(...buttons);
  const supported = primitiveState.primitives.filter(item => item.analysis.supported);
  $("#primitiveAvailability").textContent = copy.primitiveReady(supported.length);
  $("#primitiveAvailability").classList.add("available");
  if (supported.length) {
    selectPrimitive(supported[0].name);
  }
}

async function loadPrimitiveSchemas(generation) {
  if (primitiveState.loadingGeneration === generation) return;
  primitiveState.loadingGeneration = generation;
  const epoch = ++primitiveState.epoch;
  try {
    const payload = await requestJSON("/api/session/primitives");
    if (epoch !== primitiveState.epoch || generation !== runState.taskGeneration) return;
    primitiveState.loadedGeneration = generation;
    primitiveState.loadingGeneration = null;
    primitiveState.retryAt = 0;
    renderPrimitiveChoices(Array.isArray(payload.primitives) ? payload.primitives : []);
  } catch (error) {
    if (epoch !== primitiveState.epoch) return;
    primitiveState.loadingGeneration = null;
    primitiveState.retryAt = Date.now() + 2000;
    $("#primitiveAvailability").textContent = copy.primitiveUnavailable;
    $("#primitiveAvailability").classList.remove("available");
    setPrimitiveStatus(error.message, "error", 6000);
  }
}

function syncPrimitiveAvailability(snapshot) {
  const available = Boolean(snapshot.primitives_available);
  if (!available) {
    if (primitiveState.available || primitiveState.loadingGeneration != null) {
      resetPrimitivePanel();
    }
    return;
  }
  primitiveState.available = true;
  if (primitiveState.loadedGeneration !== runState.taskGeneration &&
      Date.now() >= primitiveState.retryAt) {
    loadPrimitiveSchemas(runState.taskGeneration);
  }
}

function setPrimitiveFormDisabled(disabled) {
  primitiveState.executing = disabled;
  for (const control of $("#primitiveForm").querySelectorAll("input, select, button")) {
    control.disabled = disabled;
  }
  for (const button of $("#primitiveButtons").querySelectorAll("button")) {
    const item = primitiveState.primitives.find(primitive => primitive.name === button.dataset.name);
    button.disabled = disabled || !item?.analysis.supported;
  }
}

async function executeSelectedPrimitive(event) {
  event.preventDefault();
  const primitive = primitiveState.selected;
  if (!primitive || primitiveState.executing) return;
  let argumentsObject;
  try {
    argumentsObject = readPrimitiveArguments(
      $("#primitiveForm"),
      primitive.analysis.fields,
      copy,
    );
  } catch (error) {
    setPrimitiveStatus(error.message, "error", 6000);
    return;
  }
  const epoch = primitiveState.epoch;
  setPrimitiveFormDisabled(true);
  $("#executePrimitive").textContent = copy.executingPrimitive;
  setPrimitiveStatus(copy.executingPrimitive);
  try {
    await requestJSON("/api/session/primitive", {
      method: "POST",
      body: {
        name: primitive.name,
        arguments: argumentsObject,
      },
    });
    if (epoch === primitiveState.epoch) {
      setPrimitiveStatus(copy.primitiveSucceeded, "success", 3500);
    }
  } catch (error) {
    if (epoch === primitiveState.epoch) {
      setPrimitiveStatus(copy.primitiveFailed(error.message), "error", 7000);
    }
  } finally {
    if (epoch === primitiveState.epoch) {
      setPrimitiveFormDisabled(false);
      $("#executePrimitive").textContent = copy.executePrimitive;
    }
  }
}

$("#primitiveForm").addEventListener("submit", executeSelectedPrimitive);

const ACTION_RETURN_DELAY_MS = 300;

// --- Double-buffered media swap ------------------------------------------
// Two <img> + two <video> live in the DOM at the same position. Exactly one
// carries the `.visible` class at any moment. When switching to new media
// (new realtime frame URL, new tab, new video), we PRE-LOAD into the OTHER
// buffer and only toggle `.visible` after `load` (img) / `canplay` (video)
// fires. Result: no black flash, because the previously-visible element
// stays painted the whole time the new one is loading.
function imgA() {
  return $("#frame-a");
}

function imgB() {
  return $("#frame-b");
}

function vidA() {
  return $("#video-a");
}

function vidB() {
  return $("#video-b");
}

function cancelActionReturn() {
  if (mediaState.returnTimer) {
    clearTimeout(mediaState.returnTimer);
    mediaState.returnTimer = null;
  }
}

function _bufferPair(kind) {
  return kind === "img" ? [imgA(), imgB()] : [vidA(), vidB()];
}

function _pickTarget(kind, url) {
  const [a, b] = _bufferPair(kind);
  if (a._loadedSrc === url) return a;
  if (b._loadedSrc === url) return b;
  const active = kind === "img" ? mediaState.activeImage : mediaState.activeVideo;
  return a === active ? b : a;
}

function _showBuffer(el) {
  for (const m of document.querySelectorAll(".frame-media.visible")) {
    if (m !== el) m.classList.remove("visible");
  }
  el.classList.add("visible");
  if (el.tagName === "IMG") mediaState.activeImage = el;
  else mediaState.activeVideo = el;
  // Whenever we settle on new visible media, pause the *other* video so it
  // doesn't keep playing audio underneath (matters most on video → img
  // auto-return; browsers hidden via visibility keep playing by default).
  for (const v of [vidA(), vidB()]) {
    if (v !== el && !v.paused) { try { v.pause(); } catch {} }
  }
}

// --- Sequential media swap queue ------------------------------------------
// Every swap runs to completion (image `load` or video `canplay`) before
// the next one starts, so the previously-visible element stays painted
// until the new one is decoded — no black flash.
//
// Realtime frames pushed by SSE while a swap is in flight are queued in
// order, never dropped or interrupted, so switching tabs and rolling
// updates do not cut each other short.
//
// `source: "user"` on a spec (tab click, manual step replay, click on
// episode) drops pending "auto" specs so user actions stay responsive.

function swapMedia(spec) {
  if (spec.source === "user") {
    for (let i = mediaState.swapQueue.length - 1; i >= 0; i--) {
      if (mediaState.swapQueue[i].source !== "user") mediaState.swapQueue.splice(i, 1);
    }
    if (mediaState.swapInFlight) {
      // An auto swap is still loading (finish hasn't run yet, so there's
      // nothing to display yet). Abandon it: bumping `mediaState.generation` makes the
      // in-flight swap's finish + done no-op when they eventually fire,
      // so the click's spec can start immediately without waiting for
      // the abandoned fetch to complete.
      mediaState.generation++;
      mediaState.swapInFlight = false;
    }
  }
  mediaState.swapQueue.push(spec);
  _pumpSwap();
}

function _pumpSwap() {
  if (mediaState.swapInFlight || !mediaState.swapQueue.length) return;
  const spec = mediaState.swapQueue.shift();
  mediaState.swapInFlight = true;
  const gen = ++mediaState.generation;
  _runSwap(spec, gen, () => {
    if (gen !== mediaState.generation) return;   // abandoned — skip queue advancement
    mediaState.swapInFlight = false;
    _pumpSwap();
  });
}

function _runSwap(
  { kind, url, cap, errorCap, onReady, onError },
  gen,
  done,
) {
  const target = _pickTarget(kind, url);
  let finished = false;
  let fallbackTimer = null;

  const finish = (ok) => {
    if (finished) return;
    finished = true;
    if (fallbackTimer) { clearTimeout(fallbackTimer); fallbackTimer = null; }
    if (gen !== mediaState.generation) return;   // abandoned — don't paint or advance
    if (!ok && kind === "img") {
      target.removeAttribute("src");
      for (const media of document.querySelectorAll(".frame-media.visible")) {
        media.classList.remove("visible");
      }
      mediaState.activeImage = null;
      if (errorCap != null) $("#frameCap").textContent = errorCap;
      if (onError) onError(target);
      done();
      return;
    }
    target._loadedSrc = ok ? url : null;
    _showBuffer(target);
    if (cap != null) $("#frameCap").textContent = cap;
    if (onReady) onReady(target);
    done();
  };

  // Fast path: URL already resident on this buffer (e.g. user replays the
  // same action video — no re-fetch, just play from cache).
  if (target._loadedSrc === url) {
    finish(true);
    return;
  }

  _clearMediaListeners(target);
  // Once we start loading a new URL into this buffer, its cached identity is
  // stale — clear it so an abandoned load can't leave `_loadedSrc` pointing
  // at content the buffer no longer actually holds.
  target._loadedSrc = null;
  if (kind === "img") {
    const onload = () => finish(true);
    const onerror = () => finish(false);
    target.addEventListener("load", onload, { once: true });
    target.addEventListener("error", onerror, { once: true });
    target._swapCleanup = () => {
      target.removeEventListener("load", onload);
      target.removeEventListener("error", onerror);
    };
    target.src = url;
  } else {
    // `{ once: true }` on `canplay` is CRITICAL: without it, the event
    // refires every time the video buffers or seeks, and running `onReady`
    // more than once resets `currentTime = 0` in a loop → the video would
    // stall at frame 0 and never actually play.
    const oncan = () => finish(true);
    const onerr = () => finish(false);
    target.addEventListener("canplay", oncan, { once: true });
    target.addEventListener("error", onerr, { once: true });
    target._swapCleanup = () => {
      target.removeEventListener("canplay", oncan);
      target.removeEventListener("error", onerr);
    };
    target.src = url;
    target.load();
  }

  // Fallback: if load/canplay never fires (network stall, missing keyframe),
  // swap anyway after 4s so the queue does not wedge on one bad fetch.
  fallbackTimer = setTimeout(() => finish(true), 4000);
}

function _clearMediaListeners(el) {
  if (el._swapCleanup) { el._swapCleanup(); el._swapCleanup = null; }
}

function resetMediaBuffers() {
  // Full reset for a new TaskRun. Drop the queue, invalidate any in-flight
  // swap via mediaState.generation, and wipe both buffers.
  mediaState.swapQueue.length = 0;
  mediaState.generation++;
  mediaState.swapInFlight = false;
  for (const el of [imgA(), imgB()]) {
    el.classList.remove("visible");
    _clearMediaListeners(el);
    el.removeAttribute("src");
    el._loadedSrc = null;
  }
  for (const v of [vidA(), vidB()]) {
    v.classList.remove("visible");
    _clearMediaListeners(v);
    v.onended = null; v.muted = false;
    try { v.pause(); } catch {}
    v.removeAttribute("src");
    v.load();
    v._loadedSrc = null;
  }
  mediaState.activeImage = null;
  mediaState.activeVideo = null;
}

function resetMediaForRun() {
  cancelActionReturn();
  mediaState.kind = defaultFrameKind();
  mediaState.frameIndex = -1;
  mediaState.frameAvailable = null;
  mediaState.unavailableKind = null;
  mediaState.actionVideo = null;
  mediaState.episodeVideoAvailable = false;
  mediaState.lastRealtimeKind = defaultFrameKind();
  resetMediaBuffers();
}

function resetTranscriptForRun() {
  transcriptState.epoch++;
  transcriptState.shown = 0;
  transcriptState.toolGroup = null;
  transcriptState.refreshAgain = false;
  transcriptState.initialized = false;
}

function resetTimelineForRun() {
  timelineState.initialized = false;
  timelineState.loaded = 0;
  timelineState.episodeElement = null;
}

function resetRenderedTaskProjection() {
  runState.lastStepCount = -1;
  runState.lastEventCount = -1;
  resetTranscriptForRun();
  resetTimelineForRun();
  resetMediaForRun();
  resetPrimitivePanel();
  interactionController.reset();
  $("#transcript").innerHTML = `<div class="empty">${copy.noTranscript}</div>`;
  $("#timeline").innerHTML = `<div class="empty">${copy.noActions}</div>`;
  $("#evCount").textContent = "";
  $("#stepCount").textContent = "";
  $("#usageMeta").textContent = "";
  $("#taskMeta").textContent = copy.awaitingTask(taskCommandUsage);
  $("#frameCap").textContent = copy.waitingFrame;
  setResult(false, null);
  selectFrameTab(defaultFrameKind());
}

function syncTaskGeneration(snapshot) {
  const value = snapshot.task_generation;
  if (runState.taskGeneration == null) {
    runState.taskGeneration = value;
    return "initial";
  }
  if (value < runState.taskGeneration) return "stale";
  if (value === runState.taskGeneration) return "unchanged";
  runState.taskGeneration = value;
  resetRenderedTaskProjection();
  return "changed";
}

function renderTaskMeta(task) {
  const taskMeta = $("#taskMeta");
  if (!task) {
    taskMeta.textContent = copy.awaitingTask(taskCommandUsage);
    return;
  }
  const label = document.createElement("b");
  label.textContent = task.label;
  taskMeta.replaceChildren(label);
}

function timelineDetail(item) {
  const result = item.result;
  if (result && typeof result === "object" && !Array.isArray(result)) {
    const entries = Object.entries(result)
      .filter(([, value]) =>
        value == null || ["string", "number", "boolean"].includes(typeof value)
      )
      .slice(0, 4);
    if (entries.length) return fmtArgs(Object.fromEntries(entries)).slice(0, 100);
  } else if (result != null) {
    return fmtArgs(result).slice(0, 100);
  }
  return fmtArgs(item.args).slice(0, 100);
}

const interactionController = createInteractionController({
  copy,
  select: $,
});

function isRealtimeKind(kind) {
  return frameChannels.some(channel => channel.name === kind);
}

function setBadge(state, error = null) {
  const b = $("#statusBadge");
  b.className = "badge b-" + (state || "stale");
  b.textContent = state ? (copy.stateLabels[state] || state) : "—";
  b.title = error || "";
}

function renderRuntimeStatus(runtime) {
  const container = $("#runtimeStatus");
  if (!container) return;
  if (!runtime || typeof runtime !== "object") {
    container.hidden = true;
    container.replaceChildren();
    return;
  }

  const items = runtimeComponents.map(function (component) {
    const info = runtime[component.name];
    const candidate = typeof info === "string" ? info : info?.status;
    const status = RUNTIME_STATES.includes(candidate) ? candidate : "pending";
    const item = document.createElement("span");
    item.className = `runtime-item runtime-${status}`;
    const label = component.label || component.name;
    item.textContent = `${label} ${copy.runtimeStates[status]}`;
    if (info && typeof info === "object" && info.error) {
      item.title = info.error;
      item.setAttribute("aria-label", `${item.textContent}: ${info.error}`);
    }
    return item;
  });
  container.hidden = false;
  container.replaceChildren(...items);
}

function setResult(terminated, state) {
  const b = $("#resultBadge");
  if (state === "succeeded" || terminated) {
    b.style.display = "";
    b.className = "badge " + (terminated ? "b-ok" : "b-fail");
    b.textContent = terminated ? copy.solved : copy.notSolved;
  } else {
    b.style.display = "none";
  }
}

function renderTimeline(
  tl,
  totalSteps,
  hasEpisodeVideo = mediaState.episodeVideoAvailable,
  { animateNew = false } = {},
) {
  tl = Array.isArray(tl) ? tl : [];
  mediaState.episodeVideoAvailable = !!hasEpisodeVideo;
  const el = $("#timeline");
  const shouldAnimateNew = animateNew && timelineState.initialized;
  const total = totalSteps + (mediaState.episodeVideoAvailable ? 1 : 0);
  $("#stepCount").textContent = total ? total : "";
  if (!total) {
    el.innerHTML = `<div class="empty">${copy.noActions}</div>`;
    timelineState.loaded = 0;
    timelineState.episodeElement = null;
    timelineState.initialized = true;
    return;
  }
  if (timelineState.loaded === 0 && (tl.length || mediaState.episodeVideoAvailable)) {
    el.replaceChildren();
  }
  for (const s of tl) {
    if (s.step === 0 && !s.action) continue;
    const div = document.createElement("div");
    div.className = "step" + (s.terminated ? " term" : "");
    if (s.has_action_video) div.className += " hasclip";
    if (shouldAnimateNew) div.classList.add("entering");
    const res = s.result || {};
    const det = timelineDetail(s);
    div.innerHTML = `<span class="idx">${s.step}</span>
      <div class="body"><span class="act">${s.action ?? "—"}</span>
      <div class="det" title="${fmtArgs(res).replace(/"/g,'&quot;')}">${det}</div></div>
      <span class="el">${s.elapsed_s != null ? s.elapsed_s + "s" : ""}</span>`;
    if (s.has_action_video) {
      div.title = copy.actionReplayTitle;
      div.addEventListener("click", () => playActionVideo(s));
    }
    el.insertBefore(div, timelineState.episodeElement);
  }
  timelineState.loaded = totalSteps;
  if (mediaState.episodeVideoAvailable && !timelineState.episodeElement) {
    const div = document.createElement("div");
    div.className = "step episode";
    if (shouldAnimateNew) div.classList.add("entering");
    div.title = copy.episodeReplayTitle;
    div.innerHTML = `<span class="idx">${copy.full}</span>
      <div class="body"><span class="act">${copy.episodeVideo}</span>
      <div class="det">${copy.completeRunVideo}</div></div>
      <span class="el">${copy.finished}</span>`;
    div.addEventListener("click", playEpisodeVideo);
    el.appendChild(div);
    timelineState.episodeElement = div;
  }
  timelineState.initialized = true;
}


function makeToolEl(ev) {
  const div = document.createElement("div");
  div.className = "ev " + ev.type;
  if (ev.type === "tool_call") {
    div.innerHTML = `→ <span class="tname">${ev.tool}</span> <span class="args"></span>`;
    div.querySelector(".args").textContent = fmtArgs(ev.args);
  } else {
    const isErr = ev.result && ev.result.is_error;
    div.innerHTML = `← <span class="tname ${isErr ? "err" : ""}">${ev.tool}</span> <span class="args"></span>`;
    div.querySelector(".args").textContent = fmtArgs(ev.result);
  }
  return div;
}

function makeThinkingEl(ev) {
  const div = document.createElement("div");
  div.className = "ev thinking";
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  const text = ev.text || "";
  summary.textContent = copy.thinking(text.length);
  const pre = document.createElement("pre");
  pre.textContent = text;
  details.appendChild(summary);
  details.appendChild(pre);
  div.appendChild(details);
  return div;
}

function appendEvents(events, animateNew = false) {
  const box = $("#transcript");
  if (transcriptState.shown === 0) { box.innerHTML = ""; transcriptState.toolGroup = null; }
  for (const ev of events) {
    if (ev.type === "tool_call" || ev.type === "tool_result") {
      // collapse consecutive tool calls/results into one foldable group
      if (!transcriptState.toolGroup) {
        const g = document.createElement("div");
        g.className = "toolgroup";
        g.dataset.toolCalls = "0";
        if (animateNew) g.classList.add("entering");
        g.innerHTML = `<div class="tg-head"><span class="tg-count">0</span> ${copy.toolCalls}</div><div class="tg-body"></div>`;
        g.querySelector(".tg-head").addEventListener("click", () => g.classList.toggle("open"));
        box.appendChild(g);
        transcriptState.toolGroup = g;
      }
      transcriptState.toolGroup.querySelector(".tg-body").appendChild(makeToolEl(ev));
      if (ev.type === "tool_call") {
        const n = Number(transcriptState.toolGroup.dataset.toolCalls) + 1;
        transcriptState.toolGroup.dataset.toolCalls = String(n);
        transcriptState.toolGroup.querySelector(".tg-count").textContent = n;
      }
    } else {
      transcriptState.toolGroup = null;  // close the group; turn/text render at top level
      if (ev.type === "thinking") {
        const thinking = makeThinkingEl(ev);
        if (animateNew) thinking.classList.add("entering");
        box.appendChild(thinking);
      } else if (ev.type === "text") {
        const text = makeAssistantTextElement(ev.text);
        if (animateNew) text.classList.add("entering");
        box.appendChild(text);
      } else {
        const div = document.createElement("div");
        div.className = "ev " + ev.type;
        if (animateNew) div.classList.add("entering");
        if (ev.type === "initial_prompt") {
          div.classList.add("meta");
          div.textContent = copy.initialTaskSubmitted;
        } else if (ev.type === "meta") div.textContent = `[${ev.tag}] ${ev.text}`;
        else {
          div.textContent = ev.text;
          if (ev.type === "user") div.dataset.roleLabel = copy.you;
        }
        box.appendChild(div);
      }
    }
  }
  transcriptState.shown += events.length;
  $("#evCount").textContent = transcriptState.shown
    ? copy.eventCount(transcriptState.shown)
    : "";
  if ($("#autoscroll").checked) box.scrollTop = box.scrollHeight;
}

async function refreshTranscript() {
  // Serialize: only one fetch in flight at a time. Concurrent triggers
  // (task reset + SSE ticks) would otherwise read the same `transcriptState.shown`, fetch
  // overlapping chunks, and append in nondeterministic resolution order —
  // which is what made turns show up out of order / duplicated.
  if (transcriptState.inFlight) { transcriptState.refreshAgain = true; return; }
  transcriptState.inFlight = true;
  const taskGeneration = runState.taskGeneration;
  const epoch = transcriptState.epoch;
  try {
    const r = await requestJSON(
      `/api/session/transcript?since=${transcriptState.shown}`
    );
    if (
      taskGeneration !== runState.taskGeneration
      || epoch !== transcriptState.epoch
    ) {
      transcriptState.refreshAgain = true;
      return;
    }
    const animateNew = transcriptState.initialized;
    if (r.events && r.events.length) appendEvents(r.events, animateNew);
    else if (transcriptState.shown === 0) {
      $("#transcript").innerHTML = `<div class="empty">${copy.noTranscript}</div>`;
    }
    transcriptState.initialized = true;
  } catch (e) {
    /* transient — next tick retries */
  } finally {
    transcriptState.inFlight = false;
    if (transcriptState.refreshAgain) { transcriptState.refreshAgain = false; refreshTranscript(); }  // coalesced re-run
  }
}

function setFrameKind(kind) {
  cancelActionReturn();
  if (isRealtimeKind(kind)) {
    mediaState.lastRealtimeKind = kind;
  }
  mediaState.kind = kind;
  if (kind !== "actionVideo") mediaState.actionVideo = null;
  selectFrameTab(kind);
  mediaState.frameIndex = -1;
  refreshFrame(undefined, { source: "user" });
}

function finishActionPlayback(actionVideo) {
  if (mediaState.actionVideo !== actionVideo || mediaState.returnTimer) return;
  mediaState.returnTimer = setTimeout(() => {
    mediaState.returnTimer = null;
    if (mediaState.actionVideo !== actionVideo) return;
    const returnKind = mediaState.lastRealtimeKind || defaultFrameKind();
    mediaState.actionVideo = null;
    mediaState.kind = returnKind;
    mediaState.frameIndex = -1;
    selectFrameTab(returnKind);
    // No video reset here — swapMedia keeps the finished video's last
    // frame painted until the realtime PNG is decoded, then flips visibility
    // — the transition never exposes the black framewrap background.
    refreshMeta();
  }, ACTION_RETURN_DELAY_MS);
}

function playActionVideo(step) {
  if (!step || !step.has_action_video) return;
  cancelActionReturn();
  mediaState.actionVideo = step;
  mediaState.kind = "actionVideo";
  selectFrameTab();
  mediaState.frameIndex = -1;
  refreshFrame(undefined, { source: "user" });
}

function playEpisodeVideo() {
  if (!mediaState.episodeVideoAvailable) return;
  cancelActionReturn();
  mediaState.actionVideo = null;
  mediaState.kind = "video";
  selectFrameTab();
  mediaState.frameIndex = -1;
  refreshFrame(undefined, { source: "user" });
}

function showFrameUnavailable(kind, idx) {
  if (mediaState.unavailableKind === kind && idx === mediaState.frameIndex) return;
  mediaState.frameIndex = idx ?? mediaState.frameIndex;
  mediaState.unavailableKind = kind;
  resetMediaBuffers();
  $("#frameCap").textContent = copy.frameUnavailable(frameChannelLabel(kind));
}

function refreshFrame(idx, opts = {}) {
  const source = opts.source || "auto";

  if (mediaState.kind === "actionVideo") {
    if (!mediaState.actionVideo) return;
    const actionVideo = mediaState.actionVideo;
    // Note: no ``t=Date.now()`` cache-buster — action video files are
    // written once and never mutate, so the buffer's ``_loadedSrc`` cache
    // gives us instant replay when the same clip is re-clicked.
    const url = `/api/session/action-video?step=${encodeURIComponent(mediaState.actionVideo.step)}`;
    const cap = copy.actionCaption(
      mediaState.actionVideo.step,
      mediaState.actionVideo.action,
    );
    swapMedia({
      kind: "video",
      url,
      cap,
      source,
      onReady: (v) => {
        try { v.currentTime = 0; } catch {}
        v.playbackRate = 0.5;
        v.muted = false;
        v.onended = () => finishActionPlayback(actionVideo);
        const p = v.play();
        if (p && typeof p.catch === "function") {
          p.catch(() => finishActionPlayback(actionVideo));
        }
      },
    });
    return;
  }

  if (mediaState.kind === "video") {
    const url = "/api/session/video";
    swapMedia({
      kind: "video",
      url,
      cap: copy.episodeVideo,
      source,
      onReady: (v) => {
        v.playbackRate = 1.0;
        v.muted = false;
        v.onended = null;
      },
    });
    return;
  }

  if (mediaState.frameAvailable?.[mediaState.kind] === false) {
    showFrameUnavailable(mediaState.kind, idx);
    return;
  }

  // Realtime camera / wrist frame — PNG mutates server-side, so
  // ``t=Date.now()`` keeps the URL unique per tick and defeats caching.
  if (idx != null && idx === mediaState.frameIndex) return;
  mediaState.frameIndex = idx ?? mediaState.frameIndex;
  mediaState.unavailableKind = null;
  const url = `/api/session/frame?kind=${mediaState.kind}&t=${Date.now()}`;
  swapMedia({
    kind: "img",
    url,
    cap: copy.frameCaption(
      frameChannelLabel(mediaState.kind),
      mediaState.frameIndex,
    ),
    errorCap: copy.frameUnavailable(frameChannelLabel(mediaState.kind)),
    source,
    onReady: () => { mediaState.unavailableKind = null; },
    onError: () => { mediaState.unavailableKind = mediaState.kind; },
  });
}

function applySessionSnapshot(snapshot) {
  const generationState = syncTaskGeneration(snapshot);
  if (generationState === "stale") return;
  setBadge(snapshot.state, snapshot.control_error || snapshot.error);
  setResult(snapshot.terminated, snapshot.state);
  renderRuntimeStatus(snapshot.runtime);
  syncPrimitiveAvailability(snapshot);
  interactionController.applySnapshot(snapshot);
  mediaState.frameAvailable = snapshot.frame_available || null;
  if (snapshot.usage) $("#usageMeta").textContent = copy.usage(snapshot.usage);
  refreshTranscriptIfChanged(snapshot);
  return generationState;
}

async function refreshMeta() {
  const timelineSince = timelineState.loaded;
  const r = await requestJSON(
    `/api/session/state?timeline_since=${timelineSince}`,
  );
  const generationState = applySessionSnapshot(r);
  if (generationState == null) return;
  const currentTask = r.current_task;
  renderTaskMeta(currentTask);
  const timelineCurrent = timelineSince === timelineState.loaded;
  if (timelineCurrent) {
    renderTimeline(r.timeline || [], r.n_steps, r.has_video, {
      animateNew: timelineState.initialized,
    });
    runState.lastStepCount = r.n_steps;
  }
  if (!r.has_video && mediaState.kind === "video") setFrameKind(defaultFrameKind());
  if (
    isRealtimeKind(mediaState.kind)
    && currentTask
  ) refreshFrame(r.frame_idx);
}

function refreshTranscriptIfChanged(snapshot) {
  if (snapshot.n_events == null || snapshot.n_events === runState.lastEventCount) return;
  runState.lastEventCount = snapshot.n_events;
  refreshTranscript();
}

function connectSSE() {
  if (runState.eventSource) runState.eventSource.close();
  runState.eventSource = new EventSource("/api/session/stream");
  runState.eventSource.onmessage = (e) => {
    const sig = JSON.parse(e.data);
    const generationState = applySessionSnapshot(sig);
    if (generationState == null) return;
    $("#connMeta").textContent = copy.live;
    if (generationState === "changed") {
      refreshMeta();
      return;
    }
    // refresh timeline lazily on step change
    if (sig.n_steps !== runState.lastStepCount) {
      runState.lastStepCount = sig.n_steps;
      refreshMeta();
      return;
    }
    if (sig.has_video && !mediaState.episodeVideoAvailable) refreshMeta();
    if (
      isRealtimeKind(mediaState.kind)
      && sig.frame_idx != null
      && sig.frame_idx !== mediaState.frameIndex
    ) refreshFrame(sig.frame_idx);
  };
  runState.eventSource.onerror = () => {
    $("#connMeta").textContent = copy.reconnecting;
  };
}

function connectSession() {
  runState.taskGeneration = null;
  resetRenderedTaskProjection();
  renderRuntimeStatus(null);
  $("#transcript").innerHTML = `<div class="empty">${copy.loading}</div>`;
  $("#timeline").innerHTML = '<div class="empty">…</div>';
  connectSSE();
}

$("#showtools").addEventListener("change", (e) => {
  $("#transcript").classList.toggle("alltools", e.target.checked);
});
// --- draggable splitters ---
function setupSplitter(handle, opts) {
  // opts: { axis:'x'|'y', container, prop, min, max, store, fromEnd }
  const saved = localStorage.getItem(opts.store);
  if (saved) opts.container.style.setProperty(opts.prop, saved);
  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    handle.classList.add("dragging");
    document.body.classList.add("resizing");
    document.body.style.cursor = opts.axis === "x" ? "col-resize" : "row-resize";
    const rect = opts.container.getBoundingClientRect();
    const move = (ev) => {
      let v;
      if (opts.axis === "x") {
        v = opts.fromEnd ? rect.right - ev.clientX : ev.clientX - rect.left;
      } else {
        v = opts.fromEnd ? rect.bottom - ev.clientY : ev.clientY - rect.top;
      }
      const limit = opts.axis === "x" ? rect.width : rect.height;
      v = Math.max(opts.min, Math.min(v, limit - opts.max));
      const px = v + "px";
      opts.container.style.setProperty(opts.prop, px);
      localStorage.setItem(opts.store, px);
    };
    const up = () => {
      handle.classList.remove("dragging");
      document.body.classList.remove("resizing");
      document.body.style.cursor = "";
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  });
  // double-click to reset to default
  handle.addEventListener("dblclick", () => {
    opts.container.style.removeProperty(opts.prop);
    localStorage.removeItem(opts.store);
  });
}
setupSplitter($("#gutterV"), {
  axis: "x", container: $("main"), prop: "--leftw",
  min: 280, max: 320, store: "wm.leftw",
});
setupSplitter($("#gutterH"), {
  axis: "y", container: $(".col.right"), prop: "--frameh",
  min: 120, max: 160, store: "wm.frameh",
});
setupSplitter($("#composerGrip"), {
  axis: "y", fromEnd: true, container: $(".col.left"), prop: "--composerh",
  min: 132, max: 160, store: "wm.composerh",
});

async function boot() {
  applyStaticCopy();
  try {
    const [dashboardSpec, sessionConfig] = await Promise.all([
      requestJSON("/api/commands"),
      requestJSON("/api/session/config"),
    ]);
    configureDashboardSpec(dashboardSpec);
    renderPlannerConfig(sessionConfig);
    interactionController.configureTaskCommand(dashboardSpec.task);
    connectSession();
  } catch (error) {
    console.error("Failed to load Dashboard configuration", error);
  }
}

boot();
