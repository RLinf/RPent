import { createInteractionController } from "./interaction.js";
import { makeAssistantTextElement } from "./markdown_table.js";

function $(selector) {
  return document.querySelector(selector);
}

const LANGUAGE = document.documentElement.lang === "zh-cn" ? "zh-cn" : "en";

const COPY = {
  en: {
    pageTitle: "RPent · Live Monitor",
    newSession: "New Session",
    launcherSubtitle: "Review the session config, then start the Dashboard control Session.",
    planner: "Planner (LLM backend)",
    maxTurns: "Max turns",
    maxEpisodeSteps: "Max episode steps",
    modelPreset: "Model",
    customModel: "Custom model",
    optional: "(optional)",
    required: "(required)",
    customModelPlaceholder: "provider:model or alias",
    noImages: "Disable image input (required for text-only models)",
    claudeBudget: "Claude Code budget USD",
    plannerReasoningEffort: "Reasoning effort (higher may improve success rate)",
    plannerTimeout: "Planner timeout s",
    cudaDevice: "CUDA device",
    blankDefault: "(blank = default)",
    defaultPlaceholder: "default",
    startSession: "Start Session",
    liveMonitor: "Live Monitor",
    runtimeLabels: { env: "ENV", vla: "VLA", sam3: "SAM3" },
    runtimeStates: {
      pending: "waiting",
      starting: "starting",
      ready: "ready",
      failed: "failed",
    },
    reasoning: "Agent reasoning & tool calls",
    expandTools: "expand tool calls",
    autoScroll: "auto-scroll",
    selectRun: "Select a run to begin.",
    resizeColumns: "Drag to resize columns",
    composerLabel: "Agent message composer",
    suiteSuggestionsLabel: "Task value suggestions",
    resizeComposer: "Drag to resize composer height · double-click to reset",
    composerPlaceholder: "Message the agent…",
    composerKeys: "Enter to send · Shift+Enter for newline · Esc to interrupt",
    commandPlaceholder: (usage) => usage,
    commandKeys: (usage) => `Enter to submit · ${usage}`,
    sessionStarting: "Starting shared robot services…",
    commandReady: (usage) => `Ready for ${usage}.`,
    taskStarting: "Starting the selected TaskRun…",
    taskSwitchPending: (target) => `Task switch pending${target ? `: ${target}` : ""}.`,
    taskSelectedFeedback: (target) => `Task selected: ${target}`,
    taskRunStartingFeedback: (number) => `TaskRun ${number} starting…`,
    sessionFatal: "The Dashboard Session is unavailable.",
    dashboardConfigFailed: "Dashboard configuration is unavailable.",
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
    requiredFields: {
      maxTurns: "Max turns",
      maxEpisodeSteps: "Max episode steps",
      apiModel: "API model (provider:model)",
    },
    fieldRequired: (field) => `${field} is required.`,
    starting: "starting Session… this page will switch to the live monitor.",
    startFailed: "failed to start Session — check the terminal.",
    noRuns: (directory) => `(no runs in ${directory})`,
    awaitingTask: (usage) => `Waiting for ${usage}`,
    distance: (value) => `dist ${value}m `,
    steps: (used, maximum) => `${used}/${maximum} steps `,
    lifted: (value) => `lifted=${value} `,
    chunks: (value) => `chunks=${value} `,
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
    newSession: "新建 Session",
    launcherSubtitle: "确认 Session 配置后，启动 Dashboard 控制 Session。",
    planner: "决策大脑(大模型后端)",
    maxTurns: "最大对话轮数",
    maxEpisodeSteps: "最大仿真步数",
    modelPreset: "模型",
    customModel: "自定义模型",
    optional: "(可选)",
    required: "(必填)",
    customModelPlaceholder: "provider:model 或别名",
    noImages: "禁用图像输入（纯文本模型必需）",
    claudeBudget: "Claude Code 预算 USD",
    plannerReasoningEffort: "推理强度（提高强度可能提升成功率）",
    plannerTimeout: "Planner 超时秒数",
    cudaDevice: "CUDA 设备",
    blankDefault: "(留空=默认)",
    defaultPlaceholder: "默认",
    startSession: "启动 Session",
    liveMonitor: "实时监控",
    runtimeLabels: { env: "ENV", vla: "VLA", sam3: "SAM3" },
    runtimeStates: {
      pending: "等待中",
      starting: "启动中",
      ready: "就绪",
      failed: "启动失败",
    },
    reasoning: "智能体推理与工具调用",
    expandTools: "展开工具调用",
    autoScroll: "自动滚动",
    selectRun: "请选择一个运行以开始。",
    resizeColumns: "拖动调整左右宽度",
    composerLabel: "智能体消息输入区",
    suiteSuggestionsLabel: "任务参数候选",
    resizeComposer: "拖动调整输入区高度 · 双击复位",
    composerPlaceholder: "向智能体发送消息…",
    composerKeys: "Enter 发送 · Shift+Enter 换行 · Esc 中断",
    commandPlaceholder: (usage) => usage,
    commandKeys: (usage) => `Enter 提交 · ${usage}`,
    sessionStarting: "正在启动共享环境服务…",
    commandReady: (usage) => `可提交 ${usage}。`,
    taskStarting: "正在启动已选 TaskRun…",
    taskSwitchPending: (target) => `任务切换等待中${target ? `：${target}` : ""}。`,
    taskSelectedFeedback: (target) => `已选择任务：${target
      .replace(/\/ task /g, "/ 任务 ")
      .replace(/\/ seed /g, "/ 种子 ")}`,
    taskRunStartingFeedback: (number) => `任务运行 ${number} 正在启动…`,
    sessionFatal: "Dashboard Session 已不可用。",
    dashboardConfigFailed: "Dashboard 配置不可用。",
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
    requiredFields: {
      maxTurns: "最大对话轮数",
      maxEpisodeSteps: "最大仿真步数",
      apiModel: "API 模型（provider:model）",
    },
    fieldRequired: (field) => `请填写${field}。`,
    starting: "正在启动 Session… 页面将切换到实时监控。",
    startFailed: "Session 启动失败，请查看终端输出。",
    noRuns: (directory) => `(${directory} 中暂无运行)`,
    awaitingTask: (usage) => `等待 ${usage}`,
    distance: (value) => `距离 ${value}m `,
    steps: (used, maximum) => `${used}/${maximum} 步 `,
    lifted: (value) => `已抓取=${value} `,
    chunks: (value) => `块数=${value} `,
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

const runState = {
  id: null,
  eventSource: null,
  lastStepCount: -1,
  taskGeneration: null,
};

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
  seen: new Set(),
};

const mediaState = {
  kind: null,
  frameIndex: -1,
  frameAvailable: null,
  unavailableKind: null,
  actionVideo: null,
  episodeVideoAvailable: false,
  lastRealtimeKind: null,
  lastActionStep: 0,
  autoActionPrimed: false,
  autoPlayback: null,
  returnTimer: null,
  stepTransitioning: false,
  activeImage: null,
  activeVideo: null,
  swapQueue: [],
  swapInFlight: false,
  releaseHold: null,
  generation: 0,
};

const AUTO_ACTION_RETURN_DELAY_MS = 300;
const AUTO_PLAY_ACTION_VIDEOS = false;
const MODEL_PRESETS = {
  claude_code: [
    "deepseek-v4-flash",
    "kimi-k3",
    "claude-opus-4-7",
    "sonnet",
    "opus",
  ],
  codex: ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.3-codex"],
  api: [
    "openai-chat:deepseek-v4-flash",
    "openai-chat:deepseek-v4-pro",
    "openai-chat:kimi-k3",
    "anthropic:claude-opus-4-7",
    "openai:gpt-5.6-sol",
    "openai:gpt-5.6-terra",
    "openai:gpt-5.6-luna",
    "openai-chat:glm-5.2",
  ],
};
const launcherModelSelections = Object.fromEntries(
  Object.entries(MODEL_PRESETS).map(([planner, models]) => [planner, models[0]]),
);
let activeLauncherPlanner = "claude_code";

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

function cancelAutoActionReturn() {
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
// Videos may hold the queue until they finish playing (`holdUntilEnded`),
// preventing realtime frame updates from cutting a video short.
// `source: "user"` on a spec (tab click, manual step replay, click on
// episode) drops pending "auto" specs and releases any current hold so
// user actions stay responsive — the currently-visible element is not
// interrupted, but its hold is released as soon as the click arrives.

function swapMedia(spec) {
  if (spec.source === "user") {
    for (let i = mediaState.swapQueue.length - 1; i >= 0; i--) {
      if (mediaState.swapQueue[i].source !== "user") mediaState.swapQueue.splice(i, 1);
    }
    if (mediaState.releaseHold) {
      // Currently held on a video's end — release it so the click
      // doesn't wait through the rest of the clip. Call `mediaState.releaseHold`
      // DIRECTLY (don't null it before the call): `release` itself
      // guards against double-release by checking `mediaState.releaseHold !== release`
      // and nulls the global on entry. If we nulled it here first, that
      // guard would trip on the very first invocation and `done()` would
      // never fire → the queue would wedge with mediaState.swapInFlight stuck true
      // and no further click would take effect.
      mediaState.releaseHold();
    } else if (mediaState.swapInFlight) {
      // An auto swap is still loading (finish hasn't run yet, so there's
      // no hold to release). Abandon it: bumping `mediaState.generation` makes the
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
  { kind, url, cap, errorCap, onReady, onError, holdUntilEnded },
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
    if (holdUntilEnded && kind === "video") {
      // Realtime frames wait on this video until it ends (natural or errored)
      // — or until a user action releases the hold via `swapMedia`.
      const release = () => {
        if (mediaState.releaseHold !== release) return;
        mediaState.releaseHold = null;
        target.removeEventListener("ended", release);
        target.removeEventListener("error", release);
        done();
      };
      mediaState.releaseHold = release;
      target.addEventListener("ended", release, { once: true });
      target.addEventListener("error", release, { once: true });
    } else {
      done();
    }
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
  // Full reset — only used when selecting a new run. Drops the queue,
  // invalidates any in-flight swap (via mediaState.generation), releases any
  // pending hold, and wipes both buffers.
  mediaState.swapQueue.length = 0;
  mediaState.generation++;
  if (mediaState.releaseHold) mediaState.releaseHold();
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
  cancelAutoActionReturn();
  mediaState.kind = defaultFrameKind();
  mediaState.frameIndex = -1;
  mediaState.frameAvailable = null;
  mediaState.unavailableKind = null;
  mediaState.actionVideo = null;
  mediaState.episodeVideoAvailable = false;
  mediaState.lastRealtimeKind = defaultFrameKind();
  mediaState.lastActionStep = 0;
  mediaState.autoActionPrimed = false;
  mediaState.autoPlayback = null;
  mediaState.stepTransitioning = false;
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
  timelineState.seen.clear();
}

function resetRenderedTaskProjection() {
  runState.lastStepCount = -1;
  resetTranscriptForRun();
  resetTimelineForRun();
  resetMediaForRun();
  interactionController.reset();
  $("#transcript").innerHTML = `<div class="empty">${copy.noTranscript}</div>`;
  $("#timeline").innerHTML = `<div class="empty">${copy.noActions}</div>`;
  $("#evCount").textContent = "";
  $("#stepCount").textContent = "";
  $("#usageMeta").textContent = "";
  $("#taskMeta").textContent = copy.awaitingTask(taskCommandUsage);
  $("#frameCap").textContent = copy.waitingFrame;
  setResult(false, null);
  document.querySelectorAll(".frame-tabs button").forEach(button =>
    button.classList.toggle("active", button.dataset.kind === defaultFrameKind())
  );
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
  label.textContent = task.label || fmtArgs(task.parameters || task);
  taskMeta.replaceChildren(label);
}

function fmtArgs(o) {
  if (o == null) return "";
  if (typeof o === "string") return o;
  try { return JSON.stringify(o); } catch { return String(o); }
}

const interactionController = createInteractionController({
  copy,
  select: $,
  onRefresh: () => {
    refreshMeta().catch(() => {});
  },
});

async function loadRun() {
  const r = await fetch("/api/runs").then(x => x.json());
  if (!r.runs.length) {
    $("#taskMeta").textContent = copy.noRuns(r.runs_dir);
    return;
  }

  const run = r.runs[0];
  selectRun(run.id);
}

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
    const label = copy.runtimeLabels[component.name] || component.label || component.name;
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

function maxTimelineStep(tl) {
  if (!Array.isArray(tl) || !tl.length) return 0;
  return tl.reduce((m, s) => Math.max(m, Number(s.step) || 0), 0);
}

function maybeAutoPlayNewAction(tl, nextFrameIdx) {
  if (!Array.isArray(tl)) return false;
  const candidates = tl.filter(s =>
    s.has_action_video && (Number(s.step) || 0) > mediaState.lastActionStep);
  mediaState.lastActionStep = Math.max(mediaState.lastActionStep, maxTimelineStep(tl));
  const step = candidates[candidates.length - 1];
  if (!step || !isRealtimeKind(mediaState.kind)) return false;
  return playActionVideo(step, {
    auto: true,
    nextFrameIdx,
    returnKind: mediaState.kind,
  });
}

function timelineItemKey(item) {
  return `${item.step ?? ""}:${item.action ?? ""}`;
}

function renderTimeline(
  tl,
  hasEpisodeVideo = mediaState.episodeVideoAvailable,
  { animateNew = false } = {},
) {
  tl = Array.isArray(tl) ? tl : [];
  mediaState.episodeVideoAvailable = !!hasEpisodeVideo;
  const el = $("#timeline");
  const shouldAnimateNew = animateNew && timelineState.initialized;
  const total = tl.length + (mediaState.episodeVideoAvailable ? 1 : 0);
  $("#stepCount").textContent = total ? total : "";
  if (!tl.length && !mediaState.episodeVideoAvailable) {
    el.innerHTML = `<div class="empty">${copy.noActions}</div>`;
    timelineState.initialized = true;
    return;
  }
  el.innerHTML = "";
  for (const s of tl) {
    if (s.step === 0 && !s.action) continue;
    const key = timelineItemKey(s);
    const div = document.createElement("div");
    div.className = "step" + (s.terminated ? " term" : "");
    if (s.has_action_video) div.className += " hasclip";
    if (shouldAnimateNew && !timelineState.seen.has(key)) {
      div.classList.add("entering");
    }
    const res = s.result || {};
    let det = "";
    if (res.final_dist_m != null) det += copy.distance((+res.final_dist_m).toFixed(3));
    if (res.steps_used != null) det += copy.steps(res.steps_used, res.max_steps ?? "?");
    if (res.lifted != null) det += copy.lifted(res.lifted);
    if (res.chunks != null) det += copy.chunks(res.chunks);
    det = det.trim() || fmtArgs(s.args).slice(0, 60);
    div.innerHTML = `<span class="idx">${s.step}</span>
      <div class="body"><span class="act">${s.action ?? "—"}</span>
      <div class="det" title="${fmtArgs(res).replace(/"/g,'&quot;')}">${det}</div></div>
      <span class="el">${s.elapsed_s != null ? s.elapsed_s + "s" : ""}</span>`;
    if (s.has_action_video) {
      div.title = copy.actionReplayTitle;
      div.addEventListener("click", () => playActionVideo(s, {
        returnAfterEnd: true,
        returnKind: mediaState.lastRealtimeKind,
      }));
    }
    el.appendChild(div);
    timelineState.seen.add(key);
  }
  if (mediaState.episodeVideoAvailable) {
    const div = document.createElement("div");
    div.className = "step episode";
    if (shouldAnimateNew && !timelineState.seen.has("episode-video")) {
      div.classList.add("entering");
    }
    div.title = copy.episodeReplayTitle;
    div.innerHTML = `<span class="idx">${copy.full}</span>
      <div class="body"><span class="act">${copy.episodeVideo}</span>
      <div class="det">${copy.completeRunVideo}</div></div>
      <span class="el">${copy.finished}</span>`;
    div.addEventListener("click", playEpisodeVideo);
    el.appendChild(div);
    timelineState.seen.add("episode-video");
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
        if (animateNew) g.classList.add("entering");
        g.innerHTML = `<div class="tg-head"><span class="tg-count">0</span> ${copy.toolCalls}</div><div class="tg-body"></div>`;
        g.querySelector(".tg-head").addEventListener("click", () => g.classList.toggle("open"));
        box.appendChild(g);
        transcriptState.toolGroup = g;
      }
      transcriptState.toolGroup.querySelector(".tg-body").appendChild(makeToolEl(ev));
      const n = transcriptState.toolGroup.querySelectorAll(".tg-body .ev.tool_call").length;
      transcriptState.toolGroup.querySelector(".tg-count").textContent = n;
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
  if (!runState.id) return;
  // Serialize: only one fetch in flight at a time. Concurrent triggers
  // (selectRun + SSE ticks) would otherwise read the same `transcriptState.shown`, fetch
  // overlapping chunks, and append in nondeterministic resolution order —
  // which is what made turns show up out of order / duplicated.
  if (transcriptState.inFlight) { transcriptState.refreshAgain = true; return; }
  transcriptState.inFlight = true;
  const run = runState.id;
  const taskGeneration = runState.taskGeneration;
  const epoch = transcriptState.epoch;
  try {
    const r = await fetch(
      `/api/run/transcript?run=${encodeURIComponent(run)}&since=${transcriptState.shown}`
    ).then(x => x.json());
    if (
      run !== runState.id
      || taskGeneration !== runState.taskGeneration
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
  cancelAutoActionReturn();
  if (isRealtimeKind(kind)) {
    mediaState.lastRealtimeKind = kind;
    mediaState.autoPlayback = null;
  }
  mediaState.kind = kind;
  if (kind !== "actionVideo") mediaState.actionVideo = null;
  document.querySelectorAll(".frame-tabs button").forEach(b =>
    b.classList.toggle("active", b.dataset.kind === kind));
  mediaState.frameIndex = -1;
  refreshFrame(undefined, { source: "user" });
}

function finishAutoActionPlayback() {
  if (!mediaState.autoPlayback || mediaState.returnTimer) return;
  const playback = mediaState.autoPlayback;
  mediaState.returnTimer = setTimeout(() => {
    mediaState.returnTimer = null;
    if (mediaState.autoPlayback !== playback) return;
    const nextFrameIdx = playback.nextFrameIdx;
    const returnKind = (
      playback.returnKind || mediaState.lastRealtimeKind || defaultFrameKind()
    );
    mediaState.autoPlayback = null;
    mediaState.actionVideo = null;
    mediaState.kind = returnKind;
    mediaState.frameIndex = -1;
    document.querySelectorAll(".frame-tabs button").forEach(b =>
      b.classList.toggle("active", b.dataset.kind === returnKind));
    // No video reset here — swapMedia keeps the finished video's last
    // frame painted until the realtime PNG is decoded, then flips visibility
    // — the transition never exposes the black framewrap background.
    refreshFrame(nextFrameIdx);
    refreshMeta({ autoPlayNewAction: true, nextFrameIdx });
  }, AUTO_ACTION_RETURN_DELAY_MS);
}

function playActionVideo(step, opts = {}) {
  if (!runState.id || !step || !step.has_action_video) return false;
  cancelAutoActionReturn();
  mediaState.actionVideo = step;
  mediaState.autoPlayback = opts.auto || opts.returnAfterEnd
    ? {
        step: Number(step.step) || 0,
        nextFrameIdx: opts.nextFrameIdx,
        auto: !!opts.auto,
        returnKind: opts.returnKind || mediaState.lastRealtimeKind,
      }
    : null;
  mediaState.kind = "actionVideo";
  document.querySelectorAll(".frame-tabs button").forEach(b => b.classList.remove("active"));
  mediaState.frameIndex = -1;
  // Auto-triggered replays (from maybeAutoPlayNewAction) queue as "auto";
  // manual timeline clicks are user actions and jump the queue.
  refreshFrame(undefined, { source: opts.auto ? "auto" : "user" });
  return true;
}

function playEpisodeVideo() {
  if (!runState.id || !mediaState.episodeVideoAvailable) return;
  cancelAutoActionReturn();
  mediaState.autoPlayback = null;
  mediaState.actionVideo = null;
  mediaState.kind = "video";
  document.querySelectorAll(".frame-tabs button").forEach(b => b.classList.remove("active"));
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
  if (!runState.id) return;
  const source = opts.source || "auto";

  if (mediaState.kind === "actionVideo") {
    if (!mediaState.actionVideo) return;
    const stepNum = Number(mediaState.actionVideo.step) || 0;
    // Note: no ``t=Date.now()`` cache-buster — action video files are
    // written once and never mutate, so the buffer's ``_loadedSrc`` cache
    // gives us instant replay when the same clip is re-clicked.
    const url = `/api/run/action-video?run=${encodeURIComponent(runState.id)}&step=${encodeURIComponent(mediaState.actionVideo.step)}`;
    const cap = copy.actionCaption(
      mediaState.actionVideo.step,
      mediaState.actionVideo.action,
    );
    swapMedia({
      kind: "video",
      url,
      cap,
      source,
      // Hold the queue until the clip ends so queued realtime frames don't
      // cut the replay short. User clicks (source: "user") still release
      // the hold on the current in-flight swap immediately.
      holdUntilEnded: true,
      onReady: (v) => {
        try { v.currentTime = 0; } catch {}
        v.playbackRate = 0.5;
        const shouldReturn =
          mediaState.autoPlayback && mediaState.autoPlayback.step === stepNum;
        v.muted = !!(shouldReturn && mediaState.autoPlayback.auto);
        v.onended = shouldReturn ? finishAutoActionPlayback : null;
        const p = v.play();
        if (p && typeof p.catch === "function") {
          p.catch(() => { if (shouldReturn) finishAutoActionPlayback(); });
        }
      },
    });
    return;
  }

  if (mediaState.kind === "video") {
    const url = `/api/run/video?run=${encodeURIComponent(runState.id)}`;
    swapMedia({
      kind: "video",
      url,
      cap: copy.episodeVideo,
      source,
      holdUntilEnded: true,
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
  const url = `/api/run/frame?run=${encodeURIComponent(runState.id)}&kind=${mediaState.kind}&t=${Date.now()}`;
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

async function refreshMeta(opts = {}) {
  if (!runState.id) return;
  const r = await fetch(`/api/run?run=${encodeURIComponent(runState.id)}`).then(x => x.json());
  const generationState = syncTaskGeneration(r);
  if (generationState === "stale") return;
  setBadge(r.state, r.control_error || r.error);
  setResult(r.terminated, r.state);
  renderRuntimeStatus(r.runtime);
  interactionController.applySnapshot(r);
  const currentTask = r.current_task;
  renderTaskMeta(currentTask);
  mediaState.frameAvailable = r.frame_available || null;
  if (r.usage) $("#usageMeta").textContent = copy.usage(r.usage);
  renderTimeline(r.timeline || [], r.has_video, {
    animateNew: timelineState.initialized,
  });
  if (opts.primeAutoActionStep) {
    mediaState.lastActionStep = maxTimelineStep(r.timeline || []);
    mediaState.autoActionPrimed = true;
  }
  const autoStarted = AUTO_PLAY_ACTION_VIDEOS && opts.autoPlayNewAction
    && mediaState.autoActionPrimed && !mediaState.autoPlayback
    ? maybeAutoPlayNewAction(r.timeline || [], opts.nextFrameIdx ?? r.frame_idx)
    : false;
  if (!r.has_video && mediaState.kind === "video") setFrameKind(defaultFrameKind());
  if (
    !autoStarted
    && !mediaState.autoPlayback
    && isRealtimeKind(mediaState.kind)
    && currentTask
  ) refreshFrame(r.frame_idx);
  if (generationState === "changed") refreshTranscript();
}

function connectSSE() {
  if (runState.eventSource) runState.eventSource.close();
  runState.eventSource = new EventSource(`/api/stream?run=${encodeURIComponent(runState.id)}`);
  runState.eventSource.onmessage = (e) => {
    const sig = JSON.parse(e.data);
    const generationState = syncTaskGeneration(sig);
    if (generationState === "stale") return;
    setBadge(sig.state, sig.control_error || sig.error);
    setResult(sig.terminated, sig.state);
    renderRuntimeStatus(sig.runtime);
    interactionController.applySnapshot(sig);
    mediaState.frameAvailable = sig.frame_available || null;
    if (sig.usage) $("#usageMeta").textContent = copy.usage(sig.usage);
    $("#connMeta").textContent = copy.live;
    refreshTranscript();
    if (generationState === "changed") {
      refreshMeta({ primeAutoActionStep: true });
      return;
    }
    // refresh timeline lazily on step change
    if (sig.n_steps !== runState.lastStepCount) {
      runState.lastStepCount = sig.n_steps;
      // Suppress realtime frame refreshes for the duration of the step
      // transition. ``refreshMeta`` is async (awaits ``/api/run``); during
      // that await, intermediate SSE snapshots carry the post-step
      // ``frame_idx`` but ``mediaState.autoPlayback`` is not set yet, so the
      // realtime branch below would queue the completion image BEFORE
      // ``refreshMeta`` resolves and queues the action video — producing
      // the wrong order (image → video). The guard holds those ticks off
      // until ``refreshMeta`` finishes; if it queued an action video the
      // video's ``holdUntilEnded`` + ``mediaState.autoPlayback`` take over the
      // suppression, and if it didn't, ``refreshMeta`` itself queues the
      // completion frame in the right place.
      mediaState.stepTransitioning = true;
      refreshMeta({
        autoPlayNewAction: mediaState.autoActionPrimed,
        primeAutoActionStep: !mediaState.autoActionPrimed,
        nextFrameIdx: sig.frame_idx,
      }).finally(() => { mediaState.stepTransitioning = false; });
      return;
    }
    if (sig.has_video && !mediaState.episodeVideoAvailable) refreshMeta();
    if (
      !mediaState.autoPlayback
      && !mediaState.stepTransitioning
      && isRealtimeKind(mediaState.kind)
      && sig.frame_idx != null
      && sig.frame_idx !== mediaState.frameIndex
    ) refreshFrame(sig.frame_idx);
  };
  runState.eventSource.onerror = () => {
    $("#connMeta").textContent = copy.reconnecting;
  };
}

function selectRun(id) {
  runState.id = id;
  runState.taskGeneration = null;
  resetRenderedTaskProjection();
  renderRuntimeStatus(null);
  $("#transcript").innerHTML = `<div class="empty">${copy.loading}</div>`;
  $("#timeline").innerHTML = '<div class="empty">…</div>';
  refreshMeta({ primeAutoActionStep: true });
  refreshTranscript();
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

// --- launcher (start screen) ---
function populateModelPresets(planner, selected = "") {
  const preset = $("#f-model_preset");
  const values = MODEL_PRESETS[planner];
  const model = selected || values[0];
  preset.innerHTML = "";
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    preset.appendChild(option);
  }
  const selectedPreset = values.includes(model);
  preset.value = selectedPreset ? model : values[0];
  $("#f-model_custom").value = selectedPreset ? "" : model;
}

function selectedModel() {
  return $("#f-model_custom").value.trim() || $("#f-model_preset").value.trim();
}

function showLauncher(defaults) {
  const d = defaults || {};
  const planner = MODEL_PRESETS[d.planner] ? d.planner : "claude_code";
  const defaultModel = d.model || MODEL_PRESETS[planner][0];
  const set = (id, val) => { $(id).value = val == null ? "" : val; };
  set("#f-max-turns", d["max-turns"]);
  set("#f-max-episode-steps", d["max-episode-steps"]);
  set("#f-planner-timeout-s", d["planner-timeout-s"]);
  set("#f-reasoning-effort", d["reasoning-effort"] || "none");
  set("#f-claude-code-max-budget-usd", d["claude-code-max-budget-usd"]);
  set("#f-cuda-device", d["cuda-device"]);
  $("#f-no-images").checked = Boolean(d["no-images"]);
  for (const name of Object.keys(launcherModelSelections)) {
    launcherModelSelections[name] = MODEL_PRESETS[name][0];
  }
  launcherModelSelections[planner] = defaultModel;
  activeLauncherPlanner = planner;
  $("#f-planner").value = planner;
  populateModelPresets(planner, defaultModel);
  updatePlannerFields();
  $("#launcher").classList.remove("hidden");
}

function updatePlannerFields() {
  const planner = $("#f-planner").value;
  $("#modelRequirement").textContent = planner === "api" ? copy.required : copy.optional;
  for (const field of document.querySelectorAll("[data-planner]")) {
    field.classList.toggle("hidden", field.dataset.planner !== planner);
  }
}

function collectLaunchConfig() {
  const numOrNull = (id) => {
    const v = $(id).value.trim();
    return v === "" ? null : Number(v);
  };
  const planner = $("#f-planner").value;
  const config = {
    planner,
    "max-turns": numOrNull("#f-max-turns"),
    "max-episode-steps": numOrNull("#f-max-episode-steps"),
    model: selectedModel(),
    "planner-timeout-s": numOrNull("#f-planner-timeout-s"),
    "reasoning-effort": $("#f-reasoning-effort").value,
    "no-images": $("#f-no-images").checked,
    "cuda-device": $("#f-cuda-device").value.trim(),
  };
  if (planner === "claude_code") {
    config["claude-code-max-budget-usd"] = numOrNull("#f-claude-code-max-budget-usd");
  }
  return config;
}

async function pollForRun() {
  try {
    const r = await fetch("/api/runs").then(x => x.json());
    if (r.runs && r.runs.length) {
      $("#launcher").classList.add("hidden");
      loadRun();
      return;
    }
  } catch (e) { /* transient — retry */ }
  setTimeout(pollForRun, 600);
}

async function onRun() {
  const config = collectLaunchConfig();
  const requiredNums = [
    [copy.requiredFields.maxTurns, config["max-turns"]],
    [copy.requiredFields.maxEpisodeSteps, config["max-episode-steps"]],
  ];
  if (config.planner === "api" && !/^[^:]+:.+$/.test(config.model)) {
    $("#launchStatus").textContent = copy.fieldRequired(copy.requiredFields.apiModel);
    return;
  }
  const badNum = requiredNums.find(([_, v]) => v == null || !Number.isFinite(v));
  if (badNum) {
    $("#launchStatus").textContent = copy.fieldRequired(badNum[0]);
    return;
  }
  $("#runBtn").disabled = true;
  $("#launchStatus").textContent = copy.starting;
  try {
    const resp = await fetch("/api/launch/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    if (!resp.ok) throw new Error(await resp.text());
  } catch (e) {
    $("#runBtn").disabled = false;
    $("#launchStatus").textContent = copy.startFailed;
    return;
  }
  pollForRun();
}
$("#runBtn").addEventListener("click", onRun);
$("#f-planner").addEventListener("change", () => {
  launcherModelSelections[activeLauncherPlanner] = selectedModel();
  activeLauncherPlanner = $("#f-planner").value;
  populateModelPresets(
    activeLauncherPlanner,
    launcherModelSelections[activeLauncherPlanner],
  );
  updatePlannerFields();
});

async function boot() {
  applyStaticCopy();
  let st;
  let dashboardSpec;
  try {
    [st, dashboardSpec] = await Promise.all([
      fetch("/api/launch/state")
        .then(response => response.json())
        .catch(() => ({ enabled: false })),
      fetch("/api/commands").then(response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      }),
    ]);
  } catch (error) {
    console.error("Failed to load Dashboard configuration", error);
    $("#launcher").classList.remove("hidden");
    $("#runBtn").disabled = true;
    $("#launchStatus").textContent = copy.dashboardConfigFailed;
    return;
  }
  configureDashboardSpec(dashboardSpec);
  interactionController.configureTaskCommand(dashboardSpec.task);
  if (st.enabled && st.pending) {
    showLauncher(st.defaults);
  } else {
    // launcher already submitted (page reloaded) or not armed — go to monitor
    loadRun();
  }
}

boot();
