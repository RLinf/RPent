import { createInteractionController } from "./interaction.js";
import { makeAssistantTextElement } from "./markdown_table.js";

function $(selector) {
  return document.querySelector(selector);
}

const LANGUAGE = document.documentElement.lang === "zh-cn" ? "zh-cn" : "en";

const COPY = {
  en: {
    pageTitle: "RPent · Live Monitor",
    newRun: "New Run",
    launcherSubtitle: "Review the config, then click Run to start the agent.",
    suite: "Suite",
    task: "Task",
    seed: "Seed",
    planner: "Planner (LLM backend)",
    maxTurns: "Max turns",
    maxTokens: "Max tokens",
    maxEpisodeSteps: "Max episode steps",
    modelPreset: "Model preset",
    customModel: "Custom model",
    optional: "(optional)",
    customModelPlaceholder: "provider:model or alias",
    claudeBudget: "Claude Code budget USD",
    plannerTimeout: "Planner timeout s",
    cudaDevice: "CUDA device",
    blankDefault: "(blank = default)",
    defaultPlaceholder: "default",
    run: "Run",
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
    resizeComposer: "Drag to resize composer height · double-click to reset",
    composerPlaceholder: "Message the agent…",
    composerKeys: "Enter to send · Shift+Enter for newline · Esc to interrupt the agent",
    interactionStarting: "Waiting for environment startup…",
    interactionReady: "The agent is ready for another message.",
    interactionBusy: "The agent is working; new messages will be queued.",
    interactionUnavailable: "The agent is not accepting messages yet.",
    interruptRequested: "Interrupt requested; it will be handled when the agent's control channel is available.",
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
    cameraView: "fixed camera",
    wristView: "wrist camera",
    waitingFrame: "waiting for first frame…",
    frameUnavailable: (kind) =>
      `${kind === "wrist" ? "wrist camera" : "fixed camera"} unavailable`,
    resizeFrame: "Drag to resize frame height",
    actionTimeline: "Action timeline",
    noActions: "No actions yet.",
    stateLabels: {
      starting: "starting",
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
    selectModel: "select a model",
    backendDefault: "backend default",
    noTranscript: "No transcript events yet.",
    you: "You",
    initialTaskSubmitted:
      "Initial task instructions were generated from the run configuration and submitted to the agent automatically.",
    loading: "Loading…",
    live: "● live",
    reconnecting: "○ reconnecting…",
    requiredFields: {
      task: "Task",
      seed: "Seed",
      maxTurns: "Max turns",
      maxEpisodeSteps: "Max episode steps",
      maxTokens: "Max tokens",
    },
    suiteRequired: "Suite is required.",
    fieldRequired: (field) => `${field} is required.`,
    apiModelRequired: "API model must include a provider prefix, for example anthropic:claude-opus-4-8.",
    starting: "starting run… this page will switch to the live monitor.",
    startFailed: "failed to start — check the terminal.",
    noRuns: (directory) => `(no runs in ${directory})`,
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
    frameCaption: (kind, index) => {
      const label = kind === "wrist" ? "wrist camera" : "fixed camera";
      return `${label} · frame #${index}`;
    },
    taskDetails: (task, seed) => ` · task ${task} · seed ${seed}`,
    usage: (usage) =>
      `token in ${usage.in.toLocaleString()} · out ${usage.out.toLocaleString()} · ${usage.tool_calls} tools`,
  },
  "zh-cn": {
    pageTitle: "RPent · 实时监控",
    newRun: "新建任务",
    launcherSubtitle: "确认配置后,点击「开始运行」启动智能体。",
    suite: "任务集",
    task: "任务编号",
    seed: "随机种子",
    planner: "决策大脑(大模型后端)",
    maxTurns: "最大对话轮数",
    maxTokens: "最大 token 数",
    maxEpisodeSteps: "最大仿真步数",
    modelPreset: "模型预设",
    customModel: "自定义模型",
    optional: "(可选)",
    customModelPlaceholder: "provider:model 或别名",
    claudeBudget: "Claude Code 预算 USD",
    plannerTimeout: "Planner 超时秒数",
    cudaDevice: "CUDA 设备",
    blankDefault: "(留空=默认)",
    defaultPlaceholder: "默认",
    run: "开始运行",
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
    resizeComposer: "拖动调整输入区高度 · 双击复位",
    composerPlaceholder: "向智能体发送消息…",
    composerKeys: "Enter 发送 · Shift+Enter 换行 · Esc 中断智能体",
    interactionStarting: "正在等待环境启动…",
    interactionReady: "智能体已准备好接收新消息。",
    interactionBusy: "智能体正在工作；新消息将进入等待队列。",
    interactionUnavailable: "智能体暂未开始接收消息。",
    interruptRequested: "已请求中断智能体；控制通道可用后将处理。",
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
    cameraView: "固定相机",
    wristView: "腕部相机",
    waitingFrame: "等待第一帧…",
    frameUnavailable: (kind) =>
      `${kind === "wrist" ? "腕部相机" : "固定相机"}画面不可用`,
    resizeFrame: "拖动调整画面高度",
    actionTimeline: "动作时间线",
    noActions: "暂无动作。",
    stateLabels: {
      starting: "启动中",
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
    selectModel: "选择模型",
    backendDefault: "后端默认",
    noTranscript: "暂无推理记录。",
    you: "你",
    initialTaskSubmitted: "已根据当前任务配置，自动向智能体提交初始任务指令。",
    loading: "加载中…",
    live: "● 实时",
    reconnecting: "○ 正在重连…",
    requiredFields: {
      task: "任务编号",
      seed: "随机种子",
      maxTurns: "最大对话轮数",
      maxEpisodeSteps: "最大仿真步数",
      maxTokens: "最大 token 数",
    },
    suiteRequired: "请选择任务集。",
    fieldRequired: (field) => `请填写${field}。`,
    apiModelRequired: "API 模型必须包含 provider 前缀,例如 anthropic:claude-opus-4-8。",
    starting: "正在启动运行… 页面将切换到实时监控。",
    startFailed: "启动失败,请查看终端输出。",
    noRuns: (directory) => `(${directory} 中暂无运行)`,
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
    frameCaption: (kind, index) => {
      const label = kind === "wrist" ? "腕部相机" : "固定相机";
      return `${label} · 第 ${index} 帧`;
    },
    taskDetails: (task, seed) => ` · 任务 ${task} · 种子 ${seed}`,
    usage: (usage) =>
      `输入 ${usage.in.toLocaleString()} · 输出 ${usage.out.toLocaleString()} · ${usage.tool_calls} 次工具调用`,
  },
};

const copy = COPY[LANGUAGE];
const RUNTIME_COMPONENTS = ["env", "vla", "sam3"];
const RUNTIME_STATES = ["pending", "starting", "ready", "failed"];

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

const runState = {
  id: null,
  eventSource: null,
  lastStepCount: -1,
};

const transcriptState = {
  shown: 0,
  toolGroup: null,
  inFlight: false,
  refreshAgain: false,
  initialized: false,
};

const timelineState = {
  initialized: false,
  seen: new Set(),
};

const mediaState = {
  kind: "camera",
  frameIndex: -1,
  frameAvailable: null,
  unavailableKind: null,
  actionVideo: null,
  episodeVideoAvailable: false,
  lastRealtimeKind: "camera",
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
const MODEL_PRESETS = {
  api: [
    "",
    "deepseek:deepseek-chat",
    "anthropic:claude-opus-4-7",
    "anthropic:claude-opus-4-8",
    "anthropic:claude-sonnet-4-5",
    "openai:gpt-5.5",
    "openai-chat:glm-5.2",
  ],
  claude_code: ["", "claude-opus-4-7", "sonnet", "opus"],
  codex: [""],
};

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
  mediaState.kind = "camera";
  mediaState.frameIndex = -1;
  mediaState.frameAvailable = null;
  mediaState.unavailableKind = null;
  mediaState.actionVideo = null;
  mediaState.episodeVideoAvailable = false;
  mediaState.lastRealtimeKind = "camera";
  mediaState.lastActionStep = 0;
  mediaState.autoActionPrimed = false;
  mediaState.autoPlayback = null;
  mediaState.stepTransitioning = false;
  resetMediaBuffers();
}

function resetTranscriptForRun() {
  transcriptState.shown = 0;
  transcriptState.toolGroup = null;
  transcriptState.refreshAgain = false;
  transcriptState.initialized = false;
}

function resetTimelineForRun() {
  timelineState.initialized = false;
  timelineState.seen.clear();
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
  return kind === "camera" || kind === "wrist";
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

  const items = RUNTIME_COMPONENTS.map(function (component) {
    const info = runtime[component];
    const candidate = typeof info === "string" ? info : info?.status;
    const status = RUNTIME_STATES.includes(candidate) ? candidate : "pending";
    const item = document.createElement("span");
    item.className = `runtime-item runtime-${status}`;
    item.textContent = `${copy.runtimeLabels[component]} ${copy.runtimeStates[status]}`;
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
          if (ev.type === "user") div.dataset.roleLabel = `${copy.you} › `;
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
  try {
    const r = await fetch(
      `/api/run/transcript?run=${encodeURIComponent(run)}&since=${transcriptState.shown}`
    ).then(x => x.json());
    if (run !== runState.id) return;            // switched run mid-flight — drop
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
    const returnKind = playback.returnKind || mediaState.lastRealtimeKind || "camera";
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
  $("#frameCap").textContent = copy.frameUnavailable(kind);
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
    cap: copy.frameCaption(mediaState.kind, mediaState.frameIndex),
    errorCap: copy.frameUnavailable(mediaState.kind),
    source,
    onReady: () => { mediaState.unavailableKind = null; },
    onError: () => { mediaState.unavailableKind = mediaState.kind; },
  });
}

async function refreshMeta(opts = {}) {
  if (!runState.id) return;
  const r = await fetch(`/api/run?run=${encodeURIComponent(runState.id)}`).then(x => x.json());
  setBadge(r.state, r.error);
  setResult(r.terminated, r.state);
  renderRuntimeStatus(r.runtime);
  interactionController.applySnapshot(r.interaction, r.state);
  const taskMeta = $("#taskMeta");
  const suite = document.createElement("b");
  suite.textContent = r.suite ?? r.name;
  taskMeta.replaceChildren(
    suite,
    document.createTextNode(copy.taskDetails(r.task ?? "?", r.seed ?? "?")),
  );
  mediaState.frameAvailable = r.frame_available || null;
  if (r.usage) $("#usageMeta").textContent = copy.usage(r.usage);
  renderTimeline(r.timeline || [], r.has_video, {
    animateNew: timelineState.initialized,
  });
  if (opts.primeAutoActionStep) {
    mediaState.lastActionStep = maxTimelineStep(r.timeline || []);
    mediaState.autoActionPrimed = true;
  }
  const autoStarted = opts.autoPlayNewAction && mediaState.autoActionPrimed && !mediaState.autoPlayback
    ? maybeAutoPlayNewAction(r.timeline || [], opts.nextFrameIdx ?? r.frame_idx)
    : false;
  if (!r.has_video && mediaState.kind === "video") setFrameKind("camera");
  if (
    !autoStarted
    && !mediaState.autoPlayback
    && isRealtimeKind(mediaState.kind)
  ) refreshFrame(r.frame_idx);
}

function connectSSE() {
  if (runState.eventSource) runState.eventSource.close();
  runState.eventSource = new EventSource(`/api/stream?run=${encodeURIComponent(runState.id)}`);
  runState.eventSource.onmessage = (e) => {
    const sig = JSON.parse(e.data);
    setBadge(sig.state, sig.error);
    setResult(sig.terminated, sig.state);
    renderRuntimeStatus(sig.runtime);
    interactionController.applySnapshot(sig.interaction, sig.state);
    mediaState.frameAvailable = sig.frame_available || null;
    if (sig.usage) $("#usageMeta").textContent = copy.usage(sig.usage);
    $("#connMeta").textContent = copy.live;
    refreshTranscript();
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
  runState.lastStepCount = -1;
  resetTranscriptForRun();
  resetTimelineForRun();
  resetMediaForRun();
  interactionController.reset();
  renderRuntimeStatus(null);
  document.querySelectorAll(".frame-tabs button").forEach(b =>
    b.classList.toggle("active", b.dataset.kind === "camera"));
  $("#transcript").innerHTML = `<div class="empty">${copy.loading}</div>`;
  $("#timeline").innerHTML = '<div class="empty">…</div>';
  $("#frameCap").textContent = copy.waitingFrame;
  refreshMeta({ primeAutoActionStep: true });
  refreshTranscript();
  connectSSE();
}

document.querySelectorAll(".frame-tabs button").forEach(btn => {
  btn.addEventListener("click", () => setFrameKind(btn.dataset.kind));
});
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
  const current = selected || preset.value || "";
  preset.innerHTML = "";
  for (const value of (MODEL_PRESETS[planner] || [""])) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value || (
      planner === "api" ? copy.selectModel : copy.backendDefault
    );
    preset.appendChild(option);
  }
  preset.value = (MODEL_PRESETS[planner] || []).includes(current) ? current : "";
}

function updateBackendFields() {
  const planner = $("#f-planner").value;
  document.querySelectorAll("[data-backends]").forEach(el => {
    const allowed = (el.dataset.backends || "").split(/\s+/);
    el.classList.toggle("hidden", !allowed.includes(planner));
  });
  const currentModel = $("#f-model_custom").value || $("#f-model_preset").value;
  populateModelPresets(planner, currentModel);
}

function selectedModel() {
  return $("#f-model_custom").value.trim() || $("#f-model_preset").value.trim();
}

function showLauncher(defaults) {
  const d = defaults || {};
  const set = (id, val) => { $(id).value = val == null ? "" : val; };
  set("#f-suite", d.suite);
  set("#f-task", d.task);
  set("#f-seed", d.seed);
  set("#f-planner", d.planner || "claude_code");
  set("#f-max-turns", d["max-turns"]);
  set("#f-max-tokens", d["max-tokens"]);
  set("#f-max-episode-steps", d["max-episode-steps"]);
  set("#f-planner-timeout-s", d["planner-timeout-s"]);
  set("#f-claude-code-max-budget-usd", d["claude-code-max-budget-usd"]);
  set("#f-cuda-device", d["cuda-device"]);
  populateModelPresets($("#f-planner").value, d.model || "");
  if ((MODEL_PRESETS[$("#f-planner").value] || []).includes(d.model || "")) {
    set("#f-model_custom", "");
  } else {
    set("#f-model_custom", d.model);
  }
  updateBackendFields();
  $("#launcher").classList.remove("hidden");
}

function collectLaunchConfig() {
  const numOrNull = (id) => {
    const v = $(id).value.trim();
    return v === "" ? null : Number(v);
  };
  return {
    suite: $("#f-suite").value.trim(),
    task: numOrNull("#f-task"),
    seed: numOrNull("#f-seed"),
    planner: $("#f-planner").value,
    "max-turns": numOrNull("#f-max-turns"),
    "max-tokens": numOrNull("#f-max-tokens"),
    "max-episode-steps": numOrNull("#f-max-episode-steps"),
    model: selectedModel(),
    "planner-timeout-s": numOrNull("#f-planner-timeout-s"),
    "claude-code-max-budget-usd": numOrNull("#f-claude-code-max-budget-usd"),
    "cuda-device": $("#f-cuda-device").value.trim(),
  };
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
    [copy.requiredFields.task, config.task],
    [copy.requiredFields.seed, config.seed],
    [copy.requiredFields.maxTurns, config["max-turns"]],
    [copy.requiredFields.maxEpisodeSteps, config["max-episode-steps"]],
  ];
  if (config.planner === "api") {
    requiredNums.push([copy.requiredFields.maxTokens, config["max-tokens"]]);
  }
  const badNum = requiredNums.find(([_, v]) => v == null || !Number.isFinite(v));
  if (!config.suite || badNum) {
    $("#launchStatus").textContent = !config.suite
      ? copy.suiteRequired
      : copy.fieldRequired(badNum[0]);
    return;
  }
  if (config.planner === "api" && (!config.model || !config.model.includes(":"))) {
    $("#launchStatus").textContent = copy.apiModelRequired;
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
  $("#f-model_custom").value = "";
  $("#launchStatus").textContent = "";
  updateBackendFields();
});

async function boot() {
  applyStaticCopy();
  let st = { enabled: false };
  try { st = await fetch("/api/launch/state").then(x => x.json()); } catch (e) {}
  if (st.enabled && st.pending) {
    showLauncher(st.defaults);
  } else {
    // launcher already submitted (page reloaded) or not armed — go to monitor
    loadRun();
  }
}

boot();
