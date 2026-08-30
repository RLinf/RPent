function $(selector) {
  return document.querySelector(selector);
}

const TARGET_ACTIONS = {
  chassis: ["forward", "backward", "turn_left", "turn_right", "up", "down", "observe"],
  left_arm: ["up", "down", "rotate_left", "rotate_right", "open", "close", "observe"],
  right_arm: ["up", "down", "rotate_left", "rotate_right", "open", "close", "observe"],
};

const KEY_ACTIONS = {
  ArrowUp: ["chassis", "forward"],
  ArrowDown: ["chassis", "backward"],
  ArrowLeft: ["chassis", "turn_left"],
  ArrowRight: ["chassis", "turn_right"],
  PageUp: ["chassis", "up"],
  PageDown: ["chassis", "down"],
};

const EDITABLE_TAGS = new Set(["INPUT", "TEXTAREA", "SELECT"]);

const controlState = {
  run: null,
  leaseId: `lease_${globalThis.crypto && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(16).slice(2)}`,
  sequence: 1,
  target: "chassis",
  action: "forward",
  camera: "head",
  preparedPlanId: null,
  commandId: null,
  busy: false,
  activeInteraction: null,
  available: false,
  motionAvailable: false,
  observeAvailable: false,
  unavailableReason: "manual control unavailable",
  capabilities: {},
  controlsExpanded: true,
};

function controlsRoot() {
  return $("#interactiveControls") || $("#behaviorControls");
}

function framewrap() {
  return $("#framewrap") || $(".framewrap.behavior-mode");
}

function setReceipt(text, error = false) {
  const receipt = $("#behaviorReceipt");
  if (!receipt) return;
  receipt.textContent = text || "";
  receipt.classList.toggle("error", !!error);
}

function setControlStatus(text, error = false) {
  for (const status of document.querySelectorAll(".control-status")) {
    if (status.getAttribute("aria-hidden") === "true") continue;
    status.textContent = text || "";
    status.classList.toggle("error", !!error);
  }
}

function setButtons(selector, value, attr) {
  for (const button of document.querySelectorAll(selector)) {
    const selected = button.getAttribute(attr) === value;
    button.classList.toggle("active", selected);
    if (attr === "data-behavior-target") {
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    }
    if (attr === "data-target") {
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    }
    if (attr === "data-behavior-camera" || attr === "data-kind") {
      button.setAttribute("aria-pressed", String(selected));
    }
  }
}

function setTarget(target) {
  if (!Object.prototype.hasOwnProperty.call(TARGET_ACTIONS, target)) return;
  controlState.target = target;
  if (!TARGET_ACTIONS[target].includes(controlState.action)) {
    controlState.action = TARGET_ACTIONS[target][0];
  }
  setButtons("[data-behavior-target]", controlState.target, "data-behavior-target");
  setButtons("[data-target]", controlState.target, "data-target");
  renderActionAvailability();
}

function setAction(action) {
  if (!TARGET_ACTIONS[controlState.target].includes(action)) return;
  controlState.action = action;
  setButtons("[data-behavior-action]", controlState.action, "data-behavior-action");
  setButtons("[data-action]", controlState.action, "data-action");
}

function setCamera(camera) {
  controlState.camera = camera;
  setButtons("[data-behavior-camera]", controlState.camera, "data-behavior-camera");
  setButtons("[data-camera]", controlState.camera, "data-camera");
  setButtons(".behavior-frame-tabs button", controlState.camera, "data-kind");
  postCameraSelection(camera).catch(error => setReceipt(error.message, true));
}

function renderActionAvailability() {
  const allowed = new Set(TARGET_ACTIONS[controlState.target]);
  for (const button of document.querySelectorAll("[data-behavior-action], [data-action]")) {
    const action = button.getAttribute("data-behavior-action")
      || button.getAttribute("data-action");
    button.classList.toggle("active", action === controlState.action);
    button.classList.toggle("target-mismatch", !allowed.has(action));
  }
  updateControlTooltips();
}

function controlTooltip(action) {
  if (action === "observe") return "Refresh the currently selected camera view.";
  if (action === "open") return "Open the selected gripper and keep it open.";
  if (action === "close") return "Close the selected gripper and maintain gripping pressure.";
  if (controlState.target === "chassis") {
    const tips = {
      forward: "Move the chassis forward by 5 cm. Hold to continue.",
      backward: "Move the chassis backward by 5 cm. Hold to continue.",
      turn_left: "Rotate the chassis left by 5°. Hold to continue.",
      turn_right: "Rotate the chassis right by 5°. Hold to continue.",
      up: "Raise the R1Pro torso by 3 cm. Hold to continue.",
      down: "Lower the R1Pro torso by 3 cm. Hold to continue.",
    };
    return tips[action] || "Available for arm control only.";
  }
  const hand = controlState.target === "left_arm" ? "left" : "right";
  const tips = {
    up: `Move the ${hand} hand up by 3 cm. Hold to continue.`,
    down: `Move the ${hand} hand down by 3 cm. Hold to continue.`,
    rotate_left: "Rotate the selected wrist 5° counterclockwise. Hold to continue.",
    rotate_right: "Rotate the selected wrist 5° clockwise. Hold to continue.",
  };
  return tips[action] || "Available for chassis control only.";
}

function unavailableTooltip(kind) {
  const capabilities = controlState.capabilities || {};
  const specific = kind === "observe"
    ? capabilities.observe_unavailable_reason
    : capabilities.motion_unavailable_reason;
  return String(
    specific
      || capabilities.unavailable_reason
      || controlState.unavailableReason
      || `${kind === "observe" ? "Camera refresh" : "Manual motion control"} is unavailable.`,
  );
}

function targetMismatchTooltip(action) {
  if (controlState.target === "chassis"
      && ["rotate_left", "rotate_right", "open", "close"].includes(action)) {
    return "Available for arm control only.";
  }
  return "Available for chassis control only.";
}

function setButtonTooltip(button, tooltip) {
  const text = String(tooltip || "").trim();
  button.dataset.tooltip = text;
  button.removeAttribute("title");
}

function updateControlTooltips() {
  for (const button of document.querySelectorAll("[data-behavior-target], [data-target]")) {
    setButtonTooltip(button, `Control the ${button.textContent.trim().toLowerCase()}.`);
  }
  for (const button of document.querySelectorAll("[data-behavior-action], [data-action]")) {
    const action = button.getAttribute("data-behavior-action")
      || button.getAttribute("data-action");
    const allowed = TARGET_ACTIONS[controlState.target].includes(action);
    let tooltip = controlTooltip(action);
    if (!allowed) {
      tooltip = targetMismatchTooltip(action);
    } else if (action === "observe" && !controlState.observeAvailable) {
      tooltip = unavailableTooltip("observe");
    } else if (action !== "observe" && !controlState.motionAvailable) {
      tooltip = unavailableTooltip("motion");
    }
    setButtonTooltip(button, tooltip);
  }
}

function localControlSnapshot(phase, extra = {}) {
  return {
    available: controlState.available,
    motion_available: controlState.motionAvailable,
    observe_available: controlState.observeAvailable,
    unavailable_reason: controlState.unavailableReason,
    capabilities: controlState.capabilities,
    selected_camera: controlState.camera,
    prepared_plan_id: controlState.preparedPlanId,
    command_id: controlState.commandId,
    phase,
    ...extra,
  };
}

function renderControl(snapshot = {}) {
  controlState.available = !!snapshot.available;
  controlState.motionAvailable = !!snapshot.motion_available;
  controlState.observeAvailable = !!snapshot.observe_available;
  if (snapshot.capabilities && typeof snapshot.capabilities === "object") {
    controlState.capabilities = snapshot.capabilities;
  }
  if (Object.prototype.hasOwnProperty.call(snapshot, "unavailable_reason")) {
    controlState.unavailableReason = String(snapshot.unavailable_reason || "");
  }
  controlState.preparedPlanId = snapshot.prepared_plan_id || null;
  controlState.commandId = snapshot.command_id || null;
  if (snapshot.selected_camera) {
    controlState.camera = snapshot.selected_camera;
    setButtons("[data-behavior-camera]", controlState.camera, "data-behavior-camera");
    setButtons("[data-camera]", controlState.camera, "data-camera");
    setButtons(".behavior-frame-tabs button", controlState.camera, "data-kind");
  }

  const stateLabel = $("#behaviorManualControlState");
  const phase = snapshot.phase || "offline";
  const reason = snapshot.unavailable_reason ? ` · ${snapshot.unavailable_reason}` : "";
  if (stateLabel) {
    stateLabel.textContent = `${phase}${reason}`;
  }
  setControlStatus(`${phase}${reason}`, !!snapshot.unavailable_reason);

  const interactionActive = !!controlState.activeInteraction;
  const controlsBlocked = controlState.busy || interactionActive;
  const canPrepare = controlState.motionAvailable && !controlsBlocked;
  const canExecute = !!controlState.preparedPlanId && !controlsBlocked;
  const canDiscard = !!controlState.preparedPlanId && !controlsBlocked;
  const canCapture = controlState.observeAvailable && !controlsBlocked;
  setElementDisabled("#behaviorPrepare", !canPrepare);
  setElementDisabled("#behaviorExecute", !canExecute);
  setElementDisabled("#behaviorDiscard", !canDiscard);
  setElementDisabled("#behaviorCapture", !canCapture);
  setElementDisabled("#behaviorStop", controlState.busy && !interactionActive);

  for (const button of document.querySelectorAll("[data-behavior-action], [data-action]")) {
    const action = button.getAttribute("data-behavior-action")
      || button.getAttribute("data-action");
    const allowed = TARGET_ACTIONS[controlState.target].includes(action);
    const actionAvailable = action === "observe"
      ? controlState.observeAvailable
      : controlState.motionAvailable && allowed;
    button.disabled = !actionAvailable || controlsBlocked;
    button.setAttribute("aria-disabled", String(button.disabled));
  }
  const selector = "[data-behavior-target], [data-target], [data-behavior-camera], [data-camera], .behavior-frame-tabs button";
  for (const button of document.querySelectorAll(selector)) {
    button.disabled = controlsBlocked;
    button.setAttribute("aria-disabled", String(button.disabled));
  }
  updateControlTooltips();

  const terminal = snapshot.last_terminal;
  if (terminal) {
    const success = terminal.task_success === true ? "true" : "false";
    const identity = terminal.command_id || terminal.kind || "terminal";
    setReceipt(`terminal receipt: ${identity} task_success=${success}`);
  } else if (snapshot.prepared_plan_id) {
    setReceipt(`prepared: ${snapshot.prepared_plan_id}`);
  }
}

function setElementDisabled(selector, disabled) {
  const element = $(selector);
  if (element) element.disabled = !!disabled;
}

async function resolveRun() {
  const response = await fetch("/api/runs").then(item => item.json());
  const run = response.runs && response.runs[0];
  controlState.run = run ? run.id : null;
  return controlState.run;
}

async function refreshControl() {
  if (!controlsRoot()) return;
  if (!controlState.run) await resolveRun();
  if (!controlState.run) {
    renderControl({ phase: "offline", unavailable_reason: "no run" });
    return;
  }
  try {
    const url = `/api/run/control/state?run=${encodeURIComponent(controlState.run)}`;
    const snapshot = await fetch(url).then(response => response.json());
    if (snapshot.error) {
      renderControl({ phase: "offline", unavailable_reason: snapshot.error });
      return;
    }
    renderControl(snapshot);
  } catch (error) {
    renderControl({ phase: "offline", unavailable_reason: error.message });
  }
}

async function postControl(endpoint, payload = {}) {
  if (!controlState.run) await resolveRun();
  if (!controlState.run) throw new Error("no Dashboard run is registered");
  const response = await fetch(`/api/run/control/${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      run: controlState.run,
      lease_id: controlState.leaseId,
      ...payload,
    }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || data.code || "control request failed");
  }
  renderControl(data);
  return data;
}

async function postCameraSelection(camera) {
  if (!controlState.run) await resolveRun();
  if (!controlState.run) throw new Error("no Dashboard run is registered");
  const response = await fetch("/api/run/control/camera", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      run: controlState.run,
      camera,
    }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || data.code || "camera selection failed");
  }
  renderControl(data);
  return data;
}

async function prepareCommand(target, action, camera) {
  return postControl("prepare", {
    sequence: controlState.sequence++,
    target,
    action,
    camera,
  });
}

async function executeCommand(commandId = controlState.commandId, planId = controlState.preparedPlanId) {
  return postControl("execute", {
    command_id: commandId,
    plan_id: planId,
  });
}

async function discardCommand(commandId = controlState.commandId, planId = controlState.preparedPlanId) {
  return postControl("discard", {
    command_id: commandId,
    plan_id: planId,
  });
}

async function postSafeStop(reason = "dashboard_safe_stop", stopMode = "safe_stop") {
  await requestPlannerInterrupt();
  return postControl("stop", {
    reason,
    stop_mode: stopMode,
  });
}

async function requestPlannerInterrupt() {
  if (!controlState.run) await resolveRun();
  if (!controlState.run) return;
  try {
    await fetch(`/api/sessions/${encodeURIComponent(controlState.run)}/interrupt`, {
      method: "POST",
    });
  } catch (_error) {
    // The BEHAVIOR env stop route remains authoritative for the terminal receipt.
  }
}

async function prepareSelected() {
  if (controlState.busy || controlState.activeInteraction) return;
  controlState.busy = true;
  renderControl(localControlSnapshot("preparing"));
  try {
    const result = await prepareCommand(
      controlState.target,
      controlState.action,
      controlState.camera,
    );
    setReceipt(`prepared: ${result.plan_id || result.prepared_plan_id || ""}`);
  } catch (error) {
    setReceipt(error.message, true);
  } finally {
    controlState.busy = false;
    refreshControl();
  }
}

async function executePrepared() {
  if (controlState.busy || controlState.activeInteraction) return;
  controlState.busy = true;
  try {
    const result = await executeCommand();
    const terminal = result.terminal_receipt || {};
    setReceipt(`executed: ${terminal.command_id || result.command_id || ""}`);
  } catch (error) {
    setReceipt(error.message, true);
  } finally {
    controlState.busy = false;
    refreshControl();
  }
}

async function discardPrepared() {
  if (controlState.busy || controlState.activeInteraction) return;
  controlState.busy = true;
  try {
    const result = await discardCommand();
    setReceipt(`discarded: ${result.command_id || result.plan_id || ""}`);
  } catch (error) {
    setReceipt(error.message, true);
  } finally {
    controlState.busy = false;
    refreshControl();
  }
}

async function captureViews() {
  if (controlState.busy || controlState.activeInteraction) return;
  controlState.busy = true;
  try {
    const result = await postControl("capture");
    setReceipt(`captured: ${result.command_id || ""}`);
  } catch (error) {
    setReceipt(error.message, true);
  } finally {
    controlState.busy = false;
    refreshControl();
  }
}

async function safeStop(reason = "dashboard_safe_stop") {
  if (controlState.activeInteraction) {
    requestInteractionStop(reason);
    return;
  }
  if (controlState.busy) return;
  controlState.busy = true;
  try {
    const result = await postSafeStop(reason);
    const receipt = result.terminal_receipt || {};
    const success = receipt.task_success === true ? "true" : "false";
    setReceipt(`safe-stop receipt: task_success=${success}`);
  } catch (error) {
    setReceipt(error.message, true);
  } finally {
    controlState.busy = false;
    refreshControl();
  }
}

function isEditableTarget(target) {
  if (!target) return false;
  const tagName = String(target.tagName || "").toUpperCase();
  if (EDITABLE_TAGS.has(tagName)) return true;
  if (target.isContentEditable) return true;
  const closest = target.closest;
  return typeof closest === "function"
    && !!closest.call(target, "input, textarea, select, [contenteditable], [role='textbox']");
}

function keyToken(event) {
  return `key:${event.code || event.key}`;
}

function pointerToken(event) {
  return `pointer:${event.pointerId ?? "mouse"}`;
}

function beginMomentaryAction(token, target, action) {
  if (controlState.busy || controlState.activeInteraction) return false;
  if (action === "observe") {
    setTarget(target);
    setAction(action);
    captureViews();
    return true;
  }
  if (!controlState.motionAvailable) {
    setReceipt("motion unavailable", true);
    return false;
  }
  setTarget(target);
  setAction(action);
  const interaction = {
    token,
    executed: false,
    cancelRequested: false,
    stopReason: null,
  };
  controlState.activeInteraction = interaction;
  runMomentaryInteraction(interaction).catch(error => {
    if (controlState.activeInteraction === interaction) {
      controlState.activeInteraction = null;
    }
    controlState.busy = false;
    setReceipt(error.message, true);
    refreshControl();
  });
  return true;
}

async function runMomentaryInteraction(interaction) {
  controlState.busy = true;
  renderControl(localControlSnapshot("preparing"));
  try {
    const prepared = await prepareCommand(
      controlState.target,
      controlState.action,
      controlState.camera,
    );
    if (controlState.activeInteraction !== interaction) return;
    setReceipt(`prepared: ${prepared.plan_id || prepared.prepared_plan_id || ""}`);
    if (interaction.cancelRequested) {
      await finishInteractionStop(interaction);
      return;
    }

    const commandId = prepared.command_id || controlState.commandId;
    const planId = prepared.plan_id || prepared.prepared_plan_id || controlState.preparedPlanId;
    const result = await executeCommand(commandId, planId);
    if (controlState.activeInteraction !== interaction) return;
    interaction.executed = true;
    const terminal = result.terminal_receipt || {};
    setReceipt(`executed: ${terminal.command_id || result.command_id || commandId || ""}`);
    if (interaction.cancelRequested) {
      await finishInteractionStop(interaction);
    }
  } catch (error) {
    if (controlState.activeInteraction === interaction) {
      controlState.activeInteraction = null;
    }
    setReceipt(error.message, true);
  } finally {
    controlState.busy = false;
    refreshControl();
  }
}

async function finishInteractionStop(interaction) {
  const reason = interaction.stopReason || "interaction_cancelled";
  try {
    if (!interaction.executed && controlState.preparedPlanId) {
      const result = await discardCommand(controlState.commandId, controlState.preparedPlanId);
      setReceipt(`discarded: ${result.command_id || result.plan_id || ""}`);
    } else {
      const result = await postSafeStop(reason);
      const receipt = result.terminal_receipt || {};
      const success = receipt.task_success === true ? "true" : "false";
      setReceipt(`safe-stop receipt: task_success=${success}`);
    }
  } catch (error) {
    setReceipt(error.message, true);
  } finally {
    if (controlState.activeInteraction === interaction) {
      controlState.activeInteraction = null;
    }
  }
}

function requestInteractionStop(reason, token = null) {
  const interaction = controlState.activeInteraction;
  if (!interaction) return false;
  if (token !== null && interaction.token !== token) return false;
  interaction.cancelRequested = true;
  interaction.stopReason = reason;
  if (controlState.busy) {
    setReceipt(`cancel pending: ${reason}`);
    return true;
  }
  controlState.busy = true;
  finishInteractionStop(interaction).finally(() => {
    controlState.busy = false;
    refreshControl();
  });
  return true;
}

function syncBehaviorCameraTabs() {
  const tabs = $(".behavior-frame-tabs");
  if (tabs && !tabs.querySelector("button")) {
    const labels = [
      ["head", "head"],
      ["left_wrist", "left wrist"],
      ["right_wrist", "right wrist"],
    ];
    for (const [camera, label] of labels) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.kind = camera;
      button.dataset.camera = camera;
      button.dataset.behaviorCamera = camera;
      button.textContent = label;
      tabs.appendChild(button);
    }
  }
  for (const button of document.querySelectorAll(".behavior-frame-tabs button")) {
    const camera = button.dataset.behaviorCamera
      || button.dataset.camera
      || button.dataset.kind;
    if (!camera) continue;
    button.dataset.behaviorCamera = camera;
    button.dataset.camera = camera;
    button.dataset.kind = camera;
    button.type = "button";
  }
  setButtons("[data-behavior-camera]", controlState.camera, "data-behavior-camera");
  setButtons("[data-camera]", controlState.camera, "data-camera");
  setButtons(".behavior-frame-tabs button", controlState.camera, "data-kind");
}

function handleKeyDown(event) {
  if (event.repeat || isEditableTarget(event.target)) return;
  if (event.key === "Escape") {
    requestInteractionStop("escape");
    return;
  }
  const focusedButton = event.target && event.target.closest
    ? event.target.closest("[data-behavior-action], [data-action]")
    : null;
  if (focusedButton && (event.key === " " || event.key === "Enter")) {
    event.preventDefault();
    const target = focusedButton.dataset.behaviorTarget
      || focusedButton.dataset.target
      || controlState.target;
    const action = focusedButton.dataset.behaviorAction || focusedButton.dataset.action;
    if (action) beginMomentaryAction(keyToken(event), target, action);
    return;
  }
  const mapped = KEY_ACTIONS[event.key];
  if (!mapped) return;
  event.preventDefault();
  beginMomentaryAction(keyToken(event), mapped[0], mapped[1]);
}

function handleKeyUp(event) {
  const focusedButton = event.target && event.target.closest
    ? event.target.closest("[data-behavior-action], [data-action]")
    : null;
  if (focusedButton && (event.key === " " || event.key === "Enter")) {
    event.preventDefault();
    requestInteractionStop("keyup", keyToken(event));
    return;
  }
  const mapped = KEY_ACTIONS[event.key];
  if (!mapped) return;
  event.preventDefault();
  requestInteractionStop("keyup", keyToken(event));
}

function handleActionPointerDown(button, event) {
  if (button.disabled) return;
  const action = button.dataset.behaviorAction || button.dataset.action;
  if (!action) return;
  event.preventDefault();
  button.classList.add("pressed");
  if (typeof button.setPointerCapture === "function" && event.pointerId !== undefined) {
    try {
      button.setPointerCapture(event.pointerId);
    } catch (_error) {
      // Best effort only: release/cancel/blur handlers still safe-stop.
    }
  }
  beginMomentaryAction(pointerToken(event), controlState.target, action);
}

function handleActionPointerRelease(event, reason) {
  event.preventDefault();
  const button = event.currentTarget;
  if (button && button.classList) button.classList.remove("pressed");
  requestInteractionStop(reason, pointerToken(event));
}

function setControlsExpanded(expanded) {
  const framewrap = $("#framewrap") || $(".framewrap.behavior-mode");
  if (!framewrap) return;
  controlState.controlsExpanded = !!expanded;
  framewrap.classList.toggle("controls-collapsed", !controlState.controlsExpanded);
  for (const button of document.querySelectorAll(".controls-toggle")) {
    button.setAttribute("aria-expanded", String(controlState.controlsExpanded));
  }
}

function handleControlsToggle(event) {
  event.preventDefault();
  const nextExpanded = !controlState.controlsExpanded;
  if (!nextExpanded) requestInteractionStop("controls_collapsed");
  setControlsExpanded(nextExpanded);
}

function installControls() {
  if (!controlsRoot()) return;
  setControlsExpanded(true);
  for (const button of document.querySelectorAll(".controls-toggle")) {
    button.addEventListener("click", handleControlsToggle);
  }
  for (const button of document.querySelectorAll("[data-behavior-target], [data-target]")) {
    button.addEventListener("click", () =>
      setTarget(button.dataset.behaviorTarget || button.dataset.target));
  }
  for (const button of document.querySelectorAll("[data-behavior-action], [data-action]")) {
    button.addEventListener("click", () =>
      setAction(button.dataset.behaviorAction || button.dataset.action));
    button.addEventListener("pointerdown", event => handleActionPointerDown(button, event));
    button.addEventListener("pointerup", event => handleActionPointerRelease(event, "pointerup"));
    button.addEventListener("pointercancel", event => handleActionPointerRelease(event, "pointercancel"));
    button.addEventListener("lostpointercapture", event => handleActionPointerRelease(event, "lostpointercapture"));
    button.addEventListener("mouseleave", () => button.classList.remove("pressed"));
  }
  document.addEventListener("click", event => {
    const button = event.target && event.target.closest
      ? event.target.closest(".behavior-frame-tabs button")
      : null;
    if (!button) return;
    const camera = button.dataset.behaviorCamera || button.dataset.camera || button.dataset.kind;
    if (camera) setCamera(camera);
  });
  $("#behaviorPrepare")?.addEventListener("click", prepareSelected);
  $("#behaviorExecute")?.addEventListener("click", executePrepared);
  $("#behaviorDiscard")?.addEventListener("click", discardPrepared);
  $("#behaviorCapture")?.addEventListener("click", captureViews);
  $("#behaviorStop")?.addEventListener("click", () => safeStop("dashboard_safe_stop"));
  window.addEventListener("keydown", handleKeyDown);
  window.addEventListener("keyup", handleKeyUp);
  window.addEventListener("blur", () => requestInteractionStop("window_blur"));
  window.addEventListener("pagehide", () => requestInteractionStop("pagehide"));
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") {
      requestInteractionStop("visibility_hidden");
    }
  });
  syncBehaviorCameraTabs();
  const tabs = $(".behavior-frame-tabs");
  if (tabs && typeof MutationObserver !== "undefined") {
    new MutationObserver(syncBehaviorCameraTabs).observe(tabs, {
      childList: true,
      subtree: true,
    });
  }
  renderActionAvailability();
  refreshControl();
  setInterval(refreshControl, 700);
}

if (typeof globalThis !== "undefined") {
  globalThis.__behaviorDashboardControls = {
    controlState,
    beginMomentaryAction,
    requestInteractionStop,
    handleKeyDown,
    handleKeyUp,
    setControlsExpanded,
  };
}

installControls();
