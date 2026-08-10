function formatValue(value) {
  if (value == null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function createTaskCommandSuggester() {
  let command = "";
  let fields = [];

  function configure(config) {
    command = typeof config?.command === "string" ? config.command : "";
    if (!Array.isArray(config?.fields)) {
      fields = [];
      return;
    }
    fields = config.fields.map(field => ({
      ...field,
      suggestions: Array.isArray(field?.suggestions)
        ? [...new Set(field.suggestions.filter(item => typeof item === "string"))]
        : [],
    }));
  }

  function suggestionContext(value, selectionStart, selectionEnd) {
    if (
      !command
      || value.includes("\n")
      || selectionStart !== selectionEnd
      || selectionEnd !== value.length
      || !value.startsWith(command)
    ) return null;

    const suffix = value.slice(command.length);
    if (!/^[\t ]/.test(suffix)) return null;
    const body = suffix.trimStart();
    const endsInSpace = /[\t ]$/.test(value);
    const tokens = body ? body.split(/[\t ]+/) : [];
    const fieldIndex = endsInSpace ? tokens.length : Math.max(tokens.length - 1, 0);
    const field = fields[fieldIndex];
    const suggestions = field?.suggestions || [];
    if (!suggestions.length) return null;

    const prefix = endsInSpace || !tokens.length ? "" : tokens[tokens.length - 1];
    return {
      leading: value.slice(0, value.length - prefix.length),
      prefix,
      suggestions,
    };
  }

  function suggest(value, selectionStart, selectionEnd) {
    const context = suggestionContext(value, selectionStart, selectionEnd);
    if (!context) return [];
    return context.suggestions.filter(item => item.startsWith(context.prefix));
  }

  function select(value, selectionStart, selectionEnd, item) {
    const context = suggestionContext(value, selectionStart, selectionEnd);
    if (
      !context
      || !context.suggestions.includes(item)
      || !item.startsWith(context.prefix)
    ) return null;
    const selected = `${context.leading}${item} `;
    return { value: selected, cursor: selected.length };
  }

  return { configure, select, suggest };
}

export function createInteractionController({ copy, select, onRefresh }) {
  const taskCommandSuggester = createTaskCommandSuggester();
  const state = {
    snapshot: null,
    submissionInFlight: false,
    interruptInFlight: false,
    withdrawalsInFlight: new Set(),
    requestError: null,
    notice: null,
    noticeTimer: null,
    pendingRenderKey: null,
    suggestionRenderKey: null,
    taskUsage: "",
  };

  function errorText(error) {
    if (error == null || error === "") return "";
    if (typeof error === "string") return error;
    if (typeof error === "object") {
      if (typeof error.message === "string") return error.message;
      if (typeof error.detail === "string") return error.detail;
      if (typeof error.error === "string") return error.error;
    }
    return formatValue(error);
  }

  function controlFeedbackText(feedback) {
    const text = errorText(feedback);
    const taskSelectedPrefix = "Task selected: ";
    if (text.startsWith(taskSelectedPrefix)) {
      return copy.taskSelectedFeedback(text.slice(taskSelectedPrefix.length));
    }
    const taskRunStarting = /^TaskRun (\d+) starting(?:…|\.\.\.)$/.exec(text);
    if (taskRunStarting) {
      return copy.taskRunStartingFeedback(taskRunStarting[1]);
    }
    return text;
  }

  async function responseErrorText(response) {
    let body = "";
    try {
      body = await response.text();
    } catch {}
    if (!body) return `${response.status} ${response.statusText}`.trim();
    try {
      const parsed = JSON.parse(body);
      return errorText(parsed.detail ?? parsed.error ?? parsed.message ?? parsed);
    } catch {
      return body;
    }
  }

  async function requestJSON(url, { method = "GET", body } = {}) {
    const options = { method };
    if (body !== undefined) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }
    const response = await fetch(url, options);
    if (!response.ok) throw new Error(await responseErrorText(response));
    if (response.status === 204) return null;
    const text = await response.text();
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }

  function taskTargetLabel(target) {
    if (!target) return "";
    if (typeof target.label === "string") return target.label;
    return formatValue(target.parameters || target);
  }

  function renderPendingMessages(messages) {
    const visible = (Array.isArray(messages) ? messages : []).filter(message =>
      ["pending", "sending", "failed", "unsent"].includes(message?.status)
    );
    const renderKey = JSON.stringify(visible.map(message => [
      message.message_id,
      message.text,
      message.status,
      message.error,
      state.withdrawalsInFlight.has(message.message_id),
    ]));
    if (renderKey === state.pendingRenderKey) return;
    state.pendingRenderKey = renderKey;

    const area = select("#pendingArea");
    area.hidden = visible.length === 0;
    if (!visible.length) {
      select("#pendingMessages").replaceChildren();
      select("#pendingCount").textContent = "";
      return;
    }

    const hasActive = visible.some(message =>
      message.status === "pending" || message.status === "sending"
    );
    select("#pendingHeading").textContent = hasActive
      ? copy.pendingHeading
      : copy.inactiveMessagesHeading;
    select("#pendingCount").textContent = String(visible.length);

    const elements = visible.map(message => {
      const item = document.createElement("div");
      item.className = "pending-message";
      item.dataset.status = message.status;
      if (message.message_id != null) item.dataset.messageId = message.message_id;

      const text = document.createElement("div");
      text.className = "pending-message-text";
      text.textContent = message.text || "";

      const meta = document.createElement("div");
      meta.className = "pending-message-meta";
      const status = document.createElement("span");
      status.className = "pending-message-status";
      status.textContent = copy.messageStates[message.status] || message.status;
      meta.appendChild(status);
      const error = errorText(message.error);
      if (error) meta.append(document.createTextNode(` · ${error}`));

      item.append(text, meta);
      if (message.status === "pending" || message.status === "sending") {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "withdraw-message";
        button.textContent = copy.withdraw;
        button.setAttribute("aria-label", copy.withdrawMessage(message.text || ""));
        button.disabled =
          message.status !== "pending"
          || state.withdrawalsInFlight.has(message.message_id);
        if (message.status === "pending") {
          button.addEventListener("click", () => withdrawMessage(message.message_id));
        }
        item.appendChild(button);
      }
      return item;
    });
    select("#pendingMessages").replaceChildren(...elements);
  }

  function renderTaskSuggestions() {
    const area = select("#suiteSuggestions");
    const input = select("#chatInput");
    const suggestions = input.disabled
      ? []
      : taskCommandSuggester.suggest(
        input.value,
        input.selectionStart,
        input.selectionEnd,
      );
    const renderKey = JSON.stringify(suggestions);

    area.hidden = suggestions.length === 0;
    if (renderKey === state.suggestionRenderKey) return;
    state.suggestionRenderKey = renderKey;
    if (!suggestions.length) {
      area.replaceChildren();
      return;
    }

    const candidates = suggestions.map(suggestion => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "suite-suggestion";
      button.setAttribute("role", "option");
      button.textContent = suggestion;
      button.addEventListener("click", () => {
        const selection = taskCommandSuggester.select(
          input.value,
          input.selectionStart,
          input.selectionEnd,
          suggestion,
        );
        if (!selection) return;
        input.value = selection.value;
        input.setSelectionRange(selection.cursor, selection.cursor);
        input.focus();
        renderTaskSuggestions();
      });
      return button;
    });
    area.replaceChildren(...candidates);
  }

  function render() {
    const session = state.snapshot;
    const interaction = session?.interaction;
    const composer = select("#composer");
    const input = select("#chatInput");

    if (!session) {
      composer.hidden = true;
      input.disabled = true;
      renderTaskSuggestions();
      return;
    }

    composer.hidden = false;
    const activity = interaction.planner_activity;
    const mode = interaction.input_mode;
    const fatal = session.session_state === "fatal";
    const commandContext = mode === "command_only"
      || session.session_state !== "running";
    const inputEnabled = !!interaction.session_id
      && mode !== "disabled";
    composer.dataset.inputMode = mode;
    input.placeholder = commandContext
      ? copy.commandPlaceholder(state.taskUsage)
      : copy.composerPlaceholder;
    select("#chatHint").textContent = commandContext
      ? copy.commandKeys(state.taskUsage)
      : copy.composerKeys;
    input.disabled = !inputEnabled || state.submissionInFlight;
    input.setAttribute("aria-busy", String(
      state.submissionInFlight || state.interruptInFlight
    ));

    const status = select("#interactionStatus");
    status.className = "composer-status";
    const backendError = errorText(interaction.last_error);
    const controlError = errorText(session.control_error);
    const feedback = session.control_feedback.map(controlFeedbackText).filter(Boolean);
    if (state.requestError) {
      status.textContent = state.requestError;
      status.classList.add("is-error");
    } else if (controlError) {
      status.textContent = controlError;
      status.classList.add("is-error");
    } else if (backendError) {
      status.textContent = copy.interactionError(backendError);
      status.classList.add("is-error");
    } else if (fatal) {
      status.textContent = copy.sessionFatal;
      status.classList.add("is-error");
    } else if (state.submissionInFlight) {
      status.textContent = copy.submittingMessage;
      status.classList.add("is-busy");
    } else if (interaction.interrupt_requested || state.interruptInFlight) {
      status.textContent = copy.interruptRequested;
      status.classList.add("is-busy");
    } else if (state.notice) {
      status.textContent = state.notice;
      status.classList.add("is-ready");
    } else if (feedback.length) {
      status.textContent = feedback.join("\n");
      status.classList.add("is-ready", "is-control-feedback");
    } else if (session.session_state === "starting_shared_services") {
      status.textContent = copy.sessionStarting;
      status.classList.add("is-busy");
    } else if (session.session_state === "task_starting") {
      status.textContent = copy.taskStarting;
      status.classList.add("is-busy");
    } else if (session.session_state === "switch_pending") {
      status.textContent = copy.taskSwitchPending(
        taskTargetLabel(session.pending_task)
      );
      status.classList.add("is-busy");
    } else if (mode === "command_only") {
      status.textContent = copy.commandReady(state.taskUsage);
      status.classList.add("is-ready");
    } else if (activity === "starting") {
      status.textContent = copy.interactionStarting;
      status.classList.add("is-busy");
    } else if (activity === "busy") {
      status.textContent = copy.interactionBusy;
      status.classList.add("is-busy");
    } else if (inputEnabled) {
      status.textContent = copy.interactionReady;
      status.classList.add("is-ready");
    } else {
      status.textContent = copy.interactionUnavailable;
    }

    renderPendingMessages(interaction.messages);
    renderTaskSuggestions();
  }

  function setNotice(message) {
    state.notice = message;
    if (state.noticeTimer) clearTimeout(state.noticeTimer);
    state.noticeTimer = setTimeout(() => {
      state.notice = null;
      state.noticeTimer = null;
      render();
    }, 4500);
  }

  function reset() {
    if (state.noticeTimer) clearTimeout(state.noticeTimer);
    state.snapshot = null;
    state.submissionInFlight = false;
    state.interruptInFlight = false;
    state.withdrawalsInFlight.clear();
    state.requestError = null;
    state.notice = null;
    state.noticeTimer = null;
    state.pendingRenderKey = null;
    state.suggestionRenderKey = null;
    select("#composer").hidden = true;
    select("#chatInput").disabled = true;
    select("#chatInput").value = "";
    select("#pendingArea").hidden = true;
    select("#pendingMessages").replaceChildren();
    select("#suiteSuggestions").hidden = true;
    select("#suiteSuggestions").replaceChildren();
    select("#interactionStatus").textContent = "";
  }

  function applySnapshot(snapshot) {
    const previous = state.snapshot?.interaction;
    state.snapshot = snapshot;
    const interaction = snapshot.interaction;

    if (
      previous?.interrupt_requested
      && !interaction.interrupt_requested
      && !errorText(interaction.last_error)
    ) {
      setNotice(copy.interruptSucceeded);
    } else if (interaction.interrupt_requested) {
      state.notice = null;
      if (state.noticeTimer) {
        clearTimeout(state.noticeTimer);
        state.noticeTimer = null;
      }
    }
    render();
  }

  function configureTaskCommand(config) {
    taskCommandSuggester.configure(config);
    state.taskUsage = typeof config?.usage === "string" ? config.usage : "";
    state.suggestionRenderKey = null;
    render();
  }

  async function submitMessage() {
    const interaction = state.snapshot?.interaction;
    const input = select("#chatInput");
    const draft = input.value;
    const text = draft.trim();
    if (
      !text
      || !interaction?.session_id
      || interaction.input_mode === "disabled"
      || state.submissionInFlight
    ) return;

    state.submissionInFlight = true;
    state.requestError = null;
    render();
    try {
      await requestJSON(
        `/api/sessions/${encodeURIComponent(interaction.session_id)}/messages`,
        { method: "POST", body: { text } },
      );
      if (input.value === draft) input.value = "";
      onRefresh();
    } catch (error) {
      state.requestError = copy.submitFailed(
        errorText(error) || copy.unknownRequestError
      );
    } finally {
      state.submissionInFlight = false;
      render();
    }
  }

  async function withdrawMessage(messageId) {
    const interaction = state.snapshot?.interaction;
    if (
      !interaction?.session_id
      || !messageId
      || state.withdrawalsInFlight.has(messageId)
    ) return;

    state.withdrawalsInFlight.add(messageId);
    state.requestError = null;
    render();
    try {
      await requestJSON(
        `/api/sessions/${encodeURIComponent(interaction.session_id)}/messages/${encodeURIComponent(messageId)}`,
        { method: "DELETE" },
      );
      onRefresh();
    } catch (error) {
      state.requestError = copy.withdrawFailed(
        errorText(error) || copy.unknownRequestError
      );
    } finally {
      state.withdrawalsInFlight.delete(messageId);
      render();
    }
  }

  async function requestInterrupt() {
    const interaction = state.snapshot?.interaction;
    if (
      !interaction?.session_id
      || interaction.planner_activity !== "busy"
      || interaction.interrupt_requested
      || state.interruptInFlight
    ) return;

    state.interruptInFlight = true;
    state.requestError = null;
    render();
    try {
      const result = await requestJSON(
        `/api/sessions/${encodeURIComponent(interaction.session_id)}/interrupt`,
        { method: "POST" },
      );
      if (
        result?.interrupt_requested
        && state.snapshot?.interaction.session_id === interaction.session_id
      ) {
        applySnapshot({
          ...state.snapshot,
          interaction: {
            ...state.snapshot.interaction,
            interrupt_requested: true,
          },
        });
      }
      onRefresh();
    } catch (error) {
      state.requestError = copy.interruptFailed(
        errorText(error) || copy.unknownRequestError
      );
    } finally {
      state.interruptInFlight = false;
      render();
    }
  }

  function handleKeydown(event) {
    if (event.key === "Escape") {
      event.preventDefault();
      requestInterrupt();
      return;
    }
    if (
      event.key === "Enter"
      && !event.shiftKey
      && !event.isComposing
      && event.keyCode !== 229
    ) {
      event.preventDefault();
      submitMessage();
    }
  }

  const input = select("#chatInput");
  input.addEventListener("keydown", handleKeydown);
  input.addEventListener("input", renderTaskSuggestions);
  input.addEventListener("click", renderTaskSuggestions);
  input.addEventListener("select", renderTaskSuggestions);
  return {
    applySnapshot,
    configureTaskCommand,
    reset,
  };
}
