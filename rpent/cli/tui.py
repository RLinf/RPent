"""Interactive terminal input helpers for the Physical Agent CLI."""
from __future__ import annotations

import contextlib
import importlib
import logging
import queue
import sys
import threading
from collections.abc import Callable

from rpent.utils.logging import get_logger

logger = get_logger("agent.tui")

#: Interactive-mode command tokens (case-insensitive). This module is the single
#: source of truth; the cerebrum queue helpers import ``QUIT_TOKENS`` and
#: ``START_TOKENS`` from here.
QUIT_TOKENS = frozenset({"/quit", "/exit", "/q"})
START_TOKENS = frozenset({"/start"})
HELP_TOKENS = frozenset({"/help", "/h", "help", "?"})
_HELP_TEXT = """Interactive commands:
    /help, /h, help, ? Show this help.
    /start             Restore the built-in default task prompt.
    /quit, /exit, /q   End interactive mode.

At the first prompt, the built-in task is pre-filled — edit it and press Enter,
submit it as-is, or clear it to type your own task ('/start' restores the default).
While the agent runs, type to steer it at the next turn.
"""


def handle_local_command(line: str) -> bool:
    """Handle TUI-local commands; return True when the line was consumed."""
    if line.strip().lower() not in HELP_TOKENS:
        return False
    print(_HELP_TEXT, end="")
    return True


@contextlib.contextmanager
def _route_console_logs_to_current_stdout():
    """Send console logs through the currently patched ``sys.stdout``."""
    swapped: list[tuple[logging.StreamHandler, object]] = []
    package_logger = logging.getLogger("rpent")
    for handler in package_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        if isinstance(handler, logging.StreamHandler):
            swapped.append((handler, handler.stream))
            handler.setStream(sys.stdout)
    try:
        yield
    finally:
        for handler, stream in swapped:
            handler.setStream(stream)


def build_interactive_key_bindings():
    """Return extra key bindings for common Option/Alt-arrow sequences."""
    key_binding = importlib.import_module("prompt_toolkit.key_binding")
    named_commands = importlib.import_module(
        "prompt_toolkit.key_binding.bindings.named_commands"
    )

    bindings = key_binding.KeyBindings()
    backward_word = named_commands.get_by_name("backward-word")
    forward_word = named_commands.get_by_name("forward-word")

    @bindings.add("escape", "left", eager=True)
    @bindings.add("escape", "b", eager=True)
    def _move_word_left(event):
        backward_word(event)

    @bindings.add("escape", "right", eager=True)
    @bindings.add("escape", "f", eager=True)
    def _move_word_right(event):
        forward_word(event)

    return bindings


def start_interactive_reader(
    input_queue: "queue.Queue[str | None]",
    *,
    first_prompt_default: str | None = None,
) -> threading.Thread:
    """Start a prompt-toolkit input UI and forward submitted lines."""
    if not sys.stdin.isatty():
        raise RuntimeError(
            "--interactive requires a TTY; stdin is not interactive."
        )

    try:
        prompt_toolkit = importlib.import_module("prompt_toolkit")
        history = importlib.import_module("prompt_toolkit.history")
        patch_stdout_module = importlib.import_module("prompt_toolkit.patch_stdout")
        styles = importlib.import_module("prompt_toolkit.styles")
    except ImportError as exc:
        raise RuntimeError(
            "--interactive requires prompt-toolkit; install project dependencies first."
        ) from exc

    def _read() -> None:
        session = prompt_toolkit.PromptSession(
            history=history.InMemoryHistory(),
            key_bindings=build_interactive_key_bindings(),
            style=styles.Style.from_dict({"prompt": "ansicyan bold"}),
        )
        pending_default = first_prompt_default
        try:
            with patch_stdout_module.patch_stdout(raw=True):
                with _route_console_logs_to_current_stdout():
                    while True:
                        try:
                            line = session.prompt(
                                [("class:prompt", "you> ")],
                                default=pending_default or "",
                                handle_sigint=False,
                            )
                        except (EOFError, KeyboardInterrupt):
                            break
                        if handle_local_command(line):
                            continue
                        input_queue.put(line)
                        pending_default = None
                        if line.strip().lower() in QUIT_TOKENS:
                            break
        finally:
            input_queue.put(None)

    thread = threading.Thread(target=_read, name="interactive-input", daemon=True)
    thread.start()
    return thread


def next_user_line(input_queue: "queue.Queue[str | None]") -> str | None:
    """Block for the next actionable user line from an interactive input queue.

    Returns the trimmed line, or ``None`` when the session should end (the queue
    yielded ``None`` or a quit token such as ``/quit``). Empty lines are skipped.
    This is a blocking call; async callers should wrap it with
    :func:`asyncio.to_thread`.
    """
    while True:
        line = input_queue.get()
        if line is None:
            return None
        line = line.strip()
        if line.lower() in QUIT_TOKENS:
            return None
        if line:
            return line


def initial_user_message(
    input_queue: "queue.Queue[str | None]", default_message: str
) -> str | None:
    """Block for the first user turn of an interactive session.

    Returns ``default_message`` when the user submits a start token (``/start``),
    the typed text for any other non-empty line (a custom opening prompt), or
    ``None`` when the session should end before it begins (a ``/quit`` token or a
    ``None`` sentinel). Empty lines are skipped. Blocking call.
    """
    while True:
        line = input_queue.get()
        if line is None:
            return None
        line = line.strip()
        if line.lower() in QUIT_TOKENS:
            return None
        if line.lower() in START_TOKENS:
            return default_message
        if line:
            return line


def start_first_prompt_resolver(
    input_queue: "queue.Queue[str | None]", default_message: str
) -> Callable[[], str | None]:
    """Resolve the opening user turn on a background thread.

    Returns a callable that blocks until the first turn is available and returns
    it (``default_message`` on ``/start``, the typed text for a custom prompt, or
    ``None`` if the user quit before starting). Resolving off-thread lets the
    caller boot slow resources (e.g. env/VLA servers) while the user is typing.
    """
    holder: dict[str, str | None] = {}

    def _resolve() -> None:
        holder["message"] = initial_user_message(input_queue, default_message)

    thread = threading.Thread(target=_resolve, name="first-prompt", daemon=True)
    thread.start()

    def _await() -> str | None:
        thread.join()
        return holder.get("message")

    return _await
