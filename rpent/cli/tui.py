"""Interactive terminal input helpers for the Physical Agent CLI."""
from __future__ import annotations

import contextlib
import importlib
import logging
import queue
import sys
import threading

from rpent.utils.logging import get_logger

logger = get_logger("agent.tui")

_QUIT_TOKENS = frozenset({"/quit", "/exit", "/q"})
_HELP_TOKENS = frozenset({"/help", "/h", "help", "?"})
_HELP_TEXT = """Interactive commands:
    /help, /h, help, ? Show this help.
    /start             Run the built-in default task prompt.
    /quit, /exit, /q   End interactive mode.

At the first prompt, type a task to run it (or type '/start' to use the default one).
While the agent runs, type to steer it at the next turn.
"""


def handle_local_command(line: str) -> bool:
    """Handle TUI-local commands; return True when the line was consumed."""
    if line.strip().lower() not in _HELP_TOKENS:
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
        try:
            with patch_stdout_module.patch_stdout(raw=True):
                with _route_console_logs_to_current_stdout():
                    while True:
                        try:
                            line = session.prompt(
                                [("class:prompt", "you> ")],
                                handle_sigint=False,
                            )
                        except (EOFError, KeyboardInterrupt):
                            break
                        if handle_local_command(line):
                            continue
                        input_queue.put(line)
                        if line.strip().lower() in _QUIT_TOKENS:
                            break
        finally:
            input_queue.put(None)

    thread = threading.Thread(target=_read, name="interactive-input", daemon=True)
    thread.start()
    return thread
