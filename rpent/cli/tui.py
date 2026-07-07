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


def _start_stdin_reader(input_queue: "queue.Queue[str | None]") -> threading.Thread:
    """Forward each stdin line to the agent input queue."""

    def _read() -> None:
        try:
            for line in sys.stdin:
                input_queue.put(line.rstrip("\n"))
        except Exception:
            pass
        finally:
            input_queue.put(None)

    thread = threading.Thread(target=_read, name="stdin-reader", daemon=True)
    thread.start()
    return thread


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
        logger.warning(
            "interactive prompt UI needs a TTY; falling back to plain stdin."
        )
        return _start_stdin_reader(input_queue)

    try:
        prompt_toolkit = importlib.import_module("prompt_toolkit")
        history = importlib.import_module("prompt_toolkit.history")
        patch_stdout_module = importlib.import_module("prompt_toolkit.patch_stdout")
        styles = importlib.import_module("prompt_toolkit.styles")
    except ImportError:
        logger.warning(
            "interactive prompt UI needs prompt-toolkit; falling back to plain stdin."
        )
        return _start_stdin_reader(input_queue)

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
                        input_queue.put(line)
                        if line.strip().lower() in _QUIT_TOKENS:
                            break
        finally:
            input_queue.put(None)

    thread = threading.Thread(target=_read, name="interactive-input", daemon=True)
    thread.start()
    return thread
