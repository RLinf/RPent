# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Operator replies must not be consumed as planner steering, or vice versa."""

import threading

import pytest

from rpent.tools.human_in_the_loop import HumanInTheLoopInput
from rpent.tools.toolkit import ToolCancelled


def test_operator_replies_are_request_scoped_and_do_not_consume_steering(monkeypatch):
    broker = HumanInTheLoopInput(interactive=True)
    ready = threading.Event()
    result = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: ready.set())
    worker = threading.Thread(
        target=lambda: result.append(broker("restore scene", lambda: None))
    )
    worker.start()
    assert ready.wait(2)
    request_id = broker._pending[0]
    assert not broker.route_line("success")
    assert broker.route_line("/operator stale done")
    assert not result
    assert broker.route_line(f"/operator {request_id} done")
    worker.join(2)
    assert result == ["done"] and not worker.is_alive()
    assert broker.route_line(f"/operator {request_id} success")
    assert broker._pending is None


def test_operator_eof_and_cancellation_release_pending_requests(monkeypatch):
    broker = HumanInTheLoopInput(interactive=True)
    ready = threading.Event()
    result = []
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: ready.set())
    worker = threading.Thread(
        target=lambda: result.append(broker("verdict", lambda: None))
    )
    worker.start()
    assert ready.wait(2)
    broker.close()
    worker.join(2)
    assert result == [None] and broker._pending is None
    assert broker("late request", lambda: None) is None

    broker = HumanInTheLoopInput(interactive=True)

    def cancelled():
        raise ToolCancelled("cancelled")

    with pytest.raises(ToolCancelled):
        broker("verdict", cancelled)
    assert broker._pending is None


def test_interactive_reader_routes_operator_input_without_stealing_steering(
    monkeypatch,
):
    import contextlib
    import queue
    from types import SimpleNamespace

    from rpent.cli import tui

    lines = iter(["initial task", "/operator request done", "move more slowly"])
    routed = []
    closed = []

    class Session:
        def __init__(self, **kwargs):
            pass

        def prompt(self, *args, **kwargs):
            try:
                return next(lines)
            except StopIteration:
                raise EOFError from None

    def route(line):
        if line.startswith("/operator "):
            routed.append(line)
            return True
        return False

    monkeypatch.setattr(tui, "PromptSession", Session)
    monkeypatch.setattr(
        tui.sys, "stdin", SimpleNamespace(isatty=lambda: True, fileno=lambda: 0)
    )
    monkeypatch.setattr(tui, "_restore_tty_on_exit", lambda fd: None)
    monkeypatch.setattr(tui, "patch_stdout", lambda **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(
        tui, "_route_console_logs_to_current_stdout", contextlib.nullcontext
    )
    messages = queue.Queue()
    thread = tui.start_interactive_reader(
        messages, line_handler=route, on_close=lambda: closed.append(True)
    )
    thread.join(2)
    assert not thread.is_alive()
    assert [messages.get_nowait() for _ in range(3)] == [
        "initial task",
        "move more slowly",
        None,
    ]
    assert routed == ["/operator request done"] and closed == [True]


def test_success_is_control_and_never_steering():
    broker = HumanInTheLoopInput(interactive=True)
    calls = []
    broker.bind_verdict(lambda verdict: calls.append(verdict) or True)
    assert not broker.route_line(" success ")
    assert broker.route_line("/success")
    assert len(calls) == 1
    assert not broker.route_line("the grasp was a success")
    broker.bind_verdict(None)
    assert not broker.route_line("success")
    assert len(calls) == 1


def test_only_slash_verdicts_are_control_commands():
    broker = HumanInTheLoopInput(interactive=True)
    verdicts = []
    broker.bind_verdict(lambda verdict: verdicts.append(verdict) or True)
    for text in (
        "success",
        "failure",
        "抓取成功了",
        "success 请继续",
        "/success later",
    ):
        assert not broker.route_line(text)
    assert broker.route_line("/success")
    assert broker.route_line(" /failure ")
    assert verdicts == ["success", "failure"]


@pytest.mark.parametrize(
    "kind,command,wrong,answer",
    [
        ("reset", "/done", "/continue", "done"),
        ("verdict", "/continue", "/done", "continue"),
    ],
)
def test_shortcuts_only_answer_matching_active_request(
    monkeypatch, kind, command, wrong, answer
):
    broker = HumanInTheLoopInput(interactive=True)
    ready = threading.Event()
    values = []
    monkeypatch.setattr("builtins.print", lambda *a, **kw: ready.set())
    assert broker.route_line(command)  # no buffering before a request
    ready.clear()
    thread = threading.Thread(
        target=lambda: values.append(broker.request("test", lambda: None, kind=kind))
    )
    thread.start()
    assert ready.wait(2)
    assert broker.route_line(wrong)
    assert not values and broker._pending is not None
    assert broker.route_line(command)
    thread.join(2)
    assert not thread.is_alive() and values == [answer]
    assert broker.route_line(command)  # duplicate cannot authorize another operation
    assert not broker.route_line(answer)


def test_abort_is_control_but_bare_words_are_chat():
    broker = HumanInTheLoopInput(interactive=True)
    calls = []
    broker.bind_verdict(lambda verdict: calls.append(verdict) or True)
    assert broker.route_line("/abort")
    assert calls == ["abort"]
    for word in ("done", "continue", "abort", "success", "failure"):
        assert not broker.route_line(word)


def test_help_only_shows_human_commands_when_enabled(capsys):
    from rpent.cli.tui import handle_local_command

    assert handle_local_command("/help")
    normal = capsys.readouterr().out
    assert "/quit" in normal and "/success" not in normal
    assert handle_local_command("/help", extra_help=HumanInTheLoopInput.help_text)
    active = capsys.readouterr().out
    for command in ("/done", "/continue", "/success", "/failure", "/abort"):
        assert command in active
