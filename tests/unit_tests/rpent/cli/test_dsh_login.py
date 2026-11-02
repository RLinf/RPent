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

"""Device login output and subprocess ownership, without online authorization."""

import subprocess
import sys
import types

import pytest

from rpent.cli import dsh_login


def _runtime(tmp_path, monkeypatch, body):
    script = tmp_path / "fake_runtime.py"
    script.write_text(body)
    module = types.ModuleType("deepseek_harness_runtime")
    module.resolve_bundled_launch_args = lambda: (sys.executable, str(script))
    monkeypatch.setitem(sys.modules, "deepseek_harness_runtime", module)
    processes = []
    original_popen = subprocess.Popen

    def capture(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(dsh_login.subprocess, "Popen", capture)
    return processes


def test_login_reports_device_code_and_reaps_authorized_runtime(
    tmp_path, monkeypatch, capsys
):
    processes = _runtime(
        tmp_path,
        monkeypatch,
        "import sys,time\n"
        "print('private-provider-response', file=sys.stderr, flush=True)\n"
        'print(\'RPENT_DSH_LOGIN {"event":"device_code",'
        '"url":"https://auth.openai.com/codex/device",'
        '"code":"TEST-CODE"}\', file=sys.stderr, flush=True)\n'
        'print(\'RPENT_DSH_LOGIN {"event":"settled",'
        '"status":"authorized"}\', file=sys.stderr, flush=True)\n'
        "time.sleep(30)\n",
    )
    assert dsh_login.login(tmp_path / "home", timeout_s=5)
    output = capsys.readouterr()
    assert "https://auth.openai.com/codex/device" in output.out
    assert "TEST-CODE" in output.out
    assert "private-provider-response" not in output.out + output.err
    assert processes[0].poll() is not None
    assert (tmp_path / "home").stat().st_mode & 0o077 == 0


def test_login_timeout_reaps_waiting_runtime(tmp_path, monkeypatch):
    processes = _runtime(tmp_path, monkeypatch, "import time\ntime.sleep(30)\n")
    with pytest.raises(TimeoutError, match="timed out"):
        dsh_login.login(tmp_path / "home", timeout_s=0.1)
    assert processes[0].poll() is not None


def test_login_eof_is_not_success(tmp_path, monkeypatch, capsys):
    processes = _runtime(
        tmp_path,
        monkeypatch,
        "import sys\nprint('sensitive failure body', file=sys.stderr)\n",
    )
    with pytest.raises(RuntimeError, match="before authorization"):
        dsh_login.login(tmp_path / "home", timeout_s=5)
    output = capsys.readouterr()
    assert "sensitive failure body" not in output.out + output.err
    assert processes[0].poll() is not None


def test_login_requires_explicit_home(monkeypatch):
    monkeypatch.delenv("DSH_HOME", raising=False)
    monkeypatch.setattr(sys, "argv", ["rpent-dsh-login"])
    with pytest.raises(SystemExit) as error:
        dsh_login.main()
    assert error.value.code == 2
