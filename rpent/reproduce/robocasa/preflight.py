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

"""Active, deterministic isolation preflight for the pinned Codex runtime."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .deadline_supervisor import CONTRACT_NAME, GATE_NAME, _open_gate
from .executor import (
    FROZEN_RUNTIME_SHA256,
    PINNED_CODEX_SHA256,
    PINNED_CODEX_VERSION,
    ExecutorConfig,
    _planner_command,
    _prepare_rollout_parent,
    _write_private_json,
    sha256_file,
)
from .protocol import cell_for
from .provenance import canonical_sha256

ISOLATION_ATTESTATION_NAME = "isolation-attestation.json"
EXPECTED_TOOL_NAMES = (
    "apply_patch",
    "exec_command",
    "update_plan",
    "view_image",
    "write_stdin",
)
PASS_CHECKS = (
    "scratch_write",
    "root_create_blocked",
    "rpent_read_blocked",
    "runtime_read_blocked",
    "proc_environ_read_blocked",
    "shell_network_blocked",
    "planner_secret_absent",
    "usr_write_blocked",
)


def isolation_attestation_path(config: ExecutorConfig) -> Path:
    return config.results_root / "_preflight" / ISOLATION_ATTESTATION_NAME


def _landlock_abi() -> int:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = int(libc.syscall(444, 0, 0, 1))
    if abi < 1:
        raise RuntimeError("Landlock ABI 1 is unavailable")
    return abi


def _response_base(response_id: str, status: str, output: list[dict]) -> dict:
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": "gpt-5.5",
        "output": output,
        "parallel_tool_calls": False,
        "previous_response_id": None,
        "reasoning": {"effort": "high", "summary": None},
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2,
        },
        "metadata": {},
    }


def _shell_probe(
    *, port: int, marker: Path, repo_root: Path, runtime_probe: Path
) -> str:
    return f"""set +e
scratch_ok=FAIL
root_blocked=FAIL
rpent_blocked=FAIL
runtime_blocked=FAIL
proc_blocked=FAIL
network_blocked=FAIL
secret_absent=FAIL
usr_write_blocked=FAIL

if : > scratch/isolation_preflight 2>/dev/null && test -f scratch/isolation_preflight; then scratch_ok=PASS; fi
if : > root_should_not_exist 2>/dev/null; then :; else root_blocked=PASS; fi
if /usr/bin/head -c 1 {shlex.quote(str(repo_root / "pyproject.toml"))} >/dev/null 2>&1; then :; else rpent_blocked=PASS; fi
if /usr/bin/head -c 1 {shlex.quote(str(runtime_probe))} >/dev/null 2>&1; then :; else runtime_blocked=PASS; fi
if /usr/bin/head -c 1 /proc/self/environ >/dev/null 2>&1; then :; else proc_blocked=PASS; fi
if /usr/bin/curl -fsS --max-time 2 http://127.0.0.1:{port}/shell-probe >/dev/null 2>&1; then :; else network_blocked=PASS; fi
if test -z "${{RLDX_PLANNER_API_KEY+x}}"; then secret_absent=PASS; fi
if echo changed > {shlex.quote(str(marker))} 2>/dev/null; then :; else usr_write_blocked=PASS; fi

echo "scratch_write=$scratch_ok"
echo "root_create_blocked=$root_blocked"
echo "rpent_read_blocked=$rpent_blocked"
echo "runtime_read_blocked=$runtime_blocked"
echo "proc_environ_read_blocked=$proc_blocked"
echo "shell_network_blocked=$network_blocked"
echo "planner_secret_absent=$secret_absent"
echo "usr_write_blocked=$usr_write_blocked"
"""


def _handler(
    state: dict[str, Any],
    *,
    shell_command: str,
    apply_marker: Path | None = None,
    escalation_marker: Path | None = None,
    view_path: Path | None = None,
):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            return

        def do_GET(self):
            if self.path == "/shell-probe":
                state["shell_get_count"] += 1
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            try:
                request = json.loads(self.rfile.read(length))
            except (UnicodeError, ValueError):
                self.send_error(400)
                return
            state["post_count"] += 1
            state["authorization_present"] = bool(
                self.headers.get("Authorization", "").startswith("Bearer ")
            )
            if state["post_count"] == 1:
                self._capture_tools(request.get("tools"))
                if state.get("error") is not None:
                    self._send_final_text("CAPABILITY_MISMATCH")
                else:
                    self._send_function_call(
                        "exec_command",
                        "shell_probe",
                        {"cmd": shell_command, "yield_time_ms": 10000},
                    )
                return
            pending = state.get("pending")
            if isinstance(pending, str):
                state["tool_outputs"][pending] = self._extract_tool_output(
                    request.get("input")
                )
            if state["post_count"] == 2:
                assert apply_marker is not None
                patch = (
                    "*** Begin Patch\n"
                    f"*** Update File: {apply_marker}\n"
                    "@@\n"
                    "-unchanged\n"
                    "+changed\n"
                    "*** End Patch"
                )
                self._send_apply_patch(patch)
            elif state["post_count"] == 3:
                assert view_path is not None
                self._send_function_call(
                    "view_image", "view_image_outside", {"path": str(view_path)}
                )
            elif state["post_count"] == 4:
                assert escalation_marker is not None
                self._send_function_call(
                    "exec_command",
                    "exec_escalation",
                    {
                        "cmd": f"echo changed > {shlex.quote(str(escalation_marker))}",
                        "sandbox_permissions": "require_escalated",
                        "justification": "isolation preflight must reject escalation",
                    },
                )
            elif state["post_count"] == 5:
                self._send_final_text()
            else:
                state["error"] = "unexpected additional Responses request"
                self.send_error(409)

        @staticmethod
        def _extract_tool_output(value):
            if not isinstance(value, list):
                return None
            for item in reversed(value):
                if not isinstance(item, dict):
                    continue
                if item.get("type") in {
                    "function_call_output",
                    "custom_tool_call_output",
                }:
                    output = item.get("output")
                    if isinstance(output, str):
                        return output[-16000:]
            return None

        def _capture_tools(self, tools):
            candidates = [item for item in tools or [] if isinstance(item, dict)]
            names = [item.get("name") for item in candidates]
            if (
                len(names) != len(EXPECTED_TOOL_NAMES)
                or any(not isinstance(name, str) for name in names)
                or tuple(sorted(names)) != EXPECTED_TOOL_NAMES
            ):
                state["error"] = f"unexpected tool capability set: {names!r}"
                return
            by_name = {item["name"]: item for item in candidates}
            if by_name["exec_command"].get("type", "function") != "function":
                state["error"] = "exec_command must use the function tool schema"
                return
            state["tools"] = by_name
            state["tool_names"] = sorted(names)
            state["tool_schema_sha256"] = canonical_sha256(candidates)

        def _send_apply_patch(self, patch: str):
            selected = state["tools"]["apply_patch"]
            if selected.get("type", "function") == "custom":
                self._send_custom_call("apply_patch", "apply_patch_outside", patch)
                return
            properties = (
                selected.get("parameters", {}).get("properties", {})
                if isinstance(selected.get("parameters"), dict)
                else {}
            )
            if "patch" in properties:
                arguments = {"patch": patch}
            elif "input" in properties:
                arguments = {"input": patch}
            else:
                state["error"] = "apply_patch has an unsupported schema"
                self._send_final_text("APPLY_PATCH_SCHEMA_MISMATCH")
                return
            self._send_function_call("apply_patch", "apply_patch_outside", arguments)

        def _send_function_call(self, name: str, label: str, value: dict[str, Any]):
            state["pending"] = label
            arguments = json.dumps(value, separators=(",", ":"))
            sequence = state["post_count"]
            item = {
                "id": f"fc_isolation_preflight_{sequence}",
                "type": "function_call",
                "status": "completed",
                "call_id": f"call_isolation_preflight_{sequence}",
                "name": name,
                "arguments": arguments,
            }
            self._send_call_events(item, "arguments", arguments, custom=False)

        def _send_custom_call(self, name: str, label: str, value: str):
            state["pending"] = label
            sequence = state["post_count"]
            item = {
                "id": f"fc_isolation_preflight_{sequence}",
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": f"call_isolation_preflight_{sequence}",
                "name": name,
                "input": value,
            }
            self._send_call_events(item, "input", value, custom=True)

        def _send_call_events(
            self, item: dict[str, Any], field: str, value: str, *, custom: bool
        ):
            sequence = state["post_count"]
            response_id = f"resp_preflight_{sequence}"
            events = [
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": _response_base(response_id, "in_progress", []),
                },
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": item,
                },
                {
                    "type": (
                        "response.custom_tool_call_input.done"
                        if custom
                        else "response.function_call_arguments.done"
                    ),
                    "sequence_number": 2,
                    "output_index": 0,
                    "item_id": item["id"],
                    field: value,
                },
                {
                    "type": "response.output_item.done",
                    "sequence_number": 3,
                    "output_index": 0,
                    "item": item,
                },
                {
                    "type": "response.completed",
                    "sequence_number": 4,
                    "response": _response_base(response_id, "completed", [item]),
                },
            ]
            self._send_sse(events)

        def _send_final_text(self, text: str = "PREFLIGHT_COMPLETE"):
            state["pending"] = None
            sequence = state["post_count"]
            response_id = f"resp_preflight_{sequence}"
            item = {
                "id": "msg_isolation_preflight_1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
            events = [
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": _response_base(response_id, "in_progress", []),
                },
                {
                    "type": "response.output_item.added",
                    "sequence_number": 1,
                    "output_index": 0,
                    "item": item,
                },
                {
                    "type": "response.content_part.added",
                    "sequence_number": 2,
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "part": item["content"][0],
                },
                {
                    "type": "response.output_text.done",
                    "sequence_number": 3,
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "text": text,
                },
                {
                    "type": "response.content_part.done",
                    "sequence_number": 4,
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "part": item["content"][0],
                },
                {
                    "type": "response.output_item.done",
                    "sequence_number": 5,
                    "output_index": 0,
                    "item": item,
                },
                {
                    "type": "response.completed",
                    "sequence_number": 6,
                    "response": _response_base(response_id, "completed", [item]),
                },
            ]
            self._send_sse(events)

        def _send_sse(self, events):
            payload = (
                b"".join(
                    (
                        "event: "
                        + event["type"]
                        + "\n"
                        + "data: "
                        + json.dumps(event, separators=(",", ":"))
                        + "\n\n"
                    ).encode("utf-8")
                    for event in events
                )
                + b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()

    return Handler


def _private_dummy_key(directory: Path) -> Path:
    if directory.exists() or directory.is_symlink():
        metadata = directory.lstat()
        if (
            directory.is_symlink()
            or not directory.is_dir()
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise RuntimeError("preflight output directory is unsafe")
    else:
        directory.mkdir(parents=True, mode=0o700)
        directory.chmod(0o700)
    descriptor, name = tempfile.mkstemp(prefix=".isolation-key.", dir=directory)
    try:
        os.fchmod(descriptor, 0o600)
        os.write(descriptor, b"rpent-isolation-preflight-key\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return Path(name)


def _world_writable_marker() -> Path:
    path = Path("/usr/local/share") / f"rpent-isolation-{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o666)
    try:
        os.fchmod(descriptor, 0o666)
        os.write(descriptor, b"unchanged\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=5)


def run_isolation_preflight(config: ExecutorConfig) -> dict[str, Any]:
    """Exercise the exact production command against a local fake Responses API."""
    if os.geteuid() != 0:
        raise RuntimeError("isolation preflight requires root")
    state: dict[str, Any] = {
        "post_count": 0,
        "shell_get_count": 0,
        "authorization_present": False,
        "error": None,
        "tool_outputs": {},
    }
    shell_marker = _world_writable_marker()
    apply_marker = _world_writable_marker()
    escalation_marker = _world_writable_marker()
    preflight_root = config.rollout_root / "_preflight"
    _prepare_rollout_parent(preflight_root, config.rollout_root)
    workdir = preflight_root / uuid.uuid4().hex
    workdir.mkdir(mode=0o700)
    os.chown(workdir, 0, 0)
    os.chmod(workdir, 0o700)
    gate_descriptor = _open_gate(workdir / GATE_NAME, create=True)
    os.close(gate_descriptor)
    _write_private_json(
        workdir / CONTRACT_NAME,
        {"schema_version": 1, "protocol": "isolation-preflight"},
    )
    control_inodes = {
        name: (workdir / name).stat().st_ino for name in (GATE_NAME, CONTRACT_NAME)
    }
    key = _private_dummy_key(config.results_root / "_preflight")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state, shell_command=""))
    server.daemon_threads = True
    port = int(server.server_address[1])
    server.RequestHandlerClass = _handler(
        state,
        shell_command=_shell_probe(
            port=port,
            marker=shell_marker,
            repo_root=Path(__file__).resolve().parents[3],
            runtime_probe=config.runtime.driver,
        ),
        apply_marker=apply_marker,
        escalation_marker=escalation_marker,
        view_path=config.runtime.model / "architecture.png",
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    probe_transport = replace(
        config.planner_transport,
        credential_file=key,
        request_base_url=f"http://127.0.0.1:{port}/v1",
    )
    probe_config = replace(
        config,
        planner_transport=probe_transport,
    )
    command = _planner_command(
        probe_config,
        workdir=workdir,
        run_id="isolation-preflight",
        cell=cell_for("atomic", "OpenDrawer", 1),
    )
    process: subprocess.Popen[str] | None = None
    deadline_controls_protected = False
    try:
        process = subprocess.Popen(
            command,
            cwd=config.runtime.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=90)
        except subprocess.TimeoutExpired as exc:
            _terminate(process)
            raise RuntimeError("isolation preflight timed out") from exc
    finally:
        if process is not None:
            _terminate(process)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        key.unlink(missing_ok=True)
        marker_bytes = {
            "shell": shell_marker.read_bytes() if shell_marker.exists() else b"",
            "apply_patch": (
                apply_marker.read_bytes() if apply_marker.exists() else b""
            ),
            "escalation": (
                escalation_marker.read_bytes() if escalation_marker.exists() else b""
            ),
        }
        shell_marker.unlink(missing_ok=True)
        apply_marker.unlink(missing_ok=True)
        escalation_marker.unlink(missing_ok=True)
        try:
            deadline_controls_protected = all(
                stat.S_ISREG((metadata := (workdir / name).lstat()).st_mode)
                and metadata.st_ino == control_inodes[name]
                and metadata.st_uid == 0
                and metadata.st_gid == 0
                and stat.S_IMODE(metadata.st_mode) == 0o600
                and metadata.st_nlink == 1
                for name in (GATE_NAME, CONTRACT_NAME)
            )
        except OSError:
            deadline_controls_protected = False
        if workdir.exists():
            shutil.rmtree(workdir)
    tool_outputs = state.get("tool_outputs", {})
    output = tool_outputs.get("shell_probe")
    checks = {
        name: isinstance(output, str) and f"{name}=PASS" in output
        for name in PASS_CHECKS
    }
    checks.update(
        {
            "apply_patch_outside_blocked": (
                marker_bytes["apply_patch"] == b"unchanged\n"
                and bool(tool_outputs.get("apply_patch_outside"))
            ),
            "view_image_outside_blocked": bool(tool_outputs.get("view_image_outside")),
            "exec_escalation_rejected": (
                marker_bytes["escalation"] == b"unchanged\n"
                and bool(tool_outputs.get("exec_escalation"))
            ),
            "deadline_controls_root_only": deadline_controls_protected,
        }
    )
    try:
        events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    except ValueError as exc:
        raise RuntimeError("isolation preflight produced invalid JSONL") from exc
    event_types = [event.get("type") for event in events if isinstance(event, dict)]
    observed_item_types = sorted(
        {
            item["type"]
            for event in events
            if isinstance(event, dict)
            and isinstance((item := event.get("item")), dict)
            and isinstance(item.get("type"), str)
        }
    )
    checks["view_image_outside_blocked"] = (
        checks["view_image_outside_blocked"] and "image_view" not in observed_item_types
    )
    problems = []
    assert process is not None
    if process.returncode != 0:
        problems.append(f"Codex exited {process.returncode}")
    if state["post_count"] != 5 or state["shell_get_count"] != 0:
        problems.append("Responses or generated-shell network counts differ")
    if not state["authorization_present"]:
        problems.append("Codex client omitted its API authorization")
    if state.get("error") is not None:
        problems.append(str(state["error"]))
    if tuple(state.get("tool_names", ())) != EXPECTED_TOOL_NAMES:
        problems.append("Codex direct tool capabilities differ")
    if not all(checks.values()) or (isinstance(output, str) and "=FAIL" in output):
        problems.append("one or more filesystem/network/secret checks failed")
    if any(value != b"unchanged\n" for value in marker_bytes.values()):
        problems.append("one or more /usr write markers changed")
    allowed_events = {
        "thread.started",
        "turn.started",
        "item.started",
        "item.updated",
        "item.completed",
        "turn.completed",
    }
    if not event_types or set(event_types) - allowed_events:
        problems.append("Codex emitted an unexpected JSONL event type")
    if event_types.count("turn.completed") != 1:
        problems.append("Codex did not complete exactly one preflight turn")
    if problems:
        stderr_digest = hashlib.sha256(stderr.encode("utf-8")).hexdigest()
        raise RuntimeError(
            "isolation preflight failed: "
            + "; ".join(problems)
            + f"; stderr_sha256={stderr_digest}"
        )
    codex_arguments = command[command.index(str(config.codex_bin)) + 1 : -1]
    provider_base_prefix = (
        f"model_providers.{config.planner_transport.provider_id}.base_url="
    )
    normalized_arguments = [
        (
            f"{provider_base_prefix}<loopback-preflight>"
            if item.startswith(provider_base_prefix)
            else item
        )
        for item in codex_arguments
    ]
    attestation: dict[str, Any] = {
        "schema_version": 2,
        "passed": True,
        "codex": {
            "version": PINNED_CODEX_VERSION,
            "sha256": PINNED_CODEX_SHA256,
        },
        "launcher_sha256": FROZEN_RUNTIME_SHA256["isolation_launcher"],
        "sandbox_adapter_sha256": sha256_file(Path(__file__).with_name("sandbox.py")),
        "profile": {
            "permission_profile": "rpent_outer_landlock",
            "filesystem_authority": (
                "outer_rldx_landlock_abi1_uid_gid_fixed_mailboxes"
            ),
            "codex_filesystem_view": "full_write_fast_path_no_inner_fs_sandbox",
            "command_network_authority": ("codex_0.147_restricted_network_seccomp"),
            "arguments_sha256": canonical_sha256(normalized_arguments),
        },
        "planner_transport": config.planner_transport.manifest_identity(),
        "kernel_release": os.uname().release,
        "landlock_abi": _landlock_abi(),
        "tool_names": list(EXPECTED_TOOL_NAMES),
        "tool_schema_sha256": state["tool_schema_sha256"],
        "observed_item_types": observed_item_types,
        "checks": checks,
    }
    attestation["attestation_sha256"] = canonical_sha256(attestation)
    _write_private_json(isolation_attestation_path(config), attestation)
    return attestation
