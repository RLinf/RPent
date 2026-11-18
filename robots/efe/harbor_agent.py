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

"""Harbor 0.23 installed agent for an EFE-compatible RPent trial."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext, ModelUsage

from robots.efe.constants import SANDBOX_CHOICES, is_budget_stop

# Harbor kills the agent process when its resolved ``agent_timeout_sec``
# expires. Keep connection and planner budgets inside that limit so the generic
# CLI can write its transcript and run finalizer before Harbor kills the agent:
#   connection + planner = agent_timeout - 2 * margin
#   exec wrapper   = agent_timeout - margin
# The margin covers the planner's 15 s worker join, an in-flight tool call, and
# finalization.
_TIMEOUT_MARGIN_S = 30
_DEFAULT_CONNECT_TIMEOUT_S = 60
# Fallback planner budget for trials whose Harbor timeout cannot be detected.
_DEFAULT_TIMEOUT_S = 3500


class RPentEfeAgent(BaseInstalledAgent):
    """Build and install RPent inside Harbor's isolated agent container."""

    def __init__(
        self,
        logs_dir: Path,
        *,
        source_root: str | None = None,
        workdir: str = "/app",
        timeout_s: int | None = None,
        agent_timeout_s: int | None = None,
        connect_timeout_s: float = _DEFAULT_CONNECT_TIMEOUT_S,
        max_turns: int = 200,
        reasoning_effort: str = "none",
        memory_source: str | None = None,
        skills: bool = True,
        sandbox: str = "read-only",
        **kwargs,
    ) -> None:
        """Accept RPent settings through Harbor's ``--agent-kwarg`` options."""
        super().__init__(logs_dir=logs_dir, **kwargs)
        if not self.model_name:
            raise ValueError("Supply a model with Harbor's --model option.")
        self.source_root = (
            Path(source_root).expanduser().resolve()
            if source_root
            else Path(__file__).resolve().parents[2]
        )
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be positive.")
        if agent_timeout_s is not None and agent_timeout_s <= 0:
            raise ValueError("agent_timeout_s must be positive.")
        if not math.isfinite(connect_timeout_s) or connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be positive.")
        if max_turns <= 0:
            raise ValueError("timeout_s and max_turns must be positive.")
        if sandbox not in SANDBOX_CHOICES:
            raise ValueError(f"Unsupported Codex sandbox: {sandbox}")
        self.memory_source = (
            Path(memory_source).expanduser().resolve() if memory_source else None
        )
        if self.memory_source is not None and not self.memory_source.is_dir():
            raise FileNotFoundError(
                f"The memory corpus is not a directory: {self.memory_source}"
            )
        self.workdir = workdir
        # Harbor resolves the container budget from ``--agent-timeout-sec`` or
        # the task's ``[agent] timeout_sec`` and does not pass it here, so a
        # caller that sets one must pass the same value as ``agent_timeout_s``.
        # Deriving the inner connection/planner budget keeps finalization inside
        # the Harbor kill deadline.
        self.agent_timeout_s = agent_timeout_s
        # An explicit ``timeout_s`` caps the planner budget; an unset one lets
        # the detected Harbor budget decide how long the planner may run.
        self._timeout_s_is_explicit = timeout_s is not None
        if agent_timeout_s is not None:
            budget = agent_timeout_s - 2 * _TIMEOUT_MARGIN_S
            if budget <= 0:
                raise ValueError("agent_timeout_s is too small for one session.")
            timeout_s = budget if timeout_s is None else min(timeout_s, budget)
        self.timeout_s = _DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s
        self.connect_timeout_s = connect_timeout_s
        self.max_turns = max_turns
        self.reasoning_effort = reasoning_effort
        self.skills = bool(skills)
        self.sandbox = sandbox

    @staticmethod
    def name() -> str:
        """Return the name recorded in Harbor results."""
        return "rpent-efe"

    @property
    def exec_timeout_s(self) -> int:
        """Return the exec wrapper timeout, inside Harbor's kill.

        With ``agent_timeout_s`` the wrapper stops one margin early, so the
        planner's shutdown still fits. Without it, the wrapper outlives the
        connection and planner budget by one margin, which is the best a caller
        who left Harbor's own timeout unset can do.
        """
        if self.agent_timeout_s is not None:
            return max(1, self.agent_timeout_s - _TIMEOUT_MARGIN_S)
        return self.timeout_s + _TIMEOUT_MARGIN_S

    def _resolve_harbor_agent_timeout(self) -> int | None:
        """Read Harbor's resolved timeout from this trial's local config.

        Harbor 0.23 resolves the agent timeout in its Trial object but only
        passes the value to the oracle agent; installed agents receive no
        timeout keyword argument. Read the saved trial configuration instead,
        and let ``agent_timeout_s`` override this when the config is absent.
        """
        trial_config_path = self.logs_dir.parent / "config.json"
        if not trial_config_path.is_file():
            return None

        with trial_config_path.open(encoding="utf-8") as config_file:
            trial_config = json.load(config_file)

        agent_config = trial_config.get("agent", {})
        timeout_s = agent_config.get("override_timeout_sec")
        if timeout_s is None:
            task_path = trial_config.get("task", {}).get("path")
            if not task_path:
                return None
            task_config_path = Path(task_path).expanduser() / "task.toml"
            if not task_config_path.is_file():
                return None
            with task_config_path.open("rb") as task_file:
                task_config = tomllib.load(task_file)
            timeout_s = task_config.get("agent", {}).get("timeout_sec")
        if timeout_s is None:
            return None

        max_timeout_s = agent_config.get("max_timeout_sec")
        multiplier = trial_config.get("agent_timeout_multiplier")
        if multiplier is None:
            multiplier = trial_config.get("timeout_multiplier", 1.0)
        resolved = min(float(timeout_s), float(max_timeout_s or float("inf")))
        return int(resolved * float(multiplier))

    def _resolve_task_name(self) -> str | None:
        """Return the task directory's name from this trial's saved configuration.

        The adapter uploads the instruction into the agent log directory, so
        the instruction file names the logs rather than the task. Results and
        artifact names need the task identity, which Harbor records in the
        trial configuration. A local task records its directory path; a
        dataset task records only the registry name.
        """
        trial_config_path = self.logs_dir.parent / "config.json"
        if not trial_config_path.is_file():
            return None

        with trial_config_path.open(encoding="utf-8") as config_file:
            task_config = json.load(config_file).get("task", {})
        task_path = task_config.get("path")
        if task_path:
            return Path(task_path).expanduser().name
        task_name = task_config.get("name")
        if task_name:
            # Registry names keep an ``org/name`` shape; the task is the last part.
            return str(task_name).rsplit("/", 1)[-1]
        return None

    def _apply_agent_timeout(self, agent_timeout_s: int) -> None:
        """Keep connection, planning, and finalization inside Harbor's timeout."""
        if agent_timeout_s <= 2 * _TIMEOUT_MARGIN_S:
            raise ValueError("Harbor agent timeout is too small for one session.")
        self.agent_timeout_s = agent_timeout_s
        budget = agent_timeout_s - 2 * _TIMEOUT_MARGIN_S
        # A detected Harbor budget replaces the fallback default; an explicit
        # ``timeout_s`` still caps it.
        self.timeout_s = (
            min(self.timeout_s, budget) if self._timeout_s_is_explicit else budget
        )

    async def install(self, environment: BaseEnvironment) -> None:
        """Build and install one wheel from the RPent checkout."""
        for entry in ("pyproject.toml", "README.md", "LICENSE", "rpent", "robots"):
            if not (self.source_root / entry).exists():
                raise FileNotFoundError(
                    f"RPent source is missing: {self.source_root / entry}"
                )
        with tempfile.TemporaryDirectory(prefix="rpent-efe-") as temporary_dir:
            wheel_path = self._build_wheel(Path(temporary_dir))
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / "rpent-wheel.json").write_text(
                json.dumps(
                    {
                        "filename": wheel_path.name,
                        "sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            await self.exec_as_root(environment, command="mkdir -p /opt/rpent-efe")
            await environment.upload_file(
                source_path=wheel_path,
                target_path=f"/opt/rpent-efe/{wheel_path.name}",
            )
        await self.exec_as_root(
            environment,
            command=(
                "python3 -m venv /opt/rpent-efe/venv && "
                "/opt/rpent-efe/venv/bin/python -m pip install --no-cache-dir "
                "--disable-pip-version-check "
                f"/opt/rpent-efe/{shlex.quote(wheel_path.name)}"
            ),
            timeout_sec=900,
        )
        # pip resolves dependencies inside the task image, so two trials can
        # install different versions. Record the resolved set on the host to
        # make that difference visible when results disagree.
        frozen = await self.exec_as_root(
            environment,
            command="/opt/rpent-efe/venv/bin/python -m pip freeze --all",
        )
        setup_dir = self.logs_dir / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        (setup_dir / "pip-freeze.txt").write_text(frozen.stdout or "", encoding="utf-8")
        if self.memory_source is not None:
            await self._upload_memory(environment)

    def _build_wheel(self, output_dir: Path) -> Path:
        """Build one wheel from the checkout with the host's uv."""
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError(
                "uv is required to build the RPent wheel for the EFE adapter."
            )
        completed = subprocess.run(
            [
                uv,
                "build",
                "--wheel",
                "--out-dir",
                str(output_dir),
                str(self.source_root),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Could not build the RPent wheel: {detail}")
        wheels = sorted(output_dir.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(
                "Expected one built wheel, "
                f"found {[wheel.name for wheel in wheels]} in {output_dir}"
            )
        return wheels[0]

    async def _upload_memory(self, environment: BaseEnvironment) -> None:
        """Stage the host memory corpus inside the trial logs.

        The memory root must survive the container. Harbor only syncs
        ``environment_logs_dir`` back to the trial directory, and a task may run
        the agent as a non-root ``USER``, so the root lives under the agent log
        directory and is extracted as the agent user.
        """
        remote_dir = self.environment_logs_dir / "rpent"
        remote_root = remote_dir / "memory"
        remote_archive = remote_dir / "memory.tar.gz"
        # ``docker cp`` needs the destination directory, and the agent can only
        # delete the archive from a directory it may write. Ask the agent
        # process for its ids: Harbor passes neither ``agent.user`` nor a
        # non-root image ``USER`` to this adapter, so reading the ids is the
        # only way to give the staging tree to the user that extracts into it.
        identity = await self.exec_as_agent(environment, command="id -u; id -g")
        ids = (identity.stdout or "").split()
        if len(ids) != 2:
            raise RuntimeError(
                "Cannot determine the agent user for the memory corpus: "
                f"{identity.stdout!r}"
            )
        owner = ":".join(ids)
        quoted_dir = shlex.quote(remote_dir.as_posix())
        await self.exec_as_root(
            environment,
            command=f"mkdir -p {quoted_dir} && chown -R {owner} {quoted_dir}",
        )
        with tempfile.TemporaryDirectory(prefix="rpent-efe-memory-") as temporary_dir:
            archive_path = Path(temporary_dir) / "memory.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(self.memory_source, arcname=".", recursive=True)
            await environment.upload_file(
                source_path=archive_path, target_path=str(remote_archive)
            )
        # ``--no-same-owner`` keeps the host UID/GIDs out of the container, so
        # the agent user can remove the archive and read the corpus. The archive
        # is removed before the trial reads the memory root.
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {shlex.quote(remote_root.as_posix())} && "
                f"tar -xzf {shlex.quote(str(remote_archive))} "
                f"--no-same-owner -C {shlex.quote(remote_root.as_posix())} && "
                f"rm -f {shlex.quote(str(remote_archive))}"
            ),
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run RPent in the task container and leave verification to Harbor."""
        if self.agent_timeout_s is None:
            resolved_timeout = self._resolve_harbor_agent_timeout()
            if resolved_timeout is not None:
                self._apply_agent_timeout(resolved_timeout)
        remote_output = self.environment_logs_dir / "rpent"
        remote_instruction = self.environment_logs_dir / "rpent-instruction.txt"
        connect_timeout_s = min(self.connect_timeout_s, max(1.0, self.timeout_s / 4))
        planner_timeout_s = self.timeout_s - math.ceil(connect_timeout_s)
        if planner_timeout_s <= 0:
            raise ValueError("timeout_s is too small for connection and planning.")
        with tempfile.TemporaryDirectory(prefix="rpent-efe-task-") as temporary_dir:
            instruction_path = Path(temporary_dir) / "instruction.txt"
            instruction_path.write_text(instruction, encoding="utf-8")
            await environment.upload_file(
                source_path=instruction_path, target_path=str(remote_instruction)
            )
        model = self.model_name.removeprefix("openai/")
        arguments = [
            "/opt/rpent-efe/venv/bin/rpent",
            "--robot",
            "efe",
            "--planner",
            "codex",
            "--instruction-file",
            str(remote_instruction),
            "--workdir",
            self.workdir,
            "--output-dir",
            str(remote_output),
            "--model",
            model,
            "--planner-timeout-s",
            str(planner_timeout_s),
            "--memory-profile",
            "local",
            "--memory-dir",
            str(remote_output / "memory"),
            "--efe-connect-timeout-s",
            str(connect_timeout_s),
            "--efe-trial-timeout-s",
            str(self.timeout_s),
            "--max-turns",
            str(self.max_turns),
            "--reasoning-effort",
            self.reasoning_effort,
            "--codex-sandbox",
            self.sandbox,
        ]
        if not self.skills:
            arguments.append("--efe-no-skills")
        task_name = self._resolve_task_name()
        if task_name:
            arguments += ["--task-name", task_name]
        command = shlex.join(arguments)
        forwarded = {
            name: os.environ[name]
            for name in (
                "OPENAI_API_KEY",
                "CODEX_API_KEY",
                "CODEX_BASE_URL",
                "OPENAI_BASE_URL",
                "CODEX_MODEL_CONTEXT_WINDOW",
                "CODEX_AUTO_COMPACT_TOKEN_LIMIT",
                "CODEX_SERVICE_TIER",
            )
            if name in os.environ
        }
        forwarded.update(self._extra_env)
        forwarded["CODEX_HOME"] = str(remote_output / "codex-home")
        if "CODEX_API_KEY" not in forwarded and "OPENAI_API_KEY" in forwarded:
            forwarded["CODEX_API_KEY"] = forwarded["OPENAI_API_KEY"]
        if "CODEX_API_KEY" in forwarded:
            forwarded["OPENAI_API_KEY"] = forwarded["CODEX_API_KEY"]
        if "CODEX_BASE_URL" not in forwarded and "OPENAI_BASE_URL" in forwarded:
            forwarded["CODEX_BASE_URL"] = forwarded["OPENAI_BASE_URL"]
        try:
            await self.exec_as_agent(
                environment,
                command=command,
                cwd=self.workdir,
                env=forwarded,
                timeout_sec=self.exec_timeout_s,
            )
        except Exception:
            if self._transcript_reports_budget_stop(self.logs_dir / "rpent"):
                self.logger.info(
                    "EFE planner used its time budget; Harbor will grade the episode."
                )
                return
            raise

    @staticmethod
    def _transcript_reports_budget_stop(output_dir: Path) -> bool:
        """Whether the generic CLI transcript records an intentional timeout."""
        for transcript_path in sorted(output_dir.glob("transcript_*.json")):
            try:
                transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(transcript, dict):
                continue
            if is_budget_stop(transcript.get("error")):
                return True
        return False

    @staticmethod
    def _recover_usage(output_dir: Path) -> tuple[dict[str, int], str] | None:
        """Sum the last usage event in the session trace.

        A trace without a usage event contributes nothing; ``None`` means the
        trace did not report usage.
        """
        totals = {
            "total_input_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_output_tokens": 0,
        }
        sources: list[str] = []
        observed = False
        for stream_path in sorted(output_dir.rglob("codex*.stream.jsonl")):
            last: dict[str, int | None] | None = None
            with stream_path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("method") != "thread/tokenUsage/updated":
                        continue
                    payload = event.get("payload")
                    token_usage = (
                        payload.get("token_usage")
                        if isinstance(payload, dict)
                        else None
                    )
                    total = (
                        token_usage.get("total")
                        if isinstance(token_usage, dict)
                        else None
                    )
                    if isinstance(total, dict):
                        last = total
            if last is None:
                continue
            values = {
                "total_input_tokens": last.get("input_tokens"),
                "total_cached_input_tokens": last.get("cached_input_tokens"),
                "total_output_tokens": last.get("output_tokens"),
            }
            if not any(value for value in values.values()):
                continue
            observed = True
            sources.append(stream_path.relative_to(output_dir).as_posix())
            for key, value in values.items():
                totals[key] += int(value or 0)
        if not observed:
            return None
        return totals, ",".join(sources)

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Read measured token totals after Harbor synchronizes the agent logs.

        Only ``usage_observed`` proves that the SDK reported real totals.
        Otherwise the values stay ``None`` instead of claiming zero tokens.
        """
        output_dir = self.logs_dir / "rpent"
        stats: dict[str, object] = {}
        for transcript_path in sorted(output_dir.glob("transcript_*.json")):
            try:
                transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.logger.warning(
                    "The RPent transcript is incomplete: %s", transcript_path
                )
                continue
            if not isinstance(transcript, dict):
                self.logger.warning(
                    "The RPent transcript is not a JSON object: %s", transcript_path
                )
                continue
            reported = transcript.get("stats")
            if isinstance(reported, dict):
                stats = reported
                break
            self.logger.warning(
                "The RPent transcript has no stats object: %s", transcript_path
            )
        if stats.get("usage_observed") is not True:
            recovered = self._recover_usage(output_dir)
            if recovered is None:
                return
            stats, source_name = recovered
            (output_dir / "usage-recovery.json").write_text(
                json.dumps({"complete": False, "source": source_name, "stats": stats})
                + "\n",
                encoding="utf-8",
            )
        context.n_input_tokens = stats.get("total_input_tokens")
        context.n_cache_tokens = stats.get("total_cached_input_tokens")
        context.n_output_tokens = stats.get("total_output_tokens")
        if context.n_input_tokens is not None:
            context.model_usage = {
                self.model_name: ModelUsage(
                    n_input_tokens=context.n_input_tokens,
                    n_cache_tokens=context.n_cache_tokens,
                    n_output_tokens=context.n_output_tokens,
                )
            }
