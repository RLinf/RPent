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

"""Harbor 0.23 installed agent for EFE-compatible RPent trial sessions."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import tarfile
import tempfile
import tomllib
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext, ModelUsage

from robots.efe import SANDBOX_CHOICES


class RPentEfeAgent(BaseInstalledAgent):
    """Install the local RPent source inside Harbor's isolated agent container."""

    def __init__(
        self,
        logs_dir: Path,
        *,
        source_root: str | None = None,
        workdir: str = "/app",
        timeout_s: int = 3500,
        max_turns: int = 200,
        reasoning_effort: str = "none",
        memory_source: str | None = None,
        explore_sessions: int = 1,
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
        if timeout_s <= 0 or max_turns <= 0:
            raise ValueError("timeout_s and max_turns must be positive.")
        if explore_sessions <= 0:
            raise ValueError("explore_sessions must be positive.")
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
        self.timeout_s = timeout_s
        self.max_turns = max_turns
        self.reasoning_effort = reasoning_effort
        self.explore_sessions = explore_sessions
        self.skills = bool(skills)
        self.sandbox = sandbox

    @staticmethod
    def name() -> str:
        """Return the name recorded in Harbor results."""
        return "rpent-efe"

    async def install(self, environment: BaseEnvironment) -> None:
        """Upload source and install RPent without robot extras."""
        entries = ("pyproject.toml", "README.md", "LICENSE", "rpent", "robots")
        for entry in entries:
            if not (self.source_root / entry).exists():
                raise FileNotFoundError(
                    f"RPent source is missing: {self.source_root / entry}"
                )
        source_files = {self.source_root / entry for entry in entries[:3]}
        for package in entries[3:]:
            source_files.update(
                path
                for path in (self.source_root / package).rglob("*.py")
                if not any(
                    part.startswith(".") or part == "__pycache__"
                    for part in path.relative_to(self.source_root).parts
                )
            )
        with (self.source_root / "pyproject.toml").open("rb") as project_file:
            package_data = tomllib.load(project_file)["tool"]["setuptools"].get(
                "package-data", {}
            )
        for package, patterns in package_data.items():
            package_dir = self.source_root.joinpath(*package.split("."))
            for pattern in patterns:
                source_files.update(package_dir.glob(pattern))
        await self.exec_as_root(environment, command="mkdir -p /opt/rpent-efe/source")
        with tempfile.TemporaryDirectory(prefix="rpent-efe-") as temporary_dir:
            archive_path = Path(temporary_dir) / "source.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for path in sorted(source_files):
                    if path.is_file() and not path.is_symlink():
                        archive.add(
                            path,
                            arcname=str(path.relative_to(self.source_root)),
                            recursive=False,
                        )
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / "rpent-source.json").write_text(
                json.dumps(
                    {"sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}
                )
                + "\n",
                encoding="utf-8",
            )
            await environment.upload_file(
                source_path=archive_path, target_path="/opt/rpent-efe/source.tar.gz"
            )
        await self.exec_as_root(
            environment,
            command=(
                "tar -xzf /opt/rpent-efe/source.tar.gz -C /opt/rpent-efe/source && "
                "python3 -m venv /opt/rpent-efe/venv && "
                "/opt/rpent-efe/venv/bin/python -m pip install --no-cache-dir "
                "--disable-pip-version-check /opt/rpent-efe/source"
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

    async def _upload_memory(self, environment: BaseEnvironment) -> None:
        """Copy the host memory corpus into the agent container."""
        await self.exec_as_root(environment, command="mkdir -p /opt/rpent-efe/memory")
        with tempfile.TemporaryDirectory(prefix="rpent-efe-memory-") as temporary_dir:
            archive_path = Path(temporary_dir) / "memory.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(self.memory_source, arcname=".", recursive=True)
            await environment.upload_file(
                source_path=archive_path, target_path="/opt/rpent-efe/memory.tar.gz"
            )
        await self.exec_as_root(
            environment,
            command=("tar -xzf /opt/rpent-efe/memory.tar.gz -C /opt/rpent-efe/memory"),
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run RPent in the task container and leave verification to Harbor."""
        remote_output = self.environment_logs_dir / "rpent"
        remote_instruction = self.environment_logs_dir / "rpent-instruction.txt"
        with tempfile.TemporaryDirectory(prefix="rpent-efe-task-") as temporary_dir:
            instruction_path = Path(temporary_dir) / "instruction.txt"
            instruction_path.write_text(instruction, encoding="utf-8")
            await environment.upload_file(
                source_path=instruction_path, target_path=str(remote_instruction)
            )
        model = self.model_name.removeprefix("openai/")
        arguments = [
            "/opt/rpent-efe/venv/bin/python",
            "-m",
            "robots.efe.cli",
            "--instruction-file",
            str(remote_instruction),
            "--workdir",
            self.workdir,
            "--output-dir",
            str(remote_output),
            "--model",
            model,
            "--timeout-s",
            str(self.timeout_s),
            "--max-turns",
            str(self.max_turns),
            "--reasoning-effort",
            self.reasoning_effort,
            "--explore-sessions",
            str(self.explore_sessions),
            "--codex-sandbox",
            self.sandbox,
        ]
        if self.memory_source is not None:
            arguments.extend(["--memory-dir", "/opt/rpent-efe/memory"])
        if not self.skills:
            arguments.append("--no-skills")
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
        if "CODEX_API_KEY" not in forwarded and "OPENAI_API_KEY" in forwarded:
            forwarded["CODEX_API_KEY"] = forwarded["OPENAI_API_KEY"]
        if "CODEX_API_KEY" in forwarded:
            forwarded["OPENAI_API_KEY"] = forwarded["CODEX_API_KEY"]
        if "CODEX_BASE_URL" not in forwarded and "OPENAI_BASE_URL" in forwarded:
            forwarded["CODEX_BASE_URL"] = forwarded["OPENAI_BASE_URL"]
        await self.exec_as_agent(
            environment,
            command=command,
            cwd=self.workdir,
            env=forwarded,
            timeout_sec=self.timeout_s + 30,
        )

    @staticmethod
    def _recover_usage(output_dir: Path) -> tuple[dict[str, int], str] | None:
        """Sum the last usage event in every session trace.

        Sessions are independent Codex contexts, so their totals add up. A
        trace without a usage event contributes nothing; ``None`` means no
        trace reported usage.
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
                    if event.get("method") == "thread/tokenUsage/updated":
                        last = event["payload"]["token_usage"]["total"]
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

        The session report always carries zeroed usage counters, so only
        ``usage_observed`` proves that the SDK reported real totals. Otherwise
        the values stay ``None`` instead of claiming zero tokens.
        """
        report_path = self.logs_dir / "rpent" / "session.json"
        stats: dict[str, object] = {}
        if report_path.is_file():
            try:
                stats = json.loads(report_path.read_text(encoding="utf-8"))["stats"]
            except json.JSONDecodeError:
                self.logger.warning(
                    "The RPent session report is incomplete: %s", report_path
                )
        if stats.get("usage_observed") is not True:
            recovered = self._recover_usage(report_path.parent)
            if recovered is None:
                return
            stats, source_name = recovered
            (report_path.parent / "usage-recovery.json").write_text(
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
