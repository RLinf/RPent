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

"""Declarative E2E suite loader and shell-metadata CLI."""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
SUITES_DIR = Path(__file__).resolve().parent / "suites"
DEFAULT_SHARED_ROOT = "/mnt/public/rpent_shared"
_SHARED_PREFIXES = ("assets/", "models/", "venvs/")
_ENV_PLACEHOLDER = re.compile(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}")
_GPU_INDEX_PLACEHOLDER = re.compile(r"\{gpu([0-9]+)\}")


def shared_root(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return env.get("RPENT_SHARED_ROOT", DEFAULT_SHARED_ROOT).rstrip("/")


def _load_suite_file(path: Path) -> dict[str, Any]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    suite_name = path.stem
    data.setdefault("suite", suite_name)
    data.setdefault("robot", suite_name)
    data.setdefault("optional_models", [])
    data.setdefault("env", {})
    data.setdefault("base_args", [])
    data.setdefault("setup", {})
    data.setdefault("cases", [])
    return data


def discover_suites(suites_dir: Path = SUITES_DIR) -> dict[str, dict[str, Any]]:
    """Load all declarative suite TOMLs sorted by suite name."""
    if not suites_dir.is_dir():
        return {}
    return {
        path.stem: _load_suite_file(path)
        for path in sorted(suites_dir.glob("*.toml"), key=lambda p: p.stem)
    }


def load_suite(name: str, suites_dir: Path = SUITES_DIR) -> dict[str, Any]:
    path = suites_dir / f"{name}.toml"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in suites_dir.glob("*.toml")))
        raise FileNotFoundError(
            f"Unknown E2E suite {name!r}; available: {available or 'none'}"
        )
    return _load_suite_file(path)


def resolve_value(
    value: str,
    *,
    shared: str,
    output_dir: str = "",
    gpu: str = "0",
    gpus: list[str] | None = None,
    python: str = "",
    environ: Mapping[str, str] | None = None,
) -> str:
    """Resolve placeholders ({output_dir}, {gpu}, {gpuN}, {python}, {env:VAR}) and shared paths."""
    env = os.environ if environ is None else environ
    gpu_list = list(gpus) if gpus else [gpu]
    primary_gpu = gpu_list[0]

    def _replace_gpu_idx(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if idx >= len(gpu_list):
            raise ValueError(
                f"Placeholder {{gpu{idx}}} requires at least {idx + 1} GPUs, "
                f"but only {len(gpu_list)} allocated: {gpu_list}"
            )
        return gpu_list[idx]

    text = (
        value.replace("{output_dir}", output_dir)
        .replace("{gpu}", primary_gpu)
        .replace("{python}", python)
    )
    text = _GPU_INDEX_PLACEHOLDER.sub(_replace_gpu_idx, text)
    text = _ENV_PLACEHOLDER.sub(lambda m: env.get(m.group(1), ""), text)
    if text.startswith(_SHARED_PREFIXES):
        return f"{shared}/{text}"
    return text


def _gpu_memory_gib(gpu_index: str) -> float | None:
    """Return total VRAM in GiB for one GPU ordinal, or None if unavailable."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first_line = completed.stdout.strip().splitlines()
    if not first_line:
        return None
    try:
        return float(first_line[0].strip()) / 1024.0
    except ValueError:
        return None


def resolve_case(
    suite: dict[str, Any],
    case: dict[str, Any],
    *,
    output_dir: Path,
    gpu: str,
    gpus: list[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve env and CLI args for one case and collect missing prerequisites."""
    env_src = dict(os.environ if environ is None else environ)
    shared = shared_root(env_src)
    out_str = str(output_dir)
    gpu_list = list(gpus) if gpus else [gpu]

    # [env] entries act as defaults: an already-set external env var always wins.
    case_env: dict[str, str] = {}
    for mapping in (suite.get("env") or {}, case.get("env") or {}):
        for key, raw in mapping.items():
            if env_src.get(key):
                case_env[key] = env_src[key]
            else:
                case_env[key] = resolve_value(
                    str(raw),
                    shared=shared,
                    output_dir=out_str,
                    gpu=gpu_list[0],
                    gpus=gpu_list,
                    environ={**env_src, **case_env},
                )

    combined_env = {**env_src, **case_env}

    raw_sidecar = case.get("sidecar")
    sidecar: dict[str, Any] | None = None
    external_sidecar_endpoint = ""
    if isinstance(raw_sidecar, dict):
        endpoint_env = str(raw_sidecar.get("endpoint_env", "")).strip()
        if endpoint_env and combined_env.get(endpoint_env, "").strip():
            external_sidecar_endpoint = combined_env[endpoint_env].strip()
        else:
            sidecar_env = {
                key: resolve_value(
                    str(raw),
                    shared=shared,
                    output_dir=out_str,
                    gpu=gpu_list[0],
                    gpus=gpu_list,
                    environ=combined_env,
                )
                for key, raw in (raw_sidecar.get("env") or {}).items()
            }
            sidecar_cmd = [
                resolve_value(
                    str(token),
                    shared=shared,
                    output_dir=out_str,
                    gpu=gpu_list[0],
                    gpus=gpu_list,
                    environ={**combined_env, **sidecar_env},
                )
                for token in (raw_sidecar.get("cmd") or [])
            ]
            sidecar = {
                "cmd": sidecar_cmd,
                "env": sidecar_env,
                "endpoint": str(raw_sidecar.get("endpoint", "http://127.0.0.1:{port}")),
                "ready_timeout_s": int(raw_sidecar.get("ready_timeout_s", 300)),
            }

    raw_args = [*(suite.get("base_args") or []), *(case.get("args") or [])]
    args = [
        resolve_value(
            str(arg),
            shared=shared,
            output_dir=out_str,
            gpu=gpu_list[0],
            gpus=gpu_list,
            environ=combined_env,
        )
        for arg in raw_args
    ]
    if external_sidecar_endpoint:
        args = [
            arg.replace("{sidecar_endpoint}", external_sidecar_endpoint) for arg in args
        ]
        case_env = {
            k: v.replace("{sidecar_endpoint}", external_sidecar_endpoint)
            for k, v in case_env.items()
        }

    missing: list[str] = []
    for var in case.get("require_env") or []:
        if not combined_env.get(var, "").strip():
            missing.append(f"env {var} is not set")

    opt_in = case.get("opt_in_env")
    if opt_in and combined_env.get(opt_in, "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        missing.append(f"opt-in {opt_in}=1 is not set")

    min_vram_gb = case.get("min_gpu_memory_gb")
    if min_vram_gb is not None:
        required_gib = float(min_vram_gb)
        for ordinal in gpu_list:
            total_gib = _gpu_memory_gib(ordinal)
            if total_gib is not None and total_gib < required_gib:
                missing.append(
                    f"GPU {ordinal} has {total_gib:.1f} GiB VRAM; "
                    f"case requires >= {required_gib:.0f} GiB"
                )
                break

    shared_prefix = shared + "/"
    out_clean = out_str.rstrip("/")
    out_prefix = out_clean + "/"
    seen_paths: set[str] = set()
    resource_values: list[str] = [*case_env.values(), *args]
    if sidecar is not None:
        resource_values.extend(sidecar["env"].values())
        resource_values.extend(sidecar["cmd"])
    for val in resource_values:
        if (
            val.startswith(shared_prefix)
            and val != out_clean
            and not val.startswith(out_prefix)
            and val not in seen_paths
        ):
            seen_paths.add(val)
            path = Path(val)
            if not path.exists() or (path.is_dir() and not any(path.iterdir())):
                missing.append(f"missing resource {val}")

    for raw_req in (
        *(suite.get("require_paths") or ()),
        *(case.get("require_paths") or ()),
    ):
        req = resolve_value(
            str(raw_req),
            shared=shared,
            output_dir=out_str,
            gpu=gpu,
            gpus=gpu_list,
            environ=combined_env,
        )
        if req not in seen_paths:
            seen_paths.add(req)
            path = Path(req)
            if not path.exists() or (path.is_dir() and not any(path.iterdir())):
                missing.append(f"missing resource {req}")

    return {
        "name": case["name"],
        "robot": suite["robot"],
        "args": args,
        "env": case_env,
        "sidecar": sidecar,
        "action_tool": case.get("action_tool"),
        "action_count_field": case.get("action_count_field"),
        "tool_args": {
            **(suite.get("tool_args") or {}),
            **(case.get("tool_args") or {}),
        },
        "skip_tools": sorted(
            {*(suite.get("skip_tools") or ()), *(case.get("skip_tools") or ())}
        ),
        "require_post_action_image": bool(case.get("require_post_action_image", False)),
        "missing": missing,
    }


def component_cache_hash(component: str, root: Path = REPO_ROOT) -> str:
    """Compute a deterministic short hash for a component's install inputs."""
    manifests_by_name: dict[str, tuple[Path, dict[str, Any]]] = {}
    for pattern in ("robots/*/component.toml", "components/*/*/component.toml"):
        for path in sorted(root.glob(pattern)):
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            manifests_by_name[data["name"]] = (path, data)

    visited: set[str] = set()
    files: list[Path] = []
    for base_file in (
        "pyproject.toml",
        "setup.py",
        "MANIFEST.in",
        "rpent-install.sh",
        "rpent/cli/install.py",
        "tests/e2e_tests/run_gpu_suite.sh",
        ".github/workflows/embodied-gpu-e2e.yml",
    ):
        candidate = root / base_file
        if candidate.is_file():
            files.append(candidate)
    install_dir = root / "rpent" / "install"
    if install_dir.is_dir():
        files.extend(sorted(install_dir.glob("*.py")))

    pending = [component]
    while pending:
        name = pending.pop()
        if name in visited or name not in manifests_by_name:
            continue
        visited.add(name)
        mpath, data = manifests_by_name[name]
        files.append(mpath)
        for rel in data.get("overrides") or ():
            extra = (mpath.parent / str(rel)).resolve()
            if extra.is_file():
                files.append(extra)
        pending.extend(str(r) for r in (data.get("requires") or ()))

    hasher = hashlib.sha256()
    hasher.update(component.encode("utf-8"))
    for path in sorted(set(files)):
        rel_str = (
            path.relative_to(root).as_posix()
            if path.is_relative_to(root)
            else str(path)
        )
        hasher.update(b"\0" + rel_str.encode("utf-8") + b"\0")
        hasher.update(path.read_bytes())

    # Include suite-level venv setup affecting this component.
    suites_dir = root / "tests" / "e2e_tests" / "suites"
    if suites_dir.is_dir():
        for suite_name, suite in discover_suites(suites_dir).items():
            if suite.get("component") == component:
                setup = suite.get("setup") or {}
                post_cmd = " ".join(
                    str(p) for p in (setup.get("post_install_cmd") or ())
                )
                hasher.update(
                    f"\0suite:{suite_name}:{setup.get('reinstall_copy_pkg', '')}:"
                    f"{','.join(sorted(str(p) for p in (setup.get('materialize_pkgs') or ())))}:"
                    f"{post_cmd}".encode("utf-8")
                )
    return hasher.hexdigest()[:12]


def _is_venv_level_post_cmd(cmd: list[Any]) -> bool:
    """Commands not referencing {output_dir} mutate only the venv and belong in install."""
    return bool(cmd) and not any("{output_dir}" in str(tok) for tok in cmd)


def _print_install_meta(component: str, python: str = "") -> None:
    shared = shared_root()
    materialize: list[str] = []
    reinstall_pkgs: list[str] = []
    venv_post_cmds: list[list[str]] = []
    has_eval_suites = False
    for suite in discover_suites().values():
        if suite.get("component") != component:
            continue
        has_eval_suites = True
        setup = suite.get("setup") or {}
        for pkg in setup.get("materialize_pkgs") or ():
            s = str(pkg)
            if s not in materialize:
                materialize.append(s)
        reinstall = str(setup.get("reinstall_copy_pkg") or "").strip()
        if reinstall and reinstall not in reinstall_pkgs:
            reinstall_pkgs.append(reinstall)
        raw_post = list(setup.get("post_install_cmd") or ())
        if _is_venv_level_post_cmd(raw_post):
            resolved = [
                resolve_value(str(p), shared=shared, python=python) for p in raw_post
            ]
            if resolved not in venv_post_cmds:
                venv_post_cmds.append(resolved)

    print(f"INSTALL_COMPONENT={shlex.quote(component)}")
    print(f"INSTALL_HAS_EVAL={'1' if has_eval_suites else '0'}")
    print(f"INSTALL_CACHE_HASH={shlex.quote(component_cache_hash(component))}")
    print(
        "INSTALL_MATERIALIZE_PKGS=("
        + " ".join(shlex.quote(p) for p in materialize)
        + ")"
    )
    print(
        "INSTALL_REINSTALL_PKGS=("
        + " ".join(shlex.quote(p) for p in reinstall_pkgs)
        + ")"
    )
    # Encode each venv-level post_install_cmd as a shell-quoted string.
    encoded_cmds = [" ".join(shlex.quote(tok) for tok in cmd) for cmd in venv_post_cmds]
    print("INSTALL_POST_CMDS=(" + " ".join(shlex.quote(c) for c in encoded_cmds) + ")")


def _print_eval_meta(suite_name: str, output_dir: str, python: str = "") -> None:
    suite = load_suite(suite_name)
    shared = shared_root()
    setup = suite.get("setup") or {}
    component = suite["component"]

    print(f"SUITE_COMPONENT={shlex.quote(component)}")
    print(f"SUITE_CACHE_HASH={shlex.quote(component_cache_hash(component))}")
    raw_post = list(setup.get("post_install_cmd") or ())
    if _is_venv_level_post_cmd(raw_post):
        # Already executed once during install stage; do not mutate shared venv in eval.
        post_cmd: list[str] = []
    else:
        post_cmd = [
            resolve_value(str(p), shared=shared, output_dir=output_dir, python=python)
            for p in raw_post
        ]
    print("SUITE_POST_INSTALL_CMD=(" + " ".join(shlex.quote(p) for p in post_cmd) + ")")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(
            "Usage: suite.py <list|install-meta <component> [python]"
            "|eval-meta <suite> <output_dir> [python]>",
            file=sys.stderr,
        )
        return 2
    cmd = args[0]
    if cmd == "list":
        for name in discover_suites():
            print(name)
        return 0
    if cmd == "install-meta" and len(args) in (2, 3):
        _print_install_meta(args[1], args[2] if len(args) == 3 else "")
        return 0
    if cmd == "eval-meta" and len(args) in (3, 4):
        _print_eval_meta(args[1], args[2], args[3] if len(args) == 4 else "")
        return 0
    print(f"Unknown command: {' '.join(args)}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
