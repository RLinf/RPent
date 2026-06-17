"""Hybrid LLM-in-the-loop agent runner.

Drives interactive_driver.py through tool calls. Supports:
- Starting the driver as a subprocess (or attaching to an existing one)
- Multi-turn Claude conversation with vision + tool use
- Per-turn token usage logging
- Conversation transcript persisted at the end

Use as a script (see __main__ at the bottom) or import `run_one_cell`.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

# Auto-detect project paths: this file is at
# <repo>/physical_agent/apps/runner.py
from physical_agent.utils.config import (
    get_anthropic_api_key,
    get_anthropic_base_url,
    get_anthropic_model,
    get_cuda_device,
    get_libero_type,
    get_memory_dir,
    get_openai_compat_api_key,
    get_openai_compat_base_url,
    get_openai_compat_model,
    get_openai_compat_supports_images,
    get_repl_driver_script,
    get_repo_root,
    get_vla_server_script,
)

REPO_ROOT = get_repo_root()

from physical_agent.cerebrum.adapters.anthropic import AnthropicAdapter  # noqa: E402
from physical_agent.cerebrum.adapters.openai_compat import (  # noqa: E402
    OpenAICompatibleAdapter,
)
from physical_agent.cerebrum.api_loop import ApiAgentLoop  # noqa: E402
from physical_agent.cerebrum.claude_code import ClaudeCodeCerebrum  # noqa: E402
from physical_agent.cerebrum.codex import CodexCerebrum  # noqa: E402
from physical_agent.envs.registry import (  # noqa: E402
    infer_env_from_suite,
)
from physical_agent.driver_client import (  # noqa: E402
    create_driver_client,
    get_socket_endpoint,
    set_socket_endpoint,
)
from physical_agent.driver_client.vla_client import VLAClient  # noqa: E402
from physical_agent.tools import (  # noqa: E402
    create_tool_registry,
    tool_result_to_content_blocks,
)
from physical_agent.utils.logging import get_logger, init_output_dir  # noqa: E402

logger = get_logger("agent")


def _pipe_driver_output(
    proc: subprocess.Popen,
    log_file,
    ready_events: "queue.Queue[dict]",
) -> None:
    """Copy driver stdout to log and capture machine-readable ready events."""
    assert proc.stdout is not None
    for line in proc.stdout:
        log_file.write(line)
        log_file.flush()
        try:
            event = json.loads(line)
        except Exception:
            continue
        if isinstance(event, dict) and event.get("event") == "transport_ready":
            ready_events.put(event)


def start_driver(
    suite: str,
    task: int,
    seed: int,
    output_dir: str,
    max_episode_steps: int = 600,
    libero_type: str | None = None,
    cuda_device: str | None = None,
    log_path: str | None = None,
    driver_script: str | None = None,
    ready_timeout_s: float = 300.0,
    transport_host: str = "127.0.0.1",
    transport_port: int = 0,
) -> subprocess.Popen:
    """Launch the env+model RPC driver in background.

    The driver subprocess hosts ONLY the env and model. Primitives, video
    recording, and state dumping run agent-side. The driver prints a
    machine-readable ``transport_ready`` event on stdout once its RPC
    server is listening; this function returns once that event is seen.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        log_path = str(out_dir / "driver.log")

    env = os.environ.copy()
    env["LIBERO_TYPE"] = libero_type or get_libero_type()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device or get_cuda_device())
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("ROBOT_PLATFORM", "LIBERO")

    cmd = [
        sys.executable,
        driver_script or str(get_repl_driver_script()),
        "--suite", suite,
        "--task", str(task),
        "--seed", str(seed),
        "--max_episode_steps", str(max_episode_steps),
        "--output_dir", str(out_dir),
        "--transport_host", transport_host,
        "--transport_port", str(transport_port),
    ]
    logger.info("driver cmd: %s", ' '.join(cmd))
    logger.info("driver log: %s", log_path)
    logger.info("CUDA_VISIBLE_DEVICES=%s  output_dir=%s", cuda_device, out_dir)
    log_f = open(log_path, "w")
    ready_events: queue.Queue[dict] = queue.Queue()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(REPO_ROOT),
        text=True,
        bufsize=1,
    )
    threading.Thread(
        target=_pipe_driver_output,
        args=(proc, log_f, ready_events),
        daemon=True,
    ).start()

    logger.info("waiting for transport_ready (Pi0 load ~80s)...")
    t0 = time.time()
    transport_ready = False
    while not transport_ready:
        try:
            event = ready_events.get(timeout=2.0)
        except queue.Empty:
            event = None
        if event is not None and event.get("kind") == "socket" \
                and event.get("host") and event.get("port"):
            set_socket_endpoint(out_dir, event["host"], int(event["port"]))
            transport_ready = True
            logger.info(
                "socket transport ready at %s:%s",
                event["host"],
                event["port"],
            )
            break
        if proc.poll() is not None:
            logger.error("driver EXITED before becoming ready. Last log:")
            logger.error("%s", Path(log_path).read_text()[-2000:])
            raise RuntimeError("driver exited prematurely")
        if time.time() - t0 > ready_timeout_s:
            proc.terminate()
            raise RuntimeError(f"driver not ready after {ready_timeout_s}s")
    logger.info("driver ready in %.1fs", time.time()-t0)
    return proc


def stop_driver(
    proc: subprocess.Popen,
    output_dir: str,
    stop_recording_and_save: Callable[[], None] | None = None,
    timeout: float = 15.0,
) -> None:
    if proc.poll() is not None:
        return
    # Agent-side: flush the episode video before the env+model process dies.
    if stop_recording_and_save is not None:
        try:
            stop_recording_and_save()
        except Exception:
            pass
    try:
        client = create_driver_client(output_dir)
        client.call("shutdown", timeout_s=timeout)
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def start_vla_server(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    cuda_device: str | None = None,
    log_path: str | None = None,
) -> tuple[str, subprocess.Popen]:
    """Launch the Pi0.5 VLA HTTP server in background.

    Returns ``(base_url, proc)``. ``port=0`` asks the OS for a free port.
    Caller is responsible for stopping ``proc`` via :func:`stop_vla_server`.
    """
    if port == 0:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            port = int(s.getsockname()[1])

    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

    cmd = [
        sys.executable,
        str(get_vla_server_script()),
        "--host", host,
        "--port", str(port),
    ]
    logger.info("vla_server cmd: %s", " ".join(cmd))
    if log_path:
        log_f = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    else:
        proc = subprocess.Popen(cmd, env=env)

    base_url = f"http://{host}:{port}"
    # Block until /healthz responds so callers don't race the model load.
    client = VLAClient(base_url)
    t0 = time.time()
    while time.time() - t0 < 300:
        if proc.poll() is not None:
            raise RuntimeError("vla_server exited prematurely")
        try:
            if client.healthz():
                logger.info("vla_server ready at %s after %.1fs", base_url, time.time() - t0)
                return base_url, proc
        except Exception:
            pass
        time.sleep(2.0)
    proc.terminate()
    raise RuntimeError("vla_server not ready after 300s")


def stop_vla_server(proc: subprocess.Popen | None, timeout: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


# ---------------------------------------------------------------------------
# API agent transcript serialization
# ---------------------------------------------------------------------------


def _serialize_messages(messages: list[dict]) -> list[dict]:
    """Convert SDK objects in messages to plain dicts so they're JSON-safe."""
    out = []
    for m in messages:
        message = {k: v for k, v in m.items() if k != "content"}
        message["content"] = _sanitize_transcript_content(m.get("content"))
        out.append(message)
    return out


def _sanitize_transcript_content(value):
    """Return a JSON-safe copy while omitting large inline image payloads."""
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, list):
        return [_sanitize_transcript_content(v) for v in value]
    if isinstance(value, dict):
        if value.get("type") == "image":
            return {"type": "image", "source": {"_omitted_for_transcript": True}}
        if value.get("type") == "image_url":
            return {"type": "image_url", "image_url": {"_omitted_for_transcript": True}}
        return {k: _sanitize_transcript_content(v) for k, v in value.items()}

    block: dict = {"type": getattr(value, "type", "?")}
    for attr in ("text", "name", "input", "id"):
        if hasattr(value, attr):
            block[attr] = _sanitize_transcript_content(getattr(value, attr))
    return block


# ---------------------------------------------------------------------------
# Emergency save (so a successful sim run isn't lost on agent crash)
# ---------------------------------------------------------------------------


def _emergency_save(output_dir, suite, task, seed, recipe_tag,
                    agent_error, regime="strict", verbose=True):
    """If ``<output_dir>/states.json`` has libero_terminated=True in any
    step and the recipe.jsonl / audit.json are missing, stitch them now.
    Idempotent — won't overwrite existing files.
    """
    out = Path(output_dir)
    if not out.exists():
        return
    recipe_path = out / f"recipe_{recipe_tag}.jsonl"
    audit_path = out / f"{recipe_tag}.json"
    if recipe_path.exists() and audit_path.exists():
        return  # agent already saved

    # Load the merged states.json (top-level JSON array, one entry per step).
    states: dict[int, dict] = {}
    sim_terminated = False
    states_path = out / "states.json"
    if states_path.exists():
        try:
            with open(states_path) as f:
                arr = json.load(f)
            if isinstance(arr, list):
                for i, entry in enumerate(arr):
                    if not isinstance(entry, dict):
                        continue
                    sn = int(entry.get("step_idx", i))
                    states[sn] = entry
                    if entry.get("libero_terminated"):
                        sim_terminated = True
        except Exception:
            pass

    if not states:
        return  # nothing to salvage

    # Always rebuild a recipe.jsonl from the commands actually executed
    # (commands are merged into each entry of states.json).
    if not recipe_path.exists():
        recipe_lines = []
        for sn in sorted(states.keys()):
            cmd = states[sn].get("command") or {}
            if cmd.get("action") in ("exit", "reset"):
                continue
            recipe_lines.append(json.dumps(cmd))
        if recipe_lines:
            out.mkdir(parents=True, exist_ok=True)
            recipe_path.write_text("\n".join(recipe_lines) + "\n")
            if verbose:
                logger.info("[emergency_save] wrote %s (%d cmds)", recipe_path, len(recipe_lines))

    # Build a minimal audit (PRO schema) if missing
    if not audit_path.exists():
        pick_step = next(
            (n for n in sorted(states) if (states[n].get("command") or {}).get("action") == "pi0_pick"),
            None,
        )
        release_step = next(
            (n for n in reversed(sorted(states)) if (states[n].get("command") or {}).get("action") == "release"),
            None,
        )
        move_steps = [n for n in sorted(states) if (states[n].get("command") or {}).get("action") == "move_to"]
        last_state = states[max(states)] if states else {}
        record = {
            "suite": suite,
            "task_id": task,
            "seed": seed,
            "regime": regime,
            "strategy_notes": (
                f"emergency-saved by runner after agent error: {agent_error}"
                if agent_error else "emergency-saved by runner (agent did not call finish)"
            ),
            "pick_result":     states[pick_step]["result"]   if pick_step    is not None else None,
            "post_pick_state": states[pick_step]["state"]  if pick_step    is not None and pick_step in states else None,
            "move_results":    [states[n]["result"] for n in move_steps],
            "release_result":  states[release_step]["result"] if release_step is not None else None,
            "final_state":     last_state.get("state"),
            "libero_terminated": bool(last_state.get("libero_terminated")),
            "sim_reached_terminated": sim_terminated,
            "agent_error": agent_error,
        }
        out.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(record, indent=2, default=str))
        if verbose:
            logger.info("[emergency_save] wrote %s (libero_terminated=%s)",
                         audit_path, record['libero_terminated'])


# ---------------------------------------------------------------------------
# High-level entrypoint
# ---------------------------------------------------------------------------


def run_one_cell(
    suite: str,
    task: int,
    seed: int,
    *,
    api_key: str | None = None,
    model: str | None = None,
    max_turns: int = 80,
    max_tokens: int = 4096,
    max_episode_steps: int = 600,
    cuda_device: str | None = None,
    output_dir: str | None = None,
    no_driver: bool = False,
    verbose: bool = True,
    base_url: str | None = None,
    perception: bool = False,
    libero_type: str | None = None,
    cerebrum_type: str = "anthropic",
    openai_compat_supports_images: bool | None = None,
    thinking: bool = False,
    claude_code_timeout_s: int | None = None,
    claude_code_max_budget_usd: float | None = None,
    codex_timeout_s: int | None = None,
    transport_host: str = "127.0.0.1",
    transport_port: int = 0,
    vla_endpoint: str | None = None,
    env_name: str | None = None,
) -> dict:
    """Solve one (suite, task, seed) cell end-to-end.

    Everything for the run — the driver's images/, depths/, states.json,
    camera_meta.json and episode.mp4, plus the agent's recipe/audit/transcript
    — lands in ``output_dir``. For parallel runs, give each cell its own
    ``output_dir``.

    ``cerebrum_type`` selects the LLM backend:
    - ``"anthropic"`` — Anthropic Messages API with tool-use (default).
    - ``"openai_compat"`` — OpenAI-compatible Chat Completions.
    - ``"claude_code"`` — delegates to ``claude -p`` (Claude Code).
    - ``"codex"`` — delegates to local ``codex exec``.
    """
    env_name = env_name or infer_env_from_suite(suite)
    tool_registry = create_tool_registry(env_name)
    env_spec = tool_registry.env_spec
    prompt_bundle = env_spec.prompts

    if max_episode_steps == 600 and "libero_10" in suite:
        max_episode_steps = 5000
        if verbose:
            logger.info("auto-bumped max_episode_steps to 5000 for libero_10")

    # ---- resolve output directory ----
    # Everything for the run lands under this single directory: driver
    # artifacts (images/, depths/, states.json, ...) and agent outputs
    # (recipe/audit/transcript) sit side by side.
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = REPO_ROOT / "logs" / f"{timestamp}_{suite}_t{task}_s{seed}"
    output_dir = init_output_dir(output_dir)

    if cerebrum_type == "anthropic":
        api_key = api_key or get_anthropic_api_key()
        if not api_key:
            raise ValueError("api_key is required for anthropic cerebrum")
        import anthropic
        client = anthropic.Anthropic(
            api_key=api_key,
            max_retries=8,
            timeout=120.0,
            **({"base_url": base_url} if base_url else {}),
        )
        cerebrum = ApiAgentLoop(
            adapter=AnthropicAdapter(
                client=client,
                model=model or get_anthropic_model(),
                max_tokens=max_tokens,
                thinking=thinking,
            )
        )
    elif cerebrum_type == "openai_compat":
        api_key = api_key or get_openai_compat_api_key()
        base_url = base_url or get_openai_compat_base_url()
        if not api_key:
            raise ValueError("api_key is required for openai_compat cerebrum")
        import openai

        client_kwargs = {"api_key": api_key, "max_retries": 0, "timeout": 120.0}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = openai.OpenAI(**client_kwargs)
        supports_images = openai_compat_supports_images
        if supports_images is None:
            supports_images = get_openai_compat_supports_images()
        cerebrum = ApiAgentLoop(
            adapter=OpenAICompatibleAdapter(
                client=client,
                model=model or get_openai_compat_model(),
                max_tokens=max_tokens,
                supports_images=supports_images,
                thinking=thinking,
            )
        )
    elif cerebrum_type == "claude_code":
        cc_timeout_s = claude_code_timeout_s
        if cc_timeout_s is None:
            cc_timeout_s = int(os.environ.get("CELL_TIMEOUT_S", "1200" if perception else "600"))
        cc_budget = claude_code_max_budget_usd
        if cc_budget is None:
            cc_budget = float(os.environ.get("MAX_BUDGET_USD", "10"))
        cc_output_path = Path(output_dir) / f"claude_{suite.replace('libero_', '')}_t{task}_s{seed}.txt"
        cerebrum = ClaudeCodeCerebrum(
            output_dir=output_dir,
            repo_root=REPO_ROOT,
            model=model or "sonnet",
            timeout_s=cc_timeout_s,
            max_budget_usd=cc_budget,
            extra_dirs=[str(get_memory_dir())],
            output_path=cc_output_path,
            transport_host=transport_host,
            transport_port=transport_port,
            env_name=env_spec.name,
            hide_object_coords=perception,
            video_path=str(Path(output_dir) / "episode.mp4"),
        )
    elif cerebrum_type == "codex":
        cx_timeout_s = codex_timeout_s
        if cx_timeout_s is None:
            cx_timeout_s = int(os.environ.get(
                "CODEX_TIMEOUT_S",
                os.environ.get("CELL_TIMEOUT_S", "1200" if perception else "600"),
            ))
        cx_output_path = Path(output_dir) / f"codex_{suite.replace('libero_', '')}_t{task}_s{seed}.txt"
        cerebrum = CodexCerebrum(
            output_dir=output_dir,
            repo_root=REPO_ROOT,
            model=model,
            timeout_s=cx_timeout_s,
            extra_dirs=[str(get_memory_dir())],
            output_path=cx_output_path,
            transport_host=transport_host,
            transport_port=transport_port,
            env_name=env_spec.name,
            hide_object_coords=perception,
            video_path=str(Path(output_dir) / "episode.mp4"),
        )
    else:
        raise ValueError(f"unknown cerebrum_type: {cerebrum_type}")

    # Auto-route LIBERO_TYPE if not set
    if libero_type is None:
        if any(suite.endswith(s) for s in ("_swap", "_task", "_lan")):
            libero_type = "pro"
        else:
            libero_type = "standard"

    recipe_tag = f"{suite.replace('libero_', '')}_t{task}_s{seed}"
    regime = "strict_perception" if perception else "strict"

    if cerebrum_type in {"claude_code", "codex"}:
        # CLI cerebrums get the full legacy single-shot prompt.  They interact
        # through local CLI tools plus the PhysicalAgent MCP bridge, not API
        # tool schemas.
        system_prompt = ""
        template = prompt_bundle.cli_prompt_template(perception=perception)
        user_msg = prompt_bundle.format_claude_code_prompt(
            template,
            suite=suite, task=task, seed=seed,
            output_dir=output_dir, recipe_tag=recipe_tag,
        )
    else:
        user_msg = prompt_bundle.api_user_message(
            perception=perception,
            suite=suite,
            task=task,
            seed=seed,
            output_dir=output_dir,
            recipe_tag=recipe_tag,
        )
        system_prompt = prompt_bundle.api_system_prompt(perception=perception)

    proc = None
    vla_proc = None
    if not no_driver:
        proc = start_driver(
            suite=suite, task=task, seed=seed,
            output_dir=output_dir,
            max_episode_steps=max_episode_steps,
            cuda_device=cuda_device,
            libero_type=libero_type,
            transport_host=transport_host,
            transport_port=transport_port,
        )
        if vla_endpoint is None:
            vla_endpoint, vla_proc = start_vla_server(
                cuda_device=cuda_device,
                log_path=str(Path(output_dir) / "vla_server.log"),
            )
        if cerebrum_type in {"claude_code", "codex"}:
            endpoint = get_socket_endpoint(output_dir)
            if endpoint is None:
                raise RuntimeError(
                    f"socket endpoint not registered for output_dir: {output_dir}"
                )
            cerebrum.set_socket_endpoint(*endpoint)
            cerebrum.set_vla_endpoint(vla_endpoint)
            cerebrum.set_driver_process(proc)
        else:
            env_spec.set_driver_client(
                create_driver_client(output_dir),
                model=VLAClient(vla_endpoint),
                hide_object_coords=perception,
                video_path=str(Path(output_dir) / "episode.mp4"),
            )
    else:
        if transport_port <= 0:
            raise RuntimeError(
                "--no_driver requires --transport_port pointing at an existing driver"
            )
        if vla_endpoint is None:
            raise RuntimeError(
                "--no_driver requires --vla_endpoint pointing at an existing vla_server"
            )
        set_socket_endpoint(output_dir, transport_host, transport_port)
        if cerebrum_type in {"claude_code", "codex"}:
            cerebrum.set_socket_endpoint(transport_host, transport_port)
            cerebrum.set_vla_endpoint(vla_endpoint)
        else:
            env_spec.set_driver_client(
                create_driver_client(output_dir),
                model=VLAClient(vla_endpoint),
                hide_object_coords=perception,
                video_path=str(Path(output_dir) / "episode.mp4"),
            )

    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats: dict = {}
    try:
        result = cerebrum.solve(
            system_prompt=system_prompt,
            user_message=user_msg,
            tools_spec=tool_registry.get_tools_spec(),
            tool_handler=tool_registry.execute_tool,
            tool_result_formatter=tool_result_to_content_blocks,
            max_turns=max_turns,
        )
        finish_result = result.finish_result
        messages = result.messages
        stats = result.stats
        agent_error = result.error
    except Exception as e:
        agent_error = f"{type(e).__name__}: {e}"
        logger.error("EXCEPTION in agent loop: %s", agent_error)
    finally:
        # Salvage: if the sim reached libero_terminated=True before the
        # agent crashed (or before it called finish), still write a
        # minimal recipe + audit so the run isn't lost.
        try:
            _emergency_save(output_dir, suite, task, seed, recipe_tag,
                            agent_error, regime=regime, verbose=verbose)
        except Exception as e:
            logger.error("emergency save failed: %s", e)
        if proc is not None:
            stop_driver(
                proc,
                output_dir=output_dir,
                stop_recording_and_save=env_spec.stop_recording_and_save,
            )
        stop_vla_server(vla_proc)

    elapsed = time.time() - t0

    transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
    record = {
        "suite": suite, "task": task, "seed": seed, "model": model,
        "elapsed_s": round(elapsed, 1),
        "finish": finish_result,
        "stats": stats,
        "messages": _serialize_messages(messages),
    }
    with open(transcript_path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    if verbose:
        logger.info("elapsed: %.1fs", elapsed)
        logger.info("usage: in=%s out=%s tool_calls=%s",
                     stats.get('total_input_tokens', '?'),
                     stats.get('total_output_tokens', '?'),
                     stats.get('tool_calls', '?'))
        logger.info("transcript: %s", transcript_path)
        if agent_error:
            logger.error("error: %s", agent_error)
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Standalone hybrid LLM-in-the-loop agent for LIBERO PRO",
    )
    ap.add_argument("--suite", required=True,
                    help="e.g. libero_object_task, libero_spatial_swap")
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env", dest="env_name", default=None,
                    help="Environment backend. Defaults to suite inference/libero.")
    ap.add_argument("--model", default=None,
                    help="Model id. Defaults to the selected backend's model env var.")
    ap.add_argument("--max_turns", type=int, default=80)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--max_episode_steps", type=int, default=600)
    ap.add_argument("--cuda_device", default=None,
                    help="GPU device. Defaults to CUDA_DEVICE env or 0.")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--api_key", default=None,
                    help="API key. Defaults to the selected backend's API key env var.")
    ap.add_argument("--base_url", default=None,
                    help="API base URL. Defaults to the selected backend's base URL env var.")
    ap.add_argument("--cerebrum", default="anthropic",
                    choices=["anthropic", "openai_compat", "claude_code", "codex"],
                    help="LLM backend: anthropic | openai_compat | claude_code | codex.")
    ap.add_argument("--openai_compat_no_images", action="store_true",
                    help="Do not send tool-result images to an openai_compat model.")
    ap.add_argument("--thinking", action="store_true",
                    help="Enable extended thinking / reasoning for anthropic and "
                         "openai_compat backends (no-op for claude_code/codex).")
    ap.add_argument("--claude_code_timeout_s", type=int, default=None,
                    help="Wall-clock cap for claude -p. Defaults to CELL_TIMEOUT_S, "
                         "or 1200 in --perception mode / 600 otherwise.")
    ap.add_argument("--claude_code_max_budget_usd", type=float, default=None,
                    help="Budget passed to claude -p --max-budget-usd. "
                         "Defaults to MAX_BUDGET_USD env or 10.")
    ap.add_argument("--codex_timeout_s", type=int, default=None,
                    help="Wall-clock cap for codex exec. Defaults to CODEX_TIMEOUT_S, "
                         "or CELL_TIMEOUT_S, or 1200 in --perception mode / 600 otherwise.")
    ap.add_argument("--no_driver", action="store_true",
                    help="Don't spawn driver; attach to existing output dir")
    ap.add_argument("--transport_host", default="127.0.0.1",
                    help="Socket transport bind/connect host.")
    ap.add_argument("--transport_port", type=int, default=0,
                    help="Socket transport port. 0 asks the OS for a free port; "
                         "required with --no_driver to point at an existing driver.")
    ap.add_argument("--vla_endpoint", default=None,
                    help="Base URL of an existing vla_server (e.g. http://host:8000). "
                         "If omitted with a spawned driver, a local vla_server is started; "
                         "required with --no_driver.")
    ap.add_argument("--perception", action="store_true",
                    help="PERCEPTION-ISOLATED mode: hide object coords, "
                         "use camera+depth+back_project for localization.")
    ap.add_argument("--libero_type", default=None,
                    choices=["standard", "pro", "plus"],
                    help="LIBERO variant (auto-routed from suite suffix if not set).")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    ap = _build_argparser()
    args = ap.parse_args()

    if args.cerebrum == "openai_compat":
        api_key = args.api_key or get_openai_compat_api_key()
        base_url = args.base_url or get_openai_compat_base_url()
    elif args.cerebrum == "anthropic":
        api_key = args.api_key or get_anthropic_api_key()
        base_url = args.base_url or get_anthropic_base_url()
    else:
        api_key = args.api_key
        base_url = args.base_url
    if args.cerebrum == "anthropic" and not api_key:
        logger.error("ANTHROPIC_API_KEY env var or --api_key must be set")
        return 2
    if args.cerebrum == "openai_compat" and not api_key:
        logger.error("OPENAI_COMPAT_API_KEY or OPENAI_API_KEY or --api_key must be set")
        return 2

    run_one_cell(
        suite=args.suite, task=args.task, seed=args.seed,
        api_key=api_key,
        model=args.model,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        max_episode_steps=args.max_episode_steps,
        cuda_device=args.cuda_device,
        output_dir=args.output_dir,
        no_driver=args.no_driver,
        verbose=not args.quiet,
        base_url=base_url,
        perception=args.perception,
        libero_type=args.libero_type,
        cerebrum_type=args.cerebrum,
        openai_compat_supports_images=not args.openai_compat_no_images,
        thinking=args.thinking,
        claude_code_timeout_s=args.claude_code_timeout_s,
        claude_code_max_budget_usd=args.claude_code_max_budget_usd,
        codex_timeout_s=args.codex_timeout_s,
        transport_host=args.transport_host,
        transport_port=args.transport_port,
        vla_endpoint=args.vla_endpoint,
        env_name=args.env_name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
