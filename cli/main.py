"""Physical agent main CLI entrypoint."""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from physical_agent.utils.config import (
    get_cuda_device,
    get_libero_type,
    get_repo_root,
)

from physical_agent.cerebrum.base import build_cerebrum  # noqa: E402
from physical_agent.envs import get_env_spec, get_toolkit  # noqa: E402
from physical_agent.driver_client import (  # noqa: E402
    create_driver_client,
    get_socket_endpoint,
    set_socket_endpoint,
)
from physical_agent.driver_client.vla_client import VLAClient  # noqa: E402
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


def start_env_server(
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
    """Launch the env server in background. The env server hosts the 
    env, and prints a machine-readable ``transport_ready`` event on stdout
    once its RPC server is listening; this function returns once that event
    is seen.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if log_path is None:
        log_path = str(out_dir / "env_server.log")

    env = os.environ.copy()
    env["LIBERO_TYPE"] = libero_type
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device or get_cuda_device())
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("ROBOT_PLATFORM", "LIBERO")

    cmd = [
        sys.executable,
        driver_script or str(get_repo_root() / "deployment" / "rlinf" / "env_server.py"),
        "--suite", suite,
        "--task", str(task),
        "--seed", str(seed),
        "--max_episode_steps", str(max_episode_steps),
        "--output_dir", str(out_dir),
        "--transport_host", transport_host,
        "--transport_port", str(transport_port),
    ]
    logger.info("env server cmd: %s", ' '.join(cmd))
    logger.info("env server log: %s", log_path)
    logger.info("CUDA_VISIBLE_DEVICES=%s  output_dir=%s", cuda_device, out_dir)
    log_f = open(log_path, "w")
    ready_events: queue.Queue[dict] = queue.Queue()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=get_repo_root(),
        text=True,
        bufsize=1,
    )
    threading.Thread(
        target=_pipe_driver_output,
        args=(proc, log_f, ready_events),
        daemon=True,
    ).start()

    logger.info("waiting for env server...")
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
                "env server ready at %s:%s",
                event["host"],
                event["port"],
            )
            break
        if proc.poll() is not None:
            logger.error("env server EXITED before becoming ready. Last log:")
            logger.error("%s", Path(log_path).read_text()[-2000:])
            raise RuntimeError("env server exited prematurely")
        if time.time() - t0 > ready_timeout_s:
            proc.terminate()
            raise RuntimeError(f"env server not ready after {ready_timeout_s}s")
    logger.info("env server ready in %.1fs", time.time()-t0)
    return proc


def stop_env_server(
    proc: subprocess.Popen,
    output_dir: str,
    timeout: float = 15.0,
) -> None:
    if proc.poll() is not None:
        return
    try:
        client = create_driver_client(output_dir)
        client.call("shutdown", timeout_s=timeout)
    except Exception:
        pass
    try:
        proc.wait(timeout=timeout)
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
        str(get_repo_root() / "deployment" / "rlinf" / "vla_server.py"),
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
            raise RuntimeError("vla server exited prematurely")
        try:
            if client.healthz():
                logger.info("vla server ready at %s after %.1fs", base_url, time.time() - t0)
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


def _strip_images(value):
    """Return a copy of ``value`` with inline image payloads omitted.

    SDK objects are left untouched; ``json.dump(..., default=str)`` handles
    them at write time. Only the bulky base64 image blocks are replaced.
    """
    if isinstance(value, list):
        return [_strip_images(v) for v in value]
    if isinstance(value, dict):
        if value.get("type") == "image":
            return {"type": "image", "source": {"_omitted_for_transcript": True}}
        if value.get("type") == "image_url":
            return {"type": "image_url", "image_url": {"_omitted_for_transcript": True}}
        return {k: _strip_images(v) for k, v in value.items()}
    return value


def _serialize_messages(messages: list[dict]) -> list[dict]:
    """Strip inline image payloads from messages before writing the transcript."""
    return [
        {**{k: v for k, v in m.items() if k != "content"},
         "content": _strip_images(m.get("content"))}
        for m in messages
    ]


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
    ap.add_argument("--env", dest="env_name", default="libero",
                    help="Environment backend. Defaults to libero.")
    ap.add_argument("--model", default=None,
                    help="Model id. Defaults to the selected backend's model env var.")
    ap.add_argument("--max_turns", type=int, default=100)
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

    suite = args.suite
    task = args.task
    seed = args.seed
    env_name = args.env_name
    toolkit = get_toolkit(env_name)
    env_spec = get_env_spec(env_name)
    prompt_bundle = env_spec.prompts

    max_episode_steps = args.max_episode_steps
    if max_episode_steps == 600 and "libero_10" in suite:
        max_episode_steps = 5000
        logger.info("auto-bumped max_episode_steps to 5000 for libero_10")

    # resolve output directory
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_{suite}_t{task}_s{seed}"
    output_dir = init_output_dir(output_dir)

    recipe_tag = f"{suite.replace('libero_', '')}_t{task}_s{seed}"

    cerebrum = build_cerebrum(
        args.cerebrum,
        output_dir=output_dir,
        env_name=env_spec.name,
        recipe_tag=recipe_tag,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        perception=args.perception,
        thinking=args.thinking,
        claude_code_timeout_s=args.claude_code_timeout_s,
        claude_code_max_budget_usd=args.claude_code_max_budget_usd,
        codex_timeout_s=args.codex_timeout_s,
        transport_host=args.transport_host,
        transport_port=args.transport_port,
    )

    # Auto-route LIBERO_TYPE if not set
    libero_type = args.libero_type or get_libero_type()

    if args.cerebrum in {"claude_code", "codex"}:
        system_prompt = ""
        template = prompt_bundle.cli_prompt_template(perception=args.perception)
        user_msg = prompt_bundle.format_claude_code_prompt(
            template,
            suite=suite, task=task, seed=seed,
            output_dir=output_dir, recipe_tag=recipe_tag,
        )
    else:
        user_msg = prompt_bundle.api_user_message(
            perception=args.perception,
            suite=suite,
            task=task,
            seed=seed,
            output_dir=output_dir,
            recipe_tag=recipe_tag,
        )
        system_prompt = prompt_bundle.api_system_prompt(perception=args.perception)

    proc = None
    vla_proc = None
    vla_endpoint = args.vla_endpoint
    if not args.no_driver:
        proc = start_env_server(
            suite=suite, task=task, seed=seed,
            output_dir=output_dir,
            max_episode_steps=max_episode_steps,
            cuda_device=args.cuda_device,
            libero_type=libero_type,
            transport_host=args.transport_host,
            transport_port=args.transport_port,
        )
        if vla_endpoint is None:
            vla_endpoint, vla_proc = start_vla_server(
                cuda_device=args.cuda_device,
                log_path=str(Path(output_dir) / "vla_server.log"),
            )
        if args.cerebrum in {"claude_code", "codex"}:
            endpoint = get_socket_endpoint(output_dir)
            if endpoint is None:
                raise RuntimeError(
                    f"socket endpoint not registered for output_dir: {output_dir}"
                )
            cerebrum.set_socket_endpoint(*endpoint)
            cerebrum.set_vla_endpoint(vla_endpoint)
        else:
            toolkit.set_driver_client(
                create_driver_client(output_dir),
                model=VLAClient(vla_endpoint),
                hide_object_coords=args.perception,
                video_path=str(Path(output_dir) / "episode.mp4"),
            )
    else:
        if args.transport_port <= 0:
            raise RuntimeError(
                "--no_driver requires --transport_port pointing at an existing driver"
            )
        if vla_endpoint is None:
            raise RuntimeError(
                "--no_driver requires --vla_endpoint pointing at an existing vla_server"
            )
        set_socket_endpoint(output_dir, args.transport_host, args.transport_port)
        if args.cerebrum in {"claude_code", "codex"}:
            cerebrum.set_socket_endpoint(args.transport_host, args.transport_port)
            cerebrum.set_vla_endpoint(vla_endpoint)
        else:
            toolkit.set_driver_client(
                create_driver_client(output_dir),
                model=VLAClient(vla_endpoint),
                hide_object_coords=args.perception,
                video_path=str(Path(output_dir) / "episode.mp4"),
            )

    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats: dict = {}
    try:
        result = cerebrum.solve(
            system_prompt=system_prompt,
            user_message=user_msg,
            toolkit=toolkit,
            max_turns=args.max_turns,
        )
        finish_result = result.finish_result
        messages = result.messages
        stats = result.stats
        agent_error = result.error
    except Exception as e:
        logger.error("EXCEPTION in agent loop: %s", e)
    finally:
        if proc is not None:
            # Agent-side: flush the episode video before the env+model
            # process dies, then shut down the env server.
            toolkit.release_driver_client()
            stop_env_server(proc, output_dir=output_dir)
        stop_vla_server(vla_proc)

    elapsed = time.time() - t0

    transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
    record = {
        "suite": suite, "task": task, "seed": seed, "model": args.model,
        "elapsed_s": round(elapsed, 1),
        "finish": finish_result,
        "stats": stats,
        "messages": _serialize_messages(messages),
    }
    with open(transcript_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    logger.info("elapsed: %.1fs", elapsed)
    logger.info("usage: in=%s out=%s tool_calls=%s",
                 stats.get('total_input_tokens', '?'),
                 stats.get('total_output_tokens', '?'),
                 stats.get('tool_calls', '?'))
    logger.info("transcript: %s", transcript_path)
    if agent_error:
        logger.error("error: %s", agent_error)
    return 0


if __name__ == "__main__":
    sys.exit(main())
