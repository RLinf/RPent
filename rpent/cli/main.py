"""Physical agent main CLI entrypoint."""
# `rpent/cli/`
#
# CLI entrypoints for RPent (currently just `main.py`).
#
# ## Run
#
# `main()` is exposed as the `rpent` console script (see `[project.scripts]`
# in `pyproject.toml`):
#
# ```bash
# rpent --suite libero_object_task --task 0 --seed 0 [...]
# ```
#
# ## Note
#
# Do not import `rpent.cli` from other `rpent` modules. `main.py` pulls in
# `rpent.planner`, `rpent.envs`, `rpent.utils`, `rpent.dashboard`, and
# `rpent.tools`, so importing the CLI back into any of them would create an
# import cycle. Nothing else should depend on this package.
from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from robots.libero.env_client import LiberoEnvClient  # noqa: E402
from rpent.envs import get_env_spec, get_toolkit  # noqa: E402
from rpent.planner.base import build_planner  # noqa: E402
from rpent.utils.config import (
    get_libero_type,
    get_repo_root,
)
from rpent.utils.logging import get_logger, init_output_dir  # noqa: E402
from rpent.utils.rpc import (  # noqa: E402
    create_rpc_client,
    set_socket_endpoint,
)
from rpent.utils.sam3_client import Sam3Client  # noqa: E402
from rpent.utils.vla_client import VLAClient  # noqa: E402

logger = get_logger("agent")


def _pipe_driver_output(
    proc: subprocess.Popen,
    log_file,
    ready_events: "queue.Queue[dict]",
) -> None:
    """Copy env_server stdout to log and capture machine-readable ready events."""
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
    max_episode_steps: int = 10000,
    libero_type: str | None = None,
    cuda_device: str | None = None,
    log_path: str | None = None,
    driver_script: str | None = None,
    ready_timeout_s: float = 300.0,
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
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("ROBOT_PLATFORM", "LIBERO")

    cmd = [
        sys.executable,
        driver_script or str(get_repo_root() / "robots" / "libero" / "env_server.py"),
        "--suite", suite,
        "--task", str(task),
        "--seed", str(seed),
        "--max-episode-steps", str(max_episode_steps),
        "--output-dir", str(out_dir),
    ]
    logger.info("env server cmd: %s", ' '.join(cmd))
    logger.info("env server log: %s", log_path)
    logger.info(
        "CUDA_VISIBLE_DEVICES=%s  output_dir=%s",
        env.get("CUDA_VISIBLE_DEVICES"),
        out_dir,
    )
    log_f = open(log_path, "a")
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
        client = create_rpc_client(output_dir)
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
        str(get_repo_root() / "robots" / "libero" / "vla_server.py"),
        "--host", host,
        "--port", str(port),
    ]
    logger.info("vla server cmd: %s", " ".join(cmd))
    if log_path:
        log_f = open(log_path, "a")
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
    stop_vla_server(proc)
    raise RuntimeError("vla_server not ready after 300s")


def stop_vla_server(proc: subprocess.Popen | None, timeout: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


def start_sam3_server(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    cuda_device: str | None = None,
    log_path: str | None = None,
    ready_timeout_s: float = 300.0,
) -> tuple[Sam3Client, subprocess.Popen]:
    """Launch RPent's local SAM3 server and wait for model readiness."""
    if port == 0:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            port = int(sock.getsockname()[1])

    cmd = [
        sys.executable,
        "-m",
        "robots.libero.sam3_server",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if cuda_device is not None:
        cmd.extend(["--cuda-device", str(cuda_device)])

    logger.info("SAM3 server cmd: %s", shlex.join(cmd))
    if log_path:
        logger.info("SAM3 server log: %s", log_path)
        log_file = open(log_path, "a")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=get_repo_root(),
        )
    else:
        proc = subprocess.Popen(cmd, cwd=get_repo_root())

    endpoint = f"http://{host}:{port}"
    client = Sam3Client(endpoint)
    started_at = time.monotonic()
    last_error: Exception | None = None
    while time.monotonic() - started_at < ready_timeout_s:
        if proc.poll() is not None:
            close_sam3(client, proc)
            raise RuntimeError(
                "SAM3 server exited before becoming ready; "
                f"inspect {log_path or 'its stderr'}"
            )
        try:
            client.healthz(timeout_s=1.0)
            logger.info(
                "SAM3 server ready at %s after %.1fs",
                endpoint,
                time.monotonic() - started_at,
            )
            return client, proc
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.5)

    close_sam3(client, proc)
    raise RuntimeError(
        f"SAM3 server not ready after {ready_timeout_s:.0f}s "
        f"(last error: {last_error})"
    )


def close_sam3(
    client: Sam3Client | None,
    proc: subprocess.Popen | None,
    timeout: float = 10.0,
) -> None:
    """Close the SAM3 client and terminate its local server, if owned."""
    if client is not None:
        try:
            client.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to close SAM3 client: %s", exc)
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


def _parse_env_endpoint(value: str) -> tuple[str, int]:
    """Parse an env_server socket endpoint written as ``HOST:PORT``."""
    raw = value.strip()
    try:
        parsed = urlsplit(f"//{raw}")
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid env endpoint: {value!r}") from exc
    if (
        not host
        or port is None
        or not 1 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise argparse.ArgumentTypeError(
            "env endpoint must be HOST:PORT, for example 127.0.0.1:9000"
        )
    return host, port


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Standalone hybrid LLM-in-the-loop agent for LIBERO PRO",
    )

    # models
    ap.add_argument("--planner", default="api",
                    choices=["api", "claude_code", "codex"],
                    help="LLM backend: api | claude_code | codex.")
    ap.add_argument("--model", default=None,
                    help="Model id. For the 'api' planner, prefix the provider "
                         "(e.g. anthropic:claude-opus-4-8, openai:gpt-5.5, "
                         "openai-chat:glm-5.2). For claude_code/codex this "
                         "overrides the backend default model.")
    ap.add_argument("--base-url", default=None,
                    help="API base URL. Defaults to the selected backend's base URL env var.")
    ap.add_argument("--api-key", default=None,
                    help="API key. Defaults to the selected backend's API key env var.")
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--planner-timeout-s", type=int, default=None,
                    help="Wall-clock cap for the claude_code/codex planner "
                         "subprocess. Defaults to CODEX_TIMEOUT_S (codex only), "
                         "CELL_TIMEOUT_S, or 1200.")
    ap.add_argument("--claude-code-max-budget-usd", type=float, default=None,
                    help="Budget passed to claude -p --max-budget-usd. "
                         "Defaults to MAX_BUDGET_USD env or 10.")

    # env_server / vla_server / sam3_server / transport
    ap.add_argument(
        "--env-endpoint",
        type=_parse_env_endpoint,
        metavar="HOST:PORT",
        default=None,
        help="Socket endpoint of an existing env_server. If omitted, a local "
        "env_server is started.",
    )
    ap.add_argument("--vla-endpoint", default=None,
                    help="Base URL of an existing vla_server (e.g. http://host:8000). "
                         "If omitted, a local vla_server is started.")
    ap.add_argument(
        "--sam3-endpoint",
        default=None,
        help="Base URL of an existing RPent SAM3 service. If omitted, a local "
        "SAM3 server is started automatically.",
    )
    ap.add_argument("--cuda-device", default=None,
                    help="GPU device(s) to expose via CUDA_VISIBLE_DEVICES.")

    # other config
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dashboard", action="store_true",
                    help="Start a local dashboard server for this single run.")
    ap.add_argument("--dashboard-host", default="127.0.0.1",
                    help="Dashboard bind host. Defaults to 127.0.0.1.")
    ap.add_argument("--dashboard-port", type=int, default=0,
                    help="Dashboard port. 0 asks the OS for a free port.")
    ap.add_argument("--dashboard-language", choices=["en", "zh-cn"], default="en",
                    help="Dashboard UI language. 'zh-cn' serves the Chinese "
                         "variant (index.zh-cn.html); defaults to English.")
    ap.add_argument("--verbose", action="store_true",
                    help="Enable DEBUG-level logging for stdout and the run.log "
                         "file. Defaults to INFO when not set.")

    # environments
    ap.add_argument("--env", dest="env_name", default="libero",
                    help="Environment backend. Defaults to libero.")
    ap.add_argument("--max-episode-steps", type=int, default=10000)

    ap.add_argument("--libero-type", default=None,
                    choices=["standard", "pro", "plus"],
                    help="LIBERO variant (auto-routed from suite suffix if not set).")
    ap.add_argument("--suite", default=None,
                    help="e.g. libero_object_task, libero_spatial_swap")
    ap.add_argument("--task", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)

    return ap


def main() -> int:
    parser = _build_argparser()
    args = parser.parse_args()

    # With --dashboard, open the launcher first: serve the start screen, then
    # block until the user clicks Run and overlay their choices onto args.
    # Everything downstream (output_dir, State, run loop) then sees final args.
    dashboard_server = None
    dashboard_url = None
    launch_config = None
    if args.dashboard:
        from rpent.dashboard import DashboardServer
        from rpent.dashboard.launcher import apply_to_args, defaults_from_args

        dashboard_server = DashboardServer(
            host=args.dashboard_host, port=args.dashboard_port,
            language=args.dashboard_language,
        )
        dashboard_url = dashboard_server.start()
        # The run directory is not final until the launcher form is submitted, so
        # print the pre-launch URL without initializing the run.log file handler.
        print(
            f"Dashboard: {dashboard_url}. "
            "Open it, adjust the run config, and click Run to start.",
            flush=True,
        )
        launch_config = dashboard_server.wait_for_launch(
            defaults=defaults_from_args(args)
        )
        apply_to_args(args, launch_config)

    if not args.suite:
        parser.error("--suite is required")
    if args.task is None:
        parser.error("--task is required")

    suite = args.suite
    task = args.task
    seed = args.seed
    env_name = args.env_name
    env_spec = get_env_spec(env_name)
    prompt_bundle = env_spec.prompts

    max_episode_steps = args.max_episode_steps

    # resolve output directory
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_{suite}_t{task}_s{seed}"
    output_dir = init_output_dir(output_dir, verbose=args.verbose)
    # Now that output_dir is fixed, repeat launcher details into this run's log.
    if dashboard_url is not None:
        logger.info("Dashboard: %s", dashboard_url)
    if launch_config is not None:
        logger.info("launcher config applied: %s", launch_config)
    logger.info("physical agent cmd: %s", shlex.join([sys.executable, *sys.argv]))

    recipe_tag = f"{suite.replace('libero_', '')}_t{task}_s{seed}"

    dashboard_state = None
    if args.dashboard and dashboard_server is not None:
        from rpent.dashboard.state import State

        dashboard_state = State(
            run_id=f"{suite}/{output_dir.name}",
            name=recipe_tag,
            suite=suite,
            task=task,
            seed=seed,
            output_dir=str(output_dir),
            video_path=str(Path(output_dir) / "episode.mp4"),
        )
        # Server is already serving the launcher; register the run so the
        # frontend can switch from the start screen to the live monitor.
        dashboard_server.register(dashboard_state)

    planner = build_planner(
        args.planner,
        output_dir=output_dir,
        recipe_tag=recipe_tag,
        env_name=env_name,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        planner_timeout_s=args.planner_timeout_s,
        claude_code_max_budget_usd=args.claude_code_max_budget_usd,
        dashboard=dashboard_state,
    )

    # Auto-route LIBERO_TYPE if not set
    libero_type = args.libero_type or get_libero_type()

    prompt_vars = {
        "suite": suite,
        "task": task,
        "seed": seed,
        "output_dir": output_dir,
        "recipe_tag": recipe_tag,
    }
    system_prompt = prompt_bundle.render(
        "system",
        variables=prompt_vars,
    )
    user_msg = prompt_bundle.render(
        "user",
        variables=prompt_vars,
    )

    env_proc = None
    vla_proc = None
    sam3_proc = None
    sam3_client = None
    vla_endpoint = args.vla_endpoint
    if args.env_endpoint is not None:
        env_host, env_port = args.env_endpoint
        set_socket_endpoint(output_dir, env_host, env_port)
    else:
        env_proc = start_env_server(
            suite=suite, task=task, seed=seed,
            output_dir=output_dir,
            max_episode_steps=max_episode_steps,
            cuda_device=args.cuda_device,
            libero_type=libero_type,
        )

    try:
        if vla_endpoint is None:
            vla_endpoint, vla_proc = start_vla_server(
                cuda_device=args.cuda_device,
                log_path=str(Path(output_dir) / "vla_server.log"),
            )

        if args.sam3_endpoint:
            sam3_client = Sam3Client(args.sam3_endpoint)
            sam3_client.wait_for_healthz(timeout_s=300.0)
        else:
            sam3_client, sam3_proc = start_sam3_server(
                cuda_device=args.cuda_device,
                log_path=str(Path(output_dir) / "sam3_server.log"),
            )

        toolkit = get_toolkit(
            env_name,
            primitives_kwargs={
                "env": LiberoEnvClient(
                    create_rpc_client(output_dir),
                    expected_meta={
                        "suite": suite,
                        "task": task,
                        "seed": seed,
                        "max_episode_steps": max_episode_steps,
                    },
                ),
                "model": VLAClient(vla_endpoint),
            },
            sam3_client=sam3_client,
            video_path=str(Path(output_dir) / "episode.mp4"),
            dashboard=dashboard_state,
        )
    except Exception:
        close_sam3(sam3_client, sam3_proc)
        if vla_proc is not None:
            stop_vla_server(vla_proc)
        if env_proc is not None:
            stop_env_server(env_proc, output_dir=output_dir)
        raise

    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats: dict = {}
    try:
        result = planner.solve(
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
        # Agent-side: flush the episode video before the env+model
        recipe_path = toolkit.write_recipe(recipe_tag)
        logger.info("recipe: %s", recipe_path)

        toolkit.close()
        close_sam3(sam3_client, sam3_proc)
        if vla_proc is not None:
            stop_vla_server(vla_proc)
        if env_proc is not None:
            stop_env_server(env_proc, output_dir=output_dir)

    elapsed = time.time() - t0

    transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
    record = {
        "suite": suite, "task": task, "seed": seed, "model": args.model,
        "elapsed_s": round(elapsed, 1),
        "finish": finish_result,
        "stats": stats,
        "messages": _serialize_messages(messages),
    }
    with open(transcript_path, "a") as f:
        json.dump(record, f, indent=2, default=str)

    logger.info("elapsed: %.1fs", elapsed)
    logger.info("usage: in=%s out=%s tool_calls=%s",
                 stats.get('total_input_tokens', '?'),
                 stats.get('total_output_tokens', '?'),
                 stats.get('tool_calls', '?'))
    logger.info("transcript: %s", transcript_path)
    if agent_error:
        logger.error("error: %s", agent_error)

    if args.dashboard and dashboard_state is not None:
        dashboard_state.mark_done()
        logger.info(
            "Run finished. Dashboard still serving at %s. Press Ctrl+C to stop.",
            dashboard_url,
        )
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
