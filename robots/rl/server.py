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

"""Warm GPU simulator worker pool HTTP service (LIBERO/RoboTwin/RoboCasa/RoboDojo).

Each worker is a persistent subprocess pinned to one GPU with the simulator
runtime already imported. Episodes open/execute/close directly in-process over
a private Unix domain socket (length-prefixed pickle), avoiding cold starts,
extra env_server IPC hops, and disk artifact serialization during RL rollouts.
Execute responses stream raw PNG bytes via binary framing when clients send
``Accept: application/vnd.rpent.execute+octet-stream``, eliminating Base64
encode/decode overhead across IPC.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import json
import os
import pickle
import resource
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from rpent.utils.logging import get_logger

logger = get_logger("rl_sim_pool")

BINARY_EXECUTE_MEDIA_TYPE = "application/vnd.rpent.execute+octet-stream"


# ---------------------------------------------------------------------------
# Binary HTTP framing for /sessions/{sid}/execute (zero Base64)
# ---------------------------------------------------------------------------


def pack_execute_response(meta: dict[str, Any], images: Sequence[bytes]) -> bytes:
    """Pack execute metadata JSON + raw PNG byte payloads without Base64."""
    meta_bytes = json.dumps(meta, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    parts = [
        struct.pack("!I", len(meta_bytes)),
        meta_bytes,
        struct.pack("!I", len(images)),
    ]
    for img in images:
        parts.append(struct.pack("!I", len(img)))
        parts.append(img)
    return b"".join(parts)


def unpack_execute_response(payload: bytes) -> tuple[dict[str, Any], list[bytes]]:
    """Unpack binary execute payload into ``(meta_dict, raw_png_images)``."""
    mv = memoryview(payload)
    if len(mv) < 8:
        raise ValueError("Truncated binary execute response")
    (meta_len,) = struct.unpack_from("!I", mv, 0)
    pos = 4 + meta_len
    meta = json.loads(bytes(mv[4:pos]).decode("utf-8"))
    (n_images,) = struct.unpack_from("!I", mv, pos)
    pos += 4
    images: list[bytes] = []
    for _ in range(n_images):
        (img_len,) = struct.unpack_from("!I", mv, pos)
        pos += 4
        images.append(bytes(mv[pos : pos + img_len]))
        pos += img_len
    return meta, images


# ---------------------------------------------------------------------------
# Length-prefixed binary framing over Unix domain socket
# ---------------------------------------------------------------------------


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        part = conn.recv(n - len(chunks))
        if not part:
            raise EOFError("worker socket closed")
        chunks.extend(part)
    return bytes(chunks)


def _send_msg(conn: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("!I", len(data)) + data)


def _recv_msg(conn: socket.socket) -> Any:
    (size,) = struct.unpack("!I", _recv_exact(conn, 4))
    return pickle.loads(_recv_exact(conn, size))


async def _async_send(writer: asyncio.StreamWriter, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    writer.write(struct.pack("!I", len(data)) + data)
    await writer.drain()


async def _async_recv(reader: asyncio.StreamReader) -> Any:
    header = await reader.readexactly(4)
    (size,) = struct.unpack("!I", header)
    body = await reader.readexactly(size)
    return pickle.loads(body)


# ---------------------------------------------------------------------------
# Persistent GPU simulator worker subprocess
# ---------------------------------------------------------------------------


def _configure_worker_gpu(environment: str, gpu: int | None) -> None:
    if gpu is None:
        return
    if environment in {"libero", "robocasa", "metaworld"}:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from rpent.utils.egl import configure_egl_device

        configure_egl_device(gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def _preload_simulator(environment: str, assets_path: str | None) -> None:
    """Import heavy simulator libraries once so episode opens stay warm."""
    if environment == "libero":
        os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import torch  # noqa: F401

        from robots.rl.envs import libero  # noqa: F401
    elif environment == "metaworld":
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import metaworld  # noqa: F401
        import mujoco  # noqa: F401

        import robots.rl.envs.metaworld  # noqa: F401
    elif environment == "robotwin":
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        if assets_path:
            os.environ["ROBOTWIN_ASSETS_PATH"] = assets_path
            os.environ["ROBOTWIN_ASSETS_ROOT"] = assets_path
        from robots.robotwin import env_server  # noqa: F401


class _WorkerSessionHost:
    """Owns one active simulator scene + RLToolkit inside a worker process."""

    def __init__(
        self,
        environment: str,
        sam3_endpoint: str | None,
        sam3_timeout_s: float,
        assets_path: str | None,
        libero_type: str | None,
        gpu: int | None,
    ) -> None:
        self.environment = environment
        self.sam3_endpoint = sam3_endpoint
        self.sam3_timeout_s = sam3_timeout_s
        self.assets_path = assets_path
        self.libero_type = libero_type
        self.gpu = gpu
        self._sam3 = None
        self._env = None
        self._daemon = None
        self._toolkit = None

    def _get_sam3(self):
        if self._sam3 is None and self.sam3_endpoint:
            from rpent.robots.components.sam3_client import Sam3Client
            from rpent.utils.rpc import make_rpc_client

            self._sam3 = Sam3Client(
                make_rpc_client(self.sam3_endpoint), timeout_s=self.sam3_timeout_s
            )
        return self._sam3

    def _build_env(
        self,
        task: dict[str, Any],
        max_episode_steps: int,
        *,
        record_artifacts: bool = False,
    ):
        env_name = self.environment
        if env_name == "metaworld":
            from robots.rl.envs.metaworld import LocalMetaWorldEnv

            seed = int(
                task.get("seed", task.get("task_index", task.get("init_state_id", 0)))
            )
            return LocalMetaWorldEnv(
                task_name=str(task["task_name"]),
                index=seed,
                max_episode_steps=max_episode_steps,
                task_language=task.get("task_language"),
            )
        if env_name == "libero":
            from robots.rl.envs.libero import LocalLiberoEnv

            return LocalLiberoEnv(
                suite=str(task["suite"]),
                task=int(task["task"]),
                seed=int(task["seed"]),
                libero_type=task.get("libero_type") or self.libero_type,
                max_episode_steps=max_episode_steps,
                record_artifacts=record_artifacts,
            )
        if env_name == "robotwin":
            from robots.rl.envs.robotwin import LocalRoboTwinEnv

            assets = str(task.get("assets_path") or self.assets_path or "")
            if not assets:
                raise ValueError("RoboTwin worker requires --assets-path")
            return LocalRoboTwinEnv(
                task_name=str(task["task_name"]),
                task_config=str(task.get("task_config", "demo_randomized")),
                seed=int(task["seed"]),
                assets_path=assets,
                max_episode_steps=max_episode_steps,
                record_artifacts=record_artifacts,
            )
        if env_name == "robocasa":
            return self._spawn_remote_env(
                "robots/robocasa/env_server.py",
                [
                    "--task-name",
                    str(task["task_name"]),
                    "--split",
                    str(task.get("split", "target")),
                    "--seed",
                    str(int(task["seed"])),
                    "--camera-h",
                    "1024",
                    "--camera-w",
                    "1024",
                ],
                lambda rpc: __import__(
                    "robots.robocasa.env_client", fromlist=["RoboCasaEnvClient"]
                ).RoboCasaEnvClient(rpc),
            )
        if env_name == "robodojo":
            return self._spawn_remote_env(
                "robots/robodojo/env_server.py",
                [
                    "--task",
                    str(task["task_name"]),
                    "--layout",
                    str(int(task.get("layout", 0))),
                    "--seed",
                    str(int(task["seed"])),
                    "--max-episode-steps",
                    str(max_episode_steps),
                    "--mode",
                    "eval-fair",
                    "--episode-protocol",
                    "--headless",
                    "--enable_cameras",
                ],
                lambda rpc: __import__(
                    "robots.robodojo.env_client", fromlist=["RoboDojoEnvClient"]
                ).RoboDojoEnvClient(rpc, record_vpp2_history=False),
            )
        raise ValueError(f"Unsupported simulator: {env_name}")

    def _spawn_remote_env(self, script: str, script_args: list[str], client_factory):
        from rpent.utils.config import get_repo_root
        from rpent.utils.daemon import ProcessDaemon
        from rpent.utils.rpc import make_rpc_client, wait_for_endpoint, wait_for_ready

        log_file = Path(tempfile.mkstemp(prefix="rpent-env-", suffix=".log")[1])
        cuda = ["--cuda-device", str(self.gpu)] if self.gpu is not None else []
        daemon = ProcessDaemon(
            name=f"{self.environment}_env",
            cmd=[
                sys.executable,
                str(get_repo_root() / script),
                *script_args,
                "--transport",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--parent-watch",
                *cuda,
            ],
            log_path=str(log_file),
        )
        daemon.start()
        try:
            endpoint = wait_for_endpoint(daemon, timeout_s=300.0)
            rpc = make_rpc_client(endpoint)
            wait_for_ready(rpc, timeout_s=60.0)
            self._daemon = daemon
            return client_factory(rpc)
        except BaseException:
            daemon.stop()
            raise

    def open(self, req: dict[str, Any]) -> dict[str, Any]:
        self.close()
        from robots.rl.toolkit import RLToolkit, openai_tool_schemas, render_prompts

        task = dict(req["task"])
        max_episode_steps = int(req.get("max_episode_steps", 10000))
        max_actions = int(req.get("max_actions_per_call", 3))
        image_size = int(req.get("image_size", 1024))
        reasoning = bool(req.get("reasoning", True))
        record_artifacts = bool(req.get("record_artifacts", False))
        output_dir = req.get("output_dir")
        from rpent.utils.logging import init_output_dir

        init_output_dir(output_dir or tempfile.gettempdir())

        env = self._build_env(
            task, max_episode_steps, record_artifacts=record_artifacts
        )
        self._env = env
        runtime_kwargs: dict[str, Any] = {
            "env": env,
            "sam3_client": self._get_sam3(),
            "max_episode_steps": max_episode_steps,
        }
        toolkit = RLToolkit(
            runtime_kwargs=runtime_kwargs,
            environment=self.environment,
            output_dir=output_dir,
            record_artifacts=record_artifacts,
            image_size=image_size,
            max_actions_per_call=max_actions,
            reasoning=reasoning,
        )
        self._toolkit = toolkit
        sys_prompt, usr_msg = render_prompts(
            self.environment, reasoning=reasoning, max_actions_per_call=max_actions
        )
        init_obs = toolkit.initial_observation()
        init_state_id = task.get("init_state_id")
        if init_state_id is None and hasattr(env, "last_info"):
            init_state_id = env.last_info.get("init_state_id", task.get("seed"))
        return {
            "environment": self.environment,
            "system_prompt": sys_prompt,
            "user_message": usr_msg,
            "tools": openai_tool_schemas(toolkit.list_tools()),
            "initial_observation": init_obs.to_text(),
            "init_state_id": init_state_id,
            "done": toolkit.completion_result() is not None,
            "success": bool(toolkit.solved()),
            "truncated": bool(toolkit.truncated),
        }

    def execute(self, req: dict[str, Any]) -> dict[str, Any]:
        if self._toolkit is None:
            raise RuntimeError("No active session in worker")
        result = self._toolkit.execute_tool(req["name"], req["arguments"])
        return {
            "text": result.to_text(),
            "images": [bytes(img) for img in result.images],
            "done": self._toolkit.completion_result() is not None,
            "success": bool(self._toolkit.solved()),
            "truncated": bool(self._toolkit.truncated),
            "infrastructure_error": self._toolkit.infrastructure_error,
            "sam3_service_errors": list(self._toolkit.sam3_service_errors),
        }

    def close(self) -> dict[str, Any]:
        if self._toolkit is not None:
            try:
                self._toolkit.close()
            finally:
                self._toolkit = None
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None
        if self._daemon is not None:
            try:
                self._daemon.stop()
            finally:
                self._daemon = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return {"closed": True}


def _run_worker(args: argparse.Namespace) -> None:
    _configure_worker_gpu(args.env, args.gpu)
    _preload_simulator(args.env, args.assets_path)
    host = _WorkerSessionHost(
        environment=args.env,
        sam3_endpoint=args.sam3_endpoint,
        sam3_timeout_s=args.sam3_timeout_s,
        assets_path=args.assets_path,
        libero_type=args.libero_type,
        gpu=args.gpu,
    )
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(args.socket)
    _send_msg(
        sock,
        {"ready": True, "worker_id": args.worker_id, "pid": os.getpid()},
    )
    try:
        while True:
            try:
                cmd = _recv_msg(sock)
            except EOFError:
                break
            op = cmd.get("op")
            if op == "shutdown":
                host.close()
                _send_msg(sock, {"ok": True, "result": {"shutdown": True}})
                break
            try:
                if op == "open":
                    result = host.open(cmd)
                elif op == "execute":
                    result = host.execute(cmd)
                elif op == "close":
                    result = host.close()
                else:
                    raise ValueError(f"Unknown worker op: {op}")
                _send_msg(sock, {"ok": True, "result": result})
            except Exception as exc:
                _send_msg(
                    sock,
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
    finally:
        host.close()
        sock.close()


# ---------------------------------------------------------------------------
# Async Warm Pool Manager + FastAPI App
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class _WorkerHandle:
    worker_id: int
    gpu: int | None
    proc: subprocess.Popen
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    episodes: int = 0
    busy: bool = False
    alive: bool = True
    replaced: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def rpc(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self.lock:
            self.busy = True
            try:
                await _async_send(self.writer, payload)
                reply = await _async_recv(self.reader)
            except BaseException:
                self.alive = False
                raise
            finally:
                self.busy = False
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "worker error"))
        return reply["result"]

    def kill(self) -> None:
        self.alive = False
        try:
            self.writer.close()
        except Exception:
            pass
        if self.proc.poll() is None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass


class SimPoolManager:
    """Manages N persistent warm GPU workers and active RL episode sessions."""

    def __init__(
        self,
        *,
        environment: str,
        gpus: list[int],
        num_workers: int,
        sim_python: str,
        sam3_endpoint: str | None,
        sam3_timeout_s: float = 120.0,
        assets_path: str | None = None,
        libero_type: str | None = None,
        max_episodes_per_worker: int = 30,
        sleepable: bool = False,
        max_concurrent_spawns: int = 16,
        log_dir: Path | None = None,
    ) -> None:
        self.environment = environment
        self.gpus = gpus
        self.num_workers = num_workers
        self.sim_python = sim_python
        self.sam3_endpoint = sam3_endpoint
        self.sam3_timeout_s = sam3_timeout_s
        self.assets_path = assets_path
        self.libero_type = libero_type
        self.max_episodes_per_worker = max_episodes_per_worker
        self.sleepable = sleepable
        self.max_concurrent_spawns = max_concurrent_spawns
        self.log_dir = log_dir
        self._sock_dir = Path(tempfile.mkdtemp(prefix=f"rpent-{environment}-pool-"))
        self._sock_path = self._sock_dir / "pool.sock"
        self._server: asyncio.AbstractServer | None = None
        self._ready_waiters: dict[
            int, asyncio.Future[tuple[asyncio.StreamReader, asyncio.StreamWriter]]
        ] = {}
        self._idle: asyncio.Queue[_WorkerHandle] = asyncio.Queue()
        self._sessions: dict[str, _WorkerHandle] = {}
        self._session_cohorts: dict[str, str | None] = {}
        self._workers: set[_WorkerHandle] = set()
        self._next_worker_id = 0
        self._shutting_down = False
        self._sleeping = False
        self._lifecycle_lock = asyncio.Lock()
        self._spawn_sem: asyncio.Semaphore | None = (
            asyncio.Semaphore(max_concurrent_spawns)
            if max_concurrent_spawns > 0
            else None
        )

    async def start(self) -> None:
        if self._sock_path.exists():
            self._sock_path.unlink()
        self._server = await asyncio.start_unix_server(
            self._on_worker_connect, path=str(self._sock_path)
        )
        await asyncio.gather(
            *(
                self._spawn_worker(self.gpus[i % len(self.gpus)] if self.gpus else None)
                for i in range(self.num_workers)
            )
        )
        logger.info(
            "SimPoolManager(%s) ready: %d warm workers on GPUs %s",
            self.environment,
            len(self._workers),
            self.gpus,
        )

    async def _on_worker_connect(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            ready = await asyncio.wait_for(_async_recv(reader), timeout=180.0)
        except Exception:
            writer.close()
            return
        wid = ready.get("worker_id") if ready.get("ready") else None
        waiter = self._ready_waiters.pop(wid, None)
        if waiter is None or waiter.done():
            writer.close()
            return
        waiter.set_result((reader, writer))

    async def _spawn_worker(self, gpu: int | None) -> _WorkerHandle:
        if self._spawn_sem is not None:
            async with self._spawn_sem:
                return await self._spawn_worker_impl(gpu)
        return await self._spawn_worker_impl(gpu)

    async def _spawn_worker_impl(self, gpu: int | None) -> _WorkerHandle:
        wid = self._next_worker_id
        self._next_worker_id += 1
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = (
            loop.create_future()
        )
        self._ready_waiters[wid] = waiter

        cmd = [
            self.sim_python,
            "-m",
            "robots.rl.server",
            "worker",
            "--env",
            self.environment,
            "--socket",
            str(self._sock_path),
            "--worker-id",
            str(wid),
            "--sam3-timeout-s",
            str(self.sam3_timeout_s),
        ]
        if gpu is not None:
            cmd.extend(["--gpu", str(gpu)])
        if self.sam3_endpoint:
            cmd.extend(["--sam3-endpoint", self.sam3_endpoint])
        if self.assets_path:
            cmd.extend(["--assets-path", self.assets_path])
        if self.libero_type:
            cmd.extend(["--libero-type", self.libero_type])

        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")

        log_fh = subprocess.DEVNULL
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_fh = (self.log_dir / f"{self.environment}_worker_{wid}.log").open("ab")

        proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
        if log_fh is not subprocess.DEVNULL:
            log_fh.close()

        try:
            reader, writer = await asyncio.wait_for(waiter, timeout=240.0)
        except BaseException:
            self._ready_waiters.pop(wid, None)
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            raise
        if self._sleeping or self._shutting_down:
            writer.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            raise asyncio.CancelledError(
                "SimPoolManager entered sleep/shutdown during spawn"
            )
        handle = _WorkerHandle(wid, gpu, proc, reader, writer)
        self._workers.add(handle)
        await self._idle.put(handle)
        return handle

    async def _replace_worker(self, handle: _WorkerHandle) -> None:
        if handle.replaced:
            return
        handle.replaced = True
        self._workers.discard(handle)
        handle.kill()
        if not self._shutting_down and not self._sleeping:
            try:
                await self._spawn_worker(handle.gpu)
            except Exception:
                logger.exception(
                    "Failed to respawn %s worker on GPU %s",
                    self.environment,
                    handle.gpu,
                )

    async def _acquire_idle_worker(self, is_disconnected=None) -> _WorkerHandle:
        while True:
            if is_disconnected is not None and await is_disconnected():
                raise asyncio.CancelledError(
                    "Client disconnected before acquiring worker"
                )
            try:
                worker = await asyncio.wait_for(self._idle.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if not worker.alive or worker.proc.poll() is not None:
                asyncio.create_task(self._replace_worker(worker))
                continue
            if is_disconnected is not None and await is_disconnected():
                await self._idle.put(worker)
                raise asyncio.CancelledError(
                    "Client disconnected upon acquiring worker"
                )
            return worker

    async def open_session(
        self, req: dict[str, Any], *, is_disconnected=None
    ) -> dict[str, Any]:
        if self._sleeping:
            raise RuntimeError("SimPoolManager is sleeping; call /pool/wake first")
        worker = await self._acquire_idle_worker(is_disconnected)
        sid = uuid4().hex
        cohort_id = req.get("cohort_id")
        self._sessions[sid] = worker
        self._session_cohorts[sid] = cohort_id
        try:
            result = await worker.rpc({"op": "open", **req})
            if sid not in self._sessions or worker.replaced:
                raise asyncio.CancelledError("Session closed during open")
            if is_disconnected is not None and await is_disconnected():
                await self.close_session(sid)
                raise asyncio.CancelledError("Client disconnected during open")
            return {"session_id": sid, **result}
        except asyncio.CancelledError:
            self._sessions.pop(sid, None)
            self._session_cohorts.pop(sid, None)
            asyncio.create_task(self._replace_worker(worker))
            raise
        except Exception:
            self._sessions.pop(sid, None)
            self._session_cohorts.pop(sid, None)
            if not worker.replaced and worker.alive and worker.proc.poll() is None:
                try:
                    await worker.rpc({"op": "close"})
                    if not worker.replaced and worker.alive:
                        await self._idle.put(worker)
                except Exception:
                    asyncio.create_task(self._replace_worker(worker))
            else:
                asyncio.create_task(self._replace_worker(worker))
            raise

    async def execute(
        self, sid: str, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        worker = self._sessions.get(sid)
        if worker is None:
            raise KeyError(f"Unknown session {sid}")
        return await worker.rpc({"op": "execute", "name": name, "arguments": arguments})

    async def close_cohort(self, cohort_id: str) -> dict[str, Any]:
        sids = [
            sid for sid, cid in list(self._session_cohorts.items()) if cid == cohort_id
        ]
        if sids:
            await asyncio.gather(
                *(self.close_session(sid) for sid in sids), return_exceptions=True
            )
        return {"cohort_id": cohort_id, "closed": len(sids)}

    async def close_session(self, sid: str) -> dict[str, Any]:
        self._session_cohorts.pop(sid, None)
        worker = self._sessions.pop(sid, None)
        if worker is None:
            return {"closed": False, "already_closed": True}
        worker.episodes += 1
        recycle = (
            not worker.alive
            or worker.busy
            or worker.proc.poll() is not None
            or (
                self.max_episodes_per_worker > 0
                and worker.episodes >= self.max_episodes_per_worker
            )
        )
        if worker.replaced or recycle:
            asyncio.create_task(self._replace_worker(worker))
            return {"closed": True, "recycled": True}
        try:
            await asyncio.wait_for(worker.rpc({"op": "close"}), timeout=30.0)
            if not worker.replaced and worker.alive:
                await self._idle.put(worker)
            return {"closed": True, "recycled": False}
        except Exception:
            asyncio.create_task(self._replace_worker(worker))
            return {"closed": True, "recycled": True}

    async def sleep(self) -> dict[str, Any]:
        """Release all GPU workers on sleepable collocated nodes before training."""
        async with self._lifecycle_lock:
            if not self.sleepable:
                return self.health()
            self._sleeping = True
            if self._sessions:
                await asyncio.gather(
                    *(self.close_session(sid) for sid in list(self._sessions)),
                    return_exceptions=True,
                )
            while not self._idle.empty():
                try:
                    self._idle.get_nowait()
                except asyncio.QueueEmpty:
                    break
            for worker in list(self._workers):
                worker.replaced = True
                worker.kill()
            self._workers.clear()
            self._sessions.clear()
            self._session_cohorts.clear()
            logger.info(
                "SimPoolManager(%s) asleep: 0 GPU workers resident", self.environment
            )
            return self.health()

    async def wake(self) -> dict[str, Any]:
        """Respawn warm GPU workers on sleepable collocated nodes after training."""
        async with self._lifecycle_lock:
            if not self.sleepable:
                return self.health()
            if not self._sleeping and len(self._workers) >= self.num_workers:
                return self.health()
            self._sleeping = False
            needed = max(0, self.num_workers - len(self._workers))
            if needed:
                await asyncio.gather(
                    *(
                        self._spawn_worker(
                            self.gpus[i % len(self.gpus)] if self.gpus else None
                        )
                        for i in range(needed)
                    )
                )
            logger.info(
                "SimPoolManager(%s) awake: %d/%d warm workers ready",
                self.environment,
                len(self._workers),
                self.num_workers,
            )
            return self.health()

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "environment": self.environment,
            "workers_target": self.num_workers,
            "workers_total": len(self._workers),
            "workers_idle": self._idle.qsize(),
            "active_sessions": len(self._sessions),
            "sleepable": self.sleepable,
            "sleeping": self._sleeping,
        }

    async def shutdown(self) -> None:
        self._shutting_down = True
        for worker in list(self._workers):
            worker.kill()
        self._workers.clear()
        self._sessions.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if self._sock_path.exists():
            self._sock_path.unlink(missing_ok=True)
        try:
            self._sock_dir.rmdir()
        except OSError:
            pass


class OpenRequest(BaseModel):
    task: dict[str, Any]
    cohort_id: str | None = None
    max_episode_steps: int = 10000
    image_size: int = 1024
    max_actions_per_call: int = 3
    reasoning: bool = True
    record_artifacts: bool = False
    output_dir: str | None = None


class ExecuteRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = {}


def create_app(manager: SimPoolManager) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await manager.start()
        try:
            yield
        finally:
            await manager.shutdown()

    app = FastAPI(title=f"RPent SimPool ({manager.environment})", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return manager.health()

    @app.post("/sessions/open")
    async def open_session(req: OpenRequest, request: Request):
        try:
            return await manager.open_session(
                req.model_dump(), is_disconnected=request.is_disconnected
            )
        except asyncio.CancelledError:
            raise HTTPException(status_code=499, detail="Client disconnected")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/cohorts/{cohort_id}/close")
    async def close_cohort(cohort_id: str):
        return await manager.close_cohort(cohort_id)

    @app.post("/sessions/{session_id}/execute")
    async def execute(session_id: str, req: ExecuteRequest, request: Request):
        try:
            result = await manager.execute(session_id, req.name, req.arguments)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        images: list[bytes] = result.pop("images", [])
        accept = request.headers.get("accept", "")
        if BINARY_EXECUTE_MEDIA_TYPE in accept or "application/octet-stream" in accept:
            return Response(
                content=pack_execute_response(result, images),
                media_type=BINARY_EXECUTE_MEDIA_TYPE,
            )
        result["images_b64"] = [base64.b64encode(img).decode("ascii") for img in images]
        return result

    @app.post("/sessions/{session_id}/close")
    async def close_session(session_id: str):
        return await manager.close_session(session_id)

    @app.post("/pool/sleep")
    async def pool_sleep():
        return await manager.sleep()

    @app.post("/pool/wake")
    async def pool_wake():
        return await manager.wake()

    return app


# ---------------------------------------------------------------------------
# Multi-Node Async SimPoolRouter (Zero-Base64 Binary Passthrough)
# ---------------------------------------------------------------------------


class SimPoolRouter:
    """Load-balance sessions across multiple remote SimPoolManager HTTP backends."""

    def __init__(
        self,
        *,
        environment: str,
        backends: Sequence[str],
        request_timeout_s: float = 1800.0,
    ) -> None:
        if not backends:
            raise ValueError("SimPoolRouter requires at least one backend URL")
        self.environment = environment
        self.backends = [url.rstrip("/") for url in backends]
        self.request_timeout_s = request_timeout_s
        self._client: Any = None
        self._capacity: dict[str, int] = dict.fromkeys(self.backends, 1)
        self._reserved: dict[str, int] = dict.fromkeys(self.backends, 0)
        self._sessions: dict[str, tuple[str, str | None]] = {}

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30.0, read=self.request_timeout_s, write=120.0, pool=60.0
            ),
            limits=httpx.Limits(
                max_connections=1024,
                max_keepalive_connections=256,
                keepalive_expiry=30.0,
            ),
            trust_env=False,
        )
        await self._refresh_capacities()
        logger.info(
            "SimPoolRouter(%s) ready across %d backends: total_target=%d (%s)",
            self.environment,
            len(self.backends),
            sum(self._capacity.values()),
            self._capacity,
        )

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _refresh_capacities(self) -> list[dict[str, Any]]:
        assert self._client is not None

        async def _probe(url: str) -> dict[str, Any]:
            try:
                resp = await self._client.get(f"{url}/health", timeout=15.0)
                resp.raise_for_status()
                data = resp.json()
                target = int(data.get("workers_target", data.get("workers_total", 0)))
                if target > 0:
                    self._capacity[url] = target
                return {"url": url, **data}
            except Exception as exc:
                return {
                    "url": url,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        return list(await asyncio.gather(*(_probe(url) for url in self.backends)))

    async def health(self) -> dict[str, Any]:
        rows = await self._refresh_capacities()
        ok = all(bool(r.get("ok")) for r in rows)
        return {
            "ok": ok,
            "environment": self.environment,
            "workers_target": sum(int(r.get("workers_target", 0)) for r in rows),
            "workers_total": sum(int(r.get("workers_total", 0)) for r in rows),
            "workers_idle": sum(int(r.get("workers_idle", 0)) for r in rows),
            "active_sessions": sum(int(r.get("active_sessions", 0)) for r in rows),
            "sleepable": any(bool(r.get("sleepable")) for r in rows),
            "sleeping": any(bool(r.get("sleeping")) for r in rows),
            "backends": rows,
        }

    def _pick_backend(self) -> str:
        def _key(url: str) -> tuple[int, float, int]:
            cap = max(self._capacity.get(url, 1), 1)
            res = self._reserved.get(url, 0)
            full = 1 if res >= cap else 0
            return (full, res / cap, res)

        return min(self.backends, key=_key)

    async def open_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self._client is not None
        backend = self._pick_backend()
        self._reserved[backend] += 1
        try:
            try:
                resp = await self._client.post(f"{backend}/sessions/open", json=payload)
            except httpx.TransportError as exc:
                logger.warning(
                    "SimPoolRouter(%s) transport error opening session on %s: %s",
                    self.environment,
                    backend,
                    exc,
                )
                raise HTTPException(
                    status_code=503,
                    detail=f"Backend {backend} transport error on open: {exc}",
                ) from exc
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            data = resp.json()
            sid = str(data["session_id"])
            self._sessions[sid] = (backend, payload.get("cohort_id"))
            return data
        except BaseException:
            self._reserved[backend] = max(0, self._reserved[backend] - 1)
            raise

    async def execute(
        self, session_id: str, raw_body: bytes, accept: str
    ) -> tuple[int, bytes, str]:
        assert self._client is not None
        entry = self._sessions.get(session_id)
        if entry is None:
            raise KeyError(f"Unknown session {session_id}")
        backend, _ = entry
        headers = {"Content-Type": "application/json"}
        if accept:
            headers["Accept"] = accept
        try:
            resp = await self._client.post(
                f"{backend}/sessions/{session_id}/execute",
                content=raw_body,
                headers=headers,
            )
        except httpx.TransportError as exc:
            logger.warning(
                "SimPoolRouter(%s) transport error executing session %s on %s: %s",
                self.environment,
                session_id,
                backend,
                exc,
            )
            raise HTTPException(
                status_code=503,
                detail=f"Backend {backend} transport error on execute: {exc}",
            ) from exc
        media_type = resp.headers.get("content-type", "application/json")
        return resp.status_code, resp.content, media_type

    async def close_session(self, session_id: str) -> dict[str, Any]:
        assert self._client is not None
        entry = self._sessions.pop(session_id, None)
        if entry is None:
            return {"closed": False, "already_closed": True}
        backend, _ = entry
        self._reserved[backend] = max(0, self._reserved[backend] - 1)
        try:
            resp = await self._client.post(
                f"{backend}/sessions/{session_id}/close", json={}, timeout=60.0
            )
        except httpx.TransportError as exc:
            logger.warning(
                "SimPoolRouter(%s) transport error closing session %s on %s: %s",
                self.environment,
                session_id,
                backend,
                exc,
            )
            raise HTTPException(
                status_code=503,
                detail=f"Backend {backend} transport error on close: {exc}",
            ) from exc
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()

    async def close_cohort(self, cohort_id: str) -> dict[str, Any]:
        assert self._client is not None
        for sid, (backend, cid) in list(self._sessions.items()):
            if cid == cohort_id:
                self._sessions.pop(sid, None)
                self._reserved[backend] = max(0, self._reserved[backend] - 1)

        async def _close(url: str) -> dict[str, Any]:
            resp = await self._client.post(
                f"{url}/cohorts/{cohort_id}/close", json={}, timeout=60.0
            )
            resp.raise_for_status()
            return resp.json()

        results = await asyncio.gather(
            *(_close(url) for url in self.backends), return_exceptions=True
        )
        closed = sum(int(r.get("closed", 0)) for r in results if isinstance(r, dict))
        return {"cohort_id": cohort_id, "closed": closed}

    async def lifecycle(self, action: str) -> dict[str, Any]:
        assert self._client is not None
        assert action in ("sleep", "wake")
        timeout_s = 600.0 if action == "wake" else 120.0
        if action == "sleep":
            self._sessions.clear()
            for url in self.backends:
                self._reserved[url] = 0

        async def _call(url: str) -> dict[str, Any]:
            resp = await self._client.post(
                f"{url}/pool/{action}", json={}, timeout=timeout_s
            )
            resp.raise_for_status()
            return {"url": url, **resp.json()}

        rows = list(await asyncio.gather(*(_call(url) for url in self.backends)))
        for row in rows:
            target = int(row.get("workers_target", 0))
            if target > 0:
                self._capacity[row["url"]] = target
        return {
            "ok": all(bool(r.get("ok")) for r in rows),
            "action": action,
            "environment": self.environment,
            "workers_target": sum(int(r.get("workers_target", 0)) for r in rows),
            "workers_total": sum(int(r.get("workers_total", 0)) for r in rows),
            "workers_idle": sum(int(r.get("workers_idle", 0)) for r in rows),
            "active_sessions": sum(int(r.get("active_sessions", 0)) for r in rows),
            "sleeping": any(bool(r.get("sleeping")) for r in rows),
            "backends": rows,
        }


def create_router_app(router: SimPoolRouter) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await router.start()
        try:
            yield
        finally:
            await router.shutdown()

    app = FastAPI(
        title=f"RPent SimPoolRouter ({router.environment})", lifespan=lifespan
    )

    @app.get("/health")
    async def health():
        return await router.health()

    @app.post("/sessions/open")
    async def open_session(req: OpenRequest):
        return await router.open_session(req.model_dump())

    @app.post("/cohorts/{cohort_id}/close")
    async def close_cohort(cohort_id: str):
        return await router.close_cohort(cohort_id)

    @app.post("/sessions/{session_id}/execute")
    async def execute(session_id: str, request: Request):
        raw_body = await request.body()
        accept = request.headers.get("accept", "")
        try:
            status_code, content, media_type = await router.execute(
                session_id, raw_body, accept
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=content, status_code=status_code, media_type=media_type)

    @app.post("/sessions/{session_id}/close")
    async def close_session(session_id: str):
        return await router.close_session(session_id)

    @app.post("/pool/sleep")
    async def pool_sleep():
        return await router.lifecycle("sleep")

    @app.post("/pool/wake")
    async def pool_wake():
        return await router.lifecycle("wake")

    return app


def _raise_nofile_limit(target: int = 1048576) -> None:
    """Raise soft RLIMIT_NOFILE up to ``min(target, hard)`` for high-concurrency pools."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if soft < desired:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
    except (OSError, ValueError):
        pass


def main() -> None:
    _raise_nofile_limit()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_serve = sub.add_parser("serve", help="Run the HTTP warm worker pool server")
    p_serve.add_argument(
        "--env",
        choices=("libero", "robotwin", "robocasa", "robodojo", "metaworld"),
        required=True,
    )
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, required=True)
    p_serve.add_argument("--gpus", type=int, nargs="*", default=[])
    p_serve.add_argument("--workers", type=int, default=4)
    p_serve.add_argument("--sim-python", default=sys.executable)
    p_serve.add_argument("--sam3-endpoint", default=None)
    p_serve.add_argument("--sam3-timeout-s", type=float, default=120.0)
    p_serve.add_argument("--assets-path", default=None)
    p_serve.add_argument("--libero-type", default=None)
    p_serve.add_argument("--max-episodes-per-worker", type=int, default=30)
    p_serve.add_argument("--sleepable", action="store_true")
    p_serve.add_argument("--max-concurrent-spawns", type=int, default=16)
    p_serve.add_argument("--log-dir", type=Path, default=None)

    p_router = sub.add_parser(
        "router", help="Run a multi-host SimPoolRouter over remote SimPoolManagers"
    )
    p_router.add_argument(
        "--env",
        choices=("libero", "robotwin", "robocasa", "robodojo", "metaworld"),
        required=True,
    )
    p_router.add_argument("--host", default="127.0.0.1")
    p_router.add_argument("--port", type=int, required=True)
    p_router.add_argument("--backends", nargs="+", required=True)
    p_router.add_argument("--request-timeout-s", type=float, default=1800.0)

    p_worker = sub.add_parser("worker", help="Internal persistent GPU worker process")
    p_worker.add_argument(
        "--env",
        choices=("libero", "robotwin", "robocasa", "robodojo", "metaworld"),
        required=True,
    )
    p_worker.add_argument("--socket", required=True)
    p_worker.add_argument("--worker-id", type=int, default=0)
    p_worker.add_argument("--gpu", type=int, default=None)
    p_worker.add_argument("--sam3-endpoint", default=None)
    p_worker.add_argument("--sam3-timeout-s", type=float, default=120.0)
    p_worker.add_argument("--assets-path", default=None)
    p_worker.add_argument("--libero-type", default=None)

    args = parser.parse_args()
    if args.mode == "worker":
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        _run_worker(args)
        return
    if args.mode == "router":
        router = SimPoolRouter(
            environment=args.env,
            backends=args.backends,
            request_timeout_s=args.request_timeout_s,
        )
        app = create_router_app(router)
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="warning",
            backlog=2048,
            timeout_keep_alive=300,
        )
        return

    manager = SimPoolManager(
        environment=args.env,
        gpus=args.gpus,
        num_workers=args.workers,
        sim_python=args.sim_python,
        sam3_endpoint=args.sam3_endpoint,
        sam3_timeout_s=args.sam3_timeout_s,
        assets_path=args.assets_path,
        libero_type=args.libero_type,
        max_episodes_per_worker=args.max_episodes_per_worker,
        sleepable=args.sleepable,
        max_concurrent_spawns=args.max_concurrent_spawns,
        log_dir=args.log_dir,
    )
    app = create_app(manager)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        backlog=2048,
        timeout_keep_alive=300,
    )


if __name__ == "__main__":
    main()
