"""Shared spawn/wait/stop helpers for robot runtime components.

These orchestrate the lifecycle of subprocess-backed servers (env, vla,
sam3) that implement the ``init_*_runtime`` contract on
:class:`~rpent.robots.robot_spec.RobotSpec`. Each helper reports state
transitions to a :class:`~rpent.dashboard.events.DashboardEventSink` so
the dashboard UI stays in sync with the actual processes.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rpent.dashboard.events import DashboardEventSink, RuntimeStatusEvent
from rpent.utils.daemon import ProcessDaemon
from rpent.utils.rpc import wait_for_ready

if TYPE_CHECKING:
    from rpent.utils.rpc import RpcClient


def stop_owned_daemons(
    daemons: dict[str, ProcessDaemon],
    dashboard_events: DashboardEventSink,
) -> None:
    """Stop owned daemons in reverse insertion order, marking each killed
    component ``pending`` on the dashboard so the UI doesn't keep showing
    ``ready``/``starting`` for processes that no longer exist.
    """
    for component, daemon in reversed(list(daemons.items())):
        try:
            daemon.stop()
        except Exception:
            pass
        dashboard_events.emit(RuntimeStatusEvent(component, "pending"))


def try_spawn_server(
    daemons: dict[str, ProcessDaemon],
    dashboard_events: DashboardEventSink,
    component: str,
    spawn_fn: Callable[[], tuple[ProcessDaemon | None, RpcClient]],
) -> tuple[ProcessDaemon | None, RpcClient]:
    dashboard_events.emit(RuntimeStatusEvent(component, "starting"))
    try:
        daemon, rpc = spawn_fn()
        if daemon is not None:
            daemons[component] = daemon
        return daemon, rpc
    except Exception as exc:
        stop_owned_daemons(daemons, dashboard_events)
        dashboard_events.emit(RuntimeStatusEvent(component, "failed", error=exc))
        raise RuntimeError(f"[{component}] spawn failed: {exc}") from exc


def try_wait_server(
    daemons: dict[str, ProcessDaemon],
    dashboard_events: DashboardEventSink,
    component: str,
    rpc: RpcClient,
    daemon: ProcessDaemon | None,
    timeout_s: float,
    *,
    post_fn: Callable[[], Any] | None = None,
):
    try:
        wait_for_ready(rpc, daemon=daemon, timeout_s=timeout_s)
        result = None
        if post_fn is not None:
            result = post_fn()
    except Exception as exc:
        stop_owned_daemons(daemons, dashboard_events)
        dashboard_events.emit(RuntimeStatusEvent(component, "failed", error=exc))
        raise RuntimeError(f"[{component}] wait / client connect failed: {exc}") from exc
    dashboard_events.emit(RuntimeStatusEvent(component, "ready"))
    return result
