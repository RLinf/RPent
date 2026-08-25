from pathlib import Path

from robots.libero.spec import LIBERO_DASHBOARD_SPEC
from rpent.dashboard.events import RunStartedEvent, StepRecordEvent, UsageEvent
from rpent.dashboard.state import DashboardState
from rpent.tools.state import EnvState


def _dashboard_state(tmp_path: Path) -> DashboardState:
    state = DashboardState(
        run_id="dashboard-session/test",
        output_dir=tmp_path,
        dashboard_spec=LIBERO_DASHBOARD_SPEC,
    )
    state.shared_services_ready()
    state.request_task({"suite": "libero_10", "task": 0, "seed": 0})
    assert state.wait_for_task() is not None
    state.emit(RunStartedEvent())
    return state


def _record_action(state: EnvState, action: str):
    with state.record_step(state={}, command={"action": action}, result={}):
        state.save(f"action_{action}.mp4", b"video")
    return state.get()


def test_dashboard_reopens_interaction_for_new_planner_session(tmp_path):
    state = _dashboard_state(tmp_path)
    state.emit(UsageEvent(inp=10, out=4, tool_calls=2))
    state.set_planner_activity("ended")

    video_path = tmp_path / "sessions" / "session_002" / "episode.mp4"
    state.begin_planner_session(video_path=video_path)
    state.emit(UsageEvent(inp=3, out=2, tool_calls=1))

    assert state.planner_activity == "starting"
    assert state.video_path == video_path
    assert state.snapshot()["usage"] == {"in": 13, "out": 6, "tool_calls": 3}


def test_dashboard_keeps_continuous_steps_across_toolkits(tmp_path):
    dashboard = _dashboard_state(tmp_path)
    first = EnvState(tmp_path / "sessions" / "session_001")
    second = EnvState(tmp_path / "sessions" / "session_002")

    with first.record_step(state={}):
        pass
    first_action = _record_action(first, "move_to")
    dashboard.emit(StepRecordEvent(first.get(0), first, {}))
    dashboard.emit(StepRecordEvent(first_action, first, {}))

    with second.record_step(state={}):
        pass
    second_action = _record_action(second, "release")
    dashboard.emit(StepRecordEvent(second.get(0), second, {}))
    dashboard.emit(StepRecordEvent(second_action, second, {}))

    detail = dashboard.run_detail()
    assert [item["step"] for item in detail["timeline"]] == [1, 3]
    assert detail["frame_idx"] == 3
    assert dashboard.action_video_path(1) == first.artifact_path(
        "action_move_to.mp4",
        step=1,
    )
    assert dashboard.action_video_path(3) == second.artifact_path(
        "action_release.mp4",
        step=1,
    )
