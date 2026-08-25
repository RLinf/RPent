from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from robots.libero.spec import LIBERO_DASHBOARD_SPEC
from rpent.cli import dashboard as dashboard_cli
from rpent.dashboard.events import RunStartedEvent, StepRecordEvent, UsageEvent
from rpent.dashboard.state import DashboardState
from rpent.planner.base import PlannerResult
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


def test_dashboard_explore_rebuilds_planner_and_toolkit_per_session(tmp_path, monkeypatch):
    dashboard = _dashboard_state(tmp_path)
    claimed = SimpleNamespace(number=1, request={}, output_dir=tmp_path / "task")
    toolkit_calls = []
    planner_calls = []

    class FakeToolkit:
        def __init__(self, solved):
            self._solved = solved
            self.closed = False

        def solved(self):
            return self._solved

        def write_recipe(self, recipe_tag):
            return f"recipe_{recipe_tag}.jsonl"

        def close(self):
            self.closed = True

    class FakePlanner:
        def __init__(self, solved):
            self.solved = solved

        def solve(self, **kwargs):
            planner_calls.append(kwargs)
            dashboard.set_planner_activity("ended")
            return PlannerResult(messages=[{"session": len(planner_calls)}])

    toolkits = [FakeToolkit(False), FakeToolkit(True)]

    def get_toolkit(*args, **kwargs):
        toolkit_calls.append(kwargs)
        return toolkits[len(toolkit_calls) - 1]

    def build_planner(*args, **kwargs):
        return FakePlanner(toolkits[len(planner_calls)].solved())

    def init_output_dir(path, verbose=False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    robot_spec = SimpleNamespace(
        parse_config=lambda args: SimpleNamespace(
            recipe_tag="10_t0_s0",
            output_dir=Path(args.output_dir),
            prompt_vars={},
            task_desc={},
        ),
        init_task_runtime=lambda *args: ([], {}),
        prompts=SimpleNamespace(render=lambda name, variables: name),
        finalize_run=None,
    )
    args = Namespace(
        robot_name="libero",
        verbose=False,
        explore=True,
        explore_sessions=2,
        explore_attempts_per_session=5,
        planner="api",
        base_url=None,
        model="test",
        max_tokens=100,
        planner_timeout_s=None,
        claude_code_max_budget_usd=None,
        no_images=False,
        max_turns=10,
    )
    monkeypatch.setattr(dashboard_cli, "get_toolkit", get_toolkit)
    monkeypatch.setattr(dashboard_cli, "build_planner", build_planner)
    monkeypatch.setattr(dashboard_cli, "init_output_dir", init_output_dir)

    error = dashboard_cli._run_dashboard_task(
        args=args,
        robot_spec=robot_spec,
        state=dashboard,
        claimed=claimed,
        shared_primitives_kwargs={},
        session_root=tmp_path,
    )

    assert error is None
    assert len(planner_calls) == 2
    assert all(toolkit.closed for toolkit in toolkits)
    assert [call["state_output_dir"] for call in toolkit_calls] == [
        claimed.output_dir / "sessions" / "session_001",
        claimed.output_dir / "sessions" / "session_002",
    ]


def test_dashboard_finalization_failure_is_non_fatal(tmp_path, monkeypatch):
    dashboard = _dashboard_state(tmp_path)
    claimed = SimpleNamespace(number=1, request={}, output_dir=tmp_path / "task")

    class FakeToolkit:
        def solved(self):
            return True

        def write_recipe(self, recipe_tag):
            return f"recipe_{recipe_tag}.jsonl"

        def close(self):
            pass

    class FakePlanner:
        def solve(self, **kwargs):
            dashboard.set_planner_activity("ended")
            return PlannerResult(messages=[])

    def init_output_dir(path, verbose=False):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def finalize_run(*args):
        raise AttributeError("invalid memory")

    robot_spec = SimpleNamespace(
        parse_config=lambda args: SimpleNamespace(
            recipe_tag="10_t0_s0",
            output_dir=Path(args.output_dir),
            prompt_vars={},
            task_desc={},
        ),
        init_task_runtime=lambda *args: ([], {}),
        prompts=SimpleNamespace(render=lambda name, variables: name),
        finalize_run=finalize_run,
    )
    args = Namespace(
        robot_name="libero",
        verbose=False,
        explore=True,
        explore_sessions=1,
        explore_attempts_per_session=5,
        planner="api",
        base_url=None,
        model="test",
        max_tokens=100,
        planner_timeout_s=None,
        claude_code_max_budget_usd=None,
        no_images=False,
        max_turns=10,
    )
    monkeypatch.setattr(
        dashboard_cli,
        "get_toolkit",
        lambda *args, **kwargs: FakeToolkit(),
    )
    monkeypatch.setattr(
        dashboard_cli,
        "build_planner",
        lambda *args, **kwargs: FakePlanner(),
    )
    monkeypatch.setattr(dashboard_cli, "init_output_dir", init_output_dir)

    error = dashboard_cli._run_dashboard_task(
        args=args,
        robot_spec=robot_spec,
        state=dashboard,
        claimed=claimed,
        shared_primitives_kwargs={},
        session_root=tmp_path,
    )

    assert error is None
    assert dashboard.snapshot()["control_feedback"][-1] == (
        "Task succeeded, but memory finalization failed: "
        "AttributeError: invalid memory"
    )
