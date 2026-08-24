"""Tests for RoboTwin native evaluator-state compatibility."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from robots.robotwin.evaluator_state import initialize_native_evaluator_state


class _PoseSource:
    def __init__(self, *, position=None, quaternion=None):
        self._pose = SimpleNamespace(p=position, q=quaternion)

    def get_pose(self):
        return self._pose


def _task_class(name: str):
    return type(name, (), {})


@pytest.fixture
def robotwin_utils(monkeypatch):
    utils = ModuleType("robotwin.envs.utils")
    utils.get_face_prod = lambda *_args: 1.0
    monkeypatch.setitem(sys.modules, "robotwin.envs.utils", utils)
    return utils


@pytest.mark.parametrize(
    ("face_product", "expected_arm"), [(1.0, "left"), (-1.0, "right")]
)
def test_open_laptop_initializes_arm_tag_without_actions(
    robotwin_utils, face_product, expected_arm
):
    task = _task_class("open_laptop")()
    task.laptop = _PoseSource(quaternion=[1.0, 0.0, 0.0, 0.0])
    task.take_action_cnt = 0
    task.policy_actions = 0
    task.native_actions = 0
    robotwin_utils.get_face_prod = lambda *_args: face_product

    initialize_native_evaluator_state(task)

    assert task.arm_tag == expected_arm
    assert task.take_action_cnt == 0
    assert task.policy_actions == 0
    assert task.native_actions == 0


@pytest.mark.parametrize(
    ("task_name", "position", "expected_arm"),
    [
        ("place_object_scale", [0.2, 0.0, 0.4], "right"),
        ("place_object_scale", [-0.2, 0.0, 0.4], "left"),
        ("put_object_cabinet", [0.2, 0.0, 0.4], "right"),
        ("put_object_cabinet", [-0.2, 0.0, 0.4], "left"),
    ],
)
def test_object_tasks_initialize_native_checker_state(
    task_name, position, expected_arm
):
    task = _task_class(task_name)()
    task.object = _PoseSource(position=position)
    task.take_action_cnt = 0
    task.policy_actions = 0
    task.native_actions = 0

    initialize_native_evaluator_state(task)

    assert task.arm_tag == expected_arm
    assert task.take_action_cnt == 0
    assert task.policy_actions == 0
    assert task.native_actions == 0
    if task_name == "put_object_cabinet":
        assert task.origin_z == position[2]
    else:
        assert not hasattr(task, "origin_z")


def test_unrelated_task_is_unchanged():
    task = _task_class("beat_block_hammer")()
    task.take_action_cnt = 0

    initialize_native_evaluator_state(task)

    assert vars(task) == {"take_action_cnt": 0}


def test_does_not_call_expert_or_privileged_helpers(robotwin_utils):
    task = _task_class("open_laptop")()
    task.laptop = _PoseSource(quaternion=[1.0, 0.0, 0.0, 0.0])
    task.play_once = lambda: pytest.fail("play_once must not be called")
    task.get_info = lambda: pytest.fail("get_info must not be called")

    initialize_native_evaluator_state(task)


def test_checker_initialization_error_propagates(robotwin_utils):
    task = _task_class("open_laptop")()
    task.laptop = _PoseSource(quaternion=[1.0, 0.0, 0.0, 0.0])

    def fail(*_args):
        raise RuntimeError("checker failure")

    robotwin_utils.get_face_prod = fail

    with pytest.raises(RuntimeError, match="checker failure"):
        initialize_native_evaluator_state(task)
