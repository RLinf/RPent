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

import ast

import pytest

from components.tools.trace_export.cells import namespace_check, project_code
from components.tools.trace_export.trace import TraceError

METHODS = {"move_pose", "move_delta", "set_gripper", "parallel", "finish"}


def _dump(code):
    return ast.dump(ast.parse(code), include_attributes=False)


def test_literal_explanation_is_removed_and_other_arguments_are_kept():
    code = (
        'r = robo.move_pose(xyz=[0.1, 0.2, 0.3], explanation="Move above the bowl.")\n'
        "robo.set_gripper(gripper=1)\n"
    )
    projected, audit = project_code(code, METHODS)

    assert "explanation" not in projected
    assert "xyz=[0.1, 0.2, 0.3]" in projected
    assert audit["removed_keywords"] == 1
    assert audit["original_sha256"] != audit["projected_sha256"]


def test_code_without_annotations_is_unchanged_in_ast():
    code = "for i in range(3):\n    robo.move_delta(dxyz=[0, 0, 0.01])\n"
    projected, audit = project_code(code, METHODS)

    assert _dump(projected) == _dump(code)
    assert audit["removed_keywords"] == 0


def test_forwarded_parameter_annotation_is_removed():
    code = (
        "def go(xyz, why):\n"
        "    return robo.move_pose(xyz=xyz, explanation=why)\n"
        'go([0, 0, 0], "lift")\n'
    )
    projected, audit = project_code(code, METHODS)

    assert "explanation" not in projected
    assert audit["removed_forwarded_keywords"] == 1


def test_fstring_annotation_keeps_its_evaluation():
    code = 'x = 2\nrobo.move_pose(xyz=[0, 0, 0], explanation=f"step {x}")\n'
    projected, audit = project_code(code, METHODS)

    assert "explanation" not in projected
    assert "f'step {x}'" in projected
    assert audit["preserved_annotation_evaluations"] == 1


def test_parallel_child_annotations_are_removed():
    code = (
        "robo.parallel(actions=[\n"
        '    {"primitive": "move_delta", "dxyz": [0, 0, 0.1], "explanation": "Up."},\n'
        '    {"primitive": "set_gripper", "gripper": 1, "explanation": "Close."},\n'
        "])\n"
    )
    projected, audit = project_code(code, METHODS)

    assert "explanation" not in projected
    assert audit["removed_child_annotations"] == 2


def test_parallel_actions_bound_to_a_name_used_once_are_projected():
    code = (
        "acts = [\n"
        '    {"primitive": "set_gripper", "gripper": 1, "explanation": "Close."},\n'
        "]\n"
        "robo.parallel(actions=acts)\n"
    )
    projected, audit = project_code(code, METHODS)

    assert "explanation" not in projected
    assert audit["removed_child_annotations"] == 1


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("robo.move_pose(xyz=[0, 0, 0], explanation=make_text())\n", "nonliteral"),
        ('robo.move_pose(xyz=[0, 0, 0], explanation="")\n', "invalid explanation"),
        (
            'robo.move_pose(xyz=[0, 0, 0], explanation="' + "a" * 801 + '")\n',
            "invalid explanation",
        ),
        ("r = robo.move_pose\nr(xyz=[0, 0, 0])\n", "aliased/dynamic robo"),
        ("robo = None\n", "aliased/dynamic robo"),
        ("def robo():\n    pass\n", "shadowed robo definition"),
        ("def f(robo):\n    pass\n", "shadowed robo parameter"),
        ("import os as robo\n", "shadowed robo import"),
        (
            'r = robo.finish(status="success")\nprint(r["explanation"])\n',
            "removed explanation field",
        ),
        ("exec('robo.finish()')\n", "dynamic namespace"),
        ("robo.move_pose(**kw)\n", "dynamic robo kwargs"),
        (
            'robo.parallel(actions=[], explanation="all")\n',
            "parallel accepts actions only",
        ),
        (
            "robo.parallel(actions=make())\n",
            "dynamic parallel actions",
        ),
        (
            'acts = [{"primitive": "set_gripper", "explanation": "x"}]\n'
            "robo.parallel(actions=acts)\n"
            "print(acts)\n",
            "reused outside its action group",
        ),
        ("robo.move_pose(\n", "does not compile"),
    ],
)
def test_unsafe_or_ambiguous_code_is_rejected(code, reason):
    with pytest.raises(TraceError, match=reason):
        project_code(code, METHODS)


def test_method_outside_the_public_set_is_rejected():
    with pytest.raises(TraceError, match="aliased/dynamic robo"):
        project_code("robo.secret_tool()\n", METHODS)


def test_dict_kwargs_without_an_annotation_key_are_accepted():
    code = 'kw = {"gripper": 1}\nrobo.move_pose(xyz=[0, 0, 0], **kw)\n'
    projected, _ = project_code(code, METHODS)

    assert _dump(projected) == _dump(code)


def test_namespace_check_accumulates_definitions_across_cells():
    defined = namespace_check("target = [0.1, 0.2, 0.3]\n", set())
    defined = namespace_check("robo.move_pose(xyz=target)\nreached = True\n", defined)

    assert {"target", "reached"} <= defined


def test_namespace_check_rejects_names_from_a_missing_cell():
    with pytest.raises(TraceError, match="missing namespace definitions: target"):
        namespace_check("robo.move_pose(xyz=target)\n", set())


def test_namespace_check_sees_globals_used_inside_functions():
    with pytest.raises(TraceError, match="missing namespace definitions: offset"):
        namespace_check("def f():\n    return offset\n", set())


def test_namespace_check_accepts_names_the_runtime_injects():
    defined = namespace_check("print(data.state())\n", set(), injected=("robo", "data"))

    assert defined == set()
