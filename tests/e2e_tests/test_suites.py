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

"""Generic pytest entrypoint for declarative GPU E2E suites."""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from tests.e2e_tests.runner import (
    publish_check,
    run_case,
    run_sidecar,
    selected_cuda_ordinals,
)
from tests.e2e_tests.suite import load_suite, resolve_case


def _selected_cases() -> list[Any]:
    suite_name = os.environ.get("RPENT_E2E_SUITE", "").strip()
    if not suite_name:
        return []
    suite = load_suite(suite_name)
    case_filter = os.environ.get("RPENT_E2E_CASE", "").strip()
    cases = suite["cases"]
    if case_filter:
        cases = [c for c in cases if c["name"] == case_filter]
        if not cases:
            available = ", ".join(c["name"] for c in suite["cases"])
            raise ValueError(
                f"Unknown case {case_filter!r} in suite {suite_name!r}; "
                f"available: {available or 'none'}"
            )
    return [
        pytest.param(suite, case, id=f"{suite_name}:{case['name']}") for case in cases
    ]


@pytest.mark.parametrize(("suite", "case"), _selected_cases())
def test_suite_case(
    suite: dict[str, Any],
    case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = Path(os.environ.get("RPENT_E2E_OUTPUT_DIR", str(tmp_path)))
    case_output = output_root / case["name"]
    gpus = [str(g) for g in selected_cuda_ordinals()]
    resolved = resolve_case(suite, case, output_dir=case_output, gpu=gpus[0], gpus=gpus)
    if resolved["missing"]:
        pytest.skip("; ".join(resolved["missing"]))

    sidecar = resolved["sidecar"]
    sidecar_ctx = (
        run_sidecar(sidecar, output_dir=case_output)
        if sidecar is not None
        else nullcontext("")
    )
    with sidecar_ctx as sidecar_endpoint:
        robot_argv = [
            arg.replace("{sidecar_endpoint}", sidecar_endpoint)
            for arg in resolved["args"]
        ]
        for key, value in resolved["env"].items():
            resolved_val = value.replace("{sidecar_endpoint}", sidecar_endpoint)
            if (
                resolved_val.startswith(f"{case_output}/")
                and not Path(resolved_val).suffix
            ):
                Path(resolved_val).mkdir(parents=True, exist_ok=True)
            monkeypatch.setenv(key, resolved_val)

        result = run_case(
            robot=resolved["robot"],
            robot_argv=robot_argv,
            output_dir=case_output,
            action_tool=resolved["action_tool"],
            action_count_field=resolved["action_count_field"],
            tool_args=resolved["tool_args"],
            skip_tools=resolved["skip_tools"],
            require_post_action_image=resolved["require_post_action_image"],
        )
    assert result["status"] == "passed"
    publish_check(f"{suite['suite']}_{case['name']}", result)
