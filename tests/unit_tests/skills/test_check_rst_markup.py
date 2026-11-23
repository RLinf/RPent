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

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytest.importorskip("docutils")

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".agents/skills/docs-check/check_rst_markup.py"
)
rst_check = runpy.run_path(str(SCRIPT))


def test_check_rst_markup_detects_cjk_adjacency_and_accepts_escaped_spaces(tmp_path):
    bad_rst = tmp_path / "bad.rst"
    bad_rst.write_text(
        "标题\n====\n\n配置 ``KEY``（说明）会导致构建报错。\n",
        encoding="utf-8",
    )
    assert rst_check["main"]([str(bad_rst)]) == 1

    good_rst = tmp_path / "good.rst"
    good_rst.write_text(
        "标题\n====\n\n配置 ``KEY``\\ （说明）可以正常构建。\n",
        encoding="utf-8",
    )
    assert rst_check["main"]([str(good_rst)]) == 0
