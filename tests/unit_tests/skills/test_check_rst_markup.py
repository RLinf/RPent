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

SCRIPT = Path(__file__).resolve().parents[3] / ".agents/skills/docs/check_rst_markup.py"
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


def test_list_table_rows_and_bullet_items_do_not_merge_blocks(tmp_path):
    # Consecutive list-table rows without blank lines must be checked as
    # independent blocks so an unclosed marker in row 1 is reported as ERROR
    # rather than paired with a marker in row 2 and downgraded to WARNING.
    table_rst = tmp_path / "table.rst"
    table_rst.write_text(
        ".. list-table::\n"
        "   :header-rows: 1\n"
        "\n"
        "   * - 参数\n"
        "     - 说明\n"
        "   * - 配置 ``KEY``（说明）\n"
        "     - 另一行 ``OTHER`` 正常\n",
        encoding="utf-8",
    )
    findings = rst_check["check_file"](table_rst)
    assert any(f.level == "ERROR" for f in findings)
    assert rst_check["main"](["--errors-only", str(table_rst)]) == 1


def test_whitespace_after_open_delimiter_does_not_false_warn_on_cjk_prefix(tmp_path):
    # A leading space inside (** 重点**) leaves ** unparsed as a start-string;
    # the valid CJK prefix '（' must not be reported as an invalid prefix.
    rst_file = tmp_path / "prefix_ws.rst"
    rst_file.write_text(
        "标题\n====\n\n说明（** 重点**）内容。\n",
        encoding="utf-8",
    )
    findings = rst_check["check_file"](rst_file)
    assert not any("preceded by" in f.message for f in findings)


def test_warnings_report_silent_misrenders_and_respect_errors_only(tmp_path, capsys):
    rst_file = tmp_path / "warn.rst"
    rst_file.write_text(
        "标题\n====\n\n通常**并不是**这样，配置 ``KEY``（说明）与 ``OTHER`` 正常。\n",
        encoding="utf-8",
    )
    findings = rst_check["check_file"](rst_file)
    assert [f.level for f in findings] == ["WARNING", "WARNING"]
    assert rst_check["main"](["--errors-only", str(rst_file)]) == 0
    assert "No inline-markup problems found." in capsys.readouterr().out


def test_even_length_asterisk_adornment_and_code_blocks_are_skipped(tmp_path):
    # Even-length runs of 4+ asterisks (****) must not emit a trailing ** marker.
    rst_file = tmp_path / "adornment.rst"
    rst_file.write_text(
        "****\n"
        "标题\n"
        "****\n"
        "\n"
        "正文段落以及 ``****`` 装饰线。\n"
        "\n"
        ".. code-block:: bash\n"
        "\n"
        "   echo ``not_rst_prose``中文\n",
        encoding="utf-8",
    )
    assert rst_check["check_file"](rst_file) == []
