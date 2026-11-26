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

import datetime
import runpy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
code_check = runpy.run_path(str(REPO_ROOT / ".agents/skills/code/check.py"))
route = runpy.run_path(str(REPO_ROOT / ".agents/skills/route.py"))


def test_check_init_detects_missing_init_and_passes_repo(tmp_path):
    assert code_check["check_init"](REPO_ROOT) == []

    pkg = tmp_path / "rpent" / "new_pkg"
    pkg.mkdir(parents=True)
    (pkg / "mod.py").write_text("x = 1\n", encoding="utf-8")
    findings = code_check["check_init"](tmp_path)
    assert len(findings) == 1
    assert findings[0].rule == "code/init"
    assert findings[0].path == "rpent/new_pkg"


def test_check_header_validates_rpent_copyright_and_year(tmp_path):
    year = datetime.date.today().year
    good = tmp_path / "good.py"
    good.write_text(f"# Copyright {year} The RPent Authors.\n", encoding="utf-8")
    assert code_check["check_header"]("good.py", tmp_path) == []

    old = tmp_path / "old.py"
    old.write_text("# Copyright 2020 The RPent Authors.\n", encoding="utf-8")
    old_findings = code_check["check_header"]("old.py", tmp_path)
    assert len(old_findings) == 1
    assert old_findings[0].severity == "minor"

    missing = tmp_path / "missing.py"
    missing.write_text("x = 1\n", encoding="utf-8")
    missing_findings = code_check["check_header"]("missing.py", tmp_path)
    assert len(missing_findings) == 1
    assert missing_findings[0].severity == "major"


def test_check_test_layout_flags_top_level_unit_test_file():
    Change = route["Change"]
    top = code_check["check_test_layout"](Change("A", "tests/unit_tests/test_stray.py"))
    assert len(top) == 1
    assert top[0].rule == "code/test-layout"

    nested = code_check["check_test_layout"](
        Change("A", "tests/unit_tests/rpent/cli/test_foo.py")
    )
    assert nested == []


def test_check_shell_syntax_detects_broken_bash_script(tmp_path):
    ok = tmp_path / "ok.sh"
    ok.write_text("#!/usr/bin/env bash\necho hi\n", encoding="utf-8")
    assert code_check["check_shell_syntax"]("ok.sh", tmp_path) == []

    bad = tmp_path / "bad.sh"
    bad.write_text("#!/usr/bin/env bash\nif [[ true; then\n", encoding="utf-8")
    findings = code_check["check_shell_syntax"]("bad.sh", tmp_path)
    assert len(findings) == 1
    assert findings[0].severity == "critical"
    assert findings[0].rule == "code/shell-syntax"
