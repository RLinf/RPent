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

"""Unit tests for component manifest parsing and discovery."""

import pytest

from rpent.install.errors import InstallConfigError
from rpent.install.manifest import discover, load_component


def test_load_valid_manifest_with_overrides_and_git_pins(write_toml, tmp_path):
    (tmp_path / "robots/alpha/override.txt").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "robots/alpha/override.txt").write_text("numpy<2\n", encoding="utf-8")
    path = write_toml(
        "robots/alpha/component.toml",
        """
        schema = 2
        name = "robot-alpha"
        probe = "robots.alpha"
        planner = true
        python = ">=3.11,<3.12"
        manylinux = "2_35"
        overrides = ["override.txt"]
        dependencies = [
          "pkg-a @ git+https://example.com/a.git@0123456789abcdef0123456789abcdef01234567",
          "pkg-b @ git+https://example.com/b.git@v1.2.3 ; sys_platform == 'linux'",
          "pkg-c @ git+https://example.com/c.git@v2.0.0#egg=pkg-c@extra",
          "pkg-d @ git+https://example.com/d.git@rpent",
          "pkg-e @ git+https://example.com/e.git@rpent_v0.2",
          "pkg-f @ git+https://example.com/f.git@rpent_v0.2.1",
          "pkg-g @ git+https://example.com/g.git@rpent/robodojo",
          "pkg-h @ git+https://example.com/h.git@release/v0.4",
          "pkg-i @ git+https://example.com/i.git@release/v0.4.1",
        ]
        """,
    )
    comp = load_component(path, "robot")
    assert comp.name == "robot-alpha"
    assert comp.planner is True
    assert comp.manylinux == "2_35"
    assert len(comp.overrides) == 1
    assert len(comp.dependencies) == 9


@pytest.mark.parametrize(
    "body, match",
    [
        ('schema = 1\nname = "robot-a"\nprobe = "robots.a"', "unsupported schema"),
        (
            'schema = 2\nname = "robot-a"\nprobe = "robots.a"\nextra = 1',
            "unknown field",
        ),
        ('schema = 2\nname = "robot-wrong"\nprobe = "robots.a"', "must be 'robot-a'"),
        (
            'schema = 2\nname = "robot-a"\nprobe = "robots.a"\n'
            'dependencies = ["pkg @ git+https://example.com/p.git@main"]',
            "instead of a generic branch",
        ),
        (
            'schema = 2\nname = "robot-a"\nprobe = "robots.a"\n'
            'dependencies = ["pkg @ git+https://example.com/p.git@master"]',
            "instead of a generic branch",
        ),
        (
            'schema = 2\nname = "robot-a"\nprobe = "robots.a"\n'
            'dependencies = ["pkg @ git+https://example.com/p.git@rpent/"]',
            "instead of a generic branch",
        ),
        (
            'schema = 2\nname = "robot-a"\nprobe = "robots.a"\n'
            'dependencies = ["pkg @ git+https://example.com/p.git@rpent/foo/bar"]',
            "instead of a generic branch",
        ),
        (
            'schema = 2\nname = "robot-a"\nprobe = "robots.a"\n'
            'dependencies = ["pkg @ git+https://example.com/p.git@rpent_v0"]',
            "instead of a generic branch",
        ),
        (
            'schema = 2\nname = "robot-a"\nprobe = "robots.a"\n'
            'dependencies = ["pkg @ git+https://example.com/p.git@release/0.4"]',
            "instead of a generic branch",
        ),
        (
            'schema = 2\nname = "robot-a"\nprobe = "robots.a"\n'
            'dependencies = ["pkg @ git+https://example.com/p.git@release/main"]',
            "instead of a generic branch",
        ),
        (
            'schema = 2\nname = "robot-a"\nprobe = "robots.a"\n'
            'dependencies = ["pkg @ git+https://example.com/p.git@feature/rpent"]',
            "instead of a generic branch",
        ),
        (
            'schema = 2\nname = "robot-a"\nprobe = "robots.a"\n'
            'dependencies = ["pkg @ git+https://example.com/p.git"]',
            "has no git ref",
        ),
        (
            'schema = 2\nname = "robot-a"\nprobe = "robots.a"\n'
            'dependencies = ["pkg @ git+ssh://git@example.com/org/p.git"]',
            "has no git ref",
        ),
        (
            'schema = 2\nname = "robot-a"\nprobe = "robots.a"\noverrides = ["/abs.txt"]',
            "must be relative",
        ),
        (
            'schema = 2\nname = "robot-a"\nprobe = "robots.a"\noverrides = ["missing.txt"]',
            "not found",
        ),
        (
            'schema = 2\nname = "robot-a"\nprobe = "robots.a"\nindex-strategy = "bad"',
            "index-strategy",
        ),
    ],
)
def test_load_component_rejects_invalid_manifests(write_toml, body, match):
    path = write_toml("robots/a/component.toml", body)
    with pytest.raises(InstallConfigError, match=match):
        load_component(path, "robot")


def test_discover_validates_cross_component_references(write_toml, tmp_path):
    write_toml(
        "robots/a/component.toml",
        'schema = 2\nname = "robot-a"\nprobe = "robots.a"\nrequires = ["sim-s"]',
    )
    write_toml(
        "components/sim/s/component.toml",
        'schema = 2\nname = "sim-s"\nprobe = "components.sim.s"',
    )
    catalog = discover(tmp_path)
    assert set(catalog.components) == {"robot-a", "sim-s"}

    write_toml(
        "robots/b/component.toml",
        'schema = 2\nname = "robot-b"\nprobe = "robots.b"\nimports = ["robot-a"]',
    )
    with pytest.raises(InstallConfigError, match="neither component requires"):
        discover(tmp_path)
