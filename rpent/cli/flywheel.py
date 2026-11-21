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

"""Entry point for the ``rpent-flywheel`` console script.

The exporter is the ``tool-flywheel`` component, which no core module needs at
import time. The import stays inside :func:`main` so a core-only wheel keeps a
working ``--help`` and reports a missing component instead of failing to load.
"""

from __future__ import annotations

import sys

from rpent.install.require import MissingComponentError, require


def main() -> int:
    """Run the flywheel exporter installed in this environment."""
    try:
        cli = require("tool-flywheel", "components.tools.flywheel.cli")
    except MissingComponentError as exc:
        print(exc, file=sys.stderr)
        return 1
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())
