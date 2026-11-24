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


def _print_uninstalled_help() -> int:
    """Describe the exporter when its component is not installed."""
    print(
        "usage: rpent-flywheel [-h] ...\n"
        "\n"
        "Export a recorded flight as training data. The options come from the\n"
        "`tool-flywheel` component, which this environment does not install;\n"
        "install it from an RPent checkout with\n"
        "`./rpent-install.sh install tool-flywheel`."
    )
    return 0


def main() -> int:
    """Run the flywheel exporter installed in this environment."""
    if "-h" in sys.argv or "--help" in sys.argv:
        # Help parses no arguments and needs no package the environment may be
        # missing, so the component guard must not replace the usage text.
        try:
            from components.tools.flywheel import cli
        except ModuleNotFoundError as exc:
            # A core-only wheel ships no components; the usage text still works.
            if (exc.name or "").split(".")[0] != "components":
                raise
            return _print_uninstalled_help()

        return cli.main()
    try:
        cli = require("tool-flywheel", "components.tools.flywheel.cli")
    except MissingComponentError as exc:
        print(exc, file=sys.stderr)
        return 1
    return cli.main()


if __name__ == "__main__":
    sys.exit(main())
