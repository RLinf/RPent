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

"""Load and bind Markdown prompt fragments in an explicit order."""

from collections.abc import Iterable, Mapping
from pathlib import Path

from rpent.utils.templates import substitute_text

_ROOT = Path(__file__).parent
COMMON = _ROOT / "common"
MODES = _ROOT / "modes"


def compose(
    paths: Iterable[Path], variables: Mapping[str, object] | None = None
) -> str:
    """Bind each UTF-8 template once and join its Markdown with blank lines.

    Args:
        paths: Module-relative resource paths, in the order the model reads them.
        variables: Values for all placeholders in the selected templates.

    Returns:
        Final prompt text. Inserted values are never interpreted as templates.

    Raises:
        KeyError: A template variable is missing; includes its file and name.
    """
    parts = []
    for path in paths:
        template = path.read_text(encoding="utf-8")
        try:
            part = substitute_text(template, variables or {}, strict=True).strip()
        except KeyError as exc:
            raise KeyError(f"{path}: missing prompt variable {exc.args[0]!r}") from exc
        if part:
            parts.append(part)
    return "\n\n".join(parts) + "\n"
