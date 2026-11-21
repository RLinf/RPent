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

"""Run records and the errors the component tooling raises.

Every run directory carries ``run.json``: what ran, in which environment, what
it wrote, and the verdict it ended with.
"""

from rpent.install.errors import InstallConfigError, InstallError
from rpent.install.snapshot import (
    RUN_MANIFEST_NAME,
    finish_run_manifest,
    installation_snapshot,
    start_run_manifest,
)

__all__ = [
    "RUN_MANIFEST_NAME",
    "InstallConfigError",
    "InstallError",
    "finish_run_manifest",
    "installation_snapshot",
    "start_run_manifest",
]
