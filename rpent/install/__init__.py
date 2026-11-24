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

"""Component manifests, resolution, and the ``rpent-install`` installer.

A component is the only install unit: one name, one manifest, one import probe,
and the packages only that component needs. An environment contains the
components named on the command line plus the closure of their ``requires``.
Installed environments carry ``rpent-install.json`` with the target components,
their manifest hashes, and the package freeze hash used by ``rpent-install
doctor`` and run artifacts.
"""

from rpent.install.discovery import Catalog, discover, discover_components
from rpent.install.errors import InstallConfigError, InstallError
from rpent.install.fingerprint import (
    FINGERPRINT_NAME,
    fingerprint_path,
    read_fingerprint,
    write_fingerprint,
)
from rpent.install.manifest import Component
from rpent.install.require import MissingComponentError, require
from rpent.install.resolve import Resolution, resolve

__all__ = [
    "FINGERPRINT_NAME",
    "Catalog",
    "Component",
    "InstallConfigError",
    "InstallError",
    "MissingComponentError",
    "Resolution",
    "discover",
    "discover_components",
    "fingerprint_path",
    "read_fingerprint",
    "require",
    "resolve",
    "write_fingerprint",
]
