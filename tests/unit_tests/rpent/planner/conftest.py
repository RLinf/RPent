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

import pytest

from ._native_helpers import FakeToolkit


@pytest.fixture
def make_toolkit(tmp_path):
    instances = []

    def make(*args, **kwargs):
        toolkit = FakeToolkit(tmp_path / str(len(instances)), *args, **kwargs)
        instances.append(toolkit)
        return toolkit

    yield make
    for toolkit in instances:
        if toolkit._scheduler._state != "closed":
            toolkit.close()
