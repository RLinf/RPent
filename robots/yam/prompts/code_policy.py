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

"""Code-policy instructions for yam."""

from rpent.prompt import code_policy as common
from rpent.prompt.utils import PromptNode


def system_prompt() -> dict[str, PromptNode]:
    """Assemble code execution rules and yam operating knowledge."""
    return {
        **common.system_prompt(),
        "API reference": "{{code_policy_api}}",
    }
