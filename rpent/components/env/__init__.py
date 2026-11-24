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

"""Environment-side client, facade and action-spec validation.

``client.py`` and ``facade.py`` are the shared RPC ends every robot adapter
speaks through; ``action_spec.py`` validates native action layouts on both.
The normalized env contracts (``EnvMeta``, ``EnvSnapshot``, ``CameraFrame``)
have not been moved out of the slices yet.
"""
