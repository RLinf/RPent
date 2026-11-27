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

"""Core-side contracts for the two service families a slice talks to.

``env/`` holds the environment RPC ends and the action-spec validation that
robot adapters and simulator servers share; ``models/`` holds the action-model
client, facade, backend and platform contracts that robot adapters and model
services share. Each mirrors a group of concrete services under the top-level
``components/``; the core defines what a service must expose and never imports
one statically. The backend launcher loads a component's runtime configuration
only when a selected backend needs it. Import the concrete modules rather than
re-exporting from here, so a process only loads the side it uses.
"""
