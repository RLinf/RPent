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

"""Model-side contracts, clients, facades, backends and platform helpers.

Import the concrete modules (``contracts``, ``client``, ``facade``,
``backends``, ``platforms``) rather than re-exporting from here, so a service
only loads what it uses. These bases are pure Python and depend on no model
service; concrete services live in ``components/models/`` and the core never
imports them.
"""
