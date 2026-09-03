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
"""LIBERO task cards: a recorded plan, replayed with live grounding.

A card records what the planner localized, what it then commanded, and the
distance between the two. That distance is task logic and survives a change of
layout; the absolute coordinate does not. Replaying a card re-reads its anchors
now and moves every waypoint with them, so the plan follows objects that moved
without a planner in the loop.
"""

from robots.libero.task_card.replay import replay_card

__all__ = ["replay_card"]
