# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rpent.robots.components.molmo_server import _parse_point


def test_parse_molmo2_point() -> None:
    assert _parse_point('<points coords="1 125 875"/>') == (125.0, 875.0)


def test_parse_first_of_multiple_points() -> None:
    assert _parse_point('<points coords="1 125 875; 2 500 600"/>') == (
        125.0,
        875.0,
    )


def test_reject_invalid_or_out_of_range_point() -> None:
    assert _parse_point("This isn't in the image.") is None
    assert _parse_point('<points coords="1 1001 200"/>') is None
