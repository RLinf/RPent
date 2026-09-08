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

"""Unit tests for :mod:`rpent.utils.serialization`."""

from __future__ import annotations

import dataclasses

import numpy as np

from rpent.utils.serialization import to_numpy_tree


class _FakeTensor:
    """Minimal duck-typed tensor-like object."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def detach(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


@dataclasses.dataclass
class _Pose:
    x: float
    y: float


def test_tensor_to_numpy() -> None:
    array = np.arange(6, dtype=np.float32)
    result = to_numpy_tree(_FakeTensor(array))
    np.testing.assert_array_equal(result, array)


def test_dataclass_to_dict() -> None:
    assert to_numpy_tree(_Pose(1.0, 2.0)) == {"x": 1.0, "y": 2.0}


def test_numpy_scalar_to_python() -> None:
    value = to_numpy_tree(np.float32(1.5))
    assert value == 1.5
    assert isinstance(value, float)


def test_nested_structure() -> None:
    value = {
        "tensor": _FakeTensor(np.array([1, 2, 3])),
        "list": [np.int64(4), (np.float32(0.5),)],
    }
    result = to_numpy_tree(value)
    np.testing.assert_array_equal(result["tensor"], np.array([1, 2, 3]))
    assert result["list"][0] == 4
    assert isinstance(result["list"][0], int)
    assert result["list"][1][0] == 0.5
    assert isinstance(result["list"][1][0], float)


def test_plain_values_pass_through() -> None:
    assert to_numpy_tree(42) == 42
    assert to_numpy_tree("hello") == "hello"
    assert to_numpy_tree(None) is None
