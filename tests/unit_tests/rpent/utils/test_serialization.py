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

"""Tests for :func:`rpent.utils.serialization.to_numpy_tree`."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from rpent.utils.serialization import to_numpy_tree


class _FakeTensor:
    """Duck-typed tensor stand-in (torch is not a unit-test dependency)."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


@dataclasses.dataclass
class _FakeObservation:
    state: np.ndarray
    reward: np.float32


def test_tensor_like_values_become_numpy_arrays() -> None:
    array = np.arange(3, dtype=np.float32)
    # CPU tensors convert zero-copy: the array shares memory with the source.
    assert to_numpy_tree(_FakeTensor(array)) is array


def test_numpy_arrays_pass_through_unchanged() -> None:
    array = np.arange(3)
    assert to_numpy_tree(array) is array


def test_numpy_scalars_pass_through_unchanged() -> None:
    # np.generic values are pickle-safe as-is and the RPC transports
    # round-trip them with dtype fidelity, so they need no unwrapping.
    for value in (np.float32(1.5), np.bool_(True), np.int64(7)):
        assert to_numpy_tree(value) is value


def test_nested_containers_recurse() -> None:
    value: Any = {
        "states": [_FakeTensor(np.zeros(2))],
        "info": (np.float32(0.25), {"done": np.bool_(False)}),
    }
    result = to_numpy_tree(value)
    assert result["states"][0].tolist() == [0.0, 0.0]
    assert result["info"][0] == np.float32(0.25)
    assert isinstance(result["info"][0], np.float32)
    assert result["info"][1] == {"done": np.bool_(False)}
    assert isinstance(result["info"][1]["done"], np.bool_)


def test_dataclasses_recurse() -> None:
    result = to_numpy_tree(_FakeObservation(np.ones(2), np.float32(3.5)))
    assert isinstance(result, dict)
    assert result["state"].tolist() == [1.0, 1.0]
    assert result["reward"] == np.float32(3.5)
    assert isinstance(result["reward"], np.float32)


def test_plain_values_pass_through() -> None:
    assert to_numpy_tree(None) is None
    assert to_numpy_tree("x") == "x"
    assert to_numpy_tree(3) == 3
