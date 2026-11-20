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

"""Installation fingerprints are written atomically and hashed canonically."""

import json

from rpent.install.fingerprint import (
    FINGERPRINT_NAME,
    fingerprint_path,
    fingerprint_sha256,
    freeze_path,
    freeze_sha256,
    read_fingerprint,
    write_fingerprint,
    write_freeze,
)


def test_fingerprint_round_trip_is_sorted_and_atomic(tmp_path):
    venv = tmp_path / ".venv"
    venv.mkdir()
    fingerprint = {
        "targets": ["robot-alpha"],
        "components": {"robot-alpha": {"manifest_sha256": "digest"}},
    }

    path = write_fingerprint(venv, fingerprint)

    assert path == fingerprint_path(venv)
    assert path.name == FINGERPRINT_NAME
    assert json.loads(path.read_text()) == fingerprint
    assert not (venv / f".{FINGERPRINT_NAME}.tmp").exists()
    assert json.loads(path.read_text()) == read_fingerprint(venv)


def test_missing_or_invalid_fingerprint_reads_as_none(tmp_path):
    assert read_fingerprint(tmp_path) is None
    (tmp_path / FINGERPRINT_NAME).write_text("{not json", encoding="utf-8")
    assert read_fingerprint(tmp_path) is None
    (tmp_path / FINGERPRINT_NAME).write_text("[]", encoding="utf-8")
    assert read_fingerprint(tmp_path) is None


def test_fingerprint_hash_ignores_key_order():
    assert fingerprint_sha256({"a": 1, "b": 2}) == fingerprint_sha256({"b": 2, "a": 1})
    assert fingerprint_sha256({"a": 1}) != fingerprint_sha256({"a": 2})


def test_freeze_listing_is_stored_beside_the_fingerprint(tmp_path):
    # The hash proves the package set did not change; the listing says which
    # versions it consisted of, which the hash alone cannot answer.
    path = write_freeze(tmp_path, "torch==2.7.0")

    assert path == freeze_path(tmp_path)
    assert path.read_text(encoding="utf-8") == "torch==2.7.0\n"
    assert not (tmp_path / f".{path.name}.tmp").exists()


def test_freeze_hash_ignores_line_order_and_blank_lines():
    first = "torch==2.7.0\nnumpy==2.1.0\n"
    second = "\nnumpy==2.1.0\n\ntorch==2.7.0\n\n"

    assert freeze_sha256(first) == freeze_sha256(second)
    assert freeze_sha256(first) != freeze_sha256("torch==2.7.0\n")
