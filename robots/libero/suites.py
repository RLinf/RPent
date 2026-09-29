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

"""Model-independent LIBERO task names and simulator variant routing."""

import os

LIBERO_SUITE_NAMES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
    "libero_object_task",
    "libero_object_swap",
    "libero_object_lan",
    "libero_object_object",
    "libero_goal_task",
    "libero_goal_swap",
    "libero_goal_lan",
    "libero_goal_object",
    "libero_spatial_task",
    "libero_spatial_swap",
    "libero_spatial_lan",
    "libero_spatial_object",
    "libero_10_task",
    "libero_10_swap",
    "libero_10_lan",
    "libero_10_object",
)


def suite_variant(suite: str) -> str:
    """Select standard base suites or Pro perturbations from the shared catalog."""
    if suite not in LIBERO_SUITE_NAMES:
        raise ValueError(f"Unknown LIBERO suite: {suite!r}")
    return "standard" if suite.count("_") == 1 else "pro"


def resolve_libero_type(suite: str, requested: str | None) -> str:
    """Honor explicit configuration, then infer the environment from the suite."""
    inferred = suite_variant(suite)
    variant = requested or os.environ.get("LIBERO_TYPE") or inferred
    if variant not in ("standard", "pro", "plus"):
        raise ValueError(f"Unknown LIBERO variant: {variant!r}")
    if inferred == "pro" and variant != "pro":
        raise ValueError(f"Suite {suite} requires --libero-type pro")
    return variant
