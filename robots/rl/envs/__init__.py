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

"""Flat simulator adapters sharing RecordedPerception."""

from __future__ import annotations


def primitive_owner(environment: str):
    """Return the primitive adapter class for ``environment``."""
    if environment == "metaworld":
        from robots.rl.envs.metaworld import MetaWorldRLPrimitives

        return MetaWorldRLPrimitives
    if environment == "libero":
        from robots.rl.envs.libero import LiberoRLPrimitives

        return LiberoRLPrimitives
    if environment == "robotwin":
        from robots.rl.envs.robotwin import RoboTwinRLPrimitives

        return RoboTwinRLPrimitives
    if environment == "robocasa":
        from robots.rl.envs.robocasa import RoboCasaRLPrimitives

        return RoboCasaRLPrimitives
    if environment == "robodojo":
        from robots.rl.envs.robodojo import RoboDojoRLPrimitives

        return RoboDojoRLPrimitives
    raise ValueError(f"Unsupported episode environment: {environment}")
