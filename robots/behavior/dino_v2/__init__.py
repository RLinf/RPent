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

"""BEHAVIOR DINOv2 encoder, RPC client, and server."""

from robots.behavior.dino_v2.client import BehaviorDinoClient
from robots.behavior.dino_v2.encoder import (
    DINOV2_DIMENSION,
    DISTANCE_METRIC,
    Dinov2DeploymentPaths,
    Dinov2Engine,
    Dinov2RevisionIdentity,
)
from robots.behavior.dino_v2.server import DinoRpc

__all__ = [
    "DINOV2_DIMENSION",
    "DISTANCE_METRIC",
    "BehaviorDinoClient",
    "DinoRpc",
    "Dinov2DeploymentPaths",
    "Dinov2Engine",
    "Dinov2RevisionIdentity",
]
