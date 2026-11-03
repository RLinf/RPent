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

"""Camera poses on R2 from the robot's own URDF and its measured joints."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

#: Every R2 command and observation is expressed in this frame.
BASE_FRAME = "base_link"


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray


def _origin(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if element is None:
        return transform
    xyz = [float(v) for v in element.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in element.get("rpy", "0 0 0").split()]
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


class UrdfCameraChains:
    """Forward kinematics from ``base_link`` to named URDF frames.

    The site URDF carries the robot's camera calibration, so a camera pose is
    the product of its fixed mounts and the measured waist, neck, and arm
    joints along the chain.
    """

    def __init__(self, urdf_path: str | Path, frames: Mapping[str, str]) -> None:
        root = ET.parse(Path(urdf_path).expanduser()).getroot()
        by_child = {
            joint.find("child").get("link"): joint for joint in root.iter("joint")
        }
        self._chains: dict[str, tuple[_Joint, ...]] = {}
        for camera, frame in frames.items():
            chain: list[_Joint] = []
            link = frame
            while link != BASE_FRAME:
                joint = by_child.get(link)
                if joint is None:
                    raise ValueError(
                        f"URDF frame {frame!r} for camera {camera!r} does not "
                        f"descend from {BASE_FRAME!r}; stopped at {link!r}."
                    )
                if joint.get("type") not in ("fixed", "revolute", "continuous"):
                    raise ValueError(
                        f"URDF joint {joint.get('name')!r} on the {camera!r} camera "
                        f"chain is {joint.get('type')!r}; expected fixed or revolute."
                    )
                axis = joint.find("axis")
                chain.append(
                    _Joint(
                        name=joint.get("name"),
                        kind=joint.get("type"),
                        origin=_origin(joint.find("origin")),
                        axis=np.array(
                            [float(v) for v in axis.get("xyz").split()]
                            if axis is not None
                            else [1.0, 0.0, 0.0]
                        ),
                    )
                )
                link = joint.find("parent").get("link")
            self._chains[camera] = tuple(reversed(chain))

    @property
    def joint_names(self) -> set[str]:
        """Return every moving joint a camera pose depends on."""
        return {
            joint.name
            for chain in self._chains.values()
            for joint in chain
            if joint.kind != "fixed"
        }

    def pose(self, camera: str, joints: Mapping[str, float]) -> np.ndarray:
        """Return the 4x4 ``base_link`` pose of ``camera``'s frame.

        ``joints`` must name every moving joint on the camera's chain.
        """
        transform = np.eye(4)
        for joint in self._chains[camera]:
            transform = transform @ joint.origin
            if joint.kind != "fixed":
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(
                    joint.axis * float(joints[joint.name])
                ).as_matrix()
                transform = transform @ motion
        return transform
