# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Calibrate the grasp centre from the loaded X5 collision geometry and live links."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from robots.rl.geometry import (
    finger_contact_centres,
    grasp_axes,
    mounting_calibration,
    pose_matrix,
)


def _vertices_in_link(stage, root, link_name):
    """Read collision vertices relative to their rigid link, not a stale USD pose."""
    from pxr import Usd, UsdGeom, UsdPhysics

    # Imported USD assets can repeat the link name under visuals/collisions.
    # Only the rigid body carries the live pose returned by the articulation.
    links = [
        prim
        for prim in Usd.PrimRange(root)
        if prim.GetName() == link_name and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if len(links) != 1:
        raise ValueError(f"Expected exactly one loaded robot link {link_name!r}")
    link = links[0]
    cache = UsdGeom.XformCache()
    inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(link)).T)
    meshes = {}
    # The X5 asset applies CollisionAPI to a parent Xform, while its geometry
    # lives on child meshes. Other assets may apply it directly to the mesh.
    for collider in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
        if not collider.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is False:
            continue
        for prim in Usd.PrimRange(collider, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                meshes[str(prim.GetPath())] = prim
    vertices = []
    for prim in meshes.values():
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        transform = inverse @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
        vertices.append(points @ transform[:3, :3].T + transform[:3, 3])
    if not vertices:
        raise ValueError(
            f"No loaded collision mesh for {link_name!r}; cannot calibrate TCP"
        )
    result = np.concatenate(vertices)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite gripper collision geometry")
    return result


def _support_surface(stage, scene_manager) -> tuple[str, np.ndarray]:
    """Measure the loaded table or the ground in scenes without a table."""
    from pxr import Usd, UsdGeom

    table = scene_manager._tables[0]
    if table is None:
        grounds = scene_manager._grounds[0]
        if len(grounds) != 1:
            raise ValueError("Expected one loaded ground in a scene without a table")
        name, surface = "floor", grounds[0]
    else:
        name, surface = "table", table

    prim = stage.GetPrimAtPath(surface.prim_path)
    if not prim:
        raise ValueError(f"Missing loaded {name} prim: {surface.prim_path}")
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(prim)
        .ComputeAlignedRange()
    )
    low, high = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
    if (
        not np.isfinite([low, high]).all()
        or np.any(high[:2] <= low[:2])
        or high[2] < low[2]
        or (name == "table" and high[2] == low[2])
    ):
        raise ValueError(f"Missing measured {name} geometry")
    top = (low + high) / 2
    top[2] = high[2]
    return name, top


def calibrate(env):
    """Calibrate from live robot links and the loaded support surface."""
    import omni.usd

    from robots.robodojo.agent_support import _motion_robot

    stage = omni.usd.get_context().get_stage()
    manager = env.robot_manager
    mounts, tools = {}, {}
    for arm in ("left", "right"):
        robot = _motion_robot(env, arm)
        articulation = manager.robot_key[manager.robot_list.index(robot)]
        roots = [
            prim
            for prim in stage.Traverse()
            if re.fullmatch(articulation.cfg.prim_path, str(prim.GetPath()))
            and str(prim.GetPath()).startswith(manager.scene.env_prim_paths[0] + "/")
        ]
        if len(roots) != 1:
            raise ValueError(
                "Cannot identify the loaded arm articulation for calibration"
            )
        root = roots[0]
        native = pose_matrix(manager.get_real_endpose(robot, [0], is_relative=False)[0])
        mount = manager.get_link_pose(robot, robot.base_link, [0], is_relative=False)[0]
        mounts[arm] = np.asarray(mount)[:3]
        urdf = ET.parse(Path(robot.urdf_path)).getroot()
        joints = {joint.attrib["name"]: joint for joint in urdf.findall("joint")}
        finger_points = []
        for name in robot.gripper_joints_name:
            joint = joints[name]
            if joint.attrib["type"] != "prismatic":
                raise ValueError(
                    "Grasp calibration currently requires a parallel-jaw gripper"
                )
            link = joint.find("child").attrib["link"]
            pose = pose_matrix(
                manager.get_link_pose(robot, link, [0], is_relative=False)[0]
            )
            vertices = _vertices_in_link(stage, root, link)
            world = vertices @ pose[:3, :3].T + pose[:3, 3]
            finger_points.append((world - native[:3, 3]) @ native[:3, :3])
        if len(finger_points) != 2:
            raise ValueError("Grasp calibration requires two measured finger links")
        # X5 and Aloha both slide along native Y. The opposing inner faces are
        # measured from meshes; no fixed wrist-to-grasp distance is assumed.
        centres = finger_contact_centres(*finger_points)
        offset = centres.mean(axis=0)
        tools[arm] = {
            "offset_in_native_m": offset.tolist(),
            "rotation_to_native": grasp_axes(centres[1] - centres[0], offset).tolist(),
            "contact_centres_in_native_m": centres.tolist(),
            "native_link": robot.ee_link_name,
            "method": "loaded_collision_inner_faces",
        }
    support, top = _support_surface(stage, env.scene_manager)
    frame = mounting_calibration(
        mounts["left"], mounts["right"], top, support_surface=support
    )
    frame["tcp"] = "calibrated_grasp_tcp"
    return {
        "coordinate_calibration": frame,
        "native_tcp": "native_wrist_ik",
        "tools": tools,
    }
