# Copyright 2026 The RPent Authors.
# Licensed under the Apache License, Version 2.0.
"""Read the X5 USD's repeated link names and parent-applied collision API."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robots.rl.coordinates import Coordinates, mounting_calibration
from robots.robodojo.coordinates import _support_surface, _vertices_in_link


def test_tableless_scene_reports_the_measured_floor_height(monkeypatch):
    floor = SimpleNamespace(prim_path="/World/envs/env_0/Ground")
    prim = object()
    bounds = SimpleNamespace(
        GetMin=lambda: [-1.0, -1.0, 0.12],
        GetMax=lambda: [1.0, 1.0, 0.12],
    )

    def world_bound(surface):
        assert surface is prim
        return SimpleNamespace(ComputeAlignedRange=lambda: bounds)

    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(
            Usd=SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: None)),
            UsdGeom=SimpleNamespace(
                BBoxCache=lambda *_: SimpleNamespace(ComputeWorldBound=world_bound),
                Tokens=SimpleNamespace(default_="default"),
            ),
        ),
    )
    stage = SimpleNamespace(GetPrimAtPath={floor.prim_path: prim}.get)
    scene = SimpleNamespace(_tables=[None], _grounds=[[floor]])
    kind, point = _support_surface(stage, scene)
    calibration = mounting_calibration(
        [0, 0.3, 0.4], [0, -0.3, 0.4], point, support_surface=kind
    )
    description = Coordinates.from_control(
        {"coordinate_calibration": calibration}
    ).description()
    assert description["support_surface"] == "floor"
    assert description["floor_height_above_base_m"] == pytest.approx(-0.28)
    assert description["table_height_above_base_m"] is None


def test_collision_xform_mesh_is_measured_in_rigid_link_frame(monkeypatch):
    rigid_api, mesh_type = object(), object()

    class CollisionAPI:
        def __init__(self, prim):
            self.prim = prim

        def GetCollisionEnabledAttr(self):
            return SimpleNamespace(Get=lambda: self.prim.enabled)

    class Prim:
        def __init__(self, path, *, apis=(), points=None, children=(), enabled=True):
            self.path, self.apis, self.points = path, apis, points
            self.children, self.enabled = children, enabled
            self.transform = np.eye(4)

        def GetName(self):
            return self.path.rsplit("/", 1)[-1]

        def GetPath(self):
            return self.path

        def HasAPI(self, api):
            return api in self.apis

        def IsA(self, kind):
            return kind is mesh_type and self.points is not None

    def prim_range(prim, *args):
        yield prim
        for child in prim.children:
            yield from prim_range(child)

    mesh = Prim("/robot/link7/collisions/link7/mesh", points=[[0, 0, 0], [1, 2, 3]])
    collider = Prim(
        "/robot/link7/collisions/link7", apis=(CollisionAPI,), children=[mesh]
    )
    visual = Prim("/robot/link7/visuals/link7", points=[[999, 999, 999]])
    link = Prim("/robot/link7", apis=(rigid_api,), children=[visual, collider])
    link.transform[:3, 3] = [5, 0, 0]
    mesh.transform[:3, :3] *= 0.01
    mesh.transform[:3, 3] = [5.1, 0.02, 0.03]
    root = Prim("/robot", children=[link])
    monkeypatch.setitem(
        sys.modules,
        "pxr",
        SimpleNamespace(
            Usd=SimpleNamespace(
                PrimRange=prim_range, TraverseInstanceProxies=lambda: None
            ),
            UsdPhysics=SimpleNamespace(
                RigidBodyAPI=rigid_api, CollisionAPI=CollisionAPI
            ),
            UsdGeom=SimpleNamespace(
                Mesh=mesh_type,
                XformCache=lambda: SimpleNamespace(
                    GetLocalToWorldTransform=lambda p: p.transform.T
                ),
            ),
        ),
    )
    # USD's Mesh is both a schema type and a wrapper around a mesh prim.

    class Mesh:
        def __init__(self, prim):
            self.prim = prim

        def GetPointsAttr(self):
            return SimpleNamespace(Get=lambda: self.prim.points)

    mesh_type = Mesh
    sys.modules["pxr"].UsdGeom.Mesh = Mesh
    np.testing.assert_allclose(
        _vertices_in_link(None, root, "link7"),
        [[0.1, 0.02, 0.03], [0.11, 0.04, 0.06]],
        atol=1e-12,
    )
