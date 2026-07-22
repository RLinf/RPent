---
name: region-ranges-table-frame
description: Targets defined relative to a visible fixture must be localized directly or transformed through the fixture pose estimated from current images.
metadata:
  node_type: memory
  type: feedback
  originSessionId: 234721d6-dc80-4ac9-806e-e06977ce7823
---

# Visual fixture-relative targeting

**Rule:** A destination described relative to a fixture, such as a rear
compartment, burner, shelf, or drawer interior, moves and rotates with that
fixture. Prefer direct visual localization of the destination. When relative
geometry is needed, transform only measurements derived from the current
observation:

```text
world_target_xy = observed_fixture_xy
                + R(observed_fixture_yaw) * visually_estimated_relative_offset_xy
```

The fixture pose, yaw, and relative offset must come from the current images,
depth projection, and visible fixture geometry. Do not reuse a world coordinate
or a layout-specific offset from a previous scene.

**How to apply:**

1. Resolve the requested fixture and destination from the task language.
2. Locate the destination surface or cavity directly in the current high-resolution
   image whenever it is visible, then back-project several interior pixels.
3. If the destination is partly occluded, estimate the fixture center and yaw
   from visible edges, walls, and surfaces, then derive a fixture-relative offset
   from that visible geometry.
4. Confirm that the resulting target lies inside the observed fixture boundaries.
5. After placement, inspect the current image and official task signal; if the
   object is displaced, re-localize instead of tuning around a stale coordinate.

Related: [[feedback_swap_perturbs_fixtures]] [[feedback_read_image_before_decide]]
