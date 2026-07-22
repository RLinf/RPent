---
name: sampled-fixture-pose-relocalization
description: Moved or rotated fixtures must be re-localized from current observations before targeting their compartments, cavities, or surfaces.
metadata:
  node_type: memory
  type: feedback
---

# Fixture pose re-localization

## Rule

A fixture's translation and yaw may differ between layouts. Do not carry a
remembered midpoint or world pose into the current scene. Localize the fixture
and its requested destination from the current visual observation before acting.

## How to apply

1. Locate the fixture in the current high-resolution image and back-project
   several pixels on its visible surfaces.
2. Estimate its current orientation from visible edges, walls, dividers, and
   openings rather than assuming a remembered yaw.
3. Identify the requested compartment, cavity, or surface relative to that
   observed geometry.
4. Prefer direct localization of visible interior points. If a relative offset
   is needed, rotate it through the currently observed fixture yaw.
5. Verify that the target lies inside the visible fixture boundaries and offers
   enough clearance for the held object.

## Related memories

- [[feedback_region_ranges_table_frame]] - transform visually estimated relative geometry through the current fixture pose.
- [[feedback_read_image_before_decide]] - use the current image as the spatial reasoning input.
- [[closed_loop_cup_back_compartment_placement]] - closed-loop placement into a rear caddy cavity.
- [[upright_book_back_compartment_placement]] - geometry-aware placement into a sampled caddy compartment.
