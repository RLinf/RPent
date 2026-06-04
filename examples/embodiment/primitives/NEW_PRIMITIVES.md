# NEW_PRIMITIVES.md — consolidated into STRICT_HYBRID_GUIDE.md

This file's content was merged into
[STRICT_HYBRID_GUIDE.md](./STRICT_HYBRID_GUIDE.md) on 2026-05-19 to keep
the operating manual self-contained. See the **Extended primitives
reference** section there for the full documentation on:

- `rotate_wrist` (world-z yaw rotation, bug-fixed)
- `rotate_pitch` (world-x pitch rotation)

> **Removed (do NOT use):** `js_move_to`, `articulate_to`, `set_object_pose`,
> and `carry_object` were teleport primitives that bypassed contact physics.
> They have been **deleted from the code** and are forbidden under
> STRICT_HYBRID_GUIDE Rule 4. Close doors/drawers/knobs with a short capped
> OSC push or `pi0_doubled`; never warp objects or the arm.

The guide also includes the verified-working t9 strict recipe (8 commands)
and a generic cavity / shelf placement template.
