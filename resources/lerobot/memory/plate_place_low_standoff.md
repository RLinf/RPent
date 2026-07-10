---
hook: "For placing in the white plate at x\u22480.25,y\u22480.105, high vertical standoffs can be unreachable; move low/reachable over the plate and release around z\u2248-0.03."
env: lerobot
updated: 2026-07-03
---

In the green-cube-to-white-plate task, the plate interior back-projected near x=0.224–0.252, y=0.103–0.109, z≈-0.067. Carrying the grasped cube to a high down-oriented standoff at [0.245, 0.105, 0.080] with yaw 90 failed, settling ~45 mm short (final z≈0.037) with a reach note. A lower sequence was reachable and successful: [0.230,0.105,0.035] -> [0.240,0.106,0.015] -> [0.248,0.106,0.005], then lower to about [0.251,0.108,-0.030] and open. The cube remained in the plate after lifting the gripper. For this plate region, avoid insisting on a high vertical standoff; use reachable lower waypoints and verify with the scene/arm images.
