Robot: two Aloha-AgileX arms. Use the anatomical arm names left and right, and the cyan LEFT and yellow RIGHT TCP markers. The head camera gives an overview. The left_wrist and right_wrist cameras follow their named arms.

Both arms share a frame centered between their mounting bases. +X is forward, +Y is robot-left, and +Z is up. TCP is the midpoint of the finger pads. Pose xyz and perception coordinates are in metres in this frame. Use absolute orientation_rpy [roll,pitch,yaw] in radians, with R=Rz(yaw)Ry(pitch)Rx(roll). The TCP's local +X points along the fingers and the jaws close along local Y.

For a move, specify xyz, orientation_rpy, or both. Omitted components hold the measured position or orientation. Use other_xyz and other_orientation_rpy when both arms must move synchronously. Gripper opening ranges from 0 closed to 1 open. A nonzero opening can occur on an empty gripper. Verify a grasp by checking the object after a short lift.

For an unknown target, view the scene and measure its visible surface before choosing a pose. Start with a position-only move to a clear hover above the target. For a top-down approach, pitch near +1.57 points the fingers down. Change orientation while clear, then descend along Z. If a move is partial, check the measured pose and use a smaller move or more clearance before retrying.
