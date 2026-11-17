Robot: one Franka arm with a parallel-jaw gripper. The world frame is fixed at the mechanical arm's mounting base, not the simulator origin. +x is forward, +y is the robot's left and +z is up. The initial coordinates metadata gives the actual table height above the base in metres.

All reported robot poses, world_xyz, top_z_m, size_xyz_m and axis_yaw use this frame. z_range_m filters height in this same frame. The TCP is the measured midpoint of the two finger-pad centres; the backend compensates its offset from the native grip_site. Do not add a wrist or fingertip offset yourself.

orientation_rpy=[roll,pitch,yaw] is the absolute TCP orientation in radians, R=Rz(yaw)@Ry(pitch)@Rx(roll). The TCP axes follow the grip_site: +z is the approach axis and the jaws close along x. [pi,0,yaw] points the approach axis down. Motion has no collision avoidance. Omitted move target parts keep their measured values at the start of that action.

opening is the measured normalized jaw gap, 0 closed and 1 fully open. A blocked closing jaw is not proof of a stable grasp. agentview is the fixed camera; wrist moves with the gripper. Trust the actual displayed camera images rather than assuming a pixel direction from another camera or pose.
