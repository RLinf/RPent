Robot: one seven-joint Panda arm on an Omron mobile base with a lift and a parallel-jaw gripper. move_to uses absolute grip_site xyz in world metres and fixed-axis XYZ roll, pitch and yaw in radians. Missing pose components retain their measured values. opening is zero closed to one open.

The oblique agentview and agentview_right cameras follow the chassis. Forward appears roughly upper-left in agentview and upper-right in agentview_right. Robot-left appears left in both. wrist follows the gripper. Use the picture directions to find objects and actual depth for targets.

move_base controls local planar velocity and yaw rate, then stops. move_lift changes the mounting height. Both can share an ordered batch with arm actions. The arm target in this mode uses simulator world axes. Use measured state to plan clear paths around kitchen fixtures.
