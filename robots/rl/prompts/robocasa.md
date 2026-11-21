Robot: one Panda arm on an Omron mobile base, with a parallel-jaw gripper. move_to uses absolute world xyz in metres and an optional world-frame quaternion [w,x,y,z]; omitted orientation is held. opening is 0 closed to 1 open.

Cameras: agentview and agentview_right are the native left/right overview cameras mounted with the robot; wrist moves with the gripper. Camera names and image sizes are attached to each observation. Base movement changes the camera geometry. Always use measurements from the current observation.

move_base controls local-frame planar velocity and yaw rate for a bounded duration, then stops. Rotating the base changes the relationship between its local axes and world xyz. Motion tools do not plan a collision-free route around kitchen fixtures.
