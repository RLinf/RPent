## Physical runtime

Treat every motion as safety-critical. The runtime or onsite deployment owns
robot and environment services. Do not start, stop, or restart robot, Ray, ROS,
camera, or VLA services. Use only the structured tools shown by the runtime.

Never use a VLA trained for another embodiment or action normalization.
If state, cameras, or motion results conflict, pause motion instead of guessing.
Follow the robot's explicit recovery and stop protocol; a documented recoverable
residual or rejected action is not by itself a terminal hardware fault.
