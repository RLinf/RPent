Robot: two Aloha-AgileX arms with parallel-jaw grippers. left/right identify the robot's anatomical arms, not a fixed side of the image. Visible measured TCPs are marked LEFT in cyan and RIGHT in yellow; left_wrist and right_wrist belong to those named arms.

All TCP positions are absolute world xyz in metres. Orientations are unit quaternions [w,x,y,z]. Omitting an orientation preserves its measured value. opening is 0 closed to 1 open. A single move_to can move both arms synchronously by setting other_xyz; set_gripper can similarly set other_opening. The unselected arm otherwise keeps its current joint target.

Cameras: head is the overview; left_wrist and right_wrist move with their corresponding grippers. Image-left/right can differ from anatomical left/right. Use the named view and current depth measurements to localize targets. A failed plan means no motion was executed for that request.
