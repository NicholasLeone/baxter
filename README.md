# Baxter Haptic Pen Device Manipulation Code

By using a haptic pen device, users could deviate from an arm’s preset nominal trajectory using a haptics device. The haptics device’s feedback force would be used to encourage and discourage specific movement of the arm’s end effector. The arm’s end effector would have preset trajectory and the user can push or pull along the direction of the preset trajectory to control the speed at which the end effector follows the trajectory.

The code for the project is incomplete. It has not yet been interfaced to work with 3DSystems' Touch Haptic Device (https://www.3dsystems.com/haptics-devices/touch).
The code currently allows the user to control the Baxter's end effector by assigning velocities to the end effector's x, y, z, raw, pitch, and yaw. Manipulability measures have been implemented in the code to prevent joint space singularities when moving the arm. 

joint_position_keyboard_2.py is the current version of the code. The code is based off of a script from Rethink Robotics used to control each individual joint of both arms using position controls.
