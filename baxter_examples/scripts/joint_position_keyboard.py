#!/usr/bin/env python

# Copyright (c) 2013-2015, Rethink Robotics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the Rethink Robotics nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Baxter RSDK Joint Position Example: keyboard
"""
import sys
sys.path.append('/home/nicholas/baxter_ws/devel/lib/python2.7/dist-packages')
import argparse

import rospy
import time

import baxter_interface
import baxter_external_devices
import baxter_pykdl

import argparse
import struct
import sys
from copy import copy

import rospy
import rospkg
import numpy as np
import tf2_ros
import scipy.spatial
from scipy.spatial.transform import Rotation as R
import geometry_msgs.msg
from geometry_msgs.msg import TwistStamped
import math


from gazebo_msgs.srv import (
    SpawnModel,
    DeleteModel,
)
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import (
    Header,
    Empty,
)

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)
from baxter_interface import CHECK_VERSION

import baxter_interface

from baxter_pykdl import baxter_kinematics

from tf.transformations import quaternion_matrix
from tf import TransformerROS
import actionlib

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)
from scipy import linalg
from scipy.linalg import null_space
import PyKDL
from baxter_kdl.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
from nav_msgs.msg import Path
from std_msgs.msg import Float32, Float64


class PickAndPlace(object):
    def __init__(self, limb, hover_distance = 0.15, verbose=True):
        self._baxter = URDF.from_parameter_server(key='robot_description')
        self._kdl_tree = kdl_tree_from_urdf_model(self._baxter)
        self._limb_name = limb # string
        self._hover_distance = hover_distance # in meters
        self._verbose = verbose # bool
        self._limb = baxter_interface.Limb(limb)
        self._gripper = baxter_interface.Gripper(limb)
        self._kin = baxter_kinematics(limb)
        #self._traj = Trajectory(self._limb_name)
        self._base_link = self._baxter.get_root()
        self._tip_link = limb + '_gripper'
        self._arm_chain = self._kdl_tree.getChain(self._base_link,
                                                  self._tip_link)
        self._jac_kdl = PyKDL.ChainJntToJacSolver(self._arm_chain)
        self._joint_names = self._limb.joint_names()
        self._trajectory = list()
        
        
        
    
        
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        #self.set_neutral()

    def move_to_start(self, start_angles=None):
        print("Moving the {0} arm to start pose...".format(self._limb_name))
        if not start_angles:
            start_angles = dict(zip(self._joint_names, [0]*7))
        self._guarded_move_to_joint_position(start_angles)
        self.gripper_open()
        rospy.sleep(1.0)
        print("Running. Ctrl-c to quit")

    def ik_request(self, pose):
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        ikreq = SolvePositionIKRequest()
        ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
        limb_joints = {}
        try:
            resp = self._iksvc(ikreq)
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False
        # Check if result valid, and type of seed ultimately used to get solution
        # convert rospy's string representation of uint8[]'s to int's
        resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
        limb_joints = {}
        if (resp_seeds[0] != resp.RESULT_INVALID):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp_seeds[0], 'None')
            if self._verbose:
                print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}".format(
                         (seed_str)))
            #Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            if self._verbose:
                print("IK Joint Solution:\n{0}".format(limb_joints))
                print("------------------")
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            return False
        return limb_joints

    def _guarded_move_to_joint_position(self, joint_angles):
        if joint_angles:
            self._limb.move_to_joint_positions(joint_angles)
        else:
            rospy.logerr("No Joint Angles provided for move_to_joint_positions. Staying put.")

    def gripper_open(self):
        self._gripper.open()
        rospy.sleep(1.0)

    def gripper_close(self):
        self._gripper.close()
        rospy.sleep(1.0)

    def _approach(self, pose):
        approach = copy.deepcopy(pose)
        # approach with a pose the hover-distance above the requested pose
        approach.position.z = approach.position.z + self._hover_distance
        joint_angles = self.ik_request(approach)
        self._guarded_move_to_joint_position(joint_angles)

    def _retract(self, delta):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z + delta
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        joint_angles = self.ik_request(ik_pose)
        # servo up from current pose
        self._guarded_move_to_joint_position(joint_angles)
    
    def joints_to_kdl(self, type, values=None):
        kdl_array = PyKDL.JntArray(len(self._limb.joint_names()))

        if values is None:
            if type == 'positions':
                cur_type_values = self._limb.joint_angles()
            elif type == 'velocities':
                cur_type_values = self._limb.joint_velocities()
            elif type == 'torques':
                cur_type_values = self._limb.joint_efforts()
        else:
            cur_type_values = values
        
        for idx, name in enumerate(self._limb.joint_names()): #changed from joint_names()
            kdl_array[idx] = cur_type_values[name]
        if type == 'velocities':
            kdl_array = PyKDL.JntArrayVel(kdl_array)
        return kdl_array

    def kdl_to_mat(self, data):
        mat =  np.mat(np.zeros((data.rows(), data.columns())))
        for i in range(data.rows()):
            for j in range(data.columns()):
                mat[i,j] = data[i,j]
        return mat

    def _delta_x_position(self, delta):
        # retrieve current pose from endpoint
        current_pose = self._limb.endpoint_pose()
        print(current_pose)
        ik_pose = Pose()
        ik_pose.position.x = current_pose['position'].x + delta
        ik_pose.position.y = current_pose['position'].y
        ik_pose.position.z = current_pose['position'].z 
        ik_pose.orientation.x = current_pose['orientation'].x
        ik_pose.orientation.y = current_pose['orientation'].y
        ik_pose.orientation.z = current_pose['orientation'].z
        ik_pose.orientation.w = current_pose['orientation'].w
        print(ik_pose)
        joint_angles = self.ik_request(ik_pose)
        print(joint_angles)
        # servo up from current pose
        self._guarded_move_to_joint_position(joint_angles)
    
    
    def _velocity_2(self, velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z):
        start_time = time.time() 
        jacobian = PyKDL.Jacobian(len(self._limb.joint_names()))
        self._jac_kdl.JntToJac(self.joints_to_kdl('positions',self._limb.joint_angles()), jacobian)
        jacobian_matrix = self.kdl_to_mat(jacobian)
        jacobian_tranpose_matrix = jacobian_matrix.T
        pseudo_jacobian_inverse_matrix = np.linalg.pinv(jacobian_matrix) 
        j_j_tranpose = np.dot(jacobian_matrix, jacobian_tranpose_matrix)
        j_j_tranpose = np.squeeze(np.asarray(j_j_tranpose))
        manipulability_index = np.linalg.det(j_j_tranpose)
        joint_names = self._limb.joint_names()
        t_joint_angle_value = {}
        t_joint_angles = self._limb.joint_angles()
        null_joint_angles = []

        for i in range(len(self._limb.joint_names())): #Append each joint velocity to a joint in the limb.
            joint_name = joint_names[i]
            t_joint_angle_value[joint_name] = self._limb.joint_angle(joint_name) + 0.001
            t_joint_angles[joint_name] = t_joint_angle_value[joint_name]
            t_jacobian = PyKDL.Jacobian(len(self._limb.joint_names()))
            self._jac_kdl.JntToJac(self.joints_to_kdl('positions', t_joint_angles), t_jacobian)
            t_jacobian_matrix = self.kdl_to_mat(t_jacobian)
            t_jacobian_tranpose_matrix = t_jacobian_matrix.T
            t_pseudo_jacobian_inverse_matrix = np.linalg.pinv(t_jacobian_matrix)
            t_j_j_tranpose = np.dot(t_jacobian_matrix, t_jacobian_tranpose_matrix)
            t_j_j_tranpose = np.squeeze(np.asarray(t_j_j_tranpose)) 
            t_manipulability_index = np.linalg.det(t_j_j_tranpose)
            null_joint_angles.append(manipulability_index - t_manipulability_index)
        
        null_joint_angles = np.asarray(null_joint_angles)
        print(null_joint_angles)
        velocity = np.array([velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z])
        velocity = np.dot(pseudo_jacobian_inverse_matrix,velocity)
        velocity = np.squeeze(np.asarray(velocity))
        adjusted_velocity = np.subtract(velocity, null_joint_angles)
        #print(adjusted_velocity)          
        end_time = time.time()
        elapsed_time = end_time - start_time
        adjusted_positions = adjusted_velocity*elapsed_time
        velocity_command = {}
        joint_names = self._limb.joint_names()
        
        for i in range(len(adjusted_positions)): #Append each joint velocity to a joint in the limb.
            joint_name = joint_names[i]
            velocity_command[joint_name] = float(adjusted_positions[i]) + self._limb.joint_angle(joint_name) 
            
        self._limb.set_joint_positions(velocity_command)

    def _set_point(self, pos_x, pos_y, pos_z, rot_matrix):
        current_pose = self._limb.endpoint_pose()
        print(current_pose)
        self._limb.set_joint_position_speed(0.5)
        pose = Pose()
        r = R.from_dcm(rot_matrix)
        q = r.as_quat()
        #print(q)

        pose.position.x = pos_x
        pose.position.y = pos_y
        pose.position.z = pos_z 
        pose.orientation.x = q[0]
        pose.orientation.y = q[1] 
        pose.orientation.z = q[2]
        pose.orientation.w = q[3] 
        #print(pose)    
                 
        
        
        #approach = copy.deepcopy(pose)
        joint_angles = self.ik_request(pose)
        #print(joint_angles)
        self._limb.set_joint_positions(joint_angles)
        new_pose = self._limb.endpoint_pose
        print(new_pose) 
    
    def _create_trajectory(self):
        starting_pose = Pose()
        starting_pose = Pose(
            position=Point(x=0.5098537001460513, y=0.1832584024743113, z=0.1832584024743113),
            orientation=Quaternion(x=0.14076339982823685, y=0.9896428314724866, z=0.011603321768361536, w=0.0256533488623898))
        self._trajectory.append(starting_pose)
        for i in range(359):
            previous_pose = self._trajectory[i]
            ik_pose = Pose()
            ik_pose.position.x = previous_pose.position.x + 0.001
            ik_pose.position.y = previous_pose.position.y
            ik_pose.position.z = previous_pose.position.z
            ik_pose.orientation.x = previous_pose.orientation.x
            ik_pose.orientation.y = previous_pose.orientation.y
            ik_pose.orientation.z = previous_pose.orientation.z
            ik_pose.orientation.w = previous_pose.orientation.w
            self._trajectory.append(ik_pose)
        return self._trajectory

    def _pose_to_array(self, pose):
        r = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        q = r.as_euler('xyz', degrees=True)
        array = np.array([pose.position.x, pose.position.y, pose.position.z, q[0], q[1], q[2]])
        return array
    
    def _endpoint_to_array(self, pose):
        r = R.from_quat([pose['orientation'].x, pose['orientation'].y, pose['orientation'].z, pose['orientation'].w])
        q = r.as_euler('xyz', degrees=True)
        array = np.array([pose['position'].x, pose['position'].y, pose['position'].z, q[0], q[1], q[2]])
        return array
        
        
    def _check(self, prev_trajectory_pose, current_trajectory_pose, next_trajectory_pose, current_pose, count):
        prev_dist_array = np.subtract(current_pose, prev_trajectory_pose)
        current_dist_array = np.subtract(current_pose, current_trajectory_pose)
        next_dist_array = np.subtract(current_pose, next_trajectory_pose)
        prev_dist = np.linalg.norm(prev_dist_array)
        current_dist = np.linalg.norm(current_dist_array)
        next_dist = np.linalg.norm(next_dist_array)
        if prev_dist < current_dist and prev_dist < next_dist:
            count = count - 1
        if next_dist < current_dist and next_dist < prev_dist:
            count = count + 1
        else:
            count = count
        
    def _forces(self, trajectory_pose, current_pose):
        force_x = rospy.Publisher('/F_x', Float64, queue_size=1)
        force_y = rospy.Publisher('/F_y', Float64, queue_size=1)
        force_z = rospy.Publisher('/F_z', Float64, queue_size=1)
        force_roll = rospy.Publisher('/F_roll', Float64, queue_size=1)
        force_pitch = rospy.Publisher('/F_pitch', Float64, queue_size=1)
        force_yaw = rospy.Publisher('/F_yaw', Float64, queue_size=1)
        cov_matrix = np.mat([[10,2,9,1,5,2],
            [8,4,9,1,10,1],
            [10,1,4,9,4,1],
            [3,10,10,8,9,5],
            [4,8,7,7,9,5],
            [10,10,5,9,10,3]])
        delta_pos = np.subtract(current_pose, trajectory_pose)
        forces = np.dot(cov_matrix, delta_pos)
        forces = np.squeeze(np.asarray(forces))
        force_x.publish(forces[0])
        force_y.publish(forces[1])
        force_z.publish(forces[2])
        force_roll.publish(forces[3])
        force_pitch.publish(forces[4])
        force_yaw.publish(forces[5])
        print(forces)




            


            

            
            

            
            
    
            
            

        

        

        
    
    def _determinant(self):
        #print(self._kin.jacobian_pseudo_inverse())
        #print(self._kin.jacobian_transpose())
        #print(self._kin.jacobian())
        j_j_transpose = np.dot(self._kin.jacobian(), self._kin.jacobian_transpose())
        j_j_transpose = np.squeeze(np.asarray(j_j_transpose))
        y = np.square(linalg.det(j_j_transpose))
        print(y)
        

    



    def _velocity(self, velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z):
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        rate = rospy.Rate(10.0)
        baxter_vel = rospy.Publisher('/cmd_vel', geometry_msgs.msg.TwistStamped, queue_size=1)  
        #whle not rospy.is_shutdown():  
        start_time = time.time()  
        try:
            #print('Starting to lookup transform')
            trans = tfBuffer.lookup_transform('left_hand', 'base', rospy.Time(), rospy.Duration(1.0)) #transforms end effector frame to base frame
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print('Did not find transform')
            rate.sleep()
        
        #matrix = quaternion_matrix()
        r = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]) #Input quaternions of the transform
        rotation_matrix = r.as_dcm() #Get rotation matrix through quaternions of the transform.
        #print(matrix)
        #print(rotation_matrix)
        #start_time = time.time()
        #count = 0
        linear_velocity_input = np.array([velocity_x, velocity_y, velocity_z]) #Input of Linear Velocity
        angular_velocity_input = np.array([angular_x, angular_y, angular_z]) #Input of Angular Velocity     
        velocity_input = np.array([velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z])       
        linear_velocity_base = np.dot(rotation_matrix, linear_velocity_input) #Put Linear Velocity Input into Base Frame
        angular_velocity_base = np.dot(rotation_matrix, angular_velocity_input) #Put Angular Velocity Input into Base Frame
        #print(linear_velocity_base) 
        #print(angular_velocity_base)
        pose = TwistStamped() 
        pose.header.frame_id = 'left_hand'
        pose.header.stamp = rospy.Time.now()         
        velocity_base = np.array([linear_velocity_base[0], linear_velocity_base[1], linear_velocity_base[2], angular_velocity_base[0], angular_velocity_base[1], angular_velocity_base[2]]) #Append the linear and angular velocities that are in the base frame into an array
        velocity_base_matrix = np.squeeze(np.asarray(velocity_base))
        pose.twist.linear.x = velocity_base_matrix[0]*10
        pose.twist.linear.y = velocity_base_matrix[1]*10
        pose.twist.linear.z = velocity_base_matrix[2]*10
        q_dot= np.dot(self._kin.jacobian_pseudo_inverse(), velocity_base_matrix) #Dot product of inverse jacobian and velocity base to get q-dot.
        q_dot = np.squeeze(np.asarray(q_dot))
        end_time = time.time()
        elapsed_time = end_time - start_time
        q = q_dot*elapsed_time
        print(q)
        #print(type(q_dot))
        velocity_command = {}
        joint_names = self._limb.joint_names()
        
        for i in range(len(q)): #Append each joint velocity to a joint in the limb.
            joint_name = joint_names[i]
            velocity_command[joint_name] = float(q[6-i]) + self._limb.joint_angle(joint_name) 
            #print(velocity_command[joint_name])
        #print(self._limb.endpoint_velocity())
        baxter_vel.publish(pose)
        self._limb.move_to_joint_positions(velocity_command) #Set the joint velocities to the joints. 
        rate.sleep()

    def _neutral_position(self):
        joint_angle = [0.192483, 1.047, 0.000806369, 0.491094, -0.178079, -0.0610333, -0.0124707]
        joint_position = {}
        joint_names = self._limb.joint_names()
        
        for i in range(len(joint_angle)): #Append each joint velocity to a joint in the limb.
            joint_name = joint_names[i]
            joint_position[joint_name] = joint_angle[i]
        
        self._limb.set_joint_positions(joint_position)
    
    def _current_pose(self):
        return self._limb.endpoint_pose()
    
    def _starting_position(self):
        current_pose = self._limb.endpoint_pose()
        print(current_pose)
        starting_joint_angles = {'left_w0': 0.6699952259595108,
                             'left_w1': 1.030009435085784,
                             'left_w2': -0.4999997247485215,
                             'left_e0': -1.189968899785275,
                             'left_e1': 1.9400238130755056,
                             'left_s0': -0.08000397926829805,
                             'left_s1': -0.9999781166910306}
        self._limb.set_joint_positions(starting_joint_angles)
    
    def _set_pose(self, pose):
        joint_angles = self.ik_request(pose)
        self._limb.set_joint_positions(joint_angles)
    
    def _get_current_endpose(self):
        return self._limb.endpoint_pose()


            
            
    def _servo_to_pose(self, pose):
        # servo down to release
        joint_angles = self.ik_request(pose)
        self._guarded_move_to_joint_position(joint_angles)

    def pick(self, pose):
        # open the gripper
        self.gripper_open()
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        # close gripper
        self.gripper_close()
        # retract to clear object
        self._retract()

    def place(self, pose):
        # servo above pose
        self._approach(pose)
        # servo to pose
        self._servo_to_pose(pose)
        # open the gripper
        self.gripper_open()
        # retract to clear object
        self._retract()


def map_keyboard():
    left = baxter_interface.Limb('left')
    right = baxter_interface.Limb('right')
    grip_left = baxter_interface.Gripper('left', CHECK_VERSION)
    grip_right = baxter_interface.Gripper('right', CHECK_VERSION)
    lj = left.joint_names()
    rj = right.joint_names()
    pnp = PickAndPlace('left', 0)
    #traj = Trajectory(pnp._limb_name)


    def set_j(limb, joint_name, delta):
        current_position = limb.joint_angle(joint_name)
        joint_command = {joint_name: current_position + delta}
        limb.set_joint_positions(joint_command)
    

    bindings = {
    #   key: (function, args, description)
        'k': (pnp._velocity_2, [0,0,1,0,0,0], "Velocity Controller Activated"),
        'l': (pnp._velocity_2, [1,0,0,0,0,0], "Velocity Controller Activated"),
        'x': (pnp._delta_x_position, [0.1], "Pos X Increasing"),
        'c': (pnp._delta_x_position, [-0.1], "Pos X Decreasing"),
        'j': (pnp._determinant, [], "Determinant"),
        'f': (pnp._set_point, [0.75, 0.15, -0.129, 
    [[-0.9987068,  -0.0499725,  0.0093573],
    [-0.0498289, 0.9986452, 0.0149960],
    [-0.0100940, 0.0145103,  -0.9998438]]], "Setting Pose"),
        'd': (pnp._starting_position, [], "Starting Pose"),
        'e': (pnp._current_pose, [], "Current EE Pose"),
        'm': (pnp._create_trajectory, [], "Created Trajectory"),
        'r': (pnp._neutral_position, [], "Resetting")
        
        
        
     }
    done = False
    print("Controlling joints. Press ? for help, Esc to quit.")
    rate = rospy.Rate(1000)    
    while not done and not rospy.is_shutdown():
        c = baxter_external_devices.getch()
        if c:
            #catch Esc or ctrl-c
            if c in ['\x1b', '\x03']:
                done = True
                rospy.signal_shutdown("Example finished.")
            elif c in bindings:
                cmd = bindings[c]
                #expand binding to something like "set_j(right, 's0', 0.1)"
                cmd[0](*cmd[1])
                print("command: %s" % (cmd[2],))
            else:
                print("key bindings: ")
                print("  Esc: Quit")
                print("  ?: Help")
                for key, val in sorted(bindings.items(),
                                       key=lambda x: x[1][2]):
                    print("  %s: %s" % (key, val[2]))


def main():
    """RSDK Joint Position Example: Keyboard Control

    Use your dev machine's keyboard to control joint positions.

    Each key corresponds to increasing or decreasing the angle
    of a joint on one of Baxter's arms. Each arm is represented
    by one side of the keyboard and inner/outer key pairings
    on each row for each joint.
    """
    epilog = """
See help inside the example with the '?' key for key bindings.
    """
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__,
                                     epilog=epilog)
    parser.parse_args(rospy.myargv()[1:])

    print("Initializing node... ")
    rospy.init_node("rsdk_joint_position_keyboard")
    print("Getting robot state... ")
    rs = baxter_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled
    end_loop = False
    manual_overide = False
    count = 0
    loop_count = 0
    pnp = PickAndPlace('left', 0)
    trajectory_list = pnp._create_trajectory()
    goal_pose = trajectory_list[359]
    pnp._neutral_position()
    time.sleep(3.0)
    pnp._starting_position()
    time.sleep(3.0)
    pnp._set_pose(trajectory_list[0])
    count = count + 1 
    print("Set-Up Complete")  
    time.sleep(3.0)

    while not rospy.is_shutdown() and end_loop == False:
        c = baxter_external_devices.getch()  
        if c in ['k'] and pnp._get_current_endpose != goal_pose and count != 0:
            print("Manual Overide")
            prev_trajectory_array = pnp._pose_to_array(trajectory_list[count-1])
            current_trajectory_array = pnp._pose_to_array(trajectory_list[count])
            next_trajectory_array = pnp._pose_to_array(trajectory_list[count+1])
            current_array = pnp._endpoint_to_array(pnp._get_current_endpose())
            pnp._check(prev_trajectory_array, current_trajectory_array, next_trajectory_array, current_array, count)
            pnp._forces(pnp._pose_to_array(trajectory_list[count]), current_array)            
            pnp._velocity_2(5,0,-5,0,0,0) 
            manual_overide = True 
            #rospy.sleep(0.5) 
            loop_count = 0
        if not c and manual_overide == True and loop_count >= 200:
            while pnp._endpoint_to_array(pnp._get_current_endpose())[2] < pnp._pose_to_array(trajectory_list[count])[2]:
                """ prev_trajectory_array = pnp._pose_to_array(trajectory_list[count-1])
                current_trajectory_array = pnp._pose_to_array(trajectory_list[count])
                next_trajectory_array = pnp._pose_to_array(trajectory_list[count+1])
                current_array = pnp._endpoint_to_array(pnp._get_current_endpose())
                pnp._check(prev_trajectory_array, current_trajectory_array, next_trajectory_array, current_array, count) """
                pnp._forces(pnp._pose_to_array(trajectory_list[count]), current_array) 
                pnp._velocity_2(5,0,5,0,0,0) 
                print("Returning to Nominal Trajectory")
            manual_overide = False   
            loop_count = 0
        if not c and count !=359 and manual_overide == False:
            pnp._set_pose(trajectory_list[count])
            current_array = pnp._endpoint_to_array(pnp._get_current_endpose())
            pnp._forces(pnp._pose_to_array(trajectory_list[count-1]), pnp._pose_to_array(trajectory_list[count]))
            
            #time.sleep(1.0)
            count = count + 1
            print(count)
            loop_count = 0        
        
        if c in ['\x1b', '\x03']:
            end_loop = True      

            

        else:
            print("error")
            loop_count = loop_count + 1
            print(loop_count)
            #current_array = pnp._endpoint_to_array(pnp._get_current_endpose())
            #pnp._forces(pnp._pose_to_array(trajectory_list[count-1]), pnp._pose_to_array(trajectory_list[count]))




    
    

    def clean_shutdown():
        print("\nExiting example...")
        if not init_state:
            print("Disabling robot...")
            rs.disable()
    rospy.on_shutdown(clean_shutdown)

    print("Enabling robot... ")
    rs.enable()

    map_keyboard()
    print("Done.")

    

            
            

        



if __name__ == '__main__':
    main()
