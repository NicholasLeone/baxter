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
from omni_msgs.msg import OmniState, OmniFeedback



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


class BaxterHapticControl(object):
    def __init__(self, limb, hover_distance = 0.15, verbose=True):
        self._baxter = URDF.from_parameter_server(key='robot_description') #Get Baxter URDF Model from ROS Parameter Server
        self._kdl_tree = kdl_tree_from_urdf_model(self._baxter) #Get kdl tree Baxter URDF Model
        self._limb_name = limb # string
        self._hover_distance = hover_distance # in meters
        self._verbose = verbose # bool
        self._limb = baxter_interface.Limb(limb) #Creates a Limb Object
        self._gripper = baxter_interface.Gripper(limb) #Creates a Gripper Object from the Limb Object 
        #self._kin = baxter_kinematics(limb) 
        #self._traj = Trajectory(self._limb_name)
        self._base_link = self._baxter.get_root() #Gets root link of the Baxter URDF model
        self._tip_link = limb + '_gripper' #Gets the gripper link of ther respective limb
        self._arm_chain = self._kdl_tree.getChain(self._base_link,
                                                  self._tip_link) #Creates chain from root link and gripper link
        self._jac_kdl = PyKDL.ChainJntToJacSolver(self._arm_chain) #Creates a jacobian solver for the Baxter from the chain
        self._joint_names = self._limb.joint_names() #List of joint names of the respective limb
        self._trajectory = list() #List of Nominal Trajectory Waypoints    
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        self._iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        rospy.wait_for_service(ns, 5.0)
        # verify robot is enabled
        print("Getting robot state... ")
        self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
        self._init_state = self._rs.state().enabled
        print("Enabling robot... ")
        self._rs.enable()
        

    def ik_request(self, pose): #IKSolver
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
    
    
    def joints_to_kdl(self, type, values=None): #
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
    
    
    
    def _velocity(self, velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z):
        start_time = time.time() #Starts Timer
        jacobian = PyKDL.Jacobian(len(self._limb.joint_names())) #Creates a Jacobian Object
        self._jac_kdl.JntToJac(self.joints_to_kdl('positions',self._limb.joint_angles()), jacobian) #Solves the Jacobian using the current joint angles as the input
        jacobian_matrix = self.kdl_to_mat(jacobian) #Converts the Jacobian from a kdl to matrix format
        jacobian_tranpose_matrix = jacobian_matrix.T #Jacobian Transpose
        pseudo_jacobian_inverse_matrix = np.linalg.pinv(jacobian_matrix) #Pseudoinverse Jacobian
        j_j_tranpose = np.dot(jacobian_matrix, jacobian_tranpose_matrix) #Dot product of Jacobian and Jacobian Transpose
        j_j_tranpose = np.squeeze(np.asarray(j_j_tranpose)) #Removes an extra pair of square brackets
        manipulability_index = np.linalg.det(j_j_tranpose) #Determinant of the dot product of the Jacobian and Jacobian Transpose
        manipulability_index = np.sqrt(manipulability_index) #Yoshikawa Manipulability Index (YMI)
        joint_names = self._limb.joint_names() 
        t_joint_angle_value = {} #Temporary Joint Angle Values 
        t_joint_angles = self._limb.joint_angles() #Limb's joint angles
        null_joint_velocities = []

        for i in range(len(self._limb.joint_names())): #Iterates through each joint (in this case, seven joints)
            joint_name = joint_names[i] #Name of current joint
            t_joint_angle_value[joint_name] = self._limb.joint_angle(joint_name) + 0.001 #Gradient Descent of the current joint in the null space
            t_joint_angles[joint_name] = t_joint_angle_value[joint_name] #Replaces the current joint angle with the temporary joint angle   
            t_jacobian = PyKDL.Jacobian(len(self._limb.joint_names())) 
            self._jac_kdl.JntToJac(self.joints_to_kdl('positions', t_joint_angles), t_jacobian) 
            t_jacobian_matrix = self.kdl_to_mat(t_jacobian)
            t_jacobian_tranpose_matrix = t_jacobian_matrix.T
            t_pseudo_jacobian_inverse_matrix = np.linalg.pinv(t_jacobian_matrix)
            t_j_j_tranpose = np.dot(t_jacobian_matrix, t_jacobian_tranpose_matrix)
            t_j_j_tranpose = np.squeeze(np.asarray(t_j_j_tranpose)) 
            t_manipulability_index = np.linalg.det(t_j_j_tranpose)
            t_manipulability_index = np.sqrt(t_manipulability_index) #YMI in the null-space
            null_joint_velocities.append(manipulability_index - t_manipulability_index) #Subtracts the two YMIs to get joint angles in the null-space
        
        null_joint_velocities = np.asarray(null_joint_velocities) 
        print(null_joint_velocities)
        velocity = np.array([velocity_x, velocity_y, velocity_z, angular_x, angular_y, angular_z])
        joint_velocities = np.dot(pseudo_jacobian_inverse_matrix, velocity)
        joint_velocities = np.squeeze(np.asarray(joint_velocities))
        adjusted_joint_velocities = np.subtract(joint_velocities, null_joint_velocities) #Corrective measure to prevent joint space singularities       
        end_time = time.time()
        elapsed_time = end_time - start_time
        adjusted_joint_positions = adjusted_joint_velocities*elapsed_time #Integrate joint velocities to get joint positions 
        position_command = {}
        joint_names = self._limb.joint_names()
        
        for i in range(len(adjusted_joint_positions)): #Append each joint position to a joint in the limb.
            joint_name = joint_names[i]
            velocity_command[joint_name] = float(adjusted_joint_positions[i]) + self._limb.joint_angle(joint_name) 
            
        self._limb.set_joint_positions(position_command) #Sets joint positions to all seven joints.

    def _set_point(self, pos_x, pos_y, pos_z, rot_matrix): #Sets the end effector at a point in the task space. 
        self._limb.set_joint_position_speed(0.5) #Controls the speed at which the joints move.
        pose = Pose() 
        r = R.from_dcm(rot_matrix) 
        q = r.as_quat() #Converts rotation matrix into quaternions
        pose.position.x = pos_x
        pose.position.y = pos_y
        pose.position.z = pos_z 
        pose.orientation.x = q[0]
        pose.orientation.y = q[1] 
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]        
        joint_angles = self.ik_request(pose) #Convert pose object into joint angles
        self._limb.set_joint_positions(joint_angles)
    
    def _create_planned_trajectory(self): #End effector moves towards the positive x direction.
        starting_pose = Pose()
        starting_pose = Pose(
            position=Point(x=0.5098537001460513, y=0.1832584024743113, z=0.1832584024743113),
            orientation=Quaternion(x=0.14076339982823685, y=0.9896428314724866, z=0.011603321768361536, w=0.0256533488623898))
        self._trajectory.append(starting_pose) #Appends the starting pose in the trajectory list
        for i in range(359): #Loops to add all waypoints and ending pose.
            previous_pose = self._trajectory[i]
            pose = Pose()
            pose.position.x = previous_pose.position.x + 0.001 #Add small increment in the x position.
            pose.position.y = previous_pose.position.y
            pose.position.z = previous_pose.position.z
            pose.orientation.x = previous_pose.orientation.x
            pose.orientation.y = previous_pose.orientation.y
            pose.orientation.z = previous_pose.orientation.z
            pose.orientation.w = previous_pose.orientation.w
            self._trajectory.append(pose) #Appends waypoint in trajectory list.
        return self._trajectory #Returns trajectory list
    
    def _create_trajectory(self,start_pose,end_pose, time): #Creates a nominal trajectory. Currently untested.
        traj_dist = np.subtract(end_pose, start_pose) #Distance between starting and ending poses. 
        traj_delta = traj_dist/(time*100) #Divides distance by time (100 waypoints per second)
        self._trajectory[0] = start_pose #Appends start pose to trajectory list.
        for i in range(time*100-1): #Creates all waypoints and endpose 
            next_waypoint = self._trajectory[i] + traj_delta #Adds small increment to previous waypoint
            self._trajectory[i+1].append(next_waypoint) #Appends waypoint to trajectory list.
        return self._trajectory
    
    def _force(self, feedback_force, input_force): #Calculates 
        force_res = np.subtract(input_force,feedback_force)
        
        
            


    def _pose_to_array(self, pose): #Convers a pose object into an array (6 x 1 matrix)
        r = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]) #Pose's quaternions
        q = r.as_euler('xyz', degrees=True) #Converts pose's quaternions into Euler angles.
        array = np.array([pose.position.x, pose.position.y, pose.position.z, q[0], q[1], q[2]]) #x,y,z,r,p,y
        return array
    
    def _endpoint_to_array(self, pose): #Converts end effector pose into an array (6 x 1 matrix)
        r = R.from_quat([pose['orientation'].x, pose['orientation'].y, pose['orientation'].z, pose['orientation'].w]) #End effector's quaternions
        q = r.as_euler('xyz', degrees=True) #Converts pose's quaternions into Euler angles.
        array = np.array([pose['position'].x, pose['position'].y, pose['position'].z, q[0], q[1], q[2]]) #x,y,z,r,p,y
        return array
        
        
    def _get_closest_waypoint(self, prev_waypoint_array, current_waypoint_array, next_waypoint_array, end_effector_point_array, count): #Find closest waypoint to end effector
        prev_dist_array = np.subtract(end_effector_point_array, prev_waypoint_array) #Distance between previous waypoint and end effector
        current_dist_array = np.subtract(end_effector_point_array, current_waypoint_array) #Distance between current waypoint and end effector
        next_dist_array = np.subtract(end_effector_point_array, next_waypoint_array) #Distance between next waypoint and end effector
        #Get magnitudes of all three distances
        prev_dist = np.linalg.norm(prev_dist_array) 
        current_dist = np.linalg.norm(current_dist_array)
        next_dist = np.linalg.norm(next_dist_array)
        if prev_dist < current_dist and prev_dist < next_dist: #If the previous waypoint is the closest to the end effector
            count = count - 1
        if next_dist < current_dist and next_dist < prev_dist: #If the next waypoint is the closest to the end effector
            count = count + 1
        else: #If the current waypoint is the closest to the end effector
            count = count
        
    def _forces(self, waypoint_array, end_effector_point_array): #Calculates the forces felt by user as end effector deviates from the nominal trajectory.
        #Creates six ROS topics for the six forces.
        force_x = rospy.Publisher('/F_x', Float64, queue_size=1) 
        force_y = rospy.Publisher('/F_y', Float64, queue_size=1)
        force_z = rospy.Publisher('/F_z', Float64, queue_size=1)
        force_roll = rospy.Publisher('/F_roll', Float64, queue_size=1)
        force_pitch = rospy.Publisher('/F_pitch', Float64, queue_size=1)
        force_yaw = rospy.Publisher('/F_yaw', Float64, queue_size=1)
        #Covariance Matrix filled with random values
        cov_matrix = np.mat([[10,2,9,1,5,2], 
            [8,4,9,1,10,1],
            [10,1,4,9,4,1],
            [3,10,10,8,9,5],
            [4,8,7,7,9,5],
            [10,10,5,9,10,3]])
        delta_pos = np.subtract(end_effector_point_array, waypoint_array) #Difference between waypoint and end effector point.
        forces = np.dot(cov_matrix, delta_pos) #Multiply the difference by the covariance matrix
        forces = np.squeeze(np.asarray(forces))
        #Publishes each force to respective topic
        force_x.publish(forces[0]) 
        force_y.publish(forces[1])
        force_z.publish(forces[2])
        force_roll.publish(forces[3])
        force_pitch.publish(forces[4])
        force_yaw.publish(forces[5])
        print(forces)
        
    def _neutral_position(self): #Sets the Baxter's arm pointing downwards.
        joint_angle = [0.192483, 1.047, 0.000806369, 0.491094, -0.178079, -0.0610333, -0.0124707]
        joint_position = {}
        joint_names = self._limb.joint_names()
        print(self._base_link)
        
        for i in range(len(joint_angle)): #Append each joint angle to a joint in the limb.
            joint_name = joint_names[i]
            joint_position[joint_name] = joint_angle[i]
        
        self._limb.set_joint_positions(joint_position) 
    
    def _starting_position(self): #Sets starting position for planned nominal trajectory
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
    
    def _set_pose(self, pose): #Sets pose to end effector
        joint_angles = self.ik_request(pose) #Converts pose object into joint angles
        self._limb.set_joint_positions(joint_angles) 
    
    def _get_current_endpose(self): #Returns the end effector's current pose
        return self._limb.endpoint_pose()  
            
   
def map_keyboard():
    left = baxter_interface.Limb('left')
    right = baxter_interface.Limb('right')
    grip_left = baxter_interface.Gripper('left', CHECK_VERSION)
    grip_right = baxter_interface.Gripper('right', CHECK_VERSION)
    lj = left.joint_names()
    rj = right.joint_names()
    bhc = BaxterHapticControl('left', 0) 

    bindings = {
    #   key: (function, args, description)
        'k': (bhc._velocity, [0,0,10,0,0,0], "Velocity Controller Activated"),
        'l': (bhc._velocity, [10,0,0,0,0,0], "Velocity Controller Activated"),
        'x': (bhc._delta_x_position, [0.1], "Pos X Increasing"),
        'c': (bhc._delta_x_position, [-0.1], "Pos X Decreasing"),
        'j': (bhc._determinant, [], "Determinant"),
        'f': (bhc._set_point, [0.75, 0.15, -0.129, 
    [[-0.9987068,  -0.0499725,  0.0093573],
    [-0.0498289, 0.9986452, 0.0149960],
    [-0.0100940, 0.0145103,  -0.9998438]]], "Setting Pose"),
        'd': (bhc._starting_position, [], "Starting Pose"),
        'e': (bhc._get_current_endpose, [], "Current EE Pose"),
        'm': (bhc._create_planned_trajectory, [], "Created Trajectory"),
        'r': (bhc._neutral_position, [], "Resetting")
        
        
        
     }
    done = False
    print("Controlling joints. Press ? for help, Esc to quit.")
    rate = rospy.Rate(1000) #Controls rate of the loop below.    
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
    count = 0 #Index in trajectory list
    loop_count = 0 #Used to determine when end effector should start moving towards the nominal trajectory.
    bhc = BaxterHapticControl('left', 0) #Create BaxterHapticControl object
    trajectory_list = bhc._create_planned_trajectory() #Create the planned trajectory.
    end_point = trajectory_list[359] #The end point of the trajectory
    bhc._neutral_position() #Set the Baxter in the neutral position
    time.sleep(3.0)
    bhc._starting_position() #Set the Baxter in the start 
    time.sleep(3.0)
    bhc._set_pose(trajectory_list[0])
    count = count + 1 
    print("Set-Up Complete")  
    time.sleep(3.0)
   
    #Loops until the end effector reaches the last point in the trajectory list
    while not rospy.is_shutdown() and end_loop == False: 
        c = baxter_external_devices.getch()  #Sets the keyboard binding
        if c in ['k'] and bhc._get_current_endpose != end_point and count != 0: #Manual Overide by user. End effector moves 10 m/s towards the negative z direction. 
            print("Manual Overide")
            #Get array (6x1 matrix) of the end effector's point and the previous, current, and next waypoint.
            prev_trajectory_array = bhc._pose_to_array(trajectory_list[count-1]) 
            current_trajectory_array = bhc._pose_to_array(trajectory_list[count])
            next_trajectory_array = bhc._pose_to_array(trajectory_list[count+1])
            end_effector_point_array = bhc._endpoint_to_array(bhc._get_current_endpose())
            #Find closest waypoint to end effector.
            bhc._get_closest_waypoint(prev_trajectory_array, current_trajectory_array, next_trajectory_array, end_effector_point_array, count)
            #Plot forces and torques produced by deviation from nominal trajectory.
            bhc._forces(bhc._pose_to_array(trajectory_list[count]), end_effector_point_array)            
            bhc._velocity(0,0,-10,0,0,0) 
            manual_overide = True  
            loop_count = 0
        if not c and manual_overide == True and loop_count >= 200:
            while bhc._endpoint_to_array(bhc._get_current_endpose())[2] < bhc._pose_to_array(trajectory_list[count])[2]:
                prev_trajectory_array = bhc._pose_to_array(trajectory_list[count-1])
                current_trajectory_array = bhc._pose_to_array(trajectory_list[count])
                next_trajectory_array = bhc._pose_to_array(trajectory_list[count+1])
                end_effector_point_array = bhc._endpoint_to_array(bhc._get_current_endpose())
                bhc._get_closest_waypoint(prev_trajectory_array, current_trajectory_array, next_trajectory_array, end_effector_point_array, count) 
                bhc._forces(bhc._pose_to_array(trajectory_list[count]), end_effector_point_array) 
                bhc._velocity(0,0,10,0,0,0) 
                print("Returning to Nominal Trajectory")
            manual_overide = False   
            loop_count = 0
        if not c and count !=359 and manual_overide == False: #End effector follows nominal trajectory. No user input.
            bhc._set_pose(trajectory_list[count]) #End effector sets position at the waypoint
            end_effector_point_array = bhc._endpoint_to_array(bhc._get_current_endpose()) 
            bhc._forces(bhc._pose_to_array(trajectory_list[count-1]), bhc._pose_to_array(trajectory_list[count]))            
            count = count + 1
            print(count)
            loop_count = 0        
        
        if c in ['\x1b', '\x03']: #Ctrl+C or Esc to stop the end effector following the nominal trajectory.
            end_loop = True           

        else:
            print("error")
            loop_count = loop_count + 1
            print(loop_count)
            end_effector_point_array = bhc._endpoint_to_array(bhc._get_current_endpose())
            bhc._forces(bhc._pose_to_array(trajectory_list[count-1]), bhc._pose_to_array(trajectory_list[count]))




    
    

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
