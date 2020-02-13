#!/usr/bin/env python
#
# Copyright 2019 DFKI GmbH.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.
import numpy as np
from transformations import quaternion_matrix, quaternion_from_matrix
from ..animation_data.quaternion_frame import convert_quaternion_frames_to_euler_frames, euler_to_quaternion, quaternion_to_euler
from ..animation_data.utils import quaternion_from_vector_to_vector
from ..animation_data.constants import LEN_QUAT, LEN_ROOT_POS


def convert_euler_to_quat(euler_frame, joints):
    quat_frame = euler_frame[:3].tolist()
    offset = 3
    step = 3
    for joint in joints:
        e = euler_frame[offset:offset+step]
        q = euler_to_quaternion(e)
        quat_frame += list(q)
        offset += step
    return np.array(quat_frame)


class SkeletonPoseModel(object):
    """
    TODO wrap parameters to allow use with constrained euler, e.g. to rotate an arm using a single parameter
    then have a skeleton model with realistic degrees of freedom and also predefined ik-chains that is initialized using a frame
    """
    def __init__(self, skeleton, use_euler=False):
        self.channels = skeleton.get_channels()
        pose_parameters = skeleton.reference_frame
        self.skeleton = skeleton
        self.use_euler = use_euler
        if self.use_euler:
            self.pose_parameters = convert_quaternion_frames_to_euler_frames([pose_parameters])[0]#change to euler
        else:
            self.pose_parameters = pose_parameters
        self.n_channels = {}
        self.channels_start = {}
        self.types = {}
        self.modelled_joints = []
        channel_idx = 0
        for joint, ch in list(self.channels.items()):
            self.channels_start[joint] = channel_idx
            self.n_channels[joint] = len(ch)
            if len(ch) == LEN_QUAT:
                self.types[joint] = "rot"
            elif len(ch) == LEN_ROOT_POS + LEN_QUAT:
                self.types[joint] = "trans"
            else:
                self.types[joint] = "rot"
            channel_idx += self.n_channels[joint]
            if self.n_channels[joint] > 0:
                self.modelled_joints.append(joint)
        self.free_joints_map = skeleton.free_joints_map
        if "head" in skeleton.skeleton_model:
            self.head_joint = skeleton.skeleton_model["joints"]["head"]
        if "neck" in skeleton.skeleton_model:
            self.neck_joint = skeleton.skeleton_model["joints"]["neck"]
        if "cos_map" in skeleton.skeleton_model:
            hand_joint = skeleton.skeleton_model["joints"]["right_wrist"]
            if hand_joint in skeleton.skeleton_model["cos_map"]:
                cos = skeleton.skeleton_model["cos_map"][hand_joint]
                self.relative_hand_dir = np.array(list(cos["y"]) + [0])
                self.relative_hand_cross = np.array(list(cos["x"]) + [0])
                up_vec = list(np.cross(cos["y"], cos["x"]))
                self.relative_hand_up = np.array(up_vec + [0])
        else:
            self.relative_hand_dir = np.array([1.0, 0.0, 0.0, 0.0])
            self.relative_hand_cross = np.array([0.0,1.0,0.0, 0.0])
            self.relative_hand_up = np.array([0.0, 0.0, 1.0, 0.0])
        if "relative_head_dir" in skeleton.skeleton_model.keys():
            vec = list(skeleton.skeleton_model["relative_head_dir"])
            self.relative_head_dir = np.array(vec+[0])
        else:
            self.relative_head_dir = np.array([0.0, 0.0, 1.0, 0.0])

        self.bounds = dict()#skeleton.bounds

    def set_pose_parameters(self, pose_parameters):
        self.pose_parameters = pose_parameters

    def set_channel_values(self, parameters, free_joints):
        p_idx = 0
        for joint_name in free_joints:
            n_channels = self.n_channels[joint_name]
            p_end = p_idx + n_channels
            f_start = self.channels_start[joint_name]
            f_end = f_start + n_channels
            #print f_start, f_end
            self.pose_parameters[f_start:f_end] = parameters[p_idx:p_end]
            p_idx += n_channels
        return self.pose_parameters

    #def set_rotation_of_joint(self, joint_name, axis, angle):
    #    n_channels = self.n_channels[joint_name]
    #    f_start = self.channels_start[joint_name]
    #    f_end = f_start + n_channels
    #    return

    def extract_parameters_indices(self, joint_name):
        f_start = self.channels_start[joint_name]
        f_end = f_start + self.n_channels[joint_name]
        return f_start, f_end

    def extract_parameters(self, joint_name):
        f_start, f_end = self.extract_parameters_indices(joint_name)
        #print joint_name, f_start, f_end, self.pose_parameters[f_start: f_end].tolist(), len(self.pose_parameters)
        return self.pose_parameters[f_start: f_end]

    def get_vector(self):
        if self.use_euler:
            return convert_euler_to_quat(self.pose_parameters, self.modelled_joints)#convert to quat
        else:
            return self.pose_parameters

    def evaluate_position_with_cache(self, target_joint, free_joints):
        for joint in free_joints:
            self.skeleton.nodes[joint].get_global_position(self.pose_parameters, use_cache=True)
        return self.skeleton.nodes[target_joint].get_global_position(self.pose_parameters, use_cache=True)#get_vector()

    def evaluate_position(self, target_joint):
        return self.skeleton.nodes[target_joint].get_global_position(self.pose_parameters)#get_vector()

    def evaluate_orientation(self, target_joint):
        return self.skeleton.nodes[target_joint].get_global_orientation_quaternion(self.pose_parameters)

    def evaluate_matrix(self, target_joint):
        return self.skeleton.nodes[target_joint].get_global_matrix(self.pose_parameters)

    def apply_bounds(self, free_joint):
        if free_joint in list(self.bounds.keys()):
            euler_angles = self.get_euler_angles(free_joint)
            for bound in self.bounds[free_joint]:
                self.apply_bound_on_joint(euler_angles, bound)

            if self.use_euler:
                self.set_channel_values(euler_angles, [free_joint])
            else:
                q = euler_to_quaternion(euler_angles)
                #print("apply bound")
                self.set_channel_values(q, [free_joint])
        return

    def apply_bound_on_joint(self, euler_angles, bound):
        #self.pose_parameters[start+bound["dim"]] =
        #print euler_angles
        if "min" in list(bound.keys()):
            euler_angles[bound["dim"]] = max(euler_angles[bound["dim"]],bound["min"])
        if "max" in list(bound.keys()):
            euler_angles[bound["dim"]] = min(euler_angles[bound["dim"]],bound["max"])
        #print "after",euler_angles

    def get_euler_angles(self, joint):
        if self.use_euler:
            euler_angles = self.extract_parameters(joint)
        else:
            q = self.extract_parameters(joint)
            euler_angles = quaternion_to_euler(q)
        return euler_angles

    def generate_constraints(self, free_joints):
        """ TODO add bounds on axis components of the quaternion according to
        Inverse Kinematics with Dual-Quaternions, Exponential-Maps, and Joint Limits by Ben Kenwright
        or try out euler based ik
        """
        cons = []
        idx = 0
        for joint_name in free_joints:
            if joint_name in list(self.bounds.keys()):
                start = idx
                for bound in self.bounds[joint_name]:
                    if "min" in list(bound.keys()):
                        cons.append(({"type": 'ineq', "fun": lambda x: x[start+bound["dim"]]-bound["min"]}))
                    if "max" in list(bound.keys()):
                        cons.append(({"type": 'ineq', "fun": lambda x: bound["max"]-x[start+bound["dim"]]}))
            idx += self.n_channels[joint_name]
        return tuple(cons)

    def clear_cache(self):
        self.skeleton.clear_cached_global_matrices()

    def lookat(self, point):
        head_position = self.evaluate_position(self.head_joint)
        target_dir = point - head_position
        target_dir /= np.linalg.norm(target_dir)
        head_direction = self.get_joint_direction(self.head_joint, self.relative_head_dir)
        delta_q = quaternion_from_vector_to_vector(head_direction, target_dir)
        delta_matrix = quaternion_matrix(delta_q)

        #delta*parent*old_local = parent*new_local
        #inv_parent*delta*parent*old_local = new_local
        parent_m = self.skeleton.nodes[self.neck_joint].get_global_matrix(self.pose_parameters, use_cache=False)
        old_local = np.dot(parent_m, self.skeleton.nodes[self.head_joint].get_local_matrix(self.pose_parameters))
        m = np.dot(delta_matrix, old_local)
        new_local = np.dot(np.linalg.inv(parent_m),m)
        new_local_q = quaternion_from_matrix(new_local)
        self.set_channel_values(new_local_q, [self.head_joint])

    def set_hand_orientation_using_direction_vectors(self, joint_name, orientation):
        m = quaternion_matrix(orientation)
        target_dir = np.dot(m, self.relative_hand_dir)
        target_dir /= np.linalg.norm(target_dir)
        cross_dir = np.dot(m, self.relative_hand_cross)
        cross_dir /= np.linalg.norm(cross_dir)
        #print "set hand orientation"
        #print target_dir, cross_dir
        self.point_in_direction(joint_name, target_dir[:3], cross_dir[:3])

    def set_hand_orientation(self, joint_name, orientation):
        #print "set hand orientation"
        m = quaternion_matrix(orientation)
        parent_joint_name = self.get_parent_joint(joint_name)
        parent_m = self.skeleton.nodes[parent_joint_name].get_global_matrix(self.pose_parameters, use_cache=False)
        local_m = np.dot(np.linalg.inv(parent_m), m)
        q = quaternion_from_matrix(local_m)
        self.set_channel_values(q, [joint_name])


    def point_in_direction(self, joint_name, target_dir, target_cross=None):
        parent_joint_name = self.get_parent_joint(joint_name)
        joint_direction = self.get_joint_direction(joint_name, self.relative_hand_dir)
        delta_q = quaternion_from_vector_to_vector(joint_direction, target_dir)
        delta_matrix = quaternion_matrix(delta_q)
        #delta*parent*old_local = parent*new_local
        #inv_parent*delta*parent*old_local = new_local
        parent_m = self.skeleton.nodes[parent_joint_name].get_global_matrix(self.pose_parameters, use_cache=False)
        old_local = np.dot(parent_m, self.skeleton.nodes[joint_name].get_local_matrix(self.pose_parameters))
        m = np.dot(delta_matrix, old_local)
        new_local = np.dot(np.linalg.inv(parent_m),m)
        new_local_q = quaternion_from_matrix(new_local)
        self.set_channel_values(new_local_q, [joint_name])

        #rotate around orientation vector
        if target_cross is not None:
            joint_cross = self.get_joint_direction(joint_name, self.relative_hand_cross)
            delta_q = quaternion_from_vector_to_vector(joint_cross, target_cross)
            delta_matrix = quaternion_matrix(delta_q)
            parent_m = self.skeleton.nodes[parent_joint_name].get_global_matrix(self.pose_parameters, use_cache=False)
            old_local = np.dot(parent_m, self.skeleton.nodes[joint_name].get_local_matrix(self.pose_parameters))
            m = np.dot(delta_matrix, old_local)
            new_local = np.dot(np.linalg.inv(parent_m),m)
            new_local_q = quaternion_from_matrix(new_local)
            self.point_hand_cross_dir_in_direction(joint_name, target_cross, parent_joint_name)

    def point_hand_cross_dir_in_direction(self,joint_name,target_cross,parent_joint_name):
        joint_cross = self.get_joint_direction(joint_name, self.relative_hand_cross)
        delta_q = quaternion_from_vector_to_vector(joint_cross, target_cross)
        delta_matrix = quaternion_matrix(delta_q)
        parent_m = self.skeleton.nodes[parent_joint_name].get_global_matrix(self.pose_parameters, use_cache=False)
        old_local = np.dot(parent_m, self.skeleton.nodes[joint_name].get_local_matrix(self.pose_parameters))
        m = np.dot(delta_matrix, old_local)
        new_local = np.dot(np.linalg.inv(parent_m), m)
        new_local_q = quaternion_from_matrix(new_local)
        self.set_channel_values(new_local_q, [joint_name])

    def point_hand_up_dir_in_direction(self, joint_name, target_up, parent_joint_name):
        joint_cross = self.get_joint_direction(joint_name, self.relative_hand_up)
        delta_q = quaternion_from_vector_to_vector(joint_cross, target_up)
        delta_matrix = quaternion_matrix(delta_q)
        parent_m = self.skeleton.nodes[parent_joint_name].get_global_matrix(self.pose_parameters, use_cache=False)
        old_local = np.dot(parent_m, self.skeleton.nodes[joint_name].get_local_matrix(self.pose_parameters))
        m = np.dot(delta_matrix, old_local)
        new_local = np.dot(np.linalg.inv(parent_m), m)
        new_local_q = quaternion_from_matrix(new_local)
        self.set_channel_values(new_local_q, [joint_name])

    def set_joint_orientation(self, joint_name, target_q):
        global_q = self.skeleton.nodes[joint_name].get_global_orientation_quaternion(self.pose_parameters, use_cache=False)
        global_m = quaternion_matrix(global_q)
        target_m = quaternion_matrix(target_q)
        delta_m = np.linalg.inv(global_m)*target_m
        local_m = self.skeleton.nodes[joint_name].get_local_matrix(self.pose_parameters)
        new_local_m = np.dot(delta_m, local_m)
        new_local_q = quaternion_from_matrix(new_local_m)
        self.set_channel_values(new_local_q, [joint_name])

    def get_joint_direction(self, joint_name, ref_vector):
        q = self.evaluate_orientation(joint_name)
        q /= np.linalg.norm(q)
        rotation_matrix = quaternion_matrix(q)
        vec = np.dot(rotation_matrix, ref_vector)[:3]
        return vec/np.linalg.norm(vec)

    def get_parent_joint(self, joint_name):
        if joint_name not in list(self.skeleton.parent_dict.keys()):
            return None
        return self.skeleton.parent_dict[joint_name]

    def orient_hands_to_each_other(self):
        l = "LeftHand"
        r = "RightHand"
        left_hand_position = self.skeleton.nodes[l].get_global_position(self.pose_parameters)
        right_hand_position = self.skeleton.nodes[r].get_global_position(self.pose_parameters, use_cache=True)
        ldir = right_hand_position - left_hand_position
        ldir /= np.linalg.norm(ldir)
        rdir = -ldir
        self.point_hand_up_dir_in_direction(l, ldir, self.get_parent_joint(l))
        self.point_hand_up_dir_in_direction(r, rdir, self.get_parent_joint(r))

