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
""" Analytical IK for arms and legs based on Section 5.3 of [1] and Section 4.4 of [2]

[1] Lee, Jehee, and Sung Yong Shin. "A hierarchical approach to interactive motion editing for human-like figures."
Proceedings of the 26th annual conference on Computer graphics and interactive techniques. 1999.
[2] Lucas Kovar, John Schreiner, and Michael Gleicher. "Footskate cleanup for motion capture editing." Proceedings of the 2002 ACM SIGGRAPH/Eurographics symposium on Computer animation. ACM, 2002.

"""
import math
import numpy as np
import scipy.integrate as integrate
from transformations import quaternion_multiply, quaternion_about_axis, quaternion_matrix, quaternion_from_matrix, quaternion_inverse


def normalize(v):
    return v/np.linalg.norm(v)


def quaternion_from_axis_angle(axis, angle):
    q = [1,0,0,0]
    q[1] = axis[0] * math.sin(angle / 2)
    q[2] = axis[1] * math.sin(angle / 2)
    q[3] = axis[2] * math.sin(angle / 2)
    q[0] = math.cos(angle / 2)
    return normalize(q)


def find_rotation_between_vectors(a, b):
    """http://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another"""
    if np.array_equiv(a, b):
        return [1, 0, 0, 0]

    axis = normalize(np.cross(a, b))
    dot = np.dot(a, b)
    if dot >= 1.0:
        return [1, 0, 0, 0]
    angle = math.acos(dot)
    q = quaternion_from_axis_angle(axis, angle)
    return q

def calculate_angle(upper_limb, lower_limb, ru, rl, target_length):
    upper_limb_sq = upper_limb * upper_limb
    lower_limb_sq = lower_limb * lower_limb
    ru_sq = ru * ru
    rl_sq = rl * rl
    lusq_rusq = upper_limb_sq - ru_sq
    lusq_rusq = max(0, lusq_rusq)
    llsq_rlsq = lower_limb_sq - rl_sq
    llsq_rlsq = max(0, llsq_rlsq)
    temp = upper_limb_sq + rl_sq
    temp += 2 * math.sqrt(lusq_rusq) * math.sqrt(llsq_rlsq)
    temp += - (target_length*target_length)
    temp /= 2 * ru * rl
    print("temp",temp)
    temp = max(-1, temp)
    return math.acos(temp)


def calculate_angle2(upper_limb,lower_limb,target_length):
    """ get angle between upper and lower limb based on desired length
    https://www.mathsisfun.com/algebra/trig-solving-sss-triangles.html"""
    a = upper_limb
    b = lower_limb
    c = target_length
    temp = (a*a + b*b - c*c) / (2 * a * b)
    temp = min(1, temp)
    temp = max(-1, temp)
    angle = math.acos(temp)
    return angle


def damp_angle(orig_angle, target_angle, p=0.1*np.pi, a=0.01):
    """ src: Kovar et al. [2] Section 4.4. eq. 10 and 11"""
    def func(x):
        if x < p:
            return 1.0
        elif p <= x <= np.pi:
            return a * ((x-p)/(np.pi-p))
        else:
            return 0.0
    res = integrate.quad(func, orig_angle, target_angle)
    return orig_angle + res[0]


def to_local_coordinate_system(skeleton, frame, joint_name, q):
    """ given a global rotation concatenate it with an existing local rotation and bring it to the local coordinate system"""
    # delta*parent*old_local = parent*new_local
    # inv_parent*delta*parent*old_local = new_local
    if skeleton.nodes[joint_name].parent is not None:
        global_m = quaternion_matrix(q)
        parent_joint = skeleton.nodes[joint_name].parent.node_name
        parent_m = skeleton.nodes[parent_joint].get_global_matrix(frame, use_cache=False)
        old_global = np.dot(parent_m, skeleton.nodes[joint_name].get_local_matrix(frame))
        new_global = np.dot(global_m, old_global)
        new_local = np.dot(np.linalg.inv(parent_m), new_global)
        return quaternion_from_matrix(new_local)
    else:
        return q




def calculate_limb_joint_rotation(skeleton, root, joint, end_effector, local_joint_axis, frame, target_position):
    """ find angle so the distance from root to end effector is equal to the distance from the root to the target"""
    root_pos = skeleton.nodes[root].get_global_position(frame)
    joint_pos = skeleton.nodes[joint].get_global_position(frame)
    end_effector_pos = skeleton.nodes[end_effector].get_global_position(frame)

    upper_limb = np.linalg.norm(root_pos - joint_pos)
    lower_limb = np.linalg.norm(joint_pos - end_effector_pos)
    target_length = np.linalg.norm(root_pos - target_position)
    #current_length = np.linalg.norm(root_pos - end_effector_pos)
    #current_angle = calculate_angle2(upper_limb, lower_limb, current_length)
    target_angle = calculate_angle2(upper_limb, lower_limb, target_length)

    joint_delta_angle = np.pi - target_angle
    joint_delta_q = quaternion_about_axis(joint_delta_angle, local_joint_axis)
    joint_delta_q = normalize(joint_delta_q)
    return joint_delta_q


def calculate_limb_root_rotation(skeleton, root, end_effector, frame, target_position):
    """ find angle between the vectors end_effector - root and target- root """

    # align vectors
    root_pos = skeleton.nodes[root].get_global_position(frame)
    end_effector_pos = skeleton.nodes[end_effector].get_global_position(frame)
    src_delta = end_effector_pos - root_pos
    src_dir = src_delta / np.linalg.norm(src_delta)

    target_delta = target_position - root_pos
    target_dir = target_delta / np.linalg.norm(target_delta)

    root_delta_q = find_rotation_between_vectors(src_dir, target_dir)
    root_delta_q = normalize(root_delta_q)

    return to_local_coordinate_system(skeleton, frame, root, root_delta_q)


class AnalyticalLimbIK(object):
    def __init__(self, skeleton, limb_root, limb_joint, end_effector, joint_axis, local_end_effector_dir, damp_angle=None, damp_factor=0.01):
        self.skeleton = skeleton
        self.limb_root = limb_root
        self.limb_joint = limb_joint
        self.end_effector = end_effector
        self.local_joint_axis = joint_axis
        self.local_end_effector_dir = local_end_effector_dir
        joint_idx = self.skeleton.animated_joints.index(self.limb_joint) * 4 + 3
        self.joint_indices = [joint_idx, joint_idx + 1, joint_idx + 2, joint_idx + 3]
        root_idx = self.skeleton.animated_joints.index(self.limb_root) * 4 + 3
        self.root_indices = [root_idx, root_idx + 1, root_idx + 2, root_idx + 3]
        end_effector_idx = self.skeleton.animated_joints.index(self.end_effector) * 4 + 3
        self.end_effector_indices = [end_effector_idx, end_effector_idx + 1, end_effector_idx + 2, end_effector_idx + 3]
        self.damp_angle = damp_angle
        self.damp_factor = damp_factor

    @classmethod
    def init_from_dict(cls, skeleton, joint_name, data, damp_angle=None, damp_factor=None):
        limb_root = data["root"]
        limb_joint = data["joint"]
        joint_axis = data["joint_axis"]
        end_effector_dir = data["end_effector_dir"]
        return AnalyticalLimbIK(skeleton, limb_root, limb_joint, joint_name, joint_axis, end_effector_dir, damp_angle, damp_factor)

    def calculate_limb_joint_rotation(self, frame, target_position):
        """ find angle so the distance from root to end effector is equal to the distance from the root to the target"""
        root_pos = self.skeleton.nodes[self.limb_root].get_global_position(frame)
        joint_pos = self.skeleton.nodes[self.limb_joint].get_global_position(frame)
        end_effector_pos = self.skeleton.nodes[self.end_effector].get_global_position(frame)

        upper_limb = np.linalg.norm(root_pos - joint_pos)
        lower_limb = np.linalg.norm(joint_pos - end_effector_pos)
        target_length = np.linalg.norm(root_pos - target_position)
        current_length = np.linalg.norm(root_pos - end_effector_pos)
        current_angle = calculate_angle2(upper_limb, lower_limb, current_length)
        target_angle = calculate_angle2(upper_limb, lower_limb, target_length)
        if self.damp_angle is not None:
            target_angle = damp_angle(current_angle, target_angle, self.damp_angle, self.damp_factor)
        #if abs(target_angle - np.pi) < self.min_angle:
        #    target_angle -= self.min_angle
        joint_delta_angle = np.pi - target_angle
        joint_delta_q = quaternion_about_axis(joint_delta_angle, self.local_joint_axis)
        joint_delta_q = normalize(joint_delta_q)
        frame[self.joint_indices] = joint_delta_q
        return joint_delta_q

    def calculate_limb_root_rotation(self, frame, target_position):
        """ find angle between the vectors end_effector - root and target- root """

        # align vectors
        root_pos = self.skeleton.nodes[self.limb_root].get_global_position(frame)
        end_effector_pos = self.skeleton.nodes[self.end_effector].get_global_position(frame)
        src_delta = end_effector_pos - root_pos
        src_dir = src_delta / np.linalg.norm(src_delta)

        target_delta = target_position - root_pos
        target_dir = target_delta / np.linalg.norm(target_delta)

        root_delta_q = find_rotation_between_vectors(src_dir, target_dir)
        root_delta_q = normalize(root_delta_q)

        frame[self.root_indices] = self._to_local_coordinate_system(frame, self.limb_root, root_delta_q)

    def _to_local_coordinate_system(self, frame, joint_name, q):
        """ given a global rotation concatenate it with an existing local rotation and bring it to the local coordinate system"""
        # delta*parent*old_local = parent*new_local
        # inv_parent*delta*parent*old_local = new_local
        global_m = quaternion_matrix(q)
        parent_joint = self.skeleton.nodes[joint_name].parent.node_name
        parent_m = self.skeleton.nodes[parent_joint].get_global_matrix(frame, use_cache=False)
        old_global = np.dot(parent_m, self.skeleton.nodes[joint_name].get_local_matrix(frame))
        new_global = np.dot(global_m, old_global)
        new_local = np.dot(np.linalg.inv(parent_m), new_global)
        return quaternion_from_matrix(new_local)

    def calculate_end_effector_rotation(self, frame, target_dir):
        #print "end effector rotation", self.end_effector, target_dir
        #end_effector_m = self.skeleton.nodes[self.end_effector].get_global_matrix(frame)[:3, :3]
        #src_dir = np.dot(end_effector_m, self.local_end_effector_dir)
        #src_dir = normalize(src_dir)
        src_dir = self.get_joint_dir(frame, self.end_effector)
        global_delta_q = find_rotation_between_vectors(src_dir, target_dir)
        new_local_q = self._to_local_coordinate_system(frame, self.end_effector, global_delta_q)
        frame[self.end_effector_indices] = new_local_q

    def set_end_effector_rotation(self, frame, target_orientation):
        #print "set orientation", target_orientation
        q = self.get_global_joint_orientation(self.end_effector, frame)
        delta_orientation = quaternion_multiply(target_orientation, quaternion_inverse(q))
        new_local_q = self._to_local_coordinate_system(frame, self.end_effector, delta_orientation)
        frame[self.end_effector_indices] = new_local_q
        #t = self.skeleton.nodes[self.skeleton.nodes[self.end_effector].children[0].node_name].get_global_position(frame)
        #h = self.skeleton.nodes[self.skeleton.nodes[self.end_effector].children[1].node_name].get_global_position(frame)
        #original_direction = normalize(t - h)
        #print original_direction

    def get_global_joint_orientation(self, joint_name, frame):
        m = self.skeleton.nodes[joint_name].get_global_matrix(frame)
        m[:3, 3] = [0, 0, 0]
        return normalize(quaternion_from_matrix(m))

    def get_joint_dir(self, frame, joint_name):
        pos1 = self.skeleton.nodes[joint_name].get_global_position(frame)
        pos2 = self.skeleton.nodes[joint_name].children[0].get_global_position(frame)
        return normalize(pos2 - pos1)


    def apply(self, frame, position, orientation):
        if position is not None:
            # 1 calculate joint angle based on the distance to target position
            self.calculate_limb_joint_rotation(frame, position)

            # 2 calculate limb root rotation to align the end effector with the target position
            self.calculate_limb_root_rotation(frame, position)

        # 3 orient end effector
        if orientation is not None:
            self.set_end_effector_rotation2(frame, orientation)
        return frame


    def set_end_effector_rotation2(self, frame, target_orientation):
        #print("set", target_orientation)
        new_local_q = self.to_local_cos2(self.end_effector, frame, target_orientation)
        frame[self.end_effector_indices] = new_local_q

    def to_local_cos2(self, joint_name, frame, q):
        # bring into parent coordinate system
        parent_joint = self.skeleton.nodes[joint_name].parent.node_name
        pm = self.skeleton.nodes[parent_joint].get_global_matrix(frame)#[:3, :3]
        inv_p = quaternion_inverse(quaternion_from_matrix(pm))
        normalize(inv_p)
        return quaternion_multiply(inv_p, q)
