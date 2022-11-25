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
"""
Functions for retargeting based on the paper "Using an Intermediate Skeleton and Inverse Kinematics for Motion Retargeting"
by Monzani et al.
See: http://www.vis.uni-stuttgart.de/plain/vdl/vdl_upload/91_35_retargeting%20monzani00using.pdf
"""
import numpy as np
import math
from .constants import OPENGL_UP_AXIS, GAME_ENGINE_SPINE_OFFSET_LIST
from .utils import normalize, align_axis, find_rotation_between_vectors, align_root_translation, to_local_cos, get_quaternion_rotation_by_name, apply_additional_rotation_on_frames, project_vector_on_axis, quaternion_from_vector_to_vector
from transformations import quaternion_matrix, quaternion_multiply, quaternion_about_axis, quaternion_from_matrix
from .point_cloud_retargeting import create_local_cos_map_from_skeleton_axes_with_map, get_parent_map, get_children_map, JOINT_CHILD_MAP




def apply_manual_fixes(joint_cos_map, joints):
    for j in joints:
        if j in joint_cos_map:
            joint_cos_map[j]["x"] *= -1


def get_child_joint2(skeleton_model, inv_joint_map, node_name, src_children_map):
    """ Warning output is random if there are more than one child joints
        and the value is not specified in the JOINT_CHILD_MAP """
    child_name = None
    if node_name in src_children_map and len(src_children_map[node_name]) > 0:
        child_name = src_children_map[node_name][-1]
    if node_name in inv_joint_map:
        joint_name = inv_joint_map[node_name]
        while joint_name in JOINT_CHILD_MAP:
            _child_joint_name = JOINT_CHILD_MAP[joint_name]

            # check if child joint is mapped
            joint_key = None
            if _child_joint_name in skeleton_model["joints"]:
                joint_key = skeleton_model["joints"][_child_joint_name]

            if joint_key is not None: # return child joint
                child_name = joint_key
                return child_name
            else: #keep traversing until end of child map is reached
                if _child_joint_name in JOINT_CHILD_MAP:
                    joint_name = JOINT_CHILD_MAP[_child_joint_name]
                    #print(joint_name)
                else:
                    break
    return child_name


class FastPointCloudRetargeting(object):
    def __init__(self, src_skeleton, src_joints, target_skeleton, target_to_src_joint_map, scale_factor=1.0, additional_rotation_map=None, constant_offset=None, place_on_ground=False, ground_height=0):
        self.src_skeleton = src_skeleton
        self.src_joints = src_joints
        self.src_model = src_skeleton.skeleton_model
        self.target_skeleton = target_skeleton
        self.target_to_src_joint_map = target_to_src_joint_map
        self.src_to_target_joint_map = {v: k for k, v in list(self.target_to_src_joint_map.items())}
        print("src to traget map", self.src_to_target_joint_map)
        print("target to src map", self.target_to_src_joint_map)
        self.scale_factor = scale_factor
        self.n_params = len(self.target_skeleton.animated_joints) * 4 + 3
        self.ground_height = ground_height
        self.additional_rotation_map = additional_rotation_map
        self.src_inv_joint_map = dict((v,k) for k, v in src_skeleton.skeleton_model["joints"].items())

        self.target_cos_map = create_local_cos_map_from_skeleton_axes_with_map(self.target_skeleton)
        self.src_cos_map = create_local_cos_map_from_skeleton_axes_with_map(self.src_skeleton, flip=1.0, project=True)

        if "cos_map" in target_skeleton.skeleton_model:
            self.target_cos_map.update(target_skeleton.skeleton_model["cos_map"])
        if "x_cos_fixes" in target_skeleton.skeleton_model:
            apply_manual_fixes(self.target_cos_map, target_skeleton.skeleton_model["x_cos_fixes"])
        if "cos_map" in src_skeleton.skeleton_model:
            self.src_cos_map.update(src_skeleton.skeleton_model["cos_map"])
        if "x_cos_fixes" in src_skeleton.skeleton_model:
            apply_manual_fixes(self.src_cos_map, src_skeleton.skeleton_model["x_cos_fixes"])
        self.correction_map = dict()
        self.create_correction_map()
        self.constant_offset = constant_offset
        self.place_on_ground = place_on_ground
        self.apply_spine_fix = self.src_skeleton.animated_joints != self.target_skeleton.animated_joints

        self.src_child_map = dict()
        self.src_parent_map = get_parent_map(src_joints)
        src_children_map = get_children_map(src_joints)
        for src_name in self.src_joints:
            src_child = get_child_joint2(self.src_model, self.src_inv_joint_map, src_name, src_children_map)
            if src_child is not None:
                self.src_parent_map[src_child] = src_name
                self.src_child_map[src_name] = src_child
            else:
                self.src_child_map[src_name] = None
        # print("ch",self.src_child_map)
        # for j in ["pelvis", "spine", "spine_1", "spine_2"]:
        #    if j in target_joints:
        src_joint_map = self.src_model["joints"]
        for j in ["neck", "spine_2", "spine_1", "spine"]:
            if j in src_joint_map:
                self.src_parent_map["spine_03"] = "pelvis"
        self.src_child_map[src_joint_map["pelvis"]] = src_joint_map["neck"]  # "pelvis" "neck_01"

        self.constant_offset = constant_offset
        self.place_on_ground = place_on_ground
        self.temp_frame_data = dict()

        self.target_cos_map = create_local_cos_map_from_skeleton_axes_with_map(self.target_skeleton)
        if "cos_map" in target_skeleton.skeleton_model:
            self.target_cos_map.update(target_skeleton.skeleton_model["cos_map"])
        if "x_cos_fixes" in target_skeleton.skeleton_model:
            apply_manual_fixes(self.target_cos_map, target_skeleton.skeleton_model["x_cos_fixes"])

        target_joints = self.target_skeleton.skeleton_model["joints"]
        self.target_spine_joints = [target_joints[j] for j in ["neck", "spine_2", "spine_1", "spine"] if
                                    j in target_joints]  # ["spine_03", "neck_01"]
        self.target_ball_joints = [target_joints[j] for j in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"] if
                                   j in target_joints]  # ["thigh_r", "thigh_l", "upperarm_r", "upperarm_l"]
        self.target_ankle_joints = [target_joints[j] for j in ["left_ankle", "right_ankle"] if j in target_joints]
        self.clavicle_joints = [target_joints[j] for j in ["right_clavicle", "left_clavicle"] if j in target_joints]
        if "neck" in target_joints:
            self.target_neck_joint = target_joints["neck"]
        else:
            self.target_neck_joint = None

    def create_correction_map(self):
        self.correction_map = dict()
        joint_map = self.target_skeleton.skeleton_model["joints"]
        for target_name in self.target_to_src_joint_map:
            src_name = self.target_to_src_joint_map[target_name]
            if src_name in self.src_cos_map and target_name is not None:
                src_zero_vector_y = self.src_cos_map[src_name]["y"]
                target_zero_vector_y = self.target_cos_map[target_name]["y"]
                src_zero_vector_x = self.src_cos_map[src_name]["x"]
                target_zero_vector_x = self.target_cos_map[target_name]["x"]
                if target_zero_vector_y is not None and src_zero_vector_y is not None:
                    q = quaternion_from_vector_to_vector(target_zero_vector_y, src_zero_vector_y)
                    q = normalize(q)

                    if target_name in [joint_map["pelvis"], joint_map["spine"], joint_map["spine_1"]] and False:#,, joint_map["spine_2"]
                        # add offset rotation to spine based on an upright reference pose
                        m = quaternion_matrix(q)[:3, :3]
                        v = normalize(np.dot(m, target_zero_vector_y))
                        node = self.target_skeleton.nodes[target_name]
                        t_pose_global_m = node.get_global_matrix(self.target_skeleton.reference_frame)[:3, :3]
                        global_original = np.dot(t_pose_global_m, v)
                        global_original = normalize(global_original)
                        qoffset = find_rotation_between_vectors(OPENGL_UP_AXIS, global_original)
                        q = quaternion_multiply(qoffset, q)
                        q = normalize(q)

                    m = quaternion_matrix(q)[:3, :3]
                    target_zero_vector_x = normalize(np.dot(m, target_zero_vector_x))
                    qx = quaternion_from_vector_to_vector(target_zero_vector_x, src_zero_vector_x)
                    q = quaternion_multiply(qx, q)
                    q = normalize(q)
                    self.correction_map[target_name] = q

    def rotate_bone(self, src_name, target_name, src_frame, target_frame, quess):
        q = quess
        #src_child_name = self.src_skeleton.nodes[src_name].children[0].node_name
        #print("in", src_child_name in self.src_to_target_joint_map)
        if target_name in self.correction_map:
            m = self.src_skeleton.nodes[src_name].get_global_matrix(src_frame)[:3, :3]
            gq = quaternion_from_matrix(m)
            correction_q = self.correction_map[target_name]
            q = quaternion_multiply(gq, correction_q)
            q = normalize(q)
            q = to_local_cos(self.target_skeleton, target_name, target_frame, q)
        return q

    def get_rotation_from_pc(self, src_name, target_name, src_pc, src_frame):
        child_name = self.src_child_map[src_name]
        global_src_up_vec, global_src_x_vec = self.estimate_src_joint_cos(src_name, child_name, target_name, src_pc)
        if "Spine" in src_name:
            print("axis",src_name, global_src_up_vec, global_src_up_vec)
        q = [1, 0, 0, 0]
        local_target_axes = self.src_cos_map[src_name]
        qy, axes = align_axis(local_target_axes, "y", global_src_up_vec)
        q = quaternion_multiply(qy, q)
        q = normalize(q)
        if global_src_x_vec is not None:
            qx, axes = align_axis(axes, "x", global_src_x_vec)
            q = quaternion_multiply(qx, q)
            q = normalize(q)
        return to_local_cos(self.src_skeleton, src_name, src_frame, q)

    def convert_pc_to_frame(self, src_pc, ref_frame):
        src_frame = np.zeros(ref_frame.shape)
        src_frame[:3] = np.array(src_frame[0])
        src_frame_offset = 3
        animated_joints = self.src_skeleton.animated_joints
        for src_name in animated_joints:
            q = get_quaternion_rotation_by_name(src_name, self.src_skeleton.reference_frame,
                                                self.src_skeleton,
                                                root_offset=3)
            if src_name in self.src_joints.keys():
                src_child_name = self.src_child_map[src_name]
                if src_child_name in self.src_joints.keys() and src_name in self.src_to_target_joint_map.keys():
                    target_name = self.src_to_target_joint_map[src_name]
                    q = self.get_rotation_from_pc(src_name, target_name, src_pc, src_frame)
                else:
                    print("ignore pc", src_name, src_name in self.src_to_target_joint_map.keys())
            if ref_frame is not None:
                q = q if np.dot(ref_frame[src_frame_offset:src_frame_offset + 4], q) >= 0 else -q
            src_frame[src_frame_offset:src_frame_offset + 4] = q
            src_frame_offset += 4
        #print("src", src_frame[3:7])
        return src_frame

    def retarget_frame(self, src_pc, ref_frame):
        print("fast retargeting")
        self.temp_frame_data = dict()
        src_frame = self.convert_pc_to_frame(src_pc, ref_frame)
        target_frame = np.zeros(self.n_params)
        # copy the root translation assuming the rocketbox skeleton with static offset on the hips is used as source
        target_frame[:3] = np.array(src_frame[:3]) * self.scale_factor

        if self.constant_offset is not None:
            target_frame[:3] += self.constant_offset
        animated_joints = self.target_skeleton.animated_joints
        target_offset = 3
        self.src_spine_joints = ["Spine", "Spine1"]
        for target_name in animated_joints:
            q = get_quaternion_rotation_by_name(target_name, self.target_skeleton.reference_frame, self.target_skeleton, root_offset=3)
            if target_name in self.target_to_src_joint_map.keys():
                src_name = self.target_to_src_joint_map[target_name]
                if src_name is not None and len(self.src_skeleton.nodes[src_name].children)>0 and src_name not in self.src_spine_joints:
                    q = self.rotate_bone(src_name, target_name, src_frame, target_frame, q)
                    print("rotate", src_name, target_name)
                else:
                    print("ignore", src_name,target_name)
            if ref_frame is not None and False:
                #  align quaternion to the reference frame to allow interpolation
                #  http://physicsforgames.blogspot.de/2010/02/quaternions.html
                ref_q = ref_frame[target_offset:target_offset + 4]
                if np.dot(ref_q, q) < 0:
                    q = -q
            target_frame[target_offset:target_offset + 4] = q
            target_offset += 4

        return target_frame


    def estimate_src_joint_cos(self, src_name, child_name, target_name, src_frame):
        joint_idx = self.src_joints[src_name]["index"]
        child_idx = self.src_joints[child_name]["index"]
        global_src_up_vec = src_frame[child_idx] - src_frame[joint_idx]
        global_src_up_vec /= np.linalg.norm(global_src_up_vec)
        self.temp_frame_data[src_name] = global_src_up_vec
        if target_name == self.target_skeleton.skeleton_model["joints"]["pelvis"]:
            left_hip = self.src_model["joints"]["left_hip"]
            right_hip = self.src_model["joints"]["right_hip"]
            left_hip_idx = self.src_joints[left_hip]["index"]
            right_hip_idx = self.src_joints[right_hip]["index"]
            global_src_x_vec = src_frame[left_hip_idx] - src_frame[right_hip_idx]
            global_src_x_vec /= np.linalg.norm(global_src_x_vec)
        elif target_name in self.target_spine_joints or target_name == "CC_Base_Waist":  # find x vector from shoulders
            left_shoulder = self.src_model["joints"]["left_shoulder"]
            right_shoulder = self.src_model["joints"]["right_shoulder"]
            left_shoulder_idx = self.src_joints[left_shoulder]["index"]
            right_shoulder_idx = self.src_joints[right_shoulder]["index"]
            global_src_x_vec = src_frame[left_shoulder_idx] - src_frame[right_shoulder_idx]
            global_src_x_vec /= np.linalg.norm(global_src_x_vec)
        elif target_name in self.target_ball_joints:  # use x vector of child
            child_child_name = self.src_child_map[child_name]
            child_child_idx = self.src_joints[child_child_name]["index"]
            child_global_src_up_vec = src_frame[child_child_idx] - src_frame[child_idx]
            child_global_src_up_vec /= np.linalg.norm(child_global_src_up_vec)

            global_src_x_vec = np.cross(global_src_up_vec, child_global_src_up_vec)

            global_src_x_vec /= np.linalg.norm(global_src_x_vec)

        else:  # find x vector by cross product with parent
            global_src_x_vec = None
            if src_name in self.src_parent_map:
                parent_joint = self.src_parent_map[src_name]
                # print("estimate cos", src_name, target_name)
                # if target_name in self.clavicle_joints:
                #    parent_joint = self.target_neck_joint
                #    print("set parent for", target_name, "to", self.target_neck_joint)
                if parent_joint in self.temp_frame_data:
                    global_parent_up_vector = self.temp_frame_data[parent_joint]
                    global_src_x_vec = np.cross(global_src_up_vec, global_parent_up_vector)
                    global_src_x_vec /= np.linalg.norm(global_src_x_vec)
                    # print("apply",src_name, parent_joint, global_src_x_vec)
                    # if target_name in ["calf_l", "calf_r","thigh_r","thigh_l", "spine_03","neck_01","lowerarm_r","lowerarm_l"]:
                    if target_name not in self.target_ankle_joints:
                        global_src_x_vec = - global_src_x_vec
                        # global_src_x_vec = None
                        # if global_src_x_vec is None:
                        #    print("did not find vector", target_name, parent_joint, self.target_skeleton.root)
                        # else:
                        #   print("ignore", target_name)

        return global_src_up_vec, global_src_x_vec

def generate_joint_map(src_model, target_model, joint_filter=None):
    joint_map = dict()
    for j in src_model["joints"]:
        if joint_filter is not None and j not in joint_filter:
            continue
        if j in target_model["joints"]:
            src = src_model["joints"][j]
            target = target_model["joints"][j]
            joint_map[target] = src
    return joint_map



