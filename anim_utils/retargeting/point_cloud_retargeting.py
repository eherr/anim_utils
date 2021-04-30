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
from transformations import quaternion_matrix, quaternion_multiply, quaternion_about_axis, quaternion_inverse, quaternion_from_matrix
from .utils import normalize, align_axis, find_rotation_between_vectors, align_root_translation, to_local_cos, get_quaternion_rotation_by_name, apply_additional_rotation_on_frames, project_vector_on_axis, quaternion_from_vector_to_vector
from ..animation_data.skeleton_models import JOINT_CHILD_MAP
from .analytical import create_local_cos_map_from_skeleton_axes_with_map

JOINT_CHILD_MAP = dict()
JOINT_CHILD_MAP["root"] = "pelvis"
JOINT_CHILD_MAP["pelvis"] = "spine_2"
JOINT_CHILD_MAP["spine_2"] = "neck"
JOINT_CHILD_MAP["neck"] = "head"
JOINT_CHILD_MAP["left_clavicle"] = "left_shoulder"
JOINT_CHILD_MAP["left_shoulder"] = "left_elbow"
JOINT_CHILD_MAP["left_elbow"] = "left_wrist"
JOINT_CHILD_MAP["left_wrist"] = "left_finger"
JOINT_CHILD_MAP["right_clavicle"] = "right_shoulder"
JOINT_CHILD_MAP["right_shoulder"] = "right_elbow"
JOINT_CHILD_MAP["right_elbow"] = "right_wrist"
JOINT_CHILD_MAP["right_wrist"] = "right_finger"
JOINT_CHILD_MAP["left_hip"] = "left_knee"
JOINT_CHILD_MAP["left_knee"] = "left_ankle"
JOINT_CHILD_MAP["right_elbow"] = "right_wrist"
JOINT_CHILD_MAP["right_hip"] = "right_knee"
JOINT_CHILD_MAP["right_knee"] = "right_ankle"
JOINT_CHILD_MAP["left_ankle"] = "left_toe"
JOINT_CHILD_MAP["right_ankle"] = "right_toe"


def estimate_correction(target_zero_vector_y, target_zero_vector_x,  src_zero_vector_y, src_zero_vector_x):
    q = quaternion_from_vector_to_vector(target_zero_vector_y, src_zero_vector_y)
    q = normalize(q)
    m = quaternion_matrix(q)[:3, :3]
    target_zero_vector_x = normalize(np.dot(m, target_zero_vector_x))
    qx = quaternion_from_vector_to_vector(target_zero_vector_x, src_zero_vector_x)
    q = quaternion_multiply(qx, q)
    q = normalize(q)
    return q

def create_correction_map(target_skeleton, target_to_src_joint_map, target_cos_map, src_cos_map):
    correction_map = dict()
    joint_map = target_skeleton.skeleton_model["joints"]
    for target_name in target_to_src_joint_map:
        src_name = target_to_src_joint_map[target_name]
        if src_name in src_cos_map and target_name is not None:
            src_zero_vector_y = src_cos_map[src_name]["y"]
            target_zero_vector_y = target_cos_map[target_name]["y"]
            src_zero_vector_x = src_cos_map[src_name]["x"]
            target_zero_vector_x = target_cos_map[target_name]["x"]
            if target_zero_vector_y is not None and src_zero_vector_y is not None:
                q = estimate_correction(target_zero_vector_y, target_zero_vector_x,  src_zero_vector_y, src_zero_vector_x)
                correction_map[target_name] = q
    return correction_map


def get_quaternion_to_axis(skeleton, joint_a, joint_b, axis):
    ident_f = skeleton.identity_frame
    ap = skeleton.nodes[joint_a].get_global_position(ident_f)
    bp = skeleton.nodes[joint_b].get_global_position(ident_f)
    delta = bp - ap
    delta /= np.linalg.norm(delta)
    return quaternion_from_vector_to_vector(axis, delta)


def rotate_axes2(cos, q):
    m = quaternion_matrix(q)[:3, :3]
    aligned_axes = dict()
    for key, a in list(cos.items()):
        aligned_axes[key] = np.dot(m, a)
        aligned_axes[key] = normalize(aligned_axes[key])
    return aligned_axes


def get_child_joint(skeleton_model, inv_joint_map, node_name, src_children_map):
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

def rotate_axes_in_place(cos, q):
    m = quaternion_matrix(q)[:3, :3]
    for key, a in list(cos.items()):
        cos[key] = np.dot(m, a)
        cos[key] = normalize(cos[key])
    return cos

def align_axis_in_place(axes, key, new_vec):
    q = quaternion_from_vector_to_vector(axes[key], new_vec)
    aligned_axes = rotate_axes_in_place(axes, q)
    return q, aligned_axes


def to_local_cos_fast(skeleton, node_name, frame, q):
    # bring into parent coordinate system
    pm = np.array(skeleton.nodes[node_name].get_global_matrix(frame, use_cache=True)[:3,:3])
    inv_p = quaternion_inverse(quaternion_from_matrix(pm))
    return quaternion_multiply(inv_p, q)

def align_root_joint(new_skeleton, free_joint_name, axes, global_src_up_vec, global_src_x_vec,joint_cos_map, max_iter_count=10):
    # handle special case for the root joint
    # apply only the y axis rotation of the Hip to the Game_engine node
    q = [1, 0, 0, 0]
    #apply first time
    qx, axes = align_axis_in_place(axes, "x", global_src_x_vec)  # first find rotation to align x axis
    q = quaternion_multiply(qx, q)
    q = normalize(q)

    qy, axes = align_axis_in_place(axes, "y", global_src_up_vec)  # then add a rotation to let the y axis point up
    q = quaternion_multiply(qy, q)
    q = normalize(q)

    #apply second time
    qx, axes = align_axis_in_place(axes, "x", global_src_x_vec)  # first find rotation to align x axis
    q = quaternion_multiply(qx, q)
    q = normalize(q)
    qy, axes = align_axis_in_place(axes, "y", global_src_up_vec)  # then add a rotation to let the y axis point up
    q = quaternion_multiply(qy, q)
    q = normalize(q)

    # print("handle special case for pelvis")
    # handle special case of applying the x axis rotation of the Hip to the pelvis
    node = new_skeleton.nodes[free_joint_name]
    t_pose_global_m = node.get_global_matrix(new_skeleton.reference_frame)[:3, :3]
    global_original = np.dot(t_pose_global_m, joint_cos_map[free_joint_name]["y"])
    global_original = normalize(global_original)
    qoffset = find_rotation_between_vectors(OPENGL_UP_AXIS, global_original)
    q = quaternion_multiply(q, qoffset)
    q = normalize(q)
    return q

def align_joint(local_target_axes, up_vec, x_vec):
    q = [1, 0, 0, 0]
    qy, axes = align_axis_in_place(local_target_axes, "y", up_vec)
    q = quaternion_multiply(qy, q)
    q = normalize(q)

    # then align the twisting angles
    if x_vec is not None:
        qx, axes = align_axis_in_place(axes, "x", x_vec)
        q = quaternion_multiply(qx, q)
        q = normalize(q)
        # print("set twist angle", free_joint_name, twist_angle)
    return q

def find_rotation_analytically(new_skeleton, free_joint_name, target, frame, joint_cos_map, is_root=False, max_iter_count=10, twist_angle=None):
    global_src_up_vec = target[0]
    if twist_angle is None:
        global_src_x_vec = target[1]
    else:
        global_src_x_vec = None
    local_target_axes = dict(joint_cos_map[free_joint_name])

    if is_root:
        q = align_root_joint(new_skeleton, free_joint_name, local_target_axes, global_src_up_vec,global_src_x_vec, joint_cos_map, max_iter_count)

    else:
        # first align the bone vectors
        q = [1, 0, 0, 0]
        qy, axes = align_axis_in_place(local_target_axes, "y", global_src_up_vec)
        q = quaternion_multiply(qy, q)
        q = normalize(q)

        # then align the twisting angles
        if global_src_x_vec is not None:
            qx, axes = align_axis_in_place(axes, "x", global_src_x_vec)
            q = quaternion_multiply(qx, q)
            q = normalize(q)

    #if "FK" in free_joint_name:
    #    q = to_local_cos(new_skeleton, free_joint_name, frame, q)
    if new_skeleton.nodes[free_joint_name].parent is not None:
        #if "upLeg" in free_joint_name: # it does not work for the legs for some reason
        #    q = to_local_cos(new_skeleton, new_skeleton.nodes[free_joint_name].parent.node_name, frame, q)
        #else:
        q = to_local_cos(new_skeleton, new_skeleton.nodes[free_joint_name].parent.node_name, frame, q)

    if twist_angle is not None:
        # separate rotation
        local_twist_axis = np.array(joint_cos_map[free_joint_name]["y"])
        swing_q, twist_q = swing_twist_decomposition(q, local_twist_axis)
        # replace
        twist_q = quaternion_about_axis(-twist_angle, local_twist_axis)
        q = quaternion_multiply(swing_q, twist_q)
        q = normalize(q)
    return q


def find_rotation_analytically_with_guess(new_skeleton, free_joint_name, target, frame, joint_cos_map, prev_global_q, is_root=False, max_iter_count = 10):
    global_src_up_vec = target[0]
    global_src_x_vec = target[1]
    local_target_axes = joint_cos_map[free_joint_name]
    rotated_axes = rotate_axes2(local_target_axes, prev_global_q)
    #print("rotate",local_target_axes, rotated_axes, prev_global_q)
    #print("")
    if is_root:
        q = align_root_joint(new_skeleton, free_joint_name, rotated_axes, global_src_up_vec,global_src_x_vec, joint_cos_map, max_iter_count)
    else:
        q = align_joint(rotated_axes, global_src_up_vec, global_src_x_vec)
    q = quaternion_multiply(q, prev_global_q)
    q = normalize(q)
    return to_local_cos_fast(new_skeleton, free_joint_name, frame, q)


def get_parent_map(joints):
    """Returns a dict of node names to their parent node's name"""
    parent_dict = dict()
    for joint_name in joints.keys():
        parent_dict[joint_name] = joints[joint_name]['parent']
    return parent_dict


def get_children_map(joints):
    """Returns a dict of node names to a list of children names"""
    child_dict = dict()
    for joint_name in joints.keys():
        parent_name = joints[joint_name]['parent']
        if parent_name not in child_dict:
            child_dict[parent_name] = list()
        child_dict[parent_name].append(joint_name)
    return child_dict

def swing_twist_decomposition(q, twist_axis):
    """ code by janis sprenger based on
        Dobrowsolski 2015 Swing-twist decomposition in Clifford algebra. https://arxiv.org/abs/1506.05481
    """
    #q = normalize(q)
    #twist_axis = np.array((q * offset))[0]
    projection = np.dot(twist_axis, np.array([q[1], q[2], q[3]])) * twist_axis
    twist_q = np.array([q[0], projection[0], projection[1],projection[2]])
    if np.linalg.norm(twist_q) == 0:
        twist_q = np.array([1,0,0,0])
    twist_q = normalize(twist_q)
    swing_q = quaternion_multiply(q, quaternion_inverse(twist_q))#q * quaternion_inverse(twist)
    return swing_q, twist_q


class PointCloudRetargeting(object):
    def __init__(self, src_joints, src_model, target_skeleton, target_to_src_joint_map, scale_factor=1.0, additional_rotation=None, constant_offset=None, place_on_ground=False, ground_height=0):
        self.src_joints = src_joints
        self.src_model = src_model
        self.target_skeleton = target_skeleton
        self.target_to_src_joint_map = target_to_src_joint_map
        if target_skeleton.skeleton_model["joints"]["pelvis"] is not None:
            self.target_skeleton_root = target_skeleton.skeleton_model["joints"]["pelvis"]
        else:
            self.target_skeleton_root = target_skeleton.root

        #FIXME: enable spine during retargeting
        for j in [ "spine_1", "spine"]:#"spine_2",
            k = self.target_skeleton.skeleton_model["joints"][j]
            self.target_to_src_joint_map[k] = None

        self.src_to_target_joint_map = {v: k for k, v in list(self.target_to_src_joint_map.items())}
        self.scale_factor = scale_factor
        self.n_params = len(self.target_skeleton.animated_joints) * 4 + 3
        self.ground_height = ground_height
        self.additional_rotation = additional_rotation
        self.src_inv_joint_map = dict((v,k) for k, v in src_model["joints"].items())
        self.src_child_map = dict()
        self.src_parent_map = get_parent_map(src_joints)
        src_children_map = get_children_map(src_joints)
        for src_name in self.src_joints:
            src_child = get_child_joint(self.src_model, self.src_inv_joint_map, src_name, src_children_map)
            if src_child is not None:
                self.src_parent_map[src_child] = src_name
                self.src_child_map[src_name] = src_child
            else:
                self.src_child_map[src_name] = None
        #print("ch",self.src_child_map)
        #for j in ["pelvis", "spine", "spine_1", "spine_2"]:
        #    if j in target_joints:
        src_joint_map = self.src_model["joints"]
        for j in ["neck", "spine_2", "spine_1", "spine"]:
            if j in src_joint_map:
                self.src_parent_map["spine_03"] = "pelvis"
        #self.src_child_map[src_joint_map["pelvis"]] = [ "left_eye", "right_eye", "left_ear", "right_ear"]#src_joint_map["neck"]#"pelvis" "neck_01"

        self.constant_offset = constant_offset
        self.place_on_ground = place_on_ground
        self.current_frame_data = dict()
        self.prev_frame_data = dict()
        self.frame_idx = 0
        self.target_cos_map = create_local_cos_map_from_skeleton_axes_with_map(self.target_skeleton)
        if "cos_map" in target_skeleton.skeleton_model:
            self.target_cos_map.update(target_skeleton.skeleton_model["cos_map"])

        target_joints = self.target_skeleton.skeleton_model["joints"]
        self.target_spine_joints = [target_joints[j] for j in ["neck", "spine_2", "spine_1", "spine"] if j in target_joints]#["spine_03", "neck_01"]
        self.target_ball_joints = [target_joints[j] for j in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"] if j in target_joints]# ["thigh_r", "thigh_l", "upperarm_r", "upperarm_l"]
        self.target_ankle_joints = [target_joints[j] for j in ["left_ankle", "right_ankle"] if j in target_joints]
        self.clavicle_joints = [target_joints[j] for j in ["right_clavicle", "left_clavicle"] if j in target_joints]
        self.twist_angle_joints = [target_joints[j] for j in ["right_knee","left_knee","left_shoulder", "right_shoulder", "right_clavicle", "left_clavicle","right_elbow", "left_elbow","left_wrist", "right_wrist"] if j in target_joints]
        if "neck" in target_joints:
            self.target_neck_joint = target_joints["neck"]
        else:
            self.target_neck_joint = None

        left_hip = self.src_model["joints"]["left_hip"]
        right_hip = self.src_model["joints"]["right_hip"]
        self.left_hip_idx = self.src_joints[left_hip]["index"]
        self.right_hip_idx = self.src_joints[right_hip]["index"]
        left_shoulder = self.src_model["joints"]["left_shoulder"]
        right_shoulder = self.src_model["joints"]["right_shoulder"]
        self.left_shoulder_idx = self.src_joints[left_shoulder]["index"]
        self.right_shoulder_idx = self.src_joints[right_shoulder]["index"]
        self.ref_rotation = dict()
        for target_name in self.target_skeleton.animated_joints:
            self.ref_rotation[target_name] = get_quaternion_rotation_by_name(target_name, self.target_skeleton.reference_frame,
                                                                             self.target_skeleton, root_offset=3)


    def estimate_src_joint_cos(self, src_name, child_name, target_name, src_frame):
        joint_idx = self.src_joints[src_name]["index"]
        child_idx = self.src_joints[child_name]["index"]
        if isinstance(child_idx, list):
            child_pos = np.mean([src_frame[idx] for idx in child_idx])
        else:
            child_pos = src_frame[joint_idx]
        global_src_up_vec = src_frame[child_idx] - child_pos
        global_src_up_vec /= np.linalg.norm(global_src_up_vec)
        if target_name == self.target_skeleton.skeleton_model["joints"]["pelvis"]:
            global_src_x_vec = src_frame[self.left_hip_idx] - src_frame[self.right_hip_idx]
            global_src_x_vec /= np.linalg.norm(global_src_x_vec)
        elif target_name in self.target_spine_joints or target_name == "CC_Base_Waist":  # find x vector from shoulders
            global_src_x_vec = src_frame[self.left_shoulder_idx] - src_frame[self.right_shoulder_idx]
            global_src_x_vec /= np.linalg.norm(global_src_x_vec)
        elif target_name in self.target_ball_joints:  # use x vector of child
            child_child_name = self.src_child_map[child_name]
            child_child_idx = self.src_joints[child_child_name]["index"]
            child_global_src_up_vec = src_frame[child_child_idx] - src_frame[child_idx]
            child_global_src_up_vec /= np.linalg.norm(child_global_src_up_vec)
            angle = abs(math.acos(np.dot(global_src_up_vec, child_global_src_up_vec)))
            if angle < 0.01 and src_name in self.prev_frame_data:#np.linalg.norm(delta) < 0.2
                print("use stored at ", self.frame_idx,"for",  src_name, "for child")
                global_src_up_vec = self.prev_frame_data[src_name][0]
                global_src_x_vec = self.prev_frame_data[src_name][1]
            else:
                global_src_x_vec = np.cross(global_src_up_vec, child_global_src_up_vec)
                norm = np.linalg.norm(global_src_x_vec)
                global_src_x_vec /= norm
        else:  # find x vector by cross product with parent
            global_src_x_vec = None
            if src_name in self.src_parent_map:
                parent_joint = self.src_parent_map[src_name]
                if parent_joint in self.current_frame_data:
                    global_parent_up_vector = self.current_frame_data[parent_joint][0]
                    global_src_x_vec = np.cross(global_src_up_vec, global_parent_up_vector)
                    norm = np.linalg.norm(global_src_x_vec)
                    global_src_x_vec /= norm
                    if target_name not in self.target_ankle_joints:
                        global_src_x_vec = -global_src_x_vec
        return global_src_up_vec, global_src_x_vec

    def get_src_cos_from_joint_map(self, src_name, target_name, src_frame):
        src_cos = None
        if src_name not in self.src_child_map.keys() or self.src_child_map[src_name] is None:
            return src_cos
        if self.src_child_map[src_name] in self.src_to_target_joint_map:#  and or target_name =="neck_01" or target_name.startswith("hand")
            child_name = self.src_child_map[src_name]
            if child_name not in self.src_joints.keys():
                return src_cos
            src_cos = self.estimate_src_joint_cos(src_name, child_name, target_name, src_frame)
            self.current_frame_data[src_name] = src_cos
           
        return src_cos


    def get_src_cos_from_multiple_points(self, src_frame, cos_def):
        up_start = np.mean([src_frame[self.src_joints[name]["index"]] for name in cos_def["up_start"]], axis=0)
        up_end = np.mean([src_frame[self.src_joints[name]["index"]] for name in cos_def["up_end"]], axis=0)
        global_src_up_vec = up_end - up_start
        x_start = np.mean([src_frame[self.src_joints[name]["index"]] for name in cos_def["x_start"]], axis=0)
        x_end = np.mean([src_frame[self.src_joints[name]["index"]] for name in cos_def["x_end"]], axis=0)
        global_src_x_vec = x_end - x_start
        global_src_up_vec /= np.linalg.norm(global_src_up_vec)
        global_src_x_vec /= np.linalg.norm(global_src_x_vec)
        return global_src_up_vec, global_src_x_vec


    def retarget_frame(self, src_frame, prev_frame, pose_angles=None):

        target_joints = self.target_skeleton.skeleton_model["joints"]
        joint_map = dict()
        for k, v in target_joints.items():
            joint_map[v] = k
        self.target_skeleton.clear_cached_global_matrices()
        target_frame = np.zeros(self.n_params)
        self.prev_frame_data = self.current_frame_data
        self.current_frame_data = dict()

        # copy the root translation assuming the rocketbox skeleton with static offset on the hips is used as source
        #target_frame[:3] = np.array(src_frame[0]) * self.scale_factor
        target_frame[:3] = src_frame[self.left_hip_idx] +  (src_frame[self.right_hip_idx] - src_frame[self.left_hip_idx])/2
        target_frame[:3] *= self.scale_factor
        if self.constant_offset is not None:
            target_frame[:3] += self.constant_offset
        target_offset = 3
        for target_name in self.target_skeleton.animated_joints:
            q = self.ref_rotation[target_name]
            src_cos = None
            if target_name in self.target_to_src_joint_map:
                src_name = self.target_to_src_joint_map[target_name]
                default_name = joint_map[target_name]
                if default_name in self.src_model["cos_defs"]:
                    src_cos = self.get_src_cos_from_multiple_points(src_frame, self.src_model["cos_defs"][default_name])
                if src_name is not None and src_name in self.src_joints:
                    src_cos = self.get_src_cos_from_joint_map(src_name, target_name, src_frame)
            if src_cos is not None and src_cos[1] is not None:
                twist_angle = None
                if pose_angles is not None and self.target_skeleton.nodes[target_name].parent is not None and target_name in self.twist_angle_joints:
                    joint_idx = self.src_joints[src_name]["index"]
                    twist_angle = pose_angles[joint_idx][0]
                is_root = target_name == self.target_skeleton_root
                q = find_rotation_analytically(self.target_skeleton, target_name, src_cos, target_frame, self.target_cos_map, is_root=is_root, twist_angle=twist_angle)
                q = q/np.linalg.norm(q)

            #if ref_frame is not None:
            #    q = q if np.dot(ref_frame[target_offset:target_offset + 4], q) >= 0 else -q

            if prev_frame is not None:
                prev_q = normalize(prev_frame[target_offset:target_offset + 4])
                if np.dot(q, prev_q) < 0:
                    q = -q
                inv_q = normalize(quaternion_inverse(q))
                delta_q = normalize(quaternion_multiply(inv_q, prev_q))
                theta = 2 * np.arccos(delta_q[0])
                if abs(theta) > np.pi:
                    print("keep", self.frame_idx, theta,src_name, q, prev_q)
                    q = prev_q
            target_frame[target_offset:target_offset + 4] = q
            target_offset += 4
        return target_frame

    def run(self, src_frames, frame_range):
        n_frames = len(src_frames)
        target_frames = []
        self.frame_idx = 0
        if n_frames > 0:
            if frame_range is None:
                frame_range = (0, n_frames)
            if self.additional_rotation is not None:
               src_frames = apply_additional_rotation_on_frames(self.src_skeleton.animated_joints, src_frames, self.additional_rotation)

            prev_frame = None
            for idx, src_frame in enumerate(src_frames[frame_range[0]:frame_range[1]]):
                target_frame = self.retarget_frame(src_frame, prev_frame)
                prev_frame = target_frame
                target_frames.append(target_frame)
                self.frame_idx += 1
                print("--------------")
            target_frames = np.array(target_frames)
            if self.place_on_ground:
                delta = target_frames[0][1] - self.ground_height
                target_frames[:, 1] -= delta
        return target_frames


def generate_joint_map(src_model, target_model):
    joint_map = dict()
    for j in src_model["joints"]:
        if j in target_model["joints"]:
            src = src_model["joints"][j]
            target = target_model["joints"][j]
            joint_map[target] = src
    return joint_map


def retarget_from_point_cloud_to_target(src_joints, src_model, target_skeleton, src_frames, joint_map=None, additional_rotation=None, scale_factor=1.0, frame_range=None, place_on_ground=False):
    if joint_map is None:
        joint_map = generate_joint_map(src_model, target_skeleton.skeleton_model)
    constant_offset = -np.array(target_skeleton.nodes[target_skeleton.root].offset)
    retargeting = PointCloudRetargeting(src_joints, src_model, target_skeleton, joint_map, scale_factor, additional_rotation=additional_rotation, place_on_ground=place_on_ground, constant_offset=constant_offset)
    return retargeting.run(src_frames, frame_range)
