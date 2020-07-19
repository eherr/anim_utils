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
from transformations import quaternion_matrix, quaternion_multiply, quaternion_about_axis, quaternion_from_matrix
from .constants import OPENGL_UP_AXIS
from .utils import normalize, align_axis, find_rotation_between_vectors, align_root_translation, to_local_cos, get_quaternion_rotation_by_name, apply_additional_rotation_on_frames, project_vector_on_axis, quaternion_from_vector_to_vector
from ..animation_data.skeleton_models import JOINT_CHILD_MAP


def create_local_cos_map_using_child_map(skeleton, up_vector, x_vector, child_map=None):
    joint_cos_map = dict()
    for j in list(skeleton.nodes.keys()):
        joint_cos_map[j] = dict()
        joint_cos_map[j]["y"] = up_vector
        joint_cos_map[j]["x"] = x_vector

        if j == skeleton.root:
            joint_cos_map[j]["x"] = (-np.array(x_vector)).tolist()
        else:
            o = np.array([0, 0, 0])
            if child_map is not None and j in child_map:
                child_name = child_map[j]
                node = skeleton.nodes[child_name]
                o = np.array(node.offset)
            elif len(skeleton.nodes[j].children) > 0:
                node = skeleton.nodes[j].children[0]
                o = np.array(node.offset)
            o = normalize(o)
            if sum(o * o) > 0:
                joint_cos_map[j]["y"] = o
    return joint_cos_map


def create_local_cos_map(skeleton, up_vector, x_vector):
    joint_cos_map = dict()
    for j in list(skeleton.nodes.keys()):
        joint_cos_map[j] = dict()
        joint_cos_map[j]["y"] = up_vector
        joint_cos_map[j]["x"] = x_vector
        if j == skeleton.root:
            joint_cos_map[j]["x"] = (-np.array(x_vector)).tolist()
    return joint_cos_map


def get_body_x_axis(skeleton):
    rh = skeleton.skeleton_model["joints"]["right_hip"]
    lh = skeleton.skeleton_model["joints"]["left_hip"]
    return get_body_axis(skeleton, rh, lh)

def get_body_y_axis(skeleton):
    a = skeleton.skeleton_model["joints"]["pelvis"]
    b = skeleton.skeleton_model["joints"]["head"]
    return get_body_axis(skeleton, a,b)

def get_quaternion_to_axis(skeleton, joint_a, joint_b, axis):
    ident_f = skeleton.identity_frame
    ap = skeleton.nodes[joint_a].get_global_position(ident_f)
    bp = skeleton.nodes[joint_b].get_global_position(ident_f)
    delta = bp - ap
    delta /= np.linalg.norm(delta)
    return quaternion_from_vector_to_vector(axis, delta)


def get_body_axis(skeleton, joint_a, joint_b, project=True):
    ident_f = skeleton.identity_frame
    ap = skeleton.nodes[joint_a].get_global_position(ident_f)
    bp = skeleton.nodes[joint_b].get_global_position(ident_f)
    delta = bp - ap
    m = np.linalg.norm(delta)
    if m != 0:
        delta /= m
        if project:
            projection = project_vector_on_axis(delta)
            return projection / np.linalg.norm(projection)
        else:
            return delta
    else:
        return None

def rotate_axes(cos, q):
    m = quaternion_matrix(q)[:3, :3]
    for key, a in list(cos.items()):
        cos[key] = np.dot(m, a)
        cos[key] = normalize(cos[key])
    return cos

def get_child_joint(skeleton, inv_joint_map, node_name):
    """ Warning output is random if there are more than one child joints
        and the value is not specified in the JOINT_CHILD_MAP """
    child_node = None
    if len(skeleton.nodes[node_name].children) > 0:
        child_node = skeleton.nodes[node_name].children[-1]
    if node_name in inv_joint_map:
        joint_name = inv_joint_map[node_name]
        while joint_name in JOINT_CHILD_MAP:
            child_joint_name = JOINT_CHILD_MAP[joint_name]

            # check if child joint is mapped
            joint_key = None
            if child_joint_name in skeleton.skeleton_model["joints"]:
                joint_key = skeleton.skeleton_model["joints"][child_joint_name]

            if joint_key is not None and joint_key in skeleton.nodes: # return child joint
                child_node = skeleton.nodes[joint_key]
                return child_node
            else: #keep traversing until end of child map is reached
                if child_joint_name in JOINT_CHILD_MAP:
                    joint_name = JOINT_CHILD_MAP[child_joint_name]
                else:
                    break
    return child_node

def create_local_cos_map_from_skeleton_axes_with_map(skeleton, flip=1.0, project=True):
    body_x_axis = get_body_x_axis(skeleton)*flip
    #print("body x axis", body_x_axis)
    body_y_axis = get_body_y_axis(skeleton)
    #print("body y axis", body_y_axis)
    inv_joint_map = dict((v,k) for k, v in skeleton.skeleton_model["joints"].items())
    joint_cos_map = dict()
    for j in list(skeleton.nodes.keys()):
        joint_cos_map[j] = dict()
        joint_cos_map[j]["y"] = body_y_axis
        joint_cos_map[j]["x"] = body_x_axis

        node = skeleton.nodes[j]
        child_node = get_child_joint(skeleton, inv_joint_map, node.node_name)
        if child_node is None:
            continue

        y_axis = get_body_axis(skeleton, j, child_node.node_name, project)
        if y_axis is not None:
            joint_cos_map[j]["y"] = y_axis
            #check if the new y axis is similar to the x axis
            z_vector = np.cross(y_axis, joint_cos_map[j]["x"])
            if np.linalg.norm(z_vector) == 0.0:
                joint_cos_map[j]["x"] = body_y_axis * -np.sum(joint_cos_map[j]["y"])
            #check for angle and rotate
            q = get_quaternion_to_axis(skeleton, j, child_node.node_name, y_axis)
            rotate_axes(joint_cos_map[j], q)
        else:
            joint_cos_map[j]["y"] = None
            joint_cos_map[j]["x"] = None
    return joint_cos_map


def align_root_joint(axes, global_src_x_vec, max_iter_count=10):
    # handle special case for the root joint
    # apply only the y axis rotation of the Hip to the Game_engine node
    not_aligned = True
    q = [1, 0, 0, 0]
    iter_count = 0
    while not_aligned:
        qx, axes = align_axis(axes, "x", global_src_x_vec)  # first find rotation to align x axis
        q = quaternion_multiply(qx, q)
        q = normalize(q)
        qy, axes = align_axis(axes, "y", OPENGL_UP_AXIS)  # then add a rotation to let the y axis point up
        q = quaternion_multiply(qy, q)
        q = normalize(q)
        dot_y = np.dot(axes["y"], OPENGL_UP_AXIS)
        dot_y = min(1, max(dot_y, -1))
        a_y = math.acos(dot_y)
        dot_x = np.dot(axes["x"], global_src_x_vec)
        dot_x = min(1, max(dot_x, -1))
        a_x = math.acos(dot_x)
        iter_count += 1
        not_aligned = a_y > 0.1 or a_x > 0.1 and iter_count < max_iter_count
    return q

def align_joint(new_skeleton, free_joint_name, local_target_axes, global_src_up_vec, global_src_x_vec, joint_cos_map, apply_spine_fix=False):
    # first align the twist axis
    q, axes = align_axis(local_target_axes, "y", global_src_up_vec)
    q = normalize(q)
    # then align the swing axis
    qx, axes = align_axis(axes, "x", global_src_x_vec)
    q = quaternion_multiply(qx, q)
    q = normalize(q)

    # handle cases when twist axis alignment was lost
    dot = np.dot(axes["y"], global_src_up_vec)
    if dot <= -1:
        q180 = quaternion_about_axis(np.deg2rad(180), global_src_x_vec)
        q180 = normalize(q180)
        q = quaternion_multiply(q180, q)
        q = normalize(q)
    elif abs(dot) != 1.0:
        qy, axes = align_axis(axes, "y", global_src_up_vec)
        q = quaternion_multiply(qy, q)
        q = normalize(q)
    return q


def find_rotation_analytically(new_skeleton, joint_name, global_src_up_vec, global_src_x_vec, frame, joint_cos_map, apply_spine_fix=False, apply_root_fix=False, max_iter_count=10):
    local_target_axes = joint_cos_map[joint_name]
    if joint_name == new_skeleton.root and apply_root_fix:
        q = align_root_joint(local_target_axes, global_src_x_vec, max_iter_count)
    else:
        q = align_joint(new_skeleton, joint_name, local_target_axes, global_src_up_vec, global_src_x_vec, joint_cos_map)
    return to_local_cos(new_skeleton, joint_name, frame, q)

        
def create_correction_map(target_skeleton,target_to_src_joint_map, src_cos_map, target_cos_map):
    correction_map = dict()
    for target_name in target_to_src_joint_map:
        src_name = target_to_src_joint_map[target_name]
        if src_name in src_cos_map and target_name is not None and target_name in target_cos_map:
            src_zero_vector_y = src_cos_map[src_name]["y"]
            target_zero_vector_y = target_cos_map[target_name]["y"]
            src_zero_vector_x = src_cos_map[src_name]["x"]
            target_zero_vector_x = target_cos_map[target_name]["x"]
            if target_zero_vector_y is not None and src_zero_vector_y is not None:
                q = quaternion_from_vector_to_vector(target_zero_vector_y, src_zero_vector_y)
                q = normalize(q)

                m = quaternion_matrix(q)[:3, :3]
                target_zero_vector_x = normalize(np.dot(m, target_zero_vector_x))
                qx = quaternion_from_vector_to_vector(target_zero_vector_x, src_zero_vector_x)
                q = quaternion_multiply(qx, q)
                q = normalize(q)
                correction_map[target_name] = q
    return correction_map


class Retargeting(object):
    def __init__(self, src_skeleton, target_skeleton, target_to_src_joint_map, scale_factor=1.0, additional_rotation_map=None, constant_offset=None, place_on_ground=False, force_root_translation=False, ground_height=0):
        self.src_skeleton = src_skeleton
        self.target_skeleton = target_skeleton
        self.target_to_src_joint_map = target_to_src_joint_map
        self.src_to_target_joint_map = {v: k for k, v in list(self.target_to_src_joint_map.items())}
        self.scale_factor = scale_factor
        self.n_params = len(self.target_skeleton.animated_joints) * 4 + 3
        self.ground_height = ground_height
        self.rotation_offsets = additional_rotation_map
        self.src_inv_joint_map = dict((v,k) for k, v in src_skeleton.skeleton_model["joints"].items())
        self.src_child_map = dict()
        for src_name in self.src_skeleton.animated_joints:
            src_child = get_child_joint(self.src_skeleton, self.src_inv_joint_map, src_name)
            if src_child is not None:
                self.src_child_map[src_name] = src_child.node_name
            else:
                self.src_child_map[src_name] = None
        self.target_cos_map = create_local_cos_map_from_skeleton_axes_with_map(self.target_skeleton)
        self.src_cos_map = create_local_cos_map_from_skeleton_axes_with_map(self.src_skeleton)

        if "cos_map" in target_skeleton.skeleton_model:
            self.target_cos_map.update(target_skeleton.skeleton_model["cos_map"])
        if "cos_map" in src_skeleton.skeleton_model:
            self.src_cos_map.update(src_skeleton.skeleton_model["cos_map"])
        self.correction_map = dict()
        
        spine_joints =  ["pelvis","spine", "spine_1","spine_2"]
        target_joint_map = self.target_skeleton.skeleton_model["joints"]
        self.target_spine_joints =[target_joint_map[j] for j in spine_joints if j in target_joint_map]
        
        self.correction_map = create_correction_map(self.target_skeleton, self.src_to_target_joint_map, self.src_cos_map, self.target_cos_map)
        self.constant_offset = constant_offset
        self.place_on_ground = place_on_ground
        self.force_root_translation = force_root_translation
        self.apply_spine_fix = "neck" in target_joint_map and self.src_skeleton.animated_joints != self.target_skeleton.animated_joints
        if "root" in target_joint_map:
            self.apply_root_fix = self.target_skeleton.skeleton_model["joints"]["root"] is not None # aligns root up axis with src skeleton up axis
            # make sure the src root joint in the target is not None
            target_root = self.target_skeleton.skeleton_model["joints"]["root"]
            if self.apply_root_fix and target_to_src_joint_map[target_root] is None:
                target_to_src_joint_map[target_root] = self.src_skeleton.root
        else:
            self.apply_root_fix = False
        if scale_factor <= 0:
            self.auto_scale_factor()

    def auto_scale_factor(self):
        """ estimate scale from leg length by gemlongman """
        target_hip_h = self.target_skeleton.get_body_hip2foot_height()
        src_hip_h = self.src_skeleton.get_body_hip2foot_height()
        self.scale_factor = target_hip_h / src_hip_h
        print("debug scale_factor :" + str(target_hip_h)+ " / " +str(src_hip_h) + " = " +str(self.scale_factor))

    def rotate_bone(self, src_name, target_name, src_frame, target_frame, guess):
        q = guess
        src_x_axis = self.src_cos_map[src_name]["x"]
        src_up_axis = self.src_cos_map[src_name]["y"]
        if self.src_cos_map[src_name]["y"] is not None and self.target_cos_map[target_name]["y"] is not None:
            global_m = self.src_skeleton.nodes[src_name].get_global_matrix(src_frame)[:3, :3]
            global_src_up_vec = normalize(np.dot(global_m, src_up_axis))
            global_src_x_vec = normalize(np.dot(global_m, src_x_axis))
            apply_spine_fix = self.apply_spine_fix and target_name in self.target_spine_joints
            q = find_rotation_analytically(self.target_skeleton, target_name, global_src_up_vec, global_src_x_vec, target_frame, self.target_cos_map, apply_spine_fix, self.apply_root_fix)
        return q

    def rotate_bone_fast(self, src_name, target_name, src_frame, target_frame, quess):
        q = quess
        src_child_name = self.src_skeleton.nodes[src_name].children[0].node_name
        if src_child_name in self.src_to_target_joint_map:
            m = self.src_skeleton.nodes[src_name].get_global_matrix(src_frame)[:3, :3]
            gq = quaternion_from_matrix(m)
            correction_q = self.correction_map[target_name]
            q = quaternion_multiply(gq, correction_q)
            q = normalize(q)
            q = to_local_cos(self.target_skeleton, target_name, target_frame, q)
        return q

    def retarget_frame(self, src_frame, ref_frame):
        target_frame = np.zeros(self.n_params)
        # copy the root translation assuming the rocketbox skeleton with static offset on the hips is used as source
        target_frame[0] = src_frame[0] * self.scale_factor
        target_frame[1] = src_frame[1] * self.scale_factor
        target_frame[2] = src_frame[2] * self.scale_factor

        if self.constant_offset is not None:
            target_frame[:3] += self.constant_offset
        animated_joints = self.target_skeleton.animated_joints
        target_offset = 3

        for target_name in animated_joints:
            q = get_quaternion_rotation_by_name(target_name, self.target_skeleton.reference_frame, self.target_skeleton, root_offset=3)

            if target_name in self.target_to_src_joint_map.keys():

                src_name = self.target_to_src_joint_map[target_name]
                if src_name is not None and len(self.src_skeleton.nodes[src_name].children)>0:
                    q = self.rotate_bone(src_name, target_name, src_frame, target_frame, q)

            if ref_frame is not None:
                #  align quaternion to the reference frame to allow interpolation
                #  http://physicsforgames.blogspot.de/2010/02/quaternions.html
                ref_q = ref_frame[target_offset:target_offset + 4]
                if np.dot(ref_q, q) < 0:
                    q = -q
            target_frame[target_offset:target_offset + 4] = q
            target_offset += 4

        # apply offset on the root taking the orientation into account
        if self.force_root_translation:
            aligning_root = self.target_skeleton.skeleton_model["joints"]["pelvis"]
            target_frame = align_root_translation(self.target_skeleton, target_frame, src_frame, aligning_root, self.scale_factor)
        return target_frame

    def run(self, src_frames, frame_range):
        n_frames = len(src_frames)
        target_frames = []
        if n_frames > 0:
            if frame_range is None:
                frame_range = (0, n_frames)
            if self.rotation_offsets is not None:
               src_frames = apply_additional_rotation_on_frames(self.src_skeleton.animated_joints, src_frames, self.rotation_offsets)

            ref_frame = None
            for idx, src_frame in enumerate(src_frames[frame_range[0]:frame_range[1]]):
                target_frame = self.retarget_frame(src_frame, ref_frame)
                if ref_frame is None:
                    ref_frame = target_frame
                target_frames.append(target_frame)
            target_frames = np.array(target_frames)
            if self.place_on_ground:
                delta = target_frames[0][1] - self.ground_height
                target_frames[:,1] -= delta
        return target_frames


def generate_joint_map(src_model, target_model, joint_filter=None):
    print(target_model.keys())
    joint_map = dict()
    for j in src_model["joints"]:
        if joint_filter is not None and j not in joint_filter:
            continue
        if j in target_model["joints"]:
            src = src_model["joints"][j]
            target = target_model["joints"][j]
            joint_map[target] = src
    return joint_map


def retarget_from_src_to_target(src_skeleton, target_skeleton, src_frames, joint_map=None, additional_rotation_map=None, scale_factor=1.0, frame_range=None, place_on_ground=False, joint_filter=None, force_root_translation=False):
    if joint_map is None:
        joint_map = generate_joint_map(src_skeleton.skeleton_model, target_skeleton.skeleton_model, joint_filter)
    retargeting = Retargeting(src_skeleton, target_skeleton, joint_map, scale_factor, additional_rotation_map=additional_rotation_map, place_on_ground=place_on_ground, force_root_translation=force_root_translation)
    return retargeting.run(src_frames, frame_range)
