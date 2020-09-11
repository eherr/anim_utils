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
import math
from transformations import quaternion_from_matrix, euler_matrix, quaternion_matrix, quaternion_multiply, euler_from_quaternion, quaternion_from_euler, quaternion_inverse, euler_from_matrix
from ..animation_data.utils import quaternion_from_vector_to_vector

AXES = [[1,0,0],[0,1,0],[0,0,1], [-1,0,0],[0,-1,0],[0,0,-1]]


def get_angle_between_vectors(v1, v2):
    q = quaternion_from_vector_to_vector(v1, v2)
    v = q[1:]
    sin_theta = np.linalg.norm(v)
    sin_theta = min(sin_theta, 1.0)
    abs_angle = 2 * math.asin(sin_theta)
    return abs_angle


def project_vector_on_axis(v):
    min_idx = -1
    min_angle = np.inf
    for idx, a in enumerate(AXES):
        angle = get_angle_between_vectors(v, a)
        if angle < min_angle:
            min_angle = angle
            min_idx = idx
    length = np.linalg.norm(v)
    return np.array(AXES[min_idx]) * length


def align_root_translation(target_skeleton, target_frame, src_frame, root_node="pelvis", src_scale_factor=1.0):
    target_pos = target_skeleton.nodes[root_node].get_global_position(target_frame)
    target_pos = np.array([target_pos[0], target_pos[2]])
    src_pos = np.array([src_frame[0], src_frame[2]])*src_scale_factor
    delta = src_pos - target_pos
    target_frame[0] += delta[0]
    target_frame[2] += delta[1]
    return target_frame


def align_quaternion_frames(target_skeleton, frames):
    """align quaternions for blending
    src: http://physicsforgames.blogspot.de/2010/02/quaternions.html
    """
    ref_frame = None
    new_frames = []
    for frame in frames:
        if ref_frame is None:
            ref_frame = frame
        else:
            offset = 3
            for joint in target_skeleton.animated_joints:
                q = frame[offset:offset + 4]
                ref_q = ref_frame[offset:offset + 4]
                dot = np.dot(ref_q, q)
                if dot < 0:
                    frame[offset:offset + 4] = -q
                offset += 4
        new_frames.append(frame)
    return new_frames


def get_coordinate_system_axes(skeleton, joint_name, frame, axes):
    global_m = skeleton.nodes[joint_name].get_global_matrix(frame)[:3,:3]
    dirs = []
    for axis in axes:
        direction = np.dot(global_m, axis)
        direction /= np.linalg.norm(direction)
        dirs.append(direction)
    return np.array(dirs)


def rotate_axes(axes, q):
    m = quaternion_matrix(q)[:3, :3]
    aligned_axes = dict()
    for key, a in axes.items():
        aligned_axes[key] = np.dot(m, a)
        aligned_axes[key] = normalize(aligned_axes[key])
    return aligned_axes


def align_axis(axes, key, new_vec):
    q = quaternion_from_vector_to_vector(axes[key], new_vec)
    aligned_axes = rotate_axes(axes, q)
    return q, aligned_axes


def get_quaternion_rotation_by_name(joint_name, frame, skeleton, root_offset=3):
    assert joint_name in skeleton.animated_joints
    joint_index = skeleton.animated_joints.index(joint_name)
    return frame[joint_index * 4 + root_offset : (joint_index + 1) * 4 + root_offset]


def normalize(v):
    return v/np.linalg.norm(v)


def filter_dofs(q, fixed_dims):
    e = list(euler_from_quaternion(q))
    for d in fixed_dims:
        e[d] = 0
    q = quaternion_from_euler(*e)
    return q


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


def to_local_cos_old(skeleton, node_name, frame, q):
    # bring into parent coordinate system
    pm = skeleton.nodes[node_name].get_global_matrix(frame)#[:3,:3]
    inv_pm = np.linalg.inv(pm)
    r = quaternion_matrix(q)[:3,:3]
    lr = np.dot(inv_pm, r)[:3,:3]
    q = quaternion_from_matrix(lr)
    return q


def to_local_cos(skeleton, node_name, frame, q):
    # bring into parent coordinate system
    pm = skeleton.nodes[node_name].get_global_matrix(frame)#[:3,:3]
    inv_p = quaternion_inverse(quaternion_from_matrix(pm))
    normalize(inv_p)
    return quaternion_multiply(inv_p, q)


def to_global_cos(skeleton, node_name, frame, q):
    pm = skeleton.nodes[node_name].get_global_matrix(frame)#[:3,:3]
    r = quaternion_matrix(q)[:3, :3]
    lr = np.dot(pm, r)[:3, :3]
    q = quaternion_from_matrix(lr)
    return q


def apply_additional_rotation_on_frames(animated_joints, frames, rotation_offsets):
    new_frames = []
    for frame in frames:
        new_frame = frame[:]
        for idx, name in enumerate(animated_joints):
            if name in rotation_offsets:
                euler = np.radians(rotation_offsets[name])
                additional_q = quaternion_from_euler(*euler)
                offset = idx * 4 + 3
                q = new_frame[offset:offset + 4]
                new_frame[offset:offset + 4] = quaternion_multiply(q, additional_q)

        new_frames.append(new_frame)
    return new_frames
