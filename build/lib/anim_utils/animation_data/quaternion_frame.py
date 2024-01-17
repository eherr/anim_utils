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
Created on Fri Nov 24 14:10:21 2014

@author: Erik Herrmann, Han Du

"""
import collections
import numpy as np
from transformations import euler_matrix, quaternion_from_matrix, quaternion_matrix, euler_from_matrix
from .utils import rotation_order_to_string
from .constants import DEFAULT_ROTATION_ORDER, LEN_EULER, LEN_QUAT, LEN_ROOT_POS




def check_quat(test_quat, ref_quat):
    """check locomotion_synthesis_test quat needs to be filpped or not
    """
    test_quat = np.asarray(test_quat)
    ref_quat = np.asarray(ref_quat)
    dot = np.dot(test_quat, ref_quat)
    if dot < 0:
        test_quat = - test_quat
    return test_quat.tolist()

def euler_to_quaternion(euler_angles, rotation_order=DEFAULT_ROTATION_ORDER,filter_values=True):
    """Convert euler angles to quaternion vector [qw, qx, qy, qz]
    Parameters
    ----------
    * euler_angles: list of floats
    \tA list of ordered euler angles in degress
    * rotation_order: Iteratable
    \t a list that specifies the rotation axis corresponding to the values in euler_angles
    * filter_values: Bool
    \t enforce a unique rotation representation

    """
    assert len(euler_angles) == 3, ('The length of euler angles should be 3!')
    euler_angles = np.deg2rad(euler_angles)
    rotmat = euler_matrix(*euler_angles, rotation_order_to_string(rotation_order))
    # convert rotation matrix R into quaternion vector (qw, qx, qy, qz)
    quat = quaternion_from_matrix(rotmat)
    # filter the quaternion see
    # http://physicsforgames.blogspot.de/2010/02/quaternions.html
    if filter_values:
        dot = np.sum(quat)
        if dot < 0:
            quat = -quat
    return [quat[0], quat[1], quat[2], quat[3]]


def quaternion_to_euler(quat, rotation_order=DEFAULT_ROTATION_ORDER):
    """
    Parameters
    ----------
    * q: list of floats
    \tQuaternion vector with form: [qw, qx, qy, qz]

    Return
    ------
    * euler_angles: list
    \tEuler angles in degree with specified order
    """
    quat = np.asarray(quat)
    quat = quat / np.linalg.norm(quat)
    rotmat_quat = quaternion_matrix(quat)
    rotation_order_str = rotation_order_to_string(rotation_order)
    euler_angles = np.rad2deg(euler_from_matrix(rotmat_quat, rotation_order_str))
    return euler_angles


def convert_euler_to_quaternion_frame(bvh_data, e_frame, filter_values=True, animated_joints=None):
    """Convert a BVH frame into an ordered dict of quaternions for each skeleton node
    Parameters
    ----------
    * bvh_data: BVHData
    \t Contains skeleton information
    * frame_vector: np.ndarray
    \t animation keyframe frame represented by Euler angles
    * filter_values: Bool
    \t enforce a unique rotation representation

    Returns:
    ----------
    * quat_frame: OrderedDict that contains a quaternion for each joint
    """
    if animated_joints is None:
        animated_joints = list(bvh_data.node_names.keys())
    n_dims = len(animated_joints)*4
    quat_frame = np.zeros((n_dims))
    offset = 0
    for node_name in animated_joints:
        if bvh_data.get_node_channels(node_name) is not None:
            angles, order = bvh_data.get_node_angles(node_name, e_frame)
            q = euler_to_quaternion(angles, order, filter_values)
            quat_frame[offset:offset+4] = q
            offset +=4
    return quat_frame


def convert_euler_frames_to_quaternion_frames(bvh_data, euler_frames, filter_values=True, animated_joints=None):
    """
    Parameters
    ----------
    * bvh_data: BVHData
    \t Contains skeleton information
    * euler_frames: np.ndarray
    \t list of euler frames
    Returns:
    ----------
    * quater_frames: a list of quaternion frames
    """
    if animated_joints is None:
        animated_joints = list(bvh_data.node_names.keys())
    quat_frames = []
    prev_frame = None
    for e_frame in euler_frames:
        q_frame = convert_euler_to_quaternion_frame(bvh_data, e_frame, filter_values, animated_joints=animated_joints)
        o = 0
        if prev_frame is not None and filter_values:
            for joint_name in animated_joints:
                q = check_quat(q_frame[o:o+4], prev_frame[o:o+4])
                q_frame[o:o+4] = q
                o+=4
        prev_frame = q_frame
        q_frame = np.concatenate([e_frame[:3],q_frame])
        quat_frames.append(q_frame)
    return quat_frames



def convert_quaternion_frames_to_euler_frames(quat_frames):
    """Returns an nparray of Euler frames

    Parameters
    ----------
     * quat_frames: List of quaternion frames
    \tQuaternion frames that shall be converted to Euler frames

    Returns
    -------
    * euler_frames: numpy array
    \tEuler frames
    """
    n_frames = len(quat_frames)
    assert n_frames > 0 
    n_joints = int((len(quat_frames[0])-LEN_ROOT_POS) /LEN_QUAT)
    n_params = n_joints*LEN_EULER + LEN_ROOT_POS
    euler_frames = np.zeros((n_frames, n_params))
    for frame_idx, quat_frame in enumerate(quat_frames):
        euler_frames[frame_idx,:LEN_ROOT_POS] = quat_frame[:LEN_ROOT_POS]
        src = LEN_ROOT_POS
        dst = LEN_ROOT_POS
        for i in range(n_joints):
            q = quat_frame[src:src+LEN_QUAT]
            euler_frames[frame_idx, dst:dst+LEN_EULER] = quaternion_to_euler(q)
            dst += LEN_EULER
            src += LEN_QUAT
    return euler_frames
