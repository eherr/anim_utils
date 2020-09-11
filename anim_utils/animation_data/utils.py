#!/usr/bin/env python
#
# Copyright 2019 DFKI GmbH, Daimler AG.
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
Created on Thu Feb 12 20:11:35 2015

@author: Erik Herrmann, Han Du
"""
import os
import glob
from math import sqrt, radians, sin, cos, isnan
from copy import deepcopy
import numpy as np
from scipy.stats  import linregress
from collections import OrderedDict
from .constants import *
from transformations import quaternion_matrix, euler_from_matrix, \
    quaternion_from_matrix, euler_matrix, \
    quaternion_multiply, \
    quaternion_about_axis, \
    rotation_matrix, \
    quaternion_conjugate, \
    quaternion_inverse, \
    rotation_from_matrix, \
    quaternion_slerp, \
    quaternion_conjugate,  quaternion_inverse, rotation_from_matrix


def rotation_order_to_string(rotation_order):
    r_order_string = "r"
    for c in rotation_order:
        if c == "Xrotation":
            r_order_string += "x"
        elif c == "Yrotation":
            r_order_string += "y"
        elif c == "Zrotation":
            r_order_string += "z"
    return r_order_string


def get_arc_length_from_points(points):
    """
    Note: accuracy depends on the granulariy of points
    """
    points = np.asarray(points)
    arc_length = 0.0
    last_p = None
    for p in points:
        if last_p is not None:
            delta = p - last_p
            arc_length += np.linalg.norm(delta)
        last_p = p
    return arc_length



def quaternion_data_smoothing(quat_frames):
    smoothed_quat_frames = {}
    filenames = quat_frames.keys()
    smoothed_quat_frames_data = np.asarray(deepcopy(quat_frames.values()))
    print('quaternion frame data shape: ', smoothed_quat_frames_data.shape)
    assert len(smoothed_quat_frames_data.shape) == 3, ('The shape of quaternion frames is not correct!')
    n_samples, n_frames, n_dims = smoothed_quat_frames_data.shape
    n_joints = (n_dims - 3)/4
    for i in range(n_joints):
        ref_quat = smoothed_quat_frames_data[0, 0, i*LEN_QUAT + LEN_ROOT : (i+1)*LEN_QUAT + LEN_ROOT]
        for j in range(1, n_samples):
            test_quat = smoothed_quat_frames_data[j, 0, i*LEN_QUAT + LEN_ROOT : (i+1)*LEN_QUAT + LEN_ROOT]
            if not areQuatClose(ref_quat, test_quat):
                smoothed_quat_frames_data[j, :, i*LEN_QUAT + LEN_ROOT : (i+1)*LEN_QUAT + LEN_ROOT] *= -1
    for i in range(len(filenames)):
        smoothed_quat_frames[filenames[i]] = smoothed_quat_frames_data[i]
    return smoothed_quat_frames


def areQuatClose(quat1, quat2):
    dot = np.dot(quat1, quat2)
    if dot < 0.0:
        return False
    else:
        return True


def get_step_length(frames):
    """

    :param frames: a list of euler or quaternion frames
    :return step_len: travelled distance of root joint
    """
    root_points = extract_root_positions(frames)
    step_len = get_arc_length_from_points(root_points)
    return step_len


def quaternion_from_vector_to_vector(a, b):
    """src: http://stackoverflow.com/questions/1171849/finding-quaternion-representing-the-rotation-from-one-vector-to-another
    http://wiki.ogre3d.org/Quaternion+and+Rotation+Primer"""

    v = np.cross(a, b)
    w = np.sqrt((np.linalg.norm(a) ** 2) * (np.linalg.norm(b) ** 2)) + np.dot(a, b)
    q = np.array([w, v[0], v[1], v[2]])
    if np.dot(q,q) != 0:
        return q/ np.linalg.norm(q)
    else:
        idx = np.nonzero(a)[0]
        q = np.array([0, 0, 0, 0])
        q[1 + ((idx + 1) % 2)] = 1 # [0, 0, 1, 0] for a rotation of 180 around y axis
        return q


def convert_euler_frame_to_reduced_euler_frame(bvhreader, euler_frame):
    if type(euler_frame) != list:
        euler_frame = list(euler_frame)
    reduced_euler_frame = euler_frame[:LEN_ROOT]
    for node_name in bvhreader.node_names:
        if not node_name.startswith('Bip') and 'EndSite' not in node_name:
            x_idx = bvhreader.node_channels.index((node_name, 'Xrotation'))
            reduced_euler_frame += euler_frame[x_idx: x_idx + LEN_EULER]
    return reduced_euler_frame


def convert_euler_frames_to_reduced_euler_frames(bvhreader, euler_frames):
    reduced_euler_frames = []
    for euler_frame in euler_frames:
        reduced_euler_frames.append(convert_euler_frame_to_reduced_euler_frame(bvhreader, euler_frame))
    return reduced_euler_frames


def euler_substraction(theta1, theta2):
    ''' compute the angular distance from theta2 to theta1, positive value is anti-clockwise, negative is clockwise
    @param theta1, theta2: angles in degree
    '''
    theta1 %= 360
    theta2 %= 360
    if theta1 > 180:
        theta1 -= 360
    elif theta1 < -180:
        theta1 += 360

    if theta2 > 180:
        theta2 -= 360
    elif theta2 < - 180:
        theta2 += 360

    theta = theta1 - theta2
    if theta > 180:
        theta -= 360
    elif theta < -180:
        theta += 360
    if theta > 180 or theta < - 180:
        raise ValueError(' exception value')
    return theta


def get_cartesian_coordinates_from_quaternion(skeleton, node_name, quaternion_frame, return_gloabl_matrix=False):
    """Returns cartesian coordinates for one node at one frame. Modified to
     handle frames with omitted values for joints starting with "Bip"

    Parameters
    ----------
    * node_name: String
    \tName of node
     * skeleton: Skeleton
    \tBVH data structure read from a file

    """
    if skeleton.node_names[node_name]["level"] == 0:
        root_frame_position = quaternion_frame[:3]
        root_node_offset = skeleton.node_names[node_name]["offset"]

        return [t + o for t, o in
                zip(root_frame_position, root_node_offset)]

    else:
        # Names are generated bottom to up --> reverse
        chain_names = list(skeleton.gen_all_parents(node_name))
        chain_names.reverse()
        chain_names += [node_name]  # Node is not in its parent list

        offsets = [skeleton.node_names[nodename]["offset"]
                   for nodename in chain_names]
        root_position = quaternion_frame[:3].flatten()
        offsets[0] = [r + o for r, o in zip(root_position, offsets[0])]

        j_matrices = []
        count = 0
        for node_name in chain_names:
            index = skeleton.node_name_frame_map[node_name] * 4 + 3
            j_matrix = quaternion_matrix(quaternion_frame[index: index + 4])
            j_matrix[:, 3] = offsets[count] + [1]
            j_matrices.append(j_matrix)
            count += 1

        global_matrix = np.identity(4)
        for j_matrix in j_matrices:
            global_matrix = np.dot(global_matrix, j_matrix)
        if return_gloabl_matrix:
            return global_matrix
        else:
            point = np.array([0, 0, 0, 1])
            point = np.dot(global_matrix, point)
            return point[:3].tolist()


def convert_euler_frame_to_cartesian_frame(skeleton, euler_frame, animated_joints=None):
    """
    converts euler frames to cartesian frames by calling get_cartesian_coordinates for each joint
    """
    if animated_joints is None:
        n_joints = len(skeleton.animated_joints)
        cartesian_frame = np.zeros([n_joints, LEN_ROOT_POS])
        for i in range(n_joints):
            cartesian_frame[i] = skeleton.nodes[skeleton.animated_joints[i]].get_global_position_from_euler(euler_frame)
    else:
        n_joints = len(animated_joints)
        cartesian_frame = np.zeros([n_joints, LEN_ROOT_POS])
        for i in range(n_joints):
            cartesian_frame[i] = skeleton.nodes[animated_joints[i]].get_global_position_from_euler(euler_frame)
    return cartesian_frame


def convert_quaternion_frame_to_cartesian_frame(skeleton, quat_frame):
    """
    Converts quaternion frames to cartesian frames by calling get_cartesian_coordinates_from_quaternion for each joint
    """
    cartesian_frame = []
    for node_name in skeleton.node_names:
        # ignore Bip joints and end sites
        if not node_name.startswith(
                "Bip") and "children" in list(skeleton.node_names[node_name].keys()):
            cartesian_frame.append(
                get_cartesian_coordinates_from_quaternion(
                    skeleton,
                    node_name,
                    quat_frame))  # get_cartesian_coordinates2

    return cartesian_frame


def convert_quaternion_frames_to_cartesian_frames(skeleton, quat_frames):
    cartesian_frames = []
    for i in range(len(quat_frames)):
        cartesian_frames.append(convert_quaternion_frame_to_cartesian_frame(skeleton, quat_frames[i]))
    return cartesian_frames


def transform_invariant_point_cloud_distance_2d(a, b, weights=None):
    '''

    :param a:
    :param b:
    :param weights:
    :return:
    '''
    theta, offset_x, offset_z = align_point_clouds_2D(a, b, weights)
    transformed_b = transform_point_cloud(b, theta, offset_x, offset_z)
    dist = calculate_point_cloud_distance(a, transformed_b)
    return dist


def align_point_clouds_2D(a, b, weights=None):
    '''
     Finds aligning 2d transformation of two equally sized point clouds.
     from Motion Graphs paper by Kovar et al.
     Parameters
     ---------
     *a: list
     \t point cloud
     *b: list
     \t point cloud
     *weights: list
     \t weights of correspondences
     Returns
     -------
     *theta: float
     \t angle around y axis in radians
     *offset_x: float
     \t
     *offset_z: float

     '''
    if len(a) != len(b):
        raise ValueError("two point cloud should have the same number points: "+str(len(a))+","+str(len(b)))
    n_points = len(a)
    if weights is None:
        weights = np.ones(n_points)
    numerator_left = 0.0
    denominator_left = 0.0
    #    if not weights:
    #        weight = 1.0/n_points # todo set weight base on joint level
    weighted_sum_a_x = 0.0
    weighted_sum_b_x = 0.0
    weighted_sum_a_z = 0.0
    weighted_sum_b_z = 0.0
    sum_of_weights = 0.0
    for index in range(n_points):
        numerator_left += weights[index] * (a[index][0] * b[index][2] -
                                            b[index][0] * a[index][2])
        denominator_left += weights[index] * (a[index][0] * b[index][0] +
                                              a[index][2] * b[index][2])
        sum_of_weights += weights[index]
        weighted_sum_a_x += weights[index] * a[index][0]
        weighted_sum_b_x += weights[index] * b[index][0]
        weighted_sum_a_z += weights[index] * a[index][2]
        weighted_sum_b_z += weights[index] * b[index][2]
    numerator_right = 1.0 / sum_of_weights * \
        (weighted_sum_a_x * weighted_sum_b_z -
         weighted_sum_b_x * weighted_sum_a_z)
    numerator = numerator_left - numerator_right
    denominator_right = 1.0 / sum_of_weights * \
        (weighted_sum_a_x * weighted_sum_b_x +
         weighted_sum_a_z * weighted_sum_b_z)
    denominator = denominator_left - denominator_right
    theta = np.arctan2(numerator, denominator)
    offset_x = (weighted_sum_a_x - weighted_sum_b_x * np.cos(theta) - weighted_sum_b_z * np.sin(theta)) / sum_of_weights
    offset_z = (weighted_sum_a_z + weighted_sum_b_x * np.sin(theta) - weighted_sum_b_z * np.cos(theta)) / sum_of_weights
    return theta, offset_x, offset_z


def convert_euler_frames_to_cartesian_frames(skeleton, euler_frames, animated_joints=None):
    """
    converts to euler frames to cartesian frames
    """
    cartesian_frames = []
    for euler_frame in euler_frames:
        cartesian_frames.append(
            convert_euler_frame_to_cartesian_frame(skeleton, euler_frame, animated_joints))
    return np.array(cartesian_frames)



def transform_point_by_quaternion(point, quaternion, offset, origin=None):
    """
    http://math.stackexchange.com/questions/40164/how-do-you-rotate-a-vector-by-a-unit-quaternion
    :param point:
    :param quaternion:
    :return:
    """
    if origin is not None:
        origin = np.asarray(origin)
        # print "point",point,origin
        point = point - origin
    else:
        origin = [0,0,0]
    homogenous_point = np.append([0], point)
    tmp_q = quaternion_multiply(quaternion, homogenous_point)
    temp_q = quaternion_multiply(tmp_q, quaternion_conjugate(quaternion))
    new_point = [temp_q[i+1] + offset[i] + origin[i] for i in range(3)]
    return new_point


def transform_point_by_quaternion_faster(point, quaternion, offset, origin=None):
    """
    http://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication/
    :param point:
    :param quaternion:
    :return:
    """
    if origin is not None:
        origin = np.asarray(origin)
        # print "point",point,origin
        point = point - origin
    else:
        origin = [0,0,0]
    t = 2 * np.cross(quaternion[1:], point)
    new_point = np.array(point) + quaternion[0] * t + np.cross(quaternion[1:], t)
    new_point = [new_point[i] + offset[i] + origin[i] for i in range(3)]
    return new_point


def transform_point_by_quaternion_faster2(point, quaternion, offset, origin=None):
    """
    http://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
    :param point:
    :param quaternion:
    :return:
    """
    if origin is not None:
        origin = np.asarray(origin)
        # print "point",point,origin
        point = point - origin
    else:
        origin = [0,0,0]
    u = quaternion[1:]
    s = quaternion[0]
    new_point = 2.0 * np.dot(u, point) * u + (s*s - np.dot(u, u)) * point + 2.0 * s * np.cross(u, point)
    new_point = [new_point[i] + offset[i] + origin[i] for i in range(3)]
    return new_point


def euler_angles_to_rotation_matrix(euler_angles, rotation_order=DEFAULT_ROTATION_ORDER):
        # generate rotation matrix based on rotation order
    assert len(euler_angles) == 3, ('The length of rotation angles should be 3')
    if round(euler_angles[0], 3) == 0 and round(euler_angles[2], 3) == 0:
        # rotation about y axis
        R = rotation_matrix(np.deg2rad(euler_angles[1]), [0, 1, 0])
    else:
        euler_angles = np.deg2rad(euler_angles)
        R = euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes=rotation_order_to_string(rotation_order))
    return R


def create_transformation_matrix(translation, euler_angles, rotation_order=DEFAULT_ROTATION_ORDER):
    #m = np.eye(4,4)
    m = euler_angles_to_rotation_matrix(euler_angles, rotation_order)
    m[:3,3] = translation
    return m


def transform_point(point, euler_angles, offset, origin=None, rotation_order=DEFAULT_ROTATION_ORDER):
    """
    rotate point around y axis and translate it by an offset
    Parameters
    ---------
    *point: numpy.array
    \t coordinates
    *angles: list of floats
    \tRotation angles in degrees
    *offset: list of floats
    \tTranslation
    """
    assert len(point) == 3, ('the point should be a list of length 3')
    # translate point to original point
    point = np.asarray(point)
    if origin is not None:
        origin = np.asarray(origin)
        point = point - origin
    point = np.insert(point, len(point), 1)
    R = euler_angles_to_rotation_matrix(euler_angles, rotation_order=rotation_order)
    rotated_point = np.dot(R, point)
    if origin is not None:
        rotated_point[:3] += origin
    #print rotated_point,
    transformed_point = rotated_point[:3] + offset
    return transformed_point


def smooth_quaternion_frames(quaternion_frames, discontinuity, window=20):
    """ Smooth quaternion frames given discontinuity frame

    Parameters
    ----------
    quaternion_frames: list
    \tA list of quaternion frames
    discontinuity : int
    The frame where the discontinuity is. (e.g. the transitionframe)
    window : (optional) int, default is 20
    The smoothing window
    Returns
    -------
    None.
    """
    n_joints = (len(quaternion_frames[0]) - 3) / 4
    # smooth quaternion
    n_frames = len(quaternion_frames)
    for i in range(n_joints):
        for j in range(n_frames - 1):
            q1 = np.array(quaternion_frames[j][3 + i * 4: 3 + (i + 1) * 4])
            q2 = np.array(quaternion_frames[j + 1][3 + i * 4:3 + (i + 1) * 4])
            if np.dot(q1, q2) < 0:
                quaternion_frames[
                    j + 1][3 + i * 4:3 + (i + 1) * 4] = -quaternion_frames[j + 1][3 + i * 4:3 + (i + 1) * 4]
    # generate curve of smoothing factors
    d = float(discontinuity)
    w = float(window)
    smoothing_factors = []
    for f in range(n_frames):
        value = 0.0
        if d - w <= f < d:
            tmp = (f - d + w) / w
            value = 0.5 * tmp ** 2
        elif d <= f <= d + w:
            tmp = (f - d + w) / w
            value = -0.5 * tmp ** 2 + 2 * tmp - 2
        smoothing_factors.append(value)
    smoothing_factors = np.array(smoothing_factors)
    new_quaternion_frames = []
    for i in range(len(quaternion_frames[0])):
        current_value = quaternion_frames[:, i]
        magnitude = current_value[int(d)] - current_value[int(d) - 1]
        new_value = current_value + (magnitude * smoothing_factors)
        new_quaternion_frames.append(new_value)
    new_quaternion_frames = np.array(new_quaternion_frames).T
    return new_quaternion_frames


def smooth_quaternion_frames_partially(quaternion_frames, joint_parameter_indices, discontinuity, window=20):
    """ Smooth quaternion frames given discontinuity frame

    Parameters
    ----------
    quaternion_frames: list
    \tA list of quaternion frames
    parameters_indices: list
    \tThe list of joint parameter indices that should be smoothed
    discontinuity : int
    The frame where the discontinuity is. (e.g. the transitionframe)
    window : (optional) int, default is 20
    The smoothing window
    Returns
    -------
    None.
    """
    n_joints = (len(quaternion_frames[0]) - 3) / 4
    # smooth quaternion
    n_frames = len(quaternion_frames)
    for i in range(n_joints):
         for j in range(n_frames - 1):
             q1 = np.array(quaternion_frames[j][3 + i * 4: 3 + (i + 1) * 4])
             q2 = np.array(quaternion_frames[j + 1][3 + i * 4:3 + (i + 1) * 4])
             if np.dot(q1, q2) < 0:
                quaternion_frames[
                    j + 1][3 + i * 4:3 + (i + 1) * 4] = -quaternion_frames[j + 1][3 + i * 4:3 + (i + 1) * 4]
    # generate curve of smoothing factors
    transition_frame = float(discontinuity)
    window_size = float(window)
    smoothing_factors = []
    for frame_number in range(n_frames):
        value = 0.0
        if transition_frame - window_size <= frame_number < transition_frame:
            tmp = (frame_number - transition_frame + window_size) / window_size
            value = 0.5 * tmp ** 2
        elif transition_frame <= frame_number <= transition_frame + window_size:
            tmp = (frame_number - transition_frame + window_size) / window_size
            value = -0.5 * tmp ** 2 + 2 * tmp - 2
        smoothing_factors.append(value)
    #smoothing_factors = np.array(smoothing_factors)
    transition_frame = int(transition_frame)
    #new_quaternion_frames = []
    magnitude_vector = np.array(quaternion_frames[transition_frame]) - quaternion_frames[transition_frame-1]
    #print "magnitude ", magnitude_vector
    for i in joint_parameter_indices:
        joint_parameter_index = i*4+3
        for frame_index in range(n_frames):
            quaternion_frames[frame_index][joint_parameter_index] = quaternion_frames[frame_index][joint_parameter_index]+magnitude_vector[joint_parameter_index] * smoothing_factors[frame_index]




def get_rotation_angle(point1, point2):
    """
    estimate the rotation angle from point2 to point1
    point1, point2 are normalized points
    rotate point2 to be the same as point1

    Parameters
    ----------
    *point1, point2: list or numpy array
    \tUnit 2d points

    Return
    ------
    *rotation_angle: float (in degree)
    \tRotation angle from point2 to point1
    """
    theta1 = point_to_euler_angle(point1)
    theta2 = point_to_euler_angle(point2)
    rotation_angle = euler_substraction(theta2, theta1)
    return rotation_angle


def point_to_euler_angle(vec):
    '''
    @brief: covert a 2D point vec = (cos, sin) to euler angle (in degree)
    The output range is [-180, 180]
    '''
    vec = np.array(vec)
    theta = np.rad2deg(np.arctan2(vec[1], vec[0]))
    return theta


def calculate_point_cloud_distance(a, b):
    """
    calculates the distance between two point clouds with equal length and
    corresponding distances
    """
    assert len(a) == len(b)
    distance = 0
    n_points = len(a)
    for i in range(n_points):
        d = [a[i][0] - b[i][0], a[i][1] - b[i][1], a[i][2] - b[i][2]]
        distance += sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    return distance / n_points


def rotate_and_translate_point(p, theta, offset_x, offset_z):
    """rotate and translate a 3d point, first rotation them translation
       theta is in radians
    """
    rotation_angles = [0, theta, 0]
    rotation_angles = np.rad2deg(rotation_angles)
    offset = [offset_x, 0, offset_z]
    transformed_point = transform_point(p, rotation_angles, offset)
    return transformed_point


def transform_point_cloud(point_cloud, theta, offset_x, offset_z):
    """
    transforms points in a point cloud by a rotation around y and a translation
    along x and z
    """
    transformed_point_cloud = []
    for p in point_cloud:
        if p is not None:
            transformed_point_cloud.append(
                rotate_and_translate_point(p, theta, offset_x, offset_z))
    return transformed_point_cloud


def calculate_pose_distance(skeleton, euler_frames_a, euler_frames_b):
    ''' Converts euler frames to point clouds and finds the aligning transformation
        and calculates the distance after the aligning transformation
    '''

    #    theta, offset_x, offset_z = find_aligning_transformation(bvh_reader, euler_frames_a, euler_frames_b, node_name_map)
    point_cloud_a = convert_euler_frame_to_cartesian_frame(
        skeleton, euler_frames_a[-1])
    point_cloud_b = convert_euler_frame_to_cartesian_frame(
        skeleton, euler_frames_b[0])

    weights = skeleton.joint_weights
    theta, offset_x, offset_z = align_point_clouds_2D(
        point_cloud_a, point_cloud_b, weights)
    t_point_cloud_b = transform_point_cloud(
        point_cloud_b, theta, offset_x, offset_z)
    error = calculate_point_cloud_distance(point_cloud_a, t_point_cloud_b)
    return error


def calculate_frame_distance(skeleton,
                             euler_frame_a,
                             euler_frame_b,
                             weights=None,
                             return_transform=False):
    point_cloud_a = convert_euler_frame_to_cartesian_frame(skeleton,
                                                           euler_frame_a)
    point_cloud_b = convert_euler_frame_to_cartesian_frame(skeleton,
                                                           euler_frame_b)
    theta, offset_x, offset_z = align_point_clouds_2D(point_cloud_a, point_cloud_b, weights=weights)
    t_point_cloud_b = transform_point_cloud(point_cloud_b, theta, offset_x, offset_z)
    error = calculate_point_cloud_distance(point_cloud_a, t_point_cloud_b)
    if return_transform:
        return error, theta, offset_x, offset_z
    else:
        return error


def quat_distance(quat_a, quat_b):
    # normalize quaternion vector first
    quat_a = np.asarray(quat_a)
    quat_b = np.asarray(quat_b)
    quat_a /= np.linalg.norm(quat_a)
    quat_b /= np.linalg.norm(quat_b)
    rotmat_a = quaternion_matrix(quat_a)
    rotmat_b = quaternion_matrix(quat_b)
    diff_mat = rotmat_a - rotmat_b
    tmp = np.ravel(diff_mat)
    diff = np.linalg.norm(tmp)
    return diff


def calculate_weighted_frame_distance_quat(quat_frame_a,
                                           quat_frame_b,
                                           weights):
    assert len(quat_frame_a) == len(quat_frame_b) and \
        len(quat_frame_a) == (len(weights) - 2) * LEN_QUAT + LEN_ROOT
    diff = 0
    for i in range(len(weights) - 2):
        quat1 = quat_frame_a[(i+1)*LEN_QUAT+LEN_ROOT_POS: (i+2)*LEN_QUAT+LEN_ROOT_POS]
        quat2 = quat_frame_b[(i+1)*LEN_QUAT+LEN_ROOT_POS: (i+2)*LEN_QUAT+LEN_ROOT_POS]
        tmp = quat_distance(quat1, quat2)*weights[i]
        diff += tmp
    return diff



def extract_root_positions(frames):
    """

    :param frames: a list of euler or quaternion frames
    :return roots_2D: a list of root position in 2D space
    """
    roots_2D = []
    for i in range(len(frames)):
        position_2D = [frames[i][0], frames[i][2]]
        roots_2D.append(position_2D)

    return roots_2D


def pose_orientation_quat(quaternion_frame, param_range=(3,7), ref_offset=[0, 0, 1, 1]):
    """Estimate pose orientation from root orientation
    """
    ref_offset = np.array(ref_offset)
    q = quaternion_frame[param_range[0]:param_range[1]]
    rotmat = quaternion_matrix(q)
    rotated_point = np.dot(rotmat, ref_offset)
    dir_vec = np.array([rotated_point[0], rotated_point[2]])
    dir_vec /= np.linalg.norm(dir_vec)
    return dir_vec

def pose_orientation_euler(euler_frame, ref_offset = np.array([0, 0, 1, 1])):
    '''
    only works for rocketbox skeleton
    :param euler_frame:
    :return:
    '''
    rot_angles = euler_frame[3:6]
    rot_angles_rad = np.deg2rad(rot_angles)
    rotmat = euler_matrix(*rot_angles_rad, 'rxyz')
    rotated_point = np.dot(rotmat, ref_offset)
    dir_vec = np.array([rotated_point[0],0, rotated_point[2]])
    dir_vec /= np.linalg.norm(dir_vec)
    return dir_vec


def pose_orientation_general(euler_frame, joints, skeleton, rotation_order=['Xrotation', 'Yrotation', 'Zrotation']):
    '''
    estimate pose heading direction, the position of joints define a body plane, the projection of the normal of
    the body plane is the heading direction
    :param euler_frame:
    :param joints:
    :return:
    '''
    points = []
    for joint in joints:
        points.append(skeleton.nodes[joint].get_global_position_from_euler(euler_frame, rotation_order))
    points = np.asarray(points)
    body_plane = BodyPlane(points)
    normal_vec = body_plane.normal_vector
    dir_vec = np.array([normal_vec[0], normal_vec[2]])
    return dir_vec/np.linalg.norm(dir_vec)


def pose_orientation_from_point_cloud(point_cloud, body_plane_joint_indices):
    assert len(point_cloud.shape) == 2
    points = []
    for index in body_plane_joint_indices:
        points.append(point_cloud[index])
    points = np.asarray(points)
    body_plane = BodyPlane(points)
    normal_vec = body_plane.normal_vector
    dir_vec = np.array([normal_vec[0], normal_vec[2]])
    return dir_vec/np.linalg.norm(dir_vec)


def get_trajectory_dir_from_2d_points(points):
    """Estimate the trajectory heading

    Parameters
    *\Points: numpy array
    Step 1: fit the points with a 2d straight line
    Step 2: estimate the direction vector from first and last point
    """
    points = np.asarray(points)
    dir_vector = points[-1] - points[0]
    slope, intercept, r_value, p_value, std_err = linregress(*points.T)
    if isnan(slope):
        orientation_vec1 = np.array([0, 1])
        orientation_vec2 = np.array([0, -1])
    else:
        orientation_vec1 = np.array([slope, 1])
        orientation_vec2 = np.array([-slope, -1])
    if np.dot(
            orientation_vec1,
            dir_vector) > np.dot(
            orientation_vec2,
            dir_vector):
        orientation_vec = orientation_vec1
    else:
        orientation_vec = orientation_vec2
    orientation_vec = orientation_vec / np.linalg.norm(orientation_vec)
    return orientation_vec


def get_dir_from_2d_points(points):
    points = np.asarray(points)
    dir = points[-1] - points[2]
    dir = dir/np.linalg.norm(dir)
    return dir

def pose_up_vector_euler(euler_frame):
    """
    Estimate up vector of given euler frame. Assume the pose is aligned in negative Z axis, so the projection of
    3D up vector into XOY plane (Y is vertical axis) is taken into consideration.
    :param euler_frame:
    :return:
    """
    ref_offset = np.array([1, 0, 0, 1])
    rot_angles = euler_frame[3:6]
    rot_angles_rad = np.deg2rad(rot_angles)
    rotmat = euler_matrix(rot_angles_rad[0],
                          rot_angles_rad[1],
                          rot_angles_rad[2],
                          'rxyz')
    rotated_point = np.dot(rotmat, ref_offset)
    up_vec = np.array([rotated_point[0], rotated_point[1]])
    up_vec /= np.linalg.norm(up_vec)
    return up_vec



def quaternion_rotate_vector(q, vector):
    """ src: http://math.stackexchange.com/questions/40164/how-do-you-rotate-a-vector-by-a-unit-quaternion
    Args:
        q:
        vector:

    Returns:

    """
    tmp_result = quaternion_multiply(q, vector)
    return quaternion_multiply(tmp_result, quaternion_conjugate(q))[1:]

 
def point_rotation_by_quaternion(point, q):
    """
    Rotate a 3D point by quaternion
    :param point: [x, y, z]
    :param q: [qw, qx, qy, qz]
    :return rotated_point: [x, y, ]
    """
    r = [0, point[0], point[1], point[2]]
    q_conj = [q[0], -1*q[1], -1*q[2], -1*q[3]]
    return quaternion_multiply(quaternion_multiply(q, r), q_conj)[1:]


def get_lookAt_direction(bvh_analyzer, frame_index):
    ref_vec = np.array([0, 0, 1, 0])
    global_trans = bvh_analyzer.get_global_transform("Head", frame_index)
    dir_vec = np.dot(global_trans, ref_vec)
    return np.asarray([dir_vec[0], dir_vec[2]])


def find_rotation_angle_between_two_vector(v1, v2):
    from scipy.optimize import minimize
    '''
    positive angle means counterclockwise, negative angle means clockwise
    :return:
    '''
    def objective_function(theta, data):
        src_dir, target_dir = data
        rotmat = np.zeros((2, 2))
        rotmat[0, 0] = np.cos(theta)
        rotmat[0, 1] = -np.sin(theta)
        rotmat[1, 0] = np.sin(theta)
        rotmat[1, 1] = np.cos(theta)
        rotated_src_dir = np.dot(rotmat, src_dir)
        return np.linalg.norm(rotated_src_dir - target_dir)

    initial_guess = 0
    params = [v1, v2]
    res = minimize(objective_function, initial_guess, args=params, method='L-BFGS-B')
    return np.rad2deg(res.x[0])


def get_rotation_angles_for_vectors(dir_vecs, ref_dir, up_axis):
    '''

    :param dir_vecs: nd array N*3
    :param ref_dir: 3d array
    :param up_axis: 3d array
    :return:
    '''
    axis_indice = np.where(up_axis == 0)
    angles = []
    for i in range(len(dir_vecs)):
        angles.append(get_rotation_angle(dir_vecs[i][axis_indice], ref_dir[axis_indice]))
    return np.deg2rad(np.asarray(angles))


def smooth_quat_frames(quat_frames):
    smoothed_quat_frames = OrderedDict()
    filenames = list(quat_frames.keys())
    smoothed_quat_frames_data = np.asarray(deepcopy(list(quat_frames.values())))
    print(('quaternion frame data shape: ', smoothed_quat_frames_data.shape))
    assert len(smoothed_quat_frames_data.shape) == 3, ('The shape of quaternion frames is not correct!')
    n_samples, n_frames, n_dims = smoothed_quat_frames_data.shape
    n_joints = int((n_dims - 3) / 4)
    for i in range(n_joints):
        ref_quat = smoothed_quat_frames_data[0, 0,
                   i * LEN_QUAT + LEN_ROOT: (i + 1) * LEN_QUAT + LEN_ROOT]
        for j in range(1, n_samples):
            test_quat = smoothed_quat_frames_data[j, 0,
                        i * LEN_QUAT + LEN_ROOT: (i + 1) * LEN_QUAT + LEN_ROOT]
            if not areQuatClose(ref_quat, test_quat):
                smoothed_quat_frames_data[j, :,
                i * LEN_QUAT + LEN_ROOT: (i + 1) * LEN_QUAT + LEN_ROOT] *= -1
    for i in range(len(filenames)):
        smoothed_quat_frames[filenames[i]] = smoothed_quat_frames_data[i]
    return smoothed_quat_frames


def combine_motion_clips(clips, motion_len, window_step):
    '''

    :param clips:
    :param motion_len:
    :param window_step:
    :return:
    '''
    clips = np.asarray(clips)
    n_clips, window_size, n_dims = clips.shape

    ## case 1: motion length is smaller than window_step
    if motion_len <= window_step:
        assert n_clips == 1
        left_index = (window_size - motion_len) // 2 + (window_size - motion_len) % 2
        right_index = window_size - (window_size - motion_len) // 2
        return clips[0][left_index: right_index]

    ## case 2: motion length is larger than window_step and smaller than window
    if motion_len > window_step and motion_len <= window_size:
        assert n_clips == 2
        left_index = (window_size - motion_len) // 2 + (window_size - motion_len) % 2
        right_index = window_size - (window_size - motion_len) // 2
        return clips[0][left_index: right_index]

    residue_frames = motion_len % window_step
    print('residue_frames: ', residue_frames)
    ## case 3: residue frames is smaller than window step
    if residue_frames <= window_step:
        residue_frames += window_step
        combined_frames = np.concatenate(clips[0:n_clips - 2, :window_step], axis=0)
        left_index = (window_size - residue_frames) // 2 + (window_size - residue_frames) % 2
        right_index = window_size - (window_size - residue_frames) // 2
        combined_frames = np.concatenate((combined_frames, clips[-2, left_index:right_index]), axis=0)
        return combined_frames
