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
from copy import deepcopy
from transformations import euler_from_matrix, quaternion_matrix, quaternion_about_axis, quaternion_multiply, quaternion_from_matrix, quaternion_from_euler, quaternion_slerp, euler_matrix, quaternion_inverse
from .utils import pose_orientation_general, transform_point, rotation_order_to_string, euler_substraction, point_to_euler_angle, euler_angles_to_rotation_matrix, get_rotation_angle, point_rotation_by_quaternion, LEN_QUAT, LEN_EULER, LEN_ROOT_POS, pose_orientation_euler, pose_up_vector_euler
from .constants import DEFAULT_SMOOTHING_WINDOW_SIZE, DEFAULT_ROTATION_ORDER
from .quaternion_frame import euler_to_quaternion, quaternion_to_euler
from .motion_blending import smooth_quaternion_frames_with_slerp, smooth_quaternion_frames, blend_quaternion_frames_linearly, smooth_quaternion_frames_with_slerp2, smooth_euler_frames
from ..motion_editing.motion_grounding import create_grounding_constraint_from_frame, generate_ankle_constraint_from_toe, interpolate_constraints
from ..motion_editing.analytical_inverse_kinematics import AnalyticalLimbIK
from ..motion_editing.utils import normalize, generate_root_constraint_for_two_feet, smooth_root_translation_at_start, smooth_root_translation_at_end, project_on_intersection_circle, global_position_to_root_translation
from .motion_blending import blend_between_frames, smooth_translation_in_quat_frames, generate_blended_frames, interpolate_frames, smooth_root_translation_around_transition, blend_quaternions_to_next_step, smooth_quaternion_frames_joint_filter, blend_towards_next_step_linear_with_original, smooth_root_positions

ALIGNMENT_MODE_FAST = 0
ALIGNMENT_MODE_PCL = 1

def concatenate_frames(prev_frames, new_frames):
    frames = prev_frames.tolist()
    for idx in range(1, len(new_frames)):  # skip first frame
        frames.append(new_frames[idx])
    frames = np.array(frames)
    return frames


def convert_quat_frame_to_point_cloud(skeleton, frame, joints=None):
    points = list()
    if joints is None:
        joints = [k for k, n in list(skeleton.nodes.items()) if len(n.children) > 0 and "Bip" not in n.node_name]
    for j in joints:
        p = skeleton.nodes[j].get_global_position(frame)
        points.append(p)
    return points


def _align_point_clouds_2D(a, b, weights):
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
    numerator_left = 0
    denominator_left = 0
    weighted_sum_a_x = 0
    weighted_sum_b_x = 0
    weighted_sum_a_z = 0
    weighted_sum_b_z = 0
    sum_of_weights = 0.0
    #    if not weights:
    #        weight = 1.0/n_points # todo set weight base on joint level
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
    offset_x = (weighted_sum_a_x - weighted_sum_b_x *
                np.cos(theta) - weighted_sum_b_z * np.sin(theta)) / sum_of_weights
    offset_z = (weighted_sum_a_z + weighted_sum_b_x *
                np.sin(theta) - weighted_sum_b_z * np.cos(theta)) / sum_of_weights

    return theta, offset_x, offset_z


def align_euler_frames(euler_frames,
                       frame_idx,
                       ref_orientation_euler):
    print("Deprecation Warning: Function marked as Deprecated!")
    pass
    new_euler_frames = deepcopy(euler_frames)
    ref_quat = euler_to_quaternion(ref_orientation_euler)
    root_rot_angles = euler_frames[frame_idx][3:6]
    root_rot_quat = euler_to_quaternion(root_rot_angles)
    quat_diff = quaternion_multiply(ref_quat, quaternion_inverse(root_rot_quat))
    for euler_frame in new_euler_frames:
        root_trans = euler_frame[:3]
        new_root_trans = point_rotation_by_quaternion(root_trans, quat_diff)
        euler_frame[:3] = new_root_trans
        root_rot_angles = euler_frame[3:6]
        root_rot_quat = euler_to_quaternion(root_rot_angles)

        new_root_quat = quaternion_multiply(quat_diff, root_rot_quat)
        new_root_euler = quaternion_to_euler(new_root_quat)
        euler_frame[3:6] = new_root_euler
    return new_euler_frames


def translate_euler_frames(euler_frames, frame_idx, ref_position, root_joint, skeleton):
    """
    translate euler frames to ref_position, tralsation is 2D on the ground
    :param euler_frames:
    :param frame_idx:
    :param ref_position:
    :return:
    """
    root_pos = skeleton.nodes[root_joint].get_global_position_from_euler(euler_frames[frame_idx])
    if len(ref_position) == 2:
        offset = np.array([ref_position[0] - root_pos[0], 0, ref_position[1] - root_pos[2]])
    elif len(ref_position) == 3:
        offset = ref_position - root_pos
        offset[1] = 0
    else:
        raise ValueError('the length of reference position is not correct! ')
    rotation = [0, 0, 0]
    return transform_euler_frames(euler_frames, rotation, offset)


def rotate_euler_frame(euler_frame, ref_orientation, body_plane_joints, skeleton):
    forward = pose_orientation_general(euler_frame, body_plane_joints, skeleton)
    rot_angle = get_rotation_angle(ref_orientation, forward)
    translation = np.array([0, 0, 0])
    return transform_euler_frame(euler_frame, [0, rot_angle, 0], translation)


def rotate_euler_frames(euler_frames, frame_idx, ref_orientation, body_plane_joints, skeleton, rotation_order=None):
    """
    Rotate a list of euler frames using the same rotation angle
    :param euler_frames: a list of euler frames to be rotated
    :param frame_idx: frame which uses to calculate rotation anlge
    :param ref_orientation: reference orientation for alignment
    :param rotation_order: define the rotation order for euler angles to calculate rotation matrix
    :global_rotation: if true, the rotation is also applied to the global position. If false, the rotation will not be applied to global position 
    :return rotated_frames: a list of rotated euler frames
    """
    forward = pose_orientation_general(euler_frames[frame_idx],
                                       body_plane_joints,
                                       skeleton)
    rot_angle = get_rotation_angle(ref_orientation, forward)
    translation = np.array([0, 0, 0])
    rotated_frames = transform_euler_frames(euler_frames,
                                            [0, rot_angle, 0],
                                            translation,
                                            rotation_order)
    return rotated_frames


def rotate_euler_frames_about_x_axis(euler_frames, frame_idx, ref_upvector):
    sample_upvector = pose_up_vector_euler(euler_frames[frame_idx])
    rot_angle = get_rotation_angle(ref_upvector, sample_upvector)
    translation = np.array([0, 0, 0])
    rotated_frames = transform_euler_frames(euler_frames,
                                            [rot_angle, 0, 0],
                                            translation)
    return rotated_frames



def smoothly_concatenate(euler_frames_a, euler_frames_b, window_size=20):
    euler_frames = np.concatenate((euler_frames_a, euler_frames_b), axis=0)
    euler_frames = smooth_euler_frames(euler_frames, len(euler_frames_a), window_size)
    return euler_frames

def fast_euler_frames_transformation(euler_frames_a,
                                     euler_frames_b):
    dir_vec_a = pose_orientation_euler(euler_frames_a[-1])
    dir_vec_b = pose_orientation_euler(euler_frames_b[0])
    angle = get_rotation_angle(dir_vec_a, dir_vec_b)
    offset_x = euler_frames_a[-1][0] - euler_frames_b[0][0]
    offset_z = euler_frames_a[-1][2] - euler_frames_b[0][2]
    offset = [offset_x, 0.0, offset_z]
    return angle, offset



def fast_euler_frames_alignment(euler_frames_a,
                                euler_frames_b,
                                smooth=True,
                                smoothing_window=DEFAULT_SMOOTHING_WINDOW_SIZE):
    angle, offset = fast_euler_frames_transformation(euler_frames_a, euler_frames_b)
    transformed_frames = transform_euler_frames(euler_frames_b, [0, angle, 0],offset)
    # concatenate the quaternion_frames_a and transformed_framess
    if smooth:
        euler_frames = smoothly_concatenate(euler_frames_a, transformed_frames, window_size=smoothing_window)
    else:
        euler_frames = np.concatenate((euler_frames_a,
                                            transformed_frames))
    return euler_frames

def transform_quaternion_frames_legacy(quat_frames, angles, offset, rotation_order=None):
    """ Applies a transformation on the root joint of a list quaternion frames.
    Parameters
    ----------
    *quat_frames: np.ndarray
    \tList of frames where the rotation is represented as euler angles in degrees.
    *angles: list of floats
    \tRotation angles in degrees
    *offset:  np.ndarray
    \tTranslation
    """
    if rotation_order is None:
        rotation_order = DEFAULT_ROTATION_ORDER
    offset = np.array(offset)
    if round(angles[0], 3) == 0 and round(angles[2], 3) == 0:
        rotation_q = quaternion_about_axis(np.deg2rad(angles[1]), [0, 1, 0])
    else:
        rotation_q = euler_to_quaternion(angles, rotation_order)
    rotation_matrix = euler_angles_to_rotation_matrix(angles, rotation_order)[:3, :3]
    for frame in quat_frames:
        ot = frame[:3]
        oq = frame[3:7]
        frame[:3] = np.dot(rotation_matrix, ot) + offset
        frame[3:7] = quaternion_multiply(rotation_q, oq)
    return quat_frames

def pose_orientation_quat(quaternion_frame):
    """Estimate pose orientation from root orientation
    """
    ref_offset = np.array([0, 0, 1, 1])
    rotmat = quaternion_matrix(quaternion_frame[3:7])
    rotated_point = np.dot(rotmat, ref_offset)
    dir_vec = np.array([rotated_point[0], rotated_point[2]])
    dir_vec /= np.linalg.norm(dir_vec)
    return dir_vec

def fast_quat_frames_transformation(quaternion_frames_a,
                                    quaternion_frames_b):
    dir_vec_a = pose_orientation_quat(quaternion_frames_a[-1])
    dir_vec_b = pose_orientation_quat(quaternion_frames_b[0])
    angle = get_rotation_angle(dir_vec_a, dir_vec_b)
    offset_x = quaternion_frames_a[-1][0] - quaternion_frames_b[0][0]
    offset_z = quaternion_frames_a[-1][2] - quaternion_frames_b[0][2]
    offset = [offset_x, 0.0, offset_z]
    return angle, offset



def get_orientation_vector_from_matrix(m, v=[0, 0, 1]):
    p = np.dot(m, v)
    dir_vec = np.array([p[0], p[2]])
    dir_vec /= np.linalg.norm(dir_vec)
    return dir_vec


def get_global_node_orientation_vector(skeleton, node_name, frame, v=[0, 0, 1]):
    v = np.array(v)
    m = skeleton.nodes[node_name].get_global_matrix(frame)[:3, :3]
    p = np.dot(m, v)
    dir_vec = np.array([p[0], p[2]])
    dir_vec /= np.linalg.norm(dir_vec)
    return dir_vec


REF_VECTOR = np.array([0,0,1])
def get_node_aligning_2d_transform(skeleton, node_name, prev_frames, new_frames, ref_vector=REF_VECTOR):
    """from last of prev frames to first of new frames"""

    m_a = skeleton.nodes[node_name].get_global_matrix(prev_frames[-1])
    m_b = skeleton.nodes[node_name].get_global_matrix(new_frames[0])
    dir_vec_a = get_orientation_vector_from_matrix(m_a[:3, :3], ref_vector)
    dir_vec_b = get_orientation_vector_from_matrix(m_b[:3, :3], ref_vector)
    angle = get_rotation_angle(dir_vec_a, dir_vec_b)
    q = quaternion_about_axis(np.deg2rad(angle), [0, 1, 0])
    m = quaternion_matrix(q)

    first_frame_pos = [new_frames[0][0], new_frames[0][1], new_frames[0][2],1.0]
    rotated_first_frame_pos = np.dot(m, first_frame_pos)[:3]
    delta = prev_frames[-1][:3] - rotated_first_frame_pos[:3]
    m[0, 3] = delta[0]
    #m[1, 3] = delta[1]
    m[2, 3] = delta[2]
    return m

def get_transform_from_point_cloud_alignment(skeleton, prev_frames, new_frames):
    weights = skeleton.get_joint_weights()
    p_a = convert_quat_frame_to_point_cloud(skeleton, prev_frames[-1])
    p_b = convert_quat_frame_to_point_cloud(skeleton, new_frames[0])
    theta, offset_x, offset_z = _align_point_clouds_2D(p_a, p_b, weights)
    euler = [0, np.radians(theta), 0]
    m = np.eye(4)
    m[:3,:3] = euler_matrix(*euler)[:3,:3]
    m[0,3] = offset_x
    m[2,3] = offset_z
    print("run point cloud alignment", theta, offset_x, offset_z, m)
    return m

def transform_quaternion_frames(frames, m,
                                translation_param_range=(0, 3),
                                orientation_param_range=(3, 7)):
    """ possibly broken because not 3,7 is the root orientation but 7,11
    """
    q = quaternion_from_matrix(m)
    for frame in frames:
        ot = frame[translation_param_range[0]:translation_param_range[1]].tolist() + [1]
        oq = frame[orientation_param_range[0]:orientation_param_range[1]]
        transformed_t = np.dot(m, ot)[:3]
        frame[translation_param_range[0]:translation_param_range[1]] = transformed_t
        frame[orientation_param_range[0]:orientation_param_range[1]] = quaternion_multiply(q, oq)
    return frames


def concatenate_frames_smoothly(new_frames, prev_frames, smoothing_window=0):
    d = len(prev_frames)
    frames = concatenate_frames(prev_frames, new_frames)
    if smoothing_window > 0:
        frames = smooth_quaternion_frames(frames, d, smoothing_window)
    return frames


def concatenate_frames_with_slerp(new_frames, prev_frames, smoothing_window=0):
    '''

    :param new_frames (numpy.array): n_frames * n_dims
    :param prev_frames (numpy.array): n_frames * n_dims
    :param smoothing_window:
    :return:
    '''
    d = len(prev_frames)
    frames = concatenate_frames(prev_frames, new_frames)
    if smoothing_window > 0:
        frames = smooth_quaternion_frames_with_slerp(frames, d, smoothing_window)
    return frames


def align_quaternion_frames_automatically(skeleton, node_name, new_frames, prev_frames, alignment_mode=ALIGNMENT_MODE_FAST):
    if alignment_mode == ALIGNMENT_MODE_FAST:
        m = get_node_aligning_2d_transform(skeleton, node_name, prev_frames, new_frames)
    else:
        m = get_transform_from_point_cloud_alignment(skeleton, prev_frames, new_frames)
    new_frames = transform_quaternion_frames(new_frames, m)
    return new_frames


def align_quaternion_frames_automatically2(skeleton, node_name, new_frames, prev_frames):
    angle, offset = fast_quat_frames_transformation(prev_frames, new_frames)
    new_frames = transform_quaternion_frames_legacy(new_frames, [0, angle, 0], offset)
    return new_frames


def align_quaternion_frames(skeleton, node_name, new_frames, prev_frames=None, start_pose=None):
    """align quaternion frames based on previous frames or
        given transformation

    Parameters
    ----------
    * new_frames: list
         A list of quaternion frames
    * prev_frames: list
        A list of quaternion frames
   *  transformation: dict
       Contains translation and orientation in Euler angles
    Returns:
    --------
    * transformed_frames: np.ndarray
        Quaternion frames resulting from the back projection of s,
        transformed to fit to prev_frames.

    """
    if prev_frames is not None:
        return align_quaternion_frames_automatically(skeleton, node_name, new_frames,  prev_frames)
    elif start_pose is not None:
        m = get_transform_from_start_pose(start_pose)
        first_frame_pos = new_frames[0][:3].tolist() + [1]
        t_pos = np.dot(m, first_frame_pos)[:3]
        delta = start_pose["position"]
        # FIXME this assumes the y translation is the up axis and can be ignored
        delta[0] -= t_pos[0]
        delta[2] -= t_pos[2]
        m[:3, 3] = delta
        transformed_frames = transform_quaternion_frames(new_frames, m)
        return transformed_frames
    else:
        return new_frames


def get_transform_from_start_pose(start_pose):
    e = np.deg2rad(start_pose["orientation"])
    p = start_pose["position"]
    if None not in e:
        q = quaternion_from_euler(*e)
        m = quaternion_matrix(q)
    else:
        m = np.eye(4)
    if None not in p:
        m[:3,3] = p
    return m


def align_and_concatenate_frames(skeleton, joint_name, new_frames, prev_frames=None, start_pose=None, smoothing_window=0,
                                 blending_method='smoothing'):
    new_frames = align_quaternion_frames(skeleton, joint_name, new_frames, prev_frames, start_pose)

    if prev_frames is not None:
        d = len(prev_frames)
        new_frames = concatenate_frames(prev_frames, new_frames)
        if smoothing_window > 0:
            if blending_method == 'smoothing':
                new_frames = smooth_quaternion_frames(new_frames, d, smoothing_window)
            elif blending_method == 'slerp':
                new_frames = smooth_quaternion_frames_with_slerp(new_frames, d, smoothing_window)
            elif blending_method == 'slerp2':
                new_frames = smooth_quaternion_frames_with_slerp2(skeleton, new_frames, d, smoothing_window)
            elif blending_method == 'linear':
                new_frames = blend_quaternion_frames_linearly(new_frames, prev_frames, skeleton, smoothing_window)
            else:
                raise KeyError('Unknown method!')
    return new_frames


blend = lambda x: 2 * x * x * x - 3 * x * x + 1


def align_joint(skeleton, frames, frame_idx, foot_joint, ik_chain, ik_window=7):
    transition_start = frame_idx + 1
    transition_end = frame_idx + ik_window
    c = create_grounding_constraint_from_frame(skeleton, frames, frame_idx, foot_joint)
    ik = AnalyticalLimbIK.init_from_dict(skeleton, c.joint_name, ik_chain)
    frames[transition_start] = ik.apply2(frames[transition_start], c.position, c.orientation)

    chain_joints = [ik_chain["root"], ik_chain["joint"], foot_joint]
    for c_joint in chain_joints:
        idx = skeleton.animated_joints.index(c_joint) * 4 + 3
        j_indices = [idx, idx + 1, idx + 2, idx + 3]
        start_q = frames[transition_start][j_indices]
        end_q = frames[transition_end][j_indices]
        for i in range(ik_window):
            t = float(i) / ik_window
            frames[transition_start + 1 + i][j_indices] = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
    return frames


def align_frames_and_fix_foot_to_prev(skeleton, aligning_joint, new_frames, prev_frames, start_pose, foot_joint, ik_chain, ik_window=7, smoothing_window=0):
    new_frames = align_quaternion_frames(skeleton, aligning_joint, new_frames, prev_frames, start_pose)
    if prev_frames is not None:
        offset = prev_frames[-1][:3] - new_frames[0][:3]

        d = len(prev_frames)
        frames = prev_frames.tolist()
        for idx in range(1, len(new_frames)):  # skip first frame
            frames.append(new_frames[idx])
        frames = np.array(frames)

        transition_start = d
        c = create_grounding_constraint_from_frame(skeleton, frames, d-1, foot_joint)
        ik = AnalyticalLimbIK.init_from_dict(skeleton, c.joint_name, ik_chain)
        before = skeleton.nodes[foot_joint].get_global_position(frames[transition_start])

        frames[transition_start] = ik.apply2(frames[transition_start], c.position, c.orientation)

        transition_end = d+ik_window
        print("allign frames", c.position, foot_joint, d-1, transition_end, before, skeleton.nodes[foot_joint].get_global_position(frames[transition_start]))
        print(skeleton.nodes[foot_joint].get_global_position(frames[d]))

        chain_joints =  [ik_chain["root"], ik_chain["joint"], foot_joint]
        for c_joint in chain_joints:
            idx = skeleton.animated_joints.index(c_joint) * 4 + 3
            j_indices = [idx, idx + 1, idx + 2, idx + 3]
            start_q = frames[transition_start][j_indices]
            end_q = frames[transition_end][j_indices]
            print(c_joint, start_q, end_q,j_indices)
            for i in range(ik_window):
                t = float(i) / ik_window
                # nlerp_q = self.nlerp(start_q, end_q, t)
                slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
                print(transition_start+i+1, frames[transition_start + 1 + i][j_indices], slerp_q)
                frames[transition_start + 1 + i][j_indices] = slerp_q
        if smoothing_window > 0 and False:
            frames = smooth_quaternion_frames(frames, d, smoothing_window)
        return frames
    else:
        return new_frames

def get_limb_length(skeleton, joint_name):
    limb_length = np.linalg.norm(skeleton.nodes[joint_name].offset)
    limb_length += np.linalg.norm(skeleton.nodes[joint_name].parent.offset)
    return limb_length

def generate_root_constraint_for_one_foot(skeleton, frame, root, c):
        root_pos = skeleton.nodes[root].get_global_position(frame)
        target_length = np.linalg.norm(c.position - root_pos)
        limb_length = get_limb_length(skeleton, c.joint_name)
        if target_length >= limb_length:
            new_root_pos = (c.position + normalize(root_pos - c.position) * limb_length)
            print("one constraint on ", c.joint_name, "- before", root_pos, "after", new_root_pos)
            return new_root_pos
            #frame[:3] = new_root_pos

        else:
            print("no change")


def translate_root(skeleton, frames, target_frame_idx, plant_heel, ground_height=0):
    """ translate the next frames closer to the previous frames root translation"""
    n_frames = len(frames)
    foot_pos = skeleton.nodes[plant_heel].get_global_position(frames[target_frame_idx-1])
    print("foot pos before", foot_pos)
    delta = ground_height - foot_pos[1]
    n_frames = len(frames)
    for f in range(target_frame_idx, n_frames):
        frames[f][1] += delta
    print("after", skeleton.nodes[plant_heel].get_global_position(frames[target_frame_idx]))
    for f in range(target_frame_idx, n_frames):
        frames[f,:3] += delta/2


def apply_constraint(skeleton, frames, c, ik_chain, frame_idx, start, end, window):
    print("apply swing foot constraint on frame", frame_idx, start, end)
    ik = AnalyticalLimbIK.init_from_dict(skeleton, c.joint_name, ik_chain)
    frames[frame_idx] = ik.apply2(frames[frame_idx], c.position, c.orientation)
    joint_list = [ik_chain["root"], ik_chain["joint"], c.joint_name]
    blend_between_frames(skeleton, frames, start, end, joint_list, window)


def apply_constraint_on_window_prev(skeleton, frames, c, ik_chain, start, end, window):
    ik = AnalyticalLimbIK.init_from_dict(skeleton, c.joint_name, ik_chain)
    indices = list(range(start, end + 1))
    print("apply on frames", indices)
    for f in indices:
        frames[f] = ik.apply2(frames[f], c.position, c.orientation)
    joint_list = [ik_chain["root"], ik_chain["joint"], c.joint_name]
    blend_between_frames(skeleton, frames, end, end+window, joint_list, window)


def apply_constraint_on_window_next(skeleton, frames, c, ik_chain, start, end, window):
    ik = AnalyticalLimbIK.init_from_dict(skeleton, c.joint_name, ik_chain)
    indices = list(range(start, end + 1))
    print("apply on frames", indices)
    for f in indices:
        frames[f] = ik.apply2(frames[f], c.position, c.orientation)
    joint_list = [ik_chain["root"], ik_chain["joint"], c.joint_name]
    print("blend between frames",start-window, start)
    blend_between_frames(skeleton, frames, start-window, start, joint_list, window)


def align_foot_to_next_step(skeleton, frames, foot_joint, ik_chain, target_frame_idx, window):
    start = target_frame_idx - window  # start of blending range
    end = target_frame_idx - 1 # modified frame

    c = create_grounding_constraint_from_frame(skeleton, frames, target_frame_idx, foot_joint)
    apply_constraint(skeleton, frames, c, ik_chain, target_frame_idx, start, end, window)


def align_foot_to_prev_step(skeleton, frames, foot_joint, ik_chain, target_frame_idx, window):
    start = target_frame_idx # modified frame
    end = target_frame_idx + window  # end of blending range
    c = create_grounding_constraint_from_frame(skeleton, frames, target_frame_idx-1, foot_joint)
    apply_constraint(skeleton, frames, c, ik_chain, target_frame_idx, start, end, window)




def generate_feet_constraints(skeleton, frames, frame_idx, plant_side, swing_side, target_ground_height):
    plant_foot_joint = skeleton.skeleton_model["joints"][plant_side + "_ankle"]
    plant_toe_joint = skeleton.skeleton_model["joints"][plant_side + "_toe"]
    plant_heel_joint = skeleton.skeleton_model["joints"][plant_side + "_heel"]
    swing_foot_joint = skeleton.skeleton_model["joints"][swing_side + "_ankle"]
    swing_toe_joint = skeleton.skeleton_model["joints"][swing_side + "_toe"]
    swing_heel_joint = skeleton.skeleton_model["joints"][swing_side + "_heel"]
    plant_constraint = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, plant_foot_joint,
                                                          plant_heel_joint, plant_toe_joint, target_ground_height)
    swing_constraint = generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, swing_foot_joint,
                                                          swing_heel_joint, swing_toe_joint, target_ground_height)
    return plant_constraint, swing_constraint


def generate_feet_constraints2(skeleton, frames, frame_idx, plant_side, swing_side):
    plant_foot_joint = skeleton.skeleton_model["joints"][plant_side + "_ankle"]
    swing_foot_joint = skeleton.skeleton_model["joints"][swing_side + "_ankle"]
    plant_constraint = create_grounding_constraint_from_frame(skeleton, frames, frame_idx - 1, plant_foot_joint)
    swing_constraint = create_grounding_constraint_from_frame(skeleton, frames, frame_idx - 1, swing_foot_joint)
    return plant_constraint, swing_constraint


def align_feet_to_prev_step(skeleton, frames, frame_idx, plant_constraint, swing_constraint, ik_chains, window):
    start = frame_idx  # modified frame
    end = frame_idx + window  # end of blending range
    apply_constraint_on_window_prev(skeleton, frames, plant_constraint, ik_chains[plant_constraint.joint_name], start, end, window)
    apply_constraint(skeleton, frames, swing_constraint, ik_chains[swing_constraint.joint_name], frame_idx, start, end, window)


def align_feet_to_next_step(skeleton, frames, frame_idx, plant_constraint, swing_constraint, ik_chains, plant_window, window):
    start = frame_idx - window  # end of blending range
    end = frame_idx  # modified frame
    apply_constraint_on_window_next(skeleton, frames, plant_constraint, ik_chains[plant_constraint.joint_name], start, end, plant_window)
    apply_constraint(skeleton, frames, swing_constraint, ik_chains[swing_constraint.joint_name], frame_idx, start, end, window)


def align_feet_to_next_step2(skeleton, frames, frame_idx, plant_constraint, swing_constraint, ik_chains, plant_window, window):
    start = frame_idx - window  # end of blending range
    end = frame_idx  # modified frame
    ik_chain = ik_chains[plant_constraint.joint_name]
    joint_list = [ik_chain["root"], ik_chain["joint"], plant_constraint.joint_name]
    blend_between_frames(skeleton, frames, start, end, joint_list, window)
    ik_chain = ik_chains[swing_constraint.joint_name]
    joint_list = [ik_chain["root"], ik_chain["joint"], plant_constraint.joint_name]
    blend_between_frames(skeleton, frames, start, end, joint_list, window)


def fix_feet_at_transition(skeleton, frames, d,  plant_side, swing_side, ik_chains, ik_window=8, plant_window=20):
    target_ground_height = 0
    smooth_root_translation_around_transition(frames, d, 2 * ik_window)

    plant_constraint, swing_constraint = generate_feet_constraints(skeleton, frames, d, plant_side, swing_side, target_ground_height)
    root_pos = generate_root_constraint_for_two_feet(skeleton, frames[d-1], plant_constraint, swing_constraint)
    if root_pos is not None:
        frames[d - 1][:3] = root_pos
        smooth_root_translation_at_end(frames, d - 1, ik_window)
        smooth_root_translation_at_start(frames, d, ik_window)

    align_feet_to_next_step(skeleton, frames, d-1, plant_constraint, swing_constraint, ik_chains, plant_window, ik_window)
    align_feet_to_prev_step(skeleton, frames, d, plant_constraint, swing_constraint, ik_chains, ik_window)
    #swing_foot = skeleton.skeleton_model["joints"][swing_side + "_ankle"]
    #align_foot_to_prev_step(skeleton, frames, swing_foot, ik_chains[swing_foot], d, ik_window)


def align_frames_using_forward_blending(skeleton, aligning_joint, new_frames, prev_frames, prev_start, start_pose, ik_chains, smoothing_window=0):
    """ applies foot ik constraint to fit the prev motion primitive to the next motion primitive
    """

    new_frames = align_quaternion_frames(skeleton, aligning_joint, new_frames, prev_frames, start_pose)
    if prev_frames is not None:
        d = len(prev_frames)
        frames = prev_frames.tolist()
        for idx in range(1, len(new_frames)):  # skip first frame
            frames.append(new_frames[idx])
        frames = np.array(frames)

        blend_end = d
        blend_start = int(prev_start + (blend_end-prev_start)/2)  # start blending from the middle of the step

        left_joint = skeleton.skeleton_model["joints"]["left_ankle"]
        right_joint = skeleton.skeleton_model["joints"]["right_ankle"]
        pelvis = skeleton.skeleton_model["joints"]["pelvis"]
        left_ik_chain = ik_chains[left_joint]
        right_ik_chain = ik_chains[right_joint]
        leg_joint_list = [skeleton.root, left_ik_chain["root"], left_ik_chain["joint"], left_joint,
                      right_ik_chain["root"], right_ik_chain["joint"], right_joint]
        if pelvis != skeleton.root:
            leg_joint_list.append(pelvis)

        frames = blend_towards_next_step_linear_with_original(skeleton, frames, blend_start, blend_end, leg_joint_list)
        joint_list = [j for j in skeleton.animated_joints if j not in leg_joint_list]

        if smoothing_window > 0:
            frames = smooth_quaternion_frames_joint_filter(skeleton, frames, d, joint_list, smoothing_window)
        return frames
    else:
        return new_frames

def blend_towards_next_step_linear(skeleton, frames, d,  plant_side, swing_side, ik_chains, window=8):
    target_ground_height = 0
    smooth_root_translation_around_transition(frames, d, 2 * window)
    plant_constraint, swing_constraint = generate_feet_constraints(skeleton, frames, d, plant_side, swing_side, target_ground_height)
    root_pos = generate_root_constraint_for_two_feet(skeleton, frames[d-1], plant_constraint, swing_constraint)
    if root_pos is not None:
        frames[d - 1][:3] = root_pos
        smooth_root_translation_at_end(frames, d - 1, window)
        smooth_root_translation_at_start(frames, d, window)
    blend_quaternions_to_next_step(skeleton, frames, d, plant_constraint.joint_name, swing_constraint.joint_name, ik_chains, window)



def blend_towards_next_step3(skeleton, frames, start, end, plant_side, swing_side, ik_chains, window=8):
    plant_joint = skeleton.skeleton_model["joints"][plant_side + "_ankle"]
    swing_joint = skeleton.skeleton_model["joints"][swing_side + "_ankle"]
    plant_ik_chain = ik_chains[plant_joint]
    swing_ik_chain = ik_chains[swing_joint]
    joint_list = [skeleton.root, "pelvis", plant_ik_chain["root"], plant_ik_chain["joint"], plant_joint,
                  swing_ik_chain["root"], swing_ik_chain["joint"], swing_joint]
    middle = int(start + (end-start)/2)
    new_frames = generate_blended_frames(skeleton, frames, middle, end, joint_list, end-middle)
    frames = interpolate_frames(skeleton, frames, new_frames, joint_list, middle, end)
    return frames



def transform_euler_frame(euler_frame, angles, offset, rotation_order=None, global_rotation=True):
    """
    Calls transform_point for the root parameters and adds theta to the y rotation
    channel of the frame.

    The offset of root is transformed by transform_point
    The orientation of root is rotated by Rotation matrix

    Parameters
    ---------
    *euler_frame: np.ndarray
    \t the parameters of a single frame
    *angles: list of floats
    \tRotation angles in degrees
    *offset: np.ndarray
    \tTranslation
    """
    if rotation_order is None:
        rotation_order = DEFAULT_ROTATION_ORDER
    transformed_frame = deepcopy(euler_frame)
    if global_rotation:
        transformed_frame[:3] = transform_point(euler_frame[:3], angles, offset, rotation_order=rotation_order)
    else:
        transformed_frame[:3] = transform_point(euler_frame[:3], np.zeros(3), offset, rotation_order=rotation_order)
    R = euler_matrix(np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2]), axes='rxyz')
    
    src_e = np.deg2rad(euler_frame[3:6])
    rot_string = rotation_order_to_string(rotation_order)
    OR = euler_matrix(*src_e, axes=rot_string)
    rotmat = np.dot(R, OR)
    eul_angles = np.rad2deg(euler_from_matrix(rotmat, rot_string))
    transformed_frame[3:6] = eul_angles
    return transformed_frame


def transform_euler_frames(euler_frames, angles, offset, rotation_order=None):
    """ Applies a transformation on the root joint of a list euler frames.
    Parameters
    ----------
    *euler_frames: np.ndarray
    \tList of frames where the rotation is represented as euler angles in degrees.
    *angles: list of floats
    \tRotation angles in degrees
    *offset:  np.ndarray
    \tTranslation
    """
    transformed_euler_frames = []
    for frame in euler_frames:
        transformed_euler_frames.append(
            transform_euler_frame(frame, angles, offset, rotation_order))
    return np.array(transformed_euler_frames)


def shift_euler_frames_to_ground(euler_frames, ground_contact_joints, skeleton, align_index=0):
    """
    shift all euler frames of motion to ground, which means the y-axis for
    gound contact joint should be 0
    Step 1: apply forward kinematic to compute global position for ground
            contact joint for each frame
    Setp 2: find the offset from ground contact joint to ground, and shift
            corresponding frame based on offset
    """
    foot_contact_heights = []
    for joint in ground_contact_joints:
        foot_contact_heights.append(skeleton.nodes[joint].get_global_position_from_euler(euler_frames[align_index])[1])
    avg_foot_height =np.array([0, -np.average(foot_contact_heights), 0])
    return transform_euler_frames(euler_frames, [0.0, 0.0, 0.0], avg_foot_height)


def shift_quat_frames_to_ground(quat_frames, ground_contact_joints, skeleton, align_index=0):
    foot_contact_heights = []
    for joint in ground_contact_joints:
        foot_contact_heights.append(skeleton.nodes[joint].get_global_position(quat_frames[align_index])[1])
    return transform_quaternion_frames(quat_frames,
                                       [0.0, 0.0, 0.0],
                                       np.array([0, -np.average(foot_contact_heights), 0]))



def find_aligning_transformation(skeleton, euler_frames_a, euler_frames_b):
    """
    performs alignment of the point clouds based on the poses at the end of
    euler_frames_a and the start of euler_frames_b
    Returns the rotation around y axis in radians, x offset and z offset
    """
    point_cloud_a = convert_euler_frame_to_cartesian_frame(skeleton, euler_frames_a[-1])
    point_cloud_b = convert_euler_frame_to_cartesian_frame(skeleton, euler_frames_b[0])
    weights = skeleton.get_joint_weights()
    theta, offset_x, offset_z = _align_point_clouds_2D(point_cloud_a, point_cloud_b, weights)
    return theta, offset_x, offset_z


def align_frames(skeleton, euler_frames_a, euler_frames_b, smooth=True):
    """
    calls find_aligning_transformation and concatenates the frames based on the
    resulting transformation
     Parameters
    ----------
    *skeleton: Skeleton
    \tUsed to extract hierarchy information.
    *euler_frames_a: np.ndarray
    \List of frames where the rotation is represented as euler angles in degrees.
    *euler_frames_b: np.ndarray
    \List of frames where the rotation is represented as euler angles in degrees.
    *smooth: bool
    \t Sets whether or not smoothing is supposed to be applied on the at the transition.
     Returns
    -------
    *aligned_frames : np.ndarray
    \tAligned and optionally smoothed motion
    """
    theta, offset_x, offset_z = find_aligning_transformation(skeleton, euler_frames_a, euler_frames_b)

    # apply 2d transformation
    offset = np.array([offset_x, 0, offset_z])
    angles = [0, np.degrees(theta), 0]
    euler_frames_b = transform_euler_frames(euler_frames_b, angles,
                                            offset)

    # concatenate frames and optionally apply smoothing
    if smooth:
        euler_frames = smoothly_concatenate(euler_frames_a, euler_frames_b)
    else:
        euler_frames = np.concatenate((euler_frames_a, euler_frames_b), axis=0)
    return euler_frames


