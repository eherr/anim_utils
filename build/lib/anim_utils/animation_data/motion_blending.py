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
from transformations import quaternion_slerp
import numpy as np
from copy import deepcopy
from .constants import LEN_QUAT, LEN_ROOT_POS


BLEND_DIRECTION_FORWARD = 0
BLEND_DIRECTION_BACKWARD = 1


def smooth_root_positions(positions, window):
    h_window = int(window/2)
    smoothed_positions = []
    n_pos = len(positions)
    for idx, p in enumerate(positions):
        start = max(idx-h_window, 0)
        end = min(idx + h_window, n_pos)
        #print start, end, positions[start:end]
        avg_p = np.average(positions[start:end], axis=0)
        smoothed_positions.append(avg_p)
    return smoothed_positions


def blend_quaternion(a, b, w):
    return quaternion_slerp(a, b, w, spin=0, shortestpath=True)


def smooth_joints_around_transition_using_slerp(quat_frames, joint_param_indices, discontinuity, window):
    h_window = int(window/2)
    start_frame = max(discontinuity-h_window, 0)
    end_frame = min(discontinuity+h_window, len(quat_frames)-1)
    start_window = discontinuity-start_frame
    end_window = end_frame-discontinuity
    if start_window > 0:
        create_transition_for_joints_using_slerp(quat_frames, joint_param_indices, start_frame, discontinuity, start_window, BLEND_DIRECTION_FORWARD)
    if end_window > 0:
        create_transition_for_joints_using_slerp(quat_frames, joint_param_indices, discontinuity, end_frame, end_window, BLEND_DIRECTION_BACKWARD)



def create_transition_for_joints_using_slerp(quat_frames, joint_param_indices, start_frame, end_frame, steps, direction=BLEND_DIRECTION_FORWARD):
    new_quats = create_frames_using_slerp(quat_frames, start_frame, end_frame, steps, joint_param_indices)
    for i in range(steps):
        if direction == BLEND_DIRECTION_FORWARD:
            t = float(i)/steps
        else:
            t = 1.0 - (i / steps)
        old_quat = quat_frames[start_frame+i, joint_param_indices]
        blended_quat = blend_quaternion(old_quat, new_quats[i], t)
        quat_frames[start_frame + i, joint_param_indices] = blended_quat
    return quat_frames


def smooth_quaternion_frames_using_slerp_(quat_frames, joint_parameter_indices, event_frame, window):
    start_frame = event_frame-window/2
    end_frame = event_frame+window/2
    start_q = quat_frames[start_frame, joint_parameter_indices]
    end_q = quat_frames[end_frame, joint_parameter_indices]
    for i in range(window):
        t = float(i)/window
        #nlerp_q = self.nlerp(start_q, end_q, t)
        slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        #print "slerp",start_q,  end_q, t, nlerp_q, slerp_q
        quat_frames[start_frame+i, joint_parameter_indices] = slerp_q


def smooth_quaternion_frames_using_slerp_old(quat_frames, joint_param_indices, event_frame, window):
    h_window = window/2
    start_frame = max(event_frame-h_window, 0)
    end_frame = min(event_frame+h_window, quat_frames.shape[0]-1)
    # create transition frames
    from_start_to_event = create_frames_using_slerp(quat_frames, start_frame, event_frame, h_window, joint_param_indices)
    from_event_to_end = create_frames_using_slerp(quat_frames, event_frame, end_frame, h_window, joint_param_indices)

    #blend transition frames with original frames
    steps = event_frame-start_frame
    for i in range(steps):
        t = float(i)/steps
        quat_frames[start_frame+i, joint_param_indices] = blend_quaternion(quat_frames[start_frame+i, joint_param_indices], from_start_to_event[i], t)

    steps = end_frame-event_frame
    for i in range(steps):
        t = 1.0-(i/steps)
        quat_frames[event_frame+i, joint_param_indices] = blend_quaternion(quat_frames[start_frame+i, joint_param_indices], from_event_to_end[i], t)


def smooth_quaternion_frames_using_slerp_overwrite_frames(quat_frames, joint_param_indices, event_frame, window):
    h_window = window/2
    start_frame = max(event_frame-h_window, 0)
    end_frame = min(event_frame+h_window, quat_frames.shape[0]-1)
    create_transition_using_slerp(quat_frames, start_frame, event_frame, joint_param_indices)
    create_transition_using_slerp(quat_frames, event_frame, end_frame, joint_param_indices)


def blend_frames(quat_frames, start, end, new_frames, joint_parameter_indices):
    steps = end-start
    for i in range(steps):
        t = i/steps
        quat_frames[start+i, joint_parameter_indices] = blend_quaternion(quat_frames[start+i, joint_parameter_indices], new_frames[i], t)


def create_frames_using_slerp(quat_frames, start_frame, end_frame, steps, joint_parameter_indices):
    start_q = quat_frames[start_frame, joint_parameter_indices]
    end_q = quat_frames[end_frame, joint_parameter_indices]
    frames = []
    for i in range(steps):
        t = float(i)/steps
        slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        frames.append(slerp_q)
    return frames


def create_transition_using_slerp(quat_frames, start_frame, end_frame, q_indices):
    start_q = quat_frames[start_frame, q_indices]
    end_q = quat_frames[end_frame, q_indices]
    steps = end_frame-start_frame
    for i in range(steps):
        t = float(i)/steps
        slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        quat_frames[start_frame+i, q_indices] = slerp_q

def create_transition_using_slerp_forward(quat_frames, start_frame, end_frame, q_indices):
    end_q = quat_frames[end_frame, q_indices]
    steps = end_frame - start_frame
    for i in range(steps):
        t = float(i) / steps
        start_q = quat_frames[i, q_indices]
        slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        quat_frames[start_frame + i, q_indices] = slerp_q


def create_transition_using_slerp_backward(quat_frames, start_frame, end_frame, q_indices):
    start_q = quat_frames[start_frame, q_indices]
    steps = end_frame - start_frame
    for i in range(steps):
        t = float(i) / steps
        end_q = quat_frames[i, q_indices]
        slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        quat_frames[start_frame + i, q_indices] = slerp_q

def smooth_quaternion_frames(frames, discontinuity, window=20, include_root=True):
    """ Smooth quaternion frames given discontinuity frame

    Parameters
    ----------
    frames: list
    \tA list of quaternion frames
    discontinuity : int
    The frame where the discontinuity is. (e.g. the transitionframe)
    window : (optional) int, default is 20
    The smoothing window
    include_root:  (optional) bool, default is False
    \tSets whether or not smoothing is applied on the x and z dimensions of the root translation
    Returns
    -------
    None.
    """
    n_joints = int((len(frames[0]) - 3) / 4)
    # smooth quaternion
    n_frames = len(frames)
    for i in range(n_joints):
        for j in range(n_frames - 1):
            start = 3 + i * 4
            end = 3 + (i + 1) * 4
            q1 = np.array(frames[j][start: end])
            q2 = np.array(frames[j + 1][start:end])
            if np.dot(q1, q2) < 0:
                frames[j + 1][start:end] = -frames[j + 1][start:end]

    smoothing_factors = generate_smoothing_factors(discontinuity, window, n_frames)
    d = int(discontinuity)
    dofs = list(range(len(frames[0])))[3:]
    if include_root:
        dofs = [0,1,2] + dofs
    else:
        dofs = [1] + dofs
    new_frames = np.array(frames)
    for dof_idx in dofs:
        curve = np.array(frames[:, dof_idx])  # extract dof curve
        magnitude = curve[d] - curve[d - 1]
        #print(dof_idx, magnitude, d)
        new_frames[:, dof_idx] = curve + (magnitude * smoothing_factors)
    return new_frames



def slerp_quaternion_frame(frame_a, frame_b, weight):
    frame_a = np.asarray(frame_a)
    frame_b = np.asarray(frame_b)
    assert len(frame_a) == len(frame_b)
    n_joints = int((len(frame_a) - 3) / 4)
    new_frame = np.zeros(len(frame_a))
    # linear interpolate root translation
    new_frame[:3] = (1 - weight) * frame_a[:3] + weight * frame_b[:3]
    for i in range(n_joints):
        new_frame[3+ i*4 : 3 + (i+1) * 4] = quaternion_slerp(frame_a[3 + i*4 : 3 + (i+1) * 4],
                                                             frame_b[3 + i*4 : 3 + (i+1) * 4],
                                                             weight)
    return new_frame


def smooth_quaternion_frames_with_slerp(frames, discontinuity, window=20):
    n_frames = len(frames)
    d = float(discontinuity)
    ref_pose = slerp_quaternion_frame(frames[int(d)-1], frames[int(d)], 0.5)
    w = float(window)
    new_quaternion_frames = []
    for f in range(n_frames):
        if f < d - w:
            new_quaternion_frames.append(frames[f])
        elif d - w <= f < d:
            tmp = (f - d + w) / w
            weight = 2 * (0.5 * tmp ** 2)
            new_quaternion_frames.append(slerp_quaternion_frame(frames[f], ref_pose, weight))
        elif d <= f <= d + w:
            tmp = (f - d + w) / w
            weight = 2 * (0.5 * tmp ** 2 - 2 * tmp + 2)
            new_quaternion_frames.append(slerp_quaternion_frame(frames[f], ref_pose, weight))
        else:
            new_quaternion_frames.append(frames[f])
    return np.asarray(new_quaternion_frames)


def smooth_quaternion_frames_with_slerp2(skeleton, frames, d, smoothing_window):
    '''

    :param new_frames (numpy.array): n_frames * n_dims
    :param prev_frames (numpy.array): n_frames * n_dims
    :param smoothing_window:
    :return:
    '''
    smooth_translation_in_quat_frames(frames, d, smoothing_window)
    for joint_idx, joint_name in enumerate(skeleton.animated_joints):
        start = joint_idx*4+3
        joint_indices = list(range(start, start+4))
        smooth_joints_around_transition_using_slerp(frames, joint_indices, d, smoothing_window)
    return frames


def generate_smoothing_factors(discontinuity, window, n_frames):
    """ Generate curve of smoothing factors
    """
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
    return np.array(smoothing_factors)


def smooth_quaternion_frames_joint_filter(skeleton, frames, discontinuity, joints, window=20):
    """ Smooth quaternion frames given discontinuity frame

    Parameters
    ----------
    skeleton: Skeleton
    frames: list
    \tA list of quaternion frames
    discontinuity : int
    The frame where the discontinuity is. (e.g. the transitionframe)
    joints: list
    \tA list of strings
    window : (optional) int, default is 20
    The smoothing window
    Returns
    -------
    None.
    """

    n_frames = len(frames)
    # generate curve of smoothing factors
    smoothing_factors = generate_smoothing_factors(discontinuity, window, n_frames)

    # align quaternions and extract dofs
    dof_filter_list = []
    if skeleton.root in joints:
        dof_filter_list += [0,1,2]
    for idx, j in enumerate(joints):
        j_idx = skeleton.animated_joints.index(j)
        q_start_idx = 3 + j_idx * 4
        q_end_idx = q_start_idx + 4
        dof_filter_list += [q_start_idx, q_start_idx + 1, q_start_idx + 2, q_start_idx + 3]
        for f in range(n_frames - 1):
            q1 = np.array(frames[f][q_start_idx: q_end_idx])
            q2 = np.array(frames[f + 1][q_start_idx:q_end_idx])
            if np.dot(q1, q2) < 0:
                frames[f + 1][q_start_idx:q_end_idx] = -q2
    d = int(discontinuity)
    new_frames = np.array(frames)
    for dof_idx in dof_filter_list:
        current_curve = np.array(frames[:, dof_idx])  # extract dof curve
        magnitude = current_curve[d] - current_curve[d - 1]
        new_curve = current_curve + (magnitude * smoothing_factors)
        new_frames[:, dof_idx] = new_curve
    return new_frames

def smooth_quaternion_frames2(skeleton, frames, discontinuity, window=20):
    """ Smooth quaternion frames given discontinuity frame

    Parameters
    ----------
    skeleton: Skeleton
    frames: list
    \tA list of quaternion frames
    discontinuity : int
    The frame where the discontinuity is. (e.g. the transitionframe)
    window : (optional) int, default is 20
    The smoothing window
    Returns
    -------
    None.
    """

    n_frames = len(frames)
    smoothing_factors = generate_smoothing_factors(discontinuity, window, n_frames)
    d = int(discontinuity)
    new_frames = np.array(frames)
    for idx in range(len(skeleton.animated_joints)):
        o = idx *4 + 3
        dofs = [o, o+1, o+2, o+3]
        q1 = frames[d, dofs]
        q2 = frames[d+1, dofs]
        if np.dot(q1, q2) < 0:
            q2 = -q2
        magnitude = q1 - q2
        new_frames[:, dofs] += magnitude * smoothing_factors
    return new_frames

def smooth_translation_in_quat_frames(frames, discontinuity, window=20, only_height=False):
    """ Smooth translation in quaternion frames given discontinuity frame

    Parameters
    ----------
    frames: list
    \tA list of quaternion frames
    discontinuity : int
    The frame where the discontinuity is. (e.g. the transitionframe)
    window : (optional) int, default is 20
    The smoothing window
    Returns
    -------
    None.
    """

    n_frames = len(frames)
    # generate curve of smoothing factors
    d = int(discontinuity)
    smoothing_factors = generate_smoothing_factors(d, window, n_frames)
    if only_height:
        dofs = [1]
    else:
        dofs = [0, 1, 2]
    for dof_idx in dofs:
        current_curve = np.array(frames[:, dof_idx])  # extract dof curve
        magnitude = current_curve[d] - current_curve[d - 1]
        new_curve = current_curve + (magnitude * smoothing_factors)
        frames[:, dof_idx] = new_curve
    return frames


def smooth_root_translation_around_transition(frames, d, window):
    hwindow = int(window/2.0)
    root_pos1 = frames[d-1, :3]
    root_pos2 = frames[d, :3]
    root_pos = (root_pos1 + root_pos2)/2
    start_idx = d-hwindow
    end_idx = d + hwindow
    start = frames[start_idx, :3]
    end = root_pos
    for i in range(hwindow):
        t = float(i) / hwindow
        frames[start_idx + i, :3] = start * (1 - t) + end * t
    start = root_pos
    end = frames[end_idx, :3]
    for i in range(hwindow):
        t = float(i) / hwindow
        frames[d + i, :3] = start * (1 - t) + end * t


def linear_blending(ref_pose, quat_frames, skeleton, weights, joint_list=None):
    '''
    Apply linear blending on quaternion motion data
    :param ref_pose (numpy.array)
    :param quat_frames:
    :param skeleton (morphablegraphs.animation_data.Skeleton):
    :param weights (numpy.array): weights used for slerp
    :param joint_list (list): animated joint to be blended
    :return:
    '''
    if joint_list is None:
        joint_list = skeleton.animated_joints
    new_frames = deepcopy(quat_frames)
    for i in range(len(quat_frames)):
        for joint in joint_list:
            joint_index = skeleton.nodes[joint].quaternion_frame_index
            start_index = LEN_ROOT_POS + LEN_QUAT * joint_index
            end_index = LEN_ROOT_POS + LEN_QUAT * (joint_index + 1)
            ref_q = ref_pose[start_index: end_index]
            motion_q = quat_frames[i, start_index: end_index]
            new_frames[i, start_index: end_index] = quaternion_slerp(ref_q, motion_q, weights[i])
    return new_frames


def blend_quaternion_frames_linearly(new_frames, prev_frames, skeleton, smoothing_window=None):
    '''
    Blend new frames linearly based on the last pose of previous frames
    :param new_frames (Quaternion Frames):
    :param prev_frames (Quaternion Frames):
    :param skeleton (morphablegraphs.animation_data.Skeleton):
    :param smoothing_window (int): smoothing window decides how many frames will be blended, if is None, then blend all
    :return:
    '''
    if smoothing_window is not None and smoothing_window != 0:
        slerp_weights = np.linspace(0, 1, smoothing_window)
    else:
        slerp_weights = np.linspace(0, 1, len(new_frames))

    return linear_blending(prev_frames[-1], new_frames, skeleton, slerp_weights)


def blend_between_frames(skeleton, frames, transition_start, transition_end, joint_list, window):
    for c_joint in joint_list:
        idx = skeleton.animated_joints.index(c_joint) * 4 + 3
        j_indices = [idx, idx + 1, idx + 2, idx + 3]
        start_q = frames[transition_start][j_indices]
        end_q = frames[transition_end][j_indices]
        for i in range(window):
            t = float(i) / window
            t = (float(t) / window)
            slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
            frames[transition_start + i][j_indices] = slerp_q


def generated_blend(start_q, end_q, window):
    blend = np.zeros((window, 4))
    for i in range(window):
        t = float(i) / window
        slerp_q = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        blend[i] = slerp_q
    return blend


def generate_blended_frames(skeleton, frames, start, end, joint_list, window):
    blended_frames = deepcopy(frames[:])
    for c_joint in joint_list:
        idx = skeleton.animated_joints.index(c_joint) * 4 + 3
        j_indices = [idx, idx + 1, idx + 2, idx + 3]
        start_q = frames[start][j_indices]
        end_q = frames[end][j_indices]
        blended_qs = generated_blend(start_q, end_q, window)
        for fi, q in enumerate(blended_qs):
            blended_frames[start+fi][j_indices] = q
    return blended_frames


def blend_quaternions_to_next_step(skeleton, frames, frame_idx, plant_joint, swing_joint, ik_chains,  window):
    start = frame_idx - window  # end of blending range
    end = frame_idx  # modified frame
    plant_ik_chain = ik_chains[plant_joint]
    swing_ik_chain = ik_chains[swing_joint]
    joint_list = [skeleton.root, "pelvis", plant_ik_chain["root"], plant_ik_chain["joint"], plant_joint, swing_ik_chain["root"], swing_ik_chain["joint"], swing_joint]
    blend_between_frames(skeleton, frames, start, end, joint_list, window)


def interpolate_frames(skeleton, frames_a, frames_b, joint_list, start, end):
    blended_frames = deepcopy(frames_a[:])
    window = end - start
    for joint in joint_list:
        idx = skeleton.animated_joints.index(joint) * 4 + 3
        j_indices = [idx, idx + 1, idx + 2, idx + 3]
        for f in range(window):
            t = (float(f) / window)
            q_a = frames_a[start + f][j_indices]
            q_b = frames_b[start + f][j_indices]
            blended_frames[start + f][j_indices] = quaternion_slerp(q_a, q_b, t, spin=0, shortestpath=True)
    return blended_frames


def blend_towards_next_step_linear_with_original(skeleton, frames, start, end,  joint_list):
    new_frames = generate_blended_frames(skeleton, frames, start, end, joint_list, end-start)
    new_frames2 = interpolate_frames(skeleton, frames, new_frames, joint_list, start, end)
    return new_frames2


def generate_frame_using_iterative_slerp(skeleton, motions, frame_idx, weights):
    """src: https://gamedev.stackexchange.com/questions/62354/method-for-interpolation-between-3-quaternions
    """
    frame = None
    w_sum = 0.0
    for n, w in weights.items():
        if frame is None:
            frame = np.zeros(len(motions[n][0]))
            frame[:] = motions[n][frame_idx][:]
            w_sum += w
        else:
            new_w_sum = w_sum + w
            if new_w_sum > 0:
                w_a = w_sum / new_w_sum
                w_b = w / new_w_sum
                frame_b = motions[n][frame_idx]
                frame[:3] = w_a * frame[:3] + w_b * frame_b[:3]
                for idx, j in enumerate(skeleton.animated_joints):
                    q_start_idx = (idx * 4) + 3
                    q_end_idx = q_start_idx + 4
                    q_a = np.array(frame[q_start_idx:q_end_idx])
                    q_b = frame_b[q_start_idx:q_end_idx]
                    new_q = quaternion_slerp(q_a, q_b, w_b)
                    new_q /= np.linalg.norm(new_q)
                    frame[q_start_idx:q_end_idx] = new_q
            w_sum = new_w_sum
    return frame


def generate_frame_using_iterative_slerp2(skeleton, frames, weights):
    """src: https://gamedev.stackexchange.com/questions/62354/method-for-interpolation-between-3-quaternions
    """
    frame = None
    w_sum = 0.0
    for frame_idx, w in enumerate(weights):
        if frame is None:
            frame = frames[frame_idx][:]
            w_sum += w
        else:
            new_w_sum = w_sum + w
            if new_w_sum > 0:
                w_a = w_sum / new_w_sum
                w_b = w / new_w_sum
                frame_b = frames[frame_idx][:]
                frame[:3] = w_a * frame[:3] + w_b * frame_b[:3]
                for idx, j in enumerate(skeleton.animated_joints):
                    q_start_idx = (idx * 4) + 3
                    q_end_idx = q_start_idx + 4
                    q_a = np.array(frame[q_start_idx:q_end_idx])
                    q_b = frame_b[q_start_idx:q_end_idx]
                    new_q = quaternion_slerp(q_a, q_b, w_b)
                    new_q /= np.linalg.norm(new_q)
                    frame[q_start_idx:q_end_idx] = new_q
            w_sum = new_w_sum
    return frame



def smooth_euler_frames(euler_frames, discontinuity, window=20):
    """ Smooth a function around the given discontinuity frame

    Parameters
    ----------
    motion : AnimationController
        The motion to be smoothed
        ATTENTION: This function changes the values of the motion
    discontinuity : int
        The frame where the discontinuity is. (e.g. the transitionframe)
    window : (optional) int, default is 20
        The smoothing window
    Returns
    -------
    None.
    """
    d = float(discontinuity)
    s = float(window)

    smoothing_faktors = []
    for f in range(len(euler_frames)):
        value = 0.0
        if d - s <= f < d:
            tmp = ((f - d + s) / s)
            value = 0.5 * tmp ** 2
        elif d <= f <= d + s:
            tmp = ((f - d + s) / s)
            value = -0.5 * tmp ** 2 + 2 * tmp - 2

        smoothing_faktors.append(value)

    smoothing_faktors = np.array(smoothing_faktors)
    new_euler_frames = []
    for i in range(len(euler_frames[0])):
        current_value = euler_frames[:, i]
        magnitude = current_value[int(d)] - current_value[int(d) - 1]
        if magnitude > 180:
            magnitude -= 360
        elif magnitude < -180:
            magnitude += 360
        new_value = current_value + (magnitude * smoothing_faktors)
        new_euler_frames.append(new_value)
    new_euler_frames = np.array(new_euler_frames).T
    return new_euler_frames
