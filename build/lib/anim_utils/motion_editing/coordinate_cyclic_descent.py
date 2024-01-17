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
from transformations import quaternion_from_matrix, quaternion_matrix, quaternion_slerp, quaternion_about_axis, quaternion_multiply, quaternion_inverse
from ..animation_data.joint_constraints import quaternion_to_axis_angle
from math import acos
import math
from ..animation_data.utils import get_rotation_angle


def normalize(v):
    return v / np.linalg.norm(v)


def to_local_coordinate_system(skeleton, frame, joint_name, q, use_cache=True):
    """ given a global rotation bring it to the local coordinate system"""
    if skeleton.nodes[joint_name].parent is not None:
        global_m = quaternion_matrix(q)
        parent_joint = skeleton.nodes[joint_name].parent.node_name

        # delete global matrix of parent_joint but use cache to save time
        skeleton.nodes[parent_joint].cached_global_matrix = None 
        parent_m = skeleton.nodes[parent_joint].get_global_matrix(frame, use_cache=use_cache)

        new_local = np.dot(np.linalg.inv(parent_m), global_m)
        return quaternion_from_matrix(new_local)
    else:
        return q


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


def orient_end_effector_to_target(skeleton, root, end_effector, frame, constraint):
    """ find angle between the vectors end_effector - root and target- root """

    # align vectors
    root_pos = skeleton.nodes[root].get_global_position(frame)
    if constraint.offset is not None:
        m = skeleton.nodes[end_effector].get_global_matrix(frame)
        end_effector_pos = np.dot(m, constraint.offset)[:3]
    else:
        end_effector_pos = skeleton.nodes[end_effector].get_global_position(frame)

    src_delta = end_effector_pos - root_pos
    src_dir = src_delta / np.linalg.norm(src_delta)

    target_delta = constraint.position - root_pos
    target_dir = target_delta / np.linalg.norm(target_delta)

    root_delta_q = quaternion_from_vector_to_vector(src_dir, target_dir)
    root_delta_q = normalize(root_delta_q)

    if skeleton.nodes[root].stiffness > 0:
        t = 1 - skeleton.nodes[root].stiffness
        root_delta_q = quaternion_slerp([1,0,0,0], root_delta_q, t)
        root_delta_q = normalize(root_delta_q)

    global_m = quaternion_matrix(root_delta_q)
    parent_joint = skeleton.nodes[root].parent.node_name
    parent_m = skeleton.nodes[parent_joint].get_global_matrix(frame, use_cache=True)
    old_global = np.dot(parent_m, skeleton.nodes[root].get_local_matrix(frame))
    new_global = np.dot(global_m, old_global)
    q = quaternion_from_matrix(new_global)
    return normalize(q)


def orient_node_to_target(skeleton,frame,node_name, end_effector, constraint):
    #o = skeleton.animated_joints.index(node_name) * 4 + 3
    o = skeleton.nodes[node_name].quaternion_frame_index * 4 + 3
    q = orient_end_effector_to_target(skeleton, node_name, end_effector, frame, constraint)
    q = to_local_coordinate_system(skeleton, frame, node_name, q)
    frame[o:o + 4] = q
    return frame


def apply_joint_constraint(skeleton,frame,node_name):
    #o = skeleton.animated_joints.index(node_name) * 4 + 3
    o = skeleton.nodes[node_name].quaternion_frame_index * 4 + 3
    q = np.array(frame[o:o + 4])
    q = skeleton.nodes[node_name].joint_constraint.apply(q)
    frame[o:o + 4] = q
    return frame

def set_global_orientation(skeleton, frame, node_name, orientation):
    """ set orientation of parent of node assuming node is an end point"""
    parent_node = skeleton.nodes[node_name].parent
    if parent_node is not None and parent_node.parent is not None:
        parent_name = parent_node.node_name
        m = quaternion_matrix(orientation)
        #o = skeleton.animated_joints.index(node_name) * 4 + 3
        o = skeleton.nodes[parent_name].quaternion_frame_index * 4 + 3
        parent_m = skeleton.nodes[parent_name].parent.get_global_matrix(frame, use_cache=True)
        local_m = np.dot(np.linalg.inv(parent_m), m)
        q = quaternion_from_matrix(local_m)
        frame[o:o + 4] = normalize(q)
    return frame

AXES = ["x", "y"]
def set_tool_orientation(skeleton, frame, node_name, constraint):
    """ set orientation of parent of node based on coordinate system assuming node is an end point"""
    parent_node = skeleton.nodes[node_name].parent
    parent_name = parent_node.node_name
    o = parent_node.quaternion_frame_index * 4 + 3
    for a in AXES:
         # delete global matrix of parent_joint but use cache to save time
        parent_node.cached_global_matrix = None
        global_m = parent_node.get_global_matrix(frame, use_cache=True)
        #add the offset of the end effector
        global_m[:3, 3] += skeleton.nodes[node_name].offset
        if a in constraint.src_tool_cos and a in constraint.dest_tool_cos:
            tool_axis_offset = constraint.src_tool_cos[a]
            global_tool_target_axis = constraint.dest_tool_cos[a]
            src_axis = np.dot(global_m, tool_axis_offset)[:3]
            delta_q = quaternion_from_vector_to_vector(src_axis,  global_tool_target_axis)
            delta_q = normalize(delta_q)
            delta_m = quaternion_matrix(delta_q)
            global_m = np.dot(delta_m, global_m)
        q = quaternion_from_matrix(global_m)
        q = to_local_coordinate_system(skeleton, frame, parent_name, q)
        frame[o:o + 4] = normalize(q)
    return frame

def run_ccd(skeleton, frame, end_effector_name, constraint, eps=0.01, n_max_iter=50, chain_end_joint=None, verbose=False):
    pos = skeleton.nodes[end_effector_name].get_global_position(frame)
    error = np.linalg.norm(constraint.position-pos)
    prev_error = error
    n_iters = 0
    while error > eps and n_iters < n_max_iter:
        node = skeleton.nodes[end_effector_name].parent
        depth = 0

        while node is not None and node.node_name != chain_end_joint and node.parent is not None:
            static = False
            if node.joint_constraint is not None and node.joint_constraint.is_static:
                static = True

            if not static:
                frame = orient_node_to_target(skeleton,frame, node.node_name, end_effector_name, constraint)
                if constraint.orientation is not None:
                    frame = set_global_orientation(skeleton, frame, end_effector_name, constraint.orientation)
                elif constraint.src_tool_cos is not None and constraint.dest_tool_cos is not None:
                    frame = set_tool_orientation(skeleton, frame, end_effector_name, constraint)

                if node.joint_constraint is not None:
                    frame = apply_joint_constraint(skeleton, frame, node.node_name)
            node = node.parent
            depth += 1

        if constraint.offset is not None:
            m = skeleton.nodes[end_effector_name].get_global_matrix(frame)
            end_effector_pos = np.dot(m, constraint.offset)[:3]
        else:
            end_effector_pos = skeleton.nodes[end_effector_name].get_global_position(frame)

        error = np.linalg.norm(constraint.position - end_effector_pos)
        n_iters += 1

    if verbose:
        print("error at", n_iters, ":", error, "c:",constraint.position,"pos:", pos, chain_end_joint)
    return frame, error


LOOK_AT_DIR = [0, -1,0]
SPINE_LOOK_AT_DIR = [0,0,1]

def look_at_target(skeleton, root, end_effector, frame, position, local_dir=LOOK_AT_DIR):
    """ find angle between the look direction and direction between end effector and target"""
    #direction of endeffector
    m = skeleton.nodes[end_effector].get_global_matrix(frame)
    #offset = skeleton.nodes[end_effector].offset
    end_effector_dir = np.dot(m[:3,:3], local_dir)
    end_effector_dir = end_effector_dir / np.linalg.norm(end_effector_dir)

    # direction from endeffector to target
    end_effector_pos = m[:3, 3]
    target_delta = position - end_effector_pos
    target_dir = target_delta / np.linalg.norm(target_delta)

    # find rotation to align vectors
    root_delta_q = quaternion_from_vector_to_vector(end_effector_dir, target_dir)
    root_delta_q = normalize(root_delta_q)

    #apply global delta to get new global matrix of joint
    global_m = quaternion_matrix(root_delta_q)
    parent_joint = skeleton.nodes[root].parent.node_name
    parent_m = skeleton.nodes[parent_joint].get_global_matrix(frame, use_cache=True)
    old_global = np.dot(parent_m, skeleton.nodes[root].get_local_matrix(frame))
    new_global = np.dot(global_m, old_global)
    q = quaternion_from_matrix(new_global)
    return normalize(q)



def orient_node_to_target_look_at(skeleton,frame,node_name, end_effector, position, local_dir=LOOK_AT_DIR):
    o = skeleton.nodes[node_name].quaternion_frame_index * 4 + 3
    q = look_at_target(skeleton, node_name, end_effector, frame, position, local_dir)
    q = to_local_coordinate_system(skeleton, frame, node_name, q)
    frame[o:o + 4] = q
    return frame



def run_ccd_look_at(skeleton, frame, end_effector_name, position, eps=0.01, n_max_iter=1, local_dir=LOOK_AT_DIR, chain_end_joint=None, verbose=False):
    error = np.inf
    n_iter = 0
    while error > eps and n_iter < n_max_iter:
        node = skeleton.nodes[end_effector_name].parent
        depth = 0
        while node is not None and node.node_name != chain_end_joint and node.parent is not None:#and node.node_name != skeleton.root:
            frame = orient_node_to_target_look_at(skeleton,frame, node.node_name, end_effector_name, position, local_dir)
            if node.joint_constraint is not None:
                frame = apply_joint_constraint(skeleton, frame, node.node_name)
            node = node.parent
            depth += 1

        m = skeleton.nodes[end_effector_name].get_global_matrix(frame)

        end_effector_dir = np.dot(m[:3, :3], local_dir)
        end_effector_dir = end_effector_dir / np.linalg.norm(end_effector_dir)

        # direction from endeffector to target
        end_effector_pos = m[:3,3]
        target_delta = position - end_effector_pos
        target_dir = target_delta / np.linalg.norm(target_delta)
        root_delta_q = quaternion_from_vector_to_vector(end_effector_dir, target_dir)
        root_delta_q = normalize(root_delta_q)
        v, a = quaternion_to_axis_angle(root_delta_q)
        error = abs(a)
        #error = np.linalg.norm(target_dir-end_effector_dir)
        #print(error)
        n_iter+=1
    if verbose:
        print("error at", n_iter, ":", error, "c:",position,"pos:")

    return frame, error




def look_at_target_projected(skeleton, root, end_effector, frame, position, local_dir=LOOK_AT_DIR):
    """ find angle between the look direction and direction from end effector to target projected on the xz plane"""
    #direction of endeffector
    m = skeleton.nodes[root].get_global_matrix(frame)
    end_effector_dir = np.dot(m[:3,:3], local_dir)
    end_effector_dir[1] = 0
    end_effector_dir = end_effector_dir / np.linalg.norm(end_effector_dir)

    # direction from endeffector to target
    end_effector_pos = m[:3, 3]
    target_delta = position - end_effector_pos
    target_delta[1] = 0
    target_dir = target_delta / np.linalg.norm(target_delta)

    # find rotation to align vectors
    root_delta_q = quaternion_from_vector_to_vector(end_effector_dir, target_dir)
    root_delta_q = normalize(root_delta_q)
    #apply global delta to get new global matrix of joint
    global_m = quaternion_matrix(root_delta_q)
    old_global = skeleton.nodes[root].get_global_matrix(frame)
    new_global = np.dot(global_m, old_global)
    q = quaternion_from_matrix(new_global)
    return normalize(q)

def quaternion_to_axis_angle(q):
    """http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/

    """
    a = 2* math.acos(q[0])
    s = math.sqrt(1- q[0]*q[0])
    if s < 0.001:
        x = q[1]
        y = q[2]
        z = q[3]
    else:
        x = q[1] / s
        y = q[2] / s
        z = q[3] / s
    v = np.array([x,y,z])
    if np.sum(v)> 0:
        return normalize(v),a
    else:
        return v, a

def swing_twist_decomposition(q, twist_axis):
    """ code by janis sprenger based on
        Dobrowsolski 2015 Swing-twist decomposition in Clifford algebra. https://arxiv.org/abs/1506.05481
    """
    q = normalize(q)
    #twist_axis = np.array((q * offset))[0]
    projection = np.dot(twist_axis, np.array([q[1], q[2], q[3]])) * twist_axis
    twist_q = np.array([q[0], projection[0], projection[1],projection[2]])
    if np.linalg.norm(twist_q) == 0:
        twist_q = np.array([1,0,0,0])
    twist_q = normalize(twist_q)
    swing_q = quaternion_multiply(q, quaternion_inverse(twist_q))#q * quaternion_inverse(twist)
    return swing_q, twist_q


def orient_node_to_target_look_at_projected(skeleton, frame,node_name, end_effector, position, local_dir=LOOK_AT_DIR, twist_axis=None, max_angle=None):
    o = skeleton.nodes[node_name].quaternion_frame_index * 4 + 3
    q = look_at_target_projected(skeleton, node_name, end_effector, frame, position, local_dir)
    q = to_local_coordinate_system(skeleton, frame, node_name, q)
    
    if max_angle is not None and twist_axis is not None:
        t_min_angle = np.radians(-max_angle)
        t_max_angle = np.radians(max_angle)
        swing_q, twist_q = swing_twist_decomposition(q, twist_axis)
        v, a = quaternion_to_axis_angle(twist_q)
        sign = 1
        if np.sum(v) < 0:
            sign = -1
        a *= sign
        a = max(a, t_min_angle)
        a = min(a, t_max_angle)
        new_twist_q = quaternion_about_axis(a, twist_axis)
        q = quaternion_multiply(swing_q, new_twist_q)
        q = normalize(q)
    frame[o:o + 4] = q
    return frame
