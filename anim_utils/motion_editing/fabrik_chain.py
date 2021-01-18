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
 https://www.sciencedirect.com/science/article/pii/S1524070311000178?via%3Dihub

 based on the pseudocode by Renzo Poddighe
 https://project.dke.maastrichtuniversity.nl/robotlab/wp-content/uploads/Bachelor-thesis-Renzo-Poddighe.pdf
"""
import math
import numpy as np
from transformations import quaternion_inverse, quaternion_multiply, quaternion_from_matrix, euler_from_quaternion
from .analytical_inverse_kinematics import calculate_limb_joint_rotation, calculate_limb_root_rotation, to_local_coordinate_system

def sign(x):
    return 1 if x >= 0 else -1

def quaternion_to_av(q):
    """ according to lee 2000
    the purely imaginary quaternion is identical to the angular velocity
    the sign of the real part gives the direction
    Since the unit quaternion space is folded by the antipodal equivalence,
    the angular velocity is twice as fast
    """
    return 2 * np.array(q[1:]) * sign(q[0])

def normalize(v):
    return v/ np.linalg.norm(v)

def get_quaternion_delta(a, b):
    return quaternion_multiply(quaternion_inverse(b), a)


def quaternion_from_axis_angle(axis, angle):
    q = [1,0,0,0]
    q[1] = axis[0] * math.sin(angle / 2)
    q[2] = axis[1] * math.sin(angle / 2)
    q[3] = axis[2] * math.sin(angle / 2)
    q[0] = math.cos(angle / 2)
    return normalize(q)



def get_offset_quat(a, b):
    a_len = np.linalg.norm(a)
    b_len = np.linalg.norm(b)
    if a_len > 0 and b_len > 0:
        q = quaternion_from_vector_to_vector(a/a_len,b/b_len)
        q /= np.linalg.norm(q)
        return q
    else:
        return [1,0,0,0]

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

def quaternion_from_vector_to_vector2(a, b):
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

def to_local_cos(skeleton, node_name, frame, q):
    # bring into parent coordinate system
    pm = skeleton.nodes[node_name].get_global_matrix(frame)[:3,:3]
    inv_p = quaternion_inverse(quaternion_from_matrix(pm))
    inv_p /= np.linalg.norm(inv_p)
    return quaternion_multiply(inv_p, q)


def orient_joint_to_target(skeleton, node, frame, src_pos, target_pos):
    if skeleton.nodes[node].parent is None:
        parent_pos = [0, 0, 0]
    else:
        parent_pos = skeleton.nodes[node].parent.get_global_position(frame)
    src_dir = normalize(src_pos - parent_pos)
    target_dir = normalize(target_pos - parent_pos)
    delta_q = quaternion_from_vector_to_vector(src_dir, target_dir)
    return normalize(delta_q)


class FABRIKBone(object):
    def __init__(self, name, child):
        self.name = name
        self.child = child
        self.position = np.array([0, 0, 0], np.float) # position of joint
        self.length = 0
        self.is_root = False
        self.is_leaf = False


ROOT_OFFSET = np.array([0,0,0], np.float)

class FABRIKChain(object):
    def __init__(self, skeleton, bones, node_order, tolerance=0.01, delta_tolerance=0.0001, max_iter=500, frame_offset=3, root_offset=ROOT_OFFSET, activate_constraints=False):
        self.skeleton = skeleton
        self.bones = bones
        self.node_order = node_order
        self.reversed_node_order = list(reversed(node_order))
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.target = None
        self.root_pos = None
        self.chain_length = 0
        self.root_offset = root_offset
        self.activate_constraints = activate_constraints
        self.frame_offset = frame_offset
        self.delta_tolerance = delta_tolerance

    def set_positions_from_frame(self, frame, parent_length):
        self.skeleton.clear_cached_global_matrices()
        for idx, node in enumerate(self.node_order):
            p = self.skeleton.nodes[node].get_global_position(frame, use_cache=True)
            #print("pos ", node, p)
            self.bones[node].position = p
            if idx ==0:
                self.root_pos = p
        self.chain_length = 0
        for node in self.node_order:
            next_node = self.bones[node].child
            if next_node is None:
                break
            d = np.linalg.norm(self.bones[next_node].position - self.bones[node].position)
            self.bones[node].length = d
            #print("length ",node, d)
            self.chain_length += d
        self.parent_length = parent_length

    def target_is_reachable(self):
        dist = np.linalg.norm(self.target - self.root_pos)
        #print("unreachable", dist, self.chain_length)
        return dist < self.chain_length+ self.parent_length

    def run(self, frame, target):
        self.target = target
        self.set_positions_from_frame(frame, 0)
        self.solve()
        return self.get_joint_parameters()

    def run_partial(self, frame, target):
        self.target = target
        self.set_positions_from_frame(frame, 0)
        self.solve()
        return self.set_partial_joint_parameters(frame)

    def run_partial_with_constraints(self, frame, target):
        self.target = target
        self.set_positions_from_frame(frame, 0)
        self.solve_partial(frame)
        return self.set_partial_joint_parameters(frame)

    def run_with_constraints(self, frame, target):
        self.target = target
        self.set_positions_from_frame(frame, 0)
        self.solve_with_constraints()
        return self.get_joint_parameters()

    def solve(self):
        if not self.target_is_reachable():
            print("unreachable")
            # if unreachable orient joints to target
            self.orient_to_target()
        else:
            print("reachable")
            # if reachable perform forward and backward reaching until tolerance is reached
            iter = 0
            distance = self.get_error()
            while distance > self.tolerance and iter< self.max_iter:
                self.backward()
                self.forward()
                iter+=1
                distance = self.get_error()
                print("iter",iter, distance)

    def solve_with_constraints(self):
        if not self.target_is_reachable():
            print("unreachable")
            # if unreachable orient joints to target
            self.orient_to_target()
        else:
            print("reachable")
            # if reachable perform forward and backward reaching until tolerance is reached
            iter = 0
            distance = self.get_error()
            while distance > self.tolerance and iter < self.max_iter:
                self.backward()
                self.forward()
                self.apply_constraints()
                iter+=1
                distance = self.get_error()
                print("iter",iter, distance)

    def solve_partial(self, frame):
        if not self.target_is_reachable():
            print("unreachable")
            # if unreachable orient joints to target
            self.orient_to_target()
        else:
            print("reachable")
            # if reachable perform forward and backward reaching until tolerance is reached
            iter = 0
            distance = self.get_error()
            while distance > self.tolerance and iter < self.max_iter:
                self.backward()
                self.forward()

                self.set_partial_joint_parameters(frame)
                self.set_positions_from_frame(frame, 0)

                iter += 1
                distance = self.get_error()
                print("iter", iter, distance)

    def get_error(self):
        end_effector = self.node_order[-1]
        return np.linalg.norm(self.bones[end_effector].position - self.target)

    def orient_to_target(self):
        for idx, node in enumerate(self.node_order[:-1]):
            next_node = self.bones[node].child
            if next_node is None:
                print("Error: none at ",node)
                break
            r = np.linalg.norm(self.target - self.bones[node].position)
            if r > 0:
                l = self.bones[node].length / r
                self.bones[next_node].position = (1 - l) * self.bones[node].position + l * self.target

    def backward(self):
        end_effector = self.node_order[-1]
        self.bones[end_effector].position = np.array(self.target)
        n_points = len(self.node_order)
        for idx in range(n_points - 2, -1, -1):
            node = self.node_order[idx]
            next_node = self.bones[node].child
            r = np.linalg.norm(self.bones[next_node].position - self.bones[node].position)
            if r > 0:
                l = self.bones[node].length / r
                self.bones[node].position = (1 - l) * self.bones[next_node].position + l * self.bones[node].position

    def forward(self):
        root_node = self.node_order[0]
        self.bones[root_node].position = self.root_pos
        for idx, node in enumerate(self.node_order[:-1]): #for p_idx in range(0, self.n_points - 1, 1):
            #next_node = self.node_order[idx + 1]
            next_node = self.bones[node].child
            r = np.linalg.norm(self.bones[next_node].position - self.bones[node].position)
            if r > 0:
                l = self.bones[node].length / r
                self.bones[next_node].position = l * self.bones[next_node].position + (1 - l) * self.bones[node].position

    def apply_constraints(self):
        frame = self.get_joint_parameters()
        self.set_positions_from_frame(frame, 0)
        return

    def get_joint_parameters(self):
        n_joints = len(self.node_order) - 1
        frame = np.zeros(n_joints*4+3)
        o = 3
        prev_point = self.root_offset
        for idx, node in enumerate(self.node_order[:-1]):
        #for node in self.skeleton.animated_joints:
            next_node = self.bones[node].child
            q = self.get_global_rotation(node, next_node)
            frame[o:o + 4] = to_local_cos(self.skeleton, node, frame, q)
            if self.skeleton.nodes[node].joint_constraint is not None and self.activate_constraints:
                self.apply_constraint_with_swing(node, frame, o)
            prev_point = self.bones[next_node].position
            o += 4
        return frame

    def set_partial_joint_parameters(self, frame):
        o = self.frame_offset
        for idx, node in enumerate(self.node_order[:-1]):
            next_node = self.bones[node].child
            q = self.get_global_rotation_non_cos(node, next_node, frame)
            frame[o:o + 4] = to_local_coordinate_system(self.skeleton,frame, node,  q)
            if self.skeleton.nodes[node].joint_constraint is not None and self.activate_constraints:
                self.apply_constraint_with_swing(node, frame, o)
            o += 4
        return frame

    def apply_constraint_with_swing(self, node, frame, o, eps=0.01):
        old_q = np.array(frame[o:o + 4])

        #remove twist rotation
        swing_q, twist_q  = self.skeleton.nodes[node].joint_constraint.split(old_q)
        frame[o:o + 4] = twist_q

        # apply swing_q to parent
        parent_q = np.array(frame[o - 4:o])
        new_parent_q = quaternion_multiply(parent_q, swing_q)
        new_parent_q = normalize(new_parent_q)
        frame[o - 4:o] = new_parent_q
        new_node_pos = self.skeleton.nodes[node].children[0].get_global_position(frame)
        diff = np.linalg.norm(new_node_pos -  self.target)

        # calculate rotation fix if necessary
        if diff > eps:
            delta_q = orient_joint_to_target(self.skeleton, node, frame, new_node_pos, self.target)
            aligned_parent_q = quaternion_multiply(delta_q, new_parent_q)
            aligned_parent_q = normalize(aligned_parent_q)
            frame[o - 4:o] = aligned_parent_q

    def apply_constraint_with_swing_global(self, node, frame, o, eps=0.01):
        old_q = np.array(frame[o:o + 4])
        next_node = self.bones[node].child
        parent_m = self.skeleton.nodes[node].get_global_matrix(frame)[:3, :3]
        node_m = self.skeleton.nodes[next_node].get_global_matrix(frame)[:3, :3]
        node_q = quaternion_from_matrix(node_m)
        node_q = normalize(node_q)
        # remove twist rotation
        swing_q, twist_q = self.skeleton.nodes[node].joint_constraint.split_global(parent_m, node_q)
        frame[o:o + 4] = twist_q

        # apply swing_q to parent
        parent_q = np.array(frame[o - 4:o])
        new_parent_q = quaternion_multiply(parent_q, swing_q)
        new_parent_q = normalize(new_parent_q)
        frame[o - 4:o] = new_parent_q
        new_node_pos = self.skeleton.nodes[node].children[0].get_global_position(frame)
        diff = np.linalg.norm(new_node_pos - self.target)
        return
        # calculate rotation fix if necessary
        if diff > eps:
            delta_q = orient_joint_to_target(self.skeleton, node, frame, new_node_pos, self.target)
            aligned_parent_q = quaternion_multiply(delta_q, new_parent_q)
            aligned_parent_q = normalize(aligned_parent_q)
            frame[o - 4:o] = aligned_parent_q


    def apply_constraint_with_swing_and_lee(self, node, frame, o, eps=0.01):
        old_q = frame[o:o + 4]
        old_node_pos = self.skeleton.nodes[node].children[0].get_global_position(frame)
        delta_q, new_q = self.skeleton.nodes[node].joint_constraint.split(old_q)
        frame[o:o + 4] = new_q
        new_node_pos = self.skeleton.nodes[node].children[0].get_global_position(frame)
        diff = np.linalg.norm(new_node_pos - old_node_pos)
        if diff > eps:
            parent_q = frame[o - 4:o]
            new_parent_q = quaternion_multiply(parent_q, delta_q)
            new_parent_q = normalize(new_parent_q)
            frame[o - 4:o] = new_parent_q
            new_node_pos = self.skeleton.nodes[node].children[0].get_global_position(frame)
            diff = np.linalg.norm(new_node_pos -old_node_pos)
            if diff > eps:
                parent_q = frame[o - 4:o]
                root = None
                if self.skeleton.nodes[node].parent is not None:
                    root = self.skeleton.nodes[node].parent.node_name
                end_effector = self.skeleton.nodes[node].children[0].node_name
                print("apply lee tolani",root, node, end_effector, diff)
                local_axis = self.skeleton.nodes[node].joint_constraint.axis
                #frame[o:o + 4] = calculate_limb_joint_rotation(self.skeleton, root, node, end_effector, local_axis, frame, self.target)
                delta_q = calculate_limb_root_rotation(self.skeleton, root, end_effector, frame, self.target)
                new_parent_q = quaternion_multiply(parent_q, delta_q)
                new_parent_q = normalize(new_parent_q)
                frame[o - 4:o] = new_parent_q

    def apply_constraint_with_swing2(self, node, parent_node, frame, o):
        next_node = self.bones[node].child
        target_global_m = self.skeleton.nodes[next_node].get_global_matrix(frame)[:3,:3]
        old_q = frame[o:o + 4]
        delta_q, new_q = self.skeleton.nodes[node].joint_constraint.split(old_q)
        frame[o:o + 4] = new_q
        new_global_m = self.skeleton.nodes[next_node].get_global_matrix(frame)[:3,:3]
        delta_global_m = np.dot(np.linalg.inv(new_global_m), target_global_m)
        actual_delta_q = normalize(quaternion_from_matrix(delta_global_m))
        parent_q = frame[o - 4:o]
        new_parent_q = quaternion_multiply(actual_delta_q, parent_q)
        new_parent_q = normalize(new_parent_q)
        frame[o - 4:o] = new_parent_q



    def apply_constraint_with_vector(self, node, parent_node, frame, o):
        next_node = self.bones[node].child
        old_pos = self.bones[next_node].position
        old_q = frame[o:o + 4]
        twist_q, swing_q = self.skeleton.nodes[node].joint_constraint.split(old_q)
        frame[o:o + 4] = swing_q

        # get global delta quaternion to apply on parent
        parent_pos = self.skeleton.nodes[parent_node].get_global_position(frame)
        next_node_pos = self.skeleton.nodes[next_node].get_global_position(frame)
        position_delta = np.linalg.norm(old_pos-next_node_pos)
        print("position delta", position_delta)
        if position_delta < 0.001:
            return
        desired_offset = normalize(old_pos - parent_pos)
        offset = normalize(next_node_pos - parent_pos)
        delta_q = quaternion_from_vector_to_vector(offset, desired_offset)
        print("deltaq", parent_node, node, next_node, next_node_pos, old_pos, twist_q, swing_q)
        # apply global delta on parent
        if True:
            global_m = self.skeleton.nodes[parent_node].get_global_matrix(frame)
            global_parent_q = normalize(quaternion_from_matrix(global_m))
            new_parent_q = quaternion_multiply(global_parent_q, delta_q)
            new_parent_q = normalize(new_parent_q)
            frame[o - 4:o] = to_local_cos(self.skeleton, parent_node, frame, new_parent_q)
        else:
            local_parent_q = frame[o - 4:o]
            local_delta = to_local_cos(self.skeleton, parent_node, frame, delta_q)

            new_parent_q = quaternion_multiply(local_parent_q, local_delta)
            new_parent_q = normalize(new_parent_q)
            frame[o - 4:o] = new_parent_q
            print(new_parent_q, local_parent_q, local_delta, delta_q)



    def get_global_rotation(self, node, next_node):
        """ FIXME works only when identity frame coordinate system is the same as the offset """

        #print("set", node, next_node)
        target = self.bones[next_node].position - self.bones[node].position
        next_offset = np.array(self.skeleton.nodes[next_node].offset)
        target_len = np.linalg.norm(target)
        if target_len > 0:
            target /= target_len
            next_offset /= np.linalg.norm(next_offset)

            # 1. sum over offsets of static nodes
            local_offset = self.get_child_offset(node, next_node)
            actual_offset = next_offset + local_offset
            actual_offset /= np.linalg.norm(actual_offset)  # actual_offset = [0.5, 0.5,0]
            # 2. get global rotation
            q = quaternion_from_vector_to_vector(actual_offset, target)
            return q

        else:
            #print("skip", target_len, self.bones[next_node].position)
            return [1, 0, 0, 0]

    def get_global_rotation_non_cos(self, node, next_node, frame):
        target_position = self.bones[next_node].position
        root_pos = self.skeleton.nodes[node].get_global_position(frame)
        #src_dir = np.dot(m, offset)
        end_effector_pos = self.skeleton.nodes[next_node].get_global_position(frame)
        src_delta = end_effector_pos - root_pos
        src_dir = src_delta / np.linalg.norm(src_delta)

        target_delta = target_position - root_pos
        target_dir = target_delta / np.linalg.norm(target_delta)

        #q = quaternion_from_vector_to_vector(offset, target_dir)
        q = quaternion_from_vector_to_vector2(src_dir, target_dir)
        q = normalize(q)
        return q


    def get_child_offset(self, node, child_node):
        """
        """
        actual_offset = np.array([0, 0, 0], np.float)
        while node is not None and self.skeleton.nodes[node].children[0].node_name != child_node:
            local_offset = np.array(self.skeleton.nodes[node].children[0].offset)
            local_offset /= np.linalg.norm(local_offset)
            actual_offset = actual_offset + local_offset
            node = self.skeleton.nodes[node].children[0].node_name
            if len(self.skeleton.nodes[node].children) < 1:
                node = None

        return actual_offset

    def get_joint_parameters_global(self):
        n_joints = len(self.node_order)-1
        frame = np.zeros(n_joints*4+3)
        o = self.frame_offset
        for idx, node in enumerate(self.node_order[:-1]):
            offset = np.array(self.skeleton.nodes[node].children[0].offset)
            offset /= np.linalg.norm(offset)
            next_node = self.bones[node].child
            if idx == 0:
                if self.skeleton.root == node:
                    dir_vector = self.bones[next_node].position
                    dir_vector_len = np.linalg.norm(dir_vector)
                    if dir_vector_len > 0 and np.linalg.norm(offset) > 0:
                        dir_vector /= dir_vector_len
                        q = quaternion_from_vector_to_vector(offset, dir_vector)
                        frame[o:o + 4] = q
                    else:
                        print("work around", offset,dir_vector_len, node)
                        frame[o:o + 4] = [1, 0, 0, 0]
                else:
                    print("work root around")
                    frame[o:o + 4] = [1, 0, 0, 0]

            else:
                q = self.get_global_rotation(node, next_node)
                frame[o:o+4] =q
            o += 4
        print(frame)
        return frame

    def get_global_positions(self):
        position_dict = dict()
        for node in self.node_order:
            position_dict[node] = self.bones[node].position
        return position_dict


    def get_end_effector_position(self):
        root_node = self.node_order[-1]
        return self.bones[root_node].position

    def get_next_nodes(self, next_nodes):
        for idx, n in enumerate(self.node_order):
            if idx+1 < len(self.node_order):
                next_nodes[n] = self.node_order[idx+1]
            else:
                next_nodes[n] = None
