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
from .fabrik_chain import FABRIKChain, to_local_cos, quaternion_from_vector_to_vector, FABRIKBone, ROOT_OFFSET


def get_child_offset(skeleton, node, child_node):
    actual_offset = np.array([0, 0, 0], np.float)
    while node is not None and skeleton.nodes[node].children[0].node_name != child_node:
        local_offset = np.array(skeleton.nodes[node].children[0].offset)
        local_offset_len = np.linalg.norm(local_offset)
        if local_offset_len > 0:
            local_offset /= local_offset_len
            actual_offset = actual_offset + local_offset
            node = skeleton.nodes[node].children[0].node_name
            if len(skeleton.nodes[node].children) < 1:
                node = None
        else:
            node = None
    return actual_offset


def get_global_joint_parameters_from_positions(skeleton, positions, next_nodes):
    n_parameters = len(skeleton.animated_joints)*4+3
    frame = np.zeros(n_parameters)
    frame[:3] = positions[skeleton.root]
    o = 3
    print(skeleton.animated_joints)
    print(next_nodes.keys())
    print(positions.keys())
    for node in skeleton.animated_joints:
        next_node = next_nodes[node]
        if next_node in positions:
            next_offset = np.array(skeleton.nodes[next_node].offset)
            next_offset /= np.linalg.norm(next_offset)

            # 1. sum over offsets of static nodes
            local_offset = get_child_offset(skeleton, node, next_node)
            #print(node, local_offset)
            actual_offset = next_offset + local_offset
            actual_offset /= np.linalg.norm(actual_offset)  # actual_offset = [0.5, 0.5,0]


            dir_to_child = positions[next_node] - positions[node]
            dir_to_child /= np.linalg.norm(dir_to_child)

            # 2. get global rotation
            global_q = quaternion_from_vector_to_vector(actual_offset, dir_to_child)
            frame[o:o + 4] = global_q
        else:
            #print("ignore", node, next_node)
            frame[o:o + 4] = [1,0,0,0]
        o += 4
    return frame


def global_to_local_frame(skeleton, global_frame):
    n_parameters = len(skeleton.animated_joints)*4+3
    frame = np.zeros(n_parameters)
    o = 3
    for node in skeleton.animated_joints:
        frame[o:o + 4] = to_local_cos(skeleton, node, frame, global_frame[o:o + 4])

        o += 4

    return frame


class FABRIKNode(object):
    def __init__(self, skeleton, root, parent_chain=None, tolerance=0.01, max_iter=100, root_offset=ROOT_OFFSET):
        self.root_pos = np.array([0,0,0], dtype=np.float)
        self.skeleton = skeleton
        self.n_parameters = len(self.skeleton.animated_joints)*4+3
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.parent_chain = parent_chain
        self.root = root
        self.root_offset = root_offset
        self.child_nodes = list()
        # self.target = None
        # self.position = None
        self.is_leaf = True
        self.construct_from_skeleton(skeleton, root, tolerance, max_iter)

    def construct_from_skeleton(self, skeleton, root, tolerance, max_iter):
        """ there need to be two end effectors"""
        children = [c.node_name for c in skeleton.nodes[root].children]
        for c in children:
            self.is_leaf = False
            current_node = c
            node_order = [self.root]
            while current_node is not None:
                n_children = len(skeleton.nodes[current_node].children)
                if n_children == 1:  # append to chain
                    child_node = skeleton.nodes[current_node].children[0].node_name
                    # only add list to joints
                    if not skeleton.nodes[current_node].fixed:
                        node_order.append(current_node)# skip fixed nodes
                    current_node = child_node
                else:  # stop chain # split up by adding child nodes
                    if n_children > 0:
                        node_order.append(current_node)
                    bones = dict()
                    for idx, node in enumerate(node_order):
                        child_node = None
                        if idx+1 < len(node_order):
                            child_node = node_order[idx + 1]
                        bones[node] = FABRIKBone(node, child_node)
                        if idx == 0 and self.parent_chain is None :
                            bones[node].is_root = True
                        else:
                            bones[node].is_root = False
                    parent_chain = FABRIKChain(skeleton, bones, node_order)
                    print("construct node at",self.root , current_node, node_order)
                    node = FABRIKNode(skeleton, current_node, parent_chain, tolerance, max_iter)
                    self.child_nodes.append(node)
                    current_node = None

    def backward_stage(self):
        for c in self.child_nodes:
            c.backward_stage()  # calculate backward update of children of child to start at the end effectors

        if not self.is_leaf:
            positions = [c.get_chain_root_position() for c in self.child_nodes]
            t = np.mean(positions, axis=0)
            #print("centroid",t)
            #print("no leaf")
        else:
            t = self.parent_chain.target
            #print("leaf")

        if self.parent_chain is not None:
            self.parent_chain.target = t
            if self.target_is_reachable():
               self.parent_chain.backward()
            else:
                print("unreachable")
                # if unreachable orient joints to target
                self.parent_chain.orient_to_target()

    def target_is_reachable(self):
        return self.parent_chain.target_is_reachable()

    def forward_stage(self):
        # calculate forward update of parent chain
        if self.parent_chain is not None:
            #if self.parent_chain.target_is_reachable():
            self.parent_chain.forward()
            #else:
            #    self.parent_chain.orient_to_target()

        # calculate forward update of children of child
        for c in self.child_nodes:
            if self.parent_chain is not None:
                c.parent_chain.root_pos = self.parent_chain.get_end_effector_position()
            else:
                c.parent_chain.root_pos = self.root_pos
            c.forward_stage()

    def get_chain_root_position(self):
        root_node = self.parent_chain.node_order[0]
        return self.parent_chain.bones[root_node].position


    def set_positions_from_frame(self, frame, parent_length):
        if self.parent_chain is not None:
            self.parent_chain.set_positions_from_frame(frame, parent_length)
            parent_length += self.parent_chain.chain_length
        for c in self.child_nodes:
            c.set_positions_from_frame(frame, parent_length)

    def get_error(self):
        error = 0
        for c in self.child_nodes:
            error += c.get_error()
        if self.is_leaf:
            if self.parent_chain.target_is_reachable():
                error += self.parent_chain.get_error()
            else:
                error += 0
        return error

    def solve(self):
        iter = 0
        distance = self.get_error()
        while distance > self.tolerance and iter < self.max_iter:
            self.backward_stage()
            self.forward_stage()
            distance = self.get_error()
            iter += 1
        n_out_of_reach = self.targets_out_of_reach()
        if n_out_of_reach > 0:
            self.orient_to_targets()
        print("solved", distance, n_out_of_reach)
        self.print_end_effectors()

    def print_end_effectors(self):
        n_targets = 0
        for c in self.child_nodes:
            if c.is_leaf:
                print(c.root, c.parent_chain.target, c.parent_chain.get_end_effector_position())
            else:
                c.print_end_effectors()
        return n_targets

    def targets_out_of_reach(self):
        n_targets = 0
        for c in self.child_nodes:
            if c.is_leaf and not c.parent_chain.target_is_reachable():
                n_targets+=1
            else:
                n_targets += c.targets_out_of_reach()
        return n_targets

    def orient_to_targets(self):
        for c in self.child_nodes:
            if c.is_leaf and not c.parent_chain.target_is_reachable():
                c.parent_chain.orient_to_target()
            else:
                c.orient_to_targets()

    def get_joint_parameters(self, frame):
        #print("get joint parameters")
        for c in self.child_nodes:
            print("from", c.parent_chain.node_order[0], "to", c.parent_chain.node_order[-1])
            global_frame = c.parent_chain.get_joint_parameters_global()
            src = 3
            if self.parent_chain is None:
                animated_joints = c.parent_chain.node_order
            else:
                animated_joints = c.parent_chain.node_order[1:]
            for j in animated_joints: #ignore root rotation
                dest = self.skeleton.animated_joints.index(j)*4 +3
                frame[dest:dest+4] = to_local_cos(self.skeleton, j, frame, global_frame[src:src+4])
                src += 4
            c.get_joint_parameters(frame)
        return frame

    def set_targets(self, targets):
        #print("set targets",targets)
        if self.is_leaf:
            if self.root in targets:
                self.target = targets[self.root]
                self.parent_chain.target = self.target
                #print("set target",self.root)
            else:
                self.target = self.parent_chain.get_end_effector_position() # keep the initial position as target for unconstrained leafs
                self.parent_chain.target = self.target
                print("set end effector position")
        else:
            for c in self.child_nodes:
                c.set_targets(targets)

    def run(self, orig_frame, targets):
        self.set_positions_from_frame(orig_frame, 0)
        self.set_targets(targets)
        self.solve()
        if False:
            frame = np.zeros(self.n_parameters)
            self.get_joint_parameters(frame)
        else:
            joint_pos = dict()
            self.get_global_positions(joint_pos)
            next_nodes = dict()
            self.get_next_nodes(next_nodes)
            frame = get_global_joint_parameters_from_positions(self.skeleton, joint_pos, next_nodes)
            frame = global_to_local_frame(self.skeleton, frame)
            #print("done")
            #self.print_frame_parameters(frame)
        return frame

    def get_next_nodes(self, next_nodes):
        if self.parent_chain is not None:
            self.parent_chain.get_next_nodes(next_nodes)
        for c in self.child_nodes:
            c.get_next_nodes(next_nodes)

    def print_frame_parameters(self, frame):
        idx = 3
        for n in self.skeleton.nodes:
            if len(self.skeleton.nodes[n].children) > 0:
                print(n, frame[idx: idx+4])
                idx +=4
        return

    def draw(self, m,v,p, l):
        if self.parent_chain is not None:
            self.parent_chain.draw(m,v,p,l)
        for c in self.child_nodes:
            c.draw(m, v, p, l)

    def get_global_positions(self, positions):
        if self.parent_chain is not None:
            positions.update(self.parent_chain.get_global_positions())
        for c in self.child_nodes:
            c.get_global_positions(positions)