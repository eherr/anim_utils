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
Created on Tue Jul 14 14:18:37 2015

@author: Erik Herrmann, Martin Manns
"""

import collections
from copy import copy
import json
import numpy as np
from transformations import quaternion_matrix
from .skeleton_node import SkeletonEndSiteNode
from .constants import ROTATION_TYPE_QUATERNION, ROTATION_TYPE_EULER, LEN_EULER, LEN_ROOT_POS, LEN_QUAT
from ..motion_editing.coordinate_cyclic_descent import run_ccd, normalize, set_global_orientation, run_ccd_look_at, orient_node_to_target_look_at, LOOK_AT_DIR, SPINE_LOOK_AT_DIR, orient_node_to_target_look_at_projected


class Skeleton(object):
    """ Data structure that stores the skeleton hierarchy information
        extracted from a BVH file with additional meta information.
    """
    def __init__(self):
        self.animated_joints = []
        self.free_joints_map = dict()
        self.skeleton_model = None
        self.frame_time = None
        self.root = None
        self.aligning_root_node = None  # Node that defines body orientation. Can be different from the root node.
        self.aligning_root_dir = None
        self.reference_frame = None
        self.reference_frame_length = None
        self.nodes = collections.OrderedDict()
        self.tool_nodes = []
        self.max_level = -1
        self.parent_dict = dict()
        self._chain_names = []
        self.identity_frame = None
        self.annotation = None

    def _get_node_desc(self, name):
        node_desc = dict()
        node = self.nodes[name]
        node_desc["name"] = node.node_name
        if node.parent is not None:
            node_desc["parent"] = node.parent.node_name
        else:
            node_desc["parent"] = None
        node_desc["quaternion_frame_index"] = node.quaternion_frame_index
        node_desc["index"] = node.index
        if type(node.offset) == np.ndarray:
            offset = node.offset.tolist()
        else:
            offset = node.offset
        node_desc["offset"] = offset
        node_desc["channels"] = node.channels
        node_desc["rotation"] = node.rotation.tolist()
        node_desc["fixed"] = node.fixed
        node_desc["level"] = node.level
        node_desc["node_type"] = node.node_type
        node_desc["children"] = []
        for c in node.children:
            c_desc = self._get_node_desc(c.node_name)
            node_desc["children"].append(c_desc)
        return node_desc

    def to_json(self):
        data = dict()
        data["animated_joints"] = self.animated_joints
        data["free_joints_map"] = self.free_joints_map
        data["skeleton_model"] = self.skeleton_model
        data["frame_time"] = self.frame_time
        data["root"] = self._get_node_desc(self.root)
        data["reference_frame"] = self.reference_frame.tolist()
        data["tool_nodes"] = self.tool_nodes
        data["aligning_root_node"] = self.aligning_root_node
        return data

    def save_to_json(self, file_name):
        with open(file_name, 'w') as outfile:
            tmp = json.dumps(self.to_json(), indent=4)
            outfile.write(tmp)
            outfile.close()

    def is_motion_vector_complete(self, frames, is_quaternion):
        if is_quaternion:
            rotation_type = ROTATION_TYPE_QUATERNION
        else:
            rotation_type = ROTATION_TYPE_EULER
        return len(frames[0]) == self.get_number_of_frame_parameters(rotation_type)

    def get_number_of_frame_parameters(self, rotation_type):
        n_parameters = 0
        for node_name in self.nodes.keys():
            local_parameters = self.nodes[node_name].get_number_of_frame_parameters(rotation_type)
            n_parameters += local_parameters
        return n_parameters

    def add_fixed_joint_parameters_to_motion(self, reduced_quat_frames):
        new_quat_frames = np.zeros((len(reduced_quat_frames), self.reference_frame_length))
        for idx, reduced_frame in enumerate(reduced_quat_frames):
            new_quat_frames[idx] = self.add_fixed_joint_parameters_to_frame(reduced_frame)
        return new_quat_frames

    def add_fixed_joint_parameters_to_frame(self, reduced_frame):
        """
        Takes parameters from the reduced frame for each joint of the complete skeleton found in the reduced skeleton
        otherwise it takes parameters from the reference frame
        :param reduced_frame:
        :return:
        """
        new_frame = np.zeros(self.reference_frame_length)
        joint_index = 0
        for joint_name in self.nodes.keys():
            if len(self.nodes[joint_name].children) > 0 and "EndSite" not in joint_name:
                if joint_name == self.root:
                    new_frame[:7] = reduced_frame[:7]
                else:
                    dest_start = joint_index * 4 + 3
                    if self.nodes[joint_name].fixed:
                        new_frame[dest_start: dest_start+4] = self.nodes[joint_name].rotation
                    else:
                        src_start = self.nodes[joint_name].quaternion_frame_index * 4 + 3
                        new_frame[dest_start: dest_start+4] = reduced_frame[src_start: src_start + 4]

                joint_index += 1
        return new_frame


    def add_fixed_joint_parameters_to_other_frame(self, reduced_frame, other_animated_joints):
        """
        Takes parameters from the reduced frame for each joint of the complete skeleton found in the reduced skeleton
        otherwise it takes parameters from the reference frame
        :param reduced_frame:
        :return:
        """
        new_frame = np.zeros(self.reference_frame_length)
        src_joint_index = 0
        dest_joint_index = 0
        for joint_name in list(self.nodes.keys()):
            if len(self.nodes[joint_name].children) > 0 and "EndSite" not in joint_name:
                if joint_name == self.root:
                    new_frame[:7] = reduced_frame[:7]
                    src_joint_index += 1
                else:
                    dest_start = dest_joint_index * 4 + 3
                    if joint_name in other_animated_joints:
                        src_start = other_animated_joints.index(joint_name) * 4 + 3
                        new_frame[dest_start: dest_start + 4] = reduced_frame[src_start: src_start + 4]
                        src_joint_index+=1
                    else:
                        new_frame[dest_start: dest_start + 4] = self.nodes[joint_name].rotation

                dest_joint_index += 1
        return new_frame

    def _get_max_level(self):
        levels = [node.level for node in list(self.nodes.values())]
        if len(levels) > 0:
            return max(levels)
        else:
            return 0

    def _get_parent_dict(self):
        """Returns a dict of node names to their parent node's name"""

        parent_dict = {}
        for node_name in list(self.nodes.keys()):
            for child_node in self.nodes[node_name].children:
                parent_dict[child_node.node_name] = node_name

        return parent_dict

    def gen_all_parents(self, node_name):
        """Generator of all parents' node names of node with node_name"""

        while node_name in self.parent_dict:
            parent_name = self.parent_dict[node_name]
            yield parent_name
            node_name = parent_name

    def _set_joint_weights(self):
        """ Gives joints weights according to their distance in the joint hierarchy
           to the root joint. The further away the smaller the weight.
        """
        self.joint_weights = [1.0/(self.nodes[n].level + 1.0) for n in list(self.nodes.keys())]
        self.joint_weight_map = collections.OrderedDict()
        weight_index = 0
        for node_name in list(self.nodes.keys()):
            self.joint_weight_map[node_name] = self.joint_weights[weight_index]
            weight_index += 1
        self.joint_weight_map["RightHand"] = 2.0
        self.joint_weight_map["LeftHand"] = 2.0
        self.joint_weights = list(self.joint_weight_map.values())

    def get_joint_weights(self):
        return list(self.joint_weight_map.values())

    def _generate_chain_names(self):
        chain_names = dict()
        for node_name in list(self.nodes.keys()):
            chain_names[node_name] = list(self.gen_all_parents(node_name))
            # Names are generated bottom to up --> reverse
            chain_names[node_name].reverse()
            chain_names[node_name] += [node_name]  # Node is not in its parent list
        return chain_names

    def get_cartesian_coordinates_from_quaternion(self, target_node_name, quaternion_frame, return_global_matrix=False):
        """Returns cartesian coordinates for one node at one frame. Modified to
         handle frames with omitted values for joints starting with "Bip"

        Parameters
        ----------

        * node_name: String
        \tName of node
         * skeleton: Skeleton
        \tBVH data structure read from a file

        """
        if self.nodes[target_node_name].level == 0:
            root_frame_position = quaternion_frame[:3]
            root_node_offset = self.nodes[target_node_name].offset
            return [t + o for t, o in
                    zip(root_frame_position, root_node_offset)]
        else:
            offsets = [list(self.nodes[node_name].offset)
                       for node_name in self._chain_names[target_node_name]]
            root_position = quaternion_frame[:3].flatten()
            offsets[0] = [r + o for r, o in zip(root_position, offsets[0])]
            j_matrices = []
            count = 0
            for node_name in self._chain_names[target_node_name]:
                if len(self.nodes[target_node_name].children) > 0:  # check if it is a joint or an end site
                    index = self.animated_joints.index(node_name) * 4 + 3
                    j_matrix = quaternion_matrix(quaternion_frame[index: index + 4])
                    j_matrix[:, 3] = offsets[count] + [1]
                else:
                    j_matrix = np.identity(4)
                    j_matrix[:, 3] = offsets[count] + [1]
                j_matrices.append(j_matrix)
                count += 1

            global_matrix = np.identity(4)
            for j_matrix in j_matrices:
                global_matrix = np.dot(global_matrix, j_matrix)
            if return_global_matrix:
                return global_matrix
            else:
                point = np.array([0, 0, 0, 1])
                point = np.dot(global_matrix, point)
                return point[:3].tolist()

    def convert_quaternion_frame_to_cartesian_frame(self, quat_frame, node_names=None):
        """
        Converts quaternion frames to cartesian frames by calling get_cartesian_coordinates_from_quaternion for each joint
        """
        if node_names is None:
            node_names = self.nodes.keys()
        cartesian_frame = []
        for node_name in node_names:
            if node_name in self.nodes:
                position = self.nodes[node_name].get_global_position(quat_frame)
                cartesian_frame.append(position)
        return cartesian_frame

    def clear_cached_global_matrices(self):
        for joint in self.nodes.values():
            joint.clear_cached_global_matrix()

    def get_root_reference_orientation(self):
        # reference orientation from BVH: 179.477078182 3.34148613293 -87.6482840381 x y z euler angles
        return self.reference_frame[3:7]

    def get_joint_indices(self, joint_names):
        indices = []
        for name in joint_names:
            if name in list(self.nodes.keys()):
                node = self.nodes[name]
                indices.append(node.index)
        return indices

    def get_n_joints(self):
        return len([node for node in list(self.nodes.values()) if len(node.channels) > 0])

    def get_joint_names(self):
        return [k for k, n in list(self.nodes.items()) if len(n.children) > 0]

    def scale(self, scale_factor):
        for node in list(self.nodes.values()):
            node.offset = np.array(node.offset) * scale_factor

    def get_channels(self, euler=False):
        channels = collections.OrderedDict()
        for node in list(self.nodes.values()):
            if node.node_name in self.animated_joints:
                node_channels = copy(node.channels)
                if not euler:
                    if np.all([ch in node_channels for ch in ["Xrotation", "Yrotation", "Zrotation"]]):
                        node_channels += ["Wrotation"]  # TODO fix order
                channels[node.node_name] = node_channels
        return channels


    def to_unity_format(self, joint_name_map=None, animated_joints=None, scale=1):
        """ Converts the skeleton into a custom json format and applies a coordinate transform to the left-handed coordinate system of Unity.
            src: http://answers.unity3d.com/questions/503407/need-to-convert-to-right-handed-coordinates.html
        """
        if animated_joints is None:
            animated_joints = [j for j, n in self.nodes.items() if "EndSite" not in j and len(n.children) > 0]  # self.animated_joints

        #print("animated joints", len(animated_joints))
        joint_descs = []
        self.nodes[self.root].to_unity_format(joint_descs, scale, joint_name_map=joint_name_map)

        data = dict()
        data["root"] = self.aligning_root_node
        data["jointDescs"] = joint_descs
        data["jointSequence"] = animated_joints
        default_pose = dict()
        default_pose["rotations"] = []
        default_pose["translations"] = []
        for node_name in animated_joints:
            node = self.nodes[node_name]
            q = node.rotation
            if len(q) == 4:
                r = {"x": -float(q[1]), "y": float(q[2]), "z": float(q[3]), "w": -float(q[0])}
            else:
                r = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
            default_pose["rotations"].append(r)
            o = node.offset
            if node_name == self.root:
                o = self.reference_frame[:3]
            t = {"x": -float(scale * o[0]), "y": float(scale * o[1]), "z": float(scale * o[2])}
            default_pose["translations"].append(t)
        
        print("new reference", default_pose["rotations"][0])
        data["referencePose"] = default_pose
        return data

    def get_reduced_reference_frame(self):
        frame = list(self.reference_frame[:3])
        for joint_name in self.animated_joints:
            frame += list(self.nodes[joint_name].rotation)
        return frame

    def set_reference_frame(self, frame):
        self.reference_frame = frame
        o = 3
        for joint_name in self.animated_joints:
            self.nodes[joint_name].rotation = frame[o:o+4]
            o += 4

    def apply_joint_constraints(self, frame):
        for n in self.animated_joints:
            if self.nodes[n].joint_constraint is not None:
                idx = self.nodes[n].quaternion_frame_index * 4 + 3
                q = frame[idx:idx + 4]
                frame[idx:idx + 4] = self.nodes[n].joint_constraint.apply(q)
        return frame

    def reach_target_position(self, frame, constraint, eps=0.01, max_iter=50, verbose=False, chain_end_joint=None):
        frame, error = run_ccd(self, frame, constraint.joint_name, constraint, eps, max_iter, chain_end_joint, verbose)
        print("reached with error", error)
        return frame

    def reach_target_positions(self, frame, constraints, chain_end_joints=None, eps=0.0001, n_max_iter=500, ccd_iters=20, verbose=False):
        error = np.inf
        prev_error = error
        n_iters = 0
        is_stuck = False
        if chain_end_joints is None:
            chain_end_joints = dict()
            for c in constraints:
                chain_end_joints[c.joint_name] = self.root
        while n_iters < n_max_iter and error > eps and not is_stuck:
            error = 0
            #print("iter", n_iters)
            for c in constraints:
                joint_error = 0
                if c.look_at:
                    pos = c.look_at_pos
                    if pos is None:
                        pos = c.position
                    joint_name = self.skeleton_model["joints"]["head"]
                    if joint_name is not None:
                        local_dir = LOOK_AT_DIR
                        if "look_at_dir" in self.skeleton_model:
                            local_dir = self.skeleton_model["look_at_dir"]
                        parent_node = self.nodes[joint_name].parent
                        if parent_node is not None:
                            frame = orient_node_to_target_look_at(self, frame, parent_node.node_name, joint_name, pos, local_dir=local_dir)
                if c.position is not None and c.relative_parent_joint_name is None:
                    frame, _joint_error = run_ccd(self, frame, c.joint_name, c, eps, ccd_iters, chain_end_joints[c.joint_name], verbose)
                    joint_error += _joint_error
                elif c.relative_parent_joint_name is not None: # run ccd on relative constraint
                    #turn relative constraint into a normal constraint
                    _c = c.instantiate_relative_constraint(self, frame)
                    #print("create relative constraint", _c.joint_name, c.relative_offset)
                    frame, _joint_error = run_ccd(self, frame, _c.joint_name, _c, eps, ccd_iters, chain_end_joints[c.joint_name], verbose)
                    joint_error += _joint_error
                elif c.orientation is not None:
                    frame = set_global_orientation(self, frame, c.joint_name, c.orientation)
                else:
                    print("ignore constraint on", c.joint_name, c.position)
                error += joint_error
            if abs(prev_error - error) < eps:
                is_stuck = True
            #print("iter", is_stuck, error,eps, prev_error, len(constraints))
            prev_error = error
            n_iters += 1
        print("reached with error", error, n_iters)
        return frame

    def set_joint_orientation(self, frame, joint_name, orientation):
        m = quaternion_matrix(orientation)
        parent = self.nodes[joint_name].parent
        if parent is not None:
            parent_m = parent.get_global_matrix(frame, use_cache=False)
            local_m = np.dot(np.linalg.inv(parent_m), m)
            q = quaternion_from_matrix(local_m)
            offset = self.nodes[joint_name].quaternion_frame_index*4+3
            frame[offset:offset+4] = normalize(q)
        return frame

    def look_at(self, frame, joint_name, position, eps=0.0001, n_max_iter=1, local_dir=LOOK_AT_DIR, chain_end_joint=None):
        #self.clear_cached_global_matrices()
        frame, error = run_ccd_look_at(self, frame, joint_name, position, eps, n_max_iter, local_dir, chain_end_joint)
        #frame = orient_node_to_target_look_at(self,frame,joint_name, joint_name, position)
        return frame

    def look_at_projected(self, frame, joint_name, position, local_dir=LOOK_AT_DIR):
        """ apply a 2d rotation around global y """
        twist_axis = None
        max_angle = None
        p_name = self.nodes[joint_name].parent.node_name
        if self.nodes[p_name ].joint_constraint is not None:
            twist_axis = self.nodes[p_name ].joint_constraint.axis
            max_angle = np.rad2deg(self.nodes[p_name ].joint_constraint.twist_max)
        frame = orient_node_to_target_look_at_projected(self, frame, self.nodes[joint_name].parent.node_name, joint_name, position, local_dir, twist_axis, max_angle)
        return frame

    def get_reduced_euler_frames(self, euler_frames, has_root=True):
        '''
        create reduced size of euler frames based on animated joints
        :param euler_frames: n_frames * n_dims
        :return:
        '''
        nonEndSite_joints = self.get_joints_without_EndSite()
        euler_frames = np.asarray(euler_frames)
        if has_root:
            assert euler_frames.shape[1] == LEN_ROOT_POS + LEN_EULER * len(nonEndSite_joints)
            new_euler_frames = np.zeros([euler_frames.shape[0], LEN_ROOT_POS + LEN_EULER * len(self.animated_joints)])
        else:
            assert euler_frames.shape[1] == LEN_EULER * len(nonEndSite_joints)
            new_euler_frames = np.zeros([euler_frames.shape[0], LEN_EULER * len(self.animated_joints)])
        for i in range(euler_frames.shape[0]):
            new_euler_frames[i] = self.get_reduced_euler_frame(euler_frames[i], has_root=has_root)
        return new_euler_frames

    def get_reduced_euler_frame(self, euler_frame, has_root=True):
        '''
        create a reduced size of euler frame based on animated joints
        :param euler_frames: n_frames * n_dims
        :return:
        '''
        if has_root:
            new_euler_frame = np.zeros(LEN_ROOT_POS + LEN_EULER * len(self.animated_joints))
            new_euler_frame[:LEN_ROOT_POS] = euler_frame[:LEN_ROOT_POS]
            i = 0
            for joint in self.animated_joints:
                if joint == self.root:
                    new_euler_frame[LEN_ROOT_POS+i*LEN_EULER: LEN_ROOT_POS+(i+1)*LEN_EULER] = euler_frame[self.nodes[joint].channel_indices[LEN_ROOT_POS:]]
                else:
                    new_euler_frame[LEN_ROOT_POS+i*LEN_EULER: LEN_ROOT_POS+(i+1)*LEN_EULER] = euler_frame[self.nodes[joint].channel_indices]
                i += 1
        else:
            new_euler_frame = np.zeros(LEN_EULER * len(self.animated_joints))
            i = 0
            for joint in self.animated_joints:
                if joint == self.root:
                    new_euler_frame[LEN_ROOT_POS + i * LEN_EULER: LEN_ROOT_POS + (i + 1) * LEN_EULER] = euler_frame[
                        self.nodes[joint].channel_indices[LEN_ROOT_POS:]]
                else:
                    new_euler_frame[LEN_ROOT_POS + i * LEN_EULER: LEN_ROOT_POS + (i + 1) * LEN_EULER] = euler_frame[
                        self.nodes[joint].channel_indices]
                i += 1
        return new_euler_frame

    def get_joints_without_EndSite(self):
        joints = []
        for joint in self.nodes.keys():
            if 'EndSite' not in joint:
                joints.append(joint)
        return joints

    def generate_bone_list_description(self, animated_joints=None):
        '''
        Generate a list of bone description for point cloud visualization
        :return:
        '''
        bones = []
        index = 0
        if animated_joints is None:
            animated_joints = self.animated_joints
        for node_name, node in self.nodes.items():
            if node_name in animated_joints:
                parent_joint_name = node.get_parent_name(animated_joints)
                bone_desc = {'name': node_name, 'parent': parent_joint_name, 'index': index}
                index += 1
                bones.append(bone_desc)
        return bones

    def get_body_hip2foot_height(self):
        """ helper function for autoscale by gemlongman"""
        if self.skeleton_model is None:
            print("skeleton.skeleton_model is None")
            return None
        joint_pelvis = self.skeleton_model["joints"]["pelvis"] # upperleg maybe better
        joint_right_toe = self.skeleton_model["joints"]["right_ankle"]# ["right_toe"] 
        ident_f = self.identity_frame
        joint_pelvis_p = self.nodes[joint_pelvis].get_global_position(ident_f)
        joint_right_toe_p = self.nodes[joint_right_toe].get_global_position(ident_f)
        delta = joint_pelvis_p - joint_right_toe_p
        #it should be same dircetion as body up,there is a simple way to get length
        if delta[1] < 0.01:
            return  max(abs(delta.max()), abs(delta.min()))
        return delta[1]
