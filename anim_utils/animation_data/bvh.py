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

BVH
===

Biovision file format classes for reading and writing.
BVH Reader by Martin Manns
BVH Writer by Erik Herrmann

"""

import os
from collections import OrderedDict
import numpy as np
from transformations import quaternion_matrix, euler_from_matrix, euler_matrix
from .utils import rotation_order_to_string
from .quaternion_frame import quaternion_to_euler

EULER_LEN = 3
QUAT_LEN = 4
TRANSLATION_LEN = 3
TOE_NODES = ["Bip01_R_Toe0", "Bip01_L_Toe0"]
DEFAULT_FRAME_TIME = 0.013889
DEFAULT_ROTATION_ORDER = ['Xrotation','Yrotation','Zrotation']




class BVHReader(object):
    """Biovision file format class

    Parameters
    ----------
     * infile: string
    \t path to BVH file that is loaded initially
    """
    def __init__(self, infilename=""):
        self.node_names = OrderedDict()
        self.node_channels = []
        self.parent_dict = {}
        self.frame_time = None
        self.frames = None
        self.root = ""  # needed for the bvh writer
        if infilename != "":
            infile = open(infilename, "r")
            lines = infile.readlines()
            _lines = []
            for l in lines:
                if l.strip() != "":
                    _lines.append(l)
            self.process_lines(_lines)
            infile.close()
        self.filename = os.path.split(infilename)[-1]
        self.animated_joints = None

    @classmethod
    def init_from_string(cls, skeleton_string):
        bvh_reader = cls(infilename="")
        lines = skeleton_string.split("\n")
        bvh_reader.process_lines(lines)
        return bvh_reader

    def _read_skeleton(self, lines, line_index=0, n_lines=-1):
        """Reads the skeleton part of a BVH file"""
        line_index = line_index
        parents = []
        level = 0
        name = None
        if n_lines == -1:
            n_lines = len(lines)

        while line_index < n_lines:
            if lines[line_index].startswith("MOTION"):
                break

            else:
                if "{" in lines[line_index]:
                    parents.append(name)
                    level += 1

                if "}" in lines[line_index]:
                    level -= 1
                    parents.pop(-1)
                    if level == 0:
                        break

                line_split = lines[line_index].strip().split()

                if line_split:

                    if line_split[0] == "ROOT":
                        name = line_split[1]
                        self.root = name
                        self.node_names[name] = {
                            "children": [], "level": level, "channels": [], "channel_indices": []}

                    elif line_split[0] == "JOINT":
                        name = line_split[1]
                        self.node_names[name] = {
                            "children": [], "level": level, "channels": [], "channel_indices": []}
                        self.node_names[parents[-1]]["children"].append(name)

                    elif line_split[0] == "CHANNELS":
                        for channel in line_split[2:]:
                            self.node_channels.append((name, channel))
                            self.node_names[name]["channels"].append(channel)
                            self.node_names[name]["channel_indices"].append(len(self.node_channels) - 1)

                    elif line_split == ["End", "Site"]:
                        name += "_" + "".join(line_split)
                        self.node_names[name] = {"level": level}
                        # also the end sites need to be adde as children
                        self.node_names[parents[-1]]["children"].append(name)

                    elif line_split[0] == "OFFSET" and name in list(self.node_names.keys()):
                        self.node_names[name]["offset"] = list(map(float, line_split[1:]))
                line_index += 1
        return line_index

    def _read_frametime(self, lines, line_index):
        """Reads the frametime part of a BVH file"""

        if lines[line_index].startswith("Frame Time:"):
            self.frame_time = float(lines[line_index].split(":")[-1].strip())
        else:
            self.frame_time = DEFAULT_FRAME_TIME

    def _read_frames(self, lines, line_index, n_lines=-1):
        """Reads the frames part of a BVH file"""
        line_index = line_index
        if n_lines == -1:
            n_lines = len(lines)
        frames = []
        while line_index < n_lines:
            line_split = lines[line_index].strip().split()
            frames.append(np.array(list(map(float, line_split))))
            line_index += 1

        self.frames = np.array(frames)
        return line_index

    def process_lines(self, lines):
        """Reads BVH file infile

        Parameters
        ----------
         * infile: Filelike object, optional
        \tBVH file

        """
        line_index = 0
        n_lines = len(lines)
        while line_index < n_lines:
            if lines[line_index].startswith("HIERARCHY"):
                line_index = self._read_skeleton(lines, line_index, n_lines)
            if lines[line_index].startswith("MOTION"):
                self._read_frametime(lines, line_index+2)
                line_index = self._read_frames(lines, line_index+3, n_lines)
            else:
                line_index += 1

    def get_channel_indices(self, node_channels):
        """Returns indices for specified channels

        Parameters
        ----------
         * node_channels: 2-tuples of strings
        \tEach tuple contains joint name and channel name
        \te.g. ("hip", "Xposition")

        """
        return [self.node_channels.index(nc) for nc in node_channels]

    def get_node_channels(self, node_name):
        channels = None
        if node_name in self.node_names and "channels" in self.node_names[node_name]:
            channels = self.node_names[node_name]["channels"]
        return channels

    def get_node_angles(self, node_name, frame):
        """Returns the rotation for one node at one frame of an animation
        Parameters
        ----------
        * node_name: String
        \tName of node
        * bvh_reader: BVHReader
        \t BVH data structure read from a file
        * frame: np.ndarray
        \t animation keyframe frame

        """
        channels = self.node_names[node_name]["channels"]
        euler_angles = []
        rotation_order = []
        for ch in channels:
            if ch.lower().endswith("rotation"):
                idx = self.node_channels.index((node_name, ch))
                rotation_order.append(ch)
                euler_angles.append(frame[idx])
        return euler_angles, rotation_order

    def get_angles(self, node_channels):
        """Returns numpy array of angles in all frames for specified channels

        Parameters
        ----------
         * node_channels: 2-tuples of strings
        \tEach tuple contains joint name and channel name
        \te.g. ("hip", "Xposition")

        """
        indices = self.get_channel_indices(node_channels)
        return self.frames[:, indices]

    def get_animated_joints(self):
        """Returns an ordered list of joints which have animation channels"""
        if self.animated_joints is not None:
            for joint in self.animated_joints:
                yield joint
        else:
            for name, node in self.node_names.items():
                if "channels" in node.keys() and len(node["channels"]) > 0:
                    yield name

    def set_animated_joints(self, animated_joints):
        self.animated_joints = animated_joints

    def get_animated_frames(self):
        channel_indices = []
        for joint in self.get_animated_joints():
            channel_indices += self.node_names[joint]["channel_indices"]
        return self.frames[:, channel_indices]

    def convert_rotation_order(self, rotation_order):
        self.convert_skeleton_rotation_order(rotation_order)
        self.convert_motion_rotation_order(rotation_order_to_string(rotation_order))

    def convert_skeleton_rotation_order(self, rotation_order):
        # update channel indices
        rotation_list = self.node_names[self.root]["channels"][3:]
        new_indices = sorted(range(len(rotation_list)), key=lambda k : rotation_list[k])
        for node_name, node in self.node_names.items():
            if 'End' not in node_name:
                if len(node['channels']) == 6:
                    rotation_list = node['channels'][3:]
                    node['channels'][3:] = rotation_order
                    node['rotation_order'] = rotation_order_to_string(rotation_list)
                else:
                    rotation_list = node['channels']
                    node['channels'] = rotation_order
                    node['rotation_order'] = rotation_order_to_string(rotation_list)


    def convert_motion_rotation_order(self, rotation_order_str):
        new_frames = np.zeros(self.frames.shape)
        for i in range(len(new_frames)):
            for node_name, node in self.node_names.items():
                if 'End' not in node_name:
                    if len(node['channels']) == 6:
                        rot_mat = euler_matrix(*np.deg2rad(self.frames[i, node['channel_indices'][3:]]),
                                               axes=node['rotation_order'])
                        new_frames[i, node['channel_indices'][:3]] = self.frames[i, node['channel_indices'][:3]]
                        new_frames[i, node['channel_indices'][3:]] = np.rad2deg(euler_from_matrix(rot_mat, rotation_order_str))
                    else:
                        rot_mat = euler_matrix(*np.deg2rad(self.frames[i, node['channel_indices']]),
                                               axes=node['rotation_order'])
                        new_frames[i, node['channel_indices']] = np.rad2deg(euler_from_matrix(rot_mat, rotation_order_str))
        self.frames = new_frames

    def scale(self, scale):
        for node in self.node_names:
            self.node_names[node]["offset"] = [scale * o for o in self.node_names[node]["offset"]]
            if "channels" not in self.node_names[node]:
                continue
            ch = [(node, c) for c in self.node_names[node]["channels"] if "position" in c]
            if len(ch) > 0:
                ch_indices = self.get_channel_indices(ch)
                scaled_params = [scale * o for o in self.frames[:, ch_indices]]
                self.frames[:, ch_indices] = scaled_params


class BVHWriter(object):

    """ Saves an input motion defined either as an array of euler or quaternion
    frame vectors as a BVH file.

    Parameters
    ----------
    * filename: String or None
        Name of the created bvh file. Can be None.
    * skeleton: Skeleton
        Skeleton structure needed to copy the hierarchy
    * frame_data: np.ndarray
        array of motion vectors, either with euler or quaternion as
        rotation parameters
    * frame_time: float
        time in seconds for the display of each keyframe
    * is_quaternion: Boolean
        Defines wether the frame_data is quaternion data or euler data
    """

    def __init__(self, filename, skeleton, frame_data, frame_time, is_quaternion=False):
        self.skeleton = skeleton
        self.frame_data = frame_data
        self.frame_time = frame_time
        self.is_quaternion = is_quaternion
        if filename is not None:
            self.write(filename)

    def write(self, filename):
        """ Write the hierarchy string and the frame parameter string to file
        """
        bvh_string = self.generate_bvh_string()
        if filename[-4:] == '.bvh':
            filename = filename
        else:
            filename = filename + '.bvh'
        with open(filename, 'w') as outfile:
            outfile.write(bvh_string)

    def generate_bvh_string(self):
        bvh_string = self._generate_hierarchy_string(self.skeleton) + "\n"
        if self.is_quaternion:
            #euler_frames = self.convert_quaternion_to_euler_frames_skipping_fixed_joints(self.frame_data, self.is_quaternion)
            euler_frames = self.convert_quaternion_to_euler_frames(self.skeleton, self.frame_data)

        else:
            euler_frames = self.frame_data
        bvh_string += self._generate_bvh_frame_string(euler_frames, self.frame_time)
        return bvh_string

    def _generate_hierarchy_string(self, skeleton):
        """ Initiates the recursive generation of the skeleton structure string
            by calling _generate_joint_string with the root joint
        """
        hierarchy_string = "HIERARCHY\n"
        hierarchy_string += self._generate_joint_string(skeleton.root, skeleton, 0)
        return hierarchy_string

    def _generate_joint_string(self, joint, skeleton, joint_level):
        """ Recursive traversing of the joint hierarchy to create a
            skeleton structure string in the BVH format
        """
        joint_string = ""
        temp_level = 0
        tab_string = ""
        while temp_level < joint_level:
            tab_string += "\t"
            temp_level += 1

        # determine joint type
        if joint_level == 0:
            joint_string += tab_string + "ROOT " + joint + "\n"
        else:
            if len(skeleton.nodes[joint].children) > 0:
                joint_string += tab_string + "JOINT " + joint + "\n"
            else:
                joint_string += tab_string + "End Site" + "\n"

        # open bracket add offset
        joint_string += tab_string + "{" + "\n"
        offset = skeleton.nodes[joint].offset
        joint_string += tab_string + "\t" + "OFFSET " + "\t " + \
            str(offset[0]) + "\t " + str(offset[1]) + "\t " + str(offset[2]) + "\n"

        if len(skeleton.nodes[joint].children) > 0:
            # channel information
            channels = skeleton.nodes[joint].channels
            joint_string += tab_string + "\t" + \
                "CHANNELS " + str(len(channels)) + " "
            for tok in channels:
                joint_string += tok + " "
            joint_string += "\n"

            joint_level += 1
            # recursive call for all children
            for child in skeleton.nodes[joint].children:
                joint_string += self._generate_joint_string(child.node_name, skeleton, joint_level)

        # close the bracket
        joint_string += tab_string + "}" + "\n"
        return joint_string

    def convert_quaternion_to_euler_frames(self, skeleton, quat_frames):
        """ Converts the joint rotations from quaternion to euler rotations
            * quat_frames: array of motion vectors with rotations represented as quaternion
        """
        joint_names = self.skeleton.get_joint_names()
        n_frames = len(quat_frames)
        n_params = sum([len(skeleton.nodes[j].channels) for j in joint_names])
        euler_frames = np.zeros((n_frames, n_params))
        for frame_idx, quat_frame in enumerate(quat_frames):
            euler_frames[frame_idx,:TRANSLATION_LEN] = quat_frame[:TRANSLATION_LEN]
            src = TRANSLATION_LEN
            dst = 0 # the translation offset will be added
            for joint_name in joint_names:
                channels = skeleton.nodes[joint_name].channels
                n_channels = len(channels)
                rotation_order = []
                rotation_offset = None
                for idx, ch in enumerate(channels):
                    if ch.lower().endswith("rotation"):
                        rotation_order.append(ch)
                        if rotation_offset is None:
                            rotation_offset = idx

                q = quat_frame[src:src+QUAT_LEN]
                e = quaternion_to_euler(q, rotation_order)
                params_start = dst + rotation_offset
                params_end = params_start + EULER_LEN
                euler_frames[frame_idx, params_start:params_end] = e
                dst += n_channels
                src += QUAT_LEN
        return euler_frames

    def _generate_bvh_frame_string(self,euler_frames, frame_time):
        """ Converts a list of euler frames into the BVH file representation.
            * frame_time: time in seconds for the display of each keyframe
        """

        frame_parameter_string = "MOTION\n"
        frame_parameter_string += "Frames: " + str(len(euler_frames)) + "\n"
        frame_parameter_string += "Frame Time: " + str(frame_time) + "\n"
        for frame in euler_frames:
            frame_parameter_string += ' '.join([str(f) for f in frame])
            frame_parameter_string += '\n'

        return frame_parameter_string

    def convert_quaternion_to_euler_frames_skipping_fixed_joints(self, frame_data, is_quaternion=False):
        """ Converts the joint rotations from quaternion to euler rotations
            Note: for the toe joints of the rocketbox skeleton a hard set value is used
            * frame_data: array of motion vectors, either as euler or quaternion
            * node_names: OrderedDict containing the nodes of the skeleton accessible by their name
            * is_quaternion: defines wether the frame_data is quaternion data
                            or euler data
        """
        skip_joints = not self.skeleton.is_motion_vector_complete(frame_data, is_quaternion)
        if not is_quaternion:
            if not skip_joints:
                euler_frames = frame_data
            else:
                euler_frames = []
                for frame in frame_data:
                    euler_frame = self._get_euler_frame_from_partial_euler_frame(frame, skip_joints)
                    euler_frames.append(euler_frame)
        else:
            # check whether or not "Bip" frames should be ignored
            euler_frames = []
            for frame in frame_data:
                if skip_joints:
                    euler_frame = self._get_euler_frame_from_partial_quaternion_frame(frame)
                else:
                    euler_frame = self._get_euler_frame_from_quaternion_frame(frame)
                # print len(euler_frame), euler_frame
                euler_frames.append(euler_frame)
        return euler_frames

    def _get_euler_frame_from_partial_euler_frame(self, frame, skip_joints):
        euler_frame = frame[:3]
        joint_idx = 0
        for node_name in list(self.skeleton.nodes.keys()):
            if len(self.skeleton.nodes[node_name].channels) > 0:# ignore end sites
                if not node_name.startswith("Bip") or not skip_joints:
                    if node_name in TOE_NODES:
                        # special fix for unused toe parameters
                        euler_frame = np.concatenate((euler_frame,([0.0, -90.0, 0.0])),axis=0)
                    else:
                        i = joint_idx * EULER_LEN + TRANSLATION_LEN
                        euler_frame = np.concatenate((euler_frame, frame[i:i + EULER_LEN]), axis=0)
                    joint_idx += 1
                else:
                    if node_name in TOE_NODES:
                        # special fix for unused toe parameters
                        euler_frame = np.concatenate((euler_frame,([0.0, -90.0, 0.0])),axis=0)
                    else:
                        euler_frame = np.concatenate((euler_frame,([0, 0, 0])),axis=0)  # set rotation to 0
        return euler_frame

    def _get_euler_frame_from_partial_quaternion_frame(self, frame):
        euler_frame = frame[:3]     # copy root
        joint_idx = 0
        for node_name in list(self.skeleton.nodes.keys()):
            if len(self.skeleton.nodes[node_name].channels) > 0:# ignore end sites completely
                if not node_name.startswith("Bip"):
                    i = joint_idx * QUAT_LEN + TRANSLATION_LEN
                    if node_name == self.skeleton.root:
                        channels = self.skeleton.nodes[node_name].channels[TRANSLATION_LEN:]
                    else:
                        channels = self.skeleton.nodes[node_name].channels
                    euler = quaternion_to_euler(frame[i:i + QUAT_LEN], channels)
                    euler_frame = np.concatenate((euler_frame, euler), axis=0)
                    joint_idx += 1
                else:
                    if node_name in TOE_NODES:
                        # special fix for unused toe parameters
                        euler_frame = np.concatenate((euler_frame,([0.0, -90.0, 0.0])),axis=0)
                    else:
                        euler_frame = np.concatenate((euler_frame,([0, 0, 0])),axis=0)  # set rotation to 0
        return euler_frame

    def _get_euler_frame_from_quaternion_frame(self, frame):
        euler_frame = frame[:3]  # copy root
        joint_idx = 0
        for node_name in list(self.skeleton.nodes.keys()):
            if len(self.skeleton.nodes[node_name].channels) > 0:  # ignore end sites completely
                if node_name in TOE_NODES:
                    # special fix for unused toe parameters
                    euler_frame = np.concatenate((euler_frame, ([0.0, -90.0, 0.0])), axis=0)
                else:
                    i = joint_idx * QUAT_LEN + TRANSLATION_LEN
                    if node_name == self.skeleton.root:
                        channels = self.skeleton.nodes[node_name].channels[TRANSLATION_LEN:]
                    else:
                        channels = self.skeleton.nodes[node_name].channels
                    euler = quaternion_to_euler(frame[i:i + QUAT_LEN], channels)
                    euler_frame = np.concatenate((euler_frame, euler), axis=0)
                joint_idx += 1
        return euler_frame



