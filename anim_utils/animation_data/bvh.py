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


class BVHData:
    """Biovision data format class
    """
    def __init__(self):
        self.node_names = OrderedDict()
        self.node_channels = []
        self.parent_dict = {}
        self.root = None
        self.frame_time = DEFAULT_FRAME_TIME
        self.frames = None
        self.animated_joints = None
        self.filename = ""


    def get_node_angles(self, node_name, frame):
        """Returns the rotation for one node at one frame of an animation
        Parameters
        ----------
        * node_name: String
        \tName of node
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
        #new_indices = sorted(range(len(rotation_list)), key=lambda k : rotation_list[k])
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

class BVHReader(BVHData):
    """Biovision file format class

    Parameters
    ----------
     * filename: string
    \t path to BVH file that is loaded initially
    """
    def __init__(self, filename=""):
        super().__init__()
        if filename != "":
            self.filename = os.path.split(filename)[-1]
            with open(filename, "r") as file:
                lines = file.readlines()
            self.process_lines(lines)

    @classmethod
    def init_from_string(cls, skeleton_string):
        bvh_reader = cls(infilename="")
        lines = skeleton_string.split("\n")
        bvh_reader.process_lines(lines)
        return bvh_reader

    def process_lines(self, lines):
        """Reads lines from a BVH file

        Parameters
        ----------
         * lines: list of strings

        """
        lines = [l for l in lines if l.strip() != ""]
        line_index = 0
        n_lines = len(lines)
        while line_index < n_lines:
            if lines[line_index].startswith("HIERARCHY"):
                line_index = self._read_skeleton(lines, line_index, n_lines)
            if lines[line_index].startswith("MOTION") and n_lines > line_index+3:
                self._read_frametime(lines, line_index+2)
                line_index = self._read_frames(lines, line_index+3, n_lines)
            else:
                line_index += 1

    def _read_skeleton(self, lines, line_index, n_lines):
        """Reads the skeleton part of a BVH file"""
        line_index = line_index
        parents = []
        node_name = None
        parent_name = None
        self.node_names = OrderedDict()
        self.node_channels = []
        while line_index < n_lines and not lines[line_index].startswith("MOTION"):
            split_line = lines[line_index].strip().split()
            if not split_line:
                continue
            if "{" in split_line:
                parents.append(node_name)
            elif "}" in split_line:
                parents.pop(-1)
            level = len(parents)
            if level > 0:
                parent_name = parents[-1]
            node_name = self._read_split_joint_line(split_line, level, node_name, parent_name)
            if self.root is None:
                self.root = node_name
            line_index += 1
        return line_index


    def _read_split_joint_line(self, split_line, level, node_name, parent):
        if split_line[0] == "ROOT" or split_line[0] == "JOINT":
            node_name = split_line[1]
            node_data = {"children": [], "level": level, "channels": [], "channel_indices": []}
            self.node_names[node_name] = node_data
            if parent is not None:
                self.node_names[parent]["children"].append(node_name)

        elif split_line[0] == "CHANNELS":
            for channel in split_line[2:]:
                self.node_channels.append((node_name, channel))
                self.node_names[node_name]["channels"].append(channel)
                self.node_names[node_name]["channel_indices"].append(len(self.node_channels) - 1)

        elif split_line == ["End", "Site"]:
            end_site_name = node_name  + "_" + "".join(split_line)
            self.node_names[end_site_name] = {"level": level}
            # also the end sites need to be adde as children
            self.node_names[node_name]["children"].append(end_site_name)
            node_name = end_site_name

        elif split_line[0] == "OFFSET" and node_name is not None:
            self.node_names[node_name]["offset"] = list(map(float, split_line[1:]))
        return node_name

    def _read_frametime(self, lines, line_index):
        """Reads the frametime part of a BVH file"""
        if lines[line_index].startswith("Frame Time:"):
            self.frame_time = float(lines[line_index].split(":")[-1].strip())

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



def write_euler_frames_to_bvh_file(filename, skeleton, euler_frames, frame_time):
    """ Write the hierarchy string and the frame parameter string to a file
    Parameters
    ----------
    * filename: String 
        Name of the created bvh file. 
    * skeleton: Skeleton
        Skeleton structure needed to copy the hierarchy
    * euler_frames: np.ndarray
        array of pose rotation parameters in euler angles
    * frame_time: float
        time in seconds for the display of each keyframe
    """
    bvh_string = generate_bvh_string(skeleton, euler_frames, frame_time)
    if filename[-4:] == '.bvh':
        filename = filename
    else:
        filename = filename + '.bvh'
    with open(filename, 'w') as outfile:
        outfile.write(bvh_string)


def generate_bvh_string(skeleton, euler_frames, frame_time):
    """ Write the hierarchy string and the frame parameter string to a file
    Parameters
    ----------
    * skeleton: Skeleton
        Skeleton structure needed to copy the hierarchy
    * euler_frames: np.ndarray
        array of pose rotation parameters in euler angles
    * frame_time: float
        time in seconds for the display of each keyframe
    """
    bvh_string = _generate_hierarchy_string(skeleton) + "\n"
    bvh_string += _generate_bvh_frame_string(euler_frames, frame_time)
    return bvh_string

def _generate_hierarchy_string(skeleton):
    """ Initiates the recursive generation of the skeleton structure string
        by calling _generate_joint_string with the root joint
    """
    hierarchy_string = "HIERARCHY\n"
    hierarchy_string += _generate_joint_string(skeleton.root, skeleton, 0)
    return hierarchy_string

def _generate_joint_string(joint, skeleton, joint_level):
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
            joint_string += _generate_joint_string(child.node_name, skeleton, joint_level)

    # close the bracket
    joint_string += tab_string + "}" + "\n"
    return joint_string

def _generate_bvh_frame_string(euler_frames, frame_time):
    """ Converts a list of euler frames into the BVH file representation.
        Parameters
        ----------
        * euler_frames: np.ndarray
            array of pose rotation parameters in euler angles
        * frame_time: float
            time in seconds for the display of each keyframe
    """
    frame_parameter_string = "MOTION\n"
    frame_parameter_string += "Frames: " + str(len(euler_frames)) + "\n"
    frame_parameter_string += "Frame Time: " + str(frame_time) + "\n"
    for frame in euler_frames:
        frame_parameter_string += ' '.join([str(f) for f in frame])
        frame_parameter_string += '\n'
    return frame_parameter_string


def convert_quaternion_to_euler_frames(skeleton, quat_frames):
    """ Converts the joint rotations from quaternion to euler rotations
    Parameters
    ----------
    * skeleton: Skeleton
        Skeleton structure needed to copy the hierarchy
    * quat_frames: np.ndarray
        array of pose rotation parameters as quaternion 
    """
    joint_order = []
    rotation_info = dict()
    def get_joint_meta_info(joint_name, skeleton):
        if len(skeleton.nodes[joint_name].children) > 0:
            joint_order.append(joint_name)
        rot_order = []
        offset = 0
        for idx, ch in enumerate(skeleton.nodes[joint_name].channels):
            if ch.lower().endswith("rotation"):
                rot_order.append(ch)
            else:
                offset += 1
        rotation_info[joint_name] = rot_order, offset
        for child in skeleton.nodes[joint_name].children:
            get_joint_meta_info(child.node_name, skeleton)

    get_joint_meta_info(skeleton.root, skeleton)
    n_frames = len(quat_frames)
    n_params = sum([len(skeleton.nodes[j].channels) for j in joint_order])
    euler_frames = np.zeros((n_frames, n_params))
    for frame_idx, quat_frame in enumerate(quat_frames):
        euler_frames[frame_idx,:TRANSLATION_LEN] = quat_frame[:TRANSLATION_LEN]
        dst = 0 # the translation offset will be added
        for joint_name in joint_order:
            channels = skeleton.nodes[joint_name].channels
            n_channels = len(channels)
            rotation_order = rotation_info[joint_name][0]
            rotation_offset = rotation_info[joint_name][1]
            src = skeleton.nodes[joint_name].index * QUAT_LEN + TRANSLATION_LEN
            q = quat_frame[src:src+QUAT_LEN]
            e = quaternion_to_euler(q, rotation_order)
            params_start = dst + rotation_offset
            params_end = params_start + EULER_LEN
            euler_frames[frame_idx, params_start:params_end] = e
            dst += n_channels
    return euler_frames


class BVHWriter(object):
    """Saves an input motion defined either as an array of euler or quaternion
    frame vectors to a BVH file.
    Legacy interface that calls write_euler_frames_to_bvh_file

    Parameters
    ----------
    * filename: String or None
        Name of the created bvh file. Can be None.
    * skeleton: Skeleton
        Skeleton structure needed to copy the hierarchy
    * frame_data: np.ndarray
        array of motion vectors, either with euler or quaternion as rotation parameters
    * frame_time: float
        time in seconds for the display of each keyframe
    * is_quaternion: Boolean
        Defines wether the frame_data is quaternion data or euler data
    """
    def __init__(self, filename, skeleton, frame_data, frame_time, is_quaternion=False):
        self.skeleton = skeleton
        self.frame_data = frame_data
        self.frame_time = frame_time
        if is_quaternion:
            self.frame_data = convert_quaternion_to_euler_frames(self.skeleton, self.frame_data)
        if filename is not None:
            self.write(filename)

    def write(self, filename):
        write_euler_frames_to_bvh_file(filename, self.skeleton, self.frame_data, self.frame_time)
