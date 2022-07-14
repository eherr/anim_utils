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
from datetime import datetime
import os
import numpy as np
import imp
from transformations import quaternion_inverse, quaternion_multiply, quaternion_slerp, quaternion_from_euler, euler_matrix, quaternion_from_matrix
from .quaternion_frame import convert_euler_frames_to_quaternion_frames
from .motion_concatenation import align_and_concatenate_frames, smooth_root_positions
from .constants import ROTATION_TYPE_QUATERNION, ROTATION_TYPE_EULER
from .bvh import BVHWriter
from .acclaim import create_euler_matrix, DEFAULT_FRAME_TIME


def get_quaternion_delta(a, b):
    return quaternion_multiply(quaternion_inverse(b), a)


def add_frames(skeleton, a, b):
    """ returns c = a + b"""
    c = np.zeros(len(a))
    c[:3] = a[:3] + b[:3]
    for idx, j in enumerate(skeleton.animated_joints):
        o = idx * 4 + 3
        q_a = a[o:o + 4]
        q_b = b[o:o + 4]
        q_prod = quaternion_multiply(q_a, q_b)
        c[o:o + 4] = q_prod / np.linalg.norm(q_prod)
    return c


def substract_frames(skeleton, a, b):
    """ returns c = a - b"""
    c = np.zeros(len(a))
    c[:3] = a[:3] - b[:3]
    for idx, j in enumerate(skeleton.animated_joints):
        o = idx*4 + 3
        q_a = a[o:o+4]
        q_b = b[o:o+4]
        q_delta = get_quaternion_delta(q_a, q_b)
        c[o:o+4] = q_delta / np.linalg.norm(q_delta)
    return c

def amc_euler_to_quaternion(angles, dofs, c, c_inv, align_quat=True):
    """ https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html 
    """
    m = create_euler_matrix(angles, dofs)
    #apply joint coordinate system defined by asf skeleton: C_inv x M x C 
    m = np.dot(c, np.dot(m, c_inv)) 
    q = quaternion_from_matrix(m)
    if align_quat:
        dot = np.sum(q)
        if dot < 0:
            q = -q
    return q.tolist()

class MotionVector(object):
    """ Contains a list of skeleton animation frames. Each frame represents a list of parameters for the degrees of freedom of a skeleton.
    """
    def __init__(self, skeleton=None, algorithm_config=None, rotation_type=ROTATION_TYPE_QUATERNION):
        self.n_frames = 0
        self._prev_n_frames = 0
        self.frames = None
        self.start_pose = None
        self.rotation_type = rotation_type
        self.apply_spatial_smoothing = False
        self.apply_foot_alignment = False
        self.smoothing_window = 0
        self.spatial_smoothing_method = "smoothing"
        self.frame_time = 1.0/30.0
        self.skeleton = skeleton

        if algorithm_config is not None:
            settings = algorithm_config["smoothing_settings"]
            self.apply_spatial_smoothing = settings["spatial_smoothing"]
            self.smoothing_window = settings["spatial_smoothing_window"]

            if "spatial_smoothing_method" in settings:
                self.spatial_smoothing_method = settings["spatial_smoothing_method"]
            if "apply_foot_alignment" in settings:
                self.apply_foot_alignment = settings["apply_foot_alignment"]

    def from_bvh_reader(self, bvh_reader, filter_joints=True, animated_joints=None):
        if self.rotation_type == ROTATION_TYPE_QUATERNION:
            quat_frame = convert_euler_frames_to_quaternion_frames(bvh_reader, bvh_reader.frames, filter_joints, animated_joints)
            self.frames = np.array(quat_frame)
        elif self.rotation_type == ROTATION_TYPE_EULER:
            self.frames = bvh_reader.frames
        self.n_frames = len(self.frames)
        self._prev_n_frames = 0
        self.frame_time = bvh_reader.frame_time

    def get_relative_frames(self):
        relative_frames = []
        for idx in range(1, len(self.frames)):
            delta_frame = np.zeros(len(self.frames[0]))
            delta_frame[7:] = self.frames[idx][7:]
            delta_frame[:3] = self.frames[idx][:3] - self.frames[idx-1][:3]
            currentq = self.frames[idx][3:7] / np.linalg.norm(self.frames[idx][3:7])
            prevq = self.frames[idx-1][3:7] / np.linalg.norm(self.frames[idx-1][3:7])
            delta_q = quaternion_multiply(quaternion_inverse(prevq), currentq)
            delta_frame[3:7] = delta_q
            #print(idx, self.frames[idx][:3], self.frames[idx - 1][:3], delta_frame[:3], delta_frame[3:7])

            relative_frames.append(delta_frame)
        return relative_frames

    def append_frames_generic(self, new_frames):
        """Align quaternion frames to previous frames

        Parameters
        ----------
        * new_frames: list
            A list of frames with the same rotation format type as the motion vector
        """
        if self.apply_spatial_smoothing:
            smoothing_window = self.smoothing_window
        else:
            smoothing_window = 0
        self.frames = align_and_concatenate_frames(self.skeleton, self.skeleton.aligning_root_node, new_frames, self.frames, self.start_pose,
                                                   smoothing_window=smoothing_window, blending_method=self.spatial_smoothing_method)

        self.n_frames = len(self.frames)
        self._prev_n_frames = self.n_frames


    def append_frames_using_forward_blending(self, new_frames):
        if self.apply_spatial_smoothing:
            smoothing_window = self.smoothing_window
        else:
            smoothing_window = 0
        from . import motion_concatenation
        imp.reload(motion_concatenation)
        ik_chains = self.skeleton.skeleton_model["ik_chains"]
        self.frames = motion_concatenation.align_frames_using_forward_blending(self.skeleton, self.skeleton.aligning_root_node, new_frames,
                                                                               self.frames, self._prev_n_frames, self.start_pose,
                                                                               ik_chains, smoothing_window)
        self._prev_n_frames = self.n_frames
        self.n_frames = len(self.frames)

    def append_frames(self, new_frames, plant_foot=None):
        if self.apply_foot_alignment and self.skeleton.skeleton_model is not None:
            self.append_frames_using_forward_blending(new_frames)
        else:
            self.append_frames_generic(new_frames)

    def export(self, skeleton, output_filename, add_time_stamp=False):
        bvh_writer = BVHWriter(None, skeleton, self.frames, self.frame_time, True)
        if add_time_stamp:
            output_filename = output_filename + "_" + \
                       str(datetime.now().strftime("%d%m%y_%H%M%S")) + ".bvh"
        elif output_filename != "":
            if not output_filename.endswith("bvh"):
                output_filename = output_filename + ".bvh"
        else:
            output_filename = "output.bvh"
        bvh_writer.write(output_filename)

    def reduce_frames(self, n_frames):
        if n_frames == 0:
            self.frames = None
            self.n_frames = 0
            self._prev_n_frames = self.n_frames
        else:
            self.frames = self.frames[:n_frames]
            self.n_frames = len(self.frames)
            self._prev_n_frames = 0

    def has_frames(self):
        return self.frames is not None

    def clear(self, end_frame=0):
        if end_frame == 0:
            self.frames = None
            self.n_frames = 0
            self._prev_n_frames = 0
        else:
            self.frames = self.frames[:end_frame]
            self.n_frames = len(self.frames)
            self._prev_n_frames = 0

    def translate_root(self, offset):
        for idx in range(self.n_frames):
            self.frames[idx][:3] += offset

    def scale_root(self, scale_factor):
        for idx in range(self.n_frames):
            self.frames[idx][:3] *= scale_factor

    def from_fbx(self, animation, animated_joints):
        self.frame_time = animation["frame_time"]
        root_joint = animated_joints[0]
        self.n_frames = len(animation["curves"][root_joint])
        self.frames = []
        for idx in range(self.n_frames):
            frame = self._create_frame_from_fbx(animation, animated_joints, idx)
            self.frames.append(frame)

    def _create_frame_from_fbx(self, animation, animated_joints, idx):
        n_dims = len(animated_joints) * 4 + 3
        frame = np.zeros(n_dims)
        offset = 3
        root_name = animated_joints[0]
        frame[:3] = animation["curves"][root_name][idx]["translation"]
        for node_name in animated_joints:
            if node_name in animation["curves"].keys():
                rotation = animation["curves"][node_name][idx]["rotation"]
                frame[offset:offset+4] = rotation
            offset += 4

        return frame

    def to_unity_format(self, scale=1.0):
        """ Converts the frames into a custom json format for use in a Unity client"""
        animated_joints = [j for j, n in list(self.skeleton.nodes.items()) if
                           "EndSite" not in j and len(n.children) > 0]  # self.animated_joints
        unity_frames = []

        for node in list(self.skeleton.nodes.values()):
            node.quaternion_index = node.index

        for frame in self.frames:
            unity_frame = self._convert_frame_to_unity_format(frame, animated_joints, scale)
            unity_frames.append(unity_frame)

        result_object = dict()
        result_object["frames"] = unity_frames
        result_object["frameTime"] = self.frame_time
        result_object["jointSequence"] = animated_joints
        return result_object

    def to_db_format(self, scale=1.0, animated_joints=None):
        """ Converts the frames into a custom json format for use in a Unity client"""
        if animated_joints is None:
            animated_joints = [j for j, n in self.skeleton.nodes.items() if
                               n.children is not None and len(n.children) > 0]  # self.animated_joints
        poses = []
        for frame in self.frames:
            pose = self._convert_frame_to_db_format(frame, animated_joints, scale)
            poses.append(pose)

        result_object = dict()
        result_object["poses"] = poses
        result_object["frame_time"] = self.frame_time
        result_object["joint_sequence"] = animated_joints
        return result_object

    def _convert_frame_to_db_format(self, frame, animated_joints, scale=1.0):
        """ Converts the frame into a custom json format and converts the transformations
            to the left-handed coordinate system of Unity.
            src: http://answers.unity3d.com/questions/503407/need-to-convert-to-right-handed-coordinates.html
        """
        
        t = frame[:3] * scale
        pose = [-t[0], t[1], t[2]]
        for node_name in self.skeleton.nodes.keys():
            if node_name in animated_joints:
                node = self.skeleton.nodes[node_name]
                if node_name in self.skeleton.animated_joints:  # use rotation from frame
                    # TODO fix: the animated_joints is ordered differently than the nodes list for the latest model
                    index = self.skeleton.animated_joints.index(node_name)
                    offset = index * 4 + 3
                    r = frame[offset:offset + 4]
                    pose += [-r[0], -r[1],  r[2], r[3]]
                else:  # use fixed joint rotation
                    r = node.rotation
                    pose += [-float(r[0]), -float(r[1]), float(r[2]), float(r[3])]
        return pose


    def _convert_frame_to_unity_format(self, frame, animated_joints, scale=1.0):
        """ Converts the frame into a custom json format and converts the transformations
            to the left-handed coordinate system of Unity.
            src: http://answers.unity3d.com/questions/503407/need-to-convert-to-right-handed-coordinates.html
        """
        unity_frame = {"rotations": [], "rootTranslation": None}
        for node_name in self.skeleton.nodes.keys():
            if node_name in animated_joints:
                node = self.skeleton.nodes[node_name]
                if node_name == self.skeleton.root:
                    t = frame[:3] * scale
                    unity_frame["rootTranslation"] = {"x": -t[0], "y": t[1], "z": t[2]}

                if node_name in self.skeleton.animated_joints:  # use rotation from frame
                    # TODO fix: the animated_joints is ordered differently than the nodes list for the latest model
                    index = self.skeleton.animated_joints.index(node_name)
                    offset = index * 4 + 3
                    r = frame[offset:offset + 4]
                    unity_frame["rotations"].append({"x": -r[1], "y": r[2], "z": r[3], "w": -r[0]})
                else:  # use fixed joint rotation
                    r = node.rotation
                    unity_frame["rotations"].append(
                        {"x": -float(r[1]), "y": float(r[2]), "z": float(r[3]), "w": -float(r[0])})
        return unity_frame

    def from_custom_db_format(self, data):
        self.frames = []
        for f in data["poses"]:
            t = f[:3]
            o = 3
            new_f = [-t[0], t[1], t[2]]
            for i in data["joint_sequence"]:
                new_f.append(-f[o])
                new_f.append(-f[o+1])
                new_f.append(f[o+2])
                new_f.append(f[o+3])
                o+=4

            self.frames.append(new_f)
        self.frames = np.array(self.frames)
        self.n_frames = len(self.frames)
        self.frame_time = data["frame_time"]

    def from_custom_unity_format(self, data):
        self.frames = []
        for f in data["frames"]:
            t = f["rootTranslation"]
            new_f = [-t["x"], t["y"], t["z"]]
            for q in f["rotations"]:
                new_f.append(-q["w"])
                new_f.append(-q["x"])
                new_f.append(q["y"])
                new_f.append(q["z"])

            self.frames.append(new_f)
        self.frames = np.array(self.frames)
        self.n_frames = len(self.frames)
        self.frame_time = data["frameTime"]

    def apply_low_pass_filter_on_root(self, window):
        self.frames[:, :3] = smooth_root_positions(self.frames[:, :3], window)

    def apply_delta_frame(self, skeleton, delta_frame):
        for f in range(self.n_frames):
            self.frames[f] = add_frames(skeleton, self.frames[f], delta_frame)

    def interpolate(self, start_idx, end_idx, t):
        new_frame = np.zeros(self.frames[0].shape)
        new_frame[:3] = (1-t) * self.frames[start_idx][:3] + t * self.frames[end_idx][:3]
        for i in range(3, new_frame.shape[0], 4):
            start_q = self.frames[start_idx][i:i+4]
            end_q = self.frames[end_idx][i:i+4]
            new_frame[i:i+4] = quaternion_slerp(start_q, end_q, t, spin=0, shortestpath=True)
        return new_frame

    def from_amc_data(self, skeleton, amc_frames, frame_time=DEFAULT_FRAME_TIME):
        identity = np.eye(4, dtype=np.float32)
        self.frames = []
        for f in amc_frames:
            new_f = []
            for key in skeleton.animated_joints:
                b_data = skeleton.nodes[key].format_meta_data
                if "coordinate_transform" in b_data and "inv_coordinate_transform" in b_data:
                    c = b_data["coordinate_transform"]
                    c_inv = b_data["inv_coordinate_transform"]
                else:
                    c = identity
                    c_inv = identity
                if key == "root":
                    new_f += f["root"][:3]
                    dof = b_data["asf_channels"]
                    values = amc_euler_to_quaternion(f[key][3:], dof, c, c_inv)
                    new_f += values
                elif key in f:
                    dof = b_data["asf_channels"]
                    new_f += amc_euler_to_quaternion(f[key], dof, c, c_inv)
                else:
                    new_f += [1,0,0,0]
            self.frames.append(new_f)
        self.frames = np.array(self.frames)
        self.n_frames = len(self.frames)
        self.frame_time = frame_time
