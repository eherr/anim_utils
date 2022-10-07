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
import collections
from copy import deepcopy
import json
import numpy as np
from .skeleton import Skeleton
from .skeleton_node import SkeletonRootNode, SkeletonJointNode, SkeletonEndSiteNode, SKELETON_NODE_TYPE_JOINT, SKELETON_NODE_TYPE_END_SITE
from .quaternion_frame import convert_euler_to_quaternion_frame
from .joint_constraints import HingeConstraint2
from .acclaim import asf_to_bvh_channels


DEFAULT_ROOT_DIR = [0, 0, 1]


def create_identity_frame(skeleton):
    skeleton.identity_frame = np.zeros(skeleton.reference_frame_length)
    offset = 3
    for j in skeleton.animated_joints:
        skeleton.identity_frame[offset:offset + 4] = [1, 0, 0, 0]
        offset += 4


def create_euler_frame_indices(skeleton):
    nodes_without_endsite = [node for node in list(skeleton.nodes.values()) if node.node_type != SKELETON_NODE_TYPE_END_SITE]
    for node in nodes_without_endsite:
        node.euler_frame_index = nodes_without_endsite.index(node)


def read_reference_frame_from_bvh_reader(bvh_reader, frame_index=0):
    quat_frame = convert_euler_to_quaternion_frame(bvh_reader, bvh_reader.frames[frame_index], False, animated_joints=None)
    return np.array(bvh_reader.frames[0][:3].tolist() + quat_frame.tolist())


def add_tool_nodes(skeleton, node_names, new_tool_bones):
    for b in new_tool_bones:
        add_new_end_site(skeleton, node_names, b["parent_node_name"], b["new_node_offset"])
        skeleton.tool_nodes.append(b["new_node_name"])


def add_new_end_site(skeleton, node_names, parent_node_name, offset):
    if parent_node_name in list(node_names.keys()):
        level = node_names[parent_node_name]["level"] + 1
        node_desc = dict()
        node_desc["level"] = level
        node_desc["offset"] = offset

def reference_frame_from_unity(data):
    n_j = len(data["rotations"])
    q_frame = np.zeros(n_j*4+3)
    q_frame[:3] = arr_from_unity_t(data["translations"][0])
    o = 3
    for q in data["rotations"]:
        q_frame[o:o+4] = arr_from_unity_q(q)
        o+=4
    return q_frame


def generate_reference_frame(skeleton, animated_joints):
    identity_frame = [0,0,0]
    frame = [0, 0, 0]
    joint_idx = 0
    node_list = [(skeleton.nodes[n].index, n) for n in skeleton.nodes.keys() if skeleton.nodes[n].index >= 0]
    node_list.sort()
    for idx, node in node_list:
        frame += list(skeleton.nodes[node].rotation)
        if node in animated_joints:
            identity_frame += [1.0,0.0,0.0,0.0]
            skeleton.nodes[node].quaternion_frame_index = joint_idx
            joint_idx += 1
        else:
            skeleton.nodes[node].quaternion_frame_index = -1
    skeleton.reference_frame = np.array(frame)
    skeleton.reference_frame_length = len(frame)
    skeleton.identity_frame = np.array(identity_frame)


def arr_from_unity_q(_q):
    q = np.zeros(4)
    q[0] = - _q["w"]
    q[1] = - _q["x"]
    q[2] = _q["y"]
    q[3] = _q["z"]
    return q


def arr_from_unity_t(_t):
    t = np.zeros(3)
    t[0] = - _t["x"]
    t[1] = _t["y"]
    t[2] = _t["z"]
    return t

class SkeletonBuilder(object):

    def load_from_bvh(self, bvh_reader, animated_joints=None, reference_frame=None, skeleton_model=None, tool_bones=None):
        skeleton = Skeleton()
        if animated_joints is None:
            animated_joints = list(bvh_reader.get_animated_joints())
        skeleton.animated_joints = animated_joints
        skeleton.frame_time = bvh_reader.frame_time
        skeleton.root = bvh_reader.root
        skeleton.aligning_root_dir = DEFAULT_ROOT_DIR
        if reference_frame is None:
            skeleton.reference_frame = read_reference_frame_from_bvh_reader(bvh_reader)
        else:
            skeleton.reference_frame = reference_frame
        skeleton.reference_frame_length = len(skeleton.reference_frame)
        skeleton.tool_nodes = []
        if tool_bones is not None:
            add_tool_nodes(skeleton, bvh_reader.node_names, tool_bones)
        skeleton.nodes = collections.OrderedDict()
        joint_list = list(bvh_reader.get_animated_joints())
        self.construct_hierarchy_from_bvh(skeleton, joint_list, bvh_reader.node_names, skeleton.root, 0)

        create_euler_frame_indices(skeleton)
        SkeletonBuilder.set_meta_info(skeleton)

        if skeleton_model is not None:
            skeleton.skeleton_model = skeleton_model
            #skeleton.add_heels(skeleton_model)
        return skeleton

    def construct_hierarchy_from_bvh(self, skeleton, joint_list, node_info, node_name, level):
        
        channels = []
        if "channels" in node_info[node_name]:
            channels = node_info[node_name]["channels"]
        is_fixed = True
        quaternion_frame_index = -1
        if node_name in skeleton.animated_joints:
            is_fixed = False
            quaternion_frame_index = skeleton.animated_joints.index(node_name)
        joint_index = -1
        if node_name in joint_list:
            joint_index = joint_list.index(node_name)
        if node_name == skeleton.root:
            node = SkeletonRootNode(node_name, channels, None, level)
        elif "children" in list(node_info[node_name].keys()) and len(node_info[node_name]["children"]) > 0:
            node = SkeletonJointNode(node_name, channels, None, level)
            offset = joint_index * 4 + 3
            node.rotation = skeleton.reference_frame[offset: offset + 4]
        else:
            node = SkeletonEndSiteNode(node_name, channels, None, level)
        node.fixed = is_fixed
        node.quaternion_frame_index = quaternion_frame_index
        node.index = joint_index
        node.offset = node_info[node_name]["offset"]
        skeleton.nodes[node_name] = node
        if "children" in node_info[node_name]:
            for c in node_info[node_name]["children"]:
                c_node = self.construct_hierarchy_from_bvh(skeleton, joint_list, node_info, c, level+1)
                c_node.parent = node
                node.children.append(c_node)
        return node

    def load_from_json_file(self, filename):
        with open(filename) as infile:
            data = json.load(infile)
            skeleton = self.load_from_custom_unity_format(data)
            return skeleton

    def load_from_custom_unity_format(self, data, frame_time=1.0/30, add_extra_end_site=False):
        skeleton = Skeleton()
        animated_joints = data["jointSequence"]
        if len(animated_joints) == 0:
            print("Error no joints defined")
            return skeleton

        print("load from json", len(animated_joints))
        skeleton.animated_joints = animated_joints

        skeleton.skeleton_model = collections.OrderedDict()
        skeleton.skeleton_model["joints"] = dict()
        if "head_joint" in data:
            skeleton.skeleton_model["joints"]["head"] = data["head_joint"]
        if "neck_joint" in data:
            skeleton.skeleton_model["joints"]["neck"] = data["neck_joint"]

        skeleton.frame_time = frame_time
        skeleton.nodes = collections.OrderedDict()
        skeleton.root = animated_joints[0]
        root = self._create_node_from_unity_desc(skeleton, skeleton.root, data, None, 0, add_extra_end_site)

        SkeletonBuilder.set_meta_info(skeleton)
        create_euler_frame_indices(skeleton)
        if "root" in data:
            skeleton.aligning_root_node = data["root"]
        else:
            skeleton.aligning_root_node = skeleton.root
        skeleton.aligning_root_dir = DEFAULT_ROOT_DIR

        skeleton.reference_frame = reference_frame_from_unity(data["referencePose"])
        print("reference", skeleton.reference_frame[3:7],data["referencePose"]["rotations"][0])
        skeleton.reference_frame_length = len(skeleton.reference_frame)
        return skeleton

    def load_from_json_data(self, data, animated_joints=None, use_all_joints=False):
        def extract_animated_joints(node, animated_joints):
            animated_joints.append(node["name"])
            for c in node["children"]:
                if c["index"] >= 0 and len(c["children"]) > 0:
                    extract_animated_joints(c, animated_joints)

        skeleton = Skeleton()
        print("load from json")
        if animated_joints is not None:
            skeleton.animated_joints = animated_joints
        elif "animated_joints" in data:
            skeleton.animated_joints = data["animated_joints"]
        else:
            animated_joints = list()
            extract_animated_joints(data["root"], animated_joints)
            skeleton.animated_joints = animated_joints

        skeleton.free_joints_map = data.get("free_joints_map", dict())
        if "skeleton_model" in data:
            skeleton.skeleton_model = data["skeleton_model"]
        else:
            skeleton.skeleton_model = collections.OrderedDict()
            skeleton.skeleton_model["joints"] = dict()
            if "head_joint" in data:
                skeleton.skeleton_model["joints"]["head"] = data["head_joint"]
            if "neck_joint" in data:
                skeleton.skeleton_model["joints"]["neck"] = data["neck_joint"]

        skeleton.frame_time = data["frame_time"]
        skeleton.nodes = collections.OrderedDict()
        root = self._create_node_from_desc(skeleton, data, data["root"]["name"], None, 0)
        skeleton.root = root.node_name
        if "tool_nodes" in data:
            skeleton.tool_nodes = data["tool_nodes"]
        create_euler_frame_indices(skeleton)
        skeleton.aligning_root_node = data.get("aligning_root_node",skeleton.root)
        skeleton.aligning_root_dir = data.get("aligning_root_dir",DEFAULT_ROOT_DIR)
        if "reference_frame" in data:
            skeleton.reference_frame = data["reference_frame"]
            skeleton.reference_frame_length = len(skeleton.reference_frame)
        if skeleton.reference_frame is None or use_all_joints:
            generate_reference_frame(skeleton, skeleton.animated_joints)
        SkeletonBuilder.set_meta_info(skeleton)
        return skeleton

    def load_from_fbx_data(self, data):
        skeleton = Skeleton()
        skeleton.nodes = collections.OrderedDict()

        skeleton.animated_joints = data["animated_joints"]
        skeleton.root = data["root"]
        self._create_node_from_desc(skeleton, data, skeleton.root, None, 0)
        skeleton.frame_time = data["frame_time"]
        SkeletonBuilder.set_meta_info(skeleton)
        return skeleton

    def _create_node_from_desc(self, skeleton, data, node_name, parent, level=0):
        node_data = data["nodes"][node_name]
        channels = node_data["channels"]
        if parent is None:
            node = SkeletonRootNode(node_name, channels, parent, level)
        elif node_data["node_type"] == SKELETON_NODE_TYPE_JOINT:
            node = SkeletonJointNode(node_name, channels, parent, level)
        else:
            node = SkeletonEndSiteNode(node_name, channels, parent, level)
        node.fixed = node_data["fixed"]
        node.index = node_data["index"]
        node.offset = np.array(node_data["offset"])
        node.rotation = np.array(node_data["rotation"])
        if node_name in skeleton.animated_joints:
            node.quaternion_frame_index = skeleton.animated_joints.index(node_name)
            node.fixed = False
        else:
            node.quaternion_frame_index = -1
            node.fixed = True
        node.children = []
        skeleton.nodes[node_name] = node
        for c_name in node_data["children"]:
            c_node = self._create_node_from_desc(skeleton, data, c_name, node, level+1)
            node.children.append(c_node)
        
        return node

    def get_joint_desc(self, data, name):
        for idx in range(len(data["jointDescs"])):
            if data["jointDescs"][idx]["name"] == name:
                return data["jointDescs"][idx]
        return None

    def _create_node_from_unity_desc(self, skeleton, node_name, data, parent, level, add_extra_end_site=False):
        node_data = self.get_joint_desc(data, node_name)
        if node_data is None:
            return

        if parent is None:
            channels = ["Xposition","Yposition","Zposition", "Xrotation","Yrotation","Zrotation"]
        else:# len(node_data["children"]) > 0:
            channels = ["Xrotation","Yrotation","Zrotation"]

        if parent is None:
            node = SkeletonRootNode(node_name, channels, parent, level)
        else:# len(node_data["children"]) > 0:
            node = SkeletonJointNode(node_name, channels, parent, level)
        node.offset = np.array(node_data["offset"])
        node.offset[0] *= -1
        node.rotation = np.array(node_data["rotation"])
        node.rotation[0] *= -1
        node.rotation[1] *= -1
       
        if node_name in skeleton.animated_joints:
            node.quaternion_frame_index = skeleton.animated_joints.index(node_name)
            node.fixed = False
        else:
            node.quaternion_frame_index = -1
            node.fixed = True

        skeleton.nodes[node_name] = node
        skeleton.nodes[node_name].children = []
        if len(node_data["children"]) > 0:
            node.index = node.quaternion_frame_index
            for c_name in node_data["children"]:
                c_node = self._create_node_from_unity_desc(skeleton, c_name, data, node, level+1)
                if c_node is not None:
                    skeleton.nodes[node_name].children.append(c_node)
        if add_extra_end_site:
            print("add extra end site")
            node.index = -1
            channels = []
            c_name = node_name+"_EndSite"
            c_node = SkeletonEndSiteNode(c_name, channels, node,  level+1)
            skeleton.nodes[node_name].children.append(c_node)
            skeleton.nodes[c_name] = c_node
        return node


    @classmethod
    def construct_arm_with_constraints(cls, n_joints, length):
        skeleton = Skeleton()
        skeleton.frame_time = 1 / 30
        animated_joints = []
        animated_joints.append("root")
        skeleton.root = "root"
        channels = ["rotationX", "rotationY", "rotationZ", "rotationW"]
        node = SkeletonRootNode("root", channels, None, 0)
        node.fixed = False
        node.index = 0
        node.offset = [0, 0, 0]
        node.rotation = [1, 0, 0, 0]
        node.quaternion_frame_index = 0
        skeleton.nodes["root"] = node
        parent = node
        swing_axis = np.array([0,0,1])
        twist_axis = np.array([0, 1, 0])

        angle_range = [0,90]
        for n in range(1, n_joints + 1):  # start after the root joint and add one endsite
            if n + 1 < n_joints + 1:
                node_name = "joint" + str(n)
                node = SkeletonJointNode(node_name, channels, parent, n)
                animated_joints.append(node_name)
                node.fixed = False
                node.quaternion_frame_index = n
                node.index = n
                node.offset = np.array([0, length, 0], dtype=np.float)
                print("create", node_name, node.offset)
                if n in [1]:
                    node.joint_constraint = HingeConstraint2(swing_axis, twist_axis)
            else:
                node_name = "joint" + str(n - 1) + "_EndSite"
                node = SkeletonEndSiteNode(node_name, channels, parent, n)
                node.fixed = True
                node.quaternion_frame_index = -1
                node.index = -1
                node.offset = np.array([0, 0, 0], dtype=np.float)
                print("create", node_name, node.offset)
            parent.children.append(node)

            node.rotation = [1, 0, 0, 0]
            skeleton.nodes[node_name] = node
            parent = node

        skeleton.animated_joints = animated_joints
        SkeletonBuilder.set_meta_info(skeleton)
        return skeleton


    @classmethod
    def get_reference_frame(cls, animated_joints):
        n_animated_joints = len(animated_joints)
        reference_frame = np.zeros(n_animated_joints * 4 + 3)
        o = 3
        for n in range(n_animated_joints):
            reference_frame[o:o + 4] = [1, 0, 0, 0]
            o += 4
        return reference_frame



    def load_from_asf_data(self, data, frame_time=1.0/30):
        skeleton = Skeleton()
        animated_joints = ["root"]+list(data["bones"].keys())

        print("load from asf", len(animated_joints))
        skeleton.animated_joints =animated_joints
        skeleton.skeleton_model = collections.OrderedDict()
        skeleton.skeleton_model["joints"] = dict()

        skeleton.frame_time = frame_time
        skeleton.nodes = collections.OrderedDict()
        skeleton.root = "root"
        if skeleton.root is None:
            skeleton.root = animated_joints[0]
        root = self._create_node_from_asf_data(skeleton, skeleton.root, data, None, 0)

        SkeletonBuilder.set_meta_info(skeleton)
        return skeleton

    def _create_node_from_asf_data(self, skeleton, node_name, data, parent, level):
        if parent is None:
            channels = asf_to_bvh_channels(data["root"]["order"])
            node = SkeletonRootNode(node_name, channels, parent, level)
            node.format_meta_data = data["root"]
            node.quaternion_frame_index = 0
            node.fixed = False
            node.format_meta_data["asf_channels"] = deepcopy(node.rotation_order)
        elif "dof" in data["bones"][node_name]:
            channels = asf_to_bvh_channels(data["bones"][node_name]["dof"])
            asf_channels = deepcopy(channels)
            if len(channels) < 3:
                channels = ['Xrotation','Yrotation','Zrotation']
            node = SkeletonJointNode(node_name, channels, parent, level)
            node.format_meta_data = data["bones"][node_name]
            node.format_meta_data["asf_channels"] = asf_channels
            if "offset" in data["bones"][node_name]:
                node.offset = data["bones"][node_name]["offset"]
            #if parent.node_name is not "root":
            #    node.offset = np.array(data["bones"][parent.node_name]["direction"])
            #    node.offset *= data["bones"][parent.node_name]["length"]
            if node_name in skeleton.animated_joints:
                node.quaternion_frame_index = skeleton.animated_joints.index(node_name)
                node.fixed = False
        else:
            channels = []
            node = SkeletonJointNode(node_name, channels, parent, level)
            node.format_meta_data = data["bones"][node_name]
            node.quaternion_frame_index = -1
            node.fixed = True
            node.format_meta_data["asf_channels"] = channels

        n_children = 0
        if node_name in data["children"]:
            n_children = len(data["children"][node_name])

        skeleton.nodes[node_name] = node
        skeleton.nodes[node_name].children = []
        if n_children > 0:
            node.index = node.quaternion_frame_index
            for c_name in data["children"][node_name]:
                c_node = self._create_node_from_asf_data(skeleton, c_name, data, node, level+1)
                if c_node is not None:
                    skeleton.nodes[node_name].children.append(c_node)
        else:
            node.index = node.quaternion_frame_index
            channels = []
            end_site_name = node_name +"EndSite"
            end_site_node = SkeletonJointNode(end_site_name, channels, node, level+1)
            end_site_node.quaternion_frame_index = -1
            end_site_node.fixed = True
            skeleton.nodes[end_site_name] = end_site_node
            skeleton.nodes[end_site_name].children = []
            skeleton.nodes[node_name].children.append(end_site_node)
        return node


    @classmethod
    def set_meta_info(cls, skeleton):
        skeleton.max_level = skeleton._get_max_level()
        skeleton._set_joint_weights()
        skeleton.parent_dict = skeleton._get_parent_dict()
        skeleton._chain_names = skeleton._generate_chain_names()
        skeleton.aligning_root_node = skeleton.root
        if skeleton.reference_frame is None:
            skeleton.reference_frame = SkeletonBuilder.get_reference_frame(skeleton.animated_joints)
            skeleton.reference_frame_length = len(skeleton.reference_frame)
        create_identity_frame(skeleton)
