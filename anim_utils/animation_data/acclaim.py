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
""" parser of asf and amc formats
    https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html
 """  
import sys
import math
import numpy as np
from transformations import euler_matrix

DEFAULT_FRAME_TIME = 1.0/120

def asf_to_bvh_channels(dof):
    channels = []
    for d in dof:
        type_str = ""
        if d[0].lower() == "r":
            type_str = "rotation"
        elif d[0].lower() == "t":
            type_str = "translation"
        axis_str = d[-1].upper()
        channels.append(axis_str  + type_str)
    return channels

def rotate_around_x(alpha):
    #Note vectors represent columns
    cx = math.cos(alpha)
    sx = math.sin(alpha)
    m = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, cx , sx,0.0],
                    [0.0, -sx, cx,0.0],
                    [0.0,0.0,0.0,1.0]],np.float32)
    return m.T


def rotate_around_y(beta):
    #Note vectors represent columns
    cy = math.cos(beta)
    sy = math.sin(beta)
    m = np.array([[cy,0.0,-sy ,0.0],
                    [0.0,1.0,0.0,0.0],
                    [sy, 0.0, cy,0.0],
                    [0.0,0.0,0.0,1.0]],np.float32)
    return m.T


def rotate_around_z(gamma):
    #Note vectors represent columns
    cz = math.cos(gamma)
    sz = math.sin(gamma)
    m = np.array([[cz, sz,0.0,0.0],
                    [-sz, cz,0.0,0.0],
                    [0.0,0.0,1.0,0.0],
                    [0.0,0.0,0.0,1.0]],np.float32)
    return m.T


AXES = "rxyz"
def create_euler_matrix(angles, order):
    m = np.eye(4, dtype=np.float32)
    for idx, d in enumerate(order):
        a = np.radians(angles[idx])
        d = d[0].lower()
        local_rot = np.eye(4)
        if d =="x":
            local_rot = euler_matrix(a,0,0, AXES)
        elif d =="y":
            local_rot = euler_matrix(0,a,0, AXES)
        elif d =="z":
            local_rot = euler_matrix(0,0,a, AXES)
        m = np.dot(local_rot, m)
    return m

def create_euler_matrix2(angles, order):
    m = np.eye(4, dtype=np.float32)
    for idx, d in enumerate(order):
        a = np.radians(angles[idx])
        d = d[0].lower()
        local_rot = np.eye(4)
        if d =="x":
            local_rot = rotate_around_x(a)
        elif d =="y":
            local_rot = rotate_around_y(a)
        elif d =="z":
            local_rot = rotate_around_z(a)
        m = np.dot(local_rot, m)
    return m

def create_coordinate_transform_matrices(data):
    """ FROM https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html
    Precalculate some the transformation for each segment from the axis
    """
    angles = data["root"]["orientation"]
    order = data["root"]["axis"]
    order = asf_to_bvh_channels(order)
    c = create_euler_matrix(angles, order)
    data["root"]["coordinate_transform"] = c
    data["root"]["inv_coordinate_transform"] = np.linalg.inv(c)

    for key in data["bones"]:
        if "axis" in data["bones"][key]:
            angles, order = data["bones"][key]["axis"]
            order = asf_to_bvh_channels(order)
            c = create_euler_matrix(angles, order)
            data["bones"][key]["coordinate_transform"] = c
            data["bones"][key]["inv_coordinate_transform"] = np.linalg.inv(c)
    return data
    
def set_parents(data):
    for parent in data["children"]:
        for c in data["children"][parent]:
            data["bones"][c]["parent"] = parent
    return data

def convert_bones_to_joints(data):
    """ ASF stores bones that need to be converted into joints the internal skeleton class 
        based on the BVH format
        additionally helper bones for the root need to be removed
    """
    data = set_parents(data)
    for b in data["bones"]:
        if b == "root":
            continue
        # add offset to parent
        parent = data["bones"][b]["parent"]
        if parent in data["bones"] and "direction" in data["bones"][parent]:
            offset = np.array(data["bones"][parent]["direction"])
            offset *= float(data["bones"][parent]["length"])
            data["bones"][b]["offset"] = offset
    
    # remove helper bones for hips without dofs
    new_root_children = []
    old_root_children = []
    for c in data["children"]["root"]:
        if "dof" in data["bones"][c] and len(data["bones"][c]["dof"])>0:
            new_root_children.append(c)
        else: # replace bones without dofs with their children
            old_root_children.append(c)
            for cc in data["children"][c]:
                data["bones"][cc]["parent"] = "root"
                new_root_children.append(cc)
    data["children"]["root"] = new_root_children
    for c in old_root_children:
        del data["bones"][c]
    return data

def parse_asf_file(filepath):
    with open(filepath, "rt") as in_file:
        lines = in_file.readlines()
    data = dict()
    data["bones"] = dict()
    idx = 0
    while idx < len(lines):
        next_line = lines[idx].lstrip()
        if next_line.startswith(":root"):
            idx += 1
            data["root"], idx = read_root_data(lines, idx)
            idx -=1
        if next_line.startswith(":name"):
            data["name"] = next_line.split(" ")[1]
            idx+=1
        elif next_line.startswith(":bonedata"):
            idx+=1
            next_line = lines[idx].lstrip()
            while not next_line.startswith(":hierarchy") and idx+1 < len(lines):
                node, idx = read_bone_data(lines, idx)
                if "name" in node:
                    name = node["name"]
                    data["bones"][name] = node
                if idx < len(lines):
                    next_line = lines[idx].lstrip()
        elif next_line.startswith(":hierarchy"):
            data["children"], idx = read_hierarchy(lines, idx)
        else:
            idx+=1
    data = create_coordinate_transform_matrices(data)
    data = convert_bones_to_joints(data)
    return data

def read_root_data(lines, idx):
    data = dict()
    #print("start root", idx)
    next_line = lines[idx].strip()
    while not next_line.startswith(":bonedata") and idx+1 < len(lines):
        values = next_line.split(" ")
        if len(values) > 0:
            key = values[0]
            if key == "position":
                data["position"] =  [values[1], values[2], values[3]]
            elif key == "orientation":
                data["orientation"] =  [float(values[1]),float(values[2]), float(values[3])]
            elif key == "axis":
                data["axis"] =  values[1]
            elif key == "order":
                data["order"] =  [v for v in values if v != "order"]
            #elif key == "limits":
            #    data[key] =  [values[1], values[2], values[3]]
        if idx+1 < len(lines):
            idx+=1
            next_line = lines[idx].strip() # remove empty lines

    #print("end root", idx, next_line)
    return data, idx

def read_bone_data(lines, idx):
    idx +=1 #skip begin
    data = dict()
    #print("start bone", idx)
    next_line = lines[idx].strip()
    while not next_line.startswith("end") and idx+1 < len(lines):
        values = next_line.split(" ")
        values = [v for v in values if v != ""]
        if len(values) > 0:
            key = values[0]
            if key == "id":
                data["id"] = values[1]
            elif key == "name":
                data["name"] = values[1]
            elif key == "direction":
                direction =  np.array([float(v) for v in values if v != "direction"])
                direction /= np.linalg.norm(direction)
                data["direction"] = direction.tolist()
            elif key == "length":
                data["length"] =  float(values[1])
            elif key == "axis":
                data["axis"] =  [float(values[1]),  float(values[2]),  float(values[3])], values[4]
            elif key == "dof":
                data["dof"] =  [v for v in values if v != "dof"]
            #elif key == "limits":
            #    data[key] =  [values[1], values[2], values[3]]
        if idx+1 < len(lines):
            idx+=1
            next_line = lines[idx].strip() # remove empty lines
    idx +=1 #skip end
    #print("end", idx, lines[idx])
    return data, idx


def read_hierarchy(lines, idx):
    print("found hierarchy")
    idx +=1 #skip begin
    child_dict = dict()
    next_line = lines[idx].strip()
    while not next_line.startswith("end"):
        values = next_line.split(" ")
        if len(values) > 1:
            child_dict[values[0]] = values[1:] 
        idx+=1
        next_line = lines[idx].strip() # remove empty lines
    idx +=1 #skip end
    return child_dict, idx
    
def parse_amc_file(filepath):
    with open(filepath, "rt") as in_file:
        lines = in_file.readlines()
    frames = []
    current_frame = None
    idx = 0
    while idx < len(lines):
        next_line = lines[idx].strip()
        values = next_line.split(" ")
        if len(values) == 1:
            if values[0].isdigit():
                if current_frame is not None:
                    frames.append(current_frame)
                current_frame = dict()
        elif len(values) > 1 and current_frame is not None:
            key = values[0]
            current_frame[key] = [float(v) for v in values if v != key]
        idx+=1
    return frames
