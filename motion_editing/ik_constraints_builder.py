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
import numpy as np
from transformations import quaternion_from_matrix
from anim_utils.motion_editing.motion_editing import KeyframeConstraint
from .ik_constraints import SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION, SPATIAL_CONSTRAINT_TYPE_KEYFRAME_RELATIVE_POSITION, SUPPORTED_CONSTRAINT_TYPES

class IKConstraintsBuilder(object):
    def __init__(self, skeleton):
        self.skeleton = skeleton

    def convert_to_ik_constraints(self, constraints, frame_offset=0, time_function=None, constrain_orientation=True):
        ik_constraints = collections.OrderedDict()
        for c in constraints:
            if c.constraint_type not in SUPPORTED_CONSTRAINT_TYPES or "generated" in c.semantic_annotation.keys():
                print("skip unsupported constraint")
                continue
            start_frame_idx = self.get_global_frame_idx(c.canonical_keyframe, frame_offset, time_function)
            if c.canonical_end_keyframe is not None:
                print("apply ik constraint on region")
                end_frame_idx = self.get_global_frame_idx(c.canonical_end_keyframe, frame_offset, time_function)
            else:
                print("no end keyframe defined")
                end_frame_idx = start_frame_idx+1

            for frame_idx in range(start_frame_idx, end_frame_idx):
                ik_constraint = self.convert_mg_constraint_to_ik_constraint(frame_idx, c, constrain_orientation)

                ik_constraint.src_tool_cos = c.src_tool_cos
                ik_constraint.dest_tool_cos = c.dest_tool_cos

                ik_constraint.orientation = None
                if c.orientation is not None:
                    if c.constrain_orientation_in_region:
                        if start_frame_idx < frame_idx and frame_idx < end_frame_idx -1:
                            ik_constraint.inside_region_orientation = True
                        elif frame_idx == start_frame_idx:
                            ik_constraint.inside_region_orientation = True
                            ik_constraint.keep_orientation = True
                    elif frame_idx == start_frame_idx:
                        ik_constraint.orientation = c.orientation
                    
                
                ik_constraint.position = None
                if c.position is not None:
                    if c.constrain_position_in_region:
                        ik_constraint.inside_region_position = False
                        if start_frame_idx < frame_idx and frame_idx < end_frame_idx -1:
                            ik_constraint.inside_region_position = True
                        elif frame_idx == end_frame_idx-1:
                            ik_constraint.end_of_region = True
                    elif frame_idx == start_frame_idx:
                        ik_constraint.position = c.position

                if ik_constraint is not None:
                    if frame_idx not in ik_constraints:
                        ik_constraints[frame_idx] = dict()
                    if c.joint_name not in ik_constraints[frame_idx]:
                        ik_constraints[frame_idx][c.joint_name] = []
                    ik_constraints[frame_idx][c.joint_name] = ik_constraint

        return ik_constraints

    def get_global_frame_idx(self, mp_frame_idx, frame_offset, time_function):
        if time_function is not None:
            frame_idx = frame_offset + int(time_function[mp_frame_idx]) + 1
        else:
            frame_idx = frame_offset + int(mp_frame_idx)
        return frame_idx

    def convert_mg_constraint_to_ik_constraint(self, frame_idx, mg_constraint, constrain_orientation=False):
        if mg_constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION:
            ik_constraint = self._create_keyframe_ik_constraint(mg_constraint, frame_idx, constrain_orientation, look_at=mg_constraint.look_at)
        elif mg_constraint.constraint_type == SPATIAL_CONSTRAINT_TYPE_KEYFRAME_RELATIVE_POSITION:
            ik_constraint = KeyframeConstraint(frame_idx, mg_constraint.joint_name, mg_constraint.position, None, mg_constraint.look_at, mg_constraint.offset)
        else:
            ik_constraint = None
        return ik_constraint

    def _create_keyframe_ik_constraint(self, constraint, keyframe, constrain_orientation, look_at):
        if constrain_orientation:
            orientation = constraint.orientation
        else:
            orientation = None
        return KeyframeConstraint(keyframe, constraint.joint_name, constraint.position, orientation, look_at)

    def generate_relative_constraint(self, keyframe, frame, joint_name, relative_joint_name):
        joint_pos = self.skeleton.nodes[joint_name].get_global_position(frame)
        rel_joint_pos = self.skeleton.nodes[relative_joint_name].get_global_position(frame)
        #create a keyframe constraint but indicate that it is a relative constraint
        ik_constraint = KeyframeConstraint(keyframe, joint_name, None, None, None)
        ik_constraint.relative_parent_joint_name = relative_joint_name
        ik_constraint.relative_offset = joint_pos - rel_joint_pos
        return ik_constraint

    def generate_mirror_constraint(self, keyframe, ref_frame, frame, mirror_joint_name):
        """ generate a constraint on the mirror joint with the similar offset to the root as in the reference frame"""
        ref_mirror_joint_pos = self.skeleton.nodes[mirror_joint_name].get_global_position(ref_frame)
        ref_root_joint_pos = self.skeleton.nodes[self.skeleton.root].get_global_position(ref_frame)
        ref_offset_from_root = ref_mirror_joint_pos-ref_root_joint_pos

        target_root_joint_pos = self.skeleton.nodes[self.skeleton.root].get_global_position(frame)
        mirror_joint_pos = ref_offset_from_root + target_root_joint_pos
        print("generate mirror constraint on",mirror_joint_name, mirror_joint_pos, ref_mirror_joint_pos)
        #create a keyframe constraint and set the original position as constraint
        ik_constraint = KeyframeConstraint(keyframe, mirror_joint_name, mirror_joint_pos, None, None)
        return ik_constraint

        
    def generate_orientation_constraint_from_frame(self, frame, joint_name):
        if joint_name in self.skeleton.nodes:
            m  = self.skeleton.nodes[joint_name].get_global_matrix(frame)#[:3,:3]
            return quaternion_from_matrix(m)

    def convert_to_ik_constraints_with_relative(self, frames, constraints, frame_offset=0, time_function=None, constrain_orientation=True):
        """ Create an orderered dictionary containing for each constrained frame, a KeyframeConstraint for the constrained joint"""
        ik_constraints = collections.OrderedDict()
        for c in constraints:
            if c.constraint_type not in SUPPORTED_CONSTRAINT_TYPES or "generated" in c.semantic_annotation.keys():
                print("skip unsupported constraint")
                continue
            start_frame_idx = self.get_global_frame_idx(c.canonical_keyframe, frame_offset, time_function)
            if c.canonical_end_keyframe is not None:
                print("apply ik constraint on region")
                end_frame_idx = self.get_global_frame_idx(c.canonical_end_keyframe, frame_offset, time_function)
            else:
                print("no end keyframe defined")
                end_frame_idx = start_frame_idx + 1
            if c.orientation is None:
                c.orientation = self.generate_orientation_constraint_from_frame(frames[start_frame_idx], c.joint_name)

            base_ik_constraint = self.convert_mg_constraint_to_ik_constraint(start_frame_idx, c, constrain_orientation)

            relative_ik_constraint = None
            if base_ik_constraint is None:
                continue
            base_ik_constraint.src_tool_cos = c.src_tool_cos
            base_ik_constraint.dest_tool_cos = c.dest_tool_cos
            for frame_idx in range(start_frame_idx, end_frame_idx):
                ik_constraint = deepcopy(base_ik_constraint)
                if frame_idx not in ik_constraints:
                    ik_constraints[frame_idx] = dict()

                ik_constraint.orientation = None
                if c.orientation is not None and constrain_orientation:
                    if c.constrain_orientation_in_region:
                        ik_constraint.orientation = c.orientation
                        if start_frame_idx < frame_idx < end_frame_idx -1:
                            ik_constraint.inside_region_orientation = True
                        elif frame_idx == start_frame_idx:
                            ik_constraint.inside_region_orientation = True
                            ik_constraint.keep_orientation = True
                    elif frame_idx == start_frame_idx:
                        ik_constraint.orientation = c.orientation
                    
                ik_constraint.position = None
                if c.position is not None: 
                    if c.constrain_position_in_region and c.relative_joint_name is None:# ignore region if it is relative
                        ik_constraint.inside_region_position = False
                        ik_constraint.position = c.position
                        if start_frame_idx < frame_idx < end_frame_idx -1:
                            ik_constraint.inside_region_position = True
                        elif frame_idx == end_frame_idx-1:
                            ik_constraint.end_of_region = True
                    elif frame_idx == start_frame_idx:
                        ik_constraint.position = c.position
                # overwrite with relative constraint if larger than start frame
                if c.relative_joint_name is not None and frame_idx == start_frame_idx:
                    relative_ik_constraint = self.generate_relative_constraint(frame_idx, frames[frame_idx],
                                                                        c.joint_name,
                                                                        c.relative_joint_name)
                    relative_ik_constraint.orientation = ik_constraint.orientation
                    ik_constraints[frame_idx][c.joint_name] = ik_constraint

                elif relative_ik_constraint is not None and frame_idx > start_frame_idx:
                    relative_ik_constraint.frame_idx = frame_idx
                    ik_constraints[frame_idx][c.joint_name] = relative_ik_constraint
                else:
                    ik_constraints[frame_idx][c.joint_name] = ik_constraint

                    # add also a mirror constraint
                    if c.mirror_joint_name is not None:
                        _ik_constraint = self.generate_mirror_constraint(frame_idx, frames[0], frames[frame_idx], c.mirror_joint_name)
                        ik_constraints[frame_idx][c.mirror_joint_name] = _ik_constraint
                    
                    # add also a parent constraint
                    if c.constrained_parent is not None and c.vector_to_parent is not None:
                        print("generate parent constraint", c.vector_to_parent)
                        # "get length of vector and calculate constraint position"
                        a = self.skeleton.nodes[c.joint_name].get_global_position(frames[0]) 
                        b = self.skeleton.nodes[c.constrained_parent].get_global_position(frames[0]) 
                        bone_length = np.linalg.norm(b-a)
                        c.vector_to_parent /= np.linalg.norm(c.vector_to_parent)
                        parent_pos = c.position + c.vector_to_parent * bone_length 
                        
                        #create a keyframe constraint but indicate that it is a relative constraint
                        _ik_constraint = KeyframeConstraint(frame_idx, c.constrained_parent, parent_pos, None, None)
                        

                        #remove degrees of freedom of the child constraint
                        ik_constraints[frame_idx][c.joint_name].fk_chain_root = c.constrained_parent

                        #change order so that the parent constraint is applied first
                        # TODO order constraints based on joint order in fk chain
                        reversed_ordered_dict = collections.OrderedDict()
                        reversed_ordered_dict[c.constrained_parent] = _ik_constraint
                        for k, v in ik_constraints[frame_idx].items():
                            reversed_ordered_dict[k] = v
                        ik_constraints[frame_idx] = reversed_ordered_dict
        return ik_constraints
