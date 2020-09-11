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

SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION = "keyframe_position"
SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION = "keyframe_two_hands"
SPATIAL_CONSTRAINT_TYPE_KEYFRAME_RELATIVE_POSITION = "keyframe_relative_position"
SPATIAL_CONSTRAINT_TYPE_KEYFRAME_LOOK_AT = "keyframe_look_at"
SUPPORTED_CONSTRAINT_TYPES = [SPATIAL_CONSTRAINT_TYPE_KEYFRAME_POSITION,
                              SPATIAL_CONSTRAINT_TYPE_TWO_HAND_POSITION,
                              SPATIAL_CONSTRAINT_TYPE_KEYFRAME_LOOK_AT, 
                              SPATIAL_CONSTRAINT_TYPE_KEYFRAME_RELATIVE_POSITION]
class IKConstraint(object):
    @staticmethod
    def evaluate(params, data):
        pass


class JointIKConstraint(IKConstraint):
    def __init__(self, joint_name, position, orientation, keyframe, free_joints, step_idx=-1, frame_range=None, look_at=False, optimize=True, offset=None, look_at_pos=None):
        self.joint_name = joint_name
        self.position = position
        self.orientation = orientation
        self.keyframe = keyframe
        self.frame_range = frame_range
        self.free_joints = free_joints
        self.step_idx = step_idx
        self.look_at = look_at
        self.optimize = optimize
        self.offset = offset
        self.look_at_pos = look_at_pos
        # set in case it is a relative constraint
        self.relative_parent_joint_name = None  # joint the offsets points from to the target
        self.relative_offset = None
        
        # tool orientation constraint
        self.src_tool_cos = None # tool coordinate system
        self.dest_tool_cos = None # target direction
        
        # set a fk chain root to reduce the degrees of freedom 
        self.fk_chain_root = None


    @staticmethod
    def evaluate(parameters, data):
        pose, free_joints, target_joint, target_position, target_orientation, offset = data
        pose.set_channel_values(parameters, free_joints)
        #parent_joint = pose.get_parent_joint(target_joint)
        #pose.apply_bounds_on_joint(parent_joint)
        if target_orientation is not None:
            parent_joint = pose.get_parent_joint(target_joint)
            if parent_joint is not None:
                pose.set_hand_orientation(parent_joint, target_orientation)
                #pose.apply_bounds_on_joint(parent_joint)
        if offset is not None:
            m = pose.evaluate_matrix(target_joint)
            p = np.dot(m, offset)[:3]
            d = target_position - p
        else:
            d = pose.evaluate_position(target_joint) - target_position
        return np.dot(d, d)

    def data(self, ik, free_joints=None):
        if free_joints is None:
            free_joints = self.free_joints
        if ik.optimize_orientation:
            orientation = self.orientation
        else:
            orientation = None
        return ik.pose, free_joints, self.joint_name, self.position, orientation, self.offset

    def get_joint_names(self):
        return [self.joint_name]

class RelativeJointIKConstraint(IKConstraint):
    def __init__(self, ref_joint_name, target_joint_name, rel_position, keyframe, free_joints, step_idx=-1, frame_range=None):
        self.ref_joint_name = ref_joint_name
        self.target_joint_name = target_joint_name
        self.rel_position = rel_position
        self.keyframe = keyframe
        self.frame_range = frame_range
        self.free_joints = free_joints
        self.step_idx = step_idx

    @staticmethod
    def evaluate(parameters, data):
        pose, free_joints, ref_joint, target_joint, target_delta = data
        pose.set_channel_values(parameters, free_joints)
        ref_matrix = pose.skeleton.nodes[ref_joint].get_global_matrix(pose.get_vector())
        target = np.dot(ref_matrix, target_delta)[:3]
        d = pose.evaluate_position(target_joint) - target
        return np.dot(d, d)

    def data(self, ik, free_joints=None):
        if free_joints is None:
            free_joints = self.free_joints
        return ik.pose, free_joints, self.ref_joint_name, self.target_joint_name, self.rel_position

    def get_joint_names(self):
        return [self.target_joint_name]


class TwoJointIKConstraint(IKConstraint):
    def __init__(self, joint_names, target_positions, target_center, target_delta, target_direction, keyframe, free_joints):
        self.joint_names = joint_names
        self.target_positions = target_positions
        self.target_center = target_center
        self.target_delta = target_delta
        self.target_direction = target_direction
        self.keyframe = keyframe
        self.free_joints = free_joints

    @staticmethod
    def evaluate(parameters, data):
        pose, free_joints, joint_names, target_positions, target_center, target_delta, target_direction = data
        pose.set_channel_values(parameters, free_joints)
        left = pose.evaluate_position(joint_names[0])
        right = pose.evaluate_position(joint_names[1])
        delta_vector = right - left
        residual_vector = [0.0, 0.0, 0.0]
        #get distance to center
        residual_vector[0] = np.linalg.norm(target_center - (left + 0.5 * delta_vector))
        #get difference to distance between hands
        delta = np.linalg.norm(delta_vector)
        residual_vector[1] = abs(target_delta - delta)
        #print "difference", residual_vector[1]
        #get difference of global orientation
        direction = delta_vector/delta

        residual_vector[2] = abs(target_direction[0] - direction[0]) + \
                             abs(target_direction[1] - direction[1]) + \
                             abs(target_direction[2] - direction[2])
        residual_vector[2] *= 10.0

        #print (target_center, (left + 0.5 * delta_vector), left, right)
        #error = np.linalg.norm(left-target_positions[0]) + np.linalg.norm(right-target_positions[1])
        return sum(residual_vector)#error#residual_vector[0]#sum(residual_vector)

    def data(self, ik, free_joints=None):
        if free_joints is None:
            free_joints = self.free_joints
        return ik.pose, free_joints, self.joint_names, self.target_positions, self.target_center, self.target_delta, self.target_direction

    def get_joint_names(self):
        return self.joint_names


