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
import numpy as np
from copy import copy
from transformations import quaternion_from_matrix, quaternion_matrix, quaternion_multiply, quaternion_slerp
from .analytical_inverse_kinematics import AnalyticalLimbIK
from .numerical_ik_exp import NumericalInverseKinematicsExp
from .utils import normalize, project_on_intersection_circle, smooth_root_positions
from ..animation_data.skeleton_node import SkeletonEndSiteNode
from ..animation_data.utils import quaternion_from_vector_to_vector
from ..animation_data.motion_blending import create_transition_for_joints_using_slerp, BLEND_DIRECTION_FORWARD, BLEND_DIRECTION_BACKWARD, smooth_translation_in_quat_frames
from ..animation_data.skeleton_models import IK_CHAINS_DEFAULT_SKELETON

FOOT_STATE_GROUNDED = 0
FOOT_STATE_SWINGING = 1

def add_heels_to_skeleton(skeleton, left_foot, right_foot, left_heel, right_heel, offset=[0, -5, 0]):
    
    left_heel_node = SkeletonEndSiteNode(left_heel, [], skeleton.nodes[left_foot])
    left_heel_node.offset = np.array(offset)
    skeleton.nodes[left_heel] = left_heel_node
    skeleton.nodes[left_foot].children.append(left_heel_node)

    right_heel_node = SkeletonEndSiteNode(right_heel, [], skeleton.nodes[right_foot])
    right_heel_node.offset = np.array(offset)
    skeleton.nodes[right_heel] = right_heel_node
    skeleton.nodes[right_foot].children.append(right_heel_node)
    return skeleton



def get_heel_offset(skeleton, foot_name, toe_name, frame):
    """ calculate heel offset from foot assuming the "y" axis of the foot coordinate system is aligned to the ground  
    """
    m = skeleton.nodes[foot_name].get_global_matrix(frame)
    foot_position = m[:3,3]
    print("foot_position", foot_position)
    toe_offset = skeleton.nodes[toe_name].offset
    #if len(skeleton.nodes[toe_name].children) > 0:
    #   toe_offset += skeleton.nodes[toe_name].children[0].offset
    up_vector = np.array(skeleton.skeleton_model["cos_map"][foot_name]["y"])
    up_vector /= np.linalg.norm(up_vector)
    #project toe offset on up vector that should be aligned with the ground
    scale = np.dot(up_vector, toe_offset)

    # get global position of toe aligned to ground
    local_offset = scale*up_vector
    local_offset = [local_offset[0],local_offset[1],local_offset[2], 1]
    projected_toe_pos = np.dot(m, local_offset)[:3]
    print("projected_toe_pos", projected_toe_pos)
    # use offset from projected toe position to position to get the global heel position
    toe_pos = skeleton.nodes[toe_name].get_global_position(frame)
    heel_position = foot_position + (toe_pos - projected_toe_pos) 

    # bring into local coordinate system
    heel_position = [heel_position[0],heel_position[1],heel_position[2], 1]
    heel_offset = np.dot(np.linalg.inv(m), heel_position)[:3]
    print("heel_offset", heel_offset)
    print("toe_offset", toe_offset)
    return heel_offset



def get_heel_offset2(skeleton, foot_name, toe_name, frame):
    """ calculate heel offset from foot assuming the "y" axis of the foot coordinate system is aligned to the ground  
    """
    m = skeleton.nodes[foot_name].get_global_matrix(frame)
    foot_position = m[:3,3]
    print("foot_position", foot_position)
    toe_offset = skeleton.nodes[toe_name].offset
    #if len(skeleton.nodes[toe_name].children) > 0:
    #   toe_offset += skeleton.nodes[toe_name].children[0].offset
    foot_cos_map =skeleton.skeleton_model["cos_map"][foot_name]
    up_vector = np.array(foot_cos_map["y"], dtype=np.float32)
    up_vector /= np.linalg.norm(up_vector)
    x_vector = np.array(foot_cos_map["x"], dtype=np.float32)
    x_vector /= np.linalg.norm(x_vector)
    z_vector = np.cross(up_vector, x_vector)
    z_vector /= np.linalg.norm(z_vector)
    #project toe offset on up vector that should be aligned with the ground
    scale = np.dot(z_vector, toe_offset)
    heel_offset = scale*z_vector 
    # bring into local coordinate system
    print("heel_offse2", heel_offset)
    print("toe_offset", toe_offset)
    return heel_offset

def add_temporary_heels_to_skeleton(skeleton, left_foot, right_foot, left_toe, right_toe, left_heel, right_heel):
    
    left_heel_node = SkeletonEndSiteNode(left_heel, [], skeleton.nodes[left_foot])
    left_heel_node.offset = get_heel_offset2(skeleton, left_foot, left_toe, skeleton.reference_frame)
    skeleton.nodes[left_heel] = left_heel_node
    skeleton.nodes[left_foot].children.append(left_heel_node)
    skeleton.skeleton_model["joints"]["left_heel"] = left_heel

    right_heel_node = SkeletonEndSiteNode(right_heel, [], skeleton.nodes[right_foot])
    right_heel_node.offset = get_heel_offset2(skeleton, right_foot, right_toe, skeleton.reference_frame)
    skeleton.nodes[right_heel] = right_heel_node
    skeleton.nodes[right_foot].children.append(right_heel_node)
    skeleton.skeleton_model["joints"]["right_heel"] = right_heel
    return skeleton

def create_grounding_constraint_from_frame(skeleton, frames, frame_idx, joint_name):
    position = skeleton.nodes[joint_name].get_global_position(frames[frame_idx])
    m = skeleton.nodes[joint_name].get_global_matrix(frames[frame_idx])
    m[:3, 3] = [0, 0, 0]
    orientation = normalize(quaternion_from_matrix(m))
    return MotionGroundingConstraint(frame_idx, joint_name, position, None, orientation)


def generate_ankle_constraint_from_toe(skeleton, frames, frame_idx, ankle_joint_name, heel_joint, toe_joint_name, target_ground_height, toe_pos=None):
    """ create a constraint on the ankle position based on the toe constraint position"""
    #print "add toe constraint"
    if toe_pos is None:
        ct = skeleton.nodes[toe_joint_name].get_global_position(frames[frame_idx])
        ct[1] = target_ground_height  # set toe constraint on the ground
    else:
        ct = toe_pos

    a = skeleton.nodes[ankle_joint_name].get_global_position(frames[frame_idx])
    t = skeleton.nodes[toe_joint_name].get_global_position(frames[frame_idx])

    target_toe_offset = a - t  # difference between unmodified toe and ankle at the frame
    ca = ct + target_toe_offset  # move ankle so toe is on the ground

    m = skeleton.nodes[heel_joint].get_global_matrix(frames[frame_idx])
    m[:3, 3] = [0, 0, 0]
    oq = quaternion_from_matrix(m)
    oq = normalize(oq)

    return MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, None, oq)


def create_ankle_constraint_from_toe_and_heel(skeleton, frames, frame_idx, ankle_joint, heel_joint, toe_joint,heel_offset, target_ground_height, heel_pos=None, toe_pos=None, is_swinging=False):
    if toe_pos is None:
        ct = skeleton.nodes[toe_joint].get_global_position(frames[frame_idx])
        ct[1] = target_ground_height
    else:
        ct = toe_pos
    if heel_pos is None:
        ch = skeleton.nodes[heel_joint].get_global_position(frames[frame_idx])
        ch[1] = target_ground_height
    else:
        ch = heel_pos
    target_direction = normalize(ct - ch)
    t = skeleton.nodes[toe_joint].get_global_position(frames[frame_idx])
    h = skeleton.nodes[heel_joint].get_global_position(frames[frame_idx])
    original_direction = normalize(t - h)

    global_delta_q = quaternion_from_vector_to_vector(original_direction, target_direction)
    global_delta_q = normalize(global_delta_q)

    m = skeleton.nodes[heel_joint].get_global_matrix(frames[frame_idx])
    m[:3, 3] = [0, 0, 0]
    oq = quaternion_from_matrix(m)
    oq = normalize(oq)
    orientation = normalize(quaternion_multiply(global_delta_q, oq))

    # set target ankle position based on the  grounded heel and the global target orientation of the ankle
    m = quaternion_matrix(orientation)[:3, :3]
    target_heel_offset = np.dot(m, heel_offset)
    ca = ch - target_heel_offset
    print("set ankle constraint both", ch, ca, target_heel_offset, target_ground_height)
    foot_state = FOOT_STATE_GROUNDED
    if is_swinging:
        foot_state = FOOT_STATE_SWINGING
    return MotionGroundingConstraint(frame_idx, ankle_joint, ca, None, orientation, foot_state)


def interpolate_constraints(c1, c2):
    p = (c1.position + c2.position)/2
    o = quaternion_slerp(c1.orientation, c2.orientation, 0.5)
    o = normalize(o)
    return MotionGroundingConstraint(c1.frame_idx, c1.joint_name, p, None, o)


class MotionGroundingConstraint(object):
    def __init__(self, frame_idx, joint_name, position, direction=None, orientation=None, foot_state=FOOT_STATE_GROUNDED):
        self.frame_idx = frame_idx
        self.joint_name = joint_name
        self.position = position
        self.direction = direction
        self.orientation = orientation
        self.toe_position = None
        self.heel_position = None
        self.global_toe_offset = None
        self.foot_state = foot_state

    def evaluate(self, skeleton, q_frame):
        d = self.position - skeleton.nodes[self.joint_name].get_global_position(q_frame)
        return np.dot(d, d)


class IKConstraintSet(object):
    def __init__(self, frame_range, joint_names, positions):
        self.frame_range = frame_range
        self.joint_names = joint_names
        self.constraints = []
        for idx in range(frame_range[0], frame_range[1]):
            for idx, joint_name in enumerate(joint_names):
                c = MotionGroundingConstraint(idx, joint_name, positions[idx], None)
                self.constraints.append(c)

    def add_constraint(self, c):
        self.constraints.append(c)

    def evaluate(self, skeleton, q_frame):
        error = 0
        for c in self.constraints:
            d = c.position - skeleton.nodes[c.joint_name].get_global_position(q_frame)
            error += np.dot(d, d)
        return error


def add_fixed_dofs_to_frame(skeleton, frame):
    o = 3
    full_frame = frame[:3].tolist()
    for key, node in list(skeleton.nodes.items()):
        if len(node.children) == 0:
            continue
        if not node.fixed:
            full_frame += frame[o:o+4].tolist()
            o += 4
        else:
            full_frame += node.rotation.tolist()
    return full_frame

def extract_ik_chains(skeleton, damp_angle, damp_factor):
    joints_map = skeleton.skeleton_model["joints"]
    cos_map = skeleton.skeleton_model["cos_map"]
    new_ik_chains = dict()
    for j in IK_CHAINS_DEFAULT_SKELETON:
        mapped_j = joints_map[j]
        root_joint = IK_CHAINS_DEFAULT_SKELETON[j]["root"]
        free_joint = IK_CHAINS_DEFAULT_SKELETON[j]["joint"]
        if root_joint in joints_map and free_joint in joints_map:
            mapped_free_joint = joints_map[free_joint]
            if mapped_free_joint in cos_map:
                data = copy(IK_CHAINS_DEFAULT_SKELETON[j])
                data["root"] = joints_map[root_joint]
                data["joint"] = mapped_free_joint
                data["joint_axis"] = cos_map[mapped_free_joint]["x"]
                data["end_effector_dir"] = cos_map[mapped_free_joint]["y"]
                new_ik_chains[mapped_j] = AnalyticalLimbIK.init_from_dict(skeleton, mapped_j, data, damp_angle=damp_angle, damp_factor=damp_factor)
    return new_ik_chains

class MotionGrounding(object):
    def __init__(self, skeleton, ik_settings, skeleton_model, use_analytical_ik=True, damp_angle=None, damp_factor=None):
        self.skeleton = skeleton
        self._ik = NumericalInverseKinematicsExp(skeleton, ik_settings)
        self._constraints = collections.OrderedDict()
        self.transition_window = 10
        self.root_smoothing_window = 20
        self.translation_blend_window = 40
        self._blend_ranges = collections.OrderedDict()
        self.use_analytical_ik = use_analytical_ik
        self.skeleton_model = skeleton_model
        self.damp_angle = damp_angle
        self.damp_factor = damp_factor
        if "joints" in skeleton_model and "left_toe" in skeleton_model["joints"] and "right_toe" in skeleton_model["joints"]:
            joints_map = skeleton_model["joints"]
            self.ik_chains = extract_ik_chains(skeleton, self.damp_angle, self.damp_factor)  
           
            add_temporary_heels_to_skeleton(skeleton, joints_map["left_ankle"], joints_map["right_ankle"], joints_map["left_toe"], joints_map["right_toe"], "left_heel", "right_heel")
            self.initialized = True
        else:
            self.ik_chains = dict()
            self.initialized = False
        

    def set_constraints(self, constraints):
        self._constraints = constraints

    def add_constraint(self, joint_name, frame_range, position, direction=None):
        for frame_idx in range(*frame_range):
            c = MotionGroundingConstraint(frame_idx, joint_name, position, direction)
            if frame_idx not in list(self._constraints.keys()):
                self._constraints[frame_idx] = []
            self._constraints[frame_idx].append(c)

    def add_blend_range(self, joint_names, frame_range):
        if frame_range not in list(self._constraints.keys()):
            self._blend_ranges[frame_range] = []
        for j in joint_names:
            self._blend_ranges[frame_range].append(j)

    def clear_constraints(self):
        self._constraints = collections.OrderedDict()

    def clear_blend_ranges(self):
        self._blend_ranges = collections.OrderedDict()

    def clear(self):
        self.clear_constraints()
        self.clear_blend_ranges()

    def run(self, motion_vector, scene_interface=None):
        new_frames = motion_vector.frames[:]
        if scene_interface is not None:
            self.shift_root_to_ground(new_frames, scene_interface)
        self.shift_root_to_reach_constraints(new_frames)
        if len(new_frames) > 1:
            self.blend_at_transitions(new_frames)
        if self.use_analytical_ik:
            self.apply_analytical_ik(new_frames)
        else:
            self.apply_ik_constraints(new_frames)
        if len(new_frames) > 1:
            self.blend_at_transitions(new_frames)
        return new_frames

    def apply_on_frame(self, frame, scene_interface):
        x = frame[0]
        z = frame[2]
        target_ground_height = scene_interface.get_height(x, z)
        shift = target_ground_height - frame[1]
        frame[1] += shift
        #self.apply_analytical_ik_on_frame(frame, constraints)
        return frame

    def _blend_around_frame_range(self, frames, start, end, joint_names):
        for joint_name in joint_names:
            transition_start = max(start - self.transition_window, 0)
            transition_end = min(end + self.transition_window, frames.shape[0]-1) - 1
            forward_steps = start - transition_start
            backward_steps = transition_end - end
            if joint_name == self.skeleton.root:
                if start > 0:
                    frames = smooth_translation_in_quat_frames(frames, start, self.translation_blend_window)
                temp_frame = min(end + 1, frames.shape[0]-1)
                frames = smooth_translation_in_quat_frames(frames, temp_frame, self.translation_blend_window)

            idx = self._ik.skeleton.animated_joints.index(joint_name)*4+3
            joint_parameter_indices = [idx, idx+1, idx+2, idx+3]
            if start > 0:
                create_transition_for_joints_using_slerp(frames, joint_parameter_indices, transition_start, start, forward_steps, BLEND_DIRECTION_FORWARD)
            create_transition_for_joints_using_slerp(frames, joint_parameter_indices, end, transition_end, backward_steps, BLEND_DIRECTION_BACKWARD)

    def apply_ik_constraints(self, frames):
        for frame_idx, constraints in self._constraints.items():
            if 0 <= frame_idx < len(frames):
                frames[frame_idx] = self._ik.modify_frame(frames[frame_idx], constraints)

    def shift_root_to_reach_constraints(self, frames):
        root_positions = self.generate_root_positions_from_foot_constraints(frames)
        root_positions = smooth_root_positions(root_positions, self.root_smoothing_window)
        self.apply_root_constraints(frames, root_positions)

    def generate_root_positions_from_foot_constraints(self, frames):
        root_constraints = []
        for frame_idx, constraints in self._constraints.items():
            if 0 <= frame_idx < len(frames):
                grounding_constraints = [c for c in constraints if c.foot_state==FOOT_STATE_GROUNDED]
                n_constraints = len(grounding_constraints)
                p = None
                if n_constraints == 1:
                    p = self.generate_root_constraint_for_one_foot(frames[frame_idx], grounding_constraints[0])
                elif n_constraints > 1:
                    p = self.generate_root_constraint_for_two_feet(frames[frame_idx], grounding_constraints[0], grounding_constraints[1])
                if p is None:
                    p = frames[frame_idx, :3]
                root_constraints.append(p)
        return np.array(root_constraints)

    def apply_root_constraints(self, frames, constraints):
        for frame_idx, p in enumerate(constraints):
            if p is not None:
                frames[frame_idx][:3] = p

    def generate_root_constraint_for_one_foot(self, frame, c):
        pelvis = self.skeleton.skeleton_model["joints"]["pelvis"]
        pelvis_pos = self.skeleton.nodes[pelvis].get_global_position(frame)
        target_length = np.linalg.norm(c.position - pelvis_pos)
        limb_length = self.get_limb_length(c.joint_name)
        if target_length < limb_length:
            return frame[:3] # no change
        new_root_pos = (c.position + normalize(root_pos - c.position) * limb_length)
        if np.linalg.norm(self.skeleton.nodes[pelvis].offset) > 0.0:
            root_m = self.skeleton.nodes[self.skeleton.root].get_global_matrix(frame)[:3,:3]
            new_root_pos -= np.dot(root_m, self.skeleton.nodes[pelvis].offset)
        else:
            new_root_pos -= self.skeleton.nodes[pelvis].offset
        return new_root_pos

    def generate_root_constraint_for_two_feet(self, frame, constraint1, constraint2, limb_length_offset=0.0):
        """ Set the root position to the projection on the intersection of two spheres """
        
        pelvis = self.skeleton.skeleton_model["joints"]["pelvis"]
        m = self.skeleton.nodes[pelvis].get_global_matrix(frame)
        p = m[:3, 3]
        #print("root position", root, p)
        t1 = np.linalg.norm(constraint1.position - p)
        t2 = np.linalg.norm(constraint2.position - p)

        c1 = constraint1.position
        r1 = self.get_limb_length(constraint1.joint_name) + limb_length_offset
        #p1 = c1 + r1 * normalize(p-c1)
        c2 = constraint2.position
        r2 = self.get_limb_length(constraint2.joint_name) + limb_length_offset
        #(r1, r2, t1,t2)
        #p2 = c2 + r2 * normalize(p-c2)
        if t1 < r1 and t2 < r2:
            return None
        #print("adapt root for two constraints", constraint1.position, r1, constraint2.position, r2)
        new_root_pos = project_on_intersection_circle(p, c1, r1, c2, r2)
        if np.linalg.norm(self.skeleton.nodes[pelvis].offset) > 0.0:
            root_m = self.skeleton.nodes[self.skeleton.root].get_global_matrix(frame)[:3,:3]
            new_root_pos -= np.dot(root_m, self.skeleton.nodes[pelvis].offset)
        else:
            new_root_pos -= self.skeleton.nodes[pelvis].offset
        return new_root_pos

    def get_limb_length(self, joint_name):
        limb_length = np.linalg.norm(self.skeleton.nodes[joint_name].offset)
        limb_length += np.linalg.norm(self.skeleton.nodes[joint_name].parent.offset)
        return limb_length

    def apply_analytical_ik(self, frames):
        n_frames = len(frames)
        for frame_idx, constraints in self._constraints.items():
            if 0 <= frame_idx < n_frames and len(constraints)> 0:
                self.apply_analytical_ik_on_frame(frames[frame_idx], constraints)

    def apply_analytical_ik_on_frame(self, frame, constraints):
        for c in constraints:
            if c.joint_name in self.ik_chains:
                frame = self.ik_chains[c.joint_name].apply(frame, c.position, c.orientation)
            else:
                print("could not find ik chain definition for ", c.joint_name)
                frame = self._ik.modify_frame(frame, constraints)
        return frame

    def apply_orientation_constraints_on_frame(self, frame, constraints):
        for c in constraints:
            self.ik_chains[c.joint_name].set_end_effector_rotation2(frame, c.orientation)
        return frame

    def blend_at_transitions(self, frames):
        for frame_range, joint_names in self._blend_ranges.items():
            start = frame_range[0]
            end = frame_range[1]
            self._blend_around_frame_range(frames, start, end, joint_names)
        return frames

    def shift_root_to_ground(self, frames, scene_interface):
        for idx, frame in enumerate(frames):
            x = frames[idx][0]
            z = frames[idx][2]
            target_ground_height = scene_interface.get_height(x, z)
            root_pos = self.get_projected_root_pos(frames[idx])
            shift = target_ground_height - root_pos[1]
            frames[idx][1] += shift

    def get_projected_root_pos(self, frame):
        pelvis = self.skeleton.skeleton_model["joints"]["pelvis"]
        ground_pos = self.skeleton.nodes[pelvis].get_global_position(frame)
        if np.linalg.norm(self.skeleton.nodes[pelvis].offset) > 0.0:
            root_m = self.skeleton.nodes[self.skeleton.root].get_global_matrix(frame)[:3,:3]
            ground_pos -= np.dot(root_m, self.skeleton.nodes[pelvis].offset)
        else:
            ground_pos -= self.skeleton.nodes[pelvis].offset
        return ground_pos
