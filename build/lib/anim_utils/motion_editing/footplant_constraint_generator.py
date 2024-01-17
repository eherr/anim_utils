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
from scipy.interpolate import UnivariateSpline
from transformations import quaternion_from_matrix, quaternion_multiply, quaternion_matrix, quaternion_slerp, quaternion_inverse
from ..animation_data.skeleton_models import *
from .motion_grounding import MotionGroundingConstraint, FOOT_STATE_SWINGING
from .utils import get_average_joint_position, get_average_joint_direction, normalize, get_average_direction_from_target
from ..animation_data.utils import quaternion_from_vector_to_vector

def get_velocity_and_acceleration(scalars):
    """ https://stackoverflow.com/questions/40226357/second-derivative-in-python-scipy-numpy-pandas
    """
    x = np.linspace(0, len(scalars), len(scalars))
    ys = np.array(scalars, dtype=np.float32)
    y_spl = UnivariateSpline(x, ys, s=0, k=4)
    velocity = y_spl.derivative(n=1)
    acceleration = y_spl.derivative(n=2)
    return velocity(x), acceleration(x)

def get_joint_vertical_velocity_and_acceleration(skeleton, frames, joints):
    joint_heights = dict()
    for joint in joints:
        ps = []
        for frame in frames:
            p = skeleton.nodes[joint].get_global_position(frame)
            ps.append(p[1])
        if len(ps)> 1:
            v, a = get_velocity_and_acceleration(ps)
            joint_heights[joint] = ps[:-1], v, a
        elif len(ps) == 1:
            joint_heights[joint] = [ps[0]], [0], [0]
        else:
            joint_heights[joint] = [], [], []

    return joint_heights


def get_joint_height(skeleton, frames, joints):
    joint_heights = dict()
    for joint in joints:
        ps = []
        for frame in frames:
            p = skeleton.nodes[joint_name].get_global_position(frame)
            ps.append(p)
        joint_heights[joint] = positions
    return joint_heights


def guess_ground_height(skeleton, frames, start_frame, n_frames, foot_joints):
    minimum_height = np.inf
    joint_heights = get_joint_height(skeleton, frames[start_frame:start_frame+n_frames], foot_joints)
    for joint in joint_heights:
        p = joint_heights[joint]
        pT = np.array(p).T
        new_min_height = min(pT[1])
        if new_min_height < minimum_height:
            minimum_height = new_min_height
    return minimum_height


def get_quat_delta(qa, qb):
    """ get quaternion from quat a to quat b """
    return quaternion_multiply(qb, quaternion_inverse(qa))


def get_leg_state(plant_range, foot_definitions, side, frame):
    state = "swinging"
    for foot in foot_definitions[side]:
            if side in plant_range:
                if foot in plant_range[side]:
                    if plant_range[side][foot]["start"] is not None and plant_range[side][foot]["start"] <= frame <= plant_range[side][foot]["end"]:
                        state = "planted"
    return state


def create_leg_state_model(plant_range, start_frame, end_frame, foot_definitions):
    model = collections.OrderedDict()
    for f in range(start_frame, end_frame):
        model[f] = dict()
        for side in foot_definitions:
            model[f][side] = get_leg_state(plant_range, foot_definitions, side, f)
    return model


def regenerate_ankle_constraint_with_new_orientation(position, joint_offset, new_orientation):
    """ move ankle so joint is on the ground using the new orientation"""
    m = quaternion_matrix(new_orientation)[:3, :3]
    target_offset = np.dot(m, joint_offset)
    ca = position - target_offset
    return ca

def regenerate_ankle_constraint_with_new_orientation2(position, joint_offset, delta):
    """ move ankle so joint is on the ground using the new orientation"""
    m = quaternion_matrix(delta)[:3, :3]
    target_offset = np.dot(m, joint_offset)
    ca = position + target_offset
    return ca

def get_global_orientation(skeleton, joint_name, frame):
    m = skeleton.nodes[joint_name].get_global_matrix(frame)
    m[:3, 3] = [0, 0, 0]
    return normalize(quaternion_from_matrix(m))


def convert_plant_range_to_ground_contacts(start_frame, end_frame, plant_range):
    print("convert plant range", start_frame, end_frame)
    ground_contacts = collections.OrderedDict()
    for f in range(start_frame, end_frame+1):
        ground_contacts[f] = []
    for side in list(plant_range.keys()):
        for joint in list(plant_range[side].keys()):
            if plant_range[side][joint]["start"] is not None:
                start = plant_range[side][joint]["start"]
                end = plant_range[side][joint]["end"]
                print(joint, start,end)
                for f in range(start, end):
                    ground_contacts[f].append(joint)
    return ground_contacts


def find_first_frame(skeleton, frames, joint_name, start_frame, end_frame, tolerance=2.0):
    p = skeleton.nodes[joint_name].get_global_position(frames[end_frame-1])
    h = p[1]
    search_window = list(range(start_frame, end_frame))
    for f in reversed(search_window):
        tp = skeleton.nodes[joint_name].get_global_position(frames[f])
        if abs(tp[1]-h) > tolerance:
            return f
    return start_frame


def find_last_frame(skeleton, frames, joint_name, start_frame, end_frame, tolerance=2.0):
    p = skeleton.nodes[joint_name].get_global_position(frames[start_frame])
    h = p[1]
    search_window = list(range(start_frame, end_frame))
    for f in search_window:
        tp = skeleton.nodes[joint_name].get_global_position(frames[f])
        if abs(tp[1] - h) > tolerance:
            return f
    return end_frame


def merge_constraints(a,b):
    for key, item in list(b.items()):
        if key in a:
            a[key] += b[key]
        else:
            a[key] = b[key]
    return a

def align_quaternion(q, ref_q):
    if np.dot(ref_q, q) < 0:
        return -q
    else:
        return q


def blend(x):
    return 2 * x * x * x - 3 * x * x + 1


def get_plant_frame_range(step, search_window):
    start_frame = step.start_frame
    end_frame = step.end_frame + 1
    half_window = search_window / 2
    end_offset = -5
    plant_range = dict()
    L = "left"
    R = "right"
    plant_range[L] = dict()
    plant_range[R] = dict()

    plant_range[L]["start"] = None
    plant_range[L]["end"] = None
    plant_range[R]["start"] = None
    plant_range[R]["end"] = None

    if step.node_key[1] == "beginLeftStance":
        plant_range[R]["start"] = start_frame
        plant_range[R]["end"] = end_frame - half_window + end_offset
        plant_range[L]["start"] = start_frame
        plant_range[L]["end"] = start_frame + 20

    elif step.node_key[1] == "beginRightStance":
        plant_range[L]["start"] = start_frame
        plant_range[L]["end"] = end_frame - half_window + end_offset
        plant_range[R]["start"] = start_frame
        plant_range[R]["end"] = start_frame + 20

    elif step.node_key[1] == "endLeftStance":
        plant_range[R]["start"] = start_frame + half_window
        plant_range[R]["end"] = end_frame
        plant_range[L]["start"] = end_frame - 20
        plant_range[L]["end"] = end_frame

    elif step.node_key[1] == "endRightStance":
        plant_range[L]["start"] = start_frame + half_window
        plant_range[L]["end"] = end_frame
        plant_range[R]["start"] = end_frame - 20
        plant_range[R]["end"] = end_frame

    elif step.node_key[1] == "leftStance":
        plant_range[R]["start"] = start_frame + half_window
        plant_range[R]["end"] = end_frame - half_window + end_offset

    elif step.node_key[1] == "rightStance":
        plant_range[L]["start"] = start_frame + half_window
        plant_range[L]["end"] = end_frame - half_window + end_offset
    return plant_range


def get_plant_frame_range_using_search(skeleton, motion_vector, step, foot_definitions, search_window, lift_tolerance):
    """ Use the assumption that depending on the motion primitive different
        feet are grounded at the beginning, middle and end of the step to find the plant range for each joint.
        Starting from those frames search forward and backward until the foot is lifted.
    """
    start_frame = step.start_frame
    end_frame = step.end_frame + 1
    frames = motion_vector.frames
    w = search_window
    search_start = max(end_frame - w, start_frame)  # start of search range for the grounding range
    search_end = min(start_frame + w, end_frame)  # end of search range for the grounding range
    plant_range = dict()
    L = "left"
    R = "right"
    joint_types = ["heel", "toe"] # first check heel then toe
    plant_range[L] = dict()
    plant_range[R] = dict()
    for side in list(plant_range.keys()):
        plant_range[side] = dict()
        for joint_type in joint_types:
            joint = foot_definitions[side][joint_type]
            plant_range[side][joint] = dict()
            plant_range[side][joint]["start"] = None
            plant_range[side][joint]["end"] = None

            if step.node_key[1] == "beginLeftStance":
                plant_range[side][joint]["start"] = start_frame
                plant_range[side][joint]["end"] = find_last_frame(skeleton, frames, joint, start_frame, search_end, lift_tolerance)

            elif step.node_key[1] == "beginRightStance":
                plant_range[side][joint]["start"] = start_frame
                plant_range[side][joint]["end"] = find_last_frame(skeleton, frames, joint, start_frame, search_end, lift_tolerance)

            elif step.node_key[1] == "endLeftStance":
                heel = foot_definitions[side]["heel"]
                if plant_range[side][heel]["start"] is not None:
                    local_search_start = plant_range[side][heel]["start"]
                else:
                    local_search_start = start_frame
                plant_range[side][joint]["start"] = find_first_frame(skeleton, frames, joint, local_search_start, end_frame, lift_tolerance+3.0)
                plant_range[side][joint]["end"] = end_frame

            elif step.node_key[1] == "endRightStance":
                heel = foot_definitions[side]["heel"]
                if plant_range[side][heel]["start"] is not None:
                    local_search_start = plant_range[side][heel]["start"]
                else:
                    local_search_start = start_frame
                plant_range[side][joint]["start"] = find_first_frame(skeleton, frames, joint, local_search_start, end_frame, lift_tolerance+3.0)
                plant_range[side][joint]["end"] = end_frame

            elif step.node_key[1] == "leftStance" and side == R:
                middle_frame = int((end_frame-start_frame)/2) + start_frame
                plant_range[side][joint]["start"] = find_first_frame(skeleton, frames, joint, search_start, middle_frame, lift_tolerance)
                plant_range[side][joint]["end"] = find_last_frame(skeleton, frames, joint, middle_frame + 1, search_end, lift_tolerance)

            elif step.node_key[1] == "rightStance" and side == L:
                middle_frame = int((end_frame - start_frame) / 2) + start_frame
                plant_range[side][joint]["start"] = find_first_frame(skeleton, frames, joint, search_start, middle_frame, lift_tolerance)
                plant_range[side][joint]["end"] = find_last_frame(skeleton, frames, joint, middle_frame + 1, search_end, lift_tolerance)
    return plant_range



class SceneInterface(object):
    def __init__(self, height):
        self.h = height

    def get_height(self,x, z):
        return self.h

def create_ankle_constraint(skeleton, frames, ankle_joint_name, heel_joint_name, toe_joint, frame_idx, end_frame, ground_height):
    """ create constraint on ankle position and orientation """
    c = get_average_joint_position(skeleton, frames, heel_joint_name, frame_idx, end_frame)
    c[1] = ground_height
    a = get_average_joint_position(skeleton, frames, ankle_joint_name, frame_idx, end_frame)
    h = get_average_joint_position(skeleton, frames, heel_joint_name, frame_idx, end_frame)
    target_heel_offset = a - h
    ca = c + target_heel_offset
    avg_direction = None
    if len(skeleton.nodes[ankle_joint_name].children) > 0:
        avg_direction = get_average_direction_from_target(skeleton, frames, ca, toe_joint,
                                                    frame_idx, end_frame)
        toe_length = np.linalg.norm(skeleton.nodes[toe_joint].offset)
        ct = ca + avg_direction * toe_length
        ct[1] = ground_height
        avg_direction = normalize(ct - ca)
    return MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, avg_direction)


class FootplantConstraintGenerator(object):
    def __init__(self, skeleton, settings, scene_interface=None):
        self.skeleton = skeleton
        self.left_foot = skeleton.skeleton_model["joints"]["left_ankle"]
        self.right_foot = skeleton.skeleton_model["joints"]["right_ankle"]
        self.right_heel = skeleton.skeleton_model["joints"]["right_heel"]
        self.left_heel = skeleton.skeleton_model["joints"]["left_heel"]
        self.right_toe = skeleton.skeleton_model["joints"]["right_toe"]
        self.left_toe = skeleton.skeleton_model["joints"]["left_toe"]
        self.contact_joints = [self.right_heel, self.left_heel, self.right_toe, self.left_toe]

        self.foot_definitions = {"right": {"heel": self.right_heel, "toe": self.right_toe, "ankle": self.right_foot},
                                 "left": {"heel": self.left_heel, "toe": self.left_toe, "ankle": self.left_foot}}
       
        if scene_interface is None:
            self.scene_interface = SceneInterface(0)
        else:
            self.scene_interface = scene_interface

        self.contact_tolerance = settings["contact_tolerance"]
        self.foot_lift_tolerance = settings["foot_lift_tolerance"]
        self.constraint_generation_range = settings["constraint_range"]
        self.smoothing_constraints_window = settings["smoothing_constraints_window"]
        self.foot_lift_search_window = settings["foot_lift_search_window"]

        self.position_constraint_buffer = dict()
        self.orientation_constraint_buffer = dict()
        self.velocity_tolerance = 0
        if "velocity_tolerance" in settings:
            self.velocity_tolerance = settings["velocity_tolerance"]

    def detect_ground_contacts(self, frames, joints, ground_height=0):
        """https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
        """
        joint_y_vel_acc = get_joint_vertical_velocity_and_acceleration(self.skeleton, frames, joints)
        ground_contacts = []
        for f in frames:
            ground_contacts.append([])
        for joint in joints:
            ps, yv, ya = joint_y_vel_acc[joint]
            for idx, p in enumerate(ps):
                velocity = np.sqrt(yv[idx]*yv[idx])
                if p - ground_height < self.contact_tolerance or velocity < self.velocity_tolerance :#and zero_crossings[frame_idx]
                    ground_contacts[idx].append(joint)
        # copy contacts of frame not covered by velocity
        if len(ground_contacts) > 1:
            ground_contacts[-1] = ground_contacts[-2]
        return self.filter_outliers(ground_contacts, joints)

    def filter_outliers(self, ground_contacts, joints):
        n_frames = len(ground_contacts)
        filtered_ground_contacts = [[] for idx in range(n_frames)]
        filtered_ground_contacts[0] = ground_contacts[0]
        filtered_ground_contacts[-1] = ground_contacts[-1]
        frames_indices = range(1,n_frames-1)
        for frame_idx in frames_indices:
            filtered_ground_contacts[frame_idx] = []
            prev_frame = ground_contacts[frame_idx - 1]
            current_frame = ground_contacts[frame_idx]
            next_frame = ground_contacts[frame_idx + 1]
            for joint in joints:
                if joint in current_frame:
                    if joint not in prev_frame and joint not in next_frame:
                        continue
                    else:
                        filtered_ground_contacts[frame_idx].append(joint)
        return filtered_ground_contacts

    def generate_from_graph_walk(self, motion_vector):
        """ the interpolation range must start at end_frame-1 because this is the last modified frame """
        self.position_constraint_buffer = dict()
        self.orientation_constraint_buffer = dict()

        constraints = dict()
        for frame_idx in range(motion_vector.n_frames):
            self.position_constraint_buffer[frame_idx] = dict()
            self.orientation_constraint_buffer[frame_idx] = dict()
            constraints[frame_idx] = []

        ground_contacts = collections.OrderedDict()
        blend_ranges = dict()
        blend_ranges[self.right_foot] = []
        blend_ranges[self.left_foot] = []
        graph_walk = motion_vector.graph_walk
        for idx, step in enumerate(graph_walk.steps):
            step_length = step.end_frame - step.start_frame
            if step_length <= 0:
                print("small frame range ", idx, step.node_key, step.start_frame, step.end_frame)

                continue
            if step.node_key[0] in LOCOMOTION_ACTIONS:
                step_plant_range = get_plant_frame_range_using_search(self.skeleton, motion_vector, step, self.foot_definitions,
                                                                           self.foot_lift_search_window, self.foot_lift_tolerance)
                leg_state_model = create_leg_state_model(step_plant_range, step.start_frame, step.end_frame, self.foot_definitions)
                step_ground_contacts = convert_plant_range_to_ground_contacts(step.start_frame, step.end_frame, step_plant_range)
                ground_contacts.update(step_ground_contacts)
                for frame_idx, joint_names in list(step_ground_contacts.items()):
                    constraints[frame_idx] = self.generate_grounding_constraints(motion_vector.frames, frame_idx,
                                                                                joint_names)

        self.set_smoothing_constraints(motion_vector.frames, constraints)

        n_frames = len(motion_vector.frames)
        blend_ranges = self.generate_blend_ranges(constraints, n_frames)

        return constraints, blend_ranges, ground_contacts

    def generate(self, motion_vector, ground_contacts=None):
        self.position_constraint_buffer = dict()
        self.orientation_constraint_buffer = dict()

        constraints = dict()
        if ground_contacts is None:
            ground_contacts = self.detect_ground_contacts(motion_vector.frames, self.contact_joints)
        # generate constraints
        for frame_idx, joint_names in enumerate(ground_contacts):
            constraints[frame_idx] = self.generate_grounding_constraints(motion_vector.frames, frame_idx, joint_names)

        self.set_smoothing_constraints(motion_vector.frames, constraints)

        n_frames = len(motion_vector.frames)
        blend_ranges = self.generate_blend_ranges(constraints, n_frames)
        return constraints, blend_ranges

    def generate_blend_ranges(self, constraints, n_frames):
        blend_ranges = dict()
        blend_ranges[self.right_foot] = []
        blend_ranges[self.left_foot] = []

        state = dict()
        state[self.right_foot] = -1
        state[self.left_foot] = -1

        joint_names = [self.right_foot, self.left_foot]

        for frame_idx in range(n_frames):
            if frame_idx in constraints:
                for j in joint_names:
                    constrained_joints = [c.joint_name for c in constraints[frame_idx]]
                    if j in constrained_joints:
                        if state[j] == -1:
                            # start new constrained range
                            blend_ranges[j].append([frame_idx, n_frames])
                            state[j] = len(blend_ranges[j])-1
                        else:
                            # update constrained range
                            idx = state[j]
                            blend_ranges[j][idx][1] = frame_idx
                    else:
                        state[j] = -1  # stop constrained range no constraint defined
            else:
                for c in joint_names:
                    state[c.joint_name] = -1  # stop constrained range no constraint defined
        return blend_ranges

    def generate_grounding_constraints(self, frames, frame_idx, joint_names):
        self.position_constraint_buffer[frame_idx] = dict()
        self.orientation_constraint_buffer[frame_idx] = dict()

        new_constraints = []
        end_frame = frame_idx + self.constraint_generation_range
        for side in self.foot_definitions:
            foot_joints = self.foot_definitions[side]
            c = None
            if foot_joints["heel"] in joint_names and foot_joints["toe"] in joint_names:
                c = self.generate_grounding_constraint_from_heel_and_toe(frames, foot_joints["ankle"], foot_joints["heel"], foot_joints["toe"], frame_idx, end_frame)
            elif foot_joints["heel"] in joint_names:
                c = self.generate_grounding_constraint_from_heel(frames, foot_joints["ankle"], foot_joints["heel"], frame_idx, end_frame)
            elif foot_joints["toe"] in joint_names:
                c = self.generate_grounding_constraint_from_toe(frames, foot_joints["ankle"], foot_joints["toe"], frame_idx, end_frame)
            if c is not None:
                c.heel_offset = self.skeleton.nodes[foot_joints["heel"]].offset
                new_constraints.append(c)
        return new_constraints

    def generate_grounding_constraint_from_heel_and_toe(self, frames, ankle_joint_name, heel_joint_name, toe_joint_name, frame_idx, end_frame):
        """ create constraint on ankle position and orientation """
        # get target global ankle orientation based on the direction between grounded heel and toe
        ct = self.get_previous_joint_position_from_buffer(frames, frame_idx, end_frame, toe_joint_name)
        ct[1] = self.scene_interface.get_height(ct[0], ct[2])
        ch = self.get_previous_joint_position_from_buffer(frames, frame_idx, end_frame, heel_joint_name)
        ch[1] = self.scene_interface.get_height(ct[0], ct[2])
        target_direction = normalize(ct - ch)
        t = self.skeleton.nodes[toe_joint_name].get_global_position(frames[frame_idx])
        h = self.skeleton.nodes[heel_joint_name].get_global_position(frames[frame_idx])
        original_direction = normalize(t-h)

        global_delta_q = quaternion_from_vector_to_vector(original_direction, target_direction)
        global_delta_q = normalize(global_delta_q)

        m = self.skeleton.nodes[ankle_joint_name].get_global_matrix(frames[frame_idx])
        m[:3, 3] = [0, 0, 0]
        oq = quaternion_from_matrix(m)
        oq = normalize(oq)
        orientation = normalize(quaternion_multiply(global_delta_q, oq))

        self.orientation_constraint_buffer[frame_idx][ankle_joint_name] = orientation  # save orientation to buffer

        # set target ankle position based on the  grounded heel and the global target orientation of the ankle
        m = quaternion_matrix(orientation)[:3,:3]
        target_heel_offset = np.dot(m, self.skeleton.nodes[heel_joint_name].offset)
        ca = ch - target_heel_offset
        #print "set ankle constraint both", ch, ca, target_heel_offset
        constraint = MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, None, orientation)
        constraint.toe_position = ct
        constraint.heel_position = ch
        return constraint

    def generate_grounding_constraint_from_heel(self, frames, ankle_joint_name, heel_joint_name, frame_idx, end_frame):
        """ create constraint on the ankle position without an orientation constraint"""
        #print "add ankle constraint from heel"
        ch = self.get_previous_joint_position_from_buffer(frames, frame_idx, end_frame, heel_joint_name)
        a = self.skeleton.nodes[ankle_joint_name].get_global_position(frames[frame_idx])
        h = self.skeleton.nodes[heel_joint_name].get_global_position(frames[frame_idx])
        ch[1] = self.scene_interface.get_height(ch[0], ch[2])  # set heel constraint on the ground
        target_heel_offset = a - h  # difference between unmodified heel and ankle
        ca = ch + target_heel_offset  # move ankle so heel is on the ground
        #print "set ankle constraint single", ch, ca, target_heel_offset
        constraint = MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, None, None)
        constraint.heel_position = ch
        return constraint

    def generate_grounding_constraint_from_toe(self, frames, ankle_joint_name, toe_joint_name, frame_idx, end_frame):
        """ create a constraint on the ankle position based on the toe constraint position"""
        #print "add toe constraint"
        ct = self.get_previous_joint_position_from_buffer(frames, frame_idx, end_frame, toe_joint_name)
        ct[1] = self.scene_interface.get_height(ct[0], ct[2])  # set toe constraint on the ground
        a = self.skeleton.nodes[ankle_joint_name].get_global_position(frames[frame_idx])
        t = self.skeleton.nodes[toe_joint_name].get_global_position(frames[frame_idx])

        target_toe_offset = a - t  # difference between unmodified toe and ankle at the frame
        ca = ct + target_toe_offset  # move ankle so toe is on the ground
        constraint = MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, None, None)

        constraint.toe_position = ct
        constraint.global_toe_offset = target_toe_offset
        return constraint

    def set_smoothing_constraints(self, frames, constraints):
        """ Set orientation constraints to singly constrained frames to smooth the foot orientation (Section 4.2)
        """
        for frame_idx in constraints:
            for constraint in constraints[frame_idx]:
                if constraint.joint_name not in self.orientation_constraint_buffer[frame_idx]:  # singly constrained
                    self.set_smoothing_constraint(frames, constraint, frame_idx, self.smoothing_constraints_window)

    def set_smoothing_constraint(self, frames, constraint, frame_idx, window):
        backward_hit = None
        forward_hit = None

        start_window = max(frame_idx - window, 0)
        end_window = min(frame_idx + window, len(frames))

        # look backward
        search_window = list(range(start_window, frame_idx))
        for f in reversed(search_window):
            if constraint.joint_name in self.orientation_constraint_buffer[f]:
                backward_hit = f
                break
        # look forward
        for f in range(frame_idx, end_window):
            if constraint.joint_name in self.orientation_constraint_buffer[f]:
                forward_hit = f
                break

        # update q
        oq = get_global_orientation(self.skeleton, constraint.joint_name, frames[frame_idx])
        if backward_hit is not None and constraint.toe_position is not None:
            bq = self.orientation_constraint_buffer[backward_hit][constraint.joint_name]
            j = frame_idx - backward_hit
            t = float(j + 1) / (window + 1)
            global_q = normalize(quaternion_slerp(bq, oq, t, spin=0, shortestpath=True))
            constraint.orientation = normalize(global_q)
            delta = get_quat_delta(oq, global_q)
            delta = normalize(delta)
            # rotate stored vector by global delta
            if constraint.joint_name == self.left_foot:

                constraint.position = regenerate_ankle_constraint_with_new_orientation2(constraint.toe_position,
                                                                                        constraint.global_toe_offset,
                                                                                        delta)
            else:
                constraint.position = regenerate_ankle_constraint_with_new_orientation2(constraint.toe_position,
                                                                                        constraint.global_toe_offset,
                                                                                        delta)

        elif forward_hit is not None and constraint.heel_position is not None:
            k = forward_hit - frame_idx
            t = float(k + 1) / (window + 1)
            fq = self.orientation_constraint_buffer[forward_hit][constraint.joint_name]
            global_q = normalize(quaternion_slerp(oq, fq, t, spin=0, shortestpath=True))
            constraint.orientation = normalize(global_q)
            constraint.position = regenerate_ankle_constraint_with_new_orientation(constraint.heel_position,
                                                                                   constraint.heel_offset,
                                                                                   constraint.orientation)

    def get_previous_joint_position_from_buffer(self, frames, frame_idx, end_frame, joint_name):
        """ Gets the joint position of the previous frame from the buffer if it exists.
            otherwise the position is calculated for the current frame and updated in the buffer
        """
        prev_frame_idx = frame_idx - 1
        prev_p = self.get_joint_position_from_buffer(prev_frame_idx, joint_name)
        if prev_p is not None:
            self.position_constraint_buffer[frame_idx][joint_name] = prev_p
            return prev_p
        else:
            self.update_joint_position_in_buffer(frames, frame_idx, end_frame, joint_name)
            p = self.position_constraint_buffer[frame_idx][joint_name]
            p[1] = self.scene_interface.get_height(p[0], p[2])
            #print "joint constraint",joint_name, p
            return p

    def get_joint_position_from_buffer(self, frame_idx, joint_name):
        if frame_idx not in self.position_constraint_buffer:
            return None
        if joint_name not in self.position_constraint_buffer[frame_idx]:
            return None
        return self.position_constraint_buffer[frame_idx][joint_name]

    def update_joint_position_in_buffer(self, frames, frame_idx, end_frame, joint_name):
        end_frame = min(end_frame, frames.shape[0])
        if frame_idx not in self.position_constraint_buffer:
            self.position_constraint_buffer[frame_idx] = dict()
        if joint_name not in self.position_constraint_buffer[frame_idx]:
            p = get_average_joint_position(self.skeleton, frames, joint_name, frame_idx, end_frame)
            #p[1] = self.scene_interface.get_height(p[0], p[2])
            #print "add", joint_name, p, frame_idx, end_frame
            self.position_constraint_buffer[frame_idx][joint_name] = p

    def generate_ankle_constraints_legacy(self, frames, frame_idx, joint_names, prev_constraints, prev_joint_names):
        end_frame = frame_idx + 10
        new_constraints = dict()
        temp_constraints = {"left": None, "right": None}
        if self.right_heel and self.right_toe in joint_names:
            if self.right_heel and self.right_toe in prev_joint_names:
                temp_constraints["right"] = prev_constraints["right"]
            else:
                temp_constraints["right"] = self.create_ankle_constraints_from_heel_and_toe(frames, self.right_foot, self.right_heel, frame_idx, end_frame)
        if self.left_heel and self.left_toe in joint_names:
            if joint_names == prev_joint_names:
                temp_constraints["left"] = prev_constraints["left"]
            else:
                temp_constraints["left"] = self.create_ankle_constraints_from_heel_and_toe(frames, self.left_foot,self.left_heel,  frame_idx, end_frame)
        for side in temp_constraints:
            new_constraints[side] = temp_constraints[side][0]
        return new_constraints

    def create_ankle_constraints_from_heel_and_toe(self, frames, ankle_joint_name, heel_joint_name, start_frame, end_frame):
        """ create constraint on ankle position and orientation """
        constraints = dict()
        ct = get_average_joint_position(self.skeleton, frames, heel_joint_name, start_frame, end_frame)
        ct[1] = self.scene_interface.get_height(ct[0], ct[2])
        pa = get_average_joint_position(self.skeleton, frames, ankle_joint_name, start_frame, end_frame)
        ph = get_average_joint_position(self.skeleton, frames, heel_joint_name, start_frame, end_frame)
        delta = ct - ph
        ca = pa + delta
        avg_direction = None
        if len(self.skeleton.nodes[ankle_joint_name].children) > 0:
            child_joint_name = self.skeleton.nodes[ankle_joint_name].children[0].node_name
            avg_direction = get_average_joint_direction(self.skeleton, frames, ankle_joint_name, child_joint_name,
                                                        start_frame, end_frame)
        #print "constraint ankle at",pa, ct,ca, start_frame, end_frame, avg_direction
        for frame_idx in range(start_frame, end_frame):
            c = MotionGroundingConstraint(frame_idx, ankle_joint_name, ca, avg_direction)
            constraints[frame_idx] = []
            constraints[frame_idx].append(c)
        return constraints


