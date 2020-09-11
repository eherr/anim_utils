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

import time
import numpy as np
from scipy.optimize import minimize
from ..utilities.log import write_log, write_message_to_log, LOG_MODE_DEBUG


def obj_inverse_kinematics(s, data):
    pose, free_joints, target_joint, target_position, target_direction = data
    pose.set_channel_values(s, free_joints)
    if target_direction is not None:
        parent_joint = pose.get_parent_joint(target_joint)
        pose.point_in_direction(parent_joint, target_direction)
    position = pose.evaluate_position(target_joint)
    d = position - target_position
    return np.dot(d, d)


class NumericalInverseKinematicsQuat(object):
    def __init__(self, skeleton_pose_model, ik_settings):
        self._ik_settings = ik_settings
        self.verbose = False
        self.pose = skeleton_pose_model
        self.success_threshold = self._ik_settings["success_threshold"]
        self.max_retries = self._ik_settings["max_retries"]
        self.optimize_orientation = self._ik_settings["optimize_orientation"]

    def _run_optimization(self, objective, initial_guess, data, cons=None):
         return minimize(objective, initial_guess, args=(data,),
                        method=self._ik_settings["optimization_method"],#"SLSQP",#best result using L-BFGS-B
                        constraints=cons, tol=self._ik_settings["tolerance"],
                        options={'maxiter': self._ik_settings["max_iterations"], 'disp': self.verbose})#,'eps':1.0

    def modify_pose(self, joint_name, target, direction=None):
        error = 0.0
        if joint_name in list(self.pose.free_joints_map.keys()):
            free_joints = self.pose.free_joints_map[joint_name]
            error = self._modify_using_optimization(joint_name, target, free_joints, direction)
        return error

    def modify_pose_general(self, constraint):
        free_joints = constraint.free_joints
        initial_guess = self._extract_free_parameters(free_joints)
        data = constraint.data(self, free_joints)
        if self.verbose:
            self.pose.set_channel_values(initial_guess, free_joints)
            p = self.pose.evaluate_position(constraint.joint_name)
            write_message_to_log(
                "Start optimization for joint " + constraint.joint_name + " " + str(len(initial_guess))
                + " " + str(len(free_joints)) + " " + str(p), LOG_MODE_DEBUG)
        cons = None
        #start = time.clock()
        error = np.inf
        iter_counter = 0
        result = None
        while error > self.success_threshold and iter_counter < self.max_retries:
            result = self._run_optimization(constraint.evaluate, initial_guess, data, cons)
            error = constraint.evaluate(result["x"], data)
            iter_counter += 1
        # write_log("finished optimization in",time.clock()-start,"seconds with error", error)#,result["x"].tolist(), initial_guess.tolist()
        if result is not None:
            self.pose.set_channel_values(result["x"], free_joints)
        return error

    def _extract_free_parameters(self, free_joints):
        """get parameters of joints from reference frame
        """
        parameters = list()
        for joint_name in free_joints:
            parameters += self.pose.extract_parameters(joint_name).tolist()
        return np.asarray(parameters)

    def _extract_free_parameter_indices(self, free_joints):
        """get parameter indices of joints from reference frame
        """
        indices = {}
        for joint_name in free_joints:
            indices[joint_name] = list(range(*self.pose.extract_parameters_indices(joint_name)))
        return indices

    def _modify_using_optimization(self, target_joint, target_position, free_joints, target_direction=None):
        initial_guess = self._extract_free_parameters(free_joints)
        data = self.pose, free_joints, target_joint, target_position, target_direction
        if self.verbose:
            write_message_to_log("Start optimization for joint " + target_joint + " " +  str(len(initial_guess))
                                 + " " + str(len(free_joints)), LOG_MODE_DEBUG)
        start = time.clock()
        cons = None#self.pose.generate_constraints(free_joints)
        result = self._run_optimization(obj_inverse_kinematics, initial_guess, data, cons)
        position = self.pose.evaluate_position(target_joint)
        error = np.linalg.norm(position-target_position)
        if self.verbose:
            write_message_to_log("Finished optimization in " + str(time.clock()-start) + " seconds with error " + str(error), LOG_MODE_DEBUG) #,result["x"].tolist(), initial_guess.tolist()
        self.pose.set_channel_values(result["x"], free_joints)
        return error

    def optimize_joint(self, objective, target_joint, target_position, target_orientation, free_joint):
        initial_guess = self.pose.extract_parameters(free_joint)  # self._extract_free_parameters([free_joint])
        data = self.pose, [free_joint], target_joint, target_position, None
        result = self._run_optimization(objective, initial_guess, data)
        # apply constraints here
        self.pose.apply_bounds(free_joint)
        return result["x"]

    def evaluate_delta(self, parameters, target_joint, target_position, free_joints):
        self.pose.set_channel_values(parameters, free_joints)
        position = self.pose.evaluate_position(target_joint)
        d = position - target_position
        return np.dot(d, d)
