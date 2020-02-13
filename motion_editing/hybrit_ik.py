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
""" Hybrit IK using numerical IK based on an exponential displacement map and analytical IK
based on Section 5.1 of [1]

Use current configuration as q_0
Find v such that (q_i = q_0_i exp(v_i) for 0<i<n where n are the number of joints) reaches the constraints
i.e. optimize over exponential map representation of displacement based on constraints, i.e. sum of errors of constraints from v
Minimize also the displacement

[1] Lee, Jehee, and Sung Yong Shin. "A hierarchical approach to interactive motion editing for human-like figures."
Proceedings of the 26th annual conference on Computer graphics and interactive techniques. 1999.

"""

import numpy as np
from scipy.optimize import minimize
from .utils import convert_exp_frame_to_quat_frame, add_quat_frames


def ik_objective(x, skeleton, reference, constraints, weights, analytical_ik):
    d = convert_exp_frame_to_quat_frame(skeleton, x)
    d[:4] = [1,0,0,0]

    q_frame = add_quat_frames(skeleton, reference, d)

    # reduce displacement to original frame
    error = np.dot(x.T, np.dot(weights, x))

    # reduce distance to constraints
    #print q_frame.shape
    for c in constraints:
        if c.joint_name in analytical_ik:
            q_frame = analytical_ik[c.joint_name].apply2(q_frame, c.position, c.orientation)
        error += c.evaluate(skeleton, q_frame)
    return error


class HybritIK(object):
    def __init__(self, skeleton, ik_settings, verbose=False, objective=None):

        self.skeleton = skeleton
        self.n_joints = len(self.skeleton.animated_joints)
        if objective is None:
            self._objective = ik_objective
        else:
            self._objective = objective
        self.ik_settings = ik_settings
        self.verbose = verbose
        self._optimization_options = {'maxiter': self.ik_settings["max_iterations"], 'disp': self.verbose}
        self.joint_weights = np.eye(self.n_joints * 3)
        self.max_retries = 1
        self.success_threshold = 0.1
        self.analytical_ik = dict()

    def set_joint_weights(self, weights):
        self.joint_weights = np.dot(self.joint_weights, weights)

    def add_analytical_ik(self, end_effector, limb_joint, limb_root, joint_axis, local_end_effector_dir):
        from .analytical_inverse_kinematics import AnalyticalLimbIK
        self.analytical_ik[end_effector] = AnalyticalLimbIK(self.skeleton, limb_root, limb_joint, end_effector, joint_axis, local_end_effector_dir)

    def run(self, reference, constraints):
        guess = np.zeros(self.n_joints * 3)
        iter_counter = 2
        error = np.inf
        while error > self.success_threshold and iter_counter < self.max_retries:
            r = minimize(self._objective, guess, args=(self.skeleton, reference, constraints, self.joint_weights, self.analytical_ik),
                         method=self.ik_settings["optimization_method"],
                         tol=self.ik_settings["tolerance"],
                         options=self._optimization_options)
            guess = r["x"]
            error = self._objective(guess, self.skeleton, reference, constraints, self.joint_weights, self.analytical_ik)
        iter_counter += 1

        exp_frame = guess
        return exp_frame

    def modify_frame(self, reference, constraints):
        exp_frame = self.run(reference, constraints)
        d = convert_exp_frame_to_quat_frame(self.skeleton, exp_frame)
        q_frame = add_quat_frames(self.skeleton, reference, d)
        for c in constraints:
            if c.joint_name in self.analytical_ik:
                q_frame = self.analytical_ik[c.joint_name].apply2(q_frame, c.position, c.orientation)
        return q_frame
