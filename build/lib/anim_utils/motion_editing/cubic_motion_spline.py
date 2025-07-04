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
from scipy.interpolate import interp1d
from transformations import quaternion_multiply, quaternion_inverse


def get_quaternion_delta(a, b):
    return quaternion_multiply(quaternion_inverse(b), a)


def add_frames(skeleton, a, b):
    """ returns c = a + b"""
    #print("add frames", len(a), len(b))
    c = np.zeros(len(a))
    c[:3] = a[:3] + b[:3]
    for idx, j in enumerate(skeleton.animated_joints):
        o = idx * 4 + 3
        q_a = a[o:o + 4]
        q_b = b[o:o + 4]
        #print(q_a,q_b)
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


def generate_knots(n_frames, n_knots):
    """https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LSQUnivariateSpline.html"""
    knots = np.linspace(0, n_frames - 1, n_knots)
    return knots


class CubicMotionSpline(object):
    def __init__(self, skeleton, splines, n_dims):
        self.skeleton = skeleton
        self.n_dims = n_dims
        self.splines = splines

    @classmethod
    def fit_frames(cls, skeleton, times, frames, kind='cubic'):
        if type(frames) is np.ndarray:
            n_frames, n_dims = frames.shape
        else:
            n_dims = 1
        splines = []
        for d in range(n_dims):
            s = interp1d(times, frames[:, d], kind=kind)
            splines.append(s)
        return CubicMotionSpline(skeleton, splines, n_dims)

    def evaluate(self, t):
        return np.asarray([s(t) for s in self.splines]).T

    def get_displacement_map(self, frames):
        d = []
        for idx, f in enumerate(frames):
            s_f = self.evaluate(idx)
            d.append(substract_frames(self.skeleton, f, s_f))
        return np.array(d)

    def to_frames(self, times):
        frames = []
        for t in times:
            frame = self.evaluate(t)
            frames.append(frame)
        return frames

    def add_spline_discrete(self, other, times):
        new_frames = []
        for t in times:
            f = add_frames(self.skeleton, self.evaluate(t), other.evaluate(t))
            new_frames.append(f)
        new_frames = np.array(new_frames)
        splines = []
        for d in range(self.n_dims):
            s = interp1d(times, new_frames[:, d], kind='cubic')
            splines.append(s)
        self.splines = splines

    def plot(self, x):
        import matplotlib.pyplot as plt
        for s in self.splines:
            plt.plot(x, s(x))
        plt.show()
