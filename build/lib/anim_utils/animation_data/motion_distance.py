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
from math import sqrt
from transformations import euler_matrix
from .motion_concatenation import _align_point_clouds_2D


def _point_cloud_distance(a, b):
    """ Calculates the distance between two point clouds with equal length and
    corresponding indices
    """
    assert len(a) == len(b)
    distance = 0
    n_points = len(a)
    for i in range(n_points):
        d = [a[i][0] - b[i][0], a[i][1] - b[i][1], a[i][2] - b[i][2]]
        distance += sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    return distance / n_points


def convert_quat_frame_to_point_cloud(skeleton, frame, joints=None):
    points = []
    if joints is None:
        joints = [k for k, n in list(skeleton.nodes.items()) if len(n.children) > 0]
    for j in joints:
        p = skeleton.nodes[j].get_global_position(frame)
        points.append(p)
    return points


def _transform_point_cloud(point_cloud, theta, offset_x, offset_z):
    """ Transforms points in a point cloud by a rotation around the y axis and a translation
        along x and z
    """
    transformed_point_cloud = []
    for p in point_cloud:
        if p is not None:
            m = euler_matrix(0,np.radians(theta), 0)
            m[0, 3] = offset_x
            m[2, 3] = offset_z
            p = p.tolist()
            new_point = np.dot(m, p + [1])[:3]
            transformed_point_cloud.append(new_point)
    return transformed_point_cloud


def _transform_invariant_point_cloud_distance(pa,pb, weights=None):
    if weights is None:
        weights = [1.0] * len(pa)
    theta, offset_x, offset_z = _align_point_clouds_2D(pa, pb, weights)
    t_b = _transform_point_cloud(pb, theta, offset_x, offset_z)
    distance = _point_cloud_distance(pa, t_b)
    return distance


def frame_distance_measure(skeleton, a, b, weights=None):
    """ Converts frames into point clouds and calculates
        the distance between the aligned point clouds.

    Parameters
    ---------
    *skeleton: Skeleton
    \t skeleton used for forward kinematics
    *a: np.ndarray
    \t the parameters of a single frame representing a skeleton pose.
    *b: np.ndarray
    \t the parameters of a single frame representing a skeleton pose.
    *weights: list of floats
    \t weights giving the importance of points during the alignment and distance

    Returns
    -------
    Distance representing similarity of the poses.
    """
    assert len(a) == len(b)
    pa = convert_quat_frame_to_point_cloud(skeleton, a)
    pb = convert_quat_frame_to_point_cloud(skeleton, b)
    return _transform_invariant_point_cloud_distance(pa,pb, weights)
