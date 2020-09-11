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
# encoding: UTF-8
__author__ = 'hadu01'

import numpy as np
import math
from transformations.transformations import _AXES2TUPLE, _TUPLE2AXES, _NEXT_AXIS, quaternion_from_euler
LEN_CARTESIAN = 3
LEN_QUATERNION = 4


def diff_quat(q0, q1):
    """
    Args:
            q0, q1: (qw, qx, qy, qz)

    Returns:
            float: angle in radians
    """
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    d1 = w1*w1+x1*x1+y1*y1+z1*z1

    w =  x1 * x0 + y1 * y0 + z1 * z0 + w1 * w0
    x = -x1 * w0 - y1 * z0 + z1 * y0 + w1 * x0
    y =  x1 * z0 - y1 * w0 - z1 * x0 + w1 * y0
    z = -x1 * y0 + y1 * x0 - z1 * w0 + w1 * z0
    return (w, x, y, z)


def euler_matrix_jac(ai, aj, ak, axes='rxyz', der_axis='x'):
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak
    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    M = np.zeros([4, 4])
    # rotation matrix
    # if repetition:
    #     M[i, i] = cj
    #     M[i, j] = sj * si
    #     M[i, k] = sj * ci
    #     M[j, i] = sj * sk
    #     M[j, j] = -cj * ss + cc
    #     M[j, k] = -cj * cs - sc
    #     M[k, i] = -sj * ck
    #     M[k, j] = cj * sc + cs
    #     M[k, k] = cj * cc - ss
    # else:
    #     M[i, i] = cj * ck
    #     M[i, j] = sj * sc - cs
    #     M[i, k] = sj * cc + ss
    #     M[j, i] = cj * sk
    #     M[j, j] = sj * ss + cc
    #     M[j, k] = sj * cs - sc
    #     M[k, i] = -sj
    #     M[k, j] = cj * si
    #     M[k, k] = cj * ci
    '''
    first order derivatives:
    for ai:
    d(si)/d(ai) = ci, d(sj)/d(ai) = 0, d(sk)/d(ai) = 0
    d(ci)/d(ai) = -si, d(cj)/d(ai) = 0, d(ck)/d(ai) = 0
    d(cc)/d(ai) = -si * ck = -sc, d(cs)/d(ai) = -si * sk = -ss
    d(sc)/d(ai) = ci * ck = cc, d(ss)/d(ai) = ci * sk = cs

    for aj:
    d(si)/d(aj) = 0, d(sj)/d(aj) = cj, d(sk)/d(aj) = 0
    d(ci)/d(aj) = 0, d(cj)/d(aj) = -sj, d(ck)/d(aj) = 0
    d(cc)/d(aj) = 0, d(cs)/d(aj) = 0
    d(sc)/d(aj) = 0, d(ss)/d(aj) = 0

    for ak:
    d(si)/d(ak) = 0, d(sj)/d(ak) = 0, d(sk)/d(ak) = ck
    d(ci)/d(ak) = 0, d(cj)/d(ak) = 0, d(ck)/d(ak) = -sk
    d(cc)/d(ak) = -ci * sk = -cs, d(cs)/d(ak) = ci * ck = cc
    d(sc)/d(ak) = -si * sk = -ss, d(ss)/d(ak) = si * ck = sc
    '''
    if repetition:
        if der_axis == 'x':
            M[i, i] = 0
            M[i, j] = sj * ci
            M[i, k] = -sj * si
            M[j, i] = 0
            M[j, j] = -cj * cc + sc
            M[j, k] = cj * ss - cc
            M[k, i] = 0
            M[k, j] = cj * cc - ss
            M[k, k] = -cj * sc - cs
        elif der_axis == 'y':
            M[i, i] = -sj
            M[i, j] = cj * si
            M[i, k] = cj * ci
            M[j, i] = cj * sk
            M[j, j] = sj * ss
            M[j, k] = sj * cs
            M[k, i] = -cj * ck
            M[k, j] = -sj * sc
            M[k, k] = -sj * cc
        elif der_axis == 'z':
            M[i, i] = 0
            M[i, j] = 0
            M[i, k] = 0
            M[j, i] = sj * ck
            M[j, j] = -cj * sc - cs
            M[j, k] = -cj * cc + ss
            M[k, i] = sj * sk
            M[k, j] = -cj * ss + cc
            M[k, k] = -cj * cs - sc
        else:
            raise ValueError('Unknown axis type!')
    else:
        if der_axis == 'x':
            if not frame:
                M[i, i] = 0
                M[i, j] = sj * cc + ss
                M[i, k] = -sj * sc + cs
                M[j, i] = 0
                M[j, j] = sj * cs - sc
                M[j, k] = -sj * ss - cc
                M[k, i] = 0
                M[k, j] = cj * ci
                M[k, k] = -cj * si
                # if not parity:
                #     M[i, i] = 0
                #     M[i, j] = sj * cc + ss
                #     M[i, k] = -sj * sc + cs
                #     M[j, i] = 0
                #     M[j, j] = sj * cs - sc
                #     M[j, k] = -sj * ss - cc
                #     M[k, i] = 0
                #     M[k, j] = cj * ci
                #     M[k, k] = -cj * si
                # else:
                #     M[i, i] = 0
                #     M[i, j] = -sj * cc + ss
                #     M[i, k] = -sj * sc - cs
                #     M[j, i] = 0
                #     M[j, j] = -sj * cs - sc
                #     M[j, k] = -sj * ss + cc
                #     M[k, i] = 0
                #     M[k, j] = -cj * ci
                #     M[k, k] = -cj * si
            else:
                M[i, i] = -cj * sk
                M[i, j] = -sj * ss - cc
                M[i, k] = -sj * cs + sc
                M[j, i] = cj * ck
                M[j, j] = sj * sc - cs
                M[j, k] = sj * cc + ss
                M[k, i] = 0
                M[k, j] = 0
                M[k, k] = 0
                # if not parity:
                #     M[i, i] = -cj * sk
                #     M[i, j] = -sj * ss - cc
                #     M[i, k] = -sj * cs + sc
                #     M[j, i] = cj * ck
                #     M[j, j] = sj * sc - cs
                #     M[j, k] = sj * cc + ss
                #     M[k, i] = 0
                #     M[k, j] = 0
                #     M[k, k] = 0
                # else:
                #     M[i, i] = -cj * sk
                #     M[i, j] = -sj * ss + cc
                #     M[i, k] = -sj * cs - sc
                #     M[j, i] = -cj * ck
                #     M[j, j] = -sj * sc - cs
                #     M[j, k] = -sj * cc + ss
                #     M[k, i] = 0
                #     M[k, j] = 0
                #     M[k, k] = 0
        elif der_axis == 'y':
            M[i, i] = -sj * ck
            M[i, j] = cj * sc
            M[i, k] = cj * cc
            M[j, i] = -sj * sk
            M[j, j] = cj * ss
            M[j, k] = cj * cs
            M[k, i] = -cj
            M[k, j] = -sj * si
            M[k, k] = -sj * ci
            # if not parity:
            #     M[i, i] = -sj * ck
            #     M[i, j] = cj * sc
            #     M[i, k] = cj * cc
            #     M[j, i] = -sj * sk
            #     M[j, j] = cj * ss
            #     M[j, k] = cj * cs
            #     M[k, i] = -cj
            #     M[k, j] = -sj * si
            #     M[k, k] = -sj * ci
            # else:
            #     M[i, i] = -sj * ck
            #     M[i, j] = -cj * sc
            #     M[i, k] = -cj * cc
            #     M[j, i] = -sj * sk
            #     M[j, j] = -cj * ss
            #     M[j, k] = -cj * cs
            #     M[k, i] = cj
            #     M[k, j] = -sj * si
            #     M[k, k] = -sj * ci
        elif der_axis == 'z':
            if not frame:
                M[i, i] = -cj * sk
                M[i, j] = -sj * ss - cc
                M[i, k] = -sj * cs + sc
                M[j, i] = cj * ck
                M[j, j] = sj * sc - cs
                M[j, k] = sj * cc + ss
                M[k, i] = 0
                M[k, j] = 0
                M[k, k] = 0
                # if not parity:
                #     M[i, i] = -cj * sk
                #     M[i, j] = -sj * ss - cc
                #     M[i, k] = -sj * cs + sc
                #     M[j, i] = cj * ck
                #     M[j, j] = sj * sc - cs
                #     M[j, k] = sj * cc + ss
                #     M[k, i] = 0
                #     M[k, j] = 0
                #     M[k, k] = 0
                # else:
                #     M[i, i] = -cj * sk
                #     M[i, j] = -sj * ss + cc
                #     M[i, k] = -sj * cs - sc
                #     M[j, i] = -cj * ck
                #     M[j, j] = -sj * sc - cs
                #     M[j, k] = -sj * cc + ss
                #     M[k, i] = 0
                #     M[k, j] = 0
                #     M[k, k] = 0
            else:
                M[i, i] = 0
                M[i, j] = sj * cc + ss
                M[i, k] = -sj * sc + cs
                M[j, i] = 0
                M[j, j] = sj * cs - sc
                M[j, k] = -sj * ss - cc
                M[k, i] = 0
                M[k, j] = cj * ci
                M[k, k] = -cj * si
                # if not parity:
                #     M[i, i] = 0
                #     M[i, j] = sj * cc + ss
                #     M[i, k] = -sj * sc + cs
                #     M[j, i] = 0
                #     M[j, j] = sj * cs - sc
                #     M[j, k] = -sj * ss - cc
                #     M[k, i] = 0
                #     M[k, j] = cj * ci
                #     M[k, k] = -cj * si
                # else:
                #     M[i, i] = 0
                #     M[i, j] = -sj * cc + ss
                #     M[i, k] = -sj * sc - cs
                #     M[j, i] = 0
                #     M[j, j] = -sj * cs - sc
                #     M[j, k] = -sj * ss + cc
                #     M[k, i] = 0
                #     M[k, j] = -cj * ci
                #     M[k, k] = -cj * si
        else:
            raise ValueError('Unknown axis type!')
        if parity:
            M = -M
    return M


def quaternion_inv(q):
    """
    Inverse of quaternion q
    Args:
            q: (qw, qx, qy, qz)
    """
    w, x, y, z = q
    d = w*w + x*x + y*y + z*z
    q_inv = (w/d, -x/d, -y/d, -z/d)
    return q_inv


def normalize_quaternion(q):
    q = np.asarray(q)
    normalized_q = q/np.linalg.norm(q)
    return normalized_q


def error_measure_3d_mat(raw_data,
                         reconstructed_data):
    '''
    Compute the mean squared error bewteen original data and reconstructed data
    The data matrix is array3D: n_samples * n_frames * n_dims
    '''
    raw_data = np.asarray(raw_data)
    reconstructed_data = np.asarray(reconstructed_data)
    assert raw_data.shape == reconstructed_data.shape
    diff = raw_data - reconstructed_data
    n_samples, n_frames, n_dim = diff.shape
    err = 0
    for i in range(n_samples):
        for j in range(n_frames):
            err += np.linalg.norm(diff[i, j])
    err = err/(n_samples * n_frames)
    return err


def err_quat_data(raw_data,
                         reconstructed_data):
    raw_data = np.asarray(raw_data)
    reconstructed_data = np.asarray(reconstructed_data)
    assert raw_data.shape == reconstructed_data.shape
    diff = raw_data - reconstructed_data
    n_samples, n_frames, n_dim = diff.shape
    err = 0
    for i in range(n_samples):
        for j in range(n_frames):
            err += np.linalg.norm(diff[i, j][3:])

    err = err/(n_samples * n_frames)
    return err


def quat_to_expmap(q):
    """
    map quaternion to exponential map
    :param q: [qw, qx, qy, qz]
    :return:
    """
    q = q/np.linalg.norm(q)
    theta = 2 * np.arccos(q[0])
    if theta < 1e-3:
        auxillary = 1/(0.5 + theta ** 2 / 48)
    else:
        auxillary = theta / np.sin(0.5*theta)
    vx = q[1] * auxillary
    vy = q[2] * auxillary
    vz = q[3] * auxillary
    return np.array([vx, vy, vz])

def euler_to_expmap(euler_angles):
    '''
    convert euler angles to exponential map
    :param euler_angles: degree
    :return:
    '''
    euler_angles_rad = np.deg2rad(euler_angles)
    quat = quaternion_from_euler(*euler_angles)
    return quat_to_expmap(quat)


def angle_axis_from_quaternion(q):
    q = q/np.linalg.norm(q)
    theta = 2 * np.arccos(q[0])
    if theta < 1e-3:
        axis = np.array([0, 0, 0])
    else:
        x = q[1] / np.sin(0.5 * theta)
        y = q[2] / np.sin(0.5 * theta)
        z = q[3] / np.sin(0.5 * theta)
        axis = np.array([x, y, z])
    return theta, axis


def quat_to_logmap(q):
    q = q/np.linalg.norm(q)
    theta = 2 * np.arccos(q[0])
    if theta < 1e-3:
        auxillary = 1/(0.5 + theta ** 2 / 48)
    else:
        auxillary = theta / np.sin(0.5*theta)
    vx = q[1] * auxillary
    vy = q[2] * auxillary
    vz = q[3] * auxillary
    return np.array([vx, vy, vz])


def expmap_to_quat(exp_map):
    theta = np.linalg.norm(exp_map)
    print('theta is: ', theta)
    if np.abs(theta) < 1e-3:
        auxillary = 0.5 + theta ** 2 / 48
    else:
        auxillary = np.sin(0.5*theta) / theta
    qw = np.cos(theta/2)
    qx = exp_map[0]*auxillary
    qy = exp_map[1]*auxillary
    qz = exp_map[2]*auxillary
    return [qw, qx, qy, qz]


def logmap_to_quat(v):
    theta = np.linalg.norm(v)
    if theta < 1e-3:
        auxillary = 0.5 + theta ** 2 / 48
    else:
        auxillary = np.sin(0.5*theta) / theta
    qw = np.cos(theta/2)
    qx = v[0]*auxillary
    qy = v[1]*auxillary
    qz = v[2]*auxillary
    return [qw, qx, qy, qz]


def areQuatClose(quat1, quat2):
    dot = np.dot(quat1, quat2)
    if dot < 0.0:
        return False
    else:
        return True


def cartesian_splines_distance(raw_splines, reconstructed_splines, skeleton, weighted_error=True):
    """
    Calculate the Euclidean distance between motion represented as Cartesian splines
    :param raw_splines: Cartesian spline coefficience matrix
    :param reconstructed_splines: spline coefficience matrix
    :param skeleton:
    :param weighted_error:
    :return:
    """
    raw_splines = np.asarray(raw_splines)
    reconstructed_splines = np.asarray(reconstructed_splines)
    n_samples, n_basis, n_dims = raw_splines.shape
    assert n_dims%LEN_CARTESIAN == 0
    n_joints = n_dims/LEN_CARTESIAN
    if not weighted_error:
        return error_measure_3d_mat(raw_splines, reconstructed_splines)/n_joints
    else:
        joint_weights = skeleton.joint_weights[:-4]  # ignore the last two tool joints
        weight_vector = np.ones(n_dims)
        for i in range(n_joints):
            weight_vector[i*LEN_QUATERNION: (i+1)*LEN_QUATERNION] *= joint_weights[i]
        weight_mat = np.diag(weight_vector)
        return error_measure_3d_mat(np.dot(raw_splines, weight_mat),
                                    np.dot(reconstructed_splines, weight_mat))/n_joints

def angle_between_vectors(v1, v2):
    """
    Compute angle (in radians) between two 3d vectors (vq, v2)
    :param v1: (x, y, z)
    :param v2: (x, y, z)
    :return:
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    assert len(v1) == 3
    assert len(v2) == 3
    theta = np.arccos(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))
    return theta


def angle_between_2d_vector(v1, v2):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    theta = np.arccos(np.dot(v1, v2)/np.linalg.norm(v1) * np.linalg.norm(v2))
    return theta



def find_local_maximum(signal, neighbor_size):
    '''
    find local maximums of a signal in a given beighbor size
    :param signal:
    :param neighbor_size:
    :return:
    '''
    # mirror the signal on the boundary with k = stepsize/2
    signal_len = len(signal)
    localMax = []
    k = int(np.floor(neighbor_size/2))
    extendedSignal = np.zeros(signal_len + 2*k)
    extendedSignal[k: -k] = signal
    for i in range(k):
        extendedSignal[i] = signal[0]
        extendedSignal[-i-1] = signal[-1]
    for j in range(0, signal_len):
        searchArea = extendedSignal[j: j+neighbor_size]
        maximum = np.max(searchArea)
        if maximum == extendedSignal[j+k]:
            localMax.append(j)

    return localMax


def find_local_minimum(signal, neighbor_size):
    '''
    find local maximums of a signal in a given beighbor size
    :param signal:
    :param neighbor_size:
    :return:
    '''
    # mirror the signal on the boundary with k = stepsize/2
    signal_len = len(signal)
    localMin = []
    k = int(np.floor(neighbor_size/2))
    extendedSignal = np.zeros(signal_len + 2*k)
    for i in range(signal_len):
        extendedSignal[i+k] = signal[i]
    for i in range(k):
        extendedSignal[i] = signal[0]
        extendedSignal[-i-1] = signal[-1]
    for j in range(1, signal_len-1):
        searchArea = extendedSignal[j: j+neighbor_size]
        minimum = np.min(searchArea)
        if minimum == extendedSignal[j+k]:
            localMin.append(j)

    return localMin


def linear_weights_curve(window_size):
    '''
    create a linear weight curve, the starting and ending points are zero, and the peak is in the middle
    :param window_size:
    :return:
    '''
    middle_point = window_size/2.0
    weights = np.zeros(window_size)
    for i in range(0, int(np.floor(middle_point))+1):
        weights[i] = (1.0/middle_point) * i
    for i in range(int(np.ceil(middle_point)), window_size):
        weights[i] = -(1.0/middle_point) * i + 2.0
    return weights


def sin_weights_curve(window_size):
    '''
    create a sine weight curve, the starting and ending points are zero, and the peak is in the middle
    :param window_size:
    :return:
    '''
    return np.sin(np.arange(0, window_size) * np.pi / (window_size - 1))


def power_weights_curve(window_size):
    '''
    create a power function weight curve, the starting and ending points are zero, and the peak is in the middle
    :param window_size:
    :return:
    '''
    middle_point = window_size/2.0
    weights = np.zeros(window_size)
    a = (1.0/middle_point) ** 2
    for i in range(0, int(np.floor(middle_point))+1):
        weights[i] = a * (i**2)
    for i in range(int(np.ceil(middle_point)), window_size):
        weights[i] = a * (i-window_size + 1) ** 2
    return weights


def tdot_mat(mat, out=None):
    '''
    a copy of implementation of M*M_T from GPy
    :param mat:
    :param out:
    :return:
    '''
    from scipy.linalg import blas
    """returns np.dot(mat, mat.T), but faster for large 2D arrays of doubles."""
    if (mat.dtype != 'float64') or (len(mat.shape) != 2):
        return np.dot(mat, mat.T)
    nn = mat.shape[0]
    if out is None:
        out = np.zeros((nn, nn))
    else:
        assert(out.dtype == 'float64')
        assert(out.shape == (nn, nn))
        # FIXME: should allow non-contiguous out, and copy output into it:
        assert(8 in out.strides)
        # zeroing needed because of dumb way I copy across triangular answer
        out[:] = 0.0

    # # Call to DSYRK from BLAS
    mat = np.asfortranarray(mat)
    out = blas.dsyrk(alpha=1.0, a=mat, beta=0.0, c=out, overwrite_c=1,
                     trans=0, lower=0)

    symmetrify(out, upper=True)
    return np.ascontiguousarray(out)


def symmetrify(A, upper=False):
    """
    Take the square matrix A and make it symmetrical by copting elements from
    the lower half to the upper

    works IN PLACE.

    note: tries to use cython, falls back to a slower numpy version
    """
    _symmetrify_numpy(A, upper)


def _symmetrify_numpy(A, upper=False):
    triu = np.triu_indices_from(A,k=1)
    if upper:
        A.T[triu] = A[triu]
    else:
        A[triu] = A.T[triu]


def quaternion_from_vectors(vec1, vec2):
    vec1 = vec1/np.linalg.norm(vec1)
    vec2 = vec2/np.linalg.norm(vec2)
    inner_prod = np.dot(vec1, vec2)
    cross_prod = np.cross(vec1, vec2)
    cross_prod = cross_prod / np.linalg.norm(cross_prod)
    theta = np.arccos(inner_prod)
    sin_half = np.sin(theta/2.0)
    w = np.cos(theta/2.0)
    print(w)
    x = cross_prod[0] * sin_half
    y = cross_prod[1] * sin_half
    z = cross_prod[2] * sin_half
    return np.array([w, x, y, z])
