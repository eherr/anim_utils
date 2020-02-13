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
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 04 12:56:39 2015

@author: Han Du, Martin Manns, Erik Herrmann
"""
import os
import sys
import glob
import json
import collections
import math
import numpy as np
from datetime import datetime
from ..animation_data.utils import convert_euler_frames_to_cartesian_frames
#from ..animation_data.motion_concatenation import transform_euler_frames, transform_quaternion_frames
from ..animation_data.bvh import BVHWriter, BVHReader
sys.path.append(os.path.dirname(__file__))


def write_to_logfile(path, time_string, data):
    """ Appends json data to a text file.
        Creates the file if it does not exist.
        TODO use logging library instead
    """
    data_string = json.dumps(data, indent=4)
    line = time_string + ": \n" + data_string + "\n-------\n\n"
    if not os.path.isfile(path):
        file_handle = open(path, "w")
        file_handle.write(line)
        file_handle.close()
    else:
        with open(path, "a") as file_handle:
            file_handle.write(line)


def load_json_file(filename, use_ordered_dict=True):
    """ Load a dictionary from a file

    Parameters
    ----------
    * filename: string
    \tThe path to the saved json file.
    * use_ordered_dict: bool
    \tIf set to True dicts are read as OrderedDicts.
    """
    tmp = None
    with open(filename, 'r') as infile:
        if use_ordered_dict:
            tmp = json.JSONDecoder(
                object_pairs_hook=collections.OrderedDict).decode(
                infile.read())
        else:
            tmp = json.load(infile)
        infile.close()
    return tmp


def write_to_json_file(filename, serializable, indent=4):
    with open(filename, 'w') as outfile:
        tmp = json.dumps(serializable, indent=4)
        outfile.write(tmp)
        outfile.close()


def gen_file_paths(directory, mask='*mm.json'):
    """Generator of input file paths

    Parameters
    ----------

     * dir: String
    \tPath of input folder, in which the input files reside
     * mask: String, defaults to '*.bvh'
    \tMask pattern for file search

    """

    if not directory.endswith(os.sep):
        directory += os.sep

    for filepath in glob.glob(directory + mask):
        yield filepath


def get_morphable_model_directory(root_directory, morphable_model_type="motion_primitives_quaternion_PCA95"):
    """
    Return folder path to store result without trailing os.sep
    """
    data_dir_name = "data"
    process_step_dir_name = "3 - Motion primitives"
    mm_dir = os.sep.join([root_directory,
                          data_dir_name,
                          process_step_dir_name,
                          morphable_model_type])

    return mm_dir


def get_motion_primitive_directory(root_directory, elementary_action):
    """Return motion primitive file path
    """
    data_dir_name = "data"
    process_step_dir_name = "3 - Motion primitives"
    morphable_model_type = "motion_primitives_quaternion_PCA95"
    mm_path = os.sep.join([root_directory,
                           data_dir_name,
                           process_step_dir_name,
                           morphable_model_type,
                           'elementary_action_' + elementary_action
                           ])
    return mm_path


def get_aligned_motion_data(elementary_action,
                            motion_primitive,
                            data_repo=r'E:\workspace\repo'):
    aligned_folder = get_aligned_data_folder(elementary_action,
                                             motion_primitive,
                                             data_repo)
    bvhfiles = glob.glob(os.path.join(aligned_folder, '*.bvh'))
    motion_data = {}
    for bvhfile in bvhfiles:
        bvhreader = BVHReader(bvhfile)
        motion_data[bvhreader.filename] = bvhreader.frames
    return motion_data


def get_motion_primitive_path(root_directory, elementary_action,
                              motion_primitive):
    """Return motion primitive file
    """
    data_dir_name = "data"
    process_step_dir_name = "3 - Motion primitives"
    morphable_model_type = "motion_primitives_quaternion_PCA95"
    mm_path = os.sep.join([root_directory,
                           data_dir_name,
                           process_step_dir_name,
                           morphable_model_type,
                           'elementary_action_' + elementary_action,
                           '_'.join([elementary_action,
                                     motion_primitive,
                                     'quaternion',
                                     'mm.json'])
                           ])
    return mm_path


def get_transition_model_directory(root_directory):
    data_dir_name = "data"
    process_step_dir_name = "4 - Transition model"
    transition_dir = os.sep.join([root_directory,
                                  data_dir_name,
                                  process_step_dir_name, "output"])
    return transition_dir


def clean_path(path):
    path = path.replace('/', os.sep).replace('\\', os.sep)
    if os.sep == '\\' and '\\\\?\\' not in path:
        # fix for Windows 260 char limit
        relative_levels = len([directory for directory in path.split(os.sep)
                               if directory == '..'])
        cwd = [
            directory for directory in os.getcwd().split(
                os.sep)] if ':' not in path else []
        path = '\\\\?\\' + os.sep.join(cwd[:len(cwd) - relative_levels] + [
                                       directory for directory in path.split(os.sep) if directory != ''][relative_levels:])
    return path


def export_euler_frames_to_bvh(
        output_dir, skeleton, euler_frames, prefix="", start_pose=None, time_stamp=True):
    """ Exports a list of euler frames to a bvh file after transforming the frames
    to the start pose.

    Parameters
    ---------
    * output_dir : string
        directory without trailing os.sep
    * skeleton : Skeleton
        contains joint hiearchy information
    * euler_frames : np.ndarray
        Represents the motion
    * start_pose : dict
        Contains entry position and orientation each as a list with three components

    """
    if start_pose is not None:
        euler_frames = transform_euler_frames(
            euler_frames, start_pose["orientation"], start_pose["position"])
    if time_stamp:
        filepath = output_dir + os.sep + prefix + "_" + \
            str(datetime.now().strftime("%d%m%y_%H%M%S")) + ".bvh"
    elif prefix != "":
        filepath = output_dir + os.sep + prefix + ".bvh"
    else:
        filepath = output_dir + os.sep + "output" + ".bvh"
    print(filepath)
    BVHWriter(
        filepath,
        skeleton,
        euler_frames,
        skeleton.frame_time,
        is_quaternion=False)


def get_bvh_writer(skeleton, quat_frames, start_pose=None, is_quaternion=True):
    """
    Returns
    -------
    * bvh_writer: BVHWriter
        An instance of the BVHWriter class filled with Euler frames.
    """
    if start_pose is not None:
        quat_frames = transform_quaternion_frames(quat_frames,
                                                  start_pose["orientation"],
                                                  start_pose["position"])
    if len(quat_frames) > 0 and len(quat_frames[0]) < skeleton.reference_frame_length:
        quat_frames = skeleton.add_fixed_joint_parameters_to_motion(quat_frames)
    bvh_writer = BVHWriter(None, skeleton, quat_frames, skeleton.frame_time, is_quaternion)
    return bvh_writer


def export_frames_to_bvh_file(output_dir, skeleton, frames, prefix="", time_stamp=True, is_quaternion=True):
    """ Exports a list of frames to a bvh file

    Parameters
    ---------
    * output_dir : string
        directory without trailing os.sep
    * skeleton : Skeleton
        contains joint hiearchy information
    * frames : np.ndarray
        Represents the motion

    """
    bvh_writer = get_bvh_writer(skeleton, frames, is_quaternion=is_quaternion)
    if time_stamp:
        filepath = output_dir + os.sep + prefix + "_" + \
            str(datetime.now().strftime("%d%m%y_%H%M%S")) + ".bvh"
    elif prefix != "":
        filepath = output_dir + os.sep + prefix + ".bvh"
    else:
        filepath = output_dir + os.sep + "output" + ".bvh"
    print(filepath)
    bvh_writer.write(filepath)


def gen_spline_from_control_points(control_points, take=10):
    """
    :param control_points: a list of dictionary,
           each dictionary contains the position and orientation of one point
    :return: Parameterized spline
    """
    tmp = []
    count = 0
    for point in control_points:
        #print count, skip, count % skip
        if not math.isnan(sum(np.asarray(point['position']))) and count % take == 0:
            tmp.append(point['position'])
            #print "append"
        count+=1
    dim = len(tmp[0])
    print("number of points", len(tmp), len(control_points))
    spline = ParameterizedSpline(tmp, dim)
    # print tmp
    return spline


def load_collision_free_constraints(json_file):
    """
    load control points of collision free path for collided joints
    :param json_file:
    :return: a dictionary {'elementaryActionIndex': {'jointName': spline, ...}}
    """
    try:
        with open(json_file, "rb") as infile:
            json_data = json.load(infile)
    except IOError:
        print(('cannot read data from ' + json_file))
    collision_free_constraints = {}
    for action in json_data['modification']:
        collision_free_constraints[action["elementaryActionIndex"]] = {}
        for path in action["trajectories"]:
            spline = gen_spline_from_control_points(path['controlPoints'])
            collision_free_constraints[action["elementaryActionIndex"]][path['jointName']] = spline
    return collision_free_constraints


def get_aligned_data_folder(elementary_action,
                            motion_primitive,
                            repo_dir=None):
    if repo_dir is None:
        repo_dir = r'E:\workspace\repo'
    assert os.path.exists(repo_dir), ('Please configure morphablegraph repository directory!')
    data_folder = 'data'
    mocap_folder = '1 - Mocap'
    alignment_folder = '4 - Alignment'
    elementary_action_folder = 'elementary_action_' + elementary_action
    return os.sep.join([repo_dir,
                        data_folder,
                        mocap_folder,
                        alignment_folder,
                        elementary_action_folder,
                        motion_primitive])


def get_elementary_action_data_folder(elementary_action,
                                 repo_dir=None):
    if repo_dir is None:
        repo_dir = r'E:\workspace\repo'
    assert os.path.exists(repo_dir), ('Please configure morphablegraph repository directory!')
    data_folder = 'data'
    mocap_folder = '1 - Mocap'
    alignment_folder = '4 - Alignment'
    elementary_action_folder = 'elementary_action_' + elementary_action
    return os.sep.join([repo_dir,
                        data_folder,
                        mocap_folder,
                        alignment_folder,
                        elementary_action_folder])


def get_cut_data_folder(elementary_action,
                        motion_primitive,
                        repo_dir=None):
    if repo_dir is None:
        repo_dir = r'E:\workspace\repo'
    assert os.path.exists(repo_dir), ('Please configure morphablegraph repository directory!')
    data_folder = 'data'
    mocap_folder = '1 - Mocap'
    cut_folder = '3 - Cutting'
    elementary_action_folder = 'elementary_action_' + elementary_action
    return os.sep.join([repo_dir,
                        data_folder,
                        mocap_folder,
                        cut_folder,
                        elementary_action_folder,
                        motion_primitive])


def load_bvh_files_from_folder(data_folder):
    bvhfiles = glob.glob(os.path.join(data_folder, '*.bvh'))
    # order filenames alphabetically
    tmp = {}
    for item in bvhfiles:
        filename = os.path.split(item)[-1]
        bvhreader = BVHReader(item)
        tmp[filename] = bvhreader.frames
    motion_data_dic = collections.OrderedDict(sorted(tmp.items()))
    return motion_data_dic


def get_data_analysis_folder(elementary_action,
                             motion_primitive,
                             repo_dir=r'E:\workspace\repo'):
    '''
    get data analysis folder, repo_dir is the local path for data svn repository, e.g.: C:\repo
    :param elementary_action (str):
    :param motion_primitive (str):
    :param repo_dir (str):
    :return:
    '''
    mocap_data_analysis_folder = os.path.join(repo_dir, r'data\1 - MoCap\7 - Mocap analysis')
    assert os.path.exists(mocap_data_analysis_folder), ('Please configure path to mocap analysis folder!')
    elementary_action_folder = os.path.join(mocap_data_analysis_folder, 'elementary_action_' + elementary_action)
    if not os.path.exists(elementary_action_folder):
        os.mkdir(elementary_action_folder)
    motion_primitive_folder = os.path.join(elementary_action_folder, motion_primitive)
    if not os.path.exists(motion_primitive_folder):
        os.mkdir(motion_primitive_folder)
    return motion_primitive_folder


def get_semantic_motion_primitive_path(elementary_action,
                                       motion_primitive,
                                       datarepo_dir=None):
    if datarepo_dir is None:
        datarepo_dir = r'E:\workspace\repo'
    # map the old motion primitive name to the new name
    if motion_primitive == 'first':
        motion_primitive = 'reach'
    if motion_primitive == 'second':
        motion_primitive = 'retrieve'
    return os.path.join(datarepo_dir,
                        r'data\3 - Motion primitives\motion_primitives_quaternion_PCA95_temporal_semantic',
                        'elementary_action_' + elementary_action,
                        '_'.join([elementary_action,
                                  motion_primitive,
                                  'quaternion',
                                  'mm.json']))


def get_low_dimensional_spatial_file(elementary_action,
                                     motion_primitive):
    repo_dir = r'E:\workspace\repo'
    assert os.path.exists(repo_dir), ('Please configure morphablegraph repository directory!')
    data_folder = 'data'
    pca_folder = '2 - PCA'
    low_dimensional_data_folder = 'combine_spatial_temporal'
    filename = '_'.join([elementary_action,
                         motion_primitive,
                         'low_dimensional_motion_data.json'])
    return os.sep.join([repo_dir,
                        data_folder,
                        pca_folder,
                        low_dimensional_data_folder,
                        filename])


def get_motion_primitive_filepath(elementary_action,
                                  motion_primitive,
                                  repo_dir=r'E:\workspace\repo'):
    assert os.path.exists(repo_dir), ('Please configure morphablegraph repository directory!')
    data_folder = 'data'
    motion_primitive_folder = '3 - Motion primitives'
    quat_folder = 'motion_primitives_quaternion_PCA95'
    # non_semantic_folder = 'elementary_action_models'
    elementary_action_folder = 'elementary_action_' + elementary_action
    filename = '_'.join([elementary_action,
                         motion_primitive,
                         'quaternion_mm.json'])
    return os.sep.join([repo_dir,
                        data_folder,
                        motion_primitive_folder,
                        quat_folder,
                        # non_semantic_folder,
                        elementary_action_folder,
                        filename])


def create_pseudo_timewarping(aligned_data_folder):
    bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    timewarping_data = {}
    for item in bvhfiles:
        filename = os.path.split(item)[-1]
        bvhreader = BVHReader(item)
        frame_indices = range(len(bvhreader.frames))
        timewarping_data[filename] = frame_indices
    return timewarping_data


def load_aligned_data(elementary_action, motion_primitive):
    aligned_data_folder = get_aligned_data_folder(elementary_action, motion_primitive)
    aligned_motion_data = {}
    timewarping_file = os.path.join(aligned_data_folder, 'timewarping.json')
    if not os.path.exists(timewarping_file):
        print("##############  cannot find timewarping file, create pseudo timewapring data")
        timewarping_data = create_pseudo_timewarping(aligned_data_folder)
    else:
        timewarping_data = load_json_file(timewarping_file)
    bvhfiles = glob.glob(os.path.join(aligned_data_folder, '*.bvh'))
    for filename, time_index in timewarping_data.iteritems():
        aligned_motion_data[filename] = {}
        aligned_motion_data[filename]['warping_index'] = time_index
        filepath = os.path.join(aligned_data_folder, filename)
        assert filepath in bvhfiles, ('cannot find the file in the aligned folder')
        bvhreader = BVHReader(filepath)
        aligned_motion_data[filename]['frames'] = bvhreader.frames
    aligned_motion_data = collections.OrderedDict(sorted(aligned_motion_data.items()))
    return aligned_motion_data


def get_cubic_b_spline_knots(n_basis, n_canonical_frames):
    '''
    create cubic bspline knot list, the order of the spline is 4
    :param n_basis: number of knots
    :param n_canonical_frames: length of discrete samples
    :return:
    '''
    n_orders = 4
    knots = np.zeros(n_orders + n_basis)
    # there are two padding at the beginning and at the end
    knots[3: -3] = np.linspace(0, n_canonical_frames-1, n_basis-2)
    knots[-3:] = n_canonical_frames - 1
    return knots


def save_euler_frames_as_point_cloud(filename, skeleton, euler_frames):
    cartesian_frames = convert_euler_frames_to_cartesian_frames(skeleton, euler_frames)
    print(cartesian_frames.shape)
    point_cloud_data = {'has_skeleton': False,
                        'motion_data': cartesian_frames.tolist()}
    write_to_json_file(filename, point_cloud_data)


def export_joint_position_to_cloud_data(joint_position, filename, SKELETON):
    '''

    :param joint_position: n_frames * n_joints *3 (cartesian x y z)
    :param filename: 
    :return: 
    '''
    save_data = {'motion_data': joint_position.tolist(), 'has_skeleton': True, 'skeleton': SKELETON}
    write_to_json_file(filename, save_data)