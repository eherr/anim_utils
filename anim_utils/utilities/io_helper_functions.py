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
from ..animation_data.bvh import BVHWriter, BVHReader


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


def load_bvh_files_from_folder(data_folder):
    bvhfiles = glob.glob(os.path.join(data_folder, '*.bvh'))
    # order filenames alphabetically
    tmp = {}
    for item in bvhfiles:
        filename = os.path.split(item)[-1]
        bvhreader = BVHReader(item)
        tmp[filename] = bvhreader.frames
    motion_data = collections.OrderedDict(sorted(tmp.items()))
    return motion_data
