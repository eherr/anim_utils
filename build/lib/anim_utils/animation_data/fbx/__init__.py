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
from anim_utils.animation_data.motion_vector import MotionVector
from anim_utils.animation_data.skeleton_builder import SkeletonBuilder

has_fbx = True
try:
    import FbxCommon
except ImportError as e:
    has_fbx = False
    print("Info: Disable FBX IO due to missing library")
    pass


if has_fbx:
    from .fbx_import import load_fbx_file
    from .fbx_export import export_motion_vector_to_fbx_file
else:
    def load_fbx_file(file_path):
        raise NotImplementedError

    def export_motion_vector_to_fbx_file(skeleton, motion_vector, out_file_name):
        raise NotImplementedError



def load_skeleton_and_animations_from_fbx(file_path):
    skeleton_def, animations = load_fbx_file(file_path)
    skeleton = SkeletonBuilder().load_from_fbx_data(skeleton_def)
    anim_names = list(animations.keys())
    motion_vectors = dict()
    if len(anim_names) > 0:
        anim_name = anim_names[0]
        mv = MotionVector()
        mv.from_fbx(animations[anim_name], skeleton.animated_joints)
        motion_vectors[anim_name] = mv
    return skeleton, motion_vectors

