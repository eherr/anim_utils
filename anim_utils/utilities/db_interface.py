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
Created on Mon Jun 20 2017

@author: Erik Herrmann
"""
import numpy as np
import json
import requests
import collections
import bson
import warnings
from ..animation_data import BVHReader, BVHWriter, MotionVector, SkeletonBuilder


DB_URL = "https://motion.dfki.de/8888"

#DB_URL = "https://localhost:8888/"


def call_rest_interface(url, method, data):
    method_url = url+method
    r = requests.post(method_url, data=json.dumps(data))
    return r.text

def call_bson_rest_interface(url, method, data):
    method_url = url+method
    r = requests.post(method_url, data=json.dumps(data))
    return r.content


def get_bvh_str_by_id_from_remote_db(url, clip_id, session=None):
    data = {"clip_id": clip_id}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "download_motion", data)
    return result_str


def get_motion_by_id_from_remote_db(url, clip_id, is_processed=False, session=None):
    
    data = {"clip_id": clip_id, "is_processed": int(is_processed)}
    if session is not None:
        data.update(session)
    print("get motion", data)
    result_str = call_bson_rest_interface(url, "get_motion", data)
    try:
        result_data = bson.loads(result_str)
    except:
        print("exception")
        result_data = None
    return result_data


def get_annotation_by_id_from_remote_db(url, clip_id, is_processed=False, session=None):
    data = {"clip_id": clip_id, "is_processed": int(is_processed)}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "download_annotation", data)
    return result_str



def get_time_function_by_id_from_remote_db(url, clip_id, session=None):
    data = {"clip_id": clip_id}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "get_time_function", data)
    return result_str


def get_motion_list_from_remote_db(url, collection_id, skeleton="", is_processed=False, session=None):
    data = {"collection_id": collection_id, "skeleton": skeleton, "is_processed":int(is_processed)}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "get_motion_list", data)
    try:
        result_data = json.loads(result_str)
    except Exception as e:
        print("exception", e.args)
        result_data = None
    return result_data

N_MAX_SIZE = 200000#00000


def split(motion_data):
    if type(motion_data) == dict:
        motion_data = json.dumps(motion_data)
    segments = []
    n_segments = (len(motion_data) // N_MAX_SIZE)  +1
    offset = 0
    for idx in range(n_segments):
        segments.append(motion_data[offset:offset+N_MAX_SIZE])
        offset+= N_MAX_SIZE
    return segments


def upload_motion_to_db(url, name, motion_data, collection, skeleton_name, meta_data, is_processed=False, session=None):
    parts = split(motion_data)
    for idx, part in enumerate(parts):
        #part_data = bytearray(part, "utf-8")
        data = {"data": part, "name": name, 
                "skeleton_name": skeleton_name, "meta_data": meta_data, 
                "collection": collection}
        data["part_idx"] = idx
        data["n_parts"] = len(parts)
        data["is_processed"] = is_processed
        if session is not None:
            data.update(session)
        result_text = call_rest_interface(url, "upload_motion", data)

def upload_bvh_to_db(url, name, bvh_str, collection, skeleton_model, meta_info, time_function=None, session=None):
    data = {"bvh_str": bvh_str, "name": name, 
            "skeleton_name": skeleton_model, "meta_info": meta_info, 
            "collection": collection}

    if session is not None:
        data.update(session)
    if time_function is not None:
        data["is_aligned"] = 1
        data["time_function"] = time_function
    else:
        data["is_aligned"] = 0
        data["time_function"] = ""
    result_text = call_rest_interface(url, "upload_motion", data)


def get_skeleton_from_remote_db(url, skeleton_type, session=None):
    data = {"skeleton_type": skeleton_type}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "get_skeleton", data)
    try:
        result_data = json.loads(result_str)
    except:
        result_data = None
    return result_data


def get_skeleton_model_from_remote_db(url, skeleton_type, session=None):
    data = {"skeleton_type": skeleton_type}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "get_skeleton_model", data)
    try:
        result_data = json.loads(result_str)
    except:
        result_data = None
    return result_data


def delete_motion_by_id_from_remote_db(url, clip_id, is_processed=False, session=None):
    data = {"clip_id": clip_id, "is_processed": int(is_processed)}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "delete_motion", data)
    return result_str

def create_new_collection_in_remote_db(url, name, col_type, parent_id, owner, session=None):
    data = {"name": name, "type": col_type, "parent_id": parent_id, "owner": owner}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "create_new_collection", data)

def replace_collection_in_remote_db(url, collection_id, name, col_type, parent_id, owner, session=None):
    data = {"id": collection_id ,"name": name, "type": col_type, "parent": parent_id, "owner": owner}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "replace_collection", data)

def get_collection_by_id(url, collection_id, session=None):
    data = {"id": collection_id}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "get_collection", data)
    try:
        result_data = json.loads(result_str)
    except:
        result_data = None
    return result_data


def get_collections_from_remote_db(url, parent_name="", collection_type="", session=None):
    data = {"parent_name": parent_name, "type": collection_type}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "get_collection_list", data)
    try:
        result_data = json.loads(result_str)
    except:
        result_data = None
    return result_data

def get_collections_by_parent_id_from_remote_db(url, parent_id, session=None):
    data = {"parent_id": parent_id}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "get_collection_list", data)
    try:
        result_data = json.loads(result_str)
    except:
        result_data = None
    return result_data

def delete_collection_from_remote_db(url, col_id, session=None):
    data = {"id": col_id}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "remove_collection", data)

def get_skeletons_from_remote_db(url, session=None):
    data = {}
    result_str = call_rest_interface(url, "get_skeleton_list", data)
    try:
        result_data = json.loads(result_str)
    except:
        result_data = None
    return result_data

def create_new_skeleton_in_db(url, name, skeleton_data, meta_data, session=None):
    data = {"name": name, "data": skeleton_data}
    if meta_data is not None:
        data["meta_data"] = meta_data
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "create_new_skeleton", data)

def replace_skeleton_in_remote_db(url, name, skeleton_data, meta_data, session=None):
    data = dict()
    data["name"] = name
    if skeleton_data is not None:
        data["data"] = skeleton_data
    if meta_data is not None:
        data["meta_data"] = meta_data
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "replace_skeleton", data)

def delete_skeleton_from_remote_db(url, name, session=None):
    data = {"name": name}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "remove_skeleton", data)


def replace_motion_in_db(url, motion_id, name, motion_data, collection, skeleton_name, meta_data, is_processed=False, session=None):
    data = {"motion_id": motion_id,  "data": motion_data, "name": name, 
            "skeleton_name": skeleton_name, "collection": collection, "is_processed": int(is_processed)}
    if meta_data is not None:
        data["meta_data"] =  meta_data
    if session is not None:
        data.update(session)
    result_text = call_rest_interface(url, "replace_motion", data)



def load_skeleton_from_db(db_url, skeleton_name, session=None):
    skeleton_data = get_skeleton_from_remote_db(db_url, skeleton_name, session)
    if skeleton_data is not None:
        skeleton = SkeletonBuilder().load_from_custom_unity_format(skeleton_data)
        skeleton_model = get_skeleton_model_from_remote_db(db_url, skeleton_name, session)
        skeleton.skeleton_model = skeleton_model
        return skeleton



def get_bvh_string(skeleton, frames):
    print("generate bvh string", len(skeleton.animated_joints))
    frames = np.array(frames)
    frames = skeleton.add_fixed_joint_parameters_to_motion(frames)
    frame_time = skeleton.frame_time
    bvh_writer = BVHWriter(None, skeleton, frames, frame_time, True)
    return bvh_writer.generate_bvh_string()

def get_motion_vector(skeleton, frames):
    print("generate motion vector", len(skeleton.animated_joints))
    frames = np.array(frames)
    #frames = skeleton.add_fixed_joint_parameters_to_motion(frames)
    frame_time = skeleton.frame_time
    mv = MotionVector()
    mv.frames = frames
    mv.n_frames = len(frames)
    mv.skeleton = skeleton
    return mv

def create_sections_from_keyframes(keyframes):
    sorted_keyframes = collections.OrderedDict(sorted(keyframes.items(), key=lambda t: t[1]))
    start = 0
    #end = n_canonical_frames
    semantic_annotation = collections.OrderedDict()
    for k, v in sorted_keyframes.items():
        print("set key",start, v)
        semantic_annotation[start] = {"start_idx":start,  "end_idx":v}
        start = v
    #semantic_annotation[start] = {"start_idx":start,  "end_idx":end}
    return list(semantic_annotation.values())



def get_bvh_from_str(bvh_str):
    bvh_reader = BVHReader("")
    lines = bvh_str.split("\n")
    # print(len(lines))
    lines = [l for l in lines if len(l) > 0]
    bvh_reader.process_lines(lines)
    return bvh_reader


def generate_training_data_from_bvh(bvh_data, animated_joints=None):
    motions = collections.OrderedDict()
    sections = collections.OrderedDict()
    temporal_data = collections.OrderedDict()
    skeleton = None
    for name, value in bvh_data.items():
        bvh_str = value["bvh_str"]
        print("process", name)
        mv = create_motion_vector_from_bvh(bvh_str, animated_joints)
        if skeleton is None:
            skeleton = mv.skeleton
        motions[name] = mv.frames
        if value["section_annotation"] is not None:
            sections[name] = value["section_annotation"]#create_sections_from_annotation(annotation)
        if value["time_function"] is not None:
            temporal_data[name] =  value["time_function"]
    return skeleton, motions, sections, temporal_data


def generate_training_data(motion_data, animated_joints=None):
    motions = collections.OrderedDict()
    sections = collections.OrderedDict()
    temporal_data = collections.OrderedDict()
    for name, value in motion_data.items():
        data = value["data"]
        motion_vector = MotionVector()
        motion_vector.from_custom_db_format(data)
        motions[name] = motion_vector.frames
        if value["section_annotation"] is not None:#
            v_type = type(value["section_annotation"])
            if v_type == list:
                sections[name] = value["section_annotation"]
            elif v_type == dict:
                sections[name] = list()#create_sections_from_annotation(annotation)
                print(value["section_annotation"])
                for section_key in value["section_annotation"]:
                    n_sections  = len(value["section_annotation"][section_key])
                    if n_sections == 1: # take only the first segment in the list
                        sections[name].append(value["section_annotation"][section_key][0])
                    else:
                        warnings.warn("number of annotations "+str(section_key)+" "+str(n_sections))
            else:
                warnings.warn("type unknown", name, v_type)
        if value["time_function"] is not None:
            temporal_data[name] = value["time_function"]
    return motions, sections, temporal_data

def create_keyframes_from_sections(sections):
    keyframes = dict()
    for i, s in enumerate(sections):
        keyframes["contact"+str(i)] = s["end_idx"]
    return keyframes



def create_motion_vector_from_bvh(bvh_str, animated_joints=None):
    bvh_reader = get_bvh_from_str(bvh_str)
    print("loaded motion", bvh_reader.frames.shape)
    if animated_joints is None:
        animated_joints = [key for key in list(bvh_reader.node_names.keys()) if not key.endswith("EndSite")]
    skeleton = SkeletonBuilder().load_from_bvh(bvh_reader, animated_joints)

    motion_vector = MotionVector()
    motion_vector.from_bvh_reader(bvh_reader, False, animated_joints)
    motion_vector.skeleton = skeleton
    return motion_vector


def create_motion_vector_from_json(motion_data):
    motion_vector = MotionVector()
    motion_vector.from_custom_db_format(motion_data)
    return motion_vector


def retarget_motion_in_db(db_url, retargeting, motion_id, motion_name, collection, skeleton_model_name, is_aligned=False, session=None):
    motion_data = get_motion_by_id_from_remote_db(db_url, motion_id, is_processed=is_aligned)
    if motion_data is None:
        print("Error: motion data is empty")
        return
    
    meta_info_str = get_annotation_by_id_from_remote_db(db_url, motion_id, is_processed=is_aligned)
    motion_vector = MotionVector()
    motion_vector.from_custom_db_format(motion_data)
    motion_vector.skeleton = retargeting.src_skeleton
    new_frames = retargeting.run(motion_vector.frames, frame_range=None)
    target_motion = MotionVector()
    target_motion.frames = new_frames
    target_motion.skeleton = retargeting.target_skeleton
    target_motion.frame_time = motion_vector.frame_time
    target_motion.n_frames = len(new_frames)
    m_data = target_motion.to_db_format()
    upload_motion_to_db(db_url, motion_name, m_data, collection, skeleton_model_name, meta_info_str, is_processed=is_aligned, session=session)


def authenticate(url, user, pw):
    data = {"username": user, "password": pw}
    result_str = call_rest_interface(url, "authenticate", data)
    try:
        result_data = json.loads(result_str)
    except:
        result_data = None
    return result_data


def start_cluster_job(url, imagename, job_name, job_desc, resources, session=None):
    data = {"imagename": imagename, "job_name": job_name,
            "job_desc": job_desc, "resources": resources}
    if session is not None:
        data.update(session)
    result_str = call_rest_interface(url, "start_cluster_job", data)
    try:
        result_data = json.loads(result_str)
    except:
        result_data = None
    return result_data
