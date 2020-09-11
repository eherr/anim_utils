import json
import os
import argparse
from anim_utils.animation_data import BVHReader, MotionVector, SkeletonBuilder   
from anim_utils.retargeting.analytical import Retargeting, generate_joint_map

MODEL_DATA_PATH = "data"+os.sep+"models"


def load_json_file(path):
    with open(path, "rt") as in_file:
        return json.load(in_file)

def load_skeleton_model(skeleton_type):
    skeleton_model = dict()
    path = MODEL_DATA_PATH + os.sep+skeleton_type+".json"
    if os.path.isfile(path):
        data = load_json_file(path)
        skeleton_model = data["model"]
    else:
        print("Error: model unknown", path)
    return skeleton_model 


def load_skeleton(path, skeleton_type=None):
    bvh = BVHReader(path)   
    skeleton = SkeletonBuilder().load_from_bvh(bvh)
    skeleton.skeleton_model = load_skeleton_model(skeleton_type)
    return skeleton
    

def load_motion(path, skeleton_type=None):
    bvh = BVHReader(path)   
    mv = MotionVector()  
    mv.from_bvh_reader(bvh)  
    skeleton = SkeletonBuilder().load_from_bvh(bvh) 
    skeleton.skeleton_model = load_skeleton_model(skeleton_type)
    return skeleton, mv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run retargeting.')
    parser.add_argument('dest_skeleton', nargs='?', help='bvh filename')
    parser.add_argument('dest_skeleton_type', nargs='?', help='skeleton model name')
    parser.add_argument('src_motion', nargs='?', help='bvh filename')
    parser.add_argument('src_skeleton_type', nargs='?', help='skeleton model name')
    parser.add_argument('out_filename', nargs='?', help='filename')
    parser.add_argument('src_scale', nargs='?',default=1.0, help='float')
    parser.add_argument('place_on_ground', nargs='?',default=1, help='int')
    args = parser.parse_args()
    if args.src_motion is not None and args.dest_skeleton is not None and args.out_filename is not None:
        src_skeleton, src_motion = load_motion(args.src_motion, args.src_skeleton_type)
        dest_skeleton = load_skeleton(args.dest_skeleton, args.dest_skeleton_type)
        joint_map = generate_joint_map(src_skeleton.skeleton_model, dest_skeleton.skeleton_model)
        retargeting = Retargeting(src_skeleton, dest_skeleton, joint_map, float(args.src_scale), additional_rotation_map=None, place_on_ground=bool(args.place_on_ground))
        new_frames = retargeting.run(src_motion.frames, frame_range=None)
        target_motion = MotionVector()
        target_motion.frames = new_frames
        target_motion.skeleton = retargeting.target_skeleton
        target_motion.frame_time = src_motion.frame_time
        target_motion.n_frames = len(new_frames)
        target_motion.export(dest_skeleton, args.out_filename)

