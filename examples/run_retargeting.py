import json
import os
import argparse
from pathlib import Path
from anim_utils.animation_data import BVHReader, MotionVector, SkeletonBuilder   
from anim_utils.retargeting.analytical import Retargeting, generate_joint_map
from anim_utils.animation_data.bvh import write_euler_frames_to_bvh_file, convert_quaternion_to_euler_frames

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



def main(src_motion_dir, src_skeleton_type, dest_skeleton, dest_skeleton_type, out_dir, auto_scale=False, place_on_ground=False):
    
    dest_skeleton = load_skeleton(dest_skeleton, dest_skeleton_type)
    min_p, max_p = dest_skeleton.get_bounding_box()
    dest_height = (max_p[1] - min_p[1])
    os.makedirs(out_dir, exist_ok=True)
    p = Path(src_motion_dir)
    for filename in p.iterdir():
        if filename.suffix != ".bvh":
            continue
        ground_height = 5.5 -1.8 + 0.4 #0#85 
        ground_height = 5.5 -1.8 + 0.2 #0#85 
        ground_height *= 0.01
        src_skeleton, src_motion = load_motion(filename, src_skeleton_type)
        src_scale = 1.0
        if auto_scale:
            min_p, max_p = src_skeleton.get_bounding_box()
            src_height = (max_p[1] - min_p[1])
            src_scale = dest_height / src_height
        
        joint_map = generate_joint_map(src_skeleton.skeleton_model, dest_skeleton.skeleton_model)
        retargeting = Retargeting(src_skeleton, dest_skeleton, joint_map,src_scale, additional_rotation_map=None, place_on_ground=place_on_ground, ground_height=ground_height)        
        
        new_frames = retargeting.run(src_motion.frames, frame_range=None)
        frame_data = convert_quaternion_to_euler_frames(dest_skeleton, new_frames)            
        outfilename = out_dir + os.sep+filename.stem + ".bvh"
        print("write", outfilename, auto_scale, place_on_ground)
        write_euler_frames_to_bvh_file(outfilename, dest_skeleton, frame_data, src_motion.frame_time)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run retargeting.')
    parser.add_argument('dest_skeleton', nargs='?', help='BVH filename')
    parser.add_argument('dest_skeleton_type', nargs='?', help='skeleton model name')
    parser.add_argument('src_motion_dir', nargs='?', help='src BVH directory')
    parser.add_argument('src_skeleton_type', nargs='?', help='skeleton model name')
    parser.add_argument('out_dir', nargs='?', help='output BVH directory')
    parser.add_argument('--auto_scale', default=False, dest='auto_scale', action='store_true')
    parser.add_argument('--place_on_ground', default=False, dest='place_on_ground', action='store_true')
    args = parser.parse_args()
    if args.src_motion_dir is not None and args.dest_skeleton is not None and args.out_dir is not None:
        print(args.auto_scale)
        print(args.place_on_ground)
        main(args.src_motion_dir, args.src_skeleton_type, args.dest_skeleton, args.dest_skeleton_type, args.out_dir, bool(args.auto_scale), bool(args.place_on_ground))
