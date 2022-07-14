import os
import argparse
from pathlib import Path
from anim_utils.animation_data import MotionVector, SkeletonBuilder   
from anim_utils.animation_data.acclaim import parse_asf_file, parse_amc_file
from anim_utils.animation_data.bvh import write_euler_frames_to_bvh_file, convert_quaternion_to_euler_frames


def load_skeleton(filename):
    asf = parse_asf_file(filename)   
    skeleton = SkeletonBuilder().load_from_asf_data(asf)
    return skeleton
    

def load_motion(filename, skeleton):
    amc_frames = parse_amc_file(filename)   
    mv = MotionVector()  
    mv.from_amc_data(skeleton, amc_frames)  
    return mv

def main(motion_dir,out_dir):
    amc_files = []
    skeleton_file = None
    p = Path(motion_dir)
    for filename in p.iterdir():
        if filename.suffix == ".amc":
            amc_files.append(filename)
        elif filename.suffix == ".asf":
            skeleton_file = filename
    if skeleton_file is None:
        return
    skeleton = load_skeleton(skeleton_file)
    os.makedirs(out_dir, exist_ok=True)
    for filename in amc_files:
        src_motion = load_motion(str(filename), skeleton)
        outfilename = out_dir + os.sep+filename.stem + ".bvh"
        frame_data = convert_quaternion_to_euler_frames(skeleton, src_motion.frames)
        print("write", outfilename)
        write_euler_frames_to_bvh_file(outfilename, skeleton, frame_data, src_motion.frame_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export.')
    parser.add_argument('motion_dir', nargs='?', help='directory of amc files')
    parser.add_argument('out_dir', nargs='?', help='out directory')
    args = parser.parse_args()
    if args.motion_dir is not None and args.out_dir is not None:
        main(args.motion_dir, args.out_dir)
        
            