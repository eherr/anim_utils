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
""" This module contains mappings from a standard set of human joints to different skeletons

"""
import numpy as np
import collections



ROCKETBOX_TOOL_BONES = [{
    "new_node_name": 'LeftToolEndSite',
    "parent_node_name": 'LeftHand',
    "new_node_offset": [6.1522069, -0.09354633, 3.33790343]
}, {
    "new_node_name": 'RightToolEndSite',
    "parent_node_name": 'RightHand',
    "new_node_offset": [6.1522069, 0.09354633, 3.33790343]
}, {
    "new_node_name": 'RightScrewDriverEndSite',
    "parent_node_name": 'RightHand',
    "new_node_offset": [22.1522069, -9.19354633, 3.33790343]
}, {
    "new_node_name": 'LeftScrewDriverEndSite',
    "parent_node_name": 'LeftHand',
    "new_node_offset": [22.1522069, 9.19354633, 3.33790343]
}
]
ROCKETBOX_FREE_JOINTS_MAP = {"LeftHand": ["Spine", "LeftArm", "LeftForeArm"],
                           "RightHand": ["Spine", "RightArm", "RightForeArm"],
                           "LeftToolEndSite": ["Spine", "LeftArm", "LeftForeArm"],
                           "RightToolEndSite": ["Spine", "RightArm", "RightForeArm"],  # , "RightHand"
                           "Head": [],
                           "RightScrewDriverEndSite": ["Spine", "RightArm", "RightForeArm"],
                           "LeftScrewDriverEndSite": ["Spine", "LeftArm", "LeftForeArm"]
                             }
ROCKETBOX_REDUCED_FREE_JOINTS_MAP = {"LeftHand": ["LeftArm", "LeftForeArm"],
                                   "RightHand": ["RightArm", "RightForeArm"],
                                   "LeftToolEndSite": ["LeftArm", "LeftForeArm"],
                                   "RightToolEndSite": ["RightArm", "RightForeArm"],
                                   "Head": [],
                                   "RightScrewDriverEndSite": ["RightArm", "RightForeArm"],
                                   "LeftScrewDriverEndSite": ["LeftArm", "LeftForeArm"]
                                     }
DEG2RAD = np.pi / 180
hand_bounds = [{"dim": 0, "min": 30 * DEG2RAD, "max": 180 * DEG2RAD},
               {"dim": 1, "min": -15 * DEG2RAD, "max": 120 * DEG2RAD},
               {"dim": 1, "min": -40 * DEG2RAD, "max": 40 * DEG2RAD}]

ROCKETBOX_ROOT_DIR = [0, 0, 1]
ROCKETBOX_BOUNDS = {"LeftArm": [],  # {"dim": 1, "min": 0, "max": 90}
                  "RightArm": []  # {"dim": 1, "min": 0, "max": 90},{"dim": 0, "min": 0, "max": 90}
                    , "RightHand": hand_bounds,  # [[-90, 90],[0, 0],[-90,90]]
                  "LeftHand": hand_bounds  # [[-90, 90],[0, 0],[-90,90]]
                    }

ROCKETBOX_ANIMATED_JOINT_LIST = ["Hips", "Spine", "Spine_1", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm",
                               "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg",
                               "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"]


IK_CHAINS_DEFAULT_SKELETON = dict()
IK_CHAINS_DEFAULT_SKELETON["right_ankle"] = {"root": "right_hip", "joint": "right_knee", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_DEFAULT_SKELETON["left_ankle"] = {"root": "left_hip", "joint": "left_knee", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}


IK_CHAINS_RAW_SKELETON = dict()
IK_CHAINS_RAW_SKELETON["RightFoot"] = {"root": "RightUpLeg", "joint": "RightLeg", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_RAW_SKELETON["LeftFoot"] = {"root": "LeftUpLeg", "joint": "LeftLeg", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}

#IK_CHAINS_RAW_SKELETON["RightToeBase"] = {"root": "RightUpLeg", "joint": "RightLeg", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}
#IK_CHAINS_RAW_SKELETON["LeftToeBase"] = {"root": "LeftUpLeg", "joint": "LeftLeg", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}

IK_CHAINS_RAW_SKELETON["RightHand"] = {"root": "RightArm", "joint": "RightForeArm", "joint_axis": [0, 1, 0], "end_effector_dir": [1,0,0]}
IK_CHAINS_RAW_SKELETON["LeftHand"] = {"root": "LeftArm", "joint": "LeftForeArm", "joint_axis": [0, 1, 0], "end_effector_dir": [1,0,0]}


IK_CHAINS_ROCKETBOX_SKELETON = dict()
IK_CHAINS_ROCKETBOX_SKELETON["RightFoot"] = {"root": "RightUpLeg", "joint": "RightLeg", "joint_axis": [0, 1, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_ROCKETBOX_SKELETON["LeftFoot"] = {"root": "LeftUpLeg", "joint": "LeftLeg", "joint_axis": [0, 1, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_ROCKETBOX_SKELETON["RightHand"] = {"root": "RightArm", "joint": "RightForeArm", "joint_axis": [0, 1, 0], "end_effector_dir": [1,0,0]}
IK_CHAINS_ROCKETBOX_SKELETON["LeftHand"] = {"root": "LeftArm", "joint": "LeftForeArm", "joint_axis": [0, 1, 0], "end_effector_dir": [1,0,0]}


IK_CHAINS_GAME_ENGINE_SKELETON = dict()
IK_CHAINS_GAME_ENGINE_SKELETON["foot_l"] = {"root": "thigh_l", "joint": "calf_l", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_GAME_ENGINE_SKELETON["foot_r"] = {"root": "thigh_r", "joint": "calf_r", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_GAME_ENGINE_SKELETON["hand_r"] = {"root": "upperarm_r", "joint": "lowerarm_r", "joint_axis": [1, 0, 0], "end_effector_dir": [1,0,0]}
IK_CHAINS_GAME_ENGINE_SKELETON["hand_l"] = {"root": "upperarm_l", "joint": "lowerarm_l", "joint_axis": [1, 0, 0], "end_effector_dir": [1,0,0]}



IK_CHAINS_CUSTOM_SKELETON = dict()
IK_CHAINS_CUSTOM_SKELETON["L_foot_jnt"] = {"root": "L_upLeg_jnt", "joint": "L_lowLeg_jnt", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_CUSTOM_SKELETON["R_foot_jnt"] = {"root": "R_upLeg_jnt", "joint": "R_lowLeg_jnt", "joint_axis": [1, 0, 0], "end_effector_dir": [0,0,1]}
IK_CHAINS_CUSTOM_SKELETON["R_hand_jnt"] = {"root": "R_upArm_jnt", "joint": "R_lowArm_jnt", "joint_axis": [1, 0, 0], "end_effector_dir": [1,0,0]}
IK_CHAINS_CUSTOM_SKELETON["L_hand_jnt"] = {"root": "L_upArm_jnt", "joint": "L_lowArm_jnt", "joint_axis": [1, 0, 0], "end_effector_dir": [1,0,0]}


RIGHT_SHOULDER = "RightShoulder"
RIGHT_ELBOW = "RightElbow"
RIGHT_WRIST = "RightHand"
ELBOW_AXIS = [0,1,0]

LOCOMOTION_ACTIONS = ["walk", "carryRight", "carryLeft", "carryBoth"]
DEFAULT_WINDOW_SIZE = 20
LEFT_FOOT = "LeftFoot"
RIGHT_FOOT = "RightFoot"
RIGHT_TOE = "RightToeBase"
LEFT_TOE = "LeftToeBase"
LEFT_HEEL = "LeftHeel"
RIGHT_HEEL = "RightHeel"

OFFSET = 0
RAW_SKELETON_FOOT_JOINTS = [RIGHT_TOE, LEFT_TOE, RIGHT_HEEL,LEFT_HEEL]
HEEL_OFFSET = [0, -6.480602, 0]
RAW_SKELETON_JOINTS = collections.OrderedDict()
RAW_SKELETON_JOINTS["root"] = "Hips"
RAW_SKELETON_JOINTS["pelvis"] = "Hips"
RAW_SKELETON_JOINTS["spine"] = "Spine"
RAW_SKELETON_JOINTS["spine_1"] = "Spine1"
RAW_SKELETON_JOINTS["left_clavicle"] = "LeftShoulder"
RAW_SKELETON_JOINTS["right_clavicle"] = "RightShoulder"
RAW_SKELETON_JOINTS["left_shoulder"] = "LeftArm"
RAW_SKELETON_JOINTS["right_shoulder"] = "RightArm"
RAW_SKELETON_JOINTS["left_elbow"] = "LeftForeArm"
RAW_SKELETON_JOINTS["right_elbow"] = "RightForeArm"
RAW_SKELETON_JOINTS["left_wrist"] = "LeftHand"
RAW_SKELETON_JOINTS["right_wrist"] = "RightHand"
RAW_SKELETON_JOINTS["left_hip"] = "LeftUpLeg"
RAW_SKELETON_JOINTS["right_hip"] = "RightUpLeg"
RAW_SKELETON_JOINTS["left_knee"] = "LeftLeg"
RAW_SKELETON_JOINTS["right_knee"] = "RightLeg"
RAW_SKELETON_JOINTS["left_ankle"] = "LeftFoot"
RAW_SKELETON_JOINTS["right_ankle"] = "RightFoot"
RAW_SKELETON_JOINTS["left_toe"] = "LeftToeBase"
RAW_SKELETON_JOINTS["right_toe"] = "RightToeBase"
RAW_SKELETON_JOINTS["left_heel"] = "LeftHeel"
RAW_SKELETON_JOINTS["right_heel"] = "RightHeel"
RAW_SKELETON_JOINTS["neck"] = "Neck"
RAW_SKELETON_JOINTS["head"] = "Head"
RAW_SKELETON_MODEL = collections.OrderedDict()
RAW_SKELETON_MODEL["joints"] = RAW_SKELETON_JOINTS
RAW_SKELETON_MODEL["foot_joints"] = RAW_SKELETON_FOOT_JOINTS
RAW_SKELETON_MODEL["heel_offset"] = [0, -6.480602, 0]
RAW_SKELETON_MODEL["ik_chains"] = IK_CHAINS_RAW_SKELETON



GAME_ENGINE_JOINTS = collections.OrderedDict()
GAME_ENGINE_JOINTS["root"] = "Root"
GAME_ENGINE_JOINTS["pelvis"] = "pelvis"
GAME_ENGINE_JOINTS["spine"] = "spine_01"
GAME_ENGINE_JOINTS["spine_1"] = "spine_02"
GAME_ENGINE_JOINTS["spine_2"] = "spine_03"
GAME_ENGINE_JOINTS["left_clavicle"] = "clavicle_l"
GAME_ENGINE_JOINTS["right_clavicle"] = "clavicle_r"
GAME_ENGINE_JOINTS["left_shoulder"] = "upperarm_l"
GAME_ENGINE_JOINTS["right_shoulder"] = "upperarm_r"
GAME_ENGINE_JOINTS["left_elbow"] = "lowerarm_l"
GAME_ENGINE_JOINTS["right_elbow"] = "lowerarm_r"
GAME_ENGINE_JOINTS["left_wrist"] = "hand_l"
GAME_ENGINE_JOINTS["right_wrist"] = "hand_r"
GAME_ENGINE_JOINTS["left_finger"] = "middle_03_l"
GAME_ENGINE_JOINTS["right_finger"] = "middle_03_r"
GAME_ENGINE_JOINTS["left_hip"] = "thigh_l"
GAME_ENGINE_JOINTS["right_hip"] = "thigh_r"
GAME_ENGINE_JOINTS["left_knee"] = "calf_l"
GAME_ENGINE_JOINTS["right_knee"] = "calf_r"
GAME_ENGINE_JOINTS["left_ankle"] = "foot_l"
GAME_ENGINE_JOINTS["right_ankle"] = "foot_r"
GAME_ENGINE_JOINTS["left_toe"] = "ball_l"
GAME_ENGINE_JOINTS["right_toe"] = "ball_r"
GAME_ENGINE_JOINTS["left_heel"] = "heel_l"
GAME_ENGINE_JOINTS["right_heel"] = "heel_r"
GAME_ENGINE_JOINTS["neck"] = "neck_01"
GAME_ENGINE_JOINTS["head"] = "head"

GAME_ENGINE_SKELETON_MODEL = collections.OrderedDict()
GAME_ENGINE_SKELETON_MODEL["joints"] = GAME_ENGINE_JOINTS
GAME_ENGINE_SKELETON_MODEL["foot_joints"] = ["foot_l", "foot_r", "ball_r", "ball_l", "heel_r", "heel_l"]
GAME_ENGINE_SKELETON_MODEL["heel_offset"] = (np.array([0, 2.45, 3.480602]) * 2.5).tolist()
GAME_ENGINE_SKELETON_MODEL["ik_chains"] = IK_CHAINS_GAME_ENGINE_SKELETON

GAME_ENGINE_SKELETON_COS_MAP = collections.defaultdict(dict)
GAME_ENGINE_SKELETON_COS_MAP["Game_engine"]["y"] = [0, 1, 0]
GAME_ENGINE_SKELETON_COS_MAP["Game_engine"]["x"] = [1, 0, 0]
GAME_ENGINE_SKELETON_COS_MAP["head"]["y"] = [0, 1, 0]
GAME_ENGINE_SKELETON_COS_MAP["head"]["x"] = [1, 0, 0]
GAME_ENGINE_SKELETON_MODEL["cos_map"] = GAME_ENGINE_SKELETON_COS_MAP



GAME_ENGINE_WRONG_ROOT_JOINTS = collections.OrderedDict()
GAME_ENGINE_WRONG_ROOT_JOINTS["root"] = "Game_engine"
GAME_ENGINE_WRONG_ROOT_JOINTS["pelvis"] = "pelvis"
GAME_ENGINE_WRONG_ROOT_JOINTS["spine"] = "spine_01"
GAME_ENGINE_WRONG_ROOT_JOINTS["spine_1"] = "spine_02"
GAME_ENGINE_WRONG_ROOT_JOINTS["spine_2"] = "spine_03"
GAME_ENGINE_WRONG_ROOT_JOINTS["left_clavicle"] = "clavicle_l"
GAME_ENGINE_WRONG_ROOT_JOINTS["right_clavicle"] = "clavicle_r"
GAME_ENGINE_WRONG_ROOT_JOINTS["left_shoulder"] = "upperarm_l"
GAME_ENGINE_WRONG_ROOT_JOINTS["right_shoulder"] = "upperarm_r"
GAME_ENGINE_WRONG_ROOT_JOINTS["left_elbow"] = "lowerarm_l"
GAME_ENGINE_WRONG_ROOT_JOINTS["right_elbow"] = "lowerarm_r"
GAME_ENGINE_WRONG_ROOT_JOINTS["left_wrist"] = "hand_l"
GAME_ENGINE_WRONG_ROOT_JOINTS["right_wrist"] = "hand_r"
GAME_ENGINE_WRONG_ROOT_JOINTS["left_finger"] = "middle_03_l"
GAME_ENGINE_WRONG_ROOT_JOINTS["right_finger"] = "middle_03_r"
GAME_ENGINE_WRONG_ROOT_JOINTS["left_hip"] = "thigh_l"
GAME_ENGINE_WRONG_ROOT_JOINTS["right_hip"] = "thigh_r"
GAME_ENGINE_WRONG_ROOT_JOINTS["left_knee"] = "calf_l"
GAME_ENGINE_WRONG_ROOT_JOINTS["right_knee"] = "calf_r"
GAME_ENGINE_WRONG_ROOT_JOINTS["left_ankle"] = "foot_l"
GAME_ENGINE_WRONG_ROOT_JOINTS["right_ankle"] = "foot_r"
GAME_ENGINE_WRONG_ROOT_JOINTS["left_toe"] = "ball_l"
GAME_ENGINE_WRONG_ROOT_JOINTS["right_toe"] = "ball_r"
GAME_ENGINE_WRONG_ROOT_JOINTS["left_heel"] = "heel_l"
GAME_ENGINE_WRONG_ROOT_JOINTS["right_heel"] = "heel_r"
GAME_ENGINE_WRONG_ROOT_JOINTS["neck"] = "neck_01"
GAME_ENGINE_WRONG_ROOT_JOINTS["head"] = "head"

GAME_ENGINE_WRONG_ROOT_SKELETON_MODEL = collections.OrderedDict()
GAME_ENGINE_WRONG_ROOT_SKELETON_MODEL["name"] = "game_engine_wrong_root"
GAME_ENGINE_WRONG_ROOT_SKELETON_MODEL["joints"] = GAME_ENGINE_WRONG_ROOT_JOINTS
GAME_ENGINE_WRONG_ROOT_SKELETON_MODEL["foot_joints"] = ["foot_l", "foot_r", "ball_r", "ball_l", "heel_r", "heel_l"]
GAME_ENGINE_WRONG_ROOT_SKELETON_MODEL["heel_offset"] = (np.array([0, 2.45, 3.480602]) * 2.5).tolist()
GAME_ENGINE_WRONG_ROOT_SKELETON_MODEL["ik_chains"] = IK_CHAINS_GAME_ENGINE_SKELETON

GAME_ENGINE_WRONG_ROOT_SKELETON_COS_MAP = collections.defaultdict(dict)
GAME_ENGINE_WRONG_ROOT_SKELETON_COS_MAP["Game_engine"]["y"] = [0, 1, 0]
GAME_ENGINE_WRONG_ROOT_SKELETON_COS_MAP["Game_engine"]["x"] = [0, 0, -1]
GAME_ENGINE_WRONG_ROOT_SKELETON_COS_MAP["head"]["y"] = [0, 1, 0]
GAME_ENGINE_WRONG_ROOT_SKELETON_COS_MAP["head"]["x"] = [1, 0, 0]
GAME_ENGINE_WRONG_ROOT_SKELETON_MODEL["cos_map"] = GAME_ENGINE_WRONG_ROOT_SKELETON_COS_MAP


ROCKETBOX_JOINTS = collections.OrderedDict()
ROCKETBOX_JOINTS["root"] = "Hips"
ROCKETBOX_JOINTS["pelvis"] = "Hips"
ROCKETBOX_JOINTS["spine"] = "Spine"
ROCKETBOX_JOINTS["spine_1"] = "Spine_1"
ROCKETBOX_JOINTS["left_clavicle"] = "LeftShoulder"
ROCKETBOX_JOINTS["right_clavicle"] = "RightShoulder"
ROCKETBOX_JOINTS["left_shoulder"] = "LeftArm"
ROCKETBOX_JOINTS["right_shoulder"] = "RightArm"
ROCKETBOX_JOINTS["left_elbow"] = "LeftForeArm"
ROCKETBOX_JOINTS["right_elbow"] = "RightForeArm"
ROCKETBOX_JOINTS["left_wrist"] = "LeftHand"
ROCKETBOX_JOINTS["right_wrist"] = "RightHand"
ROCKETBOX_JOINTS["left_finger"] = "Bip01_L_Finger2"
ROCKETBOX_JOINTS["right_finger"] = "Bip01_R_Finger2"
ROCKETBOX_JOINTS["left_hip"] = "LeftUpLeg"
ROCKETBOX_JOINTS["right_hip"] = "RightUpLeg"
ROCKETBOX_JOINTS["left_knee"] = "LeftLeg"
ROCKETBOX_JOINTS["right_knee"] = "RightLeg"
ROCKETBOX_JOINTS["left_ankle"] = "LeftFoot"
ROCKETBOX_JOINTS["right_ankle"] = "RightFoot"
ROCKETBOX_JOINTS["left_toe"] = "Bip01_L_Toe0"
ROCKETBOX_JOINTS["right_toe"] = "Bip01_R_Toe0"
ROCKETBOX_JOINTS["left_heel"] = "LeftHeel"
ROCKETBOX_JOINTS["right_heel"] = "RightHeel"
ROCKETBOX_JOINTS["neck"] = "Neck"
ROCKETBOX_JOINTS["head"] = "Head"

ROCKETBOX_SKELETON_MODEL = collections.OrderedDict()
ROCKETBOX_SKELETON_MODEL["name"] = "rocketbox"
ROCKETBOX_SKELETON_MODEL["joints"] = ROCKETBOX_JOINTS
ROCKETBOX_SKELETON_MODEL["heel_offset"] = [0, -6.480602, 0]
ROCKETBOX_SKELETON_MODEL["foot_joints"] = RAW_SKELETON_FOOT_JOINTS
ROCKETBOX_SKELETON_MODEL["ik_chains"] = IK_CHAINS_RAW_SKELETON
ROCKETBOX_SKELETON_COS_MAP = collections.defaultdict(dict)
ROCKETBOX_SKELETON_COS_MAP["LeftHand"]["y"] = [0.99969438, -0.01520068, -0.01949593]
ROCKETBOX_SKELETON_COS_MAP["LeftHand"]["x"] = [-1.52006822e-02,  -9.99884452e-01,   1.48198341e-04]
ROCKETBOX_SKELETON_COS_MAP["RightHand"]["y"] = [0.99969439,  0.01520077, -0.01949518]
ROCKETBOX_SKELETON_COS_MAP["RightHand"]["x"] = [ 1.52007742e-02,  -9.99884451e-01,  -1.48193551e-04]

ROCKETBOX_SKELETON_COS_MAP["LeftHand"]["y"] = [1,  0,  0]
ROCKETBOX_SKELETON_COS_MAP["RightHand"]["y"] = [1,  0,  0]
ROCKETBOX_SKELETON_COS_MAP["LeftHand"]["x"] = [0,  0,  1]
ROCKETBOX_SKELETON_COS_MAP["RightHand"]["x"] = [0,  0,  -1]


ROCKETBOX_SKELETON_MODEL["cos_map"] = ROCKETBOX_SKELETON_COS_MAP


CMU_SKELETON_JOINTS = collections.OrderedDict()
CMU_SKELETON_JOINTS["root"] = "hip"
CMU_SKELETON_JOINTS["pelvis"] = "hip"
CMU_SKELETON_JOINTS["spine"] = "abdomen"
CMU_SKELETON_JOINTS["spine_1"] = "chest"
CMU_SKELETON_JOINTS["left_clavicle"] = "lCollar"
CMU_SKELETON_JOINTS["right_clavicle"] = "rCollar"
CMU_SKELETON_JOINTS["left_shoulder"] = "lShldr"
CMU_SKELETON_JOINTS["right_shoulder"] = "rShldr"
CMU_SKELETON_JOINTS["left_elbow"] = "lForeArm"
CMU_SKELETON_JOINTS["right_elbow"] = "rForeArm"
CMU_SKELETON_JOINTS["left_wrist"] = "lHand"
CMU_SKELETON_JOINTS["right_wrist"] = "rHand"
CMU_SKELETON_JOINTS["left_hip"] = "lThigh"
CMU_SKELETON_JOINTS["right_hip"] = "rThigh"
CMU_SKELETON_JOINTS["left_knee"] = "lShin"
CMU_SKELETON_JOINTS["right_knee"] = "rShin"
CMU_SKELETON_JOINTS["left_ankle"] = "lFoot"
CMU_SKELETON_JOINTS["right_ankle"] = "rFoot"
CMU_SKELETON_JOINTS["left_toe"] = "lFoot_EndSite"
CMU_SKELETON_JOINTS["right_toe"] = "rFoot_EndSite"
CMU_SKELETON_JOINTS["left_heel"] = None
CMU_SKELETON_JOINTS["right_heel"] = None
CMU_SKELETON_JOINTS["neck"] = "neck"
CMU_SKELETON_JOINTS["head"] = "head"
CMU_SKELETON_MODEL = collections.OrderedDict()
CMU_SKELETON_MODEL["cmu"] = "cmu"
CMU_SKELETON_MODEL["joints"] = CMU_SKELETON_JOINTS
CMU_SKELETON_MODEL["foot_joints"] = []
CMU_SKELETON_MODEL["foot_correction"] = {"x":-22, "y":3}

MOVIEMATION_SKELETON_JOINTS = collections.OrderedDict()
MOVIEMATION_SKELETON_JOINTS["root"] = "Hips"
MOVIEMATION_SKELETON_JOINTS["pelvis"] = "Hips"
MOVIEMATION_SKELETON_JOINTS["spine"] = "Ab"
MOVIEMATION_SKELETON_JOINTS["spine_1"] = "Chest"
MOVIEMATION_SKELETON_JOINTS["left_clavicle"] = "LeftCollar"
MOVIEMATION_SKELETON_JOINTS["right_clavicle"] = "RightCollar"
MOVIEMATION_SKELETON_JOINTS["left_shoulder"] = "LeftShoulder"
MOVIEMATION_SKELETON_JOINTS["right_shoulder"] = "RightShoulder"
MOVIEMATION_SKELETON_JOINTS["left_elbow"] = "LeftElbow"
MOVIEMATION_SKELETON_JOINTS["right_elbow"] = "RightElbow"
MOVIEMATION_SKELETON_JOINTS["left_wrist"] = "LeftWrist"
MOVIEMATION_SKELETON_JOINTS["right_wrist"] = "RightWrist"
MOVIEMATION_SKELETON_JOINTS["left_hip"] = "LeftHip"
MOVIEMATION_SKELETON_JOINTS["right_hip"] = "RightHip"
MOVIEMATION_SKELETON_JOINTS["left_knee"] = "LeftKnee"
MOVIEMATION_SKELETON_JOINTS["right_knee"] = "RightKnee"
MOVIEMATION_SKELETON_JOINTS["left_ankle"] = "LeftAnkle"
MOVIEMATION_SKELETON_JOINTS["right_ankle"] = "RightAnkle"
MOVIEMATION_SKELETON_JOINTS["left_toe"] = "LeftAnkle_EndSite"
MOVIEMATION_SKELETON_JOINTS["right_toe"] = "RightAnkle_EndSite"
MOVIEMATION_SKELETON_JOINTS["left_heel"] = None
MOVIEMATION_SKELETON_JOINTS["right_heel"] = None
MOVIEMATION_SKELETON_JOINTS["neck"] = "Neck"
MOVIEMATION_SKELETON_JOINTS["head"] = "Head"
MOVIEMATION_SKELETON_MODEL = collections.OrderedDict()
MOVIEMATION_SKELETON_MODEL["name"] = "moviemation"
MOVIEMATION_SKELETON_MODEL["joints"] = MOVIEMATION_SKELETON_JOINTS
MOVIEMATION_SKELETON_MODEL["foot_joints"] = []


MCS_SKELETON_JOINTS = collections.OrderedDict()
MCS_SKELETON_JOINTS["root"] = "Hips"
MCS_SKELETON_JOINTS["pelvis"] = "Hips"
MCS_SKELETON_JOINTS["spine"] = None
MCS_SKELETON_JOINTS["spine_1"] = "Chest"
MCS_SKELETON_JOINTS["left_clavicle"] = "LeftCollar"
MCS_SKELETON_JOINTS["right_clavicle"] = "RightCollar"
MCS_SKELETON_JOINTS["left_shoulder"] = "LeftShoulder"
MCS_SKELETON_JOINTS["right_shoulder"] = "RightShoulder"
MCS_SKELETON_JOINTS["left_elbow"] = "LeftElbow"
MCS_SKELETON_JOINTS["right_elbow"] = "RightElbow"
MCS_SKELETON_JOINTS["left_wrist"] = "LeftWrist"
MCS_SKELETON_JOINTS["right_wrist"] = "RightWrist"
MCS_SKELETON_JOINTS["left_hip"] = "LeftHip"
MCS_SKELETON_JOINTS["right_hip"] = "RightHip"
MCS_SKELETON_JOINTS["left_knee"] = "LeftKnee"
MCS_SKELETON_JOINTS["right_knee"] = "RightKnee"
MCS_SKELETON_JOINTS["left_ankle"] = "LeftAnkle"
MCS_SKELETON_JOINTS["right_ankle"] = "RightAnkle"
MCS_SKELETON_JOINTS["left_toe"] = "LeftAnkle_EndSite"
MCS_SKELETON_JOINTS["right_toe"] = "RightAnkle_EndSite"
MCS_SKELETON_JOINTS["left_heel"] = None
MCS_SKELETON_JOINTS["right_heel"] = None
MCS_SKELETON_JOINTS["neck"] = "Neck"
MCS_SKELETON_JOINTS["head"] = "Head"
MCS_SKELETON_MODEL = collections.OrderedDict()
MCS_SKELETON_MODEL["name"] = "mcs"
MCS_SKELETON_MODEL["joints"] = MCS_SKELETON_JOINTS
MCS_SKELETON_MODEL["foot_joints"] = []



MH_CMU_SKELETON_JOINTS = collections.OrderedDict()
MH_CMU_SKELETON_JOINTS["root"] = None#"CMU compliant skeleton"
MH_CMU_SKELETON_JOINTS["pelvis"] = "Hips"
MH_CMU_SKELETON_JOINTS["spine"] = "LowerBack"
MH_CMU_SKELETON_JOINTS["spine_1"] = "Spine"
MH_CMU_SKELETON_JOINTS["spine_2"] = "Spine1"
MH_CMU_SKELETON_JOINTS["left_clavicle"] = "LeftShoulder"
MH_CMU_SKELETON_JOINTS["right_clavicle"] = "RightShoulder"
MH_CMU_SKELETON_JOINTS["left_shoulder"] = "LeftArm"
MH_CMU_SKELETON_JOINTS["right_shoulder"] = "RightArm"
MH_CMU_SKELETON_JOINTS["left_elbow"] = "LeftForeArm"
MH_CMU_SKELETON_JOINTS["right_elbow"] = "RightForeArm"
MH_CMU_SKELETON_JOINTS["left_wrist"] = "LeftHand"
MH_CMU_SKELETON_JOINTS["right_wrist"] = "RightHand"
MH_CMU_SKELETON_JOINTS["left_finger"] = "LeftHandFinger1"
MH_CMU_SKELETON_JOINTS["right_finger"] = "RightHandFinger1"
MH_CMU_SKELETON_JOINTS["left_hip"] = "LeftUpLeg"
MH_CMU_SKELETON_JOINTS["right_hip"] = "RightUpLeg"
MH_CMU_SKELETON_JOINTS["left_knee"] = "LeftLeg"
MH_CMU_SKELETON_JOINTS["right_knee"] = "RightLeg"
MH_CMU_SKELETON_JOINTS["left_ankle"] = "LeftFoot"
MH_CMU_SKELETON_JOINTS["right_ankle"] = "RightFoot"
MH_CMU_SKELETON_JOINTS["left_toe"] = "LeftToeBase"
MH_CMU_SKELETON_JOINTS["right_toe"] = "RightToeBase"
MH_CMU_SKELETON_JOINTS["left_heel"] = None
MH_CMU_SKELETON_JOINTS["right_heel"] = None
MH_CMU_SKELETON_JOINTS["neck"] = "Neck"
MH_CMU_SKELETON_JOINTS["head"] = "Head"

MH_CMU_SKELETON_MODEL = collections.OrderedDict()
MH_CMU_SKELETON_MODEL["name"] = "mh_cmu"
MH_CMU_SKELETON_MODEL["joints"] = MH_CMU_SKELETON_JOINTS
MH_CMU_SKELETON_MODEL["foot_joints"] = []
MH_CMU_SKELETON_MODEL["foot_correction"] = {"x":-22, "y":3}
MH_CMU_SKELETON_MODEL["flip_x_axis"] = False

MH_CMU_2_SKELETON_JOINTS = collections.OrderedDict()
MH_CMU_2_SKELETON_JOINTS["root"] = None#"CMU compliant skeleton"
MH_CMU_2_SKELETON_JOINTS["pelvis"] = "Hips"
MH_CMU_2_SKELETON_JOINTS["spine"] = "LowerBack"
MH_CMU_2_SKELETON_JOINTS["spine_1"] = "Spine"
MH_CMU_2_SKELETON_JOINTS["spine_2"] = "Spine1"
MH_CMU_2_SKELETON_JOINTS["left_clavicle"] = "LeftShoulder"
MH_CMU_2_SKELETON_JOINTS["right_clavicle"] = "RightShoulder"
MH_CMU_2_SKELETON_JOINTS["left_shoulder"] = "LeftArm"
MH_CMU_2_SKELETON_JOINTS["right_shoulder"] = "RightArm"
MH_CMU_2_SKELETON_JOINTS["left_elbow"] = "LeftForeArm"
MH_CMU_2_SKELETON_JOINTS["right_elbow"] = "RightForeArm"
MH_CMU_2_SKELETON_JOINTS["left_wrist"] = "LeftHand"
MH_CMU_2_SKELETON_JOINTS["right_wrist"] = "RightHand"
MH_CMU_2_SKELETON_JOINTS["left_hip"] = "LeftUpLeg"
MH_CMU_2_SKELETON_JOINTS["right_hip"] = "RightUpLeg"
MH_CMU_2_SKELETON_JOINTS["left_knee"] = "LeftLeg"
MH_CMU_2_SKELETON_JOINTS["right_knee"] = "RightLeg"
MH_CMU_2_SKELETON_JOINTS["left_ankle"] = "LeftFoot"
MH_CMU_2_SKELETON_JOINTS["right_ankle"] = "RightFoot"
MH_CMU_2_SKELETON_JOINTS["left_toe"] = "LeftToeBase"
MH_CMU_2_SKELETON_JOINTS["right_toe"] = "RightToeBase"
MH_CMU_2_SKELETON_JOINTS["left_heel"] = None
MH_CMU_2_SKELETON_JOINTS["right_heel"] = None
MH_CMU_2_SKELETON_JOINTS["neck"] = "Neck"
MH_CMU_2_SKELETON_JOINTS["head"] = "Head"

MH_CMU_2_SKELETON_MODEL = collections.OrderedDict()
MH_CMU_2_SKELETON_MODEL["name"] = "mh_cmu2"
MH_CMU_2_SKELETON_MODEL["joints"] = MH_CMU_2_SKELETON_JOINTS
MH_CMU_2_SKELETON_MODEL["foot_joints"] = []
MH_CMU_2_SKELETON_MODEL["foot_correction"] = {"x":-22, "y":3}
MH_CMU_2_SKELETON_MODEL["flip_x_axis"] = False
#MH_CMU_2_SKELETON_MODEL["foot_joints"] = []
#MH_CMU_2_SKELETON_MODEL["x_cos_fixes"] = ["Neck", "Pelvis", "Spine", "Spine1", "LowerBack", "LeftArm"]
#MH_CMU_2_SKELETON_COS_MAP = collections.defaultdict(dict)
#MH_CMU_2_SKELETON_COS_MAP["Hips"]["y"] = [0, 1,0]
#MH_CMU_2_SKELETON_COS_MAP["Hips"]["x"] = [1, 0, 0]
#MH_CMU_2_SKELETON_MODEL["cos_map"] = MH_CMU_2_SKELETON_COS_MAP

ICLONE_SKELETON_JOINTS = collections.OrderedDict()
ICLONE_SKELETON_JOINTS["root"] = "CC_Base_BoneRoot"
ICLONE_SKELETON_JOINTS["pelvis"] = "CC_Base_Hip"#CC_Base_Pelvis
ICLONE_SKELETON_JOINTS["spine"] = "CC_Base_Waist"
ICLONE_SKELETON_JOINTS["spine_1"] = "CC_Base_Spine01"
ICLONE_SKELETON_JOINTS["spine_2"] = "CC_Base_Spine02"
ICLONE_SKELETON_JOINTS["left_clavicle"] = "CC_Base_L_Clavicle"
ICLONE_SKELETON_JOINTS["right_clavicle"] = "CC_Base_R_Clavicle"
ICLONE_SKELETON_JOINTS["left_shoulder"] = "CC_Base_L_Upperarm"
ICLONE_SKELETON_JOINTS["right_shoulder"] = "CC_Base_R_Upperarm"
ICLONE_SKELETON_JOINTS["left_elbow"] = "CC_Base_L_Forearm"
ICLONE_SKELETON_JOINTS["right_elbow"] = "CC_Base_R_Forearm"
ICLONE_SKELETON_JOINTS["left_wrist"] = "CC_Base_L_Hand"
ICLONE_SKELETON_JOINTS["right_wrist"] = "CC_Base_R_Hand"
ICLONE_SKELETON_JOINTS["left_hip"] = "CC_Base_L_Thigh"
ICLONE_SKELETON_JOINTS["right_hip"] = "CC_Base_R_Thigh"
ICLONE_SKELETON_JOINTS["left_knee"] = "CC_Base_L_Calf"
ICLONE_SKELETON_JOINTS["right_knee"] = "CC_Base_R_Calf"
ICLONE_SKELETON_JOINTS["left_ankle"] = "CC_Base_L_Foot"
ICLONE_SKELETON_JOINTS["right_ankle"] = "CC_Base_R_Foot"
ICLONE_SKELETON_JOINTS["left_toe"] = "CC_Base_L_ToeBase"
ICLONE_SKELETON_JOINTS["right_toe"] = "CC_Base_R_ToeBase"
ICLONE_SKELETON_JOINTS["left_heel"] = None
ICLONE_SKELETON_JOINTS["right_heel"] = None
ICLONE_SKELETON_JOINTS["neck"] = "CC_Base_NeckTwist01"
ICLONE_SKELETON_JOINTS["head"] = "CC_Base_Head"

ICLONE_SKELETON_MODEL = collections.OrderedDict()
ICLONE_SKELETON_MODEL["name"] = "iclone"
ICLONE_SKELETON_MODEL["joints"] = ICLONE_SKELETON_JOINTS
ICLONE_SKELETON_MODEL["foot_joints"] = []
ICLONE_SKELETON_MODEL["x_cos_fixes"] = ["CC_Base_L_Thigh", "CC_Base_R_Thigh", "CC_Base_L_Calf", "CC_Base_R_Calf"]
ICLONE_SKELETON_COS_MAP = collections.defaultdict(dict)
ICLONE_SKELETON_COS_MAP["CC_Base_L_Foot"]["y"] = [0, 0.4, 0.6]
ICLONE_SKELETON_COS_MAP["CC_Base_L_Foot"]["x"] = [-1, 0, 0]
ICLONE_SKELETON_COS_MAP["CC_Base_R_Foot"]["y"] = [0, 0.4, 0.6]
ICLONE_SKELETON_COS_MAP["CC_Base_R_Foot"]["x"] = [-1, 0, 0]
ICLONE_SKELETON_MODEL["cos_map"] = ICLONE_SKELETON_COS_MAP
ICLONE_SKELETON_MODEL["foot_correction"] = {"x":-40, "y":3}
ICLONE_SKELETON_MODEL["flip_x_axis"] = True


CUSTOM_SKELETON_JOINTS = collections.OrderedDict()

CUSTOM_SKELETON_JOINTS["root"] = None
CUSTOM_SKELETON_JOINTS = dict()
CUSTOM_SKELETON_JOINTS["pelvis"] = "FK_back1_jnt"
CUSTOM_SKELETON_JOINTS["spine"] = "FK_back2_jnt"
CUSTOM_SKELETON_JOINTS["spine_1"] = "FK_back3_jnt"
CUSTOM_SKELETON_JOINTS["left_clavicle"] = "L_shoulder_jnt"
CUSTOM_SKELETON_JOINTS["right_clavicle"] = "R_shoulder_jnt"
CUSTOM_SKELETON_JOINTS["left_shoulder"] = "L_upArm_jnt"
CUSTOM_SKELETON_JOINTS["right_shoulder"] = "R_upArm_jnt"
CUSTOM_SKELETON_JOINTS["left_elbow"] = "L_lowArm_jnt"
CUSTOM_SKELETON_JOINTS["right_elbow"] = "R_lowArm_jnt"
CUSTOM_SKELETON_JOINTS["left_wrist"] = "L_hand_jnt"
CUSTOM_SKELETON_JOINTS["right_wrist"] = "R_hand_jnt"
CUSTOM_SKELETON_JOINTS["left_hip"] = "L_upLeg_jnt"
CUSTOM_SKELETON_JOINTS["right_hip"] = "R_upLeg_jnt"
CUSTOM_SKELETON_JOINTS["left_knee"] = "L_lowLeg_jnt"
CUSTOM_SKELETON_JOINTS["right_knee"] = "R_lowLeg_jnt"
CUSTOM_SKELETON_JOINTS["left_ankle"] = "L_foot_jnt"
CUSTOM_SKELETON_JOINTS["right_ankle"] = "R_foot_jnt"
CUSTOM_SKELETON_JOINTS["left_toe"] = "L_toe_jnt"
CUSTOM_SKELETON_JOINTS["right_toe"] = "R_toe_jnt"
CUSTOM_SKELETON_JOINTS["neck"] = "FK_back4_jnt"
CUSTOM_SKELETON_JOINTS["head"] = "head_jnt"

CUSTOM_SKELETON_JOINTS["left_hold_point"] = "L_hand_jnt_hold_point"
CUSTOM_SKELETON_JOINTS["right_hold_point"] = "R_hand_jnt_hold_point"

CUSTOM_SKELETON_JOINTS["right_thumb_base"] = "R_thumb_base_jnt"
CUSTOM_SKELETON_JOINTS["right_thumb_mid"] = "R_thumb_mid_jnt"
CUSTOM_SKELETON_JOINTS["right_thumb_tip"] = "R_thumb_tip_jnt"
CUSTOM_SKELETON_JOINTS["right_thumb_end"] = "R_thumb_end_jnt"

CUSTOM_SKELETON_JOINTS["right_index_finger_root"] = "R_index_root_jnt"
CUSTOM_SKELETON_JOINTS["right_index_finger_base"] = "R_index_base_jnt"
CUSTOM_SKELETON_JOINTS["right_index_finger_mid"] = "R_index_mid_jnt"
CUSTOM_SKELETON_JOINTS["right_index_finger_tip"] = "R_index_tip_jnt"
CUSTOM_SKELETON_JOINTS["right_index_finger_end"] = "R_index_end_jnt"

CUSTOM_SKELETON_JOINTS["right_middle_finger_root"] = "R_middle_root_jnt"
CUSTOM_SKELETON_JOINTS["right_middle_finger_base"] = "R_middle_base_jnt"
CUSTOM_SKELETON_JOINTS["right_middle_finger_mid"] = "R_middle_mid_jnt"
CUSTOM_SKELETON_JOINTS["right_middle_finger_tip"] = "R_middle_tip_jnt"
CUSTOM_SKELETON_JOINTS["right_middle_finger_end"] = "R_middle_end_jnt"

CUSTOM_SKELETON_JOINTS["right_ring_finger_root"] = "R_ring_base_jnt"
CUSTOM_SKELETON_JOINTS["right_ring_finger_base"] = "R_ring_root_jnt"
CUSTOM_SKELETON_JOINTS["right_ring_finger_mid"] = "R_ring_mid_jnt"
CUSTOM_SKELETON_JOINTS["right_ring_finger_tip"] = "R_ring_tip_jnt"
CUSTOM_SKELETON_JOINTS["right_ring_finger_end"] = "R_ring_end_jnt"

CUSTOM_SKELETON_JOINTS["right_pinky_finger_root"] = "R_pinky_root_jnt"
CUSTOM_SKELETON_JOINTS["right_pinky_finger_base"] = "R_pinky_base_jnt"
CUSTOM_SKELETON_JOINTS["right_pinky_finger_mid"] = "R_pinky_mid_jnt"
CUSTOM_SKELETON_JOINTS["right_pinky_finger_tip"] = "R_pinky_tip_jnt"
CUSTOM_SKELETON_JOINTS["right_pinky_finger_end"] = "R_pinky_end_jnt"

CUSTOM_SKELETON_JOINTS["left_thumb_base"] = "L_thumb_base_jnt"
CUSTOM_SKELETON_JOINTS["left_thumb_mid"] = "L_thumb_mid_jnt"
CUSTOM_SKELETON_JOINTS["left_thumb_tip"] = "L_thumb_tip_jnt"
CUSTOM_SKELETON_JOINTS["left_thumb_end"] = "L_thumb_end_jnt"

CUSTOM_SKELETON_JOINTS["left_index_finger_root"] = "L_index_root_jnt"
CUSTOM_SKELETON_JOINTS["left_index_finger_base"] = "L_index_base_jnt"
CUSTOM_SKELETON_JOINTS["left_index_finger_mid"] = "L_index_mid_jnt"
CUSTOM_SKELETON_JOINTS["left_index_finger_tip"] = "L_index_tip_jnt"
CUSTOM_SKELETON_JOINTS["left_index_finger_end"] = "L_index_end_jnt"

CUSTOM_SKELETON_JOINTS["left_middle_finger_root"] = "L_middle_root_jnt"
CUSTOM_SKELETON_JOINTS["left_middle_finger_base"] = "L_middle_base_jnt"
CUSTOM_SKELETON_JOINTS["left_middle_finger_mid"] = "L_middle_mid_jnt"
CUSTOM_SKELETON_JOINTS["left_middle_finger_tip"] = "L_middle_tip_jnt"
CUSTOM_SKELETON_JOINTS["left_middle_finger_end"] = "L_middle_end_jnt"

CUSTOM_SKELETON_JOINTS["left_ring_finger_root"] = "L_ring_base_jnt"
CUSTOM_SKELETON_JOINTS["left_ring_finger_base"] = "L_ring_root_jnt"
CUSTOM_SKELETON_JOINTS["left_ring_finger_mid"] = "L_ring_mid_jnt"
CUSTOM_SKELETON_JOINTS["left_ring_finger_tip"] = "L_ring_tip_jnt"
CUSTOM_SKELETON_JOINTS["left_ring_finger_end"] = "L_ring_end_jnt"

CUSTOM_SKELETON_JOINTS["left_pinky_finger_root"] = "L_pinky_root_jnt"
CUSTOM_SKELETON_JOINTS["left_pinky_finger_base"] = "L_pinky_base_jnt"
CUSTOM_SKELETON_JOINTS["left_pinky_finger_mid"] = "L_pinky_mid_jnt"
CUSTOM_SKELETON_JOINTS["left_pinky_finger_tip"] = "L_pinky_tip_jnt"
CUSTOM_SKELETON_JOINTS["left_pinky_finger_end"] = "L_pinky_end_jnt"




CUSTOM_SKELETON_MODEL = collections.OrderedDict()
CUSTOM_SKELETON_MODEL["name"] = "custom"
CUSTOM_SKELETON_MODEL["joints"] = CUSTOM_SKELETON_JOINTS
CUSTOM_SKELETON_MODEL["foot_joints"] = []
CUSTOM_SKELETON_COS_MAP = collections.defaultdict(dict)
CUSTOM_SKELETON_COS_MAP["R_foot_jnt"]["y"] = [1, 0, 0]
#CUSTOM_SKELETON_COS_MAP["R_foot_jnt"]["y"] = [0.4, 0.6, 0]
CUSTOM_SKELETON_COS_MAP["R_foot_jnt"]["x"] = [0, 0, 1]
CUSTOM_SKELETON_COS_MAP["L_foot_jnt"]["y"] = [1, 0, 0]
#CUSTOM_SKELETON_COS_MAP["L_foot_jnt"]["y"] = [0.4, 0.6, 0]
CUSTOM_SKELETON_COS_MAP["L_foot_jnt"]["x"] = [0, 0, 1]

CUSTOM_SKELETON_COS_MAP["L_upLeg_jnt"]["y"] = [1, 0, 0]
CUSTOM_SKELETON_COS_MAP["L_upLeg_jnt"]["x"] = [0, 0, 1]
CUSTOM_SKELETON_COS_MAP["L_lowLeg_jnt"]["y"] = [1, 0, 0]
CUSTOM_SKELETON_COS_MAP["L_lowLeg_jnt"]["x"] = [0, 0, 1]


CUSTOM_SKELETON_COS_MAP["R_upLeg_jnt"]["y"] = [1, 0, 0]
CUSTOM_SKELETON_COS_MAP["R_upLeg_jnt"]["x"] = [0, 0, 1]
CUSTOM_SKELETON_COS_MAP["R_lowLeg_jnt"]["y"] = [1, 0, 0]
CUSTOM_SKELETON_COS_MAP["R_lowLeg_jnt"]["x"] = [0, 0, 1]

CUSTOM_SKELETON_COS_MAP["FK_back2_jnt"]["y"] = [0, 1, 0]
CUSTOM_SKELETON_COS_MAP["FK_back2_jnt"]["x"] = [1, 0, 0]

CUSTOM_SKELETON_COS_MAP["R_shoulder_jnt"]["y"] = [0, 0, -1]
CUSTOM_SKELETON_COS_MAP["R_shoulder_jnt"]["x"] = [0, -1, 0]
CUSTOM_SKELETON_COS_MAP["R_upArm_jnt"]["y"] = [0, 0, -1]
CUSTOM_SKELETON_COS_MAP["R_upArm_jnt"]["x"] = [0, -1, 0]
CUSTOM_SKELETON_COS_MAP["R_lowArm_jnt"]["y"] = [0, 0, -1]
CUSTOM_SKELETON_COS_MAP["R_lowArm_jnt"]["x"] = [0, -1, 0]

CUSTOM_SKELETON_COS_MAP["L_shoulder_jnt"]["y"] = [0, 0, 1]
CUSTOM_SKELETON_COS_MAP["L_shoulder_jnt"]["x"] = [0, -1, 0]
CUSTOM_SKELETON_COS_MAP["L_upArm_jnt"]["y"] = [0, 0, 1]
CUSTOM_SKELETON_COS_MAP["L_upArm_jnt"]["x"] = [0, -1, 0]
CUSTOM_SKELETON_COS_MAP["L_lowArm_jnt"]["y"] = [0, 0, 1]
CUSTOM_SKELETON_COS_MAP["L_lowArm_jnt"]["x"] = [0, -1, 0]

CUSTOM_SKELETON_COS_MAP["R_hand_jnt"]["y"] = [0, 0, -1]
CUSTOM_SKELETON_COS_MAP["R_hand_jnt"]["x"] = [0, -1, 0]

CUSTOM_SKELETON_COS_MAP["L_hand_jnt"]["y"] = [0, 0, 1]
CUSTOM_SKELETON_COS_MAP["L_hand_jnt"]["x"] = [0, -1, 0]
if False:
    CUSTOM_SKELETON_COS_MAP["FK_back3_jnt"]["y"] = [0, 0.7, -0.3]
    CUSTOM_SKELETON_COS_MAP["FK_back3_jnt"]["x"] = [1, 0, 0]
    CUSTOM_SKELETON_COS_MAP["head_jnt"]["y"] = [0.6, 0.4, 0]
    CUSTOM_SKELETON_COS_MAP["head_jnt"]["x"] = [0, 0, -1]

for k, j in CUSTOM_SKELETON_JOINTS.items():
    if j is None:
        continue
    if "finger" in k:
        if "L_" in j:
            CUSTOM_SKELETON_COS_MAP[j]["x"] = [0, 1, 0]
            CUSTOM_SKELETON_COS_MAP[j]["y"] = [0, 0, 1]
        elif "R_" in j:
            CUSTOM_SKELETON_COS_MAP[j]["x"] = [0, -1, 0]
            CUSTOM_SKELETON_COS_MAP[j]["y"] = [0, 0, -1]
    elif "thumb" in k:
        if "L_" in j:
            CUSTOM_SKELETON_COS_MAP[j]["x"] = [1, 0, 0]
            CUSTOM_SKELETON_COS_MAP[j]["y"] = [0, 0, 1]
        elif "R_" in j:
            CUSTOM_SKELETON_COS_MAP[j]["x"] = [-1, 0, 0]
            CUSTOM_SKELETON_COS_MAP[j]["y"] = [0, 0, -1]

CUSTOM_SKELETON_MODEL["cos_map"] = CUSTOM_SKELETON_COS_MAP
CUSTOM_SKELETON_MODEL["foot_joints"] = []
CUSTOM_SKELETON_MODEL["heel_offset"] = [0, -6.480602, 0]
CUSTOM_SKELETON_MODEL["ik_chains"] = IK_CHAINS_CUSTOM_SKELETON
CUSTOM_SKELETON_MODEL["aligning_root_node"] = "FK_back1_jnt"
CUSTOM_SKELETON_MODEL["free_joints_map"] = {"R_hand_jnt_hold_point":["FK_back2_jnt", "R_upArm_jnt", "R_lowArm_jnt"], "L_hand_jnt_hold_point": ["FK_back2_jnt", "L_upArm_jnt", "L_lowArm_jnt"]}
CUSTOM_SKELETON_MODEL["relative_head_dir"] = [0.0, -1.0, 0.0]
CUSTOM_SKELETON_MODEL["flip_x_axis"] = True



CAPTURY_SKELETON_JOINTS = collections.OrderedDict()
CAPTURY_SKELETON_JOINTS["root"] = "Hips"
CAPTURY_SKELETON_JOINTS["pelvis"] = "Hips"
CAPTURY_SKELETON_JOINTS["spine"] = "Spine"
CAPTURY_SKELETON_JOINTS["spine_1"] = "Spine1"
CAPTURY_SKELETON_JOINTS["spine_2"] = "Spine3"
CAPTURY_SKELETON_JOINTS["left_clavicle"] = "LeftShoulder"
CAPTURY_SKELETON_JOINTS["right_clavicle"] = "RightShoulder"
CAPTURY_SKELETON_JOINTS["left_shoulder"] = "LeftArm"
CAPTURY_SKELETON_JOINTS["right_shoulder"] = "RightArm"
CAPTURY_SKELETON_JOINTS["left_elbow"] = "LeftForeArm"
CAPTURY_SKELETON_JOINTS["right_elbow"] = "RightForeArm"
CAPTURY_SKELETON_JOINTS["left_wrist"] = "LeftHand"
CAPTURY_SKELETON_JOINTS["right_wrist"] = "RightHand"
CAPTURY_SKELETON_JOINTS["left_finger"] = "LeftHand_EndSite"
CAPTURY_SKELETON_JOINTS["right_finger"] = "RightHand_EndSite"
CAPTURY_SKELETON_JOINTS["left_hip"] = "LeftUpLeg"
CAPTURY_SKELETON_JOINTS["right_hip"] = "RightUpLeg"
CAPTURY_SKELETON_JOINTS["left_knee"] = "LeftLeg"
CAPTURY_SKELETON_JOINTS["right_knee"] = "RightLeg"
CAPTURY_SKELETON_JOINTS["left_ankle"] = "LeftFoot"
CAPTURY_SKELETON_JOINTS["right_ankle"] = "RightFoot"
CAPTURY_SKELETON_JOINTS["left_toe"] = "LeftToeBase"
CAPTURY_SKELETON_JOINTS["right_toe"] = "RightToeBase"
CAPTURY_SKELETON_JOINTS["left_heel"] = None
CAPTURY_SKELETON_JOINTS["right_heel"] = None
CAPTURY_SKELETON_JOINTS["neck"] = "Neck"
CAPTURY_SKELETON_JOINTS["head"] = "Head"
CAPTURY_SKELETON_COS_MAP = collections.defaultdict(dict)
CAPTURY_SKELETON_COS_MAP["LeftHand"]["y"] = [-1, 0, 0]
CAPTURY_SKELETON_COS_MAP["LeftHand"]["x"] = [0, -1, 0]
CAPTURY_SKELETON_COS_MAP["LeftArm"]["y"] = [-1, 0, 0]
CAPTURY_SKELETON_COS_MAP["LeftArm"]["x"] = [0, -1, 0]
CAPTURY_SKELETON_COS_MAP["LeftShoulder"]["y"] = [-1, 0, 0]
CAPTURY_SKELETON_COS_MAP["LeftShoulder"]["x"] = [0, -1, 0]

CAPTURY_SKELETON_COS_MAP["LeftForeArm"]["y"] = [-1, 0, 0]
CAPTURY_SKELETON_COS_MAP["LeftForeArm"]["x"] = [0, -1, 0]
CAPTURY_SKELETON_COS_MAP["RightHand"]["y"] = [1, 0, 0]
CAPTURY_SKELETON_COS_MAP["RightHand"]["x"] = [0, 1, 0]
CAPTURY_SKELETON_COS_MAP["RightArm"]["y"] = [1, 0, 0]
CAPTURY_SKELETON_COS_MAP["RightArm"]["x"] = [0, 1, 0]
CAPTURY_SKELETON_COS_MAP["RightShoulder"]["y"] = [1, 0, 0]
CAPTURY_SKELETON_COS_MAP["RightShoulder"]["x"] = [0, 1, 0]

CAPTURY_SKELETON_COS_MAP["RightForeArm"]["y"] = [1, 0, 0]
CAPTURY_SKELETON_COS_MAP["RightForeArm"]["x"] = [0, 1, 0]


CAPTURY_SKELETON_MODEL = collections.OrderedDict()
CAPTURY_SKELETON_MODEL["name"] = "captury"
CAPTURY_SKELETON_MODEL["joints"] = CAPTURY_SKELETON_JOINTS
CAPTURY_SKELETON_MODEL["foot_joints"] = []
CAPTURY_SKELETON_MODEL["cos_map"] = CAPTURY_SKELETON_COS_MAP


HOLDEN_SKELETON_JOINTS = collections.OrderedDict()
HOLDEN_SKELETON_JOINTS["root"] = None
HOLDEN_SKELETON_JOINTS["pelvis"] = "Hips"
HOLDEN_SKELETON_JOINTS["spine"] = "Spine"
HOLDEN_SKELETON_JOINTS["spine_1"] = "Spine1"
HOLDEN_SKELETON_JOINTS["spine_2"] = None
HOLDEN_SKELETON_JOINTS["left_clavicle"] = "LeftShoulder"
HOLDEN_SKELETON_JOINTS["right_clavicle"] = "RightShoulder"
HOLDEN_SKELETON_JOINTS["left_shoulder"] = "LeftArm"
HOLDEN_SKELETON_JOINTS["right_shoulder"] = "RightArm"
HOLDEN_SKELETON_JOINTS["left_elbow"] = "LeftForeArm"
HOLDEN_SKELETON_JOINTS["right_elbow"] = "RightForeArm"
HOLDEN_SKELETON_JOINTS["left_wrist"] = "LeftHand"
HOLDEN_SKELETON_JOINTS["right_wrist"] = "RightHand"
HOLDEN_SKELETON_JOINTS["left_finger"] = None
HOLDEN_SKELETON_JOINTS["right_finger"] = None
HOLDEN_SKELETON_JOINTS["left_hip"] = "LeftUpLeg"
HOLDEN_SKELETON_JOINTS["right_hip"] = "RightUpLeg"
HOLDEN_SKELETON_JOINTS["left_knee"] = "LeftLeg"
HOLDEN_SKELETON_JOINTS["right_knee"] = "RightLeg"
HOLDEN_SKELETON_JOINTS["left_ankle"] = "LeftFoot"
HOLDEN_SKELETON_JOINTS["right_ankle"] = "RightFoot"
HOLDEN_SKELETON_JOINTS["left_toe"] = "LeftToeBase"
HOLDEN_SKELETON_JOINTS["right_toe"] = "RightToeBase"
HOLDEN_SKELETON_JOINTS["left_heel"] = None
HOLDEN_SKELETON_JOINTS["right_heel"] = None
HOLDEN_SKELETON_JOINTS["neck"] = "Neck1"
HOLDEN_SKELETON_JOINTS["head"] = "Head"
HOLDEN_SKELETON_COS_MAP = collections.defaultdict(dict)

HOLDEN_SKELETON_MODEL = collections.OrderedDict()
HOLDEN_SKELETON_MODEL["name"] = "holden"
HOLDEN_SKELETON_MODEL["joints"] = HOLDEN_SKELETON_JOINTS
HOLDEN_SKELETON_MODEL["foot_joints"] = []
HOLDEN_SKELETON_MODEL["cos_map"] = HOLDEN_SKELETON_COS_MAP

MIXAMO_SKELETON_JOINTS = collections.OrderedDict()
MIXAMO_SKELETON_JOINTS["root"] = None
MIXAMO_SKELETON_JOINTS["pelvis"] = "mixamorig:Hips"
MIXAMO_SKELETON_JOINTS["spine"] = "mixamorig:Spine"
MIXAMO_SKELETON_JOINTS["spine_1"] = "mixamorig:Spine1"
MIXAMO_SKELETON_JOINTS["spine_2"] = "mixamorig:Spine2"
MIXAMO_SKELETON_JOINTS["left_clavicle"] = "mixamorig:LeftShoulder"
MIXAMO_SKELETON_JOINTS["right_clavicle"] = "mixamorig:RightShoulder"
MIXAMO_SKELETON_JOINTS["left_shoulder"] = "mixamorig:LeftArm"
MIXAMO_SKELETON_JOINTS["right_shoulder"] = "mixamorig:RightArm"
MIXAMO_SKELETON_JOINTS["left_elbow"] = "mixamorig:LeftForeArm"
MIXAMO_SKELETON_JOINTS["right_elbow"] = "mixamorig:RightForeArm"
MIXAMO_SKELETON_JOINTS["left_wrist"] = "mixamorig:LeftHand"
MIXAMO_SKELETON_JOINTS["right_wrist"] = "mixamorig:RightHand"
MIXAMO_SKELETON_JOINTS["left_finger"] = None
MIXAMO_SKELETON_JOINTS["right_finger"] = None
MIXAMO_SKELETON_JOINTS["left_hip"] = "mixamorig:LeftUpLeg"
MIXAMO_SKELETON_JOINTS["right_hip"] = "mixamorig:RightUpLeg"
MIXAMO_SKELETON_JOINTS["left_knee"] = "mixamorig:LeftLeg"
MIXAMO_SKELETON_JOINTS["right_knee"] = "mixamorig:RightLeg"
MIXAMO_SKELETON_JOINTS["left_ankle"] = "mixamorig:LeftFoot"
MIXAMO_SKELETON_JOINTS["right_ankle"] = "mixamorig:RightFoot"
MIXAMO_SKELETON_JOINTS["left_toe"] = "mixamorig:LeftToeBase"
MIXAMO_SKELETON_JOINTS["right_toe"] = "mixamorig:RightToeBase"
MIXAMO_SKELETON_JOINTS["left_heel"] = None
MIXAMO_SKELETON_JOINTS["right_heel"] = None
MIXAMO_SKELETON_JOINTS["neck"] = "mixamorig:Neck"
MIXAMO_SKELETON_JOINTS["head"] = "mixamorig:Head"
MIXAMO_SKELETON_COS_MAP = collections.defaultdict(dict)
MIXAMO_SKELETON_COS_MAP["mixamorig:LeftUpLeg"]["x"] = [-1, 0, 0]
MIXAMO_SKELETON_COS_MAP["mixamorig:LeftUpLeg"]["y"] = [0, 1, 0]
MIXAMO_SKELETON_COS_MAP["mixamorig:RightUpLeg"]["x"] = [-1, 0, 0]
MIXAMO_SKELETON_COS_MAP["mixamorig:RightUpLeg"]["y"] = [0, 1, 0]
MIXAMO_SKELETON_COS_MAP["mixamorig:LeftLeg"]["x"] = [-1, 0, 0]
MIXAMO_SKELETON_COS_MAP["mixamorig:LeftLeg"]["y"] = [0, 1, 0]
MIXAMO_SKELETON_COS_MAP["mixamorig:RightLeg"]["x"] = [-1, 0, 0]
MIXAMO_SKELETON_COS_MAP["mixamorig:RightLeg"]["y"] = [0, 1, 0]
MIXAMO_SKELETON_COS_MAP["mixamorig:LeftFoot"]["x"] = [-1, 0, 0]
MIXAMO_SKELETON_COS_MAP["mixamorig:LeftFoot"]["y"] = [0, 1, 0]
MIXAMO_SKELETON_COS_MAP["mixamorig:RightFoot"]["x"] = [-1, 0, 0]
MIXAMO_SKELETON_COS_MAP["mixamorig:RightFoot"]["y"] = [0, 1, 0]

MIXAMO_SKELETON_MODEL = collections.OrderedDict()
MIXAMO_SKELETON_MODEL["name"] = "mixamo"
MIXAMO_SKELETON_MODEL["joints"] = MIXAMO_SKELETON_JOINTS
MIXAMO_SKELETON_MODEL["foot_joints"] = []
MIXAMO_SKELETON_MODEL["cos_map"] = MIXAMO_SKELETON_COS_MAP
MIXAMO_SKELETON_MODEL["foot_correction"] = {"x":22, "y":3}
MIXAMO_SKELETON_MODEL["flip_x_axis"] = True
MIXAMO_SKELETON_MODEL["fixed_joint_corrections"] = dict()
MIXAMO_SKELETON_MODEL["fixed_joint_corrections"]["RightClavicle"] = -90
MIXAMO_SKELETON_MODEL["fixed_joint_corrections"]["LeftClavicle"] = 90

MAX_SKELETON_JOINTS = collections.OrderedDict()
MAX_SKELETON_JOINTS["root"] = None
MAX_SKELETON_JOINTS["pelvis"] = "Hips"
MAX_SKELETON_JOINTS["spine"] = None
MAX_SKELETON_JOINTS["spine_1"] = None
MAX_SKELETON_JOINTS["spine_2"] = "Chest"
MAX_SKELETON_JOINTS["left_clavicle"] = "LeftCollar"
MAX_SKELETON_JOINTS["right_clavicle"] = "RightCollar"
MAX_SKELETON_JOINTS["left_shoulder"] = "LeftShoulder"
MAX_SKELETON_JOINTS["right_shoulder"] = "RightShoulder"
MAX_SKELETON_JOINTS["left_elbow"] = "LeftElbow"
MAX_SKELETON_JOINTS["right_elbow"] = "RightElbow"
MAX_SKELETON_JOINTS["left_wrist"] = "LeftWrist"
MAX_SKELETON_JOINTS["right_wrist"] = "RightWrist"
MAX_SKELETON_JOINTS["left_finger"] = "LeftWrist_EndSite"
MAX_SKELETON_JOINTS["right_finger"] = "RightWrist_EndSite"
MAX_SKELETON_JOINTS["left_hip"] = "LeftHip"
MAX_SKELETON_JOINTS["right_hip"] = "RightHip"
MAX_SKELETON_JOINTS["left_knee"] = "LeftKnee"
MAX_SKELETON_JOINTS["right_knee"] = "RightKnee"
MAX_SKELETON_JOINTS["left_ankle"] = "LeftAnkle"
MAX_SKELETON_JOINTS["right_ankle"] = "RightAnkle"
MAX_SKELETON_JOINTS["left_toe"] = None
MAX_SKELETON_JOINTS["right_toe"] = None
MAX_SKELETON_JOINTS["left_heel"] = None
MAX_SKELETON_JOINTS["right_heel"] = None
MAX_SKELETON_JOINTS["neck"] = "Neck"
MAX_SKELETON_JOINTS["head"] = "Head"
MAX_SKELETON_COS_MAP = collections.defaultdict(dict)

MAX_SKELETON_COS_MAP["LeftShoulder"]["y"] = [0, -1, 0]
MAX_SKELETON_COS_MAP["RightShoulder"]["y"] = [0, -1, 0]
MAX_SKELETON_COS_MAP["LeftShoulder"]["x"] = [-1, 0, 0]
MAX_SKELETON_COS_MAP["RightShoulder"]["x"] = [-1, 0, 0]

MAX_SKELETON_COS_MAP["LeftElbow"]["y"] = [0, -1, 0]
MAX_SKELETON_COS_MAP["RightElbow"]["y"] = [0, -1, 0]
MAX_SKELETON_COS_MAP["LeftElbow"]["x"] = [-1, 0, 0]
MAX_SKELETON_COS_MAP["RightElbow"]["x"] = [-1, 0, 0]

MAX_SKELETON_COS_MAP["LeftWrist"]["y"] = [0, -1, 0]
MAX_SKELETON_COS_MAP["RightWrist"]["y"] = [0, -1, 0]
MAX_SKELETON_COS_MAP["LeftWrist"]["x"] = [-1, 0, 0]
MAX_SKELETON_COS_MAP["RightWrist"]["x"] = [-1, 0, 0]


MAX_SKELETON_MODEL = collections.OrderedDict()
MAX_SKELETON_MODEL["name"] = "max"
MAX_SKELETON_MODEL["joints"] = MAX_SKELETON_JOINTS
MAX_SKELETON_MODEL["foot_joints"] = []
MAX_SKELETON_MODEL["cos_map"] = MAX_SKELETON_COS_MAP

AACHEN_SKELETON_JOINTS = collections.OrderedDict()
AACHEN_SKELETON_JOINTS["root"] = None#"reference"
AACHEN_SKELETON_JOINTS["pelvis"] = "Hips"
AACHEN_SKELETON_JOINTS["spine"] = "LowerBack"
AACHEN_SKELETON_JOINTS["spine_1"] = "Spine"
AACHEN_SKELETON_JOINTS["spine_2"] = "Spine1"
AACHEN_SKELETON_JOINTS["left_clavicle"] = "LeftShoulder"
AACHEN_SKELETON_JOINTS["right_clavicle"] = "RightShoulder"
AACHEN_SKELETON_JOINTS["left_shoulder"] = "LeftArm"
AACHEN_SKELETON_JOINTS["right_shoulder"] = "RightArm"
AACHEN_SKELETON_JOINTS["left_elbow"] = "LeftForeArm"
AACHEN_SKELETON_JOINTS["right_elbow"] = "RightForeArm"
AACHEN_SKELETON_JOINTS["left_wrist"] = "LeftHand"
AACHEN_SKELETON_JOINTS["right_wrist"] = "RightHand"
AACHEN_SKELETON_JOINTS["left_finger"] = "LeftFingerBase"
AACHEN_SKELETON_JOINTS["right_finger"] = "RightFingerBase"
AACHEN_SKELETON_JOINTS["left_hip"] = "LeftUpLeg"
AACHEN_SKELETON_JOINTS["right_hip"] = "RightUpLeg"
AACHEN_SKELETON_JOINTS["left_knee"] = "LeftLeg"
AACHEN_SKELETON_JOINTS["right_knee"] = "RightLeg"
AACHEN_SKELETON_JOINTS["left_ankle"] = "LeftFoot"
AACHEN_SKELETON_JOINTS["right_ankle"] = "RightFoot"
AACHEN_SKELETON_JOINTS["left_toe"] = "LeftToeBase"
AACHEN_SKELETON_JOINTS["right_toe"] = "RightToeBase"
AACHEN_SKELETON_JOINTS["left_heel"] = None
AACHEN_SKELETON_JOINTS["right_heel"] = None
AACHEN_SKELETON_JOINTS["neck"] = "Neck"
AACHEN_SKELETON_JOINTS["head"] = "Head"
AACHEN_SKELETON_MODEL = collections.OrderedDict()
AACHEN_SKELETON_MODEL["name"] = "aachen"
AACHEN_SKELETON_MODEL["joints"] = AACHEN_SKELETON_JOINTS
AACHEN_SKELETON_MODEL["foot_joints"] = []

AACHEN_SKELETON_COS_MAP = collections.defaultdict(dict)
AACHEN_SKELETON_COS_MAP["Hips"]["y"] = [0, 1, 0]
AACHEN_SKELETON_COS_MAP["Hips"]["x"] = [1, 0, 0]
AACHEN_SKELETON_MODEL["cos_map"] = AACHEN_SKELETON_COS_MAP


MH_CMU_RED_SKELETON_JOINTS = collections.OrderedDict()
MH_CMU_RED_SKELETON_JOINTS["root"] = "root"
MH_CMU_RED_SKELETON_JOINTS["pelvis"] = "pelvis"
MH_CMU_RED_SKELETON_JOINTS["thorax"] = "Spine"
MH_CMU_RED_SKELETON_JOINTS["left_clavicle"] = "lclavicle"
MH_CMU_RED_SKELETON_JOINTS["right_clavicle"] = "rclavicle"
MH_CMU_RED_SKELETON_JOINTS["left_shoulder"] = "lhumerus"
MH_CMU_RED_SKELETON_JOINTS["right_shoulder"] = "rhumerus"
MH_CMU_RED_SKELETON_JOINTS["left_elbow"] = "lradius"
MH_CMU_RED_SKELETON_JOINTS["right_elbow"] = "rradius"
MH_CMU_RED_SKELETON_JOINTS["left_wrist"] = "lhand"
MH_CMU_RED_SKELETON_JOINTS["right_wrist"] = "rhand"
MH_CMU_RED_SKELETON_JOINTS["left_hip"] = "lfemur"
MH_CMU_RED_SKELETON_JOINTS["right_hip"] = "rfemur"
MH_CMU_RED_SKELETON_JOINTS["left_knee"] = "ltibia"
MH_CMU_RED_SKELETON_JOINTS["right_knee"] = "rtibia"
MH_CMU_RED_SKELETON_JOINTS["left_ankle"] = "lfoot"
MH_CMU_RED_SKELETON_JOINTS["right_ankle"] = "rfoot"
MH_CMU_RED_SKELETON_JOINTS["left_toe"] = "ltoes"
MH_CMU_RED_SKELETON_JOINTS["right_toe"] = "rtoes"
MH_CMU_RED_SKELETON_JOINTS["left_heel"] = None
MH_CMU_RED_SKELETON_JOINTS["right_heel"] = None
MH_CMU_RED_SKELETON_JOINTS["neck"] = "head"
MH_CMU_RED_SKELETON_JOINTS["head"] = "head_EndSite"
MH_CMU_RED_SKELETON_MODEL = collections.OrderedDict()
MH_CMU_RED_SKELETON_MODEL["name"] = "mh_cmu_red"
MH_CMU_RED_SKELETON_MODEL["joints"] = MH_CMU_RED_SKELETON_JOINTS
MH_CMU_RED_SKELETON_MODEL["foot_joints"] = []

RAW2_SKELETON_FOOT_JOINTS = [RIGHT_TOE, LEFT_TOE, RIGHT_HEEL,LEFT_HEEL]

RAW2_SKELETON_JOINTS = collections.OrderedDict()
RAW2_SKELETON_JOINTS["root"] = "Hips"
RAW2_SKELETON_JOINTS["pelvis"] = "ToSpine"
RAW2_SKELETON_JOINTS["spine"] = "Spine"
RAW2_SKELETON_JOINTS["spine_1"] = "Spine1"
RAW2_SKELETON_JOINTS["spine_2"] = "Neck"
RAW2_SKELETON_JOINTS["left_clavicle"] = "LeftShoulder"
RAW2_SKELETON_JOINTS["right_clavicle"] = "RightShoulder"
RAW2_SKELETON_JOINTS["left_shoulder"] = "LeftArm"
RAW2_SKELETON_JOINTS["right_shoulder"] = "RightArm"
RAW2_SKELETON_JOINTS["left_elbow"] = "LeftForeArm"
RAW2_SKELETON_JOINTS["right_elbow"] = "RightForeArm"
RAW2_SKELETON_JOINTS["left_wrist"] = "LeftHand"
RAW2_SKELETON_JOINTS["right_wrist"] = "RightHand"
RAW2_SKELETON_JOINTS["left_hip"] = "LeftUpLeg"
RAW2_SKELETON_JOINTS["right_hip"] = "RightUpLeg"
RAW2_SKELETON_JOINTS["left_knee"] = "LeftLeg"
RAW2_SKELETON_JOINTS["right_knee"] = "RightLeg"
RAW2_SKELETON_JOINTS["left_ankle"] = "LeftFoot"
RAW2_SKELETON_JOINTS["right_ankle"] = "RightFoot"
RAW2_SKELETON_JOINTS["left_toe"] = "LeftToeBase"
RAW2_SKELETON_JOINTS["right_toe"] = "RightToeBase"
RAW2_SKELETON_JOINTS["left_heel"] = "LeftHeel"
RAW2_SKELETON_JOINTS["right_heel"] = "RightHeel"
RAW2_SKELETON_JOINTS["neck"] = "Neck"
RAW2_SKELETON_JOINTS["head"] = "Head"

RAW2_SKELETON_COS_MAP = collections.defaultdict(dict)
RAW2_SKELETON_COS_MAP["Neck"]["x"] = [1, 0, 0]
RAW2_SKELETON_COS_MAP["Neck"]["y"] = [0, 1, 0]
RAW2_SKELETON_COS_MAP["Head"]["x"] = [-1, 0, 0]
RAW2_SKELETON_COS_MAP["Head"]["y"] = [0, 1, 0]

"""
RAW2_SKELETON_COS_MAP["LeftShoulder"]["x"] = [0, 1, 0]
RAW2_SKELETON_COS_MAP["LeftShoulder"]["y"] = [-1, 0, 0]
RAW2_SKELETON_COS_MAP["LeftForeArm"]["x"] = [-1, 0, 0]
RAW2_SKELETON_COS_MAP["LeftForeArm"]["y"] = [0, 1, 0]
RAW2_SKELETON_COS_MAP["LeftArm"]["x"] = [-1, 0, 0]
RAW2_SKELETON_COS_MAP["LeftArm"]["y"] = [0, 1, 0]
RAW2_SKELETON_COS_MAP["LeftHand"]["x"] = [-1, 0, 0]
RAW2_SKELETON_COS_MAP["LeftHand"]["y"] = [0, 1, 0]
RAW2_SKELETON_COS_MAP["RightShoulder"]["x"] = [0, 1, 0]
RAW2_SKELETON_COS_MAP["RightShoulder"]["y"] = [-1, 0, 0]
RAW2_SKELETON_COS_MAP["RightArm"]["x"] = [-1, 0, 0]
RAW2_SKELETON_COS_MAP["RightArm"]["y"] = [0, 1, 0]
RAW2_SKELETON_COS_MAP["RightForeArm"]["x"] = [-1, 0, 0]
RAW2_SKELETON_COS_MAP["RightForeArm"]["y"] = [0, 1, 0]
RAW2_SKELETON_COS_MAP["RightHand"]["x"] = [-1, 0, 0]
RAW2_SKELETON_COS_MAP["RightHand"]["y"] = [0, 1, 0]
"""
RAW2_SKELETON_MODEL = collections.OrderedDict()
RAW2_SKELETON_MODEL["name"] = "raw2"
RAW2_SKELETON_MODEL["joints"] = RAW2_SKELETON_JOINTS
RAW2_SKELETON_MODEL["foot_joints"] = RAW2_SKELETON_FOOT_JOINTS
RAW2_SKELETON_MODEL["heel_offset"] = [0, -6.480602, 0]
RAW2_SKELETON_MODEL["ik_chains"] = IK_CHAINS_RAW_SKELETON
RAW2_SKELETON_MODEL["cos_map"] = RAW2_SKELETON_COS_MAP




NOITOM_SKELETON_JOINTS = collections.OrderedDict()
NOITOM_SKELETON_JOINTS["root"] = None
NOITOM_SKELETON_JOINTS["pelvis"] = "Hips"
NOITOM_SKELETON_JOINTS["spine"] = "Spine"
NOITOM_SKELETON_JOINTS["spine_1"] = "Spine1"
NOITOM_SKELETON_JOINTS["spine_2"] = "Spine2"
NOITOM_SKELETON_JOINTS["left_clavicle"] = "LeftShoulder"
NOITOM_SKELETON_JOINTS["right_clavicle"] = "RightShoulder"
NOITOM_SKELETON_JOINTS["left_shoulder"] = "LeftArm"
NOITOM_SKELETON_JOINTS["right_shoulder"] = "RightArm"
NOITOM_SKELETON_JOINTS["left_elbow"] = "LeftForeArm"
NOITOM_SKELETON_JOINTS["right_elbow"] = "RightForeArm"
NOITOM_SKELETON_JOINTS["left_wrist"] = "LeftHand"
NOITOM_SKELETON_JOINTS["right_wrist"] = "RightHand"
#NOITOM_SKELETON_JOINTS["left_finger"] = "RightInHandMiddle"
#NOITOM_SKELETON_JOINTS["right_finger"] = "LeftInHandMiddle"
NOITOM_SKELETON_JOINTS["left_hip"] = "LeftUpLeg"
NOITOM_SKELETON_JOINTS["right_hip"] = "RightUpLeg"
NOITOM_SKELETON_JOINTS["left_knee"] = "LeftLeg"
NOITOM_SKELETON_JOINTS["right_knee"] = "RightLeg"
NOITOM_SKELETON_JOINTS["left_ankle"] = "LeftFoot"
NOITOM_SKELETON_JOINTS["right_ankle"] = "RightFoot"
NOITOM_SKELETON_JOINTS["left_toe"] = "LeftFoot_EndSite"
NOITOM_SKELETON_JOINTS["right_toe"] = "RightFoot_EndSite"
NOITOM_SKELETON_JOINTS["left_heel"] = None
NOITOM_SKELETON_JOINTS["right_heel"] = None
NOITOM_SKELETON_JOINTS["neck"] = "Neck"
NOITOM_SKELETON_JOINTS["head"] = "Head"

NOITOM_SKELETON_JOINTS["right_thumb_base"] = "RightHandThumb1"
NOITOM_SKELETON_JOINTS["right_thumb_mid"] = "RightHandThumb2"
NOITOM_SKELETON_JOINTS["right_thumb_tip"] = "RightHandThumb3"
NOITOM_SKELETON_JOINTS["right_thumb_end"] = "RightHandThumb3_EndSite"

NOITOM_SKELETON_JOINTS["right_index_finger_root"] = "RightInHandIndex"
NOITOM_SKELETON_JOINTS["right_index_finger_base"] = "RightHandIndex1"
NOITOM_SKELETON_JOINTS["right_index_finger_mid"] = "RightHandIndex2"
NOITOM_SKELETON_JOINTS["right_index_finger_tip"] = "RightHandIndex3"
NOITOM_SKELETON_JOINTS["right_index_finger_end"] = "RightHandIndex3_EndSite"

NOITOM_SKELETON_JOINTS["right_middle_finger_root"] = "RightInHandMiddle"
NOITOM_SKELETON_JOINTS["right_middle_finger_base"] = "RightHandMiddle1"
NOITOM_SKELETON_JOINTS["right_middle_finger_mid"] = "RightHandMiddle2"
NOITOM_SKELETON_JOINTS["right_middle_finger_tip"] = "RightHandMiddle3"
NOITOM_SKELETON_JOINTS["right_middle_finger_end"] = "RightHandMiddle3_EndSite"

NOITOM_SKELETON_JOINTS["right_ring_finger_root"] = "RightInHandRing"
NOITOM_SKELETON_JOINTS["right_ring_finger_base"] = "RightHandRing1"
NOITOM_SKELETON_JOINTS["right_ring_finger_mid"] = "RightHandRing2"
NOITOM_SKELETON_JOINTS["right_ring_finger_tip"] = "RightHandRing3"
NOITOM_SKELETON_JOINTS["right_ring_finger_end"] = "RightHandRing3_EndSite"

NOITOM_SKELETON_JOINTS["right_pinky_finger_root"] = "RightInHandPinky"
NOITOM_SKELETON_JOINTS["right_pinky_finger_base"] = "RightHandPinky1"
NOITOM_SKELETON_JOINTS["right_pinky_finger_mid"] = "RightHandPinky2"
NOITOM_SKELETON_JOINTS["right_pinky_finger_tip"] = "RightHandPinky3"
NOITOM_SKELETON_JOINTS["right_pinky_finger_end"] = "RightHandPinky3_EndSite"


NOITOM_SKELETON_JOINTS["left_thumb_base"] = "LeftHandThumb1"
NOITOM_SKELETON_JOINTS["left_thumb_mid"] = "LeftHandThumb2"
NOITOM_SKELETON_JOINTS["left_thumb_tip"] = "LeftHandThumb3"
NOITOM_SKELETON_JOINTS["left_thumb_end"] = "LeftHandThumb3_EndSite"

NOITOM_SKELETON_JOINTS["left_index_finger_root"] = "LeftInHandIndex"
NOITOM_SKELETON_JOINTS["left_index_finger_base"] = "LeftHandIndex1"
NOITOM_SKELETON_JOINTS["left_index_finger_mid"] = "LeftHandIndex2"
NOITOM_SKELETON_JOINTS["left_index_finger_tip"] = "LeftHandIndex3"
NOITOM_SKELETON_JOINTS["left_index_finger_end"] = "LeftHandIndex3_EndSite"

NOITOM_SKELETON_JOINTS["left_middle_finger_root"] = "LeftInHandMiddle"
NOITOM_SKELETON_JOINTS["left_middle_finger_base"] = "LeftHandMiddle1"
NOITOM_SKELETON_JOINTS["left_middle_finger_mid"] = "LeftHandMiddle2"
NOITOM_SKELETON_JOINTS["left_middle_finger_tip"] = "LeftHandMiddle3"
NOITOM_SKELETON_JOINTS["left_middle_finger_end"] = "LeftHandMiddle3_EndSite"

NOITOM_SKELETON_JOINTS["left_ring_finger_root"] = "LeftInHandRing"
NOITOM_SKELETON_JOINTS["left_ring_finger_base"] = "LeftHandRing1"
NOITOM_SKELETON_JOINTS["left_ring_finger_mid"] = "LeftHandRing2"
NOITOM_SKELETON_JOINTS["left_ring_finger_tip"] = "LeftHandRing3"
NOITOM_SKELETON_JOINTS["left_ring_finger_end"] = "LeftHandRing3_EndSite"

NOITOM_SKELETON_JOINTS["left_pinky_finger_root"] = "LeftInHandPinky"
NOITOM_SKELETON_JOINTS["left_pinky_finger_base"] = "LeftHandPinky1"
NOITOM_SKELETON_JOINTS["left_pinky_finger_mid"] = "LeftHandPinky2"
NOITOM_SKELETON_JOINTS["left_pinky_finger_tip"] = "LeftHandPinky3"
NOITOM_SKELETON_JOINTS["left_pinky_finger_end"] = "LeftHandPinky3_EndSite"


NOITOM_SKELETON_MODEL = collections.OrderedDict()
NOITOM_SKELETON_MODEL["name"] = "noitom"
NOITOM_SKELETON_MODEL["joints"] = NOITOM_SKELETON_JOINTS
NOITOM_SKELETON_COS_MAP = collections.defaultdict(dict)
NOITOM_SKELETON_COS_MAP["LeftHand"]["x"] = [0, -1, 0]
NOITOM_SKELETON_COS_MAP["LeftHand"]["y"] = [1, 0, 0]
NOITOM_SKELETON_COS_MAP["RightHand"]["x"] = [0, 1, 0]
NOITOM_SKELETON_COS_MAP["RightHand"]["y"] = [-1, 0, 0]
for k, j in NOITOM_SKELETON_JOINTS.items():
    if j is None:
        continue
    if "finger" in k or "thumb" in k:
        if "Left" in j:
            NOITOM_SKELETON_COS_MAP[j]["x"] = [0, 1, 0]
            NOITOM_SKELETON_COS_MAP[j]["y"] = [1, 0, 0]
        elif "Right" in j:
            NOITOM_SKELETON_COS_MAP[j]["x"] = [0, 1, 0]
            NOITOM_SKELETON_COS_MAP[j]["y"] = [-1, 0, 0]
NOITOM_SKELETON_MODEL["cos_map"] = NOITOM_SKELETON_COS_MAP







MO_SKELETON_JOINTS = collections.OrderedDict()
MO_SKELETON_JOINTS["root"] = None
MO_SKELETON_JOINTS["pelvis"] = "Hips"
MO_SKELETON_JOINTS["spine"] = "Spine"
MO_SKELETON_JOINTS["spine_1"] = "Spine1"
MO_SKELETON_JOINTS["spine_2"] = "Spine2"
MO_SKELETON_JOINTS["left_clavicle"] = "LeftShoulder"
MO_SKELETON_JOINTS["right_clavicle"] = "RightShoulder"
MO_SKELETON_JOINTS["left_shoulder"] = "LeftArm"
MO_SKELETON_JOINTS["right_shoulder"] = "RightArm"
MO_SKELETON_JOINTS["left_elbow"] = "LeftForeArm"
MO_SKELETON_JOINTS["right_elbow"] = "RightForeArm"
MO_SKELETON_JOINTS["left_wrist"] = "LeftHand"
MO_SKELETON_JOINTS["right_wrist"] = "RightHand"
#MO_SKELETON_JOINTS["left_finger"] = "RightInHandMiddle"
#MO_SKELETON_JOINTS["right_finger"] = "LeftInHandMiddle"
MO_SKELETON_JOINTS["left_hip"] = "LeftUpLeg"
MO_SKELETON_JOINTS["right_hip"] = "RightUpLeg"
MO_SKELETON_JOINTS["left_knee"] = "LeftLeg"
MO_SKELETON_JOINTS["right_knee"] = "RightLeg"
MO_SKELETON_JOINTS["left_ankle"] = "LeftFoot"
MO_SKELETON_JOINTS["right_ankle"] = "RightFoot"
MO_SKELETON_JOINTS["left_toe"] = "LeftToeBase"
MO_SKELETON_JOINTS["right_toe"] = "RightToeBase"
MO_SKELETON_JOINTS["left_heel"] = None
MO_SKELETON_JOINTS["right_heel"] = None
MO_SKELETON_JOINTS["neck"] = "Neck"
MO_SKELETON_JOINTS["head"] = "Head"

MO_SKELETON_MODEL = collections.OrderedDict()
MO_SKELETON_MODEL["name"] = "mo"
MO_SKELETON_MODEL["joints"] = MO_SKELETON_JOINTS
MO_SKELETON_COS_MAP = collections.defaultdict(dict)
MO_SKELETON_COS_MAP["LeftHand"]["x"] = [0, -1, 0]
MO_SKELETON_COS_MAP["LeftHand"]["y"] = [1, 0, 0]
MO_SKELETON_COS_MAP["RightHand"]["x"] = [0, 1, 0]
MO_SKELETON_COS_MAP["RightHand"]["y"] = [-1, 0, 0]
MO_SKELETON_MODEL["cos_map"] = NOITOM_SKELETON_COS_MAP


SKELETON_MODELS = collections.OrderedDict()
SKELETON_MODELS["rocketbox"] = ROCKETBOX_SKELETON_MODEL
SKELETON_MODELS["game_engine"] = GAME_ENGINE_SKELETON_MODEL
SKELETON_MODELS["raw"] = RAW_SKELETON_MODEL
SKELETON_MODELS["cmu"] = CMU_SKELETON_MODEL
SKELETON_MODELS["mcs"] = MCS_SKELETON_MODEL
SKELETON_MODELS["mh_cmu"] = MH_CMU_SKELETON_MODEL
SKELETON_MODELS["mh_cmu2"] = MH_CMU_2_SKELETON_MODEL
SKELETON_MODELS["iclone"] = ICLONE_SKELETON_MODEL
SKELETON_MODELS["moviemation"] = MOVIEMATION_SKELETON_MODEL
SKELETON_MODELS["custom"] = CUSTOM_SKELETON_MODEL
SKELETON_MODELS["captury"] = CAPTURY_SKELETON_MODEL
SKELETON_MODELS["holden"] = HOLDEN_SKELETON_MODEL
SKELETON_MODELS["mixamo"] = MIXAMO_SKELETON_MODEL
SKELETON_MODELS["max"] = MAX_SKELETON_MODEL
SKELETON_MODELS["aachen"] = AACHEN_SKELETON_MODEL
SKELETON_MODELS["mh_cmu_red"] = MH_CMU_RED_SKELETON_MODEL
SKELETON_MODELS["game_engine_wrong_root"] = GAME_ENGINE_WRONG_ROOT_SKELETON_MODEL
SKELETON_MODELS["raw2"] = RAW2_SKELETON_MODEL
SKELETON_MODELS["noitom"] = NOITOM_SKELETON_MODEL
SKELETON_MODELS["mo"] = MO_SKELETON_MODEL

STANDARD_JOINTS = ["root","pelvis", "spine_1", "spine_2", "neck", "left_clavicle", "head", "left_shoulder", "left_elbow", "left_wrist", "right_clavicle", "right_shoulder",
                    "left_hip", "left_knee", "left_ankle", "right_elbow", "right_hip", "right_knee", "left_ankle", "right_ankle", "left_toe", "right_toe"]

FINGER_JOINTS = ["left_thumb_base","left_thumb_mid", "left_thumb_tip","left_thumb_end",
                    "left_index_finger_root","left_index_finger_base","left_index_finger_mid", "left_index_finger_tip","left_index_finger_end",
                    "left_middle_finger_root","left_middle_finger_base","left_middle_finger_mid","left_middle_finger_tip","left_middle_finger_end",
                    "left_ring_finger_root","left_ring_finger_base","left_ring_finger_mid","left_ring_finger_tip", "left_ring_finger_end",
                    "left_pinky_finger_root","left_pinky_finger_base","left_pinky_finger_mid","left_pinky_finger_tip", "left_pinky_finger_end"
                   
                    "right_thumb_base","right_thumb_mid","right_thumb_tip","right_thumb_end",
                    "right_index_finger_root","right_index_finger_base","right_index_finger_mid","right_index_finger_tip","right_index_finger_end",
                    "right_middle_finger_root","right_middle_finger_base","right_middle_finger_mid","right_middle_finger_tip","right_middle_finger_end",
                    "right_ring_finger_root","right_ring_finger_base","right_ring_finger_mid","right_ring_finger_tip","right_ring_finger_end",
                    "right_pinky_finger_root","right_pinky_finger_base","right_pinky_finger_mid","right_pinky_finger_tip","right_pinky_finger_end"
                    ]
UPPER_BODY_JOINTS = ["spine", "spine_1", "spine_2","neck", "head"
                       "left_clavicle", "right_clavicle",
                       "left_shoulder", "right_shoulder",
                       "left_elbow", "right_elbow",
                       "left_wrist", "right_wrist",
                       "left_hold_point","right_hold_point"
]



JOINT_CHILD_MAP = dict()
JOINT_CHILD_MAP["root"] = "pelvis"
JOINT_CHILD_MAP["pelvis"] = "spine"
JOINT_CHILD_MAP["spine"] = "spine_1"
JOINT_CHILD_MAP["spine_1"] = "spine_2"
JOINT_CHILD_MAP["spine_2"] = "neck"
JOINT_CHILD_MAP["neck"] = "head"
JOINT_CHILD_MAP["left_clavicle"] = "left_shoulder"
JOINT_CHILD_MAP["left_shoulder"] = "left_elbow"
JOINT_CHILD_MAP["left_elbow"] = "left_wrist"
JOINT_CHILD_MAP["left_wrist"] = "left_finger"  # TODO rename to hand center
JOINT_CHILD_MAP["right_clavicle"] = "right_shoulder"
JOINT_CHILD_MAP["right_shoulder"] = "right_elbow"
JOINT_CHILD_MAP["right_elbow"] = "right_wrist"
JOINT_CHILD_MAP["right_wrist"] = "right_finger"  # TODO rename to hand center
JOINT_CHILD_MAP["left_hip"] = "left_knee"
JOINT_CHILD_MAP["left_knee"] = "left_ankle"
JOINT_CHILD_MAP["right_elbow"] = "right_wrist"
JOINT_CHILD_MAP["right_hip"] = "right_knee"
JOINT_CHILD_MAP["right_knee"] = "right_ankle"
JOINT_CHILD_MAP["left_ankle"] = "left_toe"
JOINT_CHILD_MAP["right_ankle"] = "right_toe"

JOINT_PARENT_MAP = {v: k for k, v in JOINT_CHILD_MAP.items()}

FABRIK_CHAINS = dict()
FABRIK_CHAINS["right_wrist"] = {"joint_order": ["right_shoulder", "right_elbow", "right_wrist"]}
FABRIK_CHAINS["left_wrist"] = {"joint_order": ["left_shoulder", "left_elbow", "left_wrist"]}

JOINT_CONSTRAINTS = dict()
JOINT_CONSTRAINTS["right_elbow"] = {"type":"hinge", "swing_axis": [0,-1,0], "twist_axis": [0,0,1], "k1":-90, "k2":90}
JOINT_CONSTRAINTS["left_elbow"] = {"type": "hinge", "swing_axis": [0,-1,0], "twist_axis": [0,0,1], "k1":-90, "k2":90}
#JOINT_CONSTRAINTS["right_shoulder"] = {"type":"cone", "axis": [0,0,1], "k": 0.4}
#JOINT_CONSTRAINTS["left_shoulder"] = {"type": "cone", "axis": [0,0,1], "k": 0.4}
#JOINT_CONSTRAINTS["spine"] = {"type":"cone", "axis": [0,1,0], "k": np.radians(45)}
JOINT_CONSTRAINTS["spine_1"] = {"type":"spine", "axis": [0,1,0], "tk1": -np.radians(90), "tk2": np.radians(90), "sk1": -np.radians(0), "sk2": np.radians(0)}
##OINT_CONSTRAINTS["spine"] = {"type":"shoulder", "axis": [0,1,0], "k": np.radians(20), "k1": np.radians(-90), "k2": np.radians(90) }
JOINT_CONSTRAINTS["left_shoulder"] = {"type":"shoulder", "axis": [0,0,1], "k": 0.4, "k1": np.radians(-180), "k2": np.radians(180) , "stiffness":0.9}
JOINT_CONSTRAINTS["right_shoulder"] = {"type":"shoulder", "axis": [0,0,1], "k": 0.4, "k1": np.radians(-180), "k2": np.radians(180), "stiffness":0.9}

JOINT_CONSTRAINTS["left_wrist"] = {"type":"cone", "axis": [0,0,1], "k": np.radians(90)}
JOINT_CONSTRAINTS["right_wrist"] = {"type":"cone", "axis": [0,0,1], "k": np.radians(90)}
#JOINT_CONSTRAINTS["head"] = {"type":"cone", "axis": [0,1,0], "k": np.radians(0)}
#JOINT_CONSTRAINTS["neck"] = {"type":"head", "axis": [0,1,0], "k1": -np.radians(65), "k2": np.radians(65)}
JOINT_CONSTRAINTS["head"] = {"type":"head", "axis": [0,1,0], "tk1": -np.radians(85), "tk2": np.radians(85), "sk1": -np.radians(45), "sk2": np.radians(45)}
JOINT_CONSTRAINTS["left_hold_point"] = {"type":"static"}
JOINT_CONSTRAINTS["right_hold_point"] = {"type":"static"}

STANDARD_MIRROR_MAP_LEFT = {"left_hip": "right_hip",
                       "left_knee": "right_knee",
                       "left_ankle":"right_ankle",
                       "left_clavicle": "right_clavicle",
                       "left_shoulder": "right_shoulder",
                       "left_elbow": "right_elbow",
                       "left_elbow": "right_elbow",
                       "left_wrist": "right_wrist",
                       "left_hold_point": "right_hold_point"
                        }

STANDARD_MIRROR_MAP_LEFT["left_thumb_base"] = "right_thumb_base"
STANDARD_MIRROR_MAP_LEFT["left_thumb_mid"] = "right_thumb_mid"
STANDARD_MIRROR_MAP_LEFT["left_thumb_tip"] = "right_thumb_tip"
STANDARD_MIRROR_MAP_LEFT["left_thumb_end"] = "right_thumb_end"

STANDARD_MIRROR_MAP_LEFT["left_index_finger_root"] = "right_index_finger_root"
STANDARD_MIRROR_MAP_LEFT["left_index_finger_base"] = "right_index_finger_base"
STANDARD_MIRROR_MAP_LEFT["left_index_finger_mid"] = "right_index_finger_mid"
STANDARD_MIRROR_MAP_LEFT["left_index_finger_tip"] = "right_index_finger_tip"
STANDARD_MIRROR_MAP_LEFT["left_index_finger_end"] = "right_index_finger_end"

STANDARD_MIRROR_MAP_LEFT["left_middle_finger_root"] = "right_middle_finger_root"
STANDARD_MIRROR_MAP_LEFT["left_middle_finger_base"] = "right_middle_finger_base"
STANDARD_MIRROR_MAP_LEFT["left_middle_finger_mid"] = "right_middle_finger_mid"
STANDARD_MIRROR_MAP_LEFT["left_middle_finger_tip"] = "right_middle_finger_tip"
STANDARD_MIRROR_MAP_LEFT["left_middle_finger_end"] = "right_middle_finger_end"

STANDARD_MIRROR_MAP_LEFT["left_ring_finger_root"] = "right_ring_finger_root"
STANDARD_MIRROR_MAP_LEFT["left_ring_finger_base"] = "right_ring_finger_base"
STANDARD_MIRROR_MAP_LEFT["left_ring_finger_mid"] = "right_ring_finger_mid"
STANDARD_MIRROR_MAP_LEFT["left_ring_finger_tip"] = "right_ring_finger_tip"
STANDARD_MIRROR_MAP_LEFT["left_ring_finger_end"] = "right_ring_finger_end"

STANDARD_MIRROR_MAP_LEFT["left_pinky_finger_root"] = "right_pinky_finger_root"
STANDARD_MIRROR_MAP_LEFT["left_pinky_finger_base"] = "right_pinky_finger_base"
STANDARD_MIRROR_MAP_LEFT["left_pinky_finger_mid"] = "right_pinky_finger_mid"
STANDARD_MIRROR_MAP_LEFT["left_pinky_finger_tip"] = "right_pinky_finger_tip"
STANDARD_MIRROR_MAP_LEFT["left_pinky_finger_end"] = "right_pinky_finger_end"

STANDARD_MIRROR_MAP_RIGHT = dict()
for key, value in STANDARD_MIRROR_MAP_LEFT.items():
    STANDARD_MIRROR_MAP_RIGHT[value] = key

STANDARD_MIRROR_MAP = dict()
STANDARD_MIRROR_MAP.update(STANDARD_MIRROR_MAP_LEFT)
STANDARD_MIRROR_MAP.update(STANDARD_MIRROR_MAP_RIGHT)
