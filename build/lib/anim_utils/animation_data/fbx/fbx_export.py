"""
Code to export an animated skeleton to an FBX file based on a skeleton and a motion vector.
The code is based on Example01 of the FBX SDK samples.
"""

from transformations import euler_from_quaternion
import numpy as np
import FbxCommon
from fbx import *


def create_scene(sdk_manager, scene, skeleton, frames, frame_time):
    info = FbxDocumentInfo.Create(sdk_manager, "SceneInfo")
    info.mTitle = "MotionExportScene"
    scene.SetSceneInfo(info)
    root_node = create_skeleton(sdk_manager, "", skeleton)
    scene.GetRootNode().LclTranslation.Set(FbxDouble3(0, 0, 0))
    scene.GetRootNode().AddChild(root_node)
    set_rest_pose(sdk_manager, scene, root_node, skeleton)
    set_animation_curves(scene, root_node, skeleton, frames, frame_time)
    return root_node


def create_skeleton(sdk_manager, name, skeleton):
    root_node = create_root_node(sdk_manager, name, skeleton)
    return root_node

def create_root_node(sdk_manager, name, skeleton):
    skeleton_root = create_skeleton_nodes_recursively(sdk_manager, name, skeleton, skeleton.root)
    node_type = FbxSkeleton.eRoot
    node_attribute = FbxSkeleton.Create(sdk_manager, name)
    node_attribute.SetSkeletonType(node_type)
    skeleton_root.SetNodeAttribute(node_attribute)
    return skeleton_root

def create_root_node2(sdk_manager, name, skeleton):
    node_type = FbxSkeleton.eRoot
    node_attribute = FbxSkeleton.Create(sdk_manager, name)
    node_attribute.SetSkeletonType(node_type)
    extra_root_node = FbxNode.Create(sdk_manager,  "Root")
    extra_root_node.SetNodeAttribute(node_attribute)
    skeleton_root = create_skeleton_nodes_recursively(sdk_manager, name, skeleton, skeleton.root)
    extra_root_node.AddChild(skeleton_root)
    return extra_root_node

def create_skeleton_nodes_recursively(sdk_manager, skeleton_name, skeleton, node_name):
    node = skeleton.nodes[node_name]
    name = skeleton_name + node_name
    skeleton_node_attribute = FbxSkeleton.Create(sdk_manager, skeleton_name)
    node_type = FbxSkeleton.eLimbNode
    skeleton_node_attribute.SetSkeletonType(node_type)
    skeleton_node = FbxNode.Create(sdk_manager, name)
    skeleton_node.SetNodeAttribute(skeleton_node_attribute)
    t = FbxDouble3(node.offset[0], node.offset[1], node.offset[2])
    skeleton_node.LclTranslation.Set(t)
    for c_node in node.children:
        c_name = c_node.node_name
        c_node = create_skeleton_nodes_recursively(sdk_manager, skeleton_name, skeleton, c_name)
        skeleton_node.AddChild(c_node)
    return skeleton_node


def set_rest_pose_recursively(pose, fbx_node, skeleton):
    name = fbx_node.GetName()
    if name in skeleton.nodes.keys():
        node = skeleton.nodes[name]
        t = node.offset

        l_t = FbxVector4(t[0], t[1], t[2])
        l_t = FbxVector4(0,0,0)
        l_r = FbxVector4(0.0, 0.0, 0.0)
        l_s = FbxVector4(1.0, 1.0, 1.0)

        transform = FbxMatrix()
        transform.SetTRS(l_t, l_r, l_s)
        pose.Add(fbx_node, transform, True)# maybe set this to false
    n_children = fbx_node.GetChildCount()
    for idx in range(n_children):
        c_node = fbx_node.GetChild(idx)
        set_rest_pose_recursively(pose, c_node, skeleton)


def set_rest_pose(sdk_manager, scene, root_node, skeleton):
    pose = FbxPose.Create(sdk_manager, "RestPose")
    set_rest_pose_recursively(pose, root_node, skeleton)
    scene.AddPose(pose)


def create_translation_curve(fbx_node, anim_layer, euler_frames, frame_time, dimension, dim_idx):
    t = FbxTime()
    curve = fbx_node.LclTranslation.GetCurve(anim_layer, dimension, True)
    curve.KeyModifyBegin()
    for idx, frame in enumerate(euler_frames):
        t.SetSecondDouble(idx * frame_time)
        key_index = curve.KeyAdd(t)[0]
        curve.KeySetValue(key_index, frame[dim_idx])
        curve.KeySetInterpolation(key_index, FbxAnimCurveDef.eInterpolationConstant)
    curve.KeyModifyEnd()


def create_euler_curve(fbx_node, anim_layer, euler_frames, frame_time, dimension, dim_idx):
    t = FbxTime()
    curve = fbx_node.LclRotation.GetCurve(anim_layer, dimension, True)
    curve.KeyModifyBegin()
    for idx, frame in enumerate(euler_frames):
        t.SetSecondDouble(idx * frame_time)
        key_index = curve.KeyAdd(t)[0]
        curve.KeySetValue(key_index, frame[dim_idx])
        curve.KeySetInterpolation(key_index, FbxAnimCurveDef.eInterpolationConstant)
    curve.KeyModifyEnd()


def create_translation_curves(fbx_node, anim_layer, euler_frames, frame_time):
    create_translation_curve(fbx_node, anim_layer, euler_frames, frame_time, "X", 0)
    create_translation_curve(fbx_node, anim_layer, euler_frames, frame_time, "Y", 1)
    create_translation_curve(fbx_node, anim_layer, euler_frames, frame_time, "Z", 2)


def create_rotation_curves(fbx_node, anim_layer, skeleton, euler_frames, frame_time):
    node_name = fbx_node.GetName()
    if node_name not in skeleton.animated_joints:
        return
    node_idx = skeleton.animated_joints.index(node_name)
    offset = node_idx * 3 + 3
    create_euler_curve(fbx_node, anim_layer, euler_frames, frame_time, "X", offset)
    create_euler_curve(fbx_node, anim_layer, euler_frames, frame_time, "Y", offset+1)
    create_euler_curve(fbx_node, anim_layer, euler_frames, frame_time, "Z", offset+2)


def add_rotation_curves_recursively(fbx_node, anim_layer, skeleton, euler_frames, frame_time):
    create_rotation_curves(fbx_node, anim_layer, skeleton, euler_frames, frame_time)
    n_children = fbx_node.GetChildCount()
    for idx in range(n_children):
        c_node = fbx_node.GetChild(idx)
        add_rotation_curves_recursively(c_node, anim_layer, skeleton, euler_frames, frame_time)


def convert_quaternion_to_euler_frame(skeleton, frame):
    n_dims = len(skeleton.animated_joints) * 3 + 3
    euler_frame = np.zeros(n_dims)
    euler_frame[:3] = frame[:3]
    target_offset = 3
    src_offset = 3
    for idx, node in enumerate(skeleton.animated_joints):
        q = frame[src_offset:src_offset + 4]
        e = euler_from_quaternion(q)
        euler_frame[target_offset:target_offset + 3] = np.degrees(e)
        target_offset += 3
        src_offset += 4
    return euler_frame

def set_animation_curves(scene, root_node, skeleton, frames, frame_time):

    # convert frames from quaternion to euler
    euler_frames = []
    for frame in frames:
        euler_frame = convert_quaternion_to_euler_frame(skeleton, frame)
        euler_frames.append(euler_frame)

    anim_stack_name = "default"
    anim_stack = FbxAnimStack.Create(scene, anim_stack_name)
    anim_layer = FbxAnimLayer.Create(scene, "Base Layer")
    anim_stack.AddMember(anim_layer)
    
    create_translation_curves(root_node, anim_layer, euler_frames, frame_time)
    add_rotation_curves_recursively(root_node, anim_layer, skeleton, euler_frames, frame_time)


def export_motion_vector_to_fbx_file(skeleton, motion_vector, out_file_name, fbx_axis_system=FbxAxisSystem.MayaZUp):
    """ Save a scene with a single skeleton and animation curves for the root and joints
    """
    sdk_manager, scene = FbxCommon.InitializeSdkObjects()
    root_node = create_scene(sdk_manager, scene, skeleton, motion_vector.frames, motion_vector.frame_time)
    fbx_axis_system.MayaZUp.ConvertScene(scene)
    #sdk_manager.GetIOSettings().SetBoolProp(EXP_FBX_EMBEDDED, True)
    save_scene(sdk_manager, scene, out_file_name, pEmbedMedia=False)
    print("finished")

    sdk_manager.Destroy()


def save_scene(pSdkManager, pScene, pFilename, pFileFormat = -1, pEmbedMedia = False):
    """ copied from fbx common
    changed settings to a Blender compatible version according to
    http://stackoverflow.com/questions/29833986/fbx-sdk-exporting-into-older-fbx-file-format
    """

    lExporter = FbxExporter.Create(pSdkManager, "")
    lExporter.SetFileExportVersion("FBX201200", FbxSceneRenamer.eFBX_TO_FBX)

    if pFileFormat < 0 or pFileFormat >= pSdkManager.GetIOPluginRegistry().GetWriterFormatCount():
        pFileFormat = pSdkManager.GetIOPluginRegistry().GetNativeWriterFormat()
        if not pEmbedMedia:
            lFormatCount = pSdkManager.GetIOPluginRegistry().GetWriterFormatCount()
            for lFormatIndex in range(lFormatCount):
                if pSdkManager.GetIOPluginRegistry().WriterIsFBX(lFormatIndex):
                    lDesc = pSdkManager.GetIOPluginRegistry().GetWriterFormatDescription(lFormatIndex)
                    if "binary" in lDesc:
                        pFileFormat = lFormatIndex
                        break

    if not pSdkManager.GetIOSettings():
        ios = FbxIOSettings.Create(pSdkManager, IOSROOT)
        pSdkManager.SetIOSettings(ios)

    pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_MATERIAL, True)
    pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_TEXTURE, True)
    pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_EMBEDDED, pEmbedMedia)
    pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_SHAPE, True)
    pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_GOBO, True)
    pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_ANIMATION, True)
    pSdkManager.GetIOSettings().SetBoolProp(EXP_FBX_GLOBAL_SETTINGS, True)

    result = lExporter.Initialize(pFilename, pFileFormat, pSdkManager.GetIOSettings())
    if result == True:
        result = lExporter.Export(pScene)

    lExporter.Destroy()
    return result