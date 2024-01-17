import FbxCommon
from fbx import *
import numpy as np
import collections
from transformations import quaternion_matrix, euler_from_quaternion, quaternion_from_euler
from anim_utils.animation_data.skeleton_node import SKELETON_NODE_TYPE_ROOT,SKELETON_NODE_TYPE_JOINT, SKELETON_NODE_TYPE_END_SITE

def load_fbx_file(file_path):
    importer = FBXImporter(file_path)
    importer.load()
    importer.destroy()
    return importer.skeleton, importer.animations


class FBXImporter(object):
    def __init__(self, file_path):
        self.skeleton = None
        self.skeleton_root_node = None
        self.skinning_data = dict()
        self.animations = dict()
        self.mesh_list = []

        (self.sdk_manager, self.fbx_scene) = FbxCommon.InitializeSdkObjects()
        FbxCommon.LoadScene(self.sdk_manager, self.fbx_scene, file_path)
        FbxAxisSystem.OpenGL.ConvertScene(self.fbx_scene)

    def destroy(self):
        self.sdk_manager.Destroy()

    def load(self):
        self.skeleton = None
        self.skinning_data = dict()
        self.animations = dict()
        self.mesh_list = []
        root_node = self.fbx_scene.GetRootNode()
        self.parseFBXNodeHierarchy(root_node, 0)
        if self.skeleton_root_node is not None:
            self.animations = self.read_animations(self.skeleton_root_node)


    def parseFBXNodeHierarchy(self, fbx_node, depth):

        n_attributes = fbx_node.GetNodeAttributeCount()
        for idx in range(n_attributes):
            attribute = fbx_node.GetNodeAttributeByIndex(idx)
            if self.skeleton is None and attribute.GetAttributeType() == FbxNodeAttribute.eSkeleton:
                self.skeleton_root_node = fbx_node
                self.skeleton = self.create_skeleton(fbx_node)

        for idx in range(fbx_node.GetChildCount()):
            self.parseFBXNodeHierarchy(fbx_node.GetChild(idx), depth + 1)


    def create_skeleton(self, node):
        current_time = FbxTime()
        current_time.SetFrame(0, FbxTime.eFrames24)
        scale = node.EvaluateGlobalTransform().GetS()[0]
        def add_child_node_recursively(skeleton, fbx_node):
            node_name = fbx_node.GetName()
            node_idx = len(skeleton["animated_joints"])
            localTransform = fbx_node.EvaluateLocalTransform(current_time)
            #lT = localTransform.GetT()
            o = fbx_node.LclTranslation.Get()
            offset = scale*np.array([o[0], o[1], o[2]])
            q = localTransform.GetQ()
            rotation = np.array([q[3],q[0], q[1], q[2]])

            node = {"name": node_name,
                        "children": [],
                        "channels": [],
                        "offset": offset,
                        "rotation": rotation,
                        "fixed": True,
                        "index": -1,
                        "quaternion_frame_index": -1,
                        "node_type": SKELETON_NODE_TYPE_JOINT}
            n_children = fbx_node.GetChildCount()
            if n_children > 0:
                node["channels"] = ["Xrotation", "Yrotation", "Zrotation"]
                node["index"] = node_idx
                node["quaternion_frame_index"] = node_idx
                node["fixed"] = False
                skeleton["animated_joints"].append(node_name)
            for idx in range(n_children):
                c_node = add_child_node_recursively(skeleton, fbx_node.GetChild(idx))
                node["children"].append(c_node["name"])
            skeleton["nodes"][node_name] = node
            return node

        o = node.LclTranslation.Get()
        offset = scale*np.array([o[0], o[1], o[2]])
        e = node.LclRotation.Get()
        rotation = quaternion_from_euler(*e, axes='sxyz')
        root_name = node.GetName()
        skeleton = dict()
        skeleton["animated_joints"] = [root_name]
        skeleton["node_names"] = dict()
        skeleton["nodes"] = collections.OrderedDict()
        skeleton["frame_time"] = 0.013889
        root_node = {"name": root_name,
                    "children": [],
                    "channels": ["Xposition", "Yposition", "Zposition",
                                 "Xrotation", "Yrotation", "Zrotation"],
                    "offset": offset,
                    "rotation": rotation,
                    "fixed": False,
                    "index": 0,
                    "quaternion_frame_index": 0,
                    "inv_bind_pose": np.eye(4),
                    "node_type": SKELETON_NODE_TYPE_ROOT}

        skeleton["nodes"][root_name] = root_node
        for idx in range(node.GetChildCount()):
            c_node = add_child_node_recursively(skeleton, node.GetChild(idx))
            root_node["children"].append(c_node["name"])

        skeleton["root"] = root_name
        print('animated_joints', len(skeleton["animated_joints"]))
        return skeleton


    def read_animations(self, temp_node):
        """src: http://gamedev.stackexchange.com/questions/59419/c-fbx-animation-importer-using-the-fbx-sdk
        """
        anims = dict()
        count = self.fbx_scene.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimStack.ClassId))
        for idx in range(count):
            anim = dict()
            anim_stack = self.fbx_scene.GetSrcObject(FbxCriteria.ObjectType(FbxAnimStack.ClassId), idx)
            self.fbx_scene.SetCurrentAnimationStack(anim_stack)
            anim_name = anim_stack.GetName()
            anim_layer = anim_stack.GetMember(FbxCriteria.ObjectType(FbxAnimLayer.ClassId), 0)
            mLocalTimeSpan = anim_stack.GetLocalTimeSpan()
            start = mLocalTimeSpan.GetStart()
            end = mLocalTimeSpan.GetStop()
            anim["n_frames"] = end.GetFrameCount(FbxTime.eFrames24) - start.GetFrameCount(FbxTime.eFrames24) + 1
            anim["duration"] = end.GetSecondCount() - start.GetSecondCount()
            anim["frame_time"] =  anim["duration"]/anim["n_frames"] # 0.013889
            anim["curves"] = collections.OrderedDict()
            print("found animation", anim_name, anim["n_frames"], anim["duration"])
            is_root = True
            nodes = []
            while temp_node is not None:
                name = temp_node.GetName()
                if has_curve(anim_layer, temp_node):
                    if is_root:
                        anim["curves"][name] = get_global_anim_curve(temp_node, start, end)
                        is_root = False
                    else:
                        anim["curves"][name] = get_local_anim_curve(temp_node, start, end)
                for i in range(temp_node.GetChildCount()):
                    nodes.append(temp_node.GetChild(i))
                if len(nodes) > 0:
                    temp_node = nodes.pop(0)
                else:
                    temp_node = None
            anims[anim_name] = anim
        return anims

def get_local_anim_curve(node, start, end):
    curve= []
    current_t = FbxTime()
    for frame_idx in range(start.GetFrameCount(FbxTime.eFrames24), end.GetFrameCount(FbxTime.eFrames24)):
        current_t.SetFrame(frame_idx, FbxTime.eFrames24)
        local_transform = node.EvaluateLocalTransform(current_t)
        q = local_transform.GetQ()
        t = local_transform.GetT()
        transform = {"rotation": [q[3], q[0], q[1], q[2]],
                        "translation": [t[0], t[1], t[2]]}
        curve.append(transform)
    return curve

def get_global_anim_curve(node, start, end):
    curve= []
    current_t = FbxTime()
    for frame_idx in range(start.GetFrameCount(FbxTime.eFrames24), end.GetFrameCount(FbxTime.eFrames24)):
        current_t.SetFrame(frame_idx, FbxTime.eFrames24)
        local_transform = node.EvaluateGlobalTransform(current_t)
        q = local_transform.GetQ()
        t = local_transform.GetT()
        transform = {"rotation": [q[3], q[0], q[1], q[2]],
                        "translation": [t[0], t[1], t[2]]}
        curve.append(transform)
    return curve

def has_curve(anim_layer, node):
    translation = node.LclTranslation.GetCurve(anim_layer, "X")
    rotation = node.LclRotation.GetCurve(anim_layer, "X")
    return rotation is not None or translation is not None


def FBXMatrixToNumpy(m):
    q = m.GetQ()
    t = m.GetT()
    m = quaternion_matrix([q[0], q[1], q[2], q[3]])
    m[:3,3] = t[0], t[1],t[2]
    return m

