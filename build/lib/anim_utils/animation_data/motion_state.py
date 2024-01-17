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
import collections
import numpy as np
from transformations import quaternion_inverse, quaternion_conjugate, quaternion_multiply, quaternion_matrix

def get_random_color():
    random_color = np.random.rand(3, )
    if np.sum(random_color) < 0.5:
        random_color += np.array([0, 0, 1])
    return random_color.tolist()


class MotionStateInterface(object):
    def update(self, dt):
        pass

    def get_pose(self, frame_idx=None):
        pass

    def reset(self):
        pass

    def get_frames(self):
        pass

    def get_n_frames(self):
        pass

    def get_frame_time(self):
        pass

    def set_frame_idx(self, frame_idx):
        pass

    def set_color_annotation_from_labels(self, labels, colors):
        pass

    def replace_current_frame(self, frame):
        pass

    def replace_frames(self, new_frames):
        pass

    def set_color_annotation(self, semantic_annotation, color_map):
        pass

    def set_time_function(self, time_function):
        pass

    def set_color_annotation_legacy(self, annotation, color_map):
        pass

    def set_random_color_annotation(self):
        pass

    def get_current_frame_idx(self):
        pass

    def get_current_annotation(self):
        return None

    def get_n_annotations(self):
        return 0

    def get_semantic_annotation(self):
        return list()

    def set_time(self, t):
        return


class MotionState(MotionStateInterface):
    """ Wrapper of a MotionVector to add functions for keeping track of the current time and frame index as well as semantic information.
    """
    def __init__(self, mv):
        self.mv = mv
        self.time = 0
        self.frame_idx = 0
        self.play = False
        self._frame_annotation = None
        self.label_color_map = dict()
        self._semantic_annotation = collections.OrderedDict()
        self.events = dict()
        self.n_annotations = 0
        self.hold_frames = list() # sorted list
        self.paused = False
        self.interpolate = False
        self.tick = None
        self._time_function = None

    def update(self, dt):
        if self.play and not self.paused:
            self.time += dt
            self.frame_idx = int(self.time / self.mv.frame_time)
            if len(self.hold_frames) > 0:
                #print("hold frame", self.frame_idx,self.hold_frame, self.mv.n_frames)
                if self.frame_idx >= self.hold_frames[0]:
                    self.paused = True
                    self.frame_idx = self.hold_frames[0]
                    self.hold_frames = self.hold_frames[1:]  # remove entry
            if self.tick is not None:
                self.tick.update(dt)
            if self.frame_idx >= self.mv.n_frames:
                self.reset()
                return True
        return False


    def get_pose(self, frame_idx=None):
        if self.interpolate:
            return self.get_pose_interpolate(frame_idx)
        else:
            if frame_idx is None:
                return self.mv.frames[self.frame_idx]
            else:
                return self.mv.frames[frame_idx]

    def get_pose_interpolate(self, frame_idx=None):
        #FIXME interpolation causes jump in translation at start and end
        if frame_idx is None:
            start_frame_idx = int(self.time / self.mv.frame_time)
            end_frame_idx = start_frame_idx+1
            t = (self.time % self.mv.frame_time)/ self.mv.frame_time
            if end_frame_idx >= self.mv.n_frames:
                end_frame_idx = start_frame_idx
                t = 0
            #print("get pose", self.time, start_frame_idx, t, end_frame_idx)
            return self.mv.interpolate(start_frame_idx, end_frame_idx, t)
            #return self.mv.frames[self.frame_idx]
        else:
            return self.mv.frames[frame_idx]

    def reset(self):
        self.time = 0
        self.frame_idx = 0
        self.paused = False
        if self.tick is not None:
            self.tick.reset()

    def get_frames(self):
        return self.mv.frames

    def get_n_frames(self):
        return self.mv.n_frames

    def get_frame_time(self):
        return self.mv.frame_time

    def get_current_annotation(self):
        return self._frame_annotation[self.frame_idx]


    def get_current_annotation_label(self):
        if self._frame_annotation is not None and 0 <= self.frame_idx < len(self._frame_annotation):
            return self._frame_annotation[self.frame_idx]["label"]
        else:
            return ""

    def set_frame_idx(self, frame_idx):
        self.frame_idx = frame_idx
        self.time = self.get_frame_time() * self.frame_idx

    def set_color_annotation_from_labels(self, labels, colors=None):

        if colors is None:
            def getRGBfromI(RGBint):
                """ https://stackoverflow.com/questions/33124347/convert-integers-to-rgb-values-and-back-with-python
                """
                blue = RGBint & 255
                green = (RGBint >> 8) & 255
                red = (RGBint >> 16) & 255
                return red, green, blue
            n_classes = int(max(labels)+1)
            step_size = 255255255 / n_classes
            self.label_color_map = dict()
            for n in range(n_classes):
                l = int(n*step_size)
                self.label_color_map[str(n)] = np.array(getRGBfromI(l), dtype=float)/255
        else:
            n_classes = len(colors)
            for n in range(n_classes):
                self.label_color_map[str(n)] = colors[n]

        self._semantic_annotation = collections.OrderedDict()
        for n in range(n_classes):
            self._semantic_annotation[str(n)] = []
        self.n_annotations = len(labels)
        self._frame_annotation = []
        for idx, label in enumerate(labels):
            label = str(int(label))
            self._frame_annotation.append({"label": label, "color": self.label_color_map[label]})
            self._semantic_annotation[label].append(idx)

    def replace_current_frame(self, frame):
        if 0 <= self.frame_idx < self.mv.n_frames:
            self.mv.frames[self.frame_idx] = frame

    def replace_frames(self, new_frames):
        self.time = 0
        self.frame_idx = 0
        self.mv.frames = new_frames
        self.mv.n_frames = len(new_frames)

    def set_color_annotation(self, semantic_annotation, color_map):
        self._semantic_annotation = semantic_annotation
        self.label_color_map = color_map
        self.n_annotations = self.mv.n_frames
        self._frame_annotation = []
        for idx in range(self.n_annotations):
            self._frame_annotation.append({"label": "none", "color": [1,1,1]})
        for label in semantic_annotation.keys():
            if label in color_map.keys():
                entry = {"label": label, "color": color_map[label]}
                if len(semantic_annotation[label]) > 0 and type(semantic_annotation[label][0]) == list:
                    for indices in semantic_annotation[label]:
                        for i in indices:
                            if i < self.n_annotations:
                                self._frame_annotation[i] = entry
                else:
                    for idx in semantic_annotation[label]:
                        if idx < self.n_annotations:
                            self._frame_annotation[idx] = entry
                        else:
                            print("warning", idx, self.n_annotations)


    def set_time_function(self, time_function):
        self._time_function = time_function

    def set_color_annotation_legacy(self, annotation, color_map):
        self._semantic_annotation = collections.OrderedDict()
        self.label_color_map = color_map
        self.n_annotations = len(annotation)
        self._frame_annotation = []
        for idx, label in enumerate(annotation):

            self._frame_annotation.append({"label": label,
                                           "color": color_map[label]})

            if label not in list(self._semantic_annotation.keys()):
                self._semantic_annotation[label] = []
            self._semantic_annotation[label].append(idx)

    def set_random_color_annotation(self):
        self._frame_annotation = []
        for idx in range(self.mv.n_frames):
            self._frame_annotation.append({"color": get_random_color()})
        self.n_annotations = self.mv.n_frames

    def get_current_frame_idx(self):
        return self.frame_idx

    def get_n_annotations(self):
        return self.n_annotations

    def get_semantic_annotation(self):
        return list(self._semantic_annotation.items())

    def get_label_color_map(self):
        return self.label_color_map

    def apply_delta_frame(self, skeleton, delta_frame):
        self.mv.apply_delta_frame(skeleton, delta_frame)

    def set_time(self, t):
        self.frame_idx = int(t / self.get_frame_time())
        self.frame_idx = min(self.frame_idx, self.mv.n_frames-1)
        #self.time = self.frame_idx*self.get_frame_time()
        self.time = t

    def set_position(self, position):
        delta = position - self.mv.frames[self.frame_idx, :3]
        self.mv.frames[:, :3] += delta

    def set_orientation(self, orientation):
        print("rotate frames")
        current_q = self.mv.frames[self.frame_idx, 3:7]
        delta_q = quaternion_multiply(orientation, quaternion_inverse(current_q))
        delta_q /= np.linalg.norm(delta_q)
        delta_m = quaternion_matrix(delta_q)[:3, :3]
        for idx in range(self.mv.n_frames):
            old_t = self.mv.frames[idx, :3]
            old_q = self.mv.frames[idx, 3:7]
            self.mv.frames[idx, :3] = np.dot(delta_m, old_t)[:3]
            self.mv.frames[idx, 3:7] = quaternion_multiply(delta_q, old_q)
            #self.mv.frames[i, 3:7] = quaternion_multiply(delta_q, old_q)

    def set_ticker(self, tick):
        self.tick = tick

class MotionSplineState(MotionStateInterface):
    def __init__(self, spline, frame_time):
        self.spline = spline
        self.time = 0.0
        self.frame_idx = 0.0
        self.frame_time = frame_time
        self.events = dict()

    def update(self, dt):
        self.time += dt
        self.frame_idx = self.time / self.frame_time
        if self.frame_idx >= self.spline.get_domain()[-1]:
            self.reset()
            return True
        else:
            return False

    def get_pose(self, frame_idx=None):
        if frame_idx is None:
            return self.spline.evaluate(self.frame_idx)
        else:
            return self.spline.evaluate(frame_idx)

    def reset(self):
        self.time = 0

    def get_frames(self):
        return self.spline.get_motion_vector()

    def get_last_frame(self):
        return self.spline.evaluate(self.spline.get_domain()[-1])

    def get_n_frames(self):
        return self.spline.get_domain()[-1]

    def get_frame_time(self):
        return self.frame_time

    def set_time(self, t):
        self.time = t

