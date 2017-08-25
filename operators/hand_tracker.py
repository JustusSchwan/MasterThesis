from operators import dlib_tracker
from dataflow import ports
import numpy as np
import dlib
from operators import clustering
import copy


class HandTracker:
    def __init__(self, left):
        self.tracker = dlib_tracker.HypothesisTracker(threshold=5)
        self.left = left
        self.tracker.sink_bounds << self._out_bounds_guess
        self.tracker.sink_hypotheses << self._out_hypothesis_check
        self.sink_skin_color = ports.StateSink()
        self.sink_clusters = ports.StateSink()
        self.sink_face_bounds = ports.StateSink()
        self.sink_image = self.tracker.sink_image

    def track(self):
        return self.tracker.track()

    # select only the clusters that are outside, below and left (or right) of the facial region
    def _centroid_criterion(self, c):
        face_box = self.sink_face_bounds.get()
        if face_box is None:
            return False
        h = face_box.bottom() - face_box.top()
        location_crit = c[0] < face_box.left() or c[0] > face_box.right() \
                        or \
                        c[1] > face_box.bottom() + h / 2 or c[1] < face_box.top() - h / 2
        if self.left:
            location_crit = location_crit and c[0] < (face_box.left() + face_box.right()) / 2
        else:
            location_crit = location_crit and c[0] > (face_box.left() + face_box.right()) / 2
        return location_crit

    def _out_bounds_guess(self):
        clusters = self.sink_clusters.get()
        if clusters is None:
            return None
        clusters = clustering.cluster_filter(copy.copy(clusters), condition_centroids=self._centroid_criterion)
        # select largest remaining cluster
        if len(clusters.counts) == 0:
            return None
        largest = np.argmax(clusters.counts)
        mean = clusters.centroids[largest, :]
        std = clusters.std[largest, :]
        box = clusters.bounds[largest, :]
        # return dlib.rectangle(int(mean[0] - 1.5 * std[0]), int(mean[1] - 1.5 * std[1]), int(mean[0] + 1.5 * std[0]),
        #                       int(mean[1] + 1.5 * std[1]))
        return dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))

    def _out_hypothesis_check(self, rect):
        skin = self.sink_skin_color.get()
        if skin is None:
            return False
        if rect.top() == rect.bottom() or rect.left() == rect.right():
            return False
        np_rect = np.clip(np.array(dlib_tracker.dlib_rect_to_array(rect), dtype=np.int32), a_min=0,
                          a_max=[skin.shape[1], skin.shape[0], skin.shape[1], skin.shape[0]])
        num_samples = min((np_rect[3] - np_rect[1]) * (np_rect[2] - [np_rect[0]]), 1000)
        if num_samples == 0:
            return False
        mean = np.mean(
            np.random.choice(
                np.reshape(skin[np_rect[1]:np_rect[3], np_rect[0]:np_rect[2]], -1), size=num_samples))
        return mean > 20
