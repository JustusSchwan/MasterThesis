import dlib
from os import path
from dataflow import ports
import numpy as np


class FaceDetector:
    def __init__(self, threshold=0.05):
        self.detector = dlib.get_frontal_face_detector()
        self.threshold = threshold
        self.sink_image = ports.StateSink()

    def out_bounds(self):
        img = self.sink_image.get()
        if img is None:
            return
        dets, _, _ = self.detector.run(img, 0, self.threshold)
        if not dets:
            return None
        return dets[0]


def _dlib_track_to_array(prediction):
    np_pts = np.zeros([68, 2], dtype=np.float32)
    for i, p in enumerate(prediction.parts()):
        np_pts[i, :] = (p.x, p.y)
    return np_pts


class LandmarkDetector:
    def __init__(self):
        model_dir = path.join(path.dirname(path.abspath(__file__)), 'models')
        self.predictor = dlib.shape_predictor(
            path.join(model_dir, 'shape_predictor_68_face_landmarks.dat'))
        self.sink_image = ports.StateSink()
        self.sink_bounds = ports.StateSink()

    def __call__(self):
        return self.detect_landmarks()

    def detect_landmarks(self):
        img = self.sink_image.get()
        bounds = self.sink_bounds.get()
        if img is None or bounds is None:
            return None
        rect = dlib.rectangle(
            int(round(bounds.left())),
            int(round(bounds.top())),
            int(round(bounds.right())),
            int(round(bounds.bottom())))

        if min(rect.top(), rect.left()) < 0 or rect.bottom() > img.shape[0] or rect.right() > img.shape[1]:
            return None

        return _dlib_track_to_array(prediction=self.predictor(img, rect))


def outline(prediction):
    return prediction[0:17, :]


def brow_l(prediction):
    return prediction[17:22, :]


def brow_r(prediction):
    return prediction[22:27, :]


def nose_above(prediction):
    return prediction[27:31, :]


def nose_below(prediction):
    return prediction[31:36, :]


def eye_l(prediction):
    return prediction[36:42, :]


def eye_r(prediction):
    return prediction[42:48, :]


def mouth_out(prediction):
    return prediction[48:60, :]


def mouth_in(prediction):
    return prediction[60:68, :]


def get_all(prediction):
    return prediction[0:68, :]
