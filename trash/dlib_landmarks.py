import dlib
from os import path
import cv2
import numpy as np
import operators.filter
from dataflow import ports


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


def _dlib_track_to_array(prediction):
    np_pts = np.zeros([68, 2], dtype=np.float32)
    for i, p in enumerate(prediction.parts()):
        np_pts[i, :] = (p.x, p.y)
    return np_pts


class Extractor:
    """
    Tracks faces in video frames, extracts facial features
    """

    def __init__(self,
                 detection_threshold=0.05,
                 tracking_threshold=10,
                 use_opencv=False):
        self.detector = dlib.get_frontal_face_detector()
        self.tracker = dlib.correlation_tracker()
        model_dir = path.join(path.dirname(path.abspath(__file__)), 'models')
        self.predictor = dlib.shape_predictor(
            path.join(model_dir, 'shape_predictor_68_face_landmarks.dat'))
        self.face_cascade = cv2.CascadeClassifier(
            path.join(model_dir, 'haarcascade_frontalface_alt2.xml'))
        self.good = False
        self.dt = detection_threshold
        self.tt = tracking_threshold
        self.u_cv2 = use_opencv
        self.roi_size = (800, 600)
        self.roi_buffer = np.zeros((self.roi_size[0], self.roi_size[1], 3), np.uint8)
        self.dlib_roi = dlib.rectangle(0, 0, self.roi_size[1], self.roi_size[0])
        self.source_bounds = ports.EventSource()
        self.source_landmarks = ports.EventSource()
        self.sink_image = ports.EventSink(self.track_face)

    def reset(self):
        self.good = False

    def track_face(self, opencv_img):
        if not self.good:
            dets, scores, _ = self.detector.run(opencv_img, 1, self.dt)
            # print('detection scores: {}'.format(scores))
            if not dets:  # try opencv detector
                if self.u_cv2:
                    faces = self.face_cascade.detectMultiScale(opencv_img, 1.2, 7, 0, (50, 50))
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        dets = [dlib.rectangle(int(x), int(y), int(x + w), int(y + h))]
            if not dets:
                return None
            self.good = True
            self.tracker.start_track(opencv_img, dets[0])
        else:
            score = self.tracker.update(opencv_img)
            # print('tracking score: {}'.format(score))
            if score < self.tt:
                self.good = False
                return self.track_face(opencv_img)

        d = self.tracker.get_position()

        # print self.history[0:self.N - 1, :]
        # print y[0:self.N - 1, :]
        # print '----------------------------------------------'

        self.source_bounds.fire(d)

        rect = dlib.rectangle(
            int(round(d.left())),
            int(round(d.top())),
            int(round(d.right())),
            int(round(d.bottom())))

        if min(rect.top(), rect.left()) < 0 \
                or rect.bottom() > opencv_img.shape[0] \
                or rect.right() > opencv_img.shape[1]:
            return None

        cv2.resize(opencv_img[rect.top():rect.bottom(), rect.left():rect.right()],
                   (self.roi_size[1], self.roi_size[0]), self.roi_buffer)

        prediction = self.predictor(self.roi_buffer, self.dlib_roi)

        scale = np.array([(d.right() - d.left()) / self.roi_size[1],
                          (d.bottom() - d.top()) / self.roi_size[0]], dtype=np.float32)
        shift = np.array([d.left(), d.top()], dtype=np.float32)

        self.source_landmarks.fire(_dlib_track_to_array(prediction) * scale + shift)
