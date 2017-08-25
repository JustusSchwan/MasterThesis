from os import path

import cv2
import dlib
import numpy as np

from trash import filter


class dlibTracker:
    """
    Tracks faces in video frames, extracts facial features
    """

    def _iter_predict(self, prediction, start, end):
        for p in range(start, end):
            yield (int(prediction[p, 0]), int(prediction[p, 1]))

    def iter_eye_l(self, prediction):
        return self._iter_predict(prediction, 36, 42)

    def iter_eye_r(self, prediction):
        return self._iter_predict(prediction, 42, 48)

    def iter_mouth(self, prediction):
        return self._iter_predict(prediction, 48, 68)

    def iter_all(self, prediction):
        return self._iter_predict(prediction, 0, 68)

    def get_pred(self, prediction):
        return prediction.parts()

    def get_rect(self, prediction):
        return prediction.rect

    def get_ndarray(self, prediction):
        np_pts = np.zeros([68, 2], dtype=np.float32)
        for i, p in enumerate(prediction.parts()):
            np_pts[i, :] = (p.x, p.y)
        return np_pts

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
        self.filterLandmarks = filter.ButterFilterArray(n=136, n_filter=1, cutoff=0.2)

    def reset(self):
        self.good = False

    def track_face(self, opencv_img):
        if not self.good:
            self.filterLandmarks.reset()
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

        # landmarks = self.get_ndarray(prediction) * scale + shift
        # mean = np.mean(landmarks, axis=0)
        # return np.reshape(
        #     self.filterLandmarks.append_and_filter(
        #         np.reshape(landmarks - mean, (1, -1))),
        #     (-1, 2)) + mean
        return self.get_ndarray(prediction) * scale + shift
