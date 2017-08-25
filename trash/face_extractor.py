import cv2
import numpy as np

import dlib_landmarks


def avg_range(r):
    o = [0, 0]
    sz = 0
    for p in r:
        o[0] = o[0] + p[0]
        o[1] = o[1] + p[1]
        sz += 1
    o[0] = o[0]/sz
    o[1] = o[1]/sz
    return o


class LandmarkExtractor:
    """
    crops faces from videos
    """
    
    def __init__(self, tt=10):
        self.tracker = dlib_landmarks.dlibFeatureExtractor(tracking_threshold=tt)

        # target positions eye and mouth
        tel = (0.38, 0.30)
        ter = (0.62, 0.30)
        tm  = (0.50, 0.60)
        
        self.target_points = np.float32([tel, ter, tm])

    def reset(self):
        self.tracker.reset_state()

    def track_landmarks(self, img):
        track = self.tracker.track_face(img)

        if track is None:
            return None
        return track

    def warp_landmarks(self, pts):
        np_pts = np.concatenate((np.transpose(pts), np.ones([1, 68], dtype=float)), axis=0)

        el = avg_range(self.tracker.iter_eye_l(pts))
        er = avg_range(self.tracker.iter_eye_r(pts))
        m = avg_range(self.tracker.iter_mouth(pts))

        mat = cv2.getAffineTransform(
            np.float32([el, er, m]),
            self.target_points)

        return np.transpose(np.dot(mat, np_pts))
