from dataflow import ports
import cv2
import numpy as np
from scipy.stats import multivariate_normal
from itertools import product


class SkinColorEstimator:
    def __init__(self):
        self.roi = np.zeros(0, dtype=np.uint8)
        self.skip = 0
        self.mean = None
        self.cov = None
        self.lut = None
        self.perm = np.array(list(product(range(256), repeat=2)), dtype=np.uint8)
        self.lut = np.zeros((256, 256))
        self.sink_image = ports.StateSink()
        self.sink_landmarks = ports.StateSink()
        self.mx = None
        self.mn = None

    def get_skin_color_mask(self):
        img = self.sink_image.get()
        landmarks = self.sink_landmarks.get()
        if img is None or landmarks is None:
            return None
        img_chroma =cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:, :, 1:]
        if self.skip <= 0:
            if self.roi.shape != img.shape:
                self.roi = np.zeros(img.shape[:2], dtype=np.uint8)
            self.roi.fill(0)
            hull = cv2.convexHull(np.array(
                landmarks[(31, 39, 42, 35), :],
                dtype=np.int32))
            cv2.fillConvexPoly(self.roi, points=hull, color=1)
            selection = cv2.GaussianBlur(img_chroma, ksize=(9, 9), sigmaX=3, sigmaY=3)[self.roi > 0, ...]
            iir = 0.9
            if self.mean is None:
                self.mean = np.mean(selection, axis=0)
                self.cov = np.cov(selection, rowvar=False)
            else:
                self.mean = np.mean(selection, axis=0) * (1 - iir) + self.mean * iir
                self.cov = np.cov(selection, rowvar=False) * (1 - iir) + self.cov * iir
            self.lut.flat = multivariate_normal.pdf(self.perm, mean=self.mean, cov=self.cov)
            self.mx = np.max(self.lut)
            self.mn = np.min(self.lut)
            self.skip = 5
        self.skip -= 1

        img_chroma = np.reshape(
            img_chroma, (-1, 2))

        probs = np.reshape(self.lut[img_chroma[:, 0], img_chroma[:, 1]], img.shape[:2])

        return np.clip(np.array((probs - self.mn) / (self.mx - self.mn) * 255, dtype=np.uint8), a_min=0, a_max=255)
