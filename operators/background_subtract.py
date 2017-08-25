from dataflow import ports
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


class BackgroundMask:
    def __init__(self):
        self.img = None
        self.source_mask = ports.EventSource()

    def in_image(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if self.img is None:
            self.img = img_hsv
            return
        self.source_mask.fire(cv2.threshold(np.asarray(np.linalg.norm((self.img - img_hsv), axis=2), dtype=np.uint8), thresh=127, maxval=255, type=cv2.THRESH_BINARY)[1])
        self.img = self.img * 0.99 + img_hsv * 0.01
