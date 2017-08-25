import dlib
import cv2
from os import path
from dataflow import ports


class ObjectDetectorOpencv:
    def __init__(self, model):
        self.detector = cv2.CascadeClassifier(model)
        self.source_detection = ports.EventSource()

    def detect_object(self, img):
        detect = self.detector.detectMultiScale(img, 1.2, 7, 0, (50, 50))
        if len(detect) > 0:
            (x, y, w, h) = detect[0]
            self.source_detection.fire(dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))


class MultiObjectDetectorOpencv:
    def __init__(self, models):
        self.detectors = []
        for model in models:
            self.detectors.append(cv2.CascadeClassifier(model))
        self.source_detection = ports.EventSource()

    def detect_object(self, img):
        for detector in self.detectors:
            detect = detector.detectMultiScale(img, 1.1, 7, 0, (50, 50))
            if len(detect) > 0:
                (x, y, w, h) = detect[0]
                self.source_detection.fire(dlib.rectangle(int(x), int(y), int(x + w), int(y + h)))
                break


def FaceDetectorOpencv():
    return ObjectDetectorOpencv(
        path.join(path.dirname(path.abspath(__file__)), 'models/haarcascade_frontalface_alt2.xml'))


def HandDetectorOpencv():
    return MultiObjectDetectorOpencv(
        (
            path.join(path.dirname(path.abspath(__file__)), 'models/palm.xml'),
            path.join(path.dirname(path.abspath(__file__)), 'models/fist.xml'),
            path.join(path.dirname(path.abspath(__file__)), 'models/closed_frontal_palm.xml'),
            path.join(path.dirname(path.abspath(__file__)), 'models/aGest.xml')
        ))

