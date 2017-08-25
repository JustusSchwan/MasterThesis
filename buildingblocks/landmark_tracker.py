from __future__ import print_function
from dataflow import ports
from operators import face_detector
from operators import dlib_tracker
from dataflow import connectables


class LandmarkTrackerBB:
    def __init__(self, image_getter, logger):
        # Face bounds detection
        self.face_detector = face_detector.FaceDetector()
        self.face_detector.sink_image << image_getter

        # Face tracking
        self.face_tracker = dlib_tracker.Tracker(threshold=15, replenish_interval=20)
        self.face_tracker.sink_bounds << self.face_detector.out_bounds
        self.face_tracker.sink_image << image_getter

        logger.sink_state('face_bounds') << self.face_tracker.track

        # Facial landmark detection
        self.landmark_detector = face_detector.LandmarkDetector()
        self.landmark_detector.sink_bounds << logger.source_state('face_bounds')
        self.landmark_detector.sink_image << image_getter

        logger.sink_state('landmarks') << self.landmark_detector.detect_landmarks

    def reset_state(self):
        self.face_tracker.reset_state()
