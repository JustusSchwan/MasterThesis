import dlib
from dataflow import ports


def dlib_rect_to_array(rect):
    return [rect.left(), rect.top(), rect.right(), rect.bottom()]


class Tracker:
    def __init__(self, threshold=10, replenish_interval=-1):
        self.tracker = dlib.correlation_tracker()
        self.sink_bounds = ports.StateSink()
        self.sink_image = ports.StateSink()
        self.bounds = None
        self.threshold = threshold
        self.interval = replenish_interval
        self.i_interval = 0

    def get_new_bounds(self, img):
        self.bounds = self.sink_bounds.get()
        if self.bounds is not None:
            self.tracker.start_track(img, self.bounds)
            return self.bounds

    def track(self):
        img = self.sink_image.get()
        if img is None:
            return None
        if self.bounds is None:
            return self.get_new_bounds(img)

        if self.interval >= 0:
            self.i_interval += 1
            if self.i_interval > self.interval:
                self.i_interval = 0
                return self.get_new_bounds(img)

        score = self.tracker.update(img)
        if score < self.threshold:
            return self.get_new_bounds(img)

        return self.tracker.get_position()

    def reset_state(self):
        self.bounds = None
        self.i_interval = 0


class HypothesisTracker:
    def __init__(self, threshold=10):
        self.tracker = dlib.correlation_tracker()
        self.sink_bounds = ports.StateSink()
        self.sink_hypotheses = ports.StateSink()
        self.sink_image = ports.StateSink()
        self.bounds = None
        self.threshold = threshold

    def get_new_bounds(self, img):
        self.bounds = self.sink_bounds.get()
        if self.bounds is not None:
            self.tracker.start_track(img, self.bounds)
            return self.bounds

    def track(self):
        img = self.sink_image.get()
        if img is None:
            return None
        # try to get something to track
        if self.bounds is None:
            return self.get_new_bounds(img)

        score = self.tracker.update(img)
        if score < self.threshold:
            return self.get_new_bounds(img)

        # Check plausibility
        d = self.tracker.get_position()
        if self.sink_hypotheses.get(d):
            return d
        else:
            return self.get_new_bounds(img)
