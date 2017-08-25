import cv2
from dataflow import ports


class BaseVideoSource:
    def __init__(self):
        self.cap = None

    def source_image(self):
        if self.cap is None:
            return None
        ret, img = self.cap.read()
        if ret:
            return img
        return None


class CameraSource(BaseVideoSource):
    def __init__(self, camera=0):
        BaseVideoSource.__init__(self)
        self.cap = cv2.VideoCapture(camera)
        self.cap.read()
        self.cap.read()
        self.cap.read()
        self.cap.read()
        self.cap.read()
        if not self.cap.isOpened():
            print('WARNING: Camera source not open')


class FileSource(BaseVideoSource):
    def __init__(self):
        BaseVideoSource.__init__(self)
        self.sink_filename = ports.StateSink()
        self.source_eof = ports.EventSource()
        self.source_file_change = ports.EventSource()
        self.cap = None

    def skip_to_frame(self, frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)

    def skip(self):
        self.cap = None

    def source_current_frame(self):
        if self.cap is None:
            return None
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES)

    def source_total_frames(self):
        if self.cap is None:
            return None
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def source_image(self):
        img = BaseVideoSource.source_image(self)
        while img is None:
            self.source_eof.fire()
            path = self.sink_filename.get()
            if path is None:
                return None
            self.cap = cv2.VideoCapture(path)
            self.source_file_change.fire(path)
            img = BaseVideoSource.source_image(self)
        return img


class FrameSource(BaseVideoSource):
    def __init__(self):
        BaseVideoSource.__init__(self)
        self.filename = None
        self.cap = None

    def source_frame(self, filename, frame):
        if self.filename != filename:
            self.cap = cv2.VideoCapture(filename)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        img = self.source_image()
        if img is not None:
            self.filename = filename
            return img
        else:
            self.filename = None
