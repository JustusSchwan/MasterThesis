import cv2
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from util.np_buffers import NpCircularArray
from dataflow import ports
import matplotlib.patches as mpatches


class ImageViewer:
    def __init__(self, name=''):
        self.img = None
        self.name = name
        cv2.startWindowThread()
        cv2.namedWindow(self.name)
        self.sink_size = ports.StateSink()
        self.sink_image = ports.StateSink()
        self.stale = True

    def try_init_image(self):
        if not self.stale:
            return
        size = self.sink_size.get()
        if size is None:
            return
        if self.img is None or self.img.shape != size:
            self.img = np.zeros(self.sink_size.get(), dtype=np.uint8)
        else:
            self.img.flat[:] = 0
        self.stale = False

    def try_get_image(self):
        if not self.stale:
            return
        img = self.sink_image.get()
        if img is None:
            return
        if self.img is None or self.img.shape[0:2] != img.shape[0:2]:
            self.img = img.copy()
        else:
            if len(img.shape) < 3 or img.shape[2] == 1:
                cv2.cvtColor(img, code=cv2.COLOR_GRAY2BGR, dst=self.img)
            else:
                self.img[:] = img[:]
        self.stale = False

    def draw_rectangle(self, pts, color):
        if self.stale:
            self.try_get_image()
            self.try_init_image()
        if self.stale:
            return
        cv2.rectangle(self.img, (int(pts[0]), int(pts[1])), (int(pts[2]), int(pts[3])), color=color)

    def draw_circle(self, c, r, color):
        if self.stale:
            self.try_get_image()
            self.try_init_image()
        if self.stale:
            return
        if len(c.shape) == 1:
            cv2.circle(self.img, center=(int(c[0]), int(c[1])), radius=r, color=color)
        else:
            for i in range(c.shape[0]):
                cv2.circle(self.img, center=(int(c[i, 0]), int(c[i, 1])), radius=r, color=color)

    def draw_polyline(self, pts, color, closed):
        if self.stale:
            self.try_get_image()
            self.try_init_image()
        if self.stale:
            return
        cv2.polylines(self.img, [np.array(pts, np.int32).reshape((-1, 1, 2))], isClosed=closed, color=color,
                      thickness=1)

    def in_rectangle(self, color):
        return partial(self.draw_rectangle, color=color)

    def in_polyline(self, color, closed):
        return partial(self.draw_polyline, color=color, closed=closed)

    def in_circle(self, color, r):
        return partial(self.draw_circle, color=color, r=r)

    def show(self, duration=1):
        if self.stale:
            self.try_get_image()
            self.try_init_image()
        if not self.stale:
            cv2.imshow(self.name, self.img)
        key = cv2.waitKey(duration)
        self.stale = True
        return key


class TimeSeriesViewer:
    def __init__(self, capacity=200):
        assert capacity > 2
        self.series_buffer = []
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        self.capacity = capacity
        self.sink_time = ports.StateSink()
        self.lines = []
        plt.ion()
        plt.pause(0.05)

    def draw(self):
        for line, buf in zip(self.lines, self.series_buffer):
            if len(buf) < 2:
                continue
            data = buf.get_cropped()
            assert data.shape[0] == len(buf)
            line.set_xdata(data[:, 0])
            line.set_ydata(data[:, 1])
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.05)

    def append_data(self, buf, elem):
        buf.append([self.sink_time.get(), elem])

    def in_series(self, label, **kwargs):
        line = plt.Line2D(xdata=[], ydata=[], label=label, **kwargs)
        self.lines.append(line)
        self.ax.add_line(line)
        self.ax.legend(handles=self.lines)

        buf = NpCircularArray(n=2, length=self.capacity)
        self.series_buffer.append(buf)
        return partial(self.append_data, buf)

    def clear(self):
        for buf in self.series_buffer:
            buf.clear()
