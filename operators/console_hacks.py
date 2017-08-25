import sys


class ProgressBar:
    def __init__(self):
        self.progress_x = 0
        self.started = False

    def start(self, title):
        self.started = True
        sys.stdout.write(title + ": [" + "-" * 40 + "]" + chr(8) * 41)
        sys.stdout.flush()
        self.progress_x = 0

    def progress(self, x):
        if not self.started:
            return
        x = int(x * 40)
        sys.stdout.write("#" * (x - self.progress_x))
        sys.stdout.flush()
        self.progress_x = x

    def end(self):
        if not self.started:
            return
        sys.stdout.write("#" * (40 - self.progress_x) + "]\n")
        sys.stdout.flush()
        self.started = False
