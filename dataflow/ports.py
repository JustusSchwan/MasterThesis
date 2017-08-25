from __future__ import print_function


class EventConnection:
    def __init__(self, task):
        assert callable(task), 'Connected objects need to be callable'
        self.task = task
        self.receiver = None

    def __call__(self, *args):
        if self.receiver is None:
            self.task(*args)
        else:
            self.receiver(self.task(*args))

    def connect(self, task):
        self.receiver = EventConnection(task)
        return self.receiver

    # noinspection PyStatementEffect
    def __rshift__(self, other):
        return self.connect(other)


class EventSource:
    def __init__(self):
        self.receivers = []

    def fire(self, *args):
        for r in self.receivers:
            r(*args)

    def connect(self, task):
        self.receivers.append(EventConnection(task))
        return self.receivers[-1]

    # noinspection PyStatementEffect
    def __rshift__(self, other):
        return self.connect(other)


class EventTerminal:
    def __init__(self):
        self.so = EventSource()

    def __call__(self, *args):
        self.so.fire(*args)

    # noinspection PyStatementEffect
    def __rshift__(self, other):
        return self.so >> other


class StateConnection:
    def __init__(self, callback):
        assert callable(callback), 'Connected objects need to be callable'
        self.callback = callback
        self.arg_provider = None

    def __call__(self, *args):
        if self.arg_provider is None:
            return self.callback(*args)
        else:
            return self.callback(self.arg_provider(*args))

    def connect(self, callback):
        self.arg_provider = StateConnection(callback)
        return self.arg_provider

    # noinspection PyStatementEffect
    def __lshift__(self, other):
        return self.connect(other)


class StateSink:
    def __init__(self, default=None):
        self.callback = lambda *args: default

    def get(self, *args):
        return self.callback(*args)

    def connect(self, callback):
        self.callback = StateConnection(callback)
        return self.callback

    # noinspection PyStatementEffect
    def __lshift__(self, other):
        return self.connect(other)


class StateTerminal:
    def __init__(self):
        self.sink = StateSink()

    def __call__(self, *args):
        return self.sink.get(*args)

    # noinspection PyStatementEffect
    def __lshift__(self, other):
        return self.sink << other


class StateBuffer:
    def __init__(self):
        self.sink = StateSink()
        self.dirty = True
        self.buf = None

    def __call__(self, *args):
        if self.dirty:
            self.buf = self.sink.get(*args)
            self.dirty = False
        return self.buf

    # noinspection PyStatementEffect
    def __lshift__(self, other):
        return self.sink << other

    def reset(self):
        self.dirty = True


class TriggerHandler:
    def __init__(self):
        self.events = []

    def attach(self, event):
        assert callable(event), 'The attached event must be callable'
        self.events.append(event)

    def trigger(self, *args):
        for event in self.events:
            event()


if __name__ == '__main__':
    def yield_range(x):
        for i in range(x):
            yield(i)

    def multiply_lazy(gen):
        for x in gen:
            yield(x*5)

    def evaluate(gen):
        return list(gen)

    def print_x(x):
        print(x)

    so = EventSource()

    so >> (lambda x: x*5) >> print
    so.fire(5)


    # so = EventSource()
    # t = EventTerminal()
    #
    # so >> (lambda x: x + 5) >> t
    # t >> print_x
    # t >> yield_range >> multiply_lazy >> evaluate >> print_x
    #
    # so.fire(0)
    #
    # state_buf = StateBuffer()
    # state_buf.sink << (lambda x: x*3) << (lambda x: x+5) << (lambda x, y: x + y)
    #
    #
    #
    # st_si = StateSink()
    # st_si << state_buf
    #
    # print st_si.get(0, 0)
    # print st_si.get(1, 1)
    # state_buf.reset()
    # print st_si.get(1, 1)

