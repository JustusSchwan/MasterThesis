from __future__ import print_function

from functools import partial

import numpy as np

from dataflow import ports
from dataflow.ports import EventSource, StateSink


class Storage:
    def __init__(self):
        self.val = None

    def __call__(self, val):
        self.val = val
        return self.val


class ArgumentZip:
    def __init__(self, nargs):
        self.args = []
        self.complete = np.ones(nargs, dtype=np.bool)
        self.complete[:] = False
        self.source = ports.EventSource()
        for i in range(nargs):
            self.args.append(None)

    def put_arg(self, i, arg):
        self.args[i] = arg
        self.complete[i] = True
        if np.all(self.complete):
            self.complete[:] = False
            self.source.fire(*tuple(self.args))

    def in_arg(self, i):
        return partial(self.put_arg, i)


class StateArgumentZip:
    def __init__(self, *args):
        self.poller = ports.StateSink()
        self.sinks = []
        for arg in args:
            self.sinks.append(StateSink())
            self.sinks[-1] << arg

    def arg(self, n_arg=0):
        assert len(self.sinks) == n_arg, 'You must specify the arguments in order'
        while len(self.sinks) <= n_arg:
            self.sinks.append(StateSink())
        return self.sinks[n_arg]

    # noinspection PyStatementEffect
    def __lshift__(self, other):
        return self.poller << other

    def __call__(self):
        if len(self.sinks) == 0:
            return self.poller.get()
        args = []
        for sink in self.sinks:
            arg = sink.get()
            if arg is None:
                return
            args.append(arg)
        return self.poller.get(*args)


class StateArgumentUnpack:
    def __init__(self, arg_source=None):
        self.to_call = ports.StateSink()
        self.arg_sink = ports.StateSink()
        if arg_source is not None:
            self.arg_sink << arg_source

    def args(self):
        return self.arg_sink

    # noinspection PyStatementEffect
    def __lshift__(self, other):
        return self.to_call << other

    def __call__(self):
        args = self.arg_sink.get()
        if args is None:
            return self.to_call.get(None)
        return self.to_call.get(*args)


class TupleTee:
    def __init__(self, fun):
        self.fun = fun

    def __call__(self, *args):
        self.fun(*args)
        return args


class Tee:
    def __init__(self, fun):
        self.fun = fun

    def __call__(self, arg):
        self.fun(arg)
        return arg


class Counter:
    def __init__(self):
        self.c = 0
        self.out_event = ports.EventSource()

    def advance(self):
        self.out_event.fire(self.c)
        self.c += 1

    def __call__(self, *args):
        return self.c

    def reset(self):
        self.c = 0


class EventEdge:
    def __init__(self):
        self.buf = None
        self.source = ports.EventSource()

    def __call__(self, *args):
        send = False
        if self.buf != args:
            send = True
            self.buf = args
        if send:
            self.source.fire(*args)


class StateToEvent:
    def __init__(self, *args):
        self.sinks = []
        self.source = EventSource()
        for state_source in args:
            self.sinks.append(StateSink())
            self.sinks[-1] << state_source

    def arg(self, n_arg=0):
        while len(self.sinks) <= n_arg:
            self.sinks.append(StateSink())
        return self.sinks[n_arg]

    def get_and_fire(self):
        args = []
        for sink in self.sinks:
            arg = sink.get()
            if arg is None:
                return
            args.append(arg)
        self.source.fire(*args)


class EventToState:
    def __init__(self):
        self.buf = None

    def in_event(self, arg):
        self.buf = arg

    def out_state(self):
        return self.buf


class EventToStateBuffer:
    def __init__(self):
        self.val = None
        self.buffer = ports.StateBuffer()
        self.buffer.sink << (lambda: self.val)

    def __call__(self):
        return self.buffer()

    def in_event(self, arg):
        self.val = arg

    def reset(self):
        self.buffer.reset()
        self.val = None


class NoneIfNone:
    def __init__(self, fun):
        self.fun = fun

    def __call__(self, *args):
        if any(arg is None for arg in args):
            return None
        return self.fun(*args)


class IterableStateSource:
    def __init__(self, iterable):
        self.iterator = iter(iterable)

    def __call__(self):
        try:
            return self.iterator.next()
        except StopIteration:
            return None


class StateGate:
    def __init__(self):
        self.sink_control = ports.StateSink()
        self.sink = ports.StateSink()

    def __call__(self, *args):
        if self.sink_control.get():
            return self.sink.get(*args)
        else:
            return None


class EventGate:
    def __init__(self):
        self.sink_control = ports.StateSink()
        self.source = ports.EventSource()

    def sink(self, *args):
        if self.sink_control.get():
            self.source.fire(*args)


if __name__ == '__main__':
    es_buf = EventToState()
    so = ports.EventSource()
    si = ports.StateSink()
    so >> es_buf.in_event
    si << es_buf.out_state

    so.fire(5)
    print(si.get())
