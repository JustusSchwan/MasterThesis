from __future__ import print_function
import ports
import connectables
import cPickle
from functools import partial
from tempfile import gettempdir


class LoggingMode:
    def __init__(self):
        pass

    RECORD = 0
    REPLAY = 1
    LIVE = 2


class LoggingEventTerminal:
    def __init__(self, mode):
        self.mode = mode
        self.source_live = ports.EventSource()
        self.source_to_file = ports.EventSource()

    def sink_live(self, *args):
        if self.mode == LoggingMode.RECORD:
            self.source_live.fire(*args)
            self.source_to_file.fire(*args)
        elif self.mode == LoggingMode.LIVE:
            self.source_live.fire(*args)

    def sink_from_file(self, *args):
        if self.mode == LoggingMode.REPLAY:
            self.source_live.fire(*args)

    def set_mode(self, mode):
        self.mode = mode


class LoggingStateBuffer:
    def __init__(self, mode):
        self.mode = mode
        self.sink_live = ports.StateSink()
        self.source_to_file = ports.EventSource()
        self.dirty = True
        self.buf = None

    def source_live(self, *args):
        assert len(args) == 0, 'State Logging only supports argument-less calls'
        if (self.mode == LoggingMode.RECORD or self.mode == LoggingMode.LIVE) and self.dirty:
            self.buf = self.sink_live.get(*args)
            if self.mode == LoggingMode.RECORD:
                self.source_to_file.fire(self.buf)
            self.dirty = False
        return self.buf

    def sink_from_file(self, arg):
        if self.mode == LoggingMode.REPLAY:
            self.buf = arg

    def reset(self):
        self.buf = None
        self.dirty = True

    def set_mode(self, mode):
        assert self.dirty, 'Mode can only be set when buffer is reset'
        self.mode = mode


class FileHandler:
    def __init__(self, mode):
        self.mode = mode
        self.f = None
        self.opened = False
        self.events = {}
        self.blob = []

    def open(self, name):
        self.close()
        rwmode = 'wb' if self.mode == LoggingMode.RECORD else 'rb'
        try:
            self.f = open(name, rwmode)
            self.opened = True
        except IOError:
            print('The File with name {} cannot be openend for {}'.format(
                name, 'writing' if self.mode == LoggingMode.RECORD else 'reading'))
        return self.opened

    def _in_event(self, name, *args):
        if self.opened and self.mode == LoggingMode.RECORD:
            self.blob.append([name, args])

    def in_event(self, name):
        return partial(self._in_event, name)

    def out_event(self, name):
        if name not in self.events:
            self.events[name] = ports.EventSource()
        return self.events[name]

    def close(self):
        if self.opened:
            self.f.close()
            self.f = None
            self.opened = False

    def read_blob(self):
        if self.opened and self.mode == LoggingMode.REPLAY:
            try:
                blob = cPickle.load(self.f)
                for datum in blob:
                    self.out_event(datum[0]).fire(*datum[1])
            except EOFError:
                return False
        return self.opened

    def write_blob(self):
        if self.opened and self.mode == LoggingMode.RECORD:
            try:
                cPickle.dump(self.blob, self.f, protocol=-1)
                self.blob = []
            except IOError:
                print('Could not dump to file')

    def set_mode(self, mode):
        assert not self.opened, 'Mode can only be set when no file is open at the moment'
        self.mode = mode


class Logger:
    def __init__(self, mode, always_poll=False):
        assert mode == LoggingMode.RECORD or mode == LoggingMode.REPLAY or mode == LoggingMode.LIVE
        self.events = {}
        self.states = {}
        self.file_handler = FileHandler(mode)
        self.state_poller = ports.TriggerHandler()
        self.resetter = ports.TriggerHandler()
        self.always_poll = always_poll
        self.mode = mode

    def _setup_event(self, name):
        assert name not in self.events, "An event channel with the id {} was already specified. " \
                                        "This should not happen".format(name)
        handler = LoggingEventTerminal(self.mode)
        self.events[name] = handler
        handler.source_to_file >> self.file_handler.in_event(name)
        self.file_handler.out_event(name) >> handler.sink_from_file

    def sink_event(self, name):
        if name not in self.events:
            self._setup_event(name)
        return self.events[name].sink_live

    def source_event(self, name):
        if name not in self.events:
            self._setup_event(name)
        return self.events[name].source_live

    def _setup_state(self, name):
        assert name not in self.states, "A state channel with the id {} was already specified " \
                                        "This should not happen".format(name)
        handler = LoggingStateBuffer(self.mode)
        self.states[name] = handler
        self.state_poller.attach(handler.source_live)
        self.resetter.attach(handler.reset)
        handler.source_to_file >> self.file_handler.in_event(name)
        self.file_handler.out_event(name) >> handler.sink_from_file

    def sink_state(self, name):
        if name not in self.states:
            self._setup_state(name)
        return self.states[name].sink_live

    def source_state(self, name):
        if name not in self.states:
            self._setup_state(name)
        return self.states[name].source_live

    def open(self, name):
        return self.file_handler.open(name)

    def close(self):
        self.file_handler.close()

    def read(self):
        return self.file_handler.read_blob()

    def write(self):
        if self.always_poll:
            self.state_poller.trigger()
        self.file_handler.write_blob()
        self.resetter.trigger()

    def set_mode(self, mode):
        assert mode == LoggingMode.RECORD or mode == LoggingMode.REPLAY
        self.mode = mode
        for op in self.events.viewvalues():
            op.set_mode(mode)
        for op in self.states.viewvalues():
            op.set_mode(mode)
        self.file_handler.set_mode(mode)


if __name__ == '__main__':
    log = Logger(LoggingMode.RECORD)
    source = ports.EventSource()
    source >> log.sink_event('int_event')
    log.source_event('int_event') >> (lambda x: print('firing {}'.format(x)))
    log.source_event('int_event') >> print
    log.source_event('int_event') >> (lambda x: x + 5) >> print

    sink = ports.StateSink()
    log.sink_state('int_state') << connectables.Tee(lambda x: print('polling {}'.format(x))) << (lambda: 1)
    sink << log.source_state('int_state')

    log.open(gettempdir() + 'logger_test.pkl')

    source.fire(5)
    log.write()

    source.fire(6)
    print('polling sink')
    print(sink.get())

    log.write()

    print('switch to read')
    log.close()
    log.set_mode(LoggingMode.REPLAY)
    log.open('D:/tmp.pkl')

    log.read()
    print('polling sink')
    print(sink.get())
    log.read()
    print('polling sink')
    print(sink.get())
    assert not log.read()
