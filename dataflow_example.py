from __future__ import print_function
from dataflow import ports


def add(x):  # Returns a callable object
    return lambda y: x + y


# Event example code
e_source = ports.EventSource()
e_terminal = ports.EventTerminal()

# 1 shall be added to the value fired from the source, then forwarded to the terminal
e_source >> add(1) >> e_terminal

# 4 shall be added to the value from the terminal, then printed
e_terminal >> add(4) >> print

# the value from the terminal shall be printed
e_terminal >> print

e_source.fire(5)  # fires 5, thereby printing 10 and 6.


# State example code
def s_source():  # Callable object
    return 5


s_sink_1 = ports.StateSink()
s_sink_2 = ports.StateSink()
s_terminal = ports.StateTerminal()

# The terminal will add 1 to the value pulled from the source
s_terminal << add(1) << s_source

# The sink will add 4 to the value pulled from the terminal
s_sink_1 << add(4) << s_terminal

# The sink will only pull the terminal
s_sink_2 << s_terminal

# Pulls the value from the sink, the chain goes through to the source, and print it
print(s_sink_1.get())
print(s_sink_2.get())

# Multi-Argument/Buffering example
from dataflow import connectables

num_invocations = 0


# "Expensive" function, the number of invocations is counted and returned
def inefficient_source():
    global num_invocations
    num_invocations += 1
    return num_invocations


s_buf = ports.StateBuffer()

# Buffer pulls the function once and holds the value
s_buf << inefficient_source

# Call buffer twice, the number of invocations does not increase
print('Number of invocations:', s_buf())
print('Number of invocations:', s_buf())


def function_with_two_args(x, y):
    return x * y


# Merges return values of s_source and s_buf into an argument list
# Alternative construction: function_args = connectables.StateArgumentZip(s_source, s_buf)
function_args = connectables.StateArgumentZip()
function_args.arg(0) << s_source
function_args.arg(1) << s_buf

# Tells the function_args object which function to call with the argument list
function_args << function_with_two_args

s_sink_multiargs = ports.StateSink()

s_sink_multiargs << function_args
print(s_sink_multiargs.get())
print('Number of invocations:', num_invocations)

# Reset buffer, invocations will now increase
s_buf.reset()
print(s_sink_multiargs.get())
print('Number of invocations:', num_invocations)

