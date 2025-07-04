#!/usr/bin/env python
#
# Copyright 2019 DFKI GmbH.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
# NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.
LOG_MODE_ERROR = -1
LOG_MODE_INFO = 1
LOG_MODE_DEBUG = 2

_lines = []
_active = True
_mode = LOG_MODE_INFO

def activate():
    global _active
    _active = True


def deactivate():
    global _active
    _active = False

def set_log_mode(mode):
    global _mode
    _mode = mode

def write_log(*args):
    global _active
    global _lines
    if _active:
        line = " ".join(map(str, args))
        print(line)
        _lines.append(line)


def write_message_to_log(message, mode=LOG_MODE_INFO):
    global _active
    global _lines
    if _active and _mode >= mode:
        print(message)
        _lines.append(message)


def save_log(filename):
    global _lines
    with open(filename, "wb") as outfile:
        for l in _lines:
            outfile.write(l+"\n")


def clear_log():
    global _lines
    _lines = []
