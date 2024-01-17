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
import numpy as np
from ..motion_vector import MotionVector


def smooth(x, window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def low_pass(src, alpha = 0.5):
    """ http://blog.thomnichols.org/2011/08/smoothing-sensor-data-with-a-low-pass-filter
        https://en.wikipedia.org/wiki/Low-pass_filter
    """

    n_frames = len(src.frames)
    n_dims = len(src.frames[0])
    new_frames = [src.frames[0]]
    for i in range(1, n_frames):
        new_f = []
        for j in range(n_dims):
            v = new_frames[i-1][j] + alpha * (src.frames[i][j] - new_frames[i-1][j])
            print("filter", v, src.frames[i][j])
            new_f.append(v)
        new_frames.append(new_f)

    target_motion = MotionVector()
    target_motion.skeleton = src.skeleton
    target_motion.frames = np.array(new_frames)
    target_motion.n_frames = len(new_frames)
    return target_motion


def moving_average(src, window=4):
    """ https://www.wavemetrics.com/products/igorpro/dataanalysis/signalprocessing/smoothing.htm#MovingAverage
    """
    n_frames = len(src.frames)
    n_dims = len(src.frames[0])
    new_frames = np.zeros(src.frames.shape)
    new_frames[0,:] = src.frames[0,:]
    hw = int(window/2)
    for i in range(1, n_frames):
        for j in range(n_dims):
            start = max(0, i-hw)
            end = min(n_frames-1, i+hw)
            w = end-start
            #print(i, j, start, end, w, n_frames, n_dims)
            #print("filter", v, src.frames[i, j])
            new_frames[i, j] = np.sum(src.frames[start:end, j])/w
    target_motion = MotionVector()
    target_motion.skeleton = src.skeleton
    target_motion.frames = np.array(new_frames)
    print("new shape", target_motion.frames.shape)
    target_motion.n_frames = len(new_frames)
    return target_motion