import multiprocessing as mpc
import matplotlib.pyplot as plt
import imutils
import numpy as np

from matplotlib.style import use

from scipy.signal import convolve
# from line_reader import FrameFoi

import scipy
import time
import glob
import sys
import cv2
import os

from picture_paths import CABIN_PICS_1, CABIN_PICS_2, ALL_HD
from multiprocoess_functions import multi_picture_export, time_formatter


use('ggplot')

CURR_SPEEC_LIMIT = slice(712, 728, None), slice(1500, 1521, None)
ROAD_SIGN_LIMIT = slice(784, 800, None), slice(1500, 1522, None)

CURR_SPEED_720 = slice(473, 485, None), slice(1000, 1018, None)


def clip_curr_speed(p):
    roi = p[CURR_SPEEC_LIMIT]
    return roi


def clip_sign(p):
    roi = p[ROAD_SIGN_LIMIT]
    return roi


def hist_over_x(pic):
    b = np.sum(pic[:, :, 0] / 255, axis=0)
    g = np.sum(pic[:, :, 1] / 255, axis=0)
    r = np.sum(pic[:, :, 2] / 255, axis=0)
    plt.plot(b, c='b', label="b")
    plt.plot(g, c='g', label='g')
    plt.plot(r, c='r', label='r')


sub = "speed_numbers_1"
# multi_picture_export(CABIN_PICS_2, subfolder=sub, clip_f=clip_curr_speed)
# multi_picture_export(CABIN_PICS_2, subfolder=sub, postfix='hst',
#                      matplot_f=hist_over_x, clip_f=clip_curr_speed,
#                      )

sub = "speed_numbers_2"

if __name__ == "__main__":
    t0 = time.time()
    POOL = mpc.Pool(10)

    multi_picture_export(ALL_HD, subfolder=sub, clip_f=clip_curr_speed, pool=POOL)
    multi_picture_export(
            ALL_HD, subfolder=sub, postfix='hist', clip_f=clip_sign, matplot_f=hist_over_x,
            pool=POOL)

    tend = time.time()

    print("Whole script took ", time_formatter(tend - t0))

    del POOL
