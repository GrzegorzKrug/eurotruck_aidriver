import matplotlib.pyplot as plt
import imutils
import numpy as np
from numba import jit

from scipy.signal import convolve, convolve2d
# from frame_foi import FrameFoi
from sklearn.cluster import MeanShift, KMeans
import multiprocessing as mpc

import scipy
import time
import glob
import sys
import cv2
import os

from picture_paths import (
    COLORS_PATHS, PICS_AUTOSTRADA,
    PICS_FRANKFURT, PICS_FRANKFURT_TRAFFIC,
    PICS_LINES,
)

from multiprocoess_functions import *
from frame_foi import (
    foi_roadsample, foi_window_view, foi_roadsample_low,
    foi_frontvision, foi_frontsample, foi_map, foi_road_model,
    foi_mirror_left, foi_mirror_right,
)


def pic_gray_to3d(out):
    out = out[:, :, np.newaxis]
    out = np.tile(out, [1, 1, 3])
    return out


def lines_IOU(l1, l2):
    # print(l1)
    x1, y1, x2, y2 = l1
    h = y2 - y1
    k = x2 - x1
    angle = np.arctan2(h, k)
    degrees = (angle * 360 / np.pi) % 180
    print(f"{degrees:>4.2f}")
    return


# @jit()
def lines_suppression(lines, intersect_lim=3):
    intersect_counter = np.zeros(len(lines), dtype=np.int32)

    sz = len(lines) - 1
    for i in range(sz):
        for j in range(i + 1, sz):
            pass
            l1 = lines[i][0]
            l2 = lines[j][0]
            lines_IOU(l1, l2)

    return lines


def hough_parabolas(pic, **kw):
    pass


HORIZ_KERNEL = np.array([[0, 1, 2]]) - 1
HORIZ_KERNEL = np.tile(HORIZ_KERNEL, [3, 1])


def convolve_vertical(pic, **kw):
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    _, bin_image = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    bin_image = convolve2d(bin_image, HORIZ_KERNEL, 'same')
    print(bin_image.shape, bin_image.dtype)
    bin_image = bin_image.round().astype(np.uint8)

    gray_3d = pic_gray_to3d(bin_image)
    return gray_3d


def find_hough_lines(orig_pic, name=None, **kw):
    """Interesting shapes must be white!"""
    # orig_pic = convolve_vertical(orig_pic)
    # return orig_pic

    pic = cv2.cvtColor(orig_pic, cv2.COLOR_BGR2GRAY)
    _, bin_pic = cv2.threshold(pic, 30, 255, cv2.THRESH_BINARY)
    # bin_pic = 255 - bin_pic  # not using convolve

    # bin_pic = cv2.Canny(bin_pic, 30, 100)

    # bin_pic = pic_gray_to3d(bin_pic)
    # return bin_pic


    # out = pic_gray_to3d(bin_pic)
    # print(out.shape)
    # return out

    # lines = cv2.HoughLinesP(
    #         bin_pic, 1, np.pi / 180,
    #         10, minLineLength=3, maxLineGap=30
    # )
    lines = cv2.HoughLinesP(
            bin_pic, 1, np.pi / 180,
            15, minLineLength=3, maxLineGap=10
    )

    # lines = lines_suppression(lines)
    # print("lines after suppression", len(lines), f"before: {sz}")

    return lines


def draw_hough_lines(orig_pic, name=None, **kw):
    # orig_pic = 255 - orig_pic

    lines = find_hough_lines(orig_pic)
    out = orig_pic.copy()

    if lines is None:
        print(f"NOT FOUND LINES IN: {name}")
        return out
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return out


if __name__ == "__main__":
    multi_picture_export(
            PICS_LINES, subfolder="hough",
            function=draw_hough_lines,

    )
