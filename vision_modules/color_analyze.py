import multiprocessing as mpc
import matplotlib.pyplot as plt
import imutils
import numpy as np
from numba import jit

from scipy.signal import convolve

from sklearn.cluster import MeanShift, KMeans

import scipy
import time
import glob
import sys
import cv2
import os

from picture_paths import (
    COLORS_PATHS, PICS_AUTOSTRADA,
    PICS_FRANKFURT, PICS_FRANKFURT_TRAFFIC, PIC_SOURCE_FOLDER,
)
from multiprocoess_functions import *
from frame_foi import (
    foi_roadsample, foi_window_view, foi_roadsample_low,
    foi_frontvision, foi_frontsample, foi_map, foi_road_model,
    foi_mirror_left, foi_mirror_right, foi_no_hud,
)

from hough_module import draw_hough_lines
from utility import (
    rolling_smooth, pic_gray_to3d, mask_hud, image_to_features, image_from_features,
    DEFINED_COLORS,
    timedecorator, time_formatter,
)

from multiprocoess_functions import *
from frame_foi import foi_roadsample, foi_window_view


def convert_hsv(img, **kw):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def red_green(img, **kw):
    """BGR"""
    diff = img[:, :, 2].astype(float) - img[:, :, 1].astype(float)
    return norm_stack_layer(diff)


def green_red(img, **kw):
    """BGR"""
    diff = img[:, :, 1].astype(float) - img[:, :, 2].astype(float)
    # diff = -diff
    return norm_stack_layer(diff)


def red_blue(img, **kw):
    """BGR"""
    diff = img[:, :, 2].astype(float) - img[:, :, 0].astype(float)
    return norm_stack_layer(diff)


def blue_red(img, **kw):
    """BGR"""
    diff = img[:, :, 0].astype(float) - img[:, :, 2].astype(float)
    return norm_stack_layer(diff)


def blue_green(img, **kw):
    """BGR"""
    diff = img[:, :, 0].astype(float) - img[:, :, 1].astype(float)
    return norm_stack_layer(diff)


def green_blue(img, **kw):
    """BGR"""
    diff = img[:, :, 1].astype(float) - img[:, :, 0].astype(float)
    return norm_stack_layer(diff)


# def red_green_diff(img):
#     """BGR"""
#     diff = img[:, :, 2].astype(float) - img[:, :, 1].astype(float)
#     return norm_stack_layer(diff)


def norm_stack_layer(diff):
    # mn, mx = diff.min(), diff.max()
    # diff = (diff - mn) / (mx - mn) * 255
    diff = np.clip(diff, 0, 255).round()
    img = np.ones((*diff.shape, 3), dtype=np.uint8)
    img[:, :, 0] = diff
    img[:, :, 1] = diff
    img[:, :, 2] = diff
    return img


def clip_mirror(img):
    return img[750:1100, 20:350, :]


def clip_pobocze(img):
    return img[300:500, 1200:1550, :]


def run_1():
    "HSV check"
    multi_picture_export(
            PICS_AUTOSTRADA, subfolder="autostrada-hsv",
            function=red_green, postfix="red-green",
            attach_original=False,
            clip_final_pic=foi_window_view.get_foi,
            pool=pool)

    multi_picture_export(
            PICS_AUTOSTRADA, subfolder="autostrada-hsv",
            function=green_red, postfix="green-red",
            attach_original=False,
            clip_final_pic=foi_window_view.get_foi,
            pool=pool)

    multi_picture_export(
            PICS_AUTOSTRADA, subfolder="autostrada-hsv",
            function=green_blue, postfix="green-blue",
            attach_original=False,
            clip_final_pic=foi_window_view.get_foi,
            pool=pool)

    multi_picture_export(
            PICS_AUTOSTRADA, subfolder="autostrada-hsv",
            function=blue_green, postfix="blue-green",
            attach_original=False,
            clip_final_pic=foi_window_view.get_foi,
            pool=pool)

    multi_picture_export(
            PICS_AUTOSTRADA, subfolder="autostrada-hsv",
            function=blue_red, postfix="blue-red",
            attach_original=False,
            clip_final_pic=foi_window_view.get_foi,
            pool=pool)
    multi_picture_export(
            PICS_AUTOSTRADA, subfolder="autostrada-hsv",
            function=red_blue, postfix="red-blue",
            attach_original=False,
            clip_final_pic=foi_window_view.get_foi,
            pool=pool)


def strip_pic_to_components(pic, funct):
    img = cv2.cvtColor(pic, funct)
    c1 = img[:, :, 0]
    c2 = img[:, :, 1]
    c3 = img[:, :, 2]
    return c1, c2, c3


def plot_colors(pic, **kw):
    blue = pic[:, :, 0]
    green = pic[:, :, 1]
    red = pic[:, :, 2]

    pic = cv2.blur(pic, (15, 15))
    # pic = cv2.medianBlur(pic, 11)
    # pic = cv2.blur(pic, (5, 5))

    hue, sat, val = strip_pic_to_components(pic, cv2.COLOR_BGR2HSV)
    hue = np.round(hue / hue.max() * 255).astype(np.uint8)
    # rbl, rr, val = strip_pic_to_components(pic, cv2.COLOR_BGR2HSV)
    # pic_hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
    # hue = pic_hsv[:, :, 0]
    # sat = pic_hsv[:, :, 1]
    # val = pic_hsv[:, :, 2]

    img1 = np.vstack([red, green, blue])
    img2 = np.vstack([hue, sat, val])

    img = np.hstack([img1, img2])
    img = pic_gray_to3d(img)
    return img


if __name__ == "__main__":
    t0 = time.time()
    # pool = mpc.Pool(5)
    pool = None

    # multi_picture_export(
    #         PICS_FRANKFURT_TRAFFIC, subfolder="color-components",
    #         function=plot_colors,
    #         attach_original=False,
    #         loop_start=199,
    #         loop_lim=200,
    #         pool=pool)

    path = PIC_SOURCE_FOLDER + os.path.join("src_images", "zima.jpg")
    pic = cv2.imread(path, cv2.IMREAD_COLOR)
    img = plot_colors(pic)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    plt.figure(figsize=(12, 7))
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    tend = time.time()
    print(f"Whole script finished in: {time_formatter(tend - t0)}")
