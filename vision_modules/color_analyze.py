import matplotlib.pyplot as plt
import imutils
import numpy as np

from scipy.signal import convolve
from sklearn.cluster import MeanShift, KMeans
import multiprocessing as mpc

import scipy
import time
import glob
import sys
import cv2
import os

from picture_paths import CABIN_PICS_1, CABIN_PICS_2, COLORS_PATHS, PICS_AUTOSTRADA
from multiprocoess_functions import *
from frame_foi import foi_roadsample, foi_window_view


def convert_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def red_green(img):
    """BGR"""
    diff = img[:, :, 2].astype(float) - img[:, :, 1].astype(float)
    return norm_stack_layer(diff)


def green_red(img):
    """BGR"""
    diff = img[:, :, 1].astype(float) - img[:, :, 2].astype(float)
    # diff = -diff
    return norm_stack_layer(diff)


def red_blue(img):
    """BGR"""
    diff = img[:, :, 2].astype(float) - img[:, :, 0].astype(float)
    return norm_stack_layer(diff)


def blue_red(img):
    """BGR"""
    diff = img[:, :, 0].astype(float) - img[:, :, 2].astype(float)
    return norm_stack_layer(diff)


def blue_green(img):
    """BGR"""
    diff = img[:, :, 0].astype(float) - img[:, :, 1].astype(float)
    return norm_stack_layer(diff)


def green_blue(img):
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


PICS = CABIN_PICS_1

if __name__ == "__main__":
    t0 = time.time()
    pool = mpc.Pool(15)

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

    # multi_picture_export(
    #         PICS_AUTOSTRADA, subfolder="autostrada-hsv",
    #         function=red_green_diff, postfix="red-green",
    #         pool=pool)

    # plot_histogram()
    # check_mean_shift_of_hisogram()
    tend = time.time()
    print(f"Whole script finished in: {time_formatter(tend - t0)}")
