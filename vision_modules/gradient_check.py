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
    CABIN_PICS_1, CABIN_PICS_2, COLORS_PATHS, PICS_AUTOSTRADA,
    PICS_FRANKFURT, PICS_FRANKFURT_TRAFFIC,
)
from multiprocoess_functions import *
from frame_foi import (
    foi_roadsample, foi_window_view, foi_roadsample_low,
    foi_frontvision, foi_frontsample, foi_map, foi_road_model,
    foi_mirror_left, foi_mirror_right,
)

from hough_module import draw_hough_lines
from utility import (
    rolling_smooth, pic_gray_to3d, mask_hud, image_to_features, image_from_features,
    DEFINED_COLORS,
)


def calc_road_gradient(pic, name=None, **kw):
    pic = mask_hud(pic)
    pic = cv2.blur(pic, (5, 5))
    pic = cv2.medianBlur(pic, 15)
    pic[:360, :, :] = 0
    # return pic

    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

    binary_pic = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 7, 1.7,
    )

    pic = pic_gray_to3d(binary_pic)
    pic = mask_hud(pic)
    return pic


def road_segmentize(orig_pic, name=None, **kw):
    """"""
    print(name)
    orig_pic = mask_hud(orig_pic)

    h, w, _ = orig_pic.shape
    ftrs = image_to_features(orig_pic, include_pos=False, pos_weight=155)
    # diff = orig_pic - img

    # ms = MeanShift(bandwidth=5)
    ms = KMeans(3)
    ret = ms.fit(ftrs)
    color_ftrs = ftrs[:, :3]

    labs = ret.labels_
    unq_lb = np.unique(labs)
    for lb in unq_lb:
        mask = labs == lb
        color = DEFINED_COLORS[lb]
        # print(color)
        color_ftrs[mask] = color

        # mask = mask.reshape(h, w)
        # plt.imshow(mask)
        # plt.colorbar()
        # plt.show()
        # mask = np.array(mask*255, dtype=np.uint8)
        # mask = pic_gray_to3d(mask)
        # print(mask.shape)
        # return mask
        # ftrs[mask] = color
    # for lb in labs:
    # centroids = ret.cluster_centers_
    # print(ftrs.shape)
    img = image_from_features(color_ftrs, h, w)
    return img

    # act1.set_array(fr[:, :, [2, 1, 0]])
    # act1.set_array(frame_gray)
    # cv2.imshow('road', road_gray)
    # cv2.imshow('road', frame_road)
    # ax2.clear()
    # ax2.plot(hist_gray)
    # ax2.set_ylim([0, 20000])
    # ax2.set_ylim([0, 200])
    # return orig_pic


if __name__ == "__main__":
    t0 = time.time()
    pool = mpc.Pool(8)
    # pool = None

    multi_picture_export(
            PICS_FRANKFURT_TRAFFIC, subfolder="traffic-gradient",
            function=road_segmentize,
            # funal_pic=foi_frontsample.get_foi,
            pool=pool,
            loop_start=100,
            loop_lim=120,
    )
    #
    # plot_histogram()
    # check_mean_shift_of_hisogram()
    tend = time.time()
    print(f"Whole script finished in: {time_formatter(tend - t0)}")
