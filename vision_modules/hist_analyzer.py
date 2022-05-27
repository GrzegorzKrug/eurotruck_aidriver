import matplotlib.pyplot as plt
import imutils
import numpy as np

from scipy.signal import convolve
from line_reader import FrameFoi
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


def hist_3(arr):
    rg = 0, 255
    his_r, lev = np.histogram(arr[:, :, 0], 255, range=rg)
    his_g, lev = np.histogram(arr[:, :, 1], 255, range=rg)
    his_b, lev = np.histogram(arr[:, :, 2], 255, range=rg)
    return his_r, his_g, his_b, lev


files = CABIN_PICS_2
# files.sort()
# print(files)

p = files[0]

# foi_roadsample = FrameFoi(390 / 720, 400 / 720, 468 / 1280, 700 / 1280)
foi_roadsample = FrameFoi(350 / 720, 445 / 720, 468 / 1280, 900 / 1280)  # Original road

foi_window_view = FrameFoi(100 / 720, 445 / 720, 468 / 1280, 900 / 1280)  # viewport

fr_full = cv2.imread(p, cv2.IMREAD_COLOR)
fr = imutils.resize(fr_full, width=800)
fr_gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)


def get_smooth_hist(pic_rgb, N=5):
    his_r, his_g, his_b, levels = hist_3(pic_rgb)
    if N >= 2:
        if N % 2:
            N += 1
        his_b = convolve(his_b, np.ones(N), 'same') / N
        his_g = convolve(his_g, np.ones(N), 'same') / N
        his_r = convolve(his_r, np.ones(N), 'same') / N

    return his_r, his_g, his_b


def plot_line_histogram(pic_rgb):
    pic = foi_roadsample.get_foi(pic_rgb)
    return plot_smooth_histogram(pic)


def plot_smooth_histogram(pic_rgb, smooth=2):
    h, w, *c = pic_rgb.shape

    if len(c) > 0 and type(c[0]) is int and c[0] == 3:
        # his_r, his_g, his_b, levels = hist_3(pic_rgb)
        his_r, his_g, his_b = get_smooth_hist(pic_rgb)
        # his_r, his_g, his_b, levels = hist_3(pic_rgb)
        plt.plot(his_r, color='r')
        plt.plot(his_g, color='g')
        plt.plot(his_b, color='b')

        xticks = np.linspace(0, 255, 24).round().astype(int)
        plt.xticks(xticks)
        # plt.semilogy()
        plt.grid()


def find_top_peak(arr):
    assert len(arr.shape) == 1, "Please pass list"
    sz = arr.shape[0]
    # print(sz)
    indexes = np.zeros((3, 2), dtype=int)
    indexes[:, 0] = sz
    indexes[:, 1] = arr[-1]

    looking_for_end = True
    looking_for_peak = False
    looking_for_start = False

    for i in range(254, -1, -1):
        v = arr[i]
        if looking_for_end:
            if v <= indexes[2, 1]:
                indexes[2, 0] = i
                indexes[2, 1] = v
            else:
                looking_for_end = False
                looking_for_peak = True

        elif looking_for_peak:
            if v >= indexes[1, 1]:
                indexes[1, 0] = i
                indexes[1, 1] = v
            else:
                looking_for_peak = False
                looking_for_start = True

        elif looking_for_start:
            if v <= indexes[0, 0]:
                indexes[0, 0] = i
                indexes[0, 1] = v
            else:
                looking_for_start = False

    # print(indexes)
    return tuple(indexes[:, 0])


def mask_lines_based_on_hist(pic):
    pic = pic.copy()
    fr = foi_roadsample.get_foi(pic)
    hr, hg, hb = get_smooth_hist(fr)

    b, g, r = pic[:, :, 0], pic[:, :, 1], pic[:, :, 2]

    mn, pk, mx = find_top_peak(hr)
    mask_r = (mn <= r) & (r <= mx)

    mn, pk, mx = find_top_peak(hg)
    mask_g = (mn <= g) & (g <= mx)

    mn, pk, mx = find_top_peak(hb)
    mask_b = (mn <= b) & (b <= mx)

    mask = mask_r & mask_g & mask_b
    pic[mask] = [0, 0, 0]

    return pic


def check_mean_shift_of_hisogram():
    R = 1
    N = 15
    hist, levels = np.histogram(fr_gray, 255, range=[0, 255])
    hist = convolve(hist, np.ones(N), 'same') / N
    # hist = convolve(hist, np.ones(N), 'same') / N
    der1 = convolve(hist, [1, 0, -1], 'same')
    der1[[0, -1]] = 0
    featrs = np.stack([np.arange(len(hist)), hist], axis=1)

    ms = MeanShift(bandwidth=100)
    ret = ms.fit(hist.reshape(-1, 1))
    # ret = ms.fit(hist.reshape(-1, 1))

    labs = ret.labels_
    centr = ret.cluster_centers_

    plt.plot(hist)
    # plt.plot(der1)
    plt.grid()
    plt.show()


def clip_mirror(img):
    return img[750:1100, 20:350, :]


def clip_pobocze(img):
    return img[300:500, 1200:1550, :]


PICS = CABIN_PICS_1

if __name__ == "__main__":
    t0 = time.time()
    pool = mpc.Pool(15)
    # pool=None
    # multi_picture_export(PICS, subfolder="hist-1", function=mask_lines_based_on_hist)
    # multi_apply(PICS, subfolder="hist-1", function=foi_roadsample.get_foi, postfix="oryg")
    multi_picture_export(
            PICS_AUTOSTRADA, subfolder="autostrada", prefix="hist",
            function=mask_lines_based_on_hist,
            matplot_f=plot_line_histogram,
            clip_final_pic=foi_window_view.get_foi,
            pool=pool)
    # print(COLORS_PATHS)
    # multi_picture_export(
    #         COLORS_PATHS, subfolder="colors-1",
    #         # function=mask_lines_based_on_hist,
    #         matplot_f=draw_line_histogram,
    #         # clip_f=foi_roadsample.get_foi,
    #         # clip_f=clip_pobocze,
    #         pool=pool)

    # plot_histogram()
    # check_mean_shift_of_hisogram()
    tend = time.time()
    print(f"Whole script finished in: {time_formatter(tend - t0)}")
