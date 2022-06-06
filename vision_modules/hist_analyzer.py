import matplotlib.pyplot as plt
import imutils
import numpy as np
from numba import jit

from scipy.signal import convolve
# from frame_foi import FrameFoi
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
from frame_foi import foi_roadsample, foi_window_view, foi_roadsample_low, foi_roadsample_high


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

fr_full = cv2.imread(p, cv2.IMREAD_COLOR)
fr = imutils.resize(fr_full, width=800)
fr_gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)


def get_smooth_hist(pic_rgb, smooth=10, smooth_f='mean'):
    his_r, his_g, his_b, levels = hist_3(pic_rgb)
    if smooth >= 2:
        if smooth % 2:
            smooth += 1

        if smooth_f == 'mean':
            his_b = convolve(his_b, np.ones(smooth), 'same') / smooth
            his_g = convolve(his_g, np.ones(smooth), 'same') / smooth
            his_r = convolve(his_r, np.ones(smooth), 'same') / smooth

        elif smooth_f == 'median':
            his_b = fast_rolling_median(his_b, smooth=smooth)
            his_g = fast_rolling_median(his_g, smooth=smooth)
            his_r = fast_rolling_median(his_r, smooth=smooth)

        elif smooth_f == 'max':
            his_b = fast_rolling_max(his_b, smooth=smooth)
            his_g = fast_rolling_max(his_g, smooth=smooth)
            his_r = fast_rolling_max(his_r, smooth=smooth)

    return his_r, his_g, his_b


@jit()
def fast_rolling_median(arr, smooth=10):
    halfN = smooth // 2
    size = len(arr)
    out = np.zeros_like(arr)
    for ind in range(size):
        i1 = ind - halfN
        i1 = 0 if i1 < 0 else i1
        i2 = ind + halfN
        i2 = size if i2 >= size else i2
        roi = arr[i1:i2]
        # print(roi.shape)
        out[ind] = np.median(roi)

    return out


@jit()
def fast_rolling_max(arr, smooth=10):
    halfN = smooth // 2
    size = len(arr)
    out = np.zeros_like(arr)
    for ind in range(size):
        i1 = ind - halfN
        i1 = 0 if i1 < 0 else i1
        i2 = ind + halfN
        i2 = size if i2 >= size else i2
        roi = arr[i1:i2]
        out[ind] = np.max(roi)

    return out


def plot_line_histogram(pic):
    # pic = foi_roadsample.get_foi(pic_rgb)
    pic = foi_roadsample_low.get_foi(pic)
    return plot_smooth_histogram(pic)


def plot_smooth_histogram(pic_bgr, smooth=3):
    """BGR"""
    h, w, *c = pic_bgr.shape

    if len(c) > 0 and type(c[0]) is int and c[0] == 3:
        # his_r, his_g, his_b, levels = hist_3(pic_rgb)
        his_b, his_g, his_r = get_smooth_hist(pic_bgr, smooth=smooth)
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
    mn, pk, mx = indexes[:, 0]

    return mn, pk, mx


def mask_lines_based_on_hist(pic):
    pic = pic.copy()
    # fr = foi_roadsample.get_foi(pic)
    fr = foi_roadsample_low.get_foi(pic)
    hb, hg, hr = get_smooth_hist(fr)

    b, g, r = pic[:, :, 0], pic[:, :, 1], pic[:, :, 2]

    mn, pk, mx = find_top_peak(hr)
    mask_r = (mn <= r) & (r <= mx)

    mn, pk, mx = find_top_peak(hg)
    mask_g = (mn <= g) & (g <= mx)

    mn, pk, mx = find_top_peak(hb)
    mask_b = (mn <= b) & (b <= mx)

    mask = mask_r & mask_g & mask_b
    pic = np.ones(pic.shape, dtype=np.uint8) * 255
    pic[mask] = [0, 0, 0]

    return pic


def mask_lines_on_hist_delta(pic):
    """BGR"""
    pic = pic.copy()
    fr = foi_roadsample.get_foi(pic)

    hb, hg, hr = get_smooth_hist(fr, )

    b, g, r = pic[:, :, 0], pic[:, :, 1], pic[:, :, 2]

    mnr, pkr, mxr = find_top_peak(hr)
    mask_r = (mnr <= r) & (r <= mxr)

    mng, pkg, mxg = find_top_peak(hg)
    mask_g = (mng <= g) & (g <= mxg)

    mnb, pkb, mxb = find_top_peak(hb)
    mask_b = (mnb <= b) & (b <= mxb)

    mask = mask_r & mask_g & mask_b

    # delta = b.astype(float) - r.astype(float)
    delta_rg = np.abs(r.astype(float) - g.astype(float))
    # mask_rg = delta_rg <= 20

    mask_rb = (r.astype(float) - b.astype(float)) > -5

    mask = mask & mask_rb

    # print(f"nm: r: {mnr}-{pkr}-{mxr}, gr:{mng}-{pkg}-{mxg}, bl:{mnb}-{pkb}-{mxb}")
    # print(f"nm: r: {pkr:>4} {mxr - mnr:>4}, gr:{pkg:>4} {mxg - mng:>4}, bl:{pkb:>4} {mxb - mnb:>4}")

    pic = np.ones(pic.shape, dtype=np.uint8) * 255
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
    pool = mpc.Pool(10)
    # pool = None

    # multi_picture_export(
    #         PICS_AUTOSTRADA, subfolder="autostrada-foi",
    #         function=foi_roadsample.get_foi,
    #         pool=pool)

    multi_picture_export(
            PICS_AUTOSTRADA, subfolder="autostrada-hist",
            function=mask_lines_based_on_hist,
            # function=foi_roadsample.get_foi,
            matplot_f=plot_line_histogram,
            clip_final_pic=foi_window_view.get_foi,
            # clip_final_pic=foi_roadsample.get_foi,
            # clip_final_pic=foi_roadsample_low.get_foi,
            pool=pool)

    # multi_picture_export(
    #         PICS_AUTOSTRADA, subfolder="autostrada-hist", postfix="with_model",
    #         function=mask_lines_on_hist_delta,
    #         matplot_f=plot_line_histogram,
    #         clip_final_pic=foi_window_view.get_foi,
    #         pool=pool)

    # plot_histogram()
    # check_mean_shift_of_hisogram()
    tend = time.time()
    print(f"Whole script finished in: {time_formatter(tend - t0)}")
