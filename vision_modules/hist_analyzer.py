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
from utility import rolling_smooth, pic_gray_to3d,mask_hud


def hist_3(arr):
    rg = 0, 255
    his_r, lev = np.histogram(arr[:, :, 0], 255, range=rg)
    his_g, lev = np.histogram(arr[:, :, 1], 255, range=rg)
    his_b, lev = np.histogram(arr[:, :, 2], 255, range=rg)
    return his_r, his_g, his_b, lev


files = CABIN_PICS_2

p = files[0]

fr_full = cv2.imread(p, cv2.IMREAD_COLOR)
fr = imutils.resize(fr_full, width=800)
fr_gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)


def get_smooth_hist(pic_rgb, smooth=0, smooth_f='mean'):
    his_r, his_g, his_b, levels = hist_3(pic_rgb)

    if smooth > 1:
        his_r = rolling_smooth(his_r, smooth, smooth_f)
        his_g = rolling_smooth(his_g, smooth, smooth_f)
        his_b = rolling_smooth(his_b, smooth, smooth_f)

        # his_r = rolling_smooth(his_r, smooth, smooth_f)
        # his_g = rolling_smooth(his_g, smooth, smooth_f)
        # his_b = rolling_smooth(his_b, smooth, smooth_f)

    return his_r, his_g, his_b


def plot_smooth_histogram(pic_bgr, smooth=0):
    """BGR"""
    h, w, *c = pic_bgr.shape

    if len(c) > 0 and type(c[0]) is int and c[0] == 3:
        # his_r, his_g, his_b, levels = hist_3(pic_rgb)
        his_b, his_g, his_r = get_smooth_hist(pic_bgr, smooth=smooth)
        # his_r, his_g, his_b, levels = hist_3(pic_rgb)
        plt.plot(his_r, color='r', label='Red')
        plt.plot(his_g, color='g', label="Green")
        plt.plot(his_b, color='b', label="Blue")

        xticks = np.linspace(0, 255, 24).round().astype(int)
        plt.xticks(xticks)
        # plt.semilogy()
        plt.grid()


def find_top_peak(arr, steps_tollerance=4):
    assert len(arr.shape) == 1, "Please pass list"
    sz = arr.shape[0]
    # print(sz)
    "Dim,  [Index value]"
    indexes = np.zeros((3, 2), dtype=int)
    indexes[:, 0] = sz
    indexes[:, 1] = arr[-1]

    pk = np.argmax(arr)

    indexes[:, 0] = pk
    indexes[:, 1] = arr[pk]

    i = pk - 1
    left_steps = steps_tollerance
    while (i >= 0) and ((arr[i] <= indexes[0, 1]) or left_steps > 0):
        if arr[i] <= indexes[0, 1]:
            indexes[0, 0] = i
            indexes[0, 1] = arr[i]
            left_steps = steps_tollerance
        else:
            left_steps -= 1
        i -= 1

    i = pk + 1
    right_steps = steps_tollerance
    while (i <= 254) and ((arr[i] <= indexes[2, 1]) or right_steps > 0):
        if arr[i] <= indexes[2, 1]:
            indexes[2, 0] = i
            indexes[2, 1] = arr[i]
            right_steps = steps_tollerance
        else:
            right_steps -= 1
        i += 1

    mn, pk, mx = indexes[:, 0]
    return mn, pk, mx


def find_last_peak(arr, scale_accept=0):
    assert len(arr.shape) == 1, "Please pass list"
    sz = arr.shape[0]
    # print(sz)
    "Dim,  [Index value]"
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
                indexes[1, 1] = v
                indexes[1, 0] = i

        elif looking_for_peak:
            if v >= indexes[1, 1]:
                indexes[1, 0] = i
                indexes[1, 1] = v
            else:
                looking_for_peak = False
                looking_for_start = True
                indexes[0, 1] = v
                indexes[0, 0] = i

        elif looking_for_start:
            if v <= indexes[0, 0]:
                indexes[0, 0] = i
                indexes[0, 1] = v
            # elif
            else:
                looking_for_start = False

    # print(indexes)
    mn, pk, mx = indexes[:, 0]
    "Extend more"
    d1 = pk - mn
    d2 = mx - pk
    D = max([d1, d2, 15])
    D = min([50, D])

    mn = pk - D
    mx = pk + D
    # mn -= d1 / 2
    # mx += d2 / 2

    return mn, pk, mx


def apply_blur(fr):
    fr = cv2.medianBlur(fr, 11)
    # fr = cv2.GaussianBlur(fr, (11, 11), 5, 5)
    return fr


def plot_line_histogram(pic):
    pic = pic.copy()
    mask = foi_map.get_mask(pic)
    pic[mask] = 0

    "SAMPLE"
    # pic = foi_roadsample.get_foi(pic)
    pic = foi_frontsample.get_foi(pic)
    pic = apply_blur(pic)
    # plt.ylim([-10, 500])
    return plot_smooth_histogram(pic, 15)


def mask_lines_based_on_hist(pic, name=None):
    pic = pic.copy()
    mask = foi_map.get_mask(pic)
    pic[mask] = 0

    "SAMPLE"
    # fr = foi_roadsample.get_foi(pic)
    fr = foi_frontsample.get_foi(pic)
    fr = apply_blur(fr)

    # return fr
    hb, hg, hr = get_smooth_hist(fr, 15, 'mean')
    # print(hr)

    b, g, r = pic[:, :, 0], pic[:, :, 1], pic[:, :, 2]
    mnr, pkr, mxr = find_last_peak(hr)
    mask_r = (mnr <= r) & (r <= mxr)

    mng, pkg, mxg = find_last_peak(hg)
    mask_g = (mng <= g) & (g <= mxg)

    mnb, pkb, mxb = find_last_peak(hb)
    mask_b = (mnb <= b) & (b <= mxb)

    mask = mask_r & mask_g & mask_b
    # mask = get_mask_from_hist(pic, hb, hg, hr)

    pic = np.ones(pic.shape, dtype=np.uint8)
    pic[mask] = [255, 255, 255]

    print(f"{name}: r: {mnr}-{pkr}-{mxr}, gr:{mng}-{pkg}-{mxg}, bl:{mnb}-{pkb}-{mxb}")
    # print(f"{name}: r: {pkr:^4} {mxr - mnr:^4}, gr:{pkg:^4} {mxg - mng:^4}, bl:{pkb:^4} {mxb - mnb:^4}")

    return pic


def get_mask_from_last_peak(pic, hb, hg, hr):
    b, g, r = pic[:, :, 0], pic[:, :, 1], pic[:, :, 2]
    mnr, pkr, mxr = find_last_peak(hr)
    mask_r = (mnr <= r) & (r <= mxr)

    mng, pkg, mxg = find_last_peak(hg)
    mask_g = (mng <= g) & (g <= mxg)

    mnb, pkb, mxb = find_last_peak(hb)
    mask_b = (mnb <= b) & (b <= mxb)

    mask = mask_r & mask_g & mask_b
    return mask


def get_mask_from_top_peak(pic, hb, hg, hr, name=None):
    b, g, r = pic[:, :, 0], pic[:, :, 1], pic[:, :, 2]
    mnr, pkr, mxr = find_top_peak(hr)
    mask_r = (mnr <= r) & (r <= mxr)

    mng, pkg, mxg = find_top_peak(hg)
    mask_g = (mng <= g) & (g <= mxg)

    mnb, pkb, mxb = find_top_peak(hb)
    mask_b = (mnb <= b) & (b <= mxb)

    # print(f"{name}: r: {mnr}-{pkr}-{mxr}, gr:{mng}-{pkg}-{mxg}, bl:{mnb}-{pkb}-{mxb}")
    mask = mask_r & mask_g & mask_b
    return mask


def mask_lines_on_hist_delta(pic):
    """BGR"""
    pic = pic.copy()
    fr = foi_roadsample.get_foi(pic)

    hb, hg, hr = get_smooth_hist(fr, )

    b, g, r = pic[:, :, 0], pic[:, :, 1], pic[:, :, 2]

    # delta = b.astype(float) - r.astype(float)
    delta_rg = np.abs(r.astype(float) - g.astype(float))
    # mask_rg = delta_rg <= 20

    mask_rb = (r.astype(float) - b.astype(float)) > -5

    mask = mask & mask_rb

    # print(f"nm: r: {mnr}-{pkr}-{mxr}, gr:{mng}-{pkg}-{mxg}, bl:{mnb}-{pkb}-{mxb}")

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


def lines_otsu(fr, **kw):
    # print(fr.dtype)
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    # print(fr.shape)
    pc = cv2.adaptiveThreshold(
            fr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 10)
    pc = pc[:, :, np.newaxis]
    pc = np.tile(pc, [1, 1, 3])
    return pc


def lines_contours(pic, **kw):
    """BRG image"""
    # fr = cv2.Canny(fr, 300, 400) # NO BLUR
    out = apply_blur(pic)
    out = cv2.Canny(out, 80, 120)
    out = pic_gray_to3d(out)

    out = mask_hud(out, expand_mask=5)

    return out


def lines_contours_with_hist(fr, **kw):
    # fr = cv2.Canny(fr, 300, 400) # NO BLUR
    fr = apply_blur(fr)
    out = cv2.Canny(fr, 100, 200)
    out = out[:, :, np.newaxis]
    out = np.tile(out, [1, 1, 3])

    hb, hg, hr, = get_smooth_hist(fr, 10)
    mask = get_mask_from_last_peak(fr, hb, hg, hr)
    out[mask] = 0

    return out


def model_road(pic, name=None, **kw):
    return pic




# CROSS_KERNEL[1, 1] = 0




HORIZON_SMOOTH = 50
HORIZON_CUT_SMOOTH = 250


def plot_histogram_over_axis(pic_bgr, smooth=0, axis=1):
    """

    Args:
        pic_bgr:
        smooth:
        axis:

    Returns:

    """
    h, w, *c = pic_bgr.shape

    if len(c) > 0 and type(c[0]) is int and c[0] == 3:
        # his_r, his_g, his_b, levels = hist_3(pic_rgb)
        his_b = np.sum(pic_bgr[:, :, 0], axis=axis)
        his_g = np.sum(pic_bgr[:, :, 1], axis=axis)
        his_r = np.sum(pic_bgr[:, :, 2], axis=axis)

        # smooth=0
        if smooth > 1:
            his_b = rolling_smooth(his_b, smooth=smooth, )  # smooth_f= 'median')
            his_r = rolling_smooth(his_r, smooth=smooth, )  # smooth_f= 'median')
            his_g = rolling_smooth(his_g, smooth=smooth, )  # smooth_f= 'median')

        gb_diff = his_g - his_b
        rg_diff = his_r - his_g
        rb_diff = his_r - his_b

        # plt.plot(his_r, color='r')
        # plt.plot(his_g, color='g')
        # plt.plot(his_b, color='b')
        Y = - np.arange(len(gb_diff))
        if axis == 1:
            plt.plot(gb_diff, Y, color='k', label="G-B")
            plt.plot(rg_diff, Y, color='cyan', label="R-G")
            plt.plot(rb_diff, Y, color='magenta', label="R-B")
            # plt.plot(his_b, Y, color='b', label="B")
            # plt.plot(his_g, Y, color='g', label="G")
            # plt.plot(his_r, Y, color='r', label="R")
            der_b = get_derivate(his_b)
            der_b = rolling_smooth(der_b, 250)
            plt.plot(der_b, Y, color='y', label='Blue der')

        else:
            plt.plot(gb_diff, color='k', label="G-B")
            plt.plot(rg_diff, color='cyan', label="R-G")
            xticks = np.linspace(0, w, 24).round().astype(int)
            plt.xticks(xticks)

        # plt.xlim([-10000, 10000])
        # size = w if axis == 0 else h

        plt.grid()


def plot_road_histogram(pic, name=None, **kw):
    pic = pic.copy()
    pic = mask_hud(arr=pic, expand_mask=3)

    "SAMPLE"
    road = foi_road_model.get_foi(pic)

    "Get road histogram to mask everything else"
    hists = get_smooth_hist(road, HORIZON_SMOOTH)
    mask = ~get_mask_from_top_peak(pic, name=name, *hists)
    pic[mask] = 0

    return plot_histogram_over_axis(pic, smooth=HORIZON_CUT_SMOOTH, axis=1)


def temp(pic, **kw):
    pic = mask_hud(arr=pic, expand_mask=3)
    return pic


@jit()
def get_derivate(arr):
    out = np.zeros_like(arr)
    out[1:] = arr[1:] - arr[:len(arr) - 1]
    # print()
    # print(arr[:10])
    # print(out[:11])
    # print(out)

    return out


def anti_horizon_model(full_pic, name=None, **kw):
    full_pic = mask_hud(arr=full_pic, expand_mask=3)
    road = foi_road_model.get_foi(full_pic)

    "Get road histogram to mask everything else"
    hists = get_smooth_hist(road, HORIZON_SMOOTH)
    mask = ~get_mask_from_top_peak(full_pic, name=name, *hists)
    masked_background = full_pic.copy()
    masked_background[mask] = 0
    # return pic

    r = masked_background[:, :, 2].sum(axis=1)
    b = masked_background[:, :, 0].sum(axis=1)
    g = masked_background[:, :, 1].sum(axis=1)

    r = rolling_smooth(r, HORIZON_CUT_SMOOTH)
    b = rolling_smooth(b, HORIZON_CUT_SMOOTH)
    g = rolling_smooth(g, HORIZON_CUT_SMOOTH)

    rg_diff = r - g
    gb_diff = g - b
    rb_diff = r - b

    der_b = get_derivate(b)
    der_b = rolling_smooth(der_b, 250)
    ind4 = np.argmax(der_b)

    ind1 = np.argmin(rg_diff)
    ind2 = np.argmax(gb_diff)

    ind5 = np.argsort(np.abs(rb_diff), )[:20]  # Ascending
    ind5_dist = np.abs(ind5 - ind4)
    closest = np.argmin(ind5_dist)
    ind5 = ind5[closest]

    rb_crossing0 = np.any(rb_diff < 0)
    # print(ind5[:10], rb_diff[ind5[:10]])

    # vertical_cut = ind1

    # print("Blue:")
    # print(b)
    blue_th = b.max() / 2.4
    # blue_th = np.sqrt(b.max()) / 1.4
    # blue_th -= np.sqrt(blue_th) * 0.2
    half_blue = b <= blue_th
    half_blue_hist = np.where(half_blue, np.arange(len(b)), np.zeros_like(b))
    ind3 = np.argmax(half_blue_hist)

    if rb_crossing0:
        vertical_cut = np.sum([ind1 * 0.3, ind5 * 0.1, ind3 * 0.1, ind4 * 0.5]).round().astype(int)
    else:
        vertical_cut = np.sum([ind1 * 0.35, ind3 * 0.15, ind4 * 0.5]).round().astype(int)
    print(
            f"Cut of {name} at {vertical_cut}: rg{ind1:>4}, gb0{ind5:>4}, halfblue{ind3:>4}, der_bl{ind4:>4}")
    # vertical_cut = np.min([vertical_cut, ind3]).round().astype(int)

    pic_no_horizon = full_pic.copy()
    pic_no_horizon[:vertical_cut, :, :] = 0

    # gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    # pic = pic_gray_to3d(gray)
    # pic[~mask] = 255
    # plt.plot

    contours = lines_contours(pic_no_horizon)
    pic_with_lines = draw_hough_lines(contours)

    pic_with_lines[vertical_cut, :, :] = 0
    return pic_with_lines


if __name__ == "__main__":
    t0 = time.time()
    pool = mpc.Pool(8)
    # pool = None

    # multi_picture_export(
    #         PICS_FRANKFURT, subfolder="traffic-canny",
    #         # PICS_AUTOSTRADA, subfolder="traffic",
    #         # PICS_FRANKFURT_TRAFFIC, subfolder="traffic",
    #         function=lines_contours,
    #         # function=mask_lines_based_on_hist,
    #         # function=foi_frontsample.get_foi,
    #         # function=lines_otsu,
    #         # matplot_f=plot_line_histogram,
    #         # clip_final_pic=foi_frontvision.get_foi,
    #         # clip_final_pic=foi_frontsample.get_foi,
    #         pool=pool,
    #         # loop_start=48,
    #         loop_lim=80,
    # )

    # multi_picture_export(
    #         PICS_FRANKFURT, subfolder="traffic",
    #         # PICS_AUTOSTRADA, subfolder="traffic",
    #         # PICS_FRANKFURT_TRAFFIC, subfolder="traffic",
    #         # function=lines_contours_with_hist,
    #         function=mask_lines_based_on_hist,
    #         # function=foi_frontsample.get_foi,
    #         # function=lines_otsu,
    #         matplot_f=plot_line_histogram,
    #         # clip_final_pic=foi_frontvision.get_foi,
    #         # clip_final_pic=foi_frontsample.get_foi,
    #         pool=pool,
    #         # loop_start=48,
    #         loop_lim=80,
    # )

    multi_picture_export(
            PICS_FRANKFURT_TRAFFIC, subfolder="traffic-horizon",
            # function=temp,
            function=anti_horizon_model,
            # function=foi_mirror_left.get_foi,
            # function=foi_road_model.get_foi,
            # function=foi_frontsample.get_foi,
            # function=lines_otsu,
            matplot_f=plot_road_histogram,
            # clip_final_pic=foi_frontvision.get_foi,
            # clip_final_pic=foi_frontsample.get_foi,
            pool=pool,
            loop_start=110,
            loop_lim=120,
    )
    #
    # plot_histogram()
    # check_mean_shift_of_hisogram()
    tend = time.time()
    print(f"Whole script finished in: {time_formatter(tend - t0)}")
