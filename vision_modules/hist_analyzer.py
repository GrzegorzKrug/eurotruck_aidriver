import matplotlib.pyplot as plt
import imutils
import numpy as np

from scipy.signal import convolve
from line_reader import FrameFoi
from sklearn.cluster import MeanShift, KMeans

import scipy
import time
import glob
import sys
import cv2
import os

from picture_paths import CABIN_PICS_1, CABIN_PICS_2, multi_picture_export


def hist_3(arr):
    his_r, lev = np.histogram(arr[:, :, 0], 255, range=[0, 255])
    his_g, lev = np.histogram(arr[:, :, 1], 255, range=[0, 255])
    his_b, lev = np.histogram(arr[:, :, 2], 255, range=[0, 255])
    return his_r, his_g, his_b, lev


files = CABIN_PICS_2
# files.sort()
# print(files)

p = files[0]

foi_roadsample = FrameFoi(390 / 720, 400 / 720, 468 / 1280, 700 / 1280)
foi_roadsample = FrameFoi(350 / 720, 445 / 720, 468 / 1280, 900 / 1280)  # Original road
# foi_roadsample = FrameFoi(350 / 720, 445 / 720, 468 / 1280, 700 / 1280)  #
foi_roadsample = FrameFoi(350 / 720, 445 / 720, 800 / 1280, 850 / 1280)  #

fr_full = cv2.imread(p, cv2.IMREAD_COLOR)
fr = imutils.resize(fr_full, width=800)
fr_gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)


def get_3d_hist(pic, N=15):
    his_b, his_g, his_r, levels = hist_3(pic)
    if N >= 2:
        if N % 2:
            N += 1
        his_b = convolve(his_b, np.ones(N), 'same') / N
        his_r = convolve(his_r, np.ones(N), 'same') / N
        his_g = convolve(his_g, np.ones(N), 'same') / N

    return his_r, his_g, his_b


def locate_threshold(his, amp=100, min_bright=100):
    pass
    # print(len(his))
    # for i in range(255)


def draw_line_histogram(pic):
    h, w, *c = pic.shape
    pic = foi_roadsample.get_foi(pic)

    if len(c) > 0 and type(c[0]) is int and c[0] == 3:
        his_r, his_b, his_g = get_3d_hist(pic, )
        plt.plot(his_r, color='r')
        plt.plot(his_g, color='g')
        plt.plot(his_b, color='b')
        plt.grid()
    # else:
    #     hist, levels = np.histogram(pic, 255, range=[0, 255])
    #     hist_mean = convolve(hist, np.ones(N), 'same') / N
    #     plt.plot(hist, label="hist")
    #     plt.plot(hist_mean, label="mean")
    #     plt.legend()


def mask_lines_based_on_hist(pic):
    fr = foi_roadsample.get_foi(pic)
    hr, hg, hb = get_3d_hist(fr)
    # print(len(hr), len(hb), len(hb))
    locate_threshold(hr)

    return fr


# def plot_histogram():
#     N = 10
#     fr_r = foi_roadsample.get_foi(fr)
#     plt.figure()
#     plt.imshow(fr_r)
#     plt.colorbar()
#     plt.show()


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


PICS = CABIN_PICS_1

if __name__ == "__main__":
    multi_picture_export(PICS, subfolder="hist-1", function=mask_lines_based_on_hist)
    # multi_apply(PICS, subfolder="hist-1", function=foi_roadsample.get_foi, postfix="oryg")
    multi_picture_export(PICS, matplot_f=draw_line_histogram, subfolder="hist-1", postfix="hist")

# plot_histogram()
# check_mean_shift_of_hisogram()
