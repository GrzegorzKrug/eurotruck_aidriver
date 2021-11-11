import matplotlib.pyplot as plt
import imutils
import numpy as np

from scipy.signal import convolve

import scipy
import time
import glob
import sys
import cv2
import os

from sklearn.cluster import MeanShift, KMeans

"3th Party"
from matplotlib.animation import FuncAnimation
from collections import deque

"Custom"
from videocap import VideoCap

path1 = "src_movies/2021-10-21 17-31-27.mkv"
path2 = "src_movies/autostrada-notsmooth.mp4"
path1 = path2

delay = int(1 / 30 * 300)


class FrameFoi:
    def __init__(self, y1, y2, x1, x2):
        """
        Args:
            y1: Height start from top
            y2: Height end from top
            x1: Width start from left
            x2: Width end from left
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.xsl = slice(None, None, None)
        self.ysl = slice(None, None, None)

    def drawing_points(self, frame):
        h, w, *_ = frame.shape
        pt1 = np.round([w * self.x1, h * self.y1, ]).astype(int)
        pt2 = np.round([w * self.x2, h * self.y2, ]).astype(int)
        return pt1, pt2

    def get_slices(self, frame):
        """

        Args:
            frame:

        Returns:
            yslice
            xslice
        """
        h, w, *c = frame.shape
        x = (np.array([self.x1, self.x2]) * w).round().astype(int)
        y = (np.array([self.y1, self.y2]) * h).round().astype(int)
        xsl = slice(*x)
        ysl = slice(*y)
        self.xsl = xsl
        self.ysl = ysl
        return ysl, xsl

    def get_foi(self, frame):
        ysl, xsl = self.get_slices(frame)
        roi = frame[ysl, xsl]
        return roi


def image_to_features(arr, include_pos=False, pos_weight=0.5):
    """Transform image pixels to features"""
    h, w, *_ = arr.shape

    y, x = np.ogrid[:h, :w]
    xx, yy = np.meshgrid(x, y)
    keys = np.stack([yy, xx], axis=-1).reshape(-1, 2).T
    translation_key = tuple(map(tuple, keys))

    out = arr[translation_key]

    if include_pos:
        mx = np.max(keys, axis=1).reshape(2, -1)
        keys = keys / mx
        out = np.concatenate([keys.T * pos_weight, out], axis=1)

    return out


if __name__ == "__main__":
    # foi1 = FrameFoi(0.5, 0.7, 0.2, 0.3)
    foi_linedetect = FrameFoi(293 / 720, 445 / 720, 468 / 1280, 1042 / 1280)  # Line detector
    foi_roadsample = FrameFoi(350 / 720, 445 / 720, 468 / 1280, 900 / 1280)  # Road samples
    foi_roadsample = FrameFoi(420 / 720, 445 / 720, 468 / 1280, 900 / 1280)  # Road smaller
    # foi_roadsample = foi_linedetect
    v1 = VideoCap(path1)
    v1.goto(1000)
    LIMIT = 100
    HIST_BINS = 255
    rgb_history = deque(maxlen=LIMIT)
    gray_history = deque(maxlen=LIMIT)

    re, fr = v1.read()

    # plt.imshow(fr)

    # fig = plt.figure(figsize=(12, 9))
    # plt.subplots(gridspec_kw={'height_prop': [5, 1]})
    plt.subplots(2, 1, figsize=(8, 3), gridspec_kw={'height_ratios': [5, 1]})

    fig = plt.gcf()
    # ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(1, 1, 1)

    # act1 = ax1.imshow(fr[:, :, [2, 1, 0]])
    his_r, lev = np.histogram(fr[:, :, 2], 255, )
    his_g, lev = np.histogram(fr[:, :, 1], 255, )
    his_b, lev = np.histogram(fr[:, :, 0], 255, )

    act2_r = ax2.plot(his_r, color='r')
    act2_g = ax2.plot(his_g, color='g')
    act2_b = ax2.plot(his_b, color='b')

    print(dir(act2_r))

    # print(dir(act1))

    # def init():
    #     ax.imshow(fr)
    N = 3

    ms = MeanShift(bandwidth=50)


    # cls=KMeans()

    def hist_animation(i):
        # print(f"i: {i:>5}")
        # while True:

        ret, fr_full = v1.read()

        if not ret:
            return []
        fr_orig = imutils.resize(fr_full, width=800)
        fr_draw = fr_orig.copy()
        fr_gray = cv2.cvtColor(fr_orig, cv2.COLOR_BGR2GRAY)

        frame_road = foi_roadsample.get_foi(fr_orig)
        frame_road_gray = foi_roadsample.get_foi(fr_gray)

        # acx.imshow(fr)
        # cv2.rectangle(fr_draw, *foi_roadsample.drawing_points(fr_orig), (150, 200, 0), 2)

        # his_r, lev = np.histogram(frame_road[:, :, 2], HIST_BINS, range=[0, 255])
        # his_g, lev = np.histogram(frame_road[:, :, 1], HIST_BINS, range=[0, 255])
        # his_b, lev = np.histogram(frame_road[:, :, 0], HIST_BINS, range=[0, 255])
        # his_r = convolve(his_r, np.ones(N), 'same') / N
        # his_g = convolve(his_g, np.ones(N), 'same') / N
        # his_b = convolve(his_b, np.ones(N), 'same') / N

        # frame_gray = cv2.cvtColor(fr_orig, cv2.COLOR_BGR2GRAY)
        # road_gray = foi_roadsample.get_foi(frame_gray)
        # hist_gray, lev = np.histogram(fr.reshape(-1, 3), HIST_BINS, )
        hist_gray, lev = np.histogram(fr_gray.ravel(), HIST_BINS, range=[0, 255])
        # hist_gray = convolve(hist_gray, [1, 1, 1], 'same') / 3
        print("mean shift")
        # features = image_to_features(frame_road, include_pos=False, pos_weight=100)
        ret = ms.fit(hist_gray.reshape((-1, 1)))

        labs = ret.labels_
        centroids = ret.cluster_centers_
        print(labs)
        print()

        # act1.set_array(fr[:, :, [2, 1, 0]])
        # act1.set_array(frame_gray)
        # cv2.imshow('road', road_gray)
        cv2.imshow('road', frame_road)
        ax2.clear()
        ax2.plot(hist_gray)
        # ax2.set_ylim([0, 20000])
        # ax2.set_ylim([0, 200])

        key = cv2.waitKey(50) & 0xFF
        if key == 32:
            cv2.imwrite("trouble_frame.png", fr_full)



    anim = FuncAnimation(
            fig,
            func=hist_animation,
            frames=30  , interval=50,
            # blit=True
            repeat=False,
    )

    plt.show()
