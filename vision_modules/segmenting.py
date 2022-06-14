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
    PICS_FRANKFURT, PICS_FRANKFURT_TRAFFIC, SOURCE_FOLDER,
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


def get_segment_color(center):
    pass


def road_segmentize(orig_pic, name=None, clip_hud=True, speed_up=6,
        use_hsv=True,
        **kw):
    """"""
    include_pos = True

    if use_hsv:
        orig_pic = cv2.cvtColor(orig_pic, cv2.COLOR_BGR2HSV)
    H, W, _ = orig_pic.shape

    # if speed_up <= 1:
    #     orig_pic = cv2.medianBlur(orig_pic, 7)
    # ftrs = image_to_features(orig_pic, include_pos=True, pos_weight=50)
    posx_w = 15  # 10-15
    posy_w = 55  # 70-55
    KN = 4  # 4-5-6
    ft_weights = [0.2, 1, 0.4, posy_w, posx_w]

    "DOWNSCALE PICTURE"
    if speed_up > 1:
        h, w = np.round([H / speed_up, W / speed_up]).astype(int)
        train_pic = cv2.resize(orig_pic, (w, h))
        ksize = 5
        train_pic = cv2.medianBlur(train_pic, ksize)
    else:
        speed_up = 1
        train_pic = orig_pic

    # return train_pic

    "GET PICTURE FEATURES"
    if clip_hud:
        sly, slx = foi_no_hud.get_slices(train_pic)
        ftrs, clip = image_to_features(train_pic, include_pos=include_pos,
                                       # posx_weight=posx_w, posy_weight=posy_w,
                                       clip_y=sly, clip_x=slx,
                                       )
        train_pic = clip
    else:
        train_pic = mask_hud(train_pic)
        ftrs, _ = image_to_features(train_pic, include_pos=include_pos,
                                    # posx_weight=posx_w, posy_weight=posy_w
                                    )

    if ft_weights:
        ftrs *= ft_weights
    # if speed_up > 1:
    #     ftrs *= np.array([1, 1, 1, speed_up, speed_up])
    # if clip_hud:
    #     ftrs *= np.array([1, 1, 1, H / sly.stop, W / slx.stop])
    #     ftrs += np.array([0, 0, 0, (sly.stop - sly.start) / H, (slx.stop - slx.start) / W])

    t0 = time.time()
    ms = KMeans(KN)
    pts = np.array([
            [0.1, 0.5],
            [0.5, 0.1],
            [0.9, 0.5],
            [0.5, 0.9]
    ])
    h, w = train_pic.shape[:2]
    pts = (pts * [h, w]).round().astype(int)
    indy, indx = pts.T
    first_centrs = train_pic[indy, indx].reshape(4, -1)
    print(pts)
    print("Centers:")
    print(first_centrs)
    first_centrs = np.hstack([first_centrs, indy.T / h, indx.T / w])
    print("Shape")
    print(first_centrs.shape)
    # ms.cluster_centers_ = first_centrs

    ret = ms.fit(ftrs)
    tfit = time.time() - t0

    cluster_colors = ret.cluster_centers_[:, :3]

    if clip_hud or speed_up > 1:
        # if clip_hud:
        #     pred_pic = foi_no_hud.get_foi(orig_pic)
        # else:
        #     pred_pic = mask_hud(orig_pic)
        pred_pic = mask_hud(orig_pic)
        ftrs, _ = image_to_features(pred_pic, include_pos=include_pos,
                                    # posx_weight=posx_w, posy_weight=posy_w
                                    )
        if ft_weights:
            ftrs *= ft_weights
        t0 = time.time()
        labs = ms.predict(ftrs)
        tpredict = time.time() - t0
        hout, wout = pred_pic.shape[:2]
    else:
        labs = ret.labels_
        tpredict = None
        hout, wout = H, W

    print(f"Kmeans fit time: {time_formatter(tfit)}, pred time: {time_formatter(tpredict)}")

    color_ftrs = ftrs[:, :3]

    unq_lb = np.unique(labs)
    cluster_colors = cluster_colors[:len(unq_lb)]
    if ft_weights:
        cluster_colors = cluster_colors / ft_weights[:3]

    for lb in unq_lb:
        mask = labs == lb
        color = cluster_colors[lb]
        color_ftrs[mask] = color

    img = image_from_features(color_ftrs, hout, wout)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # img = np.tile(img[:, :, 1].reshape(h, w, 1), [1, 1, 3])

    # if clip_hud:
    #     img = foi_no_hud.get_foi(img)

    hout = img.shape[0]

    cluster_bar = np.zeros((hout, 20, 3), dtype=np.uint8)
    step = hout / len(cluster_colors)
    for i, cl in enumerate(cluster_colors):
        ind1 = int(i * step)
        ind2 = int(i * step + step)
        cluster_bar[ind1:ind2, :] = cl
    cluster_bar[:, 0] = [0, 0, 0]
    cluster_bar[:, 1] = [255, 255, 255]

    img = np.hstack([img, cluster_bar])

    if use_hsv:
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    # plt.imshow(img)
    # plt.show()
    return img


if __name__ == "__main__":
    t0 = time.time()
    pool = mpc.Pool(4)
    # pool = None

    # multi_picture_export(
    #         PICS_FRANKFURT_TRAFFIC, subfolder="traffic-gradient",
    #         function=road_segmentize,
    #         pool=pool,
    #         # clip_final_pic=foi_no_hud.get_foi,
    #         loop_start=35,
    #         # loop_lim=36,
    #         loop_lim=40,
    #         # loop_lim=50,
    #         # loop_lim=100,
    #         # loop_lim=300,
    #         # clip_hud=False,
    #         clip_hud=True,
    # )
    pic = cv2.imread(SOURCE_FOLDER + os.path.join(["src_images", "zima.jpg"]))
    img = road_segmentize(pic)
    # multi_picture_export(
    #         PICS_FRANKFURT_TRAFFIC, subfolder="traffic-gradient",
    #         function=road_segmentize,
    #         pool=pool,
    #         # clip_final_pic=foi_no_hud.get_foi,
    #         loop_start=55,
    #         loop_lim=60,
    #         # clip_hud=False,
    #         # clip_hud=True,
    # )
    # multi_picture_export(
    #         PICS_FRANKFURT_TRAFFIC, subfolder="traffic-gradient",
    #         function=road_segmentize,
    #         pool=pool,
    #         clip_final_pic=foi_no_hud.get_foi,
    #         loop_start=110,
    #         loop_lim=120,
    # )

    # plot_histogram()
    # check_mean_shift_of_hisogram()
    tend = time.time()
    print(f"Whole script finished in: {time_formatter(tend - t0)}")
