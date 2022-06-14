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
     COLORS_PATHS, PICS_AUTOSTRADA,
    PICS_FRANKFURT, PICS_FRANKFURT_TRAFFIC,
)
from multiprocoess_functions import *
from frame_foi import (
    foi_roadsample, foi_window_view, foi_roadsample_low,
    foi_frontvision, foi_frontsample, foi_map, foi_road_model,
    foi_mirror_left, foi_mirror_right,
)

from hough_module import draw_hough_lines


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


def rolling_smooth(arr, smooth=0, smooth_f="mean"):
    if smooth >= 2:
        if smooth % 2:
            smooth += 1

        if smooth_f == 'mean':
            arr = convolve(arr, np.ones(smooth), 'same') / smooth

        elif smooth_f == 'median':
            arr = fast_rolling_median(arr, smooth=smooth)

        elif smooth_f == 'max':
            arr = fast_rolling_max(arr, smooth=smooth)
    return arr


CROSS_KERNEL = np.ones((3, 3))


def pic_gray_to3d(out):
    out = out[:, :, np.newaxis]
    out = np.tile(out, [1, 1, 3])
    return out


def mask_hud(arr, expand_mask=0):
    arr = arr.copy()
    mask1 = foi_map.get_mask(arr)
    mask2 = foi_mirror_left.get_mask(arr)
    mask3 = foi_mirror_right.get_mask(arr)
    mask = mask1 | mask2 | mask3

    if expand_mask > 0:
        mask = mask.astype(np.uint8) * 255
        mask = cv2.dilate(mask, CROSS_KERNEL, iterations=expand_mask).astype(bool)

    arr[mask] = 0
    return arr


def time_formatter(val, prec=2):
    if val is None:
        return "None"

    if val < 0.001:
        return f"{val * 1_000_000:>6.{prec}f} us"
    elif val < 1:
        return f"{val * 1_000:>6.{prec}f} ms"
    elif val < 60:
        return f"{val:>6.{prec}f} s"
    elif val < 3600:
        return f"{val / 60:>6.{prec}f} m"
    else:
        return f"{val / 60 / 60:>6.{prec}f} h"


def timedecorator(f):
    def wrapper(*a, **kw):
        t0 = time.time()
        res = f(*a, **kw)
        tend = time.time()
        dur = tend - t0
        print(f"{f.__name__:<25} was executed in {time_formatter(dur)}")
        return res

    return wrapper


def image_to_features(arr,
        include_pos=False, posx_weight=1, posy_weight=1,
        norm_keys=True,
        clip_x=None,
        clip_y=None,
):
    """

    Args:
        arr: image array
        include_pos: Include pixel position in features?
        posx_weight: weight of position
        posy_weight: weight of position
        norm_keys: (default True) Scale position keys between 0:weight.
        clip_x: pass if you want get features from region
        clip_y: pass if you want to get features from specific region

    Returns:
        features(pixels, features), copy of image / clipped image

    """
    HFULL, WFULL, *_ = arr.shape
    if clip_y:
        arr = arr[clip_y, :]
    if clip_x:
        arr = arr[:, clip_x]

    arr = arr.copy()
    h, w, *_ = arr.shape

    y, x = np.ogrid[:h, :w]
    xx, yy = np.meshgrid(x, y)
    keys = np.stack([yy, xx], axis=-1).reshape(-1, 2).T
    translation_key = tuple(map(tuple, keys))

    out = arr[translation_key]

    if include_pos:
        mx = np.max(keys, axis=1).reshape(2, -1)
        keys = keys.astype(np.float32)

        if clip_x is not None or clip_y is not None:
            # print("Clipping")
            ofx = clip_x.start if clip_x is not None else 0
            ofy = clip_y.start if clip_y is not None else 0

            keys /= np.array([[HFULL - 1, WFULL - 1]]).T
            offs = ((ofy / (HFULL - 1)), ofx / (WFULL - 1))
            keys += np.array([offs]).T

        elif norm_keys:
            keys = keys / mx

        scale = np.array([[posy_weight, posx_weight]])
        # print(keys.shape)
        out = np.concatenate([out, keys.T * scale], axis=1)

    return out, arr


def image_from_features(ftrs, h, w):
    # ft_n = ftrs.shape[1]
    ft_img = ftrs[:, :3]

    # y, x = np.ogrid[:h, :w]
    # xx, yy = np.meshgrid(x, y)
    # keys = np.stack([yy, xx], axis=-1).reshape(-1, 2).T
    # translation_key = tuple(map(tuple, keys))

    # img = np.zeros((h, w, 3), dtype=np.int32)
    # img[translation_key] = ft_img

    img = ft_img.reshape(h, w, 3)
    # img = img.astype(np.uint8)
    img = img.astype(np.uint8)
    return img


DEFINED_COLORS = [
        # R  G, B"
        (0, 0.6, 0),  # green
        (0.6, 0.1, 0.7),  # purple
        (0, 0.2, 0.7),  # dark blue
        (0.9, 0.6, 0.2),  # orange
        (0.2, 0.3, 0.5),  # dark cyan
        (0.1, 0.7, 0.7),  # cyan
        (0.9, 0.9, 0),  # yellow
        (0.2, 0, 0),
]

DEFINED_COLORS = [(np.array(c) * 255).round().astype(np.uint8) for c in DEFINED_COLORS]
DEFINED_COLORS = np.array(DEFINED_COLORS)

if __name__ == "__main__":
    y, x = 5, 2
    ar = np.arange(y * x * 3).reshape(y, x, 3) % 27
    print("Array")
    print(ar)

    np.set_printoptions(suppress=True)
    H, W, _ = ar.shape
    ftr = image_to_features(ar, include_pos=True, posy_weight=3, posx_weight=1)
    print("True keys")
    print(ftr)

    # ar = ar[1:4, 1:]
    print("Clip keys")
    ftr, _ = image_to_features(ar, include_pos=True,
                               posy_weight=4, posx_weight=1,
                               clip_y=slice(1, 4, None),
                               clip_x=slice(0, None, None),
                               )
    print(ftr)
