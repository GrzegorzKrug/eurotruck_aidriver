import matplotlib.pyplot as plt
import imutils
import numpy as np

import glob
import sys
import cv2
import os

from itertools import product, combinations

files = glob.glob(f"src_images{os.path.sep}*.png", recursive=True)

from sklearn.cluster import MeanShift, KMeans


def detector(frame):
    marked = frame
    return marked


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


def image_revert_from_features(arr, keys):
    # translation_key = tuple(map(tuple, keys))
    # temp = np.zeros_like(arr)
    # temp[translation_key] = arr
    return temp


for p in files:
    mat = cv2.imread(p, cv2.IMREAD_COLOR)
    mat = imutils.resize(mat, height=400)
    h, w, c = mat.shape
    km = KMeans(20)
    features = image_to_features(mat, pos_weight=1000)
    ret = km.fit(features)

    lbs = ret.labels_
    centroids = ret.cluster_centers_
    # print(ret.labels_.shape)
    # print(h*w)
    colors = centroids[:, 2:]
    colors = centroids
    temp = colors[ret.labels_]

    temp = temp.reshape(h, w, c)
    temp = np.round(temp).astype(np.uint8)

    # print(f"shape: {temp[:,:,0].shape}")
    # print(np.max(temp[:, :, 0], axis=0))
    # print(np.max(temp[:, :, 1], axis=0))
    # print(np.max(temp[:, :, 2], axis=0))

    # plt.imshow(temp)
    # plt.show()

    stack = np.concatenate([mat, temp], axis=1)
    # cv2.imshow("asd", temp)
    cv2.imshow("stacked", stack)
    cv2.waitKey()
    break
