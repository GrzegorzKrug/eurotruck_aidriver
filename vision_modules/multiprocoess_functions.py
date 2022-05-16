import matplotlib.pyplot as plt
import time
import cv2
import os

from picture_paths import CABIN_PICS_1, CABIN_PICS_2, PIC_OUTPUT_FOLDER
import numpy as np
import imutils


# import multiprocessing as mpc

# POOL = mpc.Pool(10)


def time_formatter(val, prec=2):
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


@timedecorator
def multi_picture_export(pic_list, subfolder=None, prefix=None, postfix=None,
        overwrite=False,
        function=None, matplot_f=None,
        clip_f=None,
        pool=None,
        **kw):
    """
    Copy picture to new location. Can apply function to change picture or generate matplot figures.
    Args:
        pic_list:
        subfolder:
        prefix:
        overwrite:
        function:
        matplot_f: Generate matplotlib plot
        **kw:

    Returns:

    """

    "Init strings needed for destination"
    if subfolder:
        subfolder = f"{subfolder}{os.path.sep}"
        os.makedirs(PIC_OUTPUT_FOLDER + subfolder, exist_ok=True)
    if prefix is None:
        prefix = ""
    else:
        prefix = f"{prefix}_"
    if postfix is None:
        postfix = ""
    else:
        postfix = f"_{postfix}"

    "Iterate over list"
    all_args = []
    all_kw = []
    all_process = []
    # print(f"Pool: {pool}")

    for ph in pic_list:
        name = os.path.basename(ph)
        *name, ext = name.split('.')
        name = ' '.join(name)

        single_args = (
                subfolder, prefix, name, postfix, overwrite,
                ph, ext,
                function, clip_f, matplot_f
        )

        if pool is None:
            # print("Pol is none")
            _multi_picture_thread_executor(*single_args, **kw)
        else:
            all_args.append([single_args, kw])

            # proc = mpc.Process(target=_multi_picture_thread, args=single_args, kwargs=kw)
            # all_process.append(proc)
            # proc.start()
            # if len(all_process) > pool_size:
            #     [p.join() for p in all_process]
            #     all_process = []

            # print(proc.is_alive())
            # proc.join()
            # return None
    if pool is not None:
        pool.map(_multi_picture_thread, all_args)


def _multi_picture_thread(a):
    """Purpose to unpack arguments"""
    a, kw = a[0], a[1]
    return _multi_picture_thread_executor(*a, **kw)


def _multi_picture_thread_executor(
        subfolder, prefix, name, postfix, overwrite,
        ph, ext,
        function=None, clip_f=None,
        matplot_f=None,
        **kw,
):
    MAX_ORIG_SIZE = 1000
    MIN_ORIG_SIZE = 1000

    dst = f"{PIC_OUTPUT_FOLDER}{subfolder}{prefix}{name}{postfix}.{ext}"
    img_full = cv2.imread(ph, cv2.IMREAD_COLOR)
    if clip_f:
        img_input = clip_f(img_full)
    else:
        img_input = img_full

    if function:
        "Change image array"
        function_im = function(img_input.copy(), **kw)
    else:
        function_im = None

    if matplot_f:
        "Generate plot in function"
        fig = plt.figure(figsize=(10, 7))
        matplot_f(img_input.copy(), **kw)
        plt.tight_layout()
        fig.canvas.draw()
        matplot_im = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        matplot_im = cv2.cvtColor(matplot_im, cv2.COLOR_RGBA2RGB)
        plt.close()
    else:
        matplot_im = None

    if function is None and matplot_f is None and not os.path.isfile(dst):
        "Just copy"
        cv2.imwrite(dst, img_input)

    elif overwrite:
        "Just copy and overwrite"
        cv2.imwrite(dst, img_input)

    "STACKING"
    stk = [st for st in [img_input, function_im, matplot_im] if st is not None]
    H, W, C = [*zip(*[st.shape for st in stk])]
    hmin = min(H)
    wmin = min(W)

    if hmin < wmin:
        arg = np.argmin(H)
        stacked_image = stk.pop(0)
        # stacked_image = imutils.resize(stacked_image, height=500)
    else:
        arg = np.argmin(W)
        stacked_image = stk.pop(0)
        # stacked_image = imutils.resize(stacked_image, width=500)

    h, w, _ = stacked_image.shape
    if h > w and h > MAX_ORIG_SIZE:
        stacked_image = imutils.resize(stacked_image, height=MAX_ORIG_SIZE)
    elif w > h and w > MAX_ORIG_SIZE:
        stacked_image = imutils.resize(stacked_image, width=MAX_ORIG_SIZE)
    elif w >= h and w < MIN_ORIG_SIZE:
        stacked_image = imutils.resize(stacked_image, width=MIN_ORIG_SIZE)
    elif w <= h and h < MIN_ORIG_SIZE:
        stacked_image = imutils.resize(stacked_image, height=MIN_ORIG_SIZE)

    for im in stk:
        sh, sw, c = stacked_image.shape
        h, w, c = im.shape
        ratio = h / w

        if ratio > 1.5:
            kw = {'height': sh}
            axis = 1
            sep = np.zeros((sh, 2, 3))
            sep[:, 1] = [255, 255, 255]

        elif ratio < 0.5:
            kw = {'width': sw}
            axis = 0
            sep = np.zeros((2, sw, 3))
            sep[1, :] = [255, 255, 255]
        elif sh > sw:
            kw = {'height': sh}
            axis = 1
            sep = np.zeros((sh, 2, 3))
            sep[:, 1] = [255, 255, 255]
        else:
            kw = {'width': sw}
            axis = 0
            sep = np.zeros((2, sw, 3))
            sep[1, :] = [255, 255, 255]

        im = imutils.resize(im, **kw)
        # print(stacked_image.shape, im.shape, axis)
        stacked_image = np.concatenate([stacked_image, sep, im], axis=axis)

    cv2.imwrite(dst, stacked_image)


__all__ = ['multi_picture_export', 'time_formatter', 'timedecorator']

if __name__ == "__main__":
    print("Main")
    # print(PICS_FOLDER)
    # print(ALL_HD)
    # print(CABIN_PICS_1)
    multi_picture_export(CABIN_PICS_1, subfolder="filter_1")
    multi_picture_export(CABIN_PICS_2, subfolder="filter_2", postfix="plot")
