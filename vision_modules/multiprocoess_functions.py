import matplotlib.pyplot as plt
import time
import cv2
import os

from picture_paths import CABIN_PICS_1, CABIN_PICS_2, PIC_OUTPUT_FOLDER
import numpy as np
import imutils

from utility import timedecorator


# import multiprocessing as mpc

# POOL = mpc.Pool(10)


@timedecorator
def multi_picture_export(pic_list, subfolder=None, prefix=None, postfix=None,
        overwrite=False, attach_original=True,
        function=None, matplot_f=None,
        clip_final_pic=None,
        pool=None,
        loop_start=None,
        loop_lim=None,
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

    for i, ph in enumerate(pic_list):
        name = os.path.basename(ph)
        *name, ext = name.split('.')
        name = ' '.join(name)

        single_args = (
                subfolder, prefix, name, postfix, overwrite,
                ph, ext,
                attach_original,
                function, clip_final_pic, matplot_f
        )

        if loop_start and i < loop_start:
            continue
        if loop_lim and i >= loop_lim:
            break
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
        attach_original=True,
        function=None,
        clip_out_pic=None,
        matplot_f=None,
        function_im_add_axes=False,
        **kw,
):
    MAX_ORIG_SIZE = 1000
    MIN_ORIG_SIZE = 1000

    dst = f"{PIC_OUTPUT_FOLDER}{subfolder}{prefix}{name}{postfix}.{ext}"
    img_input_bgr = cv2.imread(ph, cv2.IMREAD_COLOR)
    # print(name)

    if function:
        "Change image array"
        function_im = function(img_input_bgr.copy(), name=name, **kw)
        if function_im_add_axes:
            fig = plt.figure()
            plt.imshow(function_im)
            plt.tight_layout()
            plt.subplots_adjust(top=0.95, bottom=0.05)
            fig.canvas.draw()
            function_im = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            function_im = cv2.cvtColor(function_im, cv2.COLOR_RGBA2RGB)
            plt.close()
    else:
        function_im = None

    if matplot_f:
        "Generate plot in function"
        fig = plt.figure(figsize=(10, 7))
        matplot_f(img_input_bgr.copy(), **kw)
        plt.tight_layout()
        plt.legend(loc='best')
        fig.canvas.draw()
        matplot_im = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        matplot_im = cv2.cvtColor(matplot_im, cv2.COLOR_RGBA2BGR)
        plt.close()
    else:
        matplot_im = None

    if function is None and matplot_f is None and not os.path.isfile(dst):
        "Just copy"
        cv2.imwrite(dst, img_input_bgr)

    elif overwrite:
        "Just copy and overwrite"
        cv2.imwrite(dst, img_input_bgr)

    "STACKING"

    if clip_out_pic and function_im is not None:
        img_out = clip_out_pic(function_im)
    else:
        img_out = function_im

    if attach_original and clip_out_pic:
        img_input = clip_out_pic(img_input_bgr)
    elif attach_original:
        img_input = img_input_bgr
        pass
    else:
        img_input = None

    stk = [st for st in [img_input, img_out, matplot_im] if st is not None]
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

        if ratio > 1.6:
            kw = {'height': sh}
            axis = 1
            sep = np.zeros((sh, 2, 3))
            sep[:, 1] = [255, 255, 255]
        elif ratio < 0.6:
            kw = {'width': sw}
            axis = 0
            sep = np.zeros((2, sw, 3))
            sep[1, :] = [255, 255, 255]
        # elif sw > sh:
        #     kw = {'width': sw}
        #     axis = 0
        #     sep = np.zeros((2, sw, 3))
        #     sep[1, :] = [255, 255, 255]
        else:
            kw = {'height': sh}
            axis = 1
            sep = np.zeros((sh, 2, 3))
            sep[:, 1] = [255, 255, 255]

        im = imutils.resize(im, **kw)
        # print(stacked_image.shape, im.shape, axis)
        stacked_image = np.concatenate([stacked_image, sep, im], axis=axis)

    cv2.imwrite(dst, stacked_image)


__all__ = ['multi_picture_export', ]

if __name__ == "__main__":
    print("Main")
    # print(PICS_FOLDER)
    # print(ALL_HD)
    # print(CABIN_PICS_1)
    multi_picture_export(CABIN_PICS_1, subfolder="filter_1")
    multi_picture_export(CABIN_PICS_2, subfolder="filter_2", postfix="plot")
