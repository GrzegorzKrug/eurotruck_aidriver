import matplotlib.pyplot as plt
import time
import cv2
import os

from picture_paths import CABIN_PICS_1, CABIN_PICS_2, OUTPUT_FOLDER


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
        os.makedirs(OUTPUT_FOLDER + subfolder, exist_ok=True)
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
        name, ext = name.split('.')
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
    # print("unpacking")
    # print(a)
    a, kw = a[0], a[1]
    return _multi_picture_thread_executor(*a, **kw)


def _multi_picture_thread_executor(
        subfolder, prefix, name, postfix, overwrite,
        ph, ext,
        function=None, clip_f=None,
        matplot_f=None,
        **kw,
):
    dst = f"{OUTPUT_FOLDER}{subfolder}{prefix}{name}{postfix}.{ext}"
    if function:
        "Change image array"
        img = cv2.imread(ph, cv2.IMREAD_COLOR)
        if clip_f:
            img = clip_f(img)
        out = function(img, **kw)
        cv2.imwrite(dst, out)

    elif matplot_f:
        "Generate plot in function"
        img = cv2.imread(ph, cv2.IMREAD_COLOR)
        plt.figure(figsize=(10, 7))
        if clip_f:
            img = clip_f(img)
        matplot_f(img, **kw)
        plt.tight_layout()
        plt.savefig(dst)
        plt.close()

    elif not os.path.isfile(dst):
        "Just copy"
        img = cv2.imread(ph, cv2.IMREAD_COLOR)
        if clip_f:
            img = clip_f(img)
        cv2.imwrite(dst, img)

    elif overwrite:
        "Just copy and overwrite"
        img = cv2.imread(ph, cv2.IMREAD_COLOR)
        if clip_f:
            img = clip_f(img)
        cv2.imwrite(dst, img)


if __name__ == "__main__":
    print("Main")
    # print(PICS_FOLDER)
    # print(ALL_HD)
    # print(CABIN_PICS_1)
    multi_picture_export(CABIN_PICS_1, subfolder="filter_1")
    multi_picture_export(CABIN_PICS_2, subfolder="filter_2", postfix="plot")
