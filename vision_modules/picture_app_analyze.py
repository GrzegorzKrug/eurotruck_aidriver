import matplotlib.pyplot as plt
import tkinter as tk
import imutils
import numpy as np
import cv2

import tkinter.filedialog

from scipy.signal import convolve
from tkinter import ttk
from PIL import Image, ImageTk
from frame_foi import foi_window_view


def hist_3(arr):
    rg = 0, 255
    his_r, lev = np.histogram(arr[:, :, 0], 255, range=rg)
    his_g, lev = np.histogram(arr[:, :, 1], 255, range=rg)
    his_b, lev = np.histogram(arr[:, :, 2], 255, range=rg)
    return his_r, his_g, his_b, lev


def get_smooth_hist(pic_rgb, smooth=10):
    his_r, his_g, his_b, levels = hist_3(pic_rgb)
    if smooth >= 2:
        if smooth % 2:
            smooth += 1
        his_b = convolve(his_b, np.ones(smooth), 'same') / smooth
        his_g = convolve(his_g, np.ones(smooth), 'same') / smooth
        his_r = convolve(his_r, np.ones(smooth), 'same') / smooth

    return his_r, his_g, his_b


def plot_smooth_histogram(pic_bgr, smooth=10, zoom=1):
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

        plt.grid()
        ymax = np.max([his_r, his_g, his_b]) * zoom + 5
        plt.ylim([0, ymax])


def image_downscale(fr, downsize):
    h, w, *c = fr.shape

    if h > w:
        fr = imutils.resize(fr, height=downsize)
    else:
        fr = imutils.resize(fr, width=downsize)
    # print(f"Size: {fr.shape}")
    return fr


def cv_image_to_tk(frame):
    frame = frame[:, :, [2, 1, 0]]
    im_tk = ImageTk.PhotoImage(Image.fromarray(frame))
    return im_tk


class UI:
    def __init__(self):
        self.dims = "1500x800"
        self.root = tk.Tk()
        self.root.geometry(self.dims)

        self.refs = {}
        self.vars = {}

        blank = np.zeros((50, 50, 3), dtype=np.uint8)
        self.blank = cv_image_to_tk(blank)
        blank = None

        fr = tk.Frame(self.root)
        fr.pack(side='left')
        self.display = tk.Label(fr, image=blank, )
        self.preview = tk.Label(fr, image=blank, )
        self.display.pack(side='top')
        self.preview.pack(side='top')

        self.picture = None
        self.MAX_SIZE = 700

    def add_button(self, f, text=None, parent_frame=None, side="bottom"):
        fr = parent_frame if parent_frame is not None else self.root
        bt = tk.Button(fr, text=text, command=f)
        bt.pack(side=side)

    def label_decor(f):
        def wrapper(*a, label=None, label_pos='left', side="top", parent_frame=None, **kw):
            if label:
                inner = tk.Frame(parent_frame)
                inner.pack(side=side)
                # print(f"Adding inner frame: {side}")

                lb = tk.Label(inner, text=label)
                lb.pack(side=label_pos, pady=0, padx=10)
                if label_pos:
                    side = label_pos

                parent_frame = inner

            return f(*a, side=side, parent_frame=parent_frame, **kw)

        return wrapper

    @label_decor
    def add_check_box(self, parent_frame, side='top', key=None, value=None, command=None):
        fr = parent_frame if parent_frame is not None else self.root
        var = tk.BooleanVar(value)
        chk = tk.Checkbutton(fr, var=var, command=command)
        chk.pack(side=side)
        self.refs[key] = var

    @label_decor
    def add_slider(self, parent_frame=None, side="top", from_=0, to=255, key=None, value=None,
            command=None):
        fr = parent_frame if parent_frame is not None else self.root

        var = tk.IntVar(value=value)

        slider = tk.Scale(
                fr,
                from_=from_,
                to=to,
                orient='horizontal',  # horizontal
                var=var,
                length=500,
                command=command,
        )
        slider.pack(side=side)
        if key:
            if key in self.refs:
                print(f"Overwriting ref: {key}")
            self.refs[key] = var
        return slider

    def add_frame(self, parent_frame=None, side='left', key=None):
        fr = parent_frame if parent_frame is not None else self.root
        fr = tk.Frame(fr)
        fr.pack(side=side)

        if key:
            self.refs[key] = fr

        return fr

    def start(self):
        self.root.mainloop()

    @staticmethod
    def update_photo(box, photo):
        """

        Args:
            box:
            photo: TK image

        Returns:

        """
        if photo is None:
            box.configure(image="")
        else:
            box.configure(image=photo)
            box.image = photo

    def load_picture(self):
        ret = tk.filedialog.askopenfilename()
        if ret:
            print(f"Loading: {ret}")
            self.root.title(ret)
            picture = cv2.imread(ret, cv2.IMREAD_COLOR)
        else:
            return None

        self.picture = picture

        frame = image_downscale(picture, self.MAX_SIZE)

        tk_im = cv_image_to_tk(frame)
        self.update_photo(self.display, tk_im)

        # frame = 255 - frame
        # tk_im = cv_image_to_tk(frame)
        # self.update_photo(self.preview, tk_im)

        self.root.geometry(self.dims)

    def mask_picture(self, *a):
        """BGR FORMAT"""
        pic = self.picture.copy()
        clip_to_window = bool(self.refs['cliptowindow'].get())
        if clip_to_window:
            pic = foi_window_view.get_foi(pic)
        pic = image_downscale(pic, self.MAX_SIZE)

        rmin = self.refs['rmin'].get()
        rmax = self.refs['rmax'].get()
        gmin = self.refs['gmin'].get()
        gmax = self.refs['gmax'].get()
        bmin = self.refs['bmin'].get()
        bmax = self.refs['bmax'].get()

        rmask = (rmin <= pic[:, :, 2]) & (rmax >= pic[:, :, 2])
        gmask = (gmin <= pic[:, :, 1]) & (gmax >= pic[:, :, 1])
        bmask = (bmin <= pic[:, :, 0]) & (bmax >= pic[:, :, 0])
        mask = rmask & gmask & bmask

        pic = np.zeros_like(pic)
        pic[mask, :] = 255

        tkimg = cv_image_to_tk(pic)
        self.update_photo(self.preview, tkimg)

    def plot_roi_hist(self, *a):
        # print(f"Args: {a}")

        pic = self.picture.copy()

        h, w, c = pic.shape

        halfx = self.refs['roiwidth'].get() // 2
        halfy = self.refs['roiheight'].get() // 2
        x = self.refs['roix'].get()
        y = self.refs['roiy'].get()

        x1 = 0 if x - halfx < 0 else x - halfx
        x2 = w if x + halfx >= w else x + halfx

        y1 = 0 if y - halfy < 0 else y - halfy
        y2 = h if y + halfy >= h else y + halfy

        roi = pic[y1:y2, x1:x2].copy()
        alfa = 0.6

        blank = np.zeros_like(roi) + (250, 150, 0)
        blank = blank.astype(np.uint8)

        highlight = cv2.addWeighted(roi, 1 - alfa, blank, alfa, 0)
        pic[y1:y2, x1:x2] = highlight

        clip_to_window = bool(self.refs['cliptowindow'].get())
        if clip_to_window:
            pic = foi_window_view.get_foi(pic)
        pic = image_downscale(pic, self.MAX_SIZE)
        tkimage = cv_image_to_tk(pic)
        self.update_photo(self.display, tkimage)

        fig = plt.figure(figsize=(12, 8))
        zoom = self.refs['histzoom'].get()
        plot_smooth_histogram(roi, smooth=0, zoom=zoom / 100)

        plt.tight_layout()
        fig.canvas.draw()
        matplot_im = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        matplot_im = cv2.cvtColor(matplot_im, cv2.COLOR_RGBA2BGR)
        # matplot_im = cv2.cvtColor(matplot_im, cv2.COLOR_BGRA2BGR)
        matplot_im = image_downscale(matplot_im, self.MAX_SIZE)

        hist_tk = cv_image_to_tk(matplot_im)
        self.update_photo(self.hist, hist_tk)
        plt.close()

        roi = imutils.resize(roi, height=300)
        roi = image_downscale(roi, 500)
        roi_tk = cv_image_to_tk(roi)
        self.update_photo(self.hist_prev, roi_tk)

        # plt.imshow(roi)
        # plt.show()

    def update_both(self, *a):
        self.mask_picture()
        self.plot_roi_hist()


if __name__ == "__main__":
    app = UI()

    sliders_fr = app.add_frame(side='right')

    app.add_button(app.update_both, parent_frame=sliders_fr, text="Update", side='bottom')
    app.add_button(app.load_picture, parent_frame=sliders_fr, text="Load photo", side='bottom')

    "RGB MASK"
    app.add_slider(parent_frame=sliders_fr, from_=0, to=255,
                   label="R Min", key='rmin', command=app.mask_picture)
    app.add_slider(parent_frame=sliders_fr, from_=0, to=255,
                   label="R Max", key='rmax', value=255, command=app.mask_picture)
    app.add_slider(parent_frame=sliders_fr, from_=0, to=255,
                   label="G Min", key='gmin', command=app.mask_picture)
    app.add_slider(parent_frame=sliders_fr, from_=0, to=255,
                   label="G Max", key='gmax', value=255, command=app.mask_picture)
    app.add_slider(parent_frame=sliders_fr, from_=0, to=255,
                   label="B Min", key='bmin', command=app.mask_picture)
    app.add_slider(parent_frame=sliders_fr, from_=0, to=255,
                   label="B Max", key='bmax', value=255, command=app.mask_picture)

    sep = ttk.Separator(sliders_fr, orient='horizontal', style='TSeparator', )
    sep.pack(side='top', fill="x", padx=10, pady=20)

    "REGION CLIP STUFF"
    app.add_slider(parent_frame=sliders_fr, from_=0, to=2000,
                   label="X pos", key='roix', value=255, command=app.plot_roi_hist)
    app.add_slider(parent_frame=sliders_fr, from_=0, to=2000,
                   label="Y pos", key='roiy', value=255, command=app.plot_roi_hist)
    app.add_slider(parent_frame=sliders_fr, from_=0, to=300,
                   label="Width", key='roiwidth', value=55, command=app.plot_roi_hist)
    app.add_slider(parent_frame=sliders_fr, from_=0, to=300,
                   label="Height", key='roiheight', value=55, command=app.plot_roi_hist)
    app.add_slider(parent_frame=sliders_fr, from_=1, to=100,
                   label="Zoom", key='histzoom', value=100, command=app.plot_roi_hist)

    fr = app.add_frame()
    app.hist = tk.Label(fr, image=app.blank)
    app.hist.pack(side='bottom')
    app.hist_prev = tk.Label(fr, image=app.blank)
    app.hist_prev.pack(side='top')
    fr.pack(side='left')

    app.add_check_box(parent_frame=sliders_fr, label="Clip to Window",
                      command=app.update_both, key='cliptowindow')

    app.start()
