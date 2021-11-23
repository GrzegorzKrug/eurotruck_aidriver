from PIL import ImageGrab, ImageFilter

import win32api as wapi
import win32con as wcon
import win32gui as wgui
from win32com import client
# from win32con import client

# import win32con as wcon
# import win32con.client as comclt

import pyautogui
import imutils
import ctypes
# import my_mouse_control
import numpy as np
import time
import mss
import cv2
import os

from my_mouse_controll import Mouse
from my_timer import timeit_mean, single_time

INTERVAL = 500  # MS
FALL_DELAY = 0.1  # S
INITIAL_MOUSE_POS = pyautogui.position()
GAMES = 100


def old_click():
    ctypes.windll.user32.mouse_event(Mouse.MOUSEEVENTF_LEFTDOWN)
    time.sleep(0.005)
    ctypes.windll.user32.mouse_event(Mouse.MOUSEEVENTF_LEFTUP)


# from functools import

class Window:
    "Class for managing active window"

    def __init__(self, name, match_name='any'):
        self.name = name
        self.math_name = match_name

        self.window = self.find_window(self.name)
        self.box = None
        self.props = None
        self.box_points = None

        "Initialize mss for capturing"
        self.mss = mss.mss()  # Capturing object

        self.get_window_location()

    def focus(self):
        self.window.activate()
        if self.window.isMinimized:
            self.window.restore()
            time.sleep(0.5)
        self.get_window_location()

    def get_window_location(self):
        """Reads current window location"""
        "Offsets for windows apps: notepad, paint etc."
        off_l = 7
        off_wid = -off_l - 7
        off_hei = -7

        window = self.window
        self.box = window.box
        self.props = {
                'left': window.left + off_l, "top": window.top,
                'width': window.width + off_wid, 'height': window.height + off_hei,
        }
        self.box_points = [window.left, window.top + 1, window.left + window.width,
                           window.top + window.height]

    # @timeit_mean(30)
    def grab_frame(self):
        # screen = ImageGrab.grab(self.box_points, all_screens=True)
        # return screen
        # sr = pyautogui.screenshot()
        sct = self.mss
        im = sct.grab(self.props)
        return im
        # return sr
        # screen.save(fp="screen.png",format='png')

    @staticmethod
    def find_window(name):
        winds = pyautogui.getWindowsWithTitle(name)
        for wnd in winds:
            # print(wnd.title)
            if name.lower() in wnd.title.lower():
                return wnd
        return None


if __name__ == "__main__":
    for x in range(1):
        w = Window("notepad")
        fr = w.grab_frame()
        fr = np.array(fr, dtype=np.uint8)
        # fr = imutils.resize(fr, width=800)
        # print(fr)
        # client()

        # print(w.window.title)
        # w.focus()
        print(dir(w.window))

        window_id = wgui.FindWindow(None, w.window.title)
        # print(dir(client))
        # wgui.PostMessage(window_id, ord('5'))
        # client.GetObject(window_id)
        # client.Dispatch()

        cv2.imshow("frame", fr)
        cv2.imwrite("screen.png", fr)
        # cv2.waitKey(0)
        # time.sleep(0.1)

        # w.window.minimize()
