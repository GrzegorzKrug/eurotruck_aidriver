from PIL import ImageGrab, ImageFilter

import win32api as wapi
import win32con as wcon

import pyautogui
import ctypes
# import my_mouse_control
import numpy as np
import time
import cv2
import os

from my_mouse_control import Mouse
from my_timer import timeit_mean, single_time

INTERVAL = 500  # MS
FALL_DELAY = 0.1  # S
INITIAL_MOUSE_POS = pyautogui.position()
GAMES = 100


def old_click():
    ctypes.windll.user32.mouse_event(Mouse.MOUSEEVENTF_LEFTDOWN)
    time.sleep(0.005)
    ctypes.windll.user32.mouse_event(Mouse.MOUSEEVENTF_LEFTUP)


class Window:
    "Class for managing active window"

    def __init__(self, name, match_name='any'):
        self.name = name
        self.math_name = match_name

        window = self.find_window(self.name)
        self.box = window.box
        self.box_points = [window.left, window.top, window.left + window.width,
                           window.top + window.height]
        # print(dir(window))
        # print(window.box)
        # print(window.width)
        # window.activate()

    @timeit_mean(30)
    def grab_frame(self):
        screen = ImageGrab.grab(self.box_points, all_screens=True)
        return screen
        # screen.save(fp="screen.png",format='png')

    @staticmethod
    def find_window(name):
        winds = pyautogui.getWindowsWithTitle(name)
        for wnd in winds:
            # print(wnd.title)
            if name.lower() in wnd.title.lower():
                return wnd
        return None


for x in range(10):
    w = Window("greg")
    fr = w.grab_frame()
    fr = np.array(fr, dtype=np.uint8)
    # print(fr)
    cv2.imshow("frame", fr)
    # cv2.waitKey(1000)
    # time.sleep(0.1)
