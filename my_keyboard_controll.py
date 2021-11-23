import ctypes

from window_capture import Window

import win32api as wapi
import numpy as np
import time

import matplotlib.pyplot as plt


class Keyboard:
    """It simulates the mouse"""

    def _do_event(self, flags, data, extra_info):
        """generate a keyboard event"""
        return ctypes.windll.user32.keybd_event(flags, data, extra_info)

    def _get_button_value(self, button_name, ):
        """convert the name of the button into the corresponding value"""

        return buttons


win = Window('notepad')
win.focus()

kb = Keyboard()

t0 = time.time()
dur = 5
st = wapi.GetKeyboardState()
old_st = np.array([*st])

data = []
while True:
    # print(wapi.GetAsyncKeyState())

    st = wapi.GetKeyboardState()
    # st = np.array([*st])
    print(st)
    st = list(map(int, st))
    data.append(st)

    # diff = st - old_st
    # old_st = st
    # print(max(st))
    # print(max(diff))

    if wapi.GetAsyncKeyState(ord("Q")):
        print("Q")
        break

    if time.time() > (t0 + dur):
        print("Too long, stop...")
        break

    time.sleep(0.01)

plt.plot(data)
plt.show()
