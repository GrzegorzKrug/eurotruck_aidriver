from itertools import cycle
import pyvjoy
import time

MAX = 2 ** 17 - 1

j = pyvjoy.VJoyDevice(1)

# turn button number 15 on
# j.set_button(15, 1)

# Notice the args are (buttonID,state) whereas vJoy's native API is the other way around.


# turn button 15 off again
# j.set_button(7, 0)
# Set X axis to fully left
# j.set_axis(pyvjoy.HID_USAGE_X, 0x1)

# Set X axis to fully right
# j.set_axis(pyvjoy.HID_USAGE_X, 0x7000)

# Also implemented:

# j.reset()
# j.reset_buttons()
# j.reset_povs()

# The 'efficient' method as described in vJoy's docs - set multiple values at once


# j.data.lButtons = 19  # buttons number 1,2 and 5 (1+2+16)
# j.data.wAxisX = 0x5000
# j.data.wAxisY = 0x7500

diley = 1e-3
# for x in range(33):
#     v = x * 1000
#     print(v)
#     j.data.wAxisX = v
#     j.update()
#     time.sleep(diley)
#
# for x in range(33):
#     v = x * 1000
#     print(v)
#     j.data.wAxisY = v
#     j.update()
#     time.sleep(diley)
# for x in range(33):
#     v = x * 1000
#     print(v)
#     j.data.wAxisZ = v
#     j.update()
#     time.sleep(diley)

# for x in range(2 ** 15):
#     print(x)
#     j.data.lButtons = x
#     j.update()
# time.sleep(1e-6)
# j.data.lButtons = 19  # buttons number 1,2 and 5 (1+2+16)

t0 = time.time()
dur = 100
angle = -1
ANGS = [0, 1, 12, 14, 15, 16, 30, 31, 32, 33]
ANGS = list(range(15, 31))
cyc = cycle(ANGS)
j.data.wAxisX = int(32768 / 2)

while True:
    elapsed = time.time() - t0
    if elapsed >= dur:
        break
    else:
        time.sleep(0.5)

    # angle = (angle + 1) % 33
    angle = next(cyc)
    print(f"Angle: {angle}")
    # j.data.wAxisX = int(angle * 1000)
    j.data.wAxisY = int(angle * 1000)
    j.update()

# send data to vJoy device
# j.update()

"""
Steering Centered

Gas Centered
Brake Inverted Centered
"""
