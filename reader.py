import imutils
import sys
import cv2
import os

vid = cv2.VideoCapture(f"src_movies{os.path.sep}2021-10-21 17-31-27.mkv")


def detector(frame):
    marked = frame
    return marked


pause = False




def player_preview():
    while vid.isOpened():
        if not pause:
            ret, frame = vid.read()
            frame = imutils.resize(frame, height=600)
            marked = detector(frame)

        cv2.imshow("frame", marked)
        key = cv2.waitKey(25) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:
            pause = not pause
        elif key == ord('s'):
            save_frame(frame)


class State:
    pass


class Drive(State):
    pass


class ChangeLane(State):
    pass


class Turn(State):
    pass


class Event:
    "Events that change state"
    pass


class Trigger:
    "Events change params in state. Signs"
    pass


class Transition:
    pass
