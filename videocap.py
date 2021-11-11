import cv2
import os


class VideoCap:
    def __init__(self, path=None):
        self.cap = cv2.VideoCapture(path)

    def read(self):
        # ret, frame = self.cap.read()
        return self.cap.read()

    def seek(self, pos):
        p = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, p + pos)

    def goto(self, pos):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)


def save_frame(frame):
    pth = f"out_frames{os.path.sep}frame-{0:>04}.png"
    if not os.path.isfile(pth):
        cv2.imwrite(pth, frame)
    else:
        i = 0
        while os.path.isfile(pth):
            i += 1
            pth = f"out_frames{os.path.sep}frame-{i:>04}.png"
        cv2.imwrite(pth, frame)
