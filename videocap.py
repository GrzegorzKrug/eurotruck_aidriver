import imutils
import cv2
import os

from vision_modules.picture_paths import PIC_SOURCE_FOLDER


class VideoCap:
    def __init__(self, path=None):
        self.cap = cv2.VideoCapture(path)
        self._st = 1

    def read(self):
        # ret, frame = self.cap.read()
        return self.cap.read()

    @property
    def size(self):
        self.goto(0)
        ret, frame = self.cap.read()
        h, w, _ = frame.shape
        return h, w

    def seek(self, off):
        if off != 0:
            p = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, p + off)

    def goto(self, pos):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def __iter__(self):
        self.goto(0)
        print(f"ST is: {self._st}")
        return self

    def iter(self, st=1):
        print(f"Set st to: {st}")
        self._st = st
        return self

    def __next__(self):
        # print(f"Next: {self._st}")

        p = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self.cap.read()
        self.seek(self._st - 1)
        # print(f"P: {p}, ret: {ret}")
        if ret:
            return p, frame
        else:
            raise StopIteration


# def save_frame(frame):
#     pth = f"out_frames{os.path.sep}frame-{0:>04}.png"
#     if not os.path.isfile(pth):
#         cv2.imwrite(pth, frame)
#     else:
#         i = 0
#         while os.path.isfile(pth):
#             i += 1
#             pth = f"out_frames{os.path.sep}frame-{i:>04}.png"
#         cv2.imwrite(pth, frame)


def save_frames(src_path, interval=1, max_size=1024, prefix=None, max_frames=None):
    prefix = str(prefix) + "-" if prefix else ""
    basename = os.path.basename(src_path)

    "Read input"
    *name, ext = basename.split(".")
    name = ''.join(name)

    "Make output"
    dest = os.path.abspath(PIC_SOURCE_FOLDER + os.path.sep + prefix + name) + os.path.sep
    os.makedirs(dest, exist_ok=True)

    cap = VideoCap(src_path)
    h, w = cap.size
    if h > w:
        key = "height"
        v = h
    else:
        key = "width"
        v = w
    print(f"Max size: {max_size}, v:{v}")
    if v > max_size:
        dwn_f = lambda x: imutils.resize(x, **{key: max_size})
    else:
        dwn_f = lambda x: x

    for i, (p, pic) in enumerate(cap.iter(interval)):
        print(i, p)

        out_path = dest + f"frame-{p:>05}.png"
        pic = dwn_f(pic)
        cv2.imwrite(out_path, pic)
        if max_frames and i > max_frames:
            break


if __name__ == "__main__":
    # save_frames(os.path.join("vision_source", "src_movies", "autostrada-notsmooth.mp4"), interval=40)
    save_frames(
            os.path.join("vision_source", "src_movies", "autostrada-notsmooth.mp4"),
            interval=1, prefix="numbers", max_size=2000, max_frames=850)
