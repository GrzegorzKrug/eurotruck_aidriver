import numpy as np


class FrameFoi:
    def __init__(self, y1, y2, x1, x2):
        """
        Args:
            y1: Height start from top
            y2: Height end from top
            x1: Width start from left
            x2: Width end from left
        """
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.xsl = slice(None, None, None)
        self.ysl = slice(None, None, None)

    def drawing_points(self, frame):
        h, w, *_ = frame.shape
        pt1 = np.round([w * self.x1, h * self.y1, ]).astype(int)
        pt2 = np.round([w * self.x2, h * self.y2, ]).astype(int)
        return pt1, pt2

    def get_slices(self, frame):
        """

        Args:
            frame:

        Returns:
            yslice
            xslice
        """
        h, w, *c = frame.shape
        x = (np.array([self.x1, self.x2]) * w).round().astype(int)
        y = (np.array([self.y1, self.y2]) * h).round().astype(int)
        xsl = slice(*x)
        ysl = slice(*y)
        self.xsl = xsl
        self.ysl = ysl
        return ysl, xsl

    def get_foi(self, frame):
        ysl, xsl = self.get_slices(frame)
        roi = frame[ysl, xsl]
        return roi


foi_roadsample = FrameFoi(400 / 720, 445 / 720, 468 / 1280, 1000 / 1280)  # Original Road
foi_roadsample = FrameFoi(270 / 720, 445 / 720, 468 / 1280, 1000 / 1280)  # Bigger road
# foi_roadsample = FrameFoi(200 / 720, 445 / 720, 200 / 1280, 1100 / 1280)  # Wide Thin
# foi_roadsample = FrameFoi(350 / 720, 445 / 720, 468 / 1280, 1000 / 1280)  # Original road
foi_window_view = FrameFoi(100 / 720, 445 / 720, 468 / 1280, 900 / 1280)  # viewport

__all__ = ['foi_roadsample', 'foi_window_view']