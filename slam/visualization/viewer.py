import numpy as np
import cv2


class TrajectoryViewer:

    def __init__(self):

        self.canvas = np.zeros((600, 600, 3), dtype=np.uint8)
        self.origin = (300, 300)

    def update(self, t):

        x = int(t[0][0])
        z = int(t[2][0])

        draw_x = self.origin[0] + x
        draw_y = self.origin[1] - z

        cv2.circle(self.canvas, (draw_x, draw_y), 2, (0, 255, 0), -1)

        cv2.imshow("Trajectory", self.canvas)