import numpy as np


class Map:

    def __init__(self):
        self.keyframes = []
        self.points = []

    def add_keyframe(self, frame):
        self.keyframes.append(frame)

    def add_points(self, points_3d):
        for p in points_3d:
            self.points.append(p)