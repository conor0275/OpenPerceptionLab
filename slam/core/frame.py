import numpy as np


class Frame:

    def __init__(self, img, K, id):

        self.id = id
        self.img = img
        self.K = K

        self.keypoints = None
        self.descriptors = None

        self.pose_R = np.eye(3)
        self.pose_t = np.zeros((3,1))

        self.is_keyframe = False