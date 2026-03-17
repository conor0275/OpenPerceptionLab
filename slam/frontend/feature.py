import cv2


class FeatureExtractor:

    def __init__(self):
        self.orb = cv2.ORB_create(2000)

    def extract(self, img):
        kp, des = self.orb.detectAndCompute(img, None)
        return kp, des