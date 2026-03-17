import cv2


class FeatureMatcher:

    def __init__(self):

        # ORB detector
        self.orb = cv2.ORB_create(2000)

        # matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match(self, img1, img2):

        # detect features
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        # match
        matches = self.matcher.match(des1, des2)

        # sort
        matches = sorted(matches, key=lambda x: x.distance)

        return kp1, kp2, matches