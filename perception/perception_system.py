import cv2
from ultralytics import YOLO

from perception.segmentation.deeplab_segment import DeepLabSegmenter
from perception.depth.midas_depth import MiDaSDepth


class PerceptionSystem:

    def __init__(self):

        # detection
        self.detector = YOLO("yolov8n.pt")

        # segmentation
        self.segmenter = DeepLabSegmenter()

        # depth
        self.depth_model = MiDaSDepth()

    def run(self, image):

        # detection
        det_results = self.detector(image)

        # segmentation
        seg_mask = self.segmenter.predict(image)

        # depth
        depth_map = self.depth_model.predict(image)

        return det_results, seg_mask, depth_map