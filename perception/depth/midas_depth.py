import torch
import cv2
import numpy as np

class MiDaSDepth:

    def __init__(self):

        self.device = "cpu"

        # 加载MiDaS模型
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

        self.model.to(self.device)
        self.model.eval()

        # 加载预处理
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        self.transform = midas_transforms.small_transform

    def predict(self, image):

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

        depth = depth.astype(np.uint8)

        return depth