import torch
import torchvision.transforms as T
import cv2
import numpy as np
from torchvision import models

class DeepLabSegmenter:

    def __init__(self):

        self.device = "cpu"

        # 加载模型
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)

        self.model.eval()
        self.model.to(self.device)

        # 图像预处理
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((512,512)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def predict(self, image):

        img = self.transform(image)
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img)["out"][0]

        mask = output.argmax(0).byte().cpu().numpy()

        return mask