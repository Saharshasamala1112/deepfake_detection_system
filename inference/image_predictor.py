import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

class ImagePredictor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # 🔥 faster
            transforms.ToTensor(),
        ])

    def predict(self, img_bytes):
        img = Image.open(img_bytes).convert("RGB")
        img = self.transform(img).unsqueeze(0)

        with torch.no_grad():  # 🔥 important
            output = self.model(img)
            prob = torch.sigmoid(output).item()

        return {
            "prediction": "FAKE" if prob > 0.5 else "REAL",
            "confidence": float(prob)
        }
