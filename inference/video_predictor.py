import cv2
import torch
from PIL import Image
import torchvision.transforms as T

from models.multimodal_model import MultiModalModel
from utils.device import get_device


class VideoPredictor:
    def __init__(self):
        self.device = get_device()

        self.model = MultiModalModel().to(self.device)
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def predict(self, video_path):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("❌ Cannot open video")
            return "ERROR", 0.0

        preds = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 20th frame (stable)
            if frame_idx % 20 == 0:
                try:
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    img = Image.fromarray(frame)
                    img = self.transform(img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        preds.append(self.model(img).item())

                except Exception as e:
                    print("Frame error:", e)

            frame_idx += 1

        cap.release()

        print("Frames processed:", len(preds))  # DEBUG

        if len(preds) == 0:
            return "ERROR_NO_FRAMES", 0.0

        avg = sum(preds) / len(preds)

        if avg > 0.6:
            return "FAKE", avg
        elif avg < 0.4:
            return "REAL", avg
        else:
            return "UNCERTAIN", avg