import cv2
import torch
import numpy as np

class VideoPredictor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict(self, video_path):
        cap = cv2.VideoCapture(video_path)

        frames = []
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if count % 10 == 0:  # 🔥 skip frames
                frame = cv2.resize(frame, (128, 128))
                frame = frame / 255.0
                frame = torch.tensor(frame).permute(2, 0, 1).float()
                frames.append(frame)

            count += 1

            if len(frames) >= 10:  # 🔥 max 10 frames
                break

        cap.release()

        if len(frames) == 0:
            return {"prediction": "ERROR", "confidence": 0}

        batch = torch.stack(frames)

        with torch.no_grad():
            outputs = self.model(batch)
            prob = torch.sigmoid(outputs).mean().item()

        return {
            "prediction": "FAKE" if prob > 0.5 else "REAL",
            "confidence": float(prob)
        }
