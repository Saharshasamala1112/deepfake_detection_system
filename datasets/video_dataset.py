import os
import cv2
import torch
import numpy as np
from .base_dataset import BaseDataset


class VideoDataset(BaseDataset):
    def __init__(self, video_dir, num_frames=16, transform=None):
        super().__init__(transform)
        self.video_paths = [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.endswith((".mp4", ".avi", ".MOV"))
        ]
        self.num_frames = num_frames

    def __len__(self):
        return len(self.video_paths)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(total_frames // self.num_frames, 1)

        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frames.append(frame)

        cap.release()

        frames = np.array(frames)
        return torch.tensor(frames, dtype=torch.float32)

    def __getitem__(self, idx):
        video = self.load_video(self.video_paths[idx])
        label = 0  # placeholder

        return video, torch.tensor(label, dtype=torch.float32)