import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        for label in ["real", "fake"]:
            class_dir = os.path.join(root_dir, label)

            if not os.path.exists(class_dir):
                continue

            for img in os.listdir(class_dir):
                if img.lower().endswith((".jpg", ".png", ".jpeg")):
                    path = os.path.join(class_dir, img)
                    self.image_paths.append(path)
                    self.labels.append(0 if label == "real" else 1)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label