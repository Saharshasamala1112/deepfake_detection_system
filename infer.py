import os
from PIL import Image
import torch
import torchvision.transforms as T

from models.multimodal_model import MultiModalModel
from utils.device import get_device


def main():
    device = get_device()

    model = MultiModalModel().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    # Auto select image
    test_dir = "data/images/Test/real"
    img_name = os.listdir(test_dir)[0]
    img_path = os.path.join(test_dir, img_name)

    print("Testing image:", img_path)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img).item()

    print("Raw Output:", output)

    if output > 0.7:
        prediction = "FAKE"
    elif output < 0.3:
        prediction = "REAL"
    else:
        prediction = "UNCERTAIN"

    print(f"Prediction: {prediction}")
    print(f"Confidence: {output:.3f}")


if __name__ == "__main__":
    main()