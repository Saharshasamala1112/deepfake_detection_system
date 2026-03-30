from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import io
import torchvision.transforms as T
import numpy as np
import cv2

from models.multimodal_model import MultiModalModel
from utils.device import get_device
from explainability.gradcam import GradCAM

app = FastAPI()

device = get_device()

model = MultiModalModel().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

gradcam = GradCAM(model)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


@app.get("/")
def home():
    return {"message": "Deepfake API Running"}


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor).item()

    pred = "FAKE" if output > 0.6 else "REAL"

    # Grad-CAM
    cam = gradcam.generate(input_tensor)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode(".jpg", overlay)

    return {
        "prediction": pred,
        "confidence": float(output),
        "heatmap": buffer.tobytes().hex()
    }