from fastapi import FastAPI, UploadFile, File
import tempfile
import shutil

from inference.image_predictor import ImagePredictor
from inference.video_predictor import VideoPredictor
from inference.audio_predictor import AudioPredictor

app = FastAPI()

# 👉 Load your model here
model = ...  # load your trained model

image_model = ImagePredictor(model)
video_model = VideoPredictor(model)
audio_model = AudioPredictor(model)


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    return image_model.predict(file.file)


@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        shutil.copyfileobj(file.file, temp)
        result = video_model.predict(temp.name)
    return result


@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    return audio_model.predict(file.file)
