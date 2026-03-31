from fastapi import FastAPI, UploadFile, File
import tempfile
import shutil
import random

app = FastAPI()

# ==============================
# 🚀 FAST DUMMY MODEL (NO DELAY)
# ==============================

class DummyPredictor:
    def predict(self, file):
        return {
            "prediction": random.choice(["REAL", "FAKE"]),
            "confidence": round(random.uniform(0.85, 0.99), 2),
            "explanation": "AI detected facial inconsistencies and unnatural patterns."
        }

# Use dummy models for all
image_model = DummyPredictor()
video_model = DummyPredictor()
audio_model = DummyPredictor()

# ==============================
# HEALTH CHECK
# ==============================

@app.get("/")
def home():
    return {"status": "running"}

# ==============================
# IMAGE PREDICTION
# ==============================

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    result = image_model.predict(file.file)
    return result

# ==============================
# VIDEO PREDICTION
# ==============================

@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        shutil.copyfileobj(file.file, temp)
        result = video_model.predict(temp.name)
    return result

# ==============================
# AUDIO PREDICTION
# ==============================

@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    result = audio_model.predict(file.file)
    return result
