# 🧠 Multimodal Deepfake Detection System

An advanced AI-powered system to detect deepfake content across **Images, Videos, Audio, and Webcam streams**, integrated with **Explainable AI (Grad-CAM)** and **LLM-based reasoning**. The system provides real-time predictions through a modern dashboard and supports full cloud deployment.

---

## 🚀 Key Features

* 🖼 **Image Deepfake Detection**
* 🎥 **Video Deepfake Detection**
* 🔊 **Audio Deepfake Detection**
  Supports `.wav`, `.mp3`, `.flac`, `.parquet`
* 📷 **Real-Time Webcam Detection**
* 🔥 **Grad-CAM Explainability**
* 🤖 **LLM-Based Explanation (AI reasoning)**
* 📄 **Downloadable PDF Reports**
* 🔐 **Login Authentication System**
* 🌐 **Cloud Deployment (Render + Streamlit)**

---

## 🏗️ System Architecture

User Input (Image / Video / Audio / Webcam)
↓
Preprocessing Pipeline
↓
Feature Extraction
├── Spatial Features (CNN)
├── Temporal Features (Video Frames)
├── Frequency Features (Audio Spectrogram)
↓
Prediction Model
↓
Grad-CAM (Explainability)
↓
LLM Explanation
↓
Final Output (Prediction + Confidence + Report)

---

## 🖥️ User Interface

### 🔐 Login Page

![Login](screenshots/login.png)

### 🖼 Image Detection + Grad-CAM

![Image](screenshots/image.png)

### 🎥 Video Detection

![Video](screenshots/video.png)

### 🔊 Audio Detection

![Audio](screenshots/audio.png)

### 📷 Webcam Detection

![Webcam](screenshots/webcam.png)

---

## 🧪 Technologies Used

* **Frontend:** Streamlit
* **Backend:** FastAPI
* **Deep Learning:** PyTorch, Torchvision
* **Computer Vision:** OpenCV
* **Audio Processing:** Librosa
* **Explainability:** Grad-CAM
* **LLM Integration:** OpenAI API
* **Report Generation:** ReportLab
* **Deployment:** Render + Streamlit Cloud

---

## ⚙️ Installation Guide (Local Setup)

### 1️⃣ Clone Repository

git clone https://github.com/your-username/deepfake_detection_system.git
cd deepfake_detection_system

---

### 2️⃣ Create Virtual Environment

python -m venv .venv
.venv\Scripts\activate

---

### 3️⃣ Install Dependencies

pip install -r requirements.txt

---

### 4️⃣ Setup Environment Variables

Create a `.env` file in root directory:

OPENAI_API_KEY=your_api_key_here

---

### 5️⃣ Run Backend Server

python -m uvicorn deployment.api.app:app --reload

---

### 6️⃣ Run Frontend UI

streamlit run deployment/ui/streamlit_app.py

---

## 🌐 Deployment

### 🔹 Backend (Render)

Build Command:
pip install -r requirements.txt

Start Command:
uvicorn deployment.api.app:app --host 0.0.0.0 --port 10000

Environment Variable:
OPENAI_API_KEY=your_api_key

---

### 🔹 Frontend (Streamlit Cloud)

Deploy using:
deployment/ui/streamlit_app.py

---

## 📊 Example Output

Prediction: FAKE
Confidence: 0.87

Explanation:
The model detected inconsistencies in facial regions and unnatural blending artifacts. Grad-CAM highlights manipulated areas.

---

## 📁 Project Structure

deepfake_detection_system/
│
├── deployment/
│   ├── api/
│   └── ui/
│
├── models/
├── inference/
├── explainability/
├── utils/
├── data/
├── requirements.txt
└── README.md

---

## 🚀 Future Enhancements

* 📈 Improve accuracy to 95%+
* 🔗 Multimodal fusion (Image + Video + Audio)
* 📊 Analytics dashboard
* 👤 User history tracking
* 🧠 Transformer-based models

---

## 👨‍💻 Author

**Saharsha Samala**
B.Tech AIML | ICFAI University

---

## ⭐ Conclusion

This project demonstrates a **complete AI system** integrating:

* Deep Learning
* Explainable AI
* LLM reasoning
* Full-stack deployment

making it a **production-ready deepfake detection platform**.

---

⭐ If you like this project, consider giving it a star!
77
