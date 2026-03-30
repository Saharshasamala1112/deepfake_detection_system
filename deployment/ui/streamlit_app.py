import streamlit as st
import requests
import numpy as np
import cv2

API_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide")
st.title("🧠 Deepfake Detection Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["🖼 Image", "🎥 Video", "🔊 Audio", "📷 Webcam"])

# IMAGE
with tab1:
    file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if file:
        st.image(file)

        if st.button("Predict Image"):
            res = requests.post(f"{API_URL}/predict_image", files={"file": file})
            result = res.json()

            st.success(result["prediction"])
            st.metric("Confidence", f"{float(result['confidence']):.2%}")

            # Show GradCAM
            heatmap_bytes = bytes.fromhex(result["heatmap"])
            nparr = np.frombuffer(heatmap_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            st.image(img, caption="Grad-CAM")