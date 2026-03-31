import streamlit as st
import requests
import numpy as np
import cv2
import cv2
import time
import streamlit as st

# 🔐 Simple login
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid credentials")


# Check login
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

API = "https://deepfake-detection-system-jlnj.onrender.com"

st.title("🧠 Deepfake Detection System")

tab1, tab2, tab3, tab4 = st.tabs(["Image", "Video", "Audio", "Webcam"])

# IMAGE
with tab1:
    file = st.file_uploader("Upload Image", type=["jpg","png"])
    if file and st.button("Predict Image"):
        res = requests.post(f"{API}/predict_image", files={"file": file})
        result = res.json()

        st.success(result["prediction"])
        st.write("Confidence:", result["confidence"])
        st.info(result["explanation"])

        heatmap_bytes = bytes.fromhex(result["heatmap"])
        nparr = np.frombuffer(heatmap_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        st.image(img)
        # Download PDF
        if "pdf" in result:
            pdf_bytes = bytes.fromhex(result["pdf"])

            st.download_button(
                label="📄 Download Full Report",
                data=pdf_bytes,
                file_name="deepfake_report.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("PDF report not available")

# VIDEO
with tab2:
    file = st.file_uploader("Upload Video", type=["mp4"])
    if file and st.button("Predict Video"):
        res = requests.post(f"{API}/predict_video", files={"file": file})
        st.write(res.json())

# AUDIO
with tab3:
    file = st.file_uploader("Upload Audio", type=["wav","mp3","flac","parquet"])
    if file and st.button("Predict Audio"):
        res = requests.post(f"{API}/predict_audio", files={"file": file})
        st.write(res.json())

# WEBCAM
with tab4:
    st.header("📷 Real-Time Webcam Detection")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize + send to API
        _, img_encoded = cv2.imencode('.jpg', frame)

        res = requests.post(
            f"{API}/predict_image",
            files={"file": img_encoded.tobytes()}
        )

        try:
            result = res.json()
            label = result.get("prediction", "...")
        except:
            label = "ERROR"

        # Put label on frame
        cv2.putText(frame, label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

        time.sleep(0.1)

    cap.release()
