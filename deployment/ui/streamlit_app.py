import streamlit as st
import requests
import numpy as np
import cv2
import time

def safe_request(url, files, retries=3):
    for i in range(retries):
        try:
            res = requests.post(API_URL, files=files, timeout=120)
            return res
        except requests.exceptions.Timeout:
            if i == retries - 1:
                raise
            time.sleep(2)


def wake_up_server():
    try:
        requests.get(API, timeout=10)
    except:
        pass

# ---------------- CONFIG ---------------- #
API = "https://deepfake-detection-system-jlnj.onrender.com"
REQUEST_TIMEOUT = 180  # seconds

# ---------------- LOGIN ---------------- #
def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
            st.success("Login successful ✅")
        else:
            st.error("Invalid credentials ❌")


# Session state check
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# ---------------- MAIN UI ---------------- #
st.set_page_config(page_title="Deepfake Detector", layout="wide")
st.title("🧠 Deepfake Detection System")

tab1, tab2, tab3, tab4 = st.tabs(["🖼 Image", "🎥 Video", "🎵 Audio", "📷 Webcam"])

# ---------------- IMAGE ---------------- #
with tab1:
    file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if st.button("Predict Image"):

        if uploaded_file is not None:
            files = {"file": uploaded_file.getvalue()}

        with st.spinner("Analyzing Image... ⏳"):
            try:
                res = requests.post(API_URL, files=files, timeout=120)

                if res.status_code == 200:
                    result = res.json()
                    st.success(f"Prediction: {result.get('prediction', 'Done')}")

                else:
                    st.error(f"Server error: {res.status_code}")

            except requests.exceptions.Timeout:
                st.error("⏳ Request timed out. Model is slow.")

            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        st.warning("Please upload an image first.")

                result = res.json()

                st.success(result.get("prediction", "No result"))
                st.write("Confidence:", result.get("confidence", "N/A"))
                st.info(result.get("explanation", "No explanation"))

                # Heatmap
                if "heatmap" in result:
                    heatmap_bytes = bytes.fromhex(result["heatmap"])
                    nparr = np.frombuffer(heatmap_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    st.image(img, caption="Heatmap")

                # PDF Download
                if "pdf" in result:
                    pdf_bytes = bytes.fromhex(result["pdf"])
                    st.download_button(
                        label="📄 Download Report",
                        data=pdf_bytes,
                        file_name="deepfake_report.pdf",
                        mime="application/pdf"
                    )

            except requests.exceptions.Timeout:
                st.error("⏰ Request timed out. Try again.")
            except Exception as e:
                st.error(f"Error: {e}")


# ---------------- VIDEO ---------------- #
with tab2:
    file = st.file_uploader("Upload Video", type=["mp4"])

    if file and st.button("Predict Video"):
        with st.spinner("Processing Video... ⏳"):
            try:
                wake_up_server()

                res = safe_request(
                    f"{API}/predict_video",
                    {"file": file}
                )

                st.json(res.json())
            except Exception as e:
                st.error(f"Error: {e}")


# ---------------- AUDIO ---------------- #
with tab3:
    file = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac"])

    if file and st.button("Predict Audio"):
        with st.spinner("Analyzing Audio... ⏳"):
            try:
                wake_up_server()

                res = safe_request(
                    f"{API}/predict_audio",
                    {"file": file}
                )

                st.json(res.json())
            except Exception as e:
                st.error(f"Error: {e}")


# ---------------- WEBCAM ---------------- #
with tab4:
    st.subheader("📷 Real-Time Webcam Detection")

    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    cap = None

    if run:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Camera not accessible ❌")

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break

            # Resize for speed 🚀
            frame = cv2.resize(frame, (320, 240))

            # Encode
            _, img_encoded = cv2.imencode('.jpg', frame)

            try:
                res = safe_request(
                    f"{API}/predict_image",
                    {"file": img_encoded.tobytes()}
                )

                result = res.json()
                label = result.get("prediction", "...")
            except:
                label = "ERROR"

            # Draw label
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            FRAME_WINDOW.image(frame, channels="BGR")

            # Reduce API load
            time.sleep(0.2)

        if cap:
            cap.release()
