import threading
import queue
import datetime as dt
import av
import cv2
import numpy as np
import streamlit as st
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ====== CONFIGURATION ======
ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = str(ROOT / "yolov8s-face-lindevs.pt")
G_PATH = str(ROOT / "gallery_arcface.npz")

# Detection/Recognition Params
PAD = 0.0
FRAME_SKIP = 5  # INCREASED for Cloud (Cloud CPUs are slower)
K = 5
SIM_THRESH = 0.40
MARGIN = 0.06
CLASS_MINSUM = 0.50

st.set_page_config(page_title="CCS Attendance (Cloud)", layout="wide")

# ====== LOAD RESOURCES ======
@st.cache_resource
def load_models():
    # Load YOLO
    det_model = YOLO(YOLO_WEIGHTS)
    # Warmup DeepFace
    DeepFace.build_model("ArcFace")
    return det_model

try:
    DET = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Load Gallery
if not Path(G_PATH).exists():
    st.warning(f"Gallery not found at {G_PATH}. Recognition will fail.")
    # Create dummy gallery to prevent crash during testing
    G_VECS = np.zeros((1, 512), dtype=np.float32)
    G_LABELS = ["Unknown"]
    NAMES = ["Unknown"]
else:
    G = np.load(G_PATH, allow_pickle=True)
    G_VECS = G["vecs"].astype(np.float32)
    G_LABELS = G["labels"].tolist()
    NAMES = G["classes"].tolist()

# ====== HELPER FUNCTIONS ======
def embed_rgb_arcface(rgb):
    try:
        rep = DeepFace.represent(
            img_path=rgb,
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=False
        )
        v = np.array(rep[0]["embedding"], dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-9)
        return v
    except:
        return np.zeros(512, dtype=np.float32)

def classify_knn(v_unit):
    sims = G_VECS @ v_unit
    idx = np.argpartition(-sims, K)[:K]
    idx = idx[np.argsort(-sims[idx])]
    top_sims = sims[idx]
    top_labels = [G_LABELS[i] for i in idx]

    weights = np.clip(top_sims - (SIM_THRESH - 0.05), 0.0, 1.0)
    scores = {c: 0.0 for c in NAMES}
    for w, lbl in zip(weights, top_labels):
        scores[lbl] += float(w)

    order = sorted(scores.items(), key=lambda x: -x[1])
    (c1, s1) = order[0]
    (c2, s2) = order[1] if len(order) > 1 else (c1, 0.0)

    top1 = float(top_sims.max())
    if (top1 >= SIM_THRESH) and ((s1 - s2) >= MARGIN) and (s1 >= CLASS_MINSUM):
        return c1, top1
    return "Unknown", top1

def expand_box(x1, y1, x2, y2, w, h, pad=PAD):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * (1 + pad), (y2 - y1) * (1 + pad)
    nx1 = int(max(0, cx - bw / 2))
    ny1 = int(max(0, cy - bh / 2))
    nx2 = int(min(w - 1, cx + bw / 2))
    ny2 = int(min(h - 1, cy + bh / 2))
    return nx1, ny1, nx2, ny2

# ====== THREAD-SAFE QUEUE ======
# WebRTC runs in a separate thread, so we use a queue to send data to the UI
result_queue = queue.Queue()

# ====== WEBRTC PROCESSOR CLASS ======
class VideoProcessor:
    def __init__(self):
        self.frame_i = -1
        self.last_results = None
        self.event_name = "Default Event"
        self.direction = "IN"

    def update_settings(self, event, direction):
        self.event_name = event
        self.direction = direction

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror image (optional, feels more natural)
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        self.frame_i = (self.frame_i + 1) % (FRAME_SKIP + 1)
        run_det = (self.frame_i == 0)

        # 1. DETECTION
        if run_det:
            res = DET(img, conf=0.4, iou=0.5, imgsz=320, max_det=5, verbose=False)[0]
            self.last_results = res.boxes

        # 2. RECOGNITION & DRAWING
        if self.last_results is not None:
            xyxy = self.last_results.xyxy.cpu().numpy()
            for box in xyxy:
                x1, y1, x2, y2 = box.astype(int)
                x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, PAD)

                if (x2 - x1) < 50 or (y2 - y1) < 50: continue

                # Draw Box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Recognition (Only run if detection ran, to save CPU)
                if run_det:
                    face_bgr = img[y1:y2, x1:x2]
                    if face_bgr.size > 0:
                        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                        v = embed_rgb_arcface(face_rgb)
                        name, conf = classify_knn(v)
                        
                        # Put result in Queue for the main thread to read
                        if name != "Unknown":
                            result_queue.put({
                                "name": name, 
                                "event": self.event_name, 
                                "direction": self.direction,
                                "time": dt.datetime.now().strftime("%H:%M:%S")
                            })
                            
                        label = f"{name} {conf:.2f}"
                        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ====== MAIN UI ======
st.title("☁️ CCS Cloud Attendance")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    event_name = st.text_input("Event Name", value="Thesis Defense")
    direction = st.radio("Mode", ["IN", "OUT"], horizontal=True)
    
    st.divider()
    if st.button("Clear Logs"):
        st.session_state.logs = []
        st.session_state.logged_names = set()

# Initialize Session State
if "logs" not in st.session_state:
    st.session_state.logs = []
if "logged_names" not in st.session_state:
    st.session_state.logged_names = set()

col1, col2 = st.columns([2, 1])

with col1:
    # RTC Configuration (Google STUN server is standard for free usage)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # The WebRTC Component
    ctx = webrtc_streamer(
        key="attendance-scanner",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # Pass settings to the processor
    if ctx.video_processor:
        ctx.video_processor.update_settings(event_name, direction)

with col2:
    st.subheader("Live Logs")
    log_placeholder = st.empty()

# BACKGROUND LOOP to check for queue updates
# This runs whenever the script reruns (Streamlit auto-reruns on interaction)
# We need to drain the queue into session state
while not result_queue.empty():
    try:
        data = result_queue.get_nowait()
        name = data['name']
        # Basic dedup logic
        if name not in st.session_state.logged_names:
            st.session_state.logged_names.add(name)
            st.session_state.logs.append(data)
    except queue.Empty:
        break

# Render dataframe
if st.session_state.logs:
    df = pd.DataFrame(st.session_state.logs).iloc[::-1] # Reverse order
    log_placeholder.dataframe(df, hide_index=True, use_container_width=True)
    
    # Force a rerun periodically to update the table if the camera is running
    # Otherwise the table only updates when you click a button
    if ctx.state.playing:
        time.sleep(1) 
        st.rerun()
