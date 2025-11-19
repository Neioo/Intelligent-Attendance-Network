# --- MUST BE THE FIRST LINE ---
import cv2
# ------------------------------

import av
import queue
import time
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from ultralytics import YOLO
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ====== CONFIGURATION ======
ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = str(ROOT / "yolov8s-face-lindevs.pt")
G_PATH = str(ROOT / "gallery_arcface.npz")

# --- TUNING PARAMETERS ---
FRAME_SKIP = 5        # Process 1 out of 5 frames (Relieves CPU)
GRACE_PERIOD = 4      # Keeps box visible if detection flickers
LOG_COOLDOWN = 5.0    # Log the same person again after 5 seconds
REFRESH_RATE = 2.0    # Auto-refresh the UI every 2 seconds

# --- ARCFACE PARAMS ---
K = 5
SIM_THRESH = 0.40
MARGIN = 0.06
CLASS_MINSUM = 0.50

st.set_page_config(page_title="CCS Attendance", layout="centered")

# ====== LOAD RESOURCES ======
@st.cache_resource
def load_models():
    return YOLO(YOLO_WEIGHTS)

try:
    DET = load_models()
    DeepFace.build_model("ArcFace") # Warmup
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Load Gallery
if Path(G_PATH).exists():
    G = np.load(G_PATH, allow_pickle=True)
    G_VECS = G["vecs"].astype(np.float32)
    G_LABELS = G["labels"].tolist()
    NAMES = G["classes"].tolist()
else:
    st.warning("⚠️ Gallery not found! All faces will be 'Unknown'.")
    G_VECS = np.zeros((1, 512), dtype=np.float32)
    G_LABELS = ["Unknown"]
    NAMES = ["Unknown"]

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

def expand(x1,y1,x2,y2,w,h,pad=0.0):
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1)*(1+pad), (y2-y1)*(1+pad)
    nx1, ny1 = int(max(0, cx-bw/2)), int(max(0, cy-bh/2))
    nx2, ny2 = int(min(w-1, cx+bw/2)), int(min(h-1, cy+bh/2))
    return nx1, ny1, nx2, ny2

# ====== QUEUE SYSTEM ======
# This allows the background thread to talk to the main Streamlit thread
if "log_queue" not in st.session_state:
    st.session_state.log_queue = queue.Queue()

# ====== BACKGROUND VIDEO PROCESSOR ======
class AttendanceProcessor:
    def __init__(self):
        self.frame_i = -1
        self.last_results = None
        self.grace_count = 0 
        self.last_logged_time = {}
        self.event_name = ""
        self.direction = ""
        self.queue = st.session_state.log_queue # Use session state queue

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            h, w = img.shape[:2]

            # 1. Cadence Control (Run detection every N frames)
            self.frame_i = (self.frame_i + 1) % (FRAME_SKIP + 1)
            run_det = (self.frame_i == 0)

            # 2. Detection Logic
            if run_det:
                res = DET(img, conf=0.4, iou=0.5, imgsz=320, max_det=5, verbose=False)[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    self.last_results = res.boxes
                    self.grace_count = GRACE_PERIOD
                else:
                    self.grace_count -= 1

            # 3. Drawing & Recognition
            if self.last_results is not None and self.grace_count > 0:
                xyxy = self.last_results.xyxy.cpu().numpy()
                for box in xyxy:
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1, x2, y2 = expand(x1, y1, x2, y2, w, h)

                    if (x2 - x1) < 40 or (y2 - y1) < 40: continue

                    # Draw Box
                    color = (0, 255, 0) 
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    # Recognition (Only run if detection ran)
                    if run_det:
                        face_bgr = img[y1:y2, x1:x2]
                        if face_bgr.size > 0:
                            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                            
                            # Center Crop
                            ch, cw = face_rgb.shape[:2]
                            s = min(ch, cw)
                            y0, x0 = (ch-s)//2, (cw-s)//2
                            face_rgb = face_rgb[y0:y0+s, x0:x0+s]

                            v = embed_rgb_arcface(face_rgb)
                            name, conf = classify_knn(v)
                            
                            # Color unknown faces RED
                            if name == "Unknown":
                                color = (0, 0, 255)
                                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                            # --- LOGGING LOGIC ---
                            now = time.time()
                            last_time = self.last_logged_time.get(name, 0)
                            
                            # CHANGE: We now log "Unknown" too, so you see activity
                            if (now - last_time) > LOG_COOLDOWN:
                                self.last_logged_time[name] = now
                                self.queue.put({
                                    "timestamp": dt.datetime.now().strftime("%H:%M:%S"),
                                    "name": name,
                                    "direction": self.direction,
                                    "event_name": self.event_name
                                })

                            label = f"{name} {conf:.2f}"
                            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        except Exception as e:
            print(f"Error: {e}")
            return frame

# ====== UI LAYOUT ======
st.title("🎥 CCS Attendance (Live)")

if "logs" not in st.session_state:
    st.session_state.logs = []

# Inputs
event_name_input = st.text_input("Event:", value="Thesis Defense", key="evt")
direction_input = st.radio("Mode:", ["IN", "OUT"], horizontal=True, key="dir")

col1, col2 = st.columns([2, 1])

with col1:
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    ctx = webrtc_streamer(
        key="attendance-final",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=AttendanceProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    # Push settings to processor
    if ctx.video_processor:
        ctx.video_processor.event_name = event_name_input
        ctx.video_processor.direction = direction_input

with col2:
    if st.button("Clear Logs"):
        st.session_state.logs = []
        st.success("Cleared.")
    
    st.write("### Status")
    if ctx.state.playing:
        st.success("Camera Active")
    else:
        st.warning("Camera Off")

# ====== LOG DRAINING & AUTO-REFRESH ======
# 1. Drain Queue
q = st.session_state.log_queue
while not q.empty():
    try:
        data = q.get_nowait()
        st.session_state.logs.append(data)
    except queue.Empty:
        break

# 2. Display Table
st.subheader("Attendance Logs")
if st.session_state.logs:
    df = pd.DataFrame(st.session_state.logs)
    st.dataframe(df.iloc[::-1], use_container_width=True) # Newest on top
    
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "attendance.csv", "text/csv")
else:
    st.info("Waiting for faces...")

# 3. AUTO-REFRESH LOOP (The Fix)
# Only refresh if camera is on. 2 seconds is safe for Cloud.
if ctx.state.playing:
    time.sleep(REFRESH_RATE)
    st.rerun()
