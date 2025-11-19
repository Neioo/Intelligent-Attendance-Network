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

# TUNING PARAMETERS
FRAME_SKIP = 5        # Cloud needs this (process 1 out of 5 frames)
GRACE_PERIOD = 4      # Prevents box flickering when skipping frames
K = 5
SIM_THRESH = 0.40
MARGIN = 0.06
CLASS_MINSUM = 0.50

st.set_page_config(page_title="CCS Events Attendance", layout="centered")

# ====== LOAD RESOURCES (Global Cache) ======
@st.cache_resource
def load_resources():
    # 1. Load YOLO
    try:
        det_model = YOLO(YOLO_WEIGHTS)
    except Exception as e:
        st.error(f"Failed to load YOLO: {e}")
        st.stop()

    # 2. Warmup DeepFace
    try:
        DeepFace.build_model("ArcFace")
    except Exception as e:
        st.error(f"Failed to load DeepFace: {e}")
        st.stop()

    # 3. Load Gallery
    if not Path(G_PATH).exists():
        # Dummy gallery to prevent crashes
        g_vecs = np.zeros((1, 512), dtype=np.float32)
        g_labels = ["Unknown"]
        names = ["Unknown"]
        gallery_found = False
    else:
        G = np.load(G_PATH, allow_pickle=True)
        g_vecs = G["vecs"].astype(np.float32)
        g_labels = G["labels"].tolist()
        names = G["classes"].tolist()
        gallery_found = True
        
    return det_model, g_vecs, g_labels, names, gallery_found

DET, G_VECS, G_LABELS, NAMES, GALLERY_FOUND = load_resources()

if not GALLERY_FOUND:
    st.warning(f"⚠️ {G_PATH} not found. All faces will be Unknown.")

# ====== HELPER FUNCTIONS (From your Reference Code) ======
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

def center_square(img):
    h,w = img.shape[:2]
    s = min(h,w)
    y0 = (h - s)//2; x0 = (w - s)//2
    return img[y0:y0+s, x0:x0+s]

# ====== THREAD COMMUNICATION ======
# WebRTC runs in a separate thread. We use this queue to send data to the UI.
if "log_queue" not in st.session_state:
    st.session_state.log_queue = queue.Queue()

# ====== THE CLOUD VIDEO PROCESSOR ======
class AttendanceProcessor:
    def __init__(self):
        self.frame_i = -1
        self.last_results = None
        self.grace_count = 0
        
        # Settings from UI
        self.event_name = ""
        self.direction = ""
        
        # Access the global queue
        self.queue = st.session_state.log_queue
        
        # Internal memory to prevent spamming the queue
        self.local_seen = set()

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # Mirror (Selfie)
            img = cv2.flip(img, 1)
            h, w = img.shape[:2]

            # --- 1. FRAME SKIPPING (Cloud Optimization) ---
            self.frame_i = (self.frame_i + 1) % (FRAME_SKIP + 1)
            run_det = (self.frame_i == 0)

            # --- 2. YOLO DETECTION ---
            if run_det:
                # Same params as your reference code (conf=0.20, iou=0.70)
                # But reduced imgsz to 320/640 for Cloud Speed
                res = DET(img, conf=0.35, iou=0.70, imgsz=320, max_det=50, verbose=False)[0]
                
                if res.boxes is not None and len(res.boxes) > 0:
                    self.last_results = res.boxes
                    self.grace_count = GRACE_PERIOD # Reset grace period
                else:
                    self.grace_count -= 1 # Fade out boxes

            # --- 3. PROCESS DETECTIONS ---
            # Only draw if we have recent results
            if self.last_results is not None and self.grace_count > 0:
                
                xyxy = self.last_results.xyxy.cpu().numpy()
                
                for box in xyxy:
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1, x2, y2 = expand(x1, y1, x2, y2, w, h)

                    bw, bh = x2 - x1, y2 - y1
                    if bw < 40 or bh < 40: continue

                    # Recognition (Only run on the specific detection frame)
                    if run_det:
                        face_bgr = img[y1:y2, x1:x2]
                        if face_bgr.size > 0:
                            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                            
                            # Exact Helper from your code
                            face_rgb = center_square(face_rgb)

                            v = embed_rgb_arcface(face_rgb)
                            name, top1 = classify_knn(v)
                            
                            # Store for drawing text below
                            # (In a real app, we'd cache this per box ID, but simple overwrite works for single face)
                            self.current_name = name
                            self.current_score = top1
                            
                            # --- LOGGING LOGIC MATCHING YOUR REFERENCE ---
                            if name != "Unknown":
                                # Send to Main Thread to handle "should_log" logic
                                # We use a simple set here to avoid sending 30 messages per second
                                if name not in self.local_seen:
                                    self.queue.put({
                                        "name": name,
                                        "event": self.event_name,
                                        "direction": self.direction,
                                        "time": dt.datetime.now().strftime("%H:%M:%S")
                                    })
                                    self.local_seen.add(name) # Mark sent
                    else:
                        # Use cached values if skipping frames
                        name = getattr(self, 'current_name', "Scanning...")
                        top1 = getattr(self, 'current_score', 0.0)

                    # Drawing
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{name} {top1:.2f}"
                    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        except Exception as e:
            print(f"Recv Error: {e}")
            return frame

# ====== UI LAYOUT (MATCHING YOUR REFERENCE) ======
st.title("🎥 CCS Events Attendance Checker")

# Session State Init
if "logs" not in st.session_state:
    st.session_state.logs = []
if "logged_names" not in st.session_state:
    st.session_state.logged_names = set()

# Inputs
event_name = st.text_input("Event name:", placeholder="e.g., CS Orientation 2025", key="evt_input")
direction = st.radio("Current scan mode (IN/OUT):", ["IN", "OUT"], horizontal=True, key="dir_input")

col1, col2 = st.columns([2, 1])

with col1:
    # REPLACES 'Start Camera' TOGGLE
    # This is the WebRTC component
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    ctx = webrtc_streamer(
        key="attendance-reference-ui",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=AttendanceProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    # Send UI settings to the background processor
    if ctx.video_processor:
        ctx.video_processor.event_name = event_name
        ctx.video_processor.direction = direction

with col2:
    if st.button("Reset logs"):
        st.session_state.logs = []
        st.session_state.logged_names = set()
        # Also clear the processor's memory if possible, or just rely on session state
        st.success("Attendance logs reset.")

# ====== LOGIC: BRIDGE QUEUE TO TABLE ======
# This replaces the "should_log" logic in the while loop.
# We pull data from the Queue and decide HERE if we should add it to the table.

new_logs_found = False
while not st.session_state.log_queue.empty():
    try:
        data = st.session_state.log_queue.get_nowait()
        name = data["name"]
        
        # --- REFERENCE CODE LOGIC: "should_log" ---
        if name not in st.session_state.logged_names:
            st.session_state.logged_names.add(name)
            st.session_state.logs.append({
                "timestamp": data["time"],
                "name": name,
                "direction": data["direction"],
                "event_name": data["event"]
            })
            new_logs_found = True
    except queue.Empty:
        break

# Auto-refresh ONLY if we found new logs (Prevents Death Loop)
if new_logs_found:
    st.rerun()

# ====== ATTENDANCE TABLE ======
st.subheader("Attendance logs")

if st.session_state.logs:
    df = pd.DataFrame(st.session_state.logs)
    # Show newest on top
    st.dataframe(df.iloc[::-1], use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "attendance_logs.csv", "text/csv")
else:
    st.info("No attendance logs yet. Start the camera to begin logging.")
