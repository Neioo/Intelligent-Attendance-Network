# --- MAGIC FIX: MUST BE FIRST LINE ---
import cv2
# -------------------------------------

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

PAD = 0.0
FRAME_SKIP = 5  # 5 is safer for Cloud Free Tier
K = 5
SIM_THRESH = 0.40
MARGIN = 0.06
CLASS_MINSUM = 0.50

st.set_page_config(page_title="CCS Events Attendance", layout="centered")

# ====== LOAD RESOURCES ======
@st.cache_resource
def load_models():
    return YOLO(YOLO_WEIGHTS)

try:
    DET = load_models()
    # Warmup
    DeepFace.build_model("ArcFace")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

if Path(G_PATH).exists():
    G = np.load(G_PATH, allow_pickle=True)
    G_VECS = G["vecs"].astype(np.float32)
    G_LABELS = G["labels"].tolist()
    NAMES = G["classes"].tolist()
else:
    st.warning("Gallery not found. Recognition will not work.")
    G_VECS = np.zeros((1, 512), dtype=np.float32)
    G_LABELS = ["Unknown"]
    NAMES = ["Unknown"]

# ====== HELPER FUNCTIONS (EXACTLY FROM YOUR FIRST CODE) ======
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

def expand(x1,y1,x2,y2,w,h,pad=PAD):
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

# ====== QUEUE FOR LOGS ======
# This effectively replaces st.session_state for the background thread
log_queue = queue.Queue()

# ====== THE BACKGROUND ENGINE ======
class AttendanceProcessor:
    def __init__(self):
        self.frame_i = -1
        self.last_results = None
        # These will be updated from the UI
        self.event_name = ""
        self.direction = ""

    def recv(self, frame):
        # Convert WebRTC frame to OpenCV image
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror (Selfie)
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        # --- EXACT LOGIC FROM YOUR FIRST CODE STARTS HERE ---
        self.frame_i = (self.frame_i + 1) % (FRAME_SKIP + 1)
        run_det = (self.frame_i == 0)

        if run_det:
             # Uses DET from global scope
            res = DET(img, conf=0.4, iou=0.5, imgsz=320, max_det=10, verbose=False)[0]
            self.last_results = res.boxes
        
        if self.last_results is not None:
            xyxy = self.last_results.xyxy.cpu().numpy()
            for box in xyxy:
                x1, y1, x2, y2 = box.astype(int)
                x1, y1, x2, y2 = expand(x1, y1, x2, y2, w, h, PAD)

                # Skip tiny boxes
                if (x2 - x1) < 50 or (y2 - y1) < 50: continue

                # Draw Box
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Recognition (only if we ran detection this frame)
                if run_det:
                    face_bgr = img[y1:y2, x1:x2]
                    if face_bgr.size > 0:
                        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                        face_rgb = center_square(face_rgb) # Your helper

                        v = embed_rgb_arcface(face_rgb)
                        name, conf = classify_knn(v)

                        # Log logic
                        if name != "Unknown":
                            # Push to queue instead of st.session_state
                            log_queue.put({
                                "timestamp": dt.datetime.now().strftime("%H:%M:%S"),
                                "name": name,
                                "direction": self.direction,
                                "event_name": self.event_name
                            })
                            
                        # Label Logic
                        label = f"{name} {conf:.2f}"
                        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # --- LOGIC ENDS ---
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ====== UI LAYOUT (MATCHING CODE A) ======
st.title("🎥 CCS Events Attendance Checker")

if "logs" not in st.session_state:
    st.session_state.logs = []
if "logged_names" not in st.session_state:
    st.session_state.logged_names = set()

# INPUTS ON MAIN PAGE (Just like Code A)
event_name_input = st.text_input("Event name:", placeholder="e.g., CS Orientation 2025", key="evt_input")
direction_input = st.radio("Current scan mode (IN/OUT):", ["IN", "OUT"], horizontal=True, key="dir_input")

col1, col2 = st.columns([2, 1])

with col1:
    # REPLACES: st.toggle("Start Camera")
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    ctx = webrtc_streamer(
        key="attendance",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=AttendanceProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    # Pass the UI inputs to the background processor
    if ctx.video_processor:
        ctx.video_processor.event_name = event_name_input
        ctx.video_processor.direction = direction_input

with col2:
    if st.button("Reset logs"):
        st.session_state.logs = []
        st.session_state.logged_names = set()
        st.success("Logs reset.")

# ====== LOGGING LOGIC ======
# Check the queue for new names found by the background thread
while not log_queue.empty():
    try:
        data = log_queue.get_nowait()
        # De-duplicate (Session Logic)
        if data["name"] not in st.session_state.logged_names:
            st.session_state.logged_names.add(data["name"])
            st.session_state.logs.append(data)
    except:
        break

# Refresh if camera is running so logs appear live
if ctx.state.playing:
    time.sleep(1) 
    st.rerun()

# ====== ATTENDANCE TABLE (MATCHING CODE A) ======
st.subheader("Attendance logs")

if st.session_state.logs:
    df = pd.DataFrame(st.session_state.logs)
    # Show newest on top
    st.dataframe(df.iloc[::-1], use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="attendance_logs.csv",
        mime="text/csv"
    )
else:
    st.info("No logs yet. Start the camera.")
