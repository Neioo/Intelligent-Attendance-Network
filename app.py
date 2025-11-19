# --- MAGIC FIX: MUST BE FIRST LINE ---
import cv2
# -------------------------------------

import time
import datetime as dt
import queue
from pathlib import Path

import av
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = str(ROOT / "yolov8s-face-lindevs.pt")

PAD        = 0.0
MIN_BOX    = 60

# Run detector every N frames (0 = every frame)
FRAME_SKIP = 2        # try 2; increase if CPU is slow
skip_mod   = FRAME_SKIP + 1

# ---- ArcFace k-NN params ----
K            = 5
SIM_THRESH   = 0.40
MARGIN       = 0.06
CLASS_MINSUM = 0.50

# ---- Attendance logging params ----
MIN_GAP_SECONDS = 10  # minimum seconds between logs for same person (no longer used)

# Queue for passing logs from webrtc thread → main Streamlit thread
log_queue = queue.Queue()

st.set_page_config(page_title="Face ID (ArcFace gallery k-NN)", layout="centered")
st.title("🎥 CCS Events Attendance Checker ")

# ====== LOAD GALLERY ======
G_PATH = "gallery_arcface.npz"
if not Path(G_PATH).exists():
    st.error(f"{G_PATH} not found. Run build_gallery_arcface.py first.")
    st.stop()

G = np.load(G_PATH, allow_pickle=True)
G_VECS   = G["vecs"].astype(np.float32)
G_LABELS = G["labels"].tolist()
NAMES    = G["classes"].tolist()

@st.cache_resource
def load_yolo():
    return YOLO(YOLO_WEIGHTS)

@st.cache_resource
def warmup_arcface():
    # We just run this to load the model into memory.
    # DeepFace caches it internally, so we don't need to return/pass the object.
    DeepFace.build_model("ArcFace") 
    return True

DET = load_yolo()
warmup_arcface() # Warm up so first frame isn't slow

def embed_rgb_arcface(rgb):
    """
    RGB uint8 face crop -> L2-normalized ArcFace vector.
    """
    rep = DeepFace.represent(
        img_path=rgb,
        model_name="ArcFace",
        detector_backend="skip",
        enforce_detection=False
    )
    v = np.array(rep[0]["embedding"], dtype=np.float32)
    v /= (np.linalg.norm(v) + 1e-9)
    return v

def classify_knn(v_unit):
    sims = G_VECS @ v_unit
    idx = np.argpartition(-sims, K)[:K]
    idx = idx[np.argsort(-sims[idx])]
    top_sims   = sims[idx]
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

# ---- Attendance helper ----
def should_log(name: str) -> bool:
    """
    Allow only one log per person per camera activation.
    """
    if name in st.session_state.logged_names:
        return False
    st.session_state.logged_names.add(name)
    return True

# ====== STATE INIT ======
if "run" not in st.session_state: 
    st.session_state.run = False
if "logs" not in st.session_state:
    st.session_state.logs = []          # list of {timestamp, name, direction, event_name}
if "last_seen" not in st.session_state:
    st.session_state.last_seen = {}     # name -> last log datetime (kept for compatibility)
if "event_name" not in st.session_state:
    st.session_state.event_name = "My Event"
if "logged_names" not in st.session_state:
    st.session_state.logged_names = set()  # per-activation memory of who has been logged
if "prev_run" not in st.session_state:
    st.session_state.prev_run = False

# ====== UI ======
st.text_input(
    "Event name:",
    key="event_name",
    placeholder="e.g., CS Orientation 2025"
)
direction = st.radio(
    "Current scan mode (IN/OUT):",
    ["IN", "OUT"],
    horizontal=True
)

st.session_state.run = st.toggle("Start camera", value=st.session_state.run)

# Detect a new activation to reset per-run memory
if st.session_state.run and not st.session_state.prev_run:
    st.session_state.logged_names = set()
st.session_state.prev_run = st.session_state.run

if st.button("Reset logs"):
    st.session_state.logs = []
    st.session_state.last_seen = {}
    st.session_state.logged_names = set()
    st.success("Attendance logs reset.")

# ====== VIDEO PROCESSOR (WebRTC) ======
class AttendanceProcessor:
    def __init__(self):
        self.frame_i = -1
        self.last_results = None
        self.event_name = ""
        self.direction = ""

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Mirror (selfie)
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        # --- YOLO scheduling (from second code) ---
        self.frame_i = (self.frame_i + 1) % (FRAME_SKIP + 1)
        run_det = (self.frame_i == 0)

        if run_det:
            res = DET(
                img,
                conf=0.20,        # lower conf → more boxes found
                iou=0.70,         # keep more overlapping faces
                imgsz=960,        # bigger net input → small faces show up
                max_det=50,       # allow many faces
                verbose=False
            )[0]
            self.last_results = res.boxes if (res is not None and res.boxes is not None) else None

        if self.last_results is not None:
            xyxy = self.last_results.xyxy.cpu().numpy()
            for b in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[b].astype(int)
                x1, y1, x2, y2 = expand(x1, y1, x2, y2, w, h, PAD)

                bw, bh = x2 - x1, y2 - y1
                if bw < 40 or bh < 40:   # smaller than your old MIN_BOX=60
                    continue

                face_bgr = img[y1:y2, x1:x2]
                if face_bgr.size == 0:
                    continue

                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                face_rgb = center_square(face_rgb)

                v = embed_rgb_arcface(face_rgb)
                name, top1 = classify_knn(v)

                # ---- Logging: push to queue, main thread dedupes ----
                if name != "Unknown":
                    log_queue.put({
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                        "name": name,
                        "direction": self.direction,
                        "event_name": self.event_name,
                    })

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {top1:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0, 0, 0), -1)
                cv2.putText(img, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ====== START WEBRTC CAMERA (only when toggle is ON) ======
ctx = None
if st.session_state.run:
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="attendance",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=AttendanceProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.event_name = st.session_state.event_name
        ctx.video_processor.direction = direction

# ====== CONSUME LOG QUEUE & UPDATE SESSION LOGS ======
while not log_queue.empty():
    try:
        data = log_queue.get_nowait()
    except queue.Empty:
        break

    if should_log(data["name"]):
        st.session_state.logs.append(data)

# Optional: force rerun while camera is playing so logs update live
if ctx is not None and ctx.state.playing:
    time.sleep(1)
    st.rerun()

# ====== ATTENDANCE LOGS ======
st.subheader("Attendance logs")

if st.session_state.logs:
    df = pd.DataFrame(st.session_state.logs)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="attendance_logs.csv",
        mime="text/csv"
    )
else:
    st.info("No attendance logs yet. Start the camera to begin logging.")
