import  time
import datetime as dt  # <-- added
import cv2, numpy as np, streamlit as st
import pandas as pd    # <-- added
from ultralytics import YOLO
from deepface import DeepFace
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

st.set_page_config(page_title="Face ID (ArcFace gallery k-NN)", layout="centered")
frame_ph = st.empty()


ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = str(ROOT / "yolov8s-face-lindevs.pt")

PAD        = 0.0
MIN_BOX    = 60

# Run detector every N frames (0 = every frame)
FRAME_SKIP = 2        # try 2; increase if CPU is slow
skip_mod   = FRAME_SKIP + 1
frame_i    = -1       # ensures we run on the very first frame
last_results = None

# ---- ArcFace k-NN params ----
K            = 5
SIM_THRESH   = 0.40
MARGIN       = 0.06
CLASS_MINSUM = 0.50

# ---- Attendance logging params ----
MIN_GAP_SECONDS = 10  # minimum seconds between logs for same person (no longer used)


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
    # FIX IS HERE: Use 'model_name' string, NOT 'model' object
    rep = DeepFace.represent(
        img_path=rgb,
        model_name="ArcFace",     # <--- Changed from model=ARC
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

def open_cam():
    for be in [cv2.CAP_MSMF, cv2.CAP_ANY]:
        for idx in [0,1,2]:
            cap = cv2.VideoCapture(idx, be)
            if not cap.isOpened(): cap.release(); continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
            cap.set(cv2.CAP_PROP_FPS,30); time.sleep(0.2)
            ok,_ = cap.read()
            if ok: return cap
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUY2'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            time.sleep(0.2); ok,_ = cap.read()
            if ok: return cap
            cap.release()
    return None

# ---- Attendance helper ----
def should_log(name: str) -> bool:
    """
    Allow only one log per person per camera activation.
    """
    if name in st.session_state.logged_names:
        return False
    st.session_state.logged_names.add(name)
    return True

# ====== UI ======
if "run" not in st.session_state: 
    st.session_state.run = False
if "logs" not in st.session_state:
    st.session_state.logs = []          # list of {timestamp, name, direction, event_name}
if "last_seen" not in st.session_state:
    st.session_state.last_seen = {}     # name -> last log datetime (kept for compatibility)
if "event_name" not in st.session_state:
    st.session_state.event_name = "My Event"
if "logged_names" not in st.session_state:
    st.session_state.logged_names = set()  # <- per-activation memory of who has been logged

# Event + direction controls
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

if st.button("Reset logs"):
    st.session_state.logs = []
    st.session_state.last_seen = {}
    st.session_state.logged_names = set()   # <- also clear per-run memory
    st.success("Attendance logs reset.")

frame_ph = st.empty()

# Open camera if starting

# --- LIVE VIDEO FROM BROWSER (WebRTC) ---
import av  # required by streamlit-webrtc


class FaceTransformer(VideoTransformerBase):
    def __init__(self):
        self.det = DET  # your cached YOLO
        self.direction = direction  # IN / OUT from your radio

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # BGR
        h, w = img.shape[:2]

        # YOLO detect
        res = self.det(img, conf=0.20, iou=0.70, imgsz=960, max_det=50, verbose=False)[0]
        boxes = res.boxes if (res is not None and res.boxes is not None) else None

        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            for b in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[b].astype(int)
                x1, y1, x2, y2 = expand(x1, y1, x2, y2, w, h, PAD)
                if (x2 - x1) < 40 or (y2 - y1) < 40:
                    continue

                face = img[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_rgb = center_square(face_rgb)

                # ArcFace -> kNN
                v = embed_rgb_arcface(face_rgb)
                name, top1 = classify_knn(v)

                # Attendance log (once per activation)
                if name != "Unknown" and should_log(name):
                    st.session_state.logs.append({
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                        "name": name,
                        "direction": self.direction,
                        "event_name": st.session_state.event_name
                    })

                # Draw
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {top1:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0, 0, 0), -1)
                cv2.putText(img, label, (x1 + 4, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return img

# Start only when your toggle is ON
if st.session_state.get("run", False):
    webrtc_streamer(
        key="attendance-webrtc",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=FaceTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
else:
    st.info("Toggle **Start camera** to begin the live stream.")


# ====== ATTENDANCE LOGS (shown when camera loop stops) ======
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


# # ====== LIVE LOOP ======
# while st.session_state.get("run", False):
#     cap = st.session_state.cap
#     ok, frame = cap.read()
#     if not ok: break
#
#     # Mirror (Selfie view)
#     frame = cv2.flip(frame, 1)
#     h, w = frame.shape[:2]
#
#     # CHANGE 2: Frame Skipping Logic
#     skip_counter += 1
#     if skip_counter % FRAME_SKIP == 0:
#         # Run AI Detection (Expensive part)
#         # CHANGE 3: Lower confidence slightly to catch faces faster
#         res = DET(frame, conf=0.30, iou=0.65, max_det=10, verbose=False)[0]
#         last_results = res.boxes
#     
#     # Draw (Always draw, even if we skipped detection this frame)
#     if last_results is not None:
#         for box, conf in zip(last_results.xyxy.cpu().numpy(), last_results.conf.cpu().numpy()):
#             x1,y1,x2,y2 = box.astype(int)
#             x1,y1,x2,y2 = expand(x1,y1,x2,y2,w,h, PAD)
#             
#             # Skip tiny boxes
#             if (x2-x1) < MIN_BOX or (y2-y1) < MIN_BOX: continue
#
#             # OPTIONAL: Only run ArcFace recognition every few frames too if still slow
#             # For now, we run it whenever we have a box, but you could optimize this further
#             face_bgr = frame[y1:y2, x1:x2]
#             if face_bgr.size == 0: continue
#
#             face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
#             face_rgb = center_square(face_rgb)
#
#             # Recognition
#             v = embed_rgb_arcface(face_rgb)
#             name, top1 = classify_knn(v)
#
#             # Drawing Code
#             color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#             cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
#             
#             label = f"{name} {top1:.2f}"
#             (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
#             cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+8, y1), (0,0,0), -1)
#             cv2.putText(frame, label, (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
#
#     # CHANGE 4: Fix Blue Tint
#     # Pass 'frame' (BGR) directly to imencode. Do NOT convert to RGB first.
#     _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
#     
#     # Streamlit renders the JPEG bytes directly
#     frame_ph.image(jpg.tobytes(), channels="BGR", output_format="JPEG", use_container_width=True)
#     
#     # Tiny sleep to prevent UI freeze
#     time.sleep(0.001)

