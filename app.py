import time
import datetime as dt
import cv2
import numpy as np
import streamlit as st
import pandas as pd
from ultralytics import YOLO
from deepface import DeepFace
from pathlib import Path

# ====== CONFIGURATION ======
ROOT = Path(__file__).resolve().parent
YOLO_WEIGHTS = str(ROOT / "yolov8s-face-lindevs.pt")
G_PATH = str(ROOT / "gallery_arcface.npz")

PAD = 0.0
FRAME_SKIP = 2  # Run detection every 2 frames to save FPS

# ArcFace k-NN params
K = 5
SIM_THRESH = 0.40
MARGIN = 0.06
CLASS_MINSUM = 0.50

st.set_page_config(page_title="CCS Attendance", layout="wide")

# ====== LOAD RESOURCES (Cached) ======
@st.cache_resource
def load_models():
    # Load YOLO
    det_model = YOLO(YOLO_WEIGHTS)
    
    # Warmup DeepFace (builds model once)
    DeepFace.build_model("ArcFace")
    return det_model

try:
    DET = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Load Gallery
if not Path(G_PATH).exists():
    st.error(f"Gallery file not found at: {G_PATH}")
    st.info("Please run your gallery builder script first to generate 'gallery_arcface.npz'.")
    st.stop()
else:
    G = np.load(G_PATH, allow_pickle=True)
    G_VECS = G["vecs"].astype(np.float32)
    G_LABELS = G["labels"].tolist()
    NAMES = G["classes"].tolist()

# ====== HELPER FUNCTIONS ======
def embed_rgb_arcface(rgb):
    """RGB uint8 face crop -> L2-normalized ArcFace vector."""
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

def should_log(name):
    """Prevents spamming the log for the same person in the same session."""
    if name in st.session_state.logged_names:
        return False
    st.session_state.logged_names.add(name)
    return True

# ====== SESSION STATE SETUP ======
if "logs" not in st.session_state:
    st.session_state.logs = []
if "logged_names" not in st.session_state:
    st.session_state.logged_names = set()

# ====== SIDEBAR CONTROLS ======
with st.sidebar:
    st.title("Controls")
    
    event_name = st.text_input("Event Name", value="CCS Event 2025")
    direction = st.radio("Scan Mode", ["IN", "OUT"], horizontal=True)
    
    # Camera Control
    run_camera = st.toggle("Start Camera", value=False)
    
    st.divider()
    
    if st.button("Clear Logs & Reset Memory"):
        st.session_state.logs = []
        st.session_state.logged_names = set()
        st.success("Memory cleared!")

    st.divider()
    st.write("### Download Data")
    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV", 
            csv, 
            "attendance.csv", 
            "text/csv", 
            key="dl_csv"
        )

# ====== MAIN APP LOGIC ======
st.title("🎥 Face Recognition Attendance")

col1, col2 = st.columns([2, 1])

with col1:
    frame_placeholder = st.empty()

with col2:
    st.subheader("Live Logs")
    log_placeholder = st.empty()

# Camera Loop
if run_camera:
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    # Try to set resolution for better performance/quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_i = -1
    last_results = None
    
    stop_button_pressed = False
    
    while cap.isOpened() and run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera.")
            break

        # Mirror image
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        # --- Frame Skipping for Performance ---
        frame_i = (frame_i + 1) % (FRAME_SKIP + 1)
        run_det = (frame_i == 0)

        if run_det:
            # YOLO Inference
            res = DET(
                frame, 
                conf=0.4,      # Higher conf to reduce false positives
                iou=0.5, 
                imgsz=640, 
                max_det=10, 
                verbose=False
            )[0]
            last_results = res.boxes

        # --- Processing Detections ---
        if last_results is not None:
            xyxy = last_results.xyxy.cpu().numpy()
            
            for box in xyxy:
                x1, y1, x2, y2 = box.astype(int)
                x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, PAD)

                # Skip tiny faces
                if (x2 - x1) < 60 or (y2 - y1) < 60:
                    continue

                # Crop Face
                face_bgr = frame[y1:y2, x1:x2]
                if face_bgr.size == 0: continue
                
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                
                # Recognition (only runs if we are detecting this frame to save speed)
                # OR run every frame if you want smoother tracking
                if run_det: 
                    v = embed_rgb_arcface(face_rgb)
                    name, conf = classify_knn(v)
                    
                    # Log Attendance
                    if name != "Unknown" and should_log(name):
                        st.session_state.logs.append({
                            "Time": dt.datetime.now().strftime("%H:%M:%S"),
                            "Name": name,
                            "Event": event_name,
                            "Status": direction
                        })
                else:
                    # Use previous name/conf if skipping frames (simplification)
                    # Ideally, you would track IDs, but for simple use:
                    name, conf = "Scanning...", 0.0

                # Draw UI
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                if name == "Scanning...": color = (200, 200, 200)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Update Streamlit Image
        # Convert BGR to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update Live Log Table
        if st.session_state.logs:
            log_placeholder.dataframe(
                pd.DataFrame(st.session_state.logs).iloc[::-1], # Show newest on top
                height=400, 
                hide_index=True
            )
        
        # Check if the user unchecked the toggle
        # Note: In Streamlit, the loop usually blocks the UI update. 
        # Only a page rerun stops this. 
        # If you uncheck the box in the sidebar, Streamlit triggers a rerun, 
        # causing 'run_camera' to become False at the top of the NEXT execution.
        # We rely on Streamlit's reactive nature here.

    cap.release()
else:
    st.info("Camera is off. Toggle the sidebar switch to start.")
    if st.session_state.logs:
        st.dataframe(pd.DataFrame(st.session_state.logs))
