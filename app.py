from __future__ import annotations

import json
import math
import tempfile
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from ultralytics import YOLO

# Project layout expected by this application:
# E:\TrafficCollisionProject\app.py
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
DETECTOR_PATH = MODEL_DIR / "yolov8n.pt"
TRACKER_PATH = ROOT / "configs" / "bytetrack_low_conf.yaml"
CNN_DIR = MODEL_DIR / "phase_6_3_architecture_comparison"
SEQUENCE_DIR = ROOT / "processed" / "fixed_rate_sequences"
SCHEMA_PATH = SEQUENCE_DIR / "sequence_schema.json"
PREPROCESS_PATH = SEQUENCE_DIR / "preprocessing_config.json"

HORIZONS = (500, 1000, 1500)
SEQUENCE_LENGTH = 50
INTERVAL = 0.2
DIRECT_TOLERANCE = 0.05
MAX_INTERPOLATION_GAP = 0.30
MAX_COUNT_HOLD_AGE = 0.30
MIN_COVERAGE = 90.0
MAX_UNAVAILABLE_RUN = 5
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DIAGONAL = math.sqrt(FRAME_WIDTH**2 + FRAME_HEIGHT**2)
VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}
CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COUNT_FEATURES = {
    "vehicle_count", "bicycle_count", "car_count", "motorcycle_count", "bus_count", "truck_count",
    "interaction_pair_count", "approaching_pair_count", "nearby_pair_count", "tracked_object_count",
    "person_object_count", "bicycle_object_count", "car_object_count", "motorcycle_object_count",
    "bus_object_count", "truck_object_count",
}
MOTION_FEATURES = [
    "center_x_norm", "center_y_norm", "width_norm", "height_norm", "bbox_area_norm",
    "velocity_x_norm", "velocity_y_norm", "speed_norm", "acceleration_x_norm",
    "acceleration_y_norm", "acceleration_norm",
]


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(64, 64, num_layers=2, batch_first=True, dropout=0.3, bidirectional=False)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        _, (hidden, _) = self.lstm(x)
        return self.classifier(self.dropout(hidden[-1])).squeeze(1)


def read_schema():
    if not SCHEMA_PATH.exists() or not PREPROCESS_PATH.exists():
        raise FileNotFoundError(
            "Missing training artifacts. Expected processed/fixed_rate_sequences/"
            "sequence_schema.json and preprocessing_config.json."
        )
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    config = json.loads(PREPROCESS_PATH.read_text(encoding="utf-8"))
    base = schema.get("feature_columns") or schema.get("base_features") or config.get("feature_columns")
    missing = schema.get("missingness_features") or config.get("missingness_features")
    if not base or not missing or len(base) != 66 or len(missing) != 30:
        raise ValueError("Could not identify the expected 66 base features and 30 missingness features in the schema/config JSON.")
    if not all(k in config for k in ("medians", "means", "stds")):
        raise ValueError("Preprocessing JSON must contain medians, means and stds.")
    return list(base), list(missing), config


def load_models(device):
    models = {}
    for horizon in HORIZONS:
        path = CNN_DIR / f"cnn_lstm_{horizon}ms.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {path}")
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        model = CNNLSTM().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models[horizon] = model
    return models


def longest_true_run(mask):
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def resample_causal(df: pd.DataFrame, feature_columns: list[str], endpoint: float):
    df = df[df.timestamp_sec <= endpoint + 1e-9].sort_values("timestamp_sec")
    if df.empty:
        return None, 0.0, 0
    ts = df.timestamp_sec.to_numpy(dtype=float)
    vals = df[feature_columns].to_numpy(dtype=float)
    grid = endpoint - np.arange(SEQUENCE_LENGTH - 1, -1, -1, dtype=float) * INTERVAL
    out = np.full((SEQUENCE_LENGTH, len(feature_columns)), np.nan, dtype=np.float32)
    count_idx = [i for i, f in enumerate(feature_columns) if f in COUNT_FEATURES]
    cont_idx = [i for i, f in enumerate(feature_columns) if f not in COUNT_FEATURES]
    for step, target in enumerate(grid):
        right = int(np.searchsorted(ts, target, side="left"))
        left = right - 1
        if right < len(ts) and np.isclose(ts[right], target, atol=1e-8):
            out[step] = vals[right]
            continue
        if left >= 0 and 0 <= target - ts[left] <= DIRECT_TOLERANCE:
            out[step] = vals[left]
            continue
        if left >= 0 and right < len(ts):
            gap = ts[right] - ts[left]
            if 0 < gap <= MAX_INTERPOLATION_GAP:
                alpha = (target - ts[left]) / gap
                for j in cont_idx:
                    a, b = vals[left, j], vals[right, j]
                    if np.isfinite(a) and np.isfinite(b):
                        out[step, j] = a + alpha * (b - a)
        if left >= 0 and 0 <= target - ts[left] <= MAX_COUNT_HOLD_AGE:
            for j in count_idx:
                if np.isfinite(vals[left, j]):
                    out[step, j] = vals[left, j]
    available = np.isfinite(out).any(axis=1)
    coverage = float(available.mean() * 100.0)
    if coverage < MIN_COVERAGE or longest_true_run(~available) > MAX_UNAVAILABLE_RUN:
        return None, coverage, longest_true_run(~available)
    return out, coverage, longest_true_run(~available)


def transform(raw, feature_columns, missing_features, config):
    medians = np.asarray(config["medians"], dtype=np.float32)
    means = np.asarray(config["means"], dtype=np.float32)
    stds = np.asarray(config["stds"], dtype=np.float32)
    if len(medians) != len(feature_columns) or len(means) != len(feature_columns) or len(stds) != len(feature_columns):
        raise ValueError("Preprocessing arrays do not match the 66 base features.")
    missing = (~np.isfinite(raw)).astype(np.float32)
    imputed = np.where(np.isfinite(raw), raw, medians)
    normalized = (imputed - means) / stds
    indices = [feature_columns.index(f) for f in missing_features]
    return np.concatenate([normalized, missing[:, indices]], axis=1).astype(np.float32)


def safe_stats(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}


def build_scene_row(observations, previous_motion, timestamp):
    # observations: one dict per current tracked object; motion is computed at 5 Hz.
    objs = []
    for item in observations:
        tid = item["track_id"]
        cx, cy, x1, y1, x2, y2 = item["cx"], item["cy"], item["x1"], item["y1"], item["x2"], item["y2"]
        prev = previous_motion.get(tid)
        vx = vy = ax = ay = np.nan
        if prev is not None:
            dt = timestamp - prev["timestamp"]
            if 0 < dt <= 0.6:
                vx = ((cx - prev["cx"]) / FRAME_WIDTH) / dt
                vy = ((cy - prev["cy"]) / FRAME_HEIGHT) / dt
                if np.isfinite(prev["vx"]) and np.isfinite(prev["vy"]):
                    ax = (vx - prev["vx"]) / dt
                    ay = (vy - prev["vy"]) / dt
        speed = math.sqrt(vx * vx + vy * vy) if np.isfinite(vx) and np.isfinite(vy) else np.nan
        accel = math.sqrt(ax * ax + ay * ay) if np.isfinite(ax) and np.isfinite(ay) else np.nan
        item_motion = {
            "center_x_norm": cx / FRAME_WIDTH, "center_y_norm": cy / FRAME_HEIGHT,
            "width_norm": (x2 - x1) / FRAME_WIDTH, "height_norm": (y2 - y1) / FRAME_HEIGHT,
            "bbox_area_norm": max(0.0, (x2 - x1) * (y2 - y1)) / (FRAME_WIDTH * FRAME_HEIGHT),
            "velocity_x_norm": vx, "velocity_y_norm": vy, "speed_norm": speed,
            "acceleration_x_norm": ax, "acceleration_y_norm": ay, "acceleration_norm": accel,
        }
        item["motion"] = item_motion
        objs.append(item)
        previous_motion[tid] = {"timestamp": timestamp, "cx": cx, "cy": cy, "vx": vx, "vy": vy}

    vehicle_objs = [o for o in objs if o["class_id"] in VEHICLE_CLASS_IDS]
    pair_distances, ttc_values, closing_values = [], [], []
    approaching = nearby = 0
    for i in range(len(vehicle_objs)):
        for j in range(i + 1, len(vehicle_objs)):
            a, b = vehicle_objs[i], vehicle_objs[j]
            rx = (b["cx"] - a["cx"]) / DIAGONAL
            ry = (b["cy"] - a["cy"]) / DIAGONAL
            dist = math.hypot(rx, ry)
            pair_distances.append(dist)
            av, bv = a["motion"], b["motion"]
            vals = [av["velocity_x_norm"], av["velocity_y_norm"], bv["velocity_x_norm"], bv["velocity_y_norm"]]
            if dist > 0 and all(np.isfinite(v) for v in vals):
                rvx = (bv["velocity_x_norm"] - av["velocity_x_norm"]) * FRAME_WIDTH / DIAGONAL
                rvy = (bv["velocity_y_norm"] - av["velocity_y_norm"]) * FRAME_HEIGHT / DIAGONAL
                closing = -(rx * rvx + ry * rvy) / dist
                closing_values.append(closing)
                if closing > 0.005:
                    approaching += 1
                    ttc = dist / closing
                    if 0 < ttc <= 10:
                        ttc_values.append(ttc)
            if dist <= 0.10:
                nearby += 1

    row = {
        "vehicle_count": len(vehicle_objs),
        "bicycle_count": sum(o["class_id"] == 1 for o in vehicle_objs),
        "car_count": sum(o["class_id"] == 2 for o in vehicle_objs),
        "motorcycle_count": sum(o["class_id"] == 3 for o in vehicle_objs),
        "bus_count": sum(o["class_id"] == 5 for o in vehicle_objs),
        "truck_count": sum(o["class_id"] == 7 for o in vehicle_objs),
        "interaction_pair_count": len(pair_distances),
        "approaching_pair_count": approaching,
        "nearby_pair_count": nearby,
        "minimum_pair_distance_norm": min(pair_distances) if pair_distances else np.nan,
        "mean_pair_distance_norm": float(np.mean(pair_distances)) if pair_distances else np.nan,
        "minimum_ttc_proxy_sec": min(ttc_values) if ttc_values else np.nan,
        "mean_ttc_proxy_sec": float(np.mean(ttc_values)) if ttc_values else np.nan,
        "maximum_closing_speed_norm": max(closing_values) if closing_values else np.nan,
        "mean_closing_speed_norm": float(np.mean(closing_values)) if closing_values else np.nan,
        "tracked_object_count": len(objs),
        "person_object_count": sum(o["class_id"] == 0 for o in objs),
        "bicycle_object_count": sum(o["class_id"] == 1 for o in objs),
        "car_object_count": sum(o["class_id"] == 2 for o in objs),
        "motorcycle_object_count": sum(o["class_id"] == 3 for o in objs),
        "bus_object_count": sum(o["class_id"] == 5 for o in objs),
        "truck_object_count": sum(o["class_id"] == 7 for o in objs),
    }
    for feature in MOTION_FEATURES:
        stats = safe_stats([o["motion"][feature] for o in objs])
        for stat, value in stats.items():
            row[f"{feature}_{stat}"] = value
    return row, objs


def draw_overlay(frame, observations, trails):
    for obj in observations:
        tid = obj["track_id"]
        x1, y1, x2, y2 = map(int, (obj["x1"], obj["y1"], obj["x2"], obj["y2"]))
        name = CLASS_NAMES.get(obj["class_id"], str(obj["class_id"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2)
        cv2.putText(frame, f"{name} ID:{tid}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 2)
        trail = list(trails[tid])
        for p, q in zip(trail, trail[1:]):
            cv2.line(frame, p, q, (0, 200, 255), 2)
    return frame


def process_video(video_path: Path, output_path: Path, progress, status):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_features, missing_features, config = read_schema()
    models = load_models(device)
    detector = YOLO(str(DETECTOR_PATH))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the uploaded video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise RuntimeError("Could not read video dimensions.")
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create annotated MP4 output.")

    scene_rows = []
    previous_motion = {}
    trails = defaultdict(lambda: deque(maxlen=30))
    latest_observations = []
    latest_predictions = None
    latest_coverage = None
    frame_idx = 0
    next_sample_time = 0.0
    sample_number = 0
    sample_stride_note = "Detector/tracker evaluated at 5 Hz; the last detection overlay is held between sample frames."
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_idx / fps
            if timestamp + 1e-9 >= next_sample_time:
                # One persistent tracker stream; ByteTrack sees each sampled frame in temporal order.
                result = detector.track(
                    frame, persist=True, tracker=str(TRACKER_PATH), conf=0.05, iou=0.50,
                    imgsz=640, classes=[0, 1, 2, 3, 5, 7], device=device, verbose=False,
                )[0]
                observations = []
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    clss = boxes.cls.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()
                    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full(len(xyxy), -1)
                    for box, cls_id, conf, tid in zip(xyxy, clss, confs, ids):
                        if tid < 0:
                            continue
                        x1, y1, x2, y2 = map(float, box)
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        observations.append({"track_id": int(tid), "class_id": int(cls_id), "confidence": float(conf),
                                             "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": cx, "cy": cy})
                        trails[int(tid)].append((int(cx), int(cy)))
                latest_observations = observations
                row, _ = build_scene_row(observations, previous_motion, timestamp)
                row["timestamp_sec"] = timestamp
                scene_rows.append(row)
                sample_number += 1
                next_sample_time = sample_number * INTERVAL

                # Once 50 samples are available, produce a causal rolling prediction.
                if sample_number >= SEQUENCE_LENGTH:
                    scene_df = pd.DataFrame(scene_rows)
                    raw, coverage, missing_run = resample_causal(scene_df, base_features, timestamp)
                    latest_coverage = coverage
                    if raw is not None:
                        model_input = transform(raw, base_features, missing_features, config)
                        tensor = torch.from_numpy(model_input[None, ...]).to(device)
                        latest_predictions = {}
                        with torch.no_grad():
                            for horizon, model in models.items():
                                score = float(torch.sigmoid(model(tensor)).item())
                                latest_predictions[horizon] = score
                    else:
                        latest_predictions = None
                status.text(f"Processing video: {timestamp:.1f}s | sampled frames: {sample_number}")

            annotated = frame.copy()
            # Redraw the most recent sample's boxes and full available trajectory trails.
            annotated = draw_overlay(annotated, latest_observations, trails)
            if latest_predictions is None:
                banner = "WARM-UP / INSUFFICIENT HISTORY" if sample_number < SEQUENCE_LENGTH else "SEQUENCE QUALITY CHECK FAILED"
                cv2.putText(annotated, banner, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 255), 2)
            else:
                text_y = 35
                for horizon in HORIZONS:
                    score = latest_predictions[horizon]
                    verdict = "COLLISION PREDICTED" if score >= 0.5 else "NO COLLISION PREDICTED"
                    color = (0, 0, 255) if score >= 0.5 else (0, 200, 80)
                    cv2.putText(annotated, f"{horizon} ms: {verdict} ({score:.3f})", (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
                    text_y += 27
                cv2.putText(annotated, "EXPERIMENTAL MODEL OUTPUT - NOT A CONFIRMED EVENT", (20, text_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 190, 255), 2)
            writer.write(annotated)
            frame_idx += 1
            if total_frames > 0 and frame_idx % 10 == 0:
                progress.progress(min(frame_idx / total_frames, 1.0))
    finally:
        cap.release()
        writer.release()
    progress.progress(1.0)
    return {
        "frames": frame_idx,
        "samples": sample_number,
        "duration": frame_idx / fps,
        "predictions": latest_predictions,
        "coverage": latest_coverage,
        "device": device,
        "sampling_note": sample_stride_note,
    }


def make_browser_mp4(source_path: Path, target_path: Path):
    """Convert OpenCV's MP4V output to browser-friendly H.264 MP4."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg, "-y", "-i", str(source_path), "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0 or not target_path.exists() or target_path.stat().st_size == 0:
            raise RuntimeError(completed.stderr[-1500:])
        return target_path
    except Exception:
        # Keep a downloadable fallback if H.264 conversion is unavailable.
        return source_path


st.set_page_config(
    page_title="Traffic Intelligence | Collision Forecast",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stHeader"] {background: transparent;}
    .block-container {max-width: 1440px; padding-top: 2.6rem; padding-bottom: 3rem;}
    .hero {padding: 1.35rem 1.6rem; border: 1px solid rgba(128,128,128,.25); border-radius: 18px;
           background: linear-gradient(115deg, rgba(35,75,115,.13), rgba(20,150,130,.07)); margin-bottom: 1.2rem;}
    .eyebrow {font-size: .75rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; opacity: .7;}
    .hero h1 {font-size: 2rem; margin: .25rem 0 .35rem 0; letter-spacing: -.03em;}
    .hero p {margin: 0; opacity: .76;}
    .section-title {font-size: 1.08rem; font-weight: 700; margin: .6rem 0 .7rem 0;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: 1rem; border-radius: 14px; background: rgba(128,128,128,.045);}
    .verdict {border-radius: 14px; padding: 1rem 1.1rem; border: 1px solid rgba(128,128,128,.25); min-height: 112px;}
    .verdict-high {background: rgba(220,53,69,.09); border-color: rgba(220,53,69,.38);}
    .verdict-low {background: rgba(25,135,84,.08); border-color: rgba(25,135,84,.35);}
    .verdict-label {font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; opacity: .7; font-weight: 700;}
    .verdict-main {font-size: 1.05rem; font-weight: 800; margin: .3rem 0;}
    .verdict-score {font-size: .85rem; opacity: .8;}
    .stButton button, .stDownloadButton button {border-radius: 10px; min-height: 2.7rem; font-weight: 650;}
</style>
<div class="hero">
  <div class="eyebrow">Computer vision · Predictive analytics</div>
  <h1>Traffic Collision Forecast</h1>
  <p>Upload a road video to detect and track road users, inspect trajectories, and review collision forecasting performance.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Research prototype. ‘Collision predicted’ is a model classification at the selected threshold, "
    "not confirmation that a crash occurred. Scores are uncalibrated and must not be used as safety alerts."
)

with st.sidebar:
    st.markdown("## Processing pipeline")
    st.caption("Stages used for each uploaded video")
    for stage in (
        "01 · Video input",
        "02 · YOLO object detection",
        "03 · ByteTrack object tracking",
        "04 · Trajectory and interaction features",
        "05 · 50-frame sequence preparation",
        "06 · CNN-LSTM collision forecasting",
        "07 · Annotated video output",
    ):
        st.markdown(f"- {stage}")
    st.divider()
    st.caption("Inference device: CUDA when available; otherwise CPU.")

st.markdown('<div class="section-title">01 · Select a traffic video</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv", "mpeg", "mpg"], label_visibility="collapsed")

if uploaded is not None:
    st.caption(f"Selected file: **{uploaded.name}** · {uploaded.size / (1024 * 1024):.1f} MB")
    if st.button("▶  Analyse video", type="primary", use_container_width=True):
        st.session_state.pop("traffic_result", None)
        st.session_state.pop("traffic_video_bytes", None)
        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            input_path = temp_dir / uploaded.name
            raw_output = temp_dir / "annotated_raw.mp4"
            browser_output = temp_dir / "annotated_h264.mp4"
            input_path.write_bytes(uploaded.getbuffer())
            progress = st.progress(0.0, text="Starting video analysis…")
            status = st.empty()
            try:
                result = process_video(input_path, raw_output, progress, status)
                final_output = make_browser_mp4(raw_output, browser_output)
                video_bytes = final_output.read_bytes()
                st.session_state["traffic_result"] = result
                st.session_state["traffic_video_bytes"] = video_bytes
                st.session_state["traffic_filename"] = uploaded.name
                st.success("Analysis complete. Review the annotated video and model evaluation summary below.")
            except Exception as exc:
                st.error(f"Processing failed: {exc}")
                st.exception(exc)

if st.session_state.get("traffic_result") and st.session_state.get("traffic_video_bytes"):
    result = st.session_state["traffic_result"]
    video_bytes = st.session_state["traffic_video_bytes"]
    st.divider()
    

    # Previously reported test-set Average Precision values for each TTE horizon.
    # Re-run the evaluation script and update these values whenever the checkpoints or test set change.
    horizon_ap = {500: 0.6837, 1000: 0.7978, 1500: 0.7481}
    collision_map = sum(horizon_ap.values()) / len(horizon_ap)

    map_col, ap1, ap2, ap3 = st.columns(4)

    st.markdown('<div class="section-title">03 · Annotated video</div>', unsafe_allow_html=True)
    st.video(video_bytes, format="video/mp4")
    st.download_button(
        "⬇  Download annotated video",
        data=video_bytes,
        file_name="annotated_traffic_video.mp4",
        mime="video/mp4",
        use_container_width=True,
        on_click="ignore",
    )
    st.markdown('<div class="section-title">03 · Processing summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Frames processed", f"{result['frames']:,}")
    m2.metric("5 Hz samples", f"{result['samples']:,}")
    m3.metric("Video duration", f"{result['duration']:.1f} s")
    m4.metric("Inference device", result["device"].upper())
    st.caption(result["sampling_note"])

st.divider()
st.caption("Traffic Collision Forecast · Experimental academic prototype · Not for real-world safety decisions")
