import cv2
import numpy as np
import os
import subprocess
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ForceTrack Bar Path API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Process at 50% resolution for a good speed/accuracy balance
SCALE = 0.5

# Maximum allowed jump between detections (in scaled pixels) before
# the detection is considered a false positive and discarded.
MAX_JUMP_PX = 100


# ── Kalman filter ─────────────────────────────────────────────────────────────

class KalmanFilter1D:
    """Minimal 1-D Kalman filter for smoothing a single coordinate."""

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 10.0):
        self.q = process_noise       # process noise covariance
        self.r = measurement_noise   # measurement noise covariance
        self.x = 0.0                 # state estimate
        self.p = 1.0                 # estimate error covariance
        self.initialized = False

    def update(self, measurement: float) -> float:
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x

        # Predict
        p_pred = self.p + self.q

        # Update
        k = p_pred / (p_pred + self.r)   # Kalman gain
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * p_pred
        return self.x

    def predict(self) -> float:
        """Advance state without a measurement (fills detection gaps)."""
        self.p += self.q
        return self.x


# ── Video helpers ─────────────────────────────────────────────────────────────

def get_video_rotation(path: str) -> int:
    """Return the clockwise rotation degrees encoded in the video's metadata."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            # Modern ffprobe exposes rotation in side_data_list
            for sd in stream.get("side_data_list", []):
                if sd.get("side_data_type") == "Display Matrix":
                    rot = int(sd.get("rotation", 0))
                    return rot  # may be negative (e.g. -90)
            # Older ffprobe / some containers put it in tags
            tags = stream.get("tags", {})
            if "rotate" in tags:
                return int(tags["rotate"])
    except Exception:
        pass
    return 0


def correct_rotation(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate *frame* to compensate for the metadata rotation."""
    # Normalise to 0 / 90 / 180 / 270
    deg = degrees % 360
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270 or deg == -90 % 360:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ── Detection helpers ─────────────────────────────────────────────────────────

def detect_barbell_plate(
    gray: np.ndarray,
    search_region: tuple | None = None,
) -> tuple[float, float, float] | None:
    """
    Detect the largest circle in *gray* (or within *search_region*).

    search_region: (x, y, w, h) in *gray* coordinates.
    Returns (cx, cy, radius) in *gray* coordinates, or None.
    """
    if search_region is not None:
        rx, ry, rw, rh = search_region
        rx = max(0, rx)
        ry = max(0, ry)
        rw = min(rw, gray.shape[1] - rx)
        rh = min(rh, gray.shape[0] - ry)
        if rw < 8 or rh < 8:
            return None
        roi = gray[ry : ry + rh, rx : rx + rw]
    else:
        roi = gray
        rx, ry = 0, 0

    blurred = cv2.GaussianBlur(roi, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=max(10, min(roi.shape[:2]) // 2),
    )
    if circles is None:
        return None

    # Pick the largest circle (most likely the plate)
    circles = np.round(circles[0]).astype(int)
    best = max(circles, key=lambda c: c[2])
    cx = float(best[0] + rx)
    cy = float(best[1] + ry)
    r  = float(best[2])
    return cx, cy, r


def is_valid_detection(
    cx: float, cy: float,
    last_cx: float, last_cy: float,
    max_jump: float = MAX_JUMP_PX,
) -> bool:
    """Return True when the new detection is within *max_jump* pixels of the last."""
    dist = np.hypot(cx - last_cx, cy - last_cy)
    return dist <= max_jump


# ── FastAPI endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "6.0"}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    params: str = Form(...),
    api_key: str = Form(default=""),
):
    p = json.loads(params)
    tap_time = float(p["tap_time"])
    cap_w    = int(p["cap_w"])
    cap_h    = int(p["cap_h"])
    orig_cx  = float(p["orig_cx"])
    orig_cy  = float(p["orig_cy"])
    box_hw   = float(p["box_hw"])
    box_hh   = float(p["box_hh"])

    content = await video.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(content)
        tmp = f.name

    try:
        # ── Rotation metadata ────────────────────────────────────────────────
        rotation_deg = get_video_rotation(tmp)

        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video")

        fps   = max(1.0, float(cap.get(cv2.CAP_PROP_FPS)) or 30.0)
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # pre-rotation
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # pre-rotation
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # After rotation, the logical frame dimensions may swap
        deg_norm = rotation_deg % 360
        if deg_norm in (90, 270):
            vid_w, vid_h = raw_h, raw_w
        else:
            vid_w, vid_h = raw_w, raw_h

        # Scaled processing dimensions
        proc_w = max(8, int(vid_w * SCALE))
        proc_h = max(8, int(vid_h * SCALE))

        def get_proc_frame(frm: np.ndarray) -> np.ndarray:
            frm = correct_rotation(frm, rotation_deg)
            return cv2.resize(frm, (proc_w, proc_h))

        # Map the user's tap point (in capture coords) → scaled proc coords
        sx = proc_w / cap_w
        sy = proc_h / cap_h
        tap_px = orig_cx * sx
        tap_py = orig_cy * sy
        search_r_x = max(4, int(box_hw * sx))
        search_r_y = max(4, int(box_hh * sy))

        # ── Seek to tap frame ────────────────────────────────────────────────
        start = max(0, int(tap_time * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, frame0 = cap.read()
        if not ret:
            raise HTTPException(status_code=400, detail="Cannot read tap frame")

        proc0 = get_proc_frame(frame0)
        gray0 = cv2.cvtColor(proc0, cv2.COLOR_BGR2GRAY)

        # Initial detection: search within the tap bounding box
        init_region = (
            int(tap_px - search_r_x),
            int(tap_py - search_r_y),
            search_r_x * 2,
            search_r_y * 2,
        )
        det = detect_barbell_plate(gray0, search_region=init_region)
        if det is None:
            # Widen search to full frame if the tight region fails
            det = detect_barbell_plate(gray0)
        if det is None:
            # Fall back to the tap point itself
            det = (tap_px, tap_py, min(search_r_x, search_r_y))

        last_cx, last_cy, last_r = det

        # Kalman filters for x and y independently
        kf_x = KalmanFilter1D()
        kf_y = KalmanFilter1D()
        smooth_x = kf_x.update(last_cx)
        smooth_y = kf_y.update(last_cy)

        t0 = cap.get(cv2.CAP_PROP_POS_MSEC)
        results = [{"t": t0, "x": smooth_x / proc_w, "y": smooth_y / proc_h}]

        # ── Main tracking loop ───────────────────────────────────────────────
        max_frames = min(total - start - 1, int(fps * 45))
        fn = start + 1

        while fn < start + 1 + max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            proc  = get_proc_frame(frame)
            gray  = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

            # Constrained search around last known position
            margin_x = max(search_r_x, int(last_r * 3))
            margin_y = max(search_r_y, int(last_r * 3))
            region = (
                int(last_cx - margin_x),
                int(last_cy - margin_y),
                margin_x * 2,
                margin_y * 2,
            )
            det = detect_barbell_plate(gray, search_region=region)

            if det is not None and is_valid_detection(det[0], det[1], last_cx, last_cy):
                last_cx, last_cy, last_r = det
                smooth_x = kf_x.update(last_cx)
                smooth_y = kf_y.update(last_cy)
            else:
                # No valid detection — advance Kalman prediction
                smooth_x = kf_x.predict()
                smooth_y = kf_y.predict()
                last_cx, last_cy = smooth_x, smooth_y

            results.append({
                "t": t_ms,
                "x": smooth_x / proc_w,
                "y": smooth_y / proc_h,
            })
            fn += 1

        cap.release()

        # Return original (pre-rotation) dimensions so the frontend can
        # reconstruct absolute positions if needed.
        return {"frames": results, "cap_w": raw_w, "cap_h": raw_h}

    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
