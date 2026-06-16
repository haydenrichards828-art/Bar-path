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

# ── Constants ─────────────────────────────────────────────────────────────────

SCALE        = 0.5          # process at 50 % resolution
API_KEY_ENV  = "BAR_PATH_API_KEY"
MAX_UPLOAD_B = 600 * 1024 * 1024   # 600 MB


# ── 2-D Kalman filter ─────────────────────────────────────────────────────────

class KalmanFilter2D:
    """Simple constant-position 2-D Kalman filter."""

    def __init__(self, process_noise: float = 2.0, measurement_noise: float = 8.0):
        self.kf = cv2.KalmanFilter(4, 2)          # state: [x, y, vx, vy]
        dt = 1.0
        self.kf.transitionMatrix    = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)
        self.kf.measurementMatrix   = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self.initialized = False

    def init(self, x: float, y: float):
        self.kf.statePost = np.array([[x], [y], [0.], [0.]], dtype=np.float32)
        self.initialized = True

    def update(self, x: float, y: float):
        if not self.initialized:
            self.init(x, y)
            return x, y
        self.kf.predict()
        meas = np.array([[x], [y]], dtype=np.float32)
        est  = self.kf.correct(meas)
        return float(est[0]), float(est[1])

    def predict(self):
        est = self.kf.predict()
        return float(est[0]), float(est[1])


# ── Video helpers ─────────────────────────────────────────────────────────────

def get_video_rotation(path: str) -> int:
    """Return the clockwise rotation degrees encoded in the video's metadata."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", path],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            for sd in stream.get("side_data_list", []):
                if sd.get("side_data_type") == "Display Matrix":
                    return int(sd.get("rotation", 0))
            tags = stream.get("tags", {})
            if "rotate" in tags:
                return int(tags["rotate"])
    except Exception:
        pass
    return 0


def correct_rotation(frame: np.ndarray, degrees: int) -> np.ndarray:
    deg = degrees % 360
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg in (270, 360 - 90):
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


# ── Hough-circle detection ────────────────────────────────────────────────────

def detect_barbell_plate(
    gray: np.ndarray,
    search_region: tuple | None = None,
    min_r: int = 5,
    max_r: int | None = None,
) -> tuple[float, float, float] | None:
    if search_region is not None:
        rx, ry, rw, rh = search_region
        rx = max(0, rx); ry = max(0, ry)
        rw = min(rw, gray.shape[1] - rx)
        rh = min(rh, gray.shape[0] - ry)
        if rw < 8 or rh < 8:
            return None
        roi = gray[ry:ry + rh, rx:rx + rw]
    else:
        roi = gray
        rx, ry = 0, 0

    if max_r is None:
        max_r = max(10, min(roi.shape[:2]) // 2)

    blurred = cv2.GaussianBlur(roi, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=20,
        param1=50, param2=30,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)
    best = max(circles, key=lambda c: c[2])
    return float(best[0] + rx), float(best[1] + ry), float(best[2])


# ── FastAPI endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.3.0"}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    params: str = Form(...),
    api_key: str = Form(default=""),
):
    # ── API-key guard ────────────────────────────────────────────────────────
    required_key = os.environ.get(API_KEY_ENV, "")
    if required_key and api_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    p        = json.loads(params)
    tap_time = float(p["tap_time"])
    cap_w    = int(p["cap_w"])
    cap_h    = int(p["cap_h"])
    orig_cx  = float(p["orig_cx"])
    orig_cy  = float(p["orig_cy"])
    box_hw   = float(p["box_hw"])
    box_hh   = float(p["box_hh"])

    # ── Upload-size guard ────────────────────────────────────────────────────
    content = await video.read()
    if len(content) > MAX_UPLOAD_B:
        raise HTTPException(status_code=413, detail="Video exceeds 600 MB limit")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(content)
        tmp = f.name

    try:
        rotation_deg = get_video_rotation(tmp)

        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video")

        fps   = max(1.0, float(cap.get(cv2.CAP_PROP_FPS)) or 30.0)
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        deg_norm = rotation_deg % 360
        vid_w, vid_h = (raw_h, raw_w) if deg_norm in (90, 270) else (raw_w, raw_h)

        proc_w = max(8, int(vid_w * SCALE))
        proc_h = max(8, int(vid_h * SCALE))

        def get_proc_frame(frm: np.ndarray) -> np.ndarray:
            frm = correct_rotation(frm, rotation_deg)
            return cv2.resize(frm, (proc_w, proc_h))

        sx = proc_w / cap_w
        sy = proc_h / cap_h
        tap_px     = orig_cx * sx
        tap_py     = orig_cy * sy
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

        init_region = (
            int(tap_px - search_r_x), int(tap_py - search_r_y),
            search_r_x * 2, search_r_y * 2,
        )
        det = detect_barbell_plate(gray0, search_region=init_region)
        if det is None:
            det = detect_barbell_plate(gray0)
        if det is None:
            det = (tap_px, tap_py, min(search_r_x, search_r_y))

        init_cx, init_cy, plate_r = det
        max_jump_sq = (plate_r * 5) ** 2   # jump-validation threshold

        # ── Initialise CSRT tracker ──────────────────────────────────────────
        tracker = cv2.TrackerCSRT_create()
        tr_x = int(init_cx - plate_r)
        tr_y = int(init_cy - plate_r)
        tr_w = int(plate_r * 2)
        tr_h = int(plate_r * 2)
        tracker.init(proc0, (tr_x, tr_y, tr_w, tr_h))

        # ── 2-D Kalman filter ────────────────────────────────────────────────
        kf = KalmanFilter2D()
        smooth_x, smooth_y = kf.update(init_cx, init_cy)

        last_cx, last_cy = init_cx, init_cy

        t0 = cap.get(cv2.CAP_PROP_POS_MSEC)
        results = [{"t": t0, "x": smooth_x / proc_w, "y": smooth_y / proc_h}]

        # ── Frame-skip interval (speed optimisation) ─────────────────────────
        skip = max(1, total // 600)

        # ── Main tracking loop ───────────────────────────────────────────────
        max_frames = min(total - start - 1, int(fps * 45))
        fn = start + 1

        while fn < start + 1 + max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Skip frames for speed, but still record a smoothed position
            if (fn - start) % skip != 0:
                sx_pred, sy_pred = kf.predict()
                results.append({"t": t_ms, "x": sx_pred / proc_w, "y": sy_pred / proc_h})
                fn += 1
                continue

            proc = get_proc_frame(frame)
            gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

            # ── CSRT update ──────────────────────────────────────────────────
            ok, bbox = tracker.update(proc)
            csrt_cx = csrt_cy = None
            if ok:
                bx, by, bw, bh = bbox
                csrt_cx = bx + bw / 2
                csrt_cy = by + bh / 2

            # ── Hough fallback inside search box ─────────────────────────────
            margin_x = max(search_r_x, int(plate_r * 3))
            margin_y = max(search_r_y, int(plate_r * 3))
            hough_region = (
                int(last_cx - margin_x), int(last_cy - margin_y),
                margin_x * 2, margin_y * 2,
            )
            hough_det = detect_barbell_plate(
                gray, search_region=hough_region,
                min_r=max(3, int(plate_r * 0.5)),
                max_r=int(plate_r * 2.5),
            )

            # ── Fuse: prefer CSRT when valid, else Hough, else predict ───────
            det_cx = det_cy = None

            if csrt_cx is not None:
                dx = csrt_cx - last_cx; dy = csrt_cy - last_cy
                if dx * dx + dy * dy <= max_jump_sq:
                    det_cx, det_cy = csrt_cx, csrt_cy

            if det_cx is None and hough_det is not None:
                hx, hy, hr = hough_det
                dx = hx - last_cx; dy = hy - last_cy
                if dx * dx + dy * dy <= max_jump_sq:
                    det_cx, det_cy = hx, hy
                    plate_r = hr          # update radius estimate
                    max_jump_sq = (plate_r * 5) ** 2

            if det_cx is not None:
                # Adaptive reinit: if CSRT drifted far from Hough, reinit
                if (csrt_cx is not None and hough_det is not None):
                    ddx = csrt_cx - hough_det[0]
                    ddy = csrt_cy - hough_det[1]
                    if ddx * ddx + ddy * ddy > max_jump_sq * 0.25:
                        tracker = cv2.TrackerCSRT_create()
                        nr = int(plate_r)
                        tracker.init(proc, (
                            int(det_cx - nr), int(det_cy - nr),
                            nr * 2, nr * 2,
                        ))

                smooth_x, smooth_y = kf.update(det_cx, det_cy)
                last_cx, last_cy = det_cx, det_cy
            else:
                smooth_x, smooth_y = kf.predict()
                last_cx, last_cy = smooth_x, smooth_y

            results.append({"t": t_ms, "x": smooth_x / proc_w, "y": smooth_y / proc_h})
            fn += 1

        cap.release()
        return {"frames": results, "cap_w": raw_w, "cap_h": raw_h}

    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
