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

SCALE        = 0.25          # processing resolution factor
MAX_BYTES    = 600 * 1024 * 1024  # 600 MB upload limit
MAX_SECONDS  = 45            # longest clip we'll process


# ── 2-D Kalman filter (x, y, vx, vy) ────────────────────────────────────────
def make_kalman() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost       = np.eye(4, dtype=np.float32)
    return kf


# ── ffprobe rotation tag ──────────────────────────────────────────────────────
def probe_rotation(path: str) -> int:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", path],
            stderr=subprocess.DEVNULL, timeout=10
        )
        data = json.loads(out)
        for s in data.get("streams", []):
            tags = s.get("tags", s.get("side_data_list", [{}])[0] if s.get("side_data_list") else {})
            rot  = tags.get("rotate") or tags.get("rotation")
            if rot is not None:
                return int(rot)
    except Exception:
        pass
    return 0


@app.get("/health")
def health():
    return {"status": "ok", "version": "6.0"}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    params: str = Form(...),
    api_key: str = Form(default="")
):
    # ── API key guard ─────────────────────────────────────────────────────────
    required_key = os.environ.get("BARPATH_API_KEY", "")
    if required_key and api_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # ── Parse params ──────────────────────────────────────────────────────────
    p         = json.loads(params)
    tap_time  = float(p["tap_time"])
    cap_w     = int(p["cap_w"])
    cap_h     = int(p["cap_h"])
    orig_cx   = float(p["orig_cx"])
    orig_cy   = float(p["orig_cy"])
    box_hw    = float(p["box_hw"])
    box_hh    = float(p["box_hh"])
    dot_off_x = float(p["dot_offset_x"])
    dot_off_y = float(p["dot_offset_y"])

    # ── Read & size-check upload ──────────────────────────────────────────────
    content = await video.read()
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Video exceeds 600 MB limit")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(content)
        tmp = f.name
    del content  # free memory immediately

    try:
        # ── Open video ────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video")

        fps   = max(1.0, float(cap.get(cv2.CAP_PROP_FPS)) or 30.0)
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── Rotation: prefer ffprobe tag, fall back to dimension heuristic ────
        rot_deg      = probe_rotation(tmp)
        cap_portrait = cap_h > cap_w
        vid_portrait = raw_h > raw_w
        rotation     = None

        if rot_deg in (90, 270):
            rotation = cv2.ROTATE_90_COUNTERCLOCKWISE if rot_deg == 90 else cv2.ROTATE_90_CLOCKWISE
            vid_w, vid_h = raw_h, raw_w
        elif cap_portrait and not vid_portrait:
            rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
            vid_w, vid_h = raw_h, raw_w
        elif not cap_portrait and vid_portrait:
            rotation = cv2.ROTATE_90_CLOCKWISE
            vid_w, vid_h = raw_h, raw_w
        else:
            vid_w, vid_h = raw_w, raw_h

        # ── Scaled processing dimensions ──────────────────────────────────────
        proc_w = max(8, int(vid_w * SCALE))
        proc_h = max(8, int(vid_h * SCALE))

        def get_proc_frame(frm):
            if rotation is not None:
                frm = cv2.rotate(frm, rotation)
            return cv2.resize(frm, (proc_w, proc_h))

        # ── Coordinate scales: capture-space → proc-pixels ───────────────────
        sx = proc_w / cap_w
        sy = proc_h / cap_h

        # Initial plate centre & half-box in proc-pixels
        cx0 = orig_cx * sx
        cy0 = orig_cy * sy
        hw  = box_hw  * sx
        hh  = box_hh  * sy
        # Estimated plate radius from the tap bounding box
        init_r = max(4.0, (hw + hh) / 2.0)

        # ── Seek to tap frame & detect initial circle ─────────────────────────
        start = max(0, int(tap_time * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, frame0 = cap.read()
        if not ret:
            raise HTTPException(status_code=400, detail="Cannot read tap frame")

        proc0 = get_proc_frame(frame0)
        gray0 = cv2.cvtColor(proc0, cv2.COLOR_BGR2GRAY)

        # Tight ROI around tap point for initial detection
        roi_r  = int(init_r * 2.5)
        rx1, ry1 = max(0, int(cx0 - roi_r)), max(0, int(cy0 - roi_r))
        rx2, ry2 = min(proc_w, int(cx0 + roi_r)), min(proc_h, int(cy0 + roi_r))
        roi_gray = gray0[ry1:ry2, rx1:rx2]

        plate_r = init_r  # will be refined if Hough finds a circle
        circles = cv2.HoughCircles(
            roi_gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=50,
            param1=80, param2=25,
            minRadius=max(4, int(init_r * 0.5)),
            maxRadius=int(init_r * 1.8)
        )
        if circles is not None:
            c = circles[0][0]
            plate_r = float(c[2])
            cx0 = rx1 + float(c[0])
            cy0 = ry1 + float(c[1])

        # Adaptive radius bounds (±35 % of detected radius)
        r_min = max(4, int(plate_r * 0.65))
        r_max = int(plate_r * 1.35)

        # ── Initialise 2-D Kalman filter ──────────────────────────────────────
        kf = make_kalman()
        kf.statePre  = np.array([[cx0],[cy0],[0.],[0.]], np.float32)
        kf.statePost = kf.statePre.copy()

        last_cx, last_cy = cx0, cy0
        results = [{
            "t": round(start / fps, 4),
            "x": round(last_cx / sx + dot_off_x, 5),
            "y": round(last_cy / sy + dot_off_y, 5),
        }]

        # ── Frame-skip: target ≤ 500 processed frames ─────────────────────────
        max_frames = min(total - start - 1, int(fps * MAX_SECONDS))
        skip       = max(1, total // 500)   # e.g. 1500-frame video → skip=3
        jump_sq    = (plate_r * 4) ** 2     # max allowed squared displacement

        fn = start + 1
        while fn < start + 1 + max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                break

            proc  = get_proc_frame(frame)
            gray  = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

            # Kalman predict → windowed search centre
            pred  = kf.predict()
            px, py = float(pred[0]), float(pred[1])

            # Windowed search: 5× plate radius around predicted position
            win   = int(plate_r * 5)
            wx1   = max(0, int(px - win))
            wy1   = max(0, int(py - win))
            wx2   = min(proc_w, int(px + win))
            wy2   = min(proc_h, int(py + win))
            win_gray = gray[wy1:wy2, wx1:wx2]

            detected = False
            if win_gray.size > 0:
                circles = cv2.HoughCircles(
                    win_gray, cv2.HOUGH_GRADIENT, dp=1.2,
                    minDist=50,
                    param1=80, param2=25,
                    minRadius=r_min, maxRadius=r_max
                )
                if circles is not None:
                    # Pick circle closest to prediction
                    best = min(circles[0], key=lambda c: (c[0]+wx1-px)**2 + (c[1]+wy1-py)**2)
                    bx_abs = best[0] + wx1
                    by_abs = best[1] + wy1
                    # Jump validation: reject implausible leaps
                    if (bx_abs - last_cx)**2 + (by_abs - last_cy)**2 <= jump_sq:
                        meas = np.array([[bx_abs],[by_abs]], np.float32)
                        kf.correct(meas)
                        last_cx, last_cy = bx_abs, by_abs
                        detected = True

            if not detected:
                # No valid detection — carry Kalman prediction forward
                last_cx, last_cy = px, py

            results.append({
                "t": round(fn / fps, 4),
                "x": round(last_cx / sx + dot_off_x, 5),
                "y": round(last_cy / sy + dot_off_y, 5),
            })
            fn += skip

        cap.release()
        return {"frames": results, "cap_w": cap_w, "cap_h": cap_h}

    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
