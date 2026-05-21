import cv2
import numpy as np
import os
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ForceTrack Bar Path API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("BARPATH_API_KEY", "change-me")


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0"}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    params: str = Form(...),
    api_key: str = Form(...)
):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    p = json.loads(params)
    tap_time = float(p["tap_time"])
    cap_w = int(p["cap_w"])
    cap_h = int(p["cap_h"])
    orig_cx = float(p["orig_cx"])
    orig_cy = float(p["orig_cy"])
    box_hw = float(p["box_hw"])
    box_hh = float(p["box_hh"])
    dot_offset_x = float(p["dot_offset_x"])
    dot_offset_y = float(p["dot_offset_y"])

    # Save video to temp file
    content = await video.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Scale factors: capture canvas coords → video pixel coords
        sx = vid_w / cap_w
        sy = vid_h / cap_h

        # Bounding box in video pixel coordinates
        bx = max(0, int((orig_cx - box_hw) * sx))
        by = max(0, int((orig_cy - box_hh) * sy))
        bw = max(4, int(box_hw * 2 * sx))
        bh = max(4, int(box_hh * 2 * sy))
        bw = min(bw, vid_w - bx)
        bh = min(bh, vid_h - by)

        # Search radius in video pixels (mirrors JS: SR_x=8, SR_y=35 in capture coords)
        sr_x = max(4, int(8 * sx))
        sr_y = max(4, int(35 * sy))

        # Seek to tap_time
        start_frame = max(0, int(tap_time * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, first_frame = cap.read()
        if not ret:
            raise HTTPException(status_code=400, detail="Cannot read frame at tap_time")

        # Extract grayscale template from bounding box
        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        template = first_gray[by:by + bh, bx:bx + bw].copy()

        def find_plate(gray_frame, pred_cx_cap, pred_cy_cap):
            """Find plate using template matching. Args/returns in capture coord space."""
            pvx = pred_cx_cap * sx
            pvy = pred_cy_cap * sy

            rx = max(0, int(pvx - box_hw * sx - sr_x))
            ry = max(0, int(pvy - box_hh * sy - sr_y))
            rx2 = min(vid_w, int(pvx + box_hw * sx + sr_x + bw))
            ry2 = min(vid_h, int(pvy + box_hh * sy + sr_y + bh))

            region = gray_frame[ry:ry2, rx:rx2]
            if region.shape[0] < template.shape[0] or region.shape[1] < template.shape[1]:
                return pred_cx_cap, pred_cy_cap  # fallback to prediction

            result = cv2.matchTemplate(region, template, cv2.TM_SQDIFF)
            _, _, (mx, my), _ = cv2.minMaxLoc(result)

            found_vx = rx + mx + bw / 2
            found_vy = ry + my + bh / 2
            return found_vx / sx, found_vy / sy

        # Tracking state
        cx = orig_cx
        cy = orig_cy
        vx = 0.0
        vy = 0.0

        results = [{"t": start_frame / fps, "x": cx + dot_offset_x, "y": cy + dot_offset_y}]

        frame_num = start_frame + 1
        while frame_num < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pred_cx = cx + vx
            pred_cy = cy + vy

            found_cx, found_cy = find_plate(gray, pred_cx, pred_cy)

            # Velocity EMA — same formula as JS: nv = found - pred + vel, cap at 25
            nv_x = found_cx - pred_cx + vx
            nv_y = found_cy - pred_cy + vy
            mag = (nv_x ** 2 + nv_y ** 2) ** 0.5
            if mag > 25:
                nv_x = nv_x * 25 / mag
                nv_y = nv_y * 25 / mag

            vx = vx * 0.2 + nv_x * 0.8
            vy = vy * 0.2 + nv_y * 0.8
            cx = found_cx
            cy = found_cy

            results.append({
                "t": frame_num / fps,
                "x": cx + dot_offset_x,
                "y": cy + dot_offset_y
            })
            frame_num += 1

        cap.release()
        return {"frames": results, "cap_w": cap_w, "cap_h": cap_h}

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
