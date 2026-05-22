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


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0"}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    params: str = Form(...),
    api_key: str = Form(default="")
):
    p = json.loads(params)
    tap_time    = float(p["tap_time"])
    cap_w       = int(p["cap_w"])
    cap_h       = int(p["cap_h"])
    orig_cx     = float(p["orig_cx"])
    orig_cy     = float(p["orig_cy"])
    box_hw      = float(p["box_hw"])
    box_hh      = float(p["box_hh"])
    dot_off_x   = float(p["dot_offset_x"])
    dot_off_y   = float(p["dot_offset_y"])

    content = await video.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(content)
        tmp = f.name

    try:
        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video")

        fps      = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        raw_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── Rotation fix ────────────────────────────────────────────────────
        # iPhones store portrait video with a rotation flag that OpenCV on
        # Linux ignores.  Detect the mismatch and rotate frames to match.
        cap_portrait = cap_h > cap_w
        vid_portrait = raw_h > raw_w
        rotation = None
        if cap_portrait and not vid_portrait:
            rotation = cv2.ROTATE_90_CLOCKWISE
            vid_w, vid_h = raw_h, raw_w          # dimensions after rotation
        elif not cap_portrait and vid_portrait:
            rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
            vid_w, vid_h = raw_h, raw_w
        else:
            vid_w, vid_h = raw_w, raw_h

        def get_gray(frame):
            if rotation is not None:
                frame = cv2.rotate(frame, rotation)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Scale: capture-coords  ↔  video-pixels ──────────────────────────
        sx = vid_w / cap_w
        sy = vid_h / cap_h

        # Bounding box in video-pixel coords
        bx = max(0, int((orig_cx - box_hw) * sx))
        by = max(0, int((orig_cy - box_hh) * sy))
        bw = max(4, min(int(box_hw * 2 * sx), vid_w - bx))
        bh = max(4, min(int(box_hh * 2 * sy), vid_h - by))

        # Search radius  (matches JS: SR_x=8, SR_y=35 in capture space)
        sr_x = max(4, int(8  * sx))
        sr_y = max(4, int(35 * sy))

        # ── Read first frame & build template ───────────────────────────────
        start = max(0, int(tap_time * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, f0 = cap.read()
        if not ret:
            raise HTTPException(status_code=400, detail="Cannot read first frame")

        g0   = get_gray(f0)
        tmpl = g0[by:by + bh, bx:bx + bw].copy()

        # ── Template-matching helper ─────────────────────────────────────────
        def find_plate(gray, pcx, pcy):
            pvx = pcx * sx
            pvy = pcy * sy
            rx  = max(0,     int(pvx - box_hw * sx - sr_x))
            ry  = max(0,     int(pvy - box_hh * sy - sr_y))
            rx2 = min(vid_w, int(pvx + box_hw * sx + sr_x + bw))
            ry2 = min(vid_h, int(pvy + box_hh * sy + sr_y + bh))
            region = gray[ry:ry2, rx:rx2]
            if region.shape[0] < tmpl.shape[0] or region.shape[1] < tmpl.shape[1]:
                return pcx, pcy
            res = cv2.matchTemplate(region, tmpl, cv2.TM_SQDIFF)
            _, _, (mx, my), _ = cv2.minMaxLoc(res)
            return (rx + mx + bw / 2) / sx, (ry + my + bh / 2) / sy

        # ── Track every frame ────────────────────────────────────────────────
        cx, cy = orig_cx, orig_cy
        vx, vy = 0.0, 0.0
        results = [{"t": start / fps, "x": cx + dot_off_x, "y": cy + dot_off_y}]

        fn = start + 1
        while fn < total:
            ret, frame = cap.read()
            if not ret:
                break
            gray = get_gray(frame)
            pcx, pcy = cx + vx, cy + vy
            fcx, fcy = find_plate(gray, pcx, pcy)

            nvx = fcx - pcx + vx
            nvy = fcy - pcy + vy
            mag = (nvx ** 2 + nvy ** 2) ** 0.5
            if mag > 25:
                nvx, nvy = nvx * 25 / mag, nvy * 25 / mag
            vx = vx * 0.2 + nvx * 0.8
            vy = vy * 0.2 + nvy * 0.8
            cx, cy = fcx, fcy
            results.append({"t": fn / fps, "x": cx + dot_off_x, "y": cy + dot_off_y})
            fn += 1

        cap.release()
        return {"frames": results, "cap_w": cap_w, "cap_h": cap_h}

    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
