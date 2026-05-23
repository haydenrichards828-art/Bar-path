import cv2
import numpy as np
import os
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ForceTrack Bar Path API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0"}


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    params: str = Form(...),
    api_key: str = Form(default="")
):
    p = json.loads(params)
    tap_time  = float(p["tap_time"])
    cap_w     = int(p["cap_w"])
    cap_h     = int(p["cap_h"])
    orig_cx   = float(p["orig_cx"])
    orig_cy   = float(p["orig_cy"])
    box_hw    = float(p["box_hw"])
    box_hh    = float(p["box_hh"])
    dot_off_x = float(p["dot_offset_x"])
    dot_off_y = float(p["dot_offset_y"])

    content = await video.read()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(content)
        tmp = f.name

    try:
        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video")

        fps   = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── Rotation: iPhones store portrait with rotation metadata OpenCV ignores
        cap_portrait = cap_h > cap_w
        vid_portrait = raw_h > raw_w
        rotation = None
        if cap_portrait and not vid_portrait:
            rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
            vid_w, vid_h = raw_h, raw_w
        elif not cap_portrait and vid_portrait:
            rotation = cv2.ROTATE_90_CLOCKWISE
            vid_w, vid_h = raw_h, raw_w
        else:
            vid_w, vid_h = raw_w, raw_h

        def fix_frame(frm):
            if rotation is not None:
                frm = cv2.rotate(frm, rotation)
            return frm

        # ── Scale factors: capture-coords ↔ video-pixels ──
        sx = vid_w / cap_w
        sy = vid_h / cap_h

        # Bounding box in video pixels (plate region user drew)
        bx = max(0, int((orig_cx - box_hw) * sx))
        by = max(0, int((orig_cy - box_hh) * sy))
        bw = max(8, min(int(box_hw * 2 * sx), vid_w - bx))
        bh = max(8, min(int(box_hh * 2 * sy), vid_h - by))

        # ── Seek to tap frame and read it ──
        start = max(0, int(tap_time * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, frame0 = cap.read()
        if not ret:
            raise HTTPException(status_code=400, detail="Cannot read tap frame")
        frame0 = fix_frame(frame0)

        # ── Use CSRT tracker (handles lighting/reflection changes robustly) ──
        try:
            tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            tracker = cv2.TrackerKCF_create()

        # Slightly expand bbox for better CSRT initialisation
        pad  = max(4, int(min(bw, bh) * 0.15))
        tbx  = max(0, bx - pad)
        tby  = max(0, by - pad)
        tbw  = min(vid_w - tbx, bw + 2 * pad)
        tbh  = min(vid_h - tby, bh + 2 * pad)
        tracker.init(frame0, (tbx, tby, tbw, tbh))

        # Plate centre from initial bbox → capture coords
        px0 = (bx + bw / 2) / sx
        py0 = (by + bh / 2) / sy
        results = [{"t": start / fps,
                    "x": px0 + dot_off_x,
                    "y": py0 + dot_off_y}]

        last_cx = bx + bw / 2
        last_cy = by + bh / 2

        fn = start + 1
        while fn < total:
            ret, frame = cap.read()
            if not ret:
                break
            frame = fix_frame(frame)

            ok, bbox = tracker.update(frame)
            if ok:
                last_cx = bbox[0] + bbox[2] / 2
                last_cy = bbox[1] + bbox[3] / 2

            # Always emit a point (last known if tracking failed)
            results.append({
                "t": fn / fps,
                "x": last_cx / sx + dot_off_x,
                "y": last_cy / sy + dot_off_y
            })
            fn += 1

        cap.release()
        return {"frames": results, "cap_w": cap_w, "cap_h": cap_h}

    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
