import cv2
import numpy as np
import os
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ForceTrack Bar Path API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Process at 25% resolution - 16x less memory than full 1080p, CSRT still accurate
SCALE = 0.25


@app.get("/health")
def health():
    return {"status": "ok", "version": "5.0"}


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

        fps   = max(1.0, float(cap.get(cv2.CAP_PROP_FPS)) or 30.0)
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # ── Rotation fix for iPhone portrait videos ──────────────────────────
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

        # Scaled processing dimensions
        proc_w = max(8, int(vid_w * SCALE))
        proc_h = max(8, int(vid_h * SCALE))

        def get_proc_frame(frm):
            if rotation is not None:
                frm = cv2.rotate(frm, rotation)
            return cv2.resize(frm, (proc_w, proc_h))

        # Scale: capture-coords → scaled-video-pixels
        sx = proc_w / cap_w
        sy = proc_h / cap_h

        # Bounding box in scaled video pixels
        bx = max(0, int((orig_cx - box_hw) * sx))
        by = max(0, int((orig_cy - box_hh) * sy))
        bw = max(4, min(int(box_hw * 2 * sx), proc_w - bx))
        bh = max(4, min(int(box_hh * 2 * sy), proc_h - by))

        # Seek to tap frame
        start = max(0, int(tap_time * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, frame0 = cap.read()
        if not ret:
            raise HTTPException(status_code=400, detail="Cannot read tap frame")

        frame0_proc = get_proc_frame(frame0)

        # CSRT tracker on small frames - fast and memory-efficient
        try:
            tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            tracker = cv2.TrackerKCF_create()

        pad = max(2, int(min(bw, bh) * 0.1))
        tracker.init(frame0_proc, (
            max(0, bx - pad),
            max(0, by - pad),
            min(proc_w - max(0, bx - pad), bw + 2 * pad),
            min(proc_h - max(0, by - pad), bh + 2 * pad)
        ))

        last_cx = bx + bw / 2
        last_cy = by + bh / 2
        results = [{"t": start / fps, "x": last_cx / sx + dot_off_x, "y": last_cy / sy + dot_off_y}]

        # 45 second cap
        max_frames = min(total - start - 1, int(fps * 45))

        fn = start + 1
        while fn < start + 1 + max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            ok, bbox = tracker.update(get_proc_frame(frame))
            if ok:
                last_cx = bbox[0] + bbox[2] / 2
                last_cy = bbox[1] + bbox[3] / 2
            results.append({"t": fn / fps, "x": last_cx / sx + dot_off_x, "y": last_cy / sy + dot_off_y})
            fn += 1

        cap.release()
        return {"frames": results, "cap_w": cap_w, "cap_h": cap_h}

    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
