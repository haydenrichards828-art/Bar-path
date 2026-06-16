import os, cv2, numpy as np, tempfile, subprocess
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ForceTrack Bar Path API", version="0.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
VALID_KEY = os.environ.get("BARPATH_API_KEY", "")

def get_rotation(path):
    try:
        r = subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
            "-show_entries","stream_tags=rotate","-of",
            "default=noprint_wrappers=1:nokey=1",path],
            capture_output=True, text=True, timeout=10)
        v = r.stdout.strip()
        return int(v) if v else 0
    except: return 0

def rotate_frame(frame, rot):
    if rot == 90: return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rot == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if rot == 270: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

class Kalman2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.init = False
    def update(self, x, y):
        m = np.array([[x],[y]], np.float32)
        if not self.init:
            self.kf.statePre = np.array([[x],[y],[0],[0]], np.float32)
            self.init = True
        self.kf.correct(m)
        p = self.kf.predict()
        return float(p[0]), float(p[1])
    def predict(self):
        p = self.kf.predict()
        return float(p[0]), float(p[1])

def hough_detect(gray, min_r, max_r, search_box=None):
    """Detect largest circle via Hough. search_box=(x,y,w,h) restricts ROI."""
    if search_box is not None:
        sx, sy, sw, sh = [int(v) for v in search_box]
        h, w = gray.shape
        sx, sy = max(0, sx), max(0, sy)
        ex, ey = min(w, sx+sw), min(h, sy+sh)
        if ex <= sx or ey <= sy: return None
        roi = gray[sy:ey, sx:ex]
        ox, oy = sx, sy
    else:
        roi, ox, oy = gray, 0, 0
    b = cv2.GaussianBlur(roi, (9,9), 2)
    for p2 in [28, 22, 16, 12]:
        c = cv2.HoughCircles(b, cv2.HOUGH_GRADIENT, 1.2, 30,
                             param1=80, param2=p2, minRadius=min_r, maxRadius=max_r)
        if c is not None:
            best = max(c[0], key=lambda x: x[2])
            return float(best[0]+ox), float(best[1]+oy), float(best[2])
    return None

def circle_in_region(gray, bbox, min_r, max_r):
    """Return True if a Hough circle exists inside the bounding box."""
    x, y, w, h = [int(v) for v in bbox]
    gh, gw = gray.shape
    x, y = max(0, x), max(0, y)
    ex, ey = min(gw, x+w), min(gh, y+h)
    if ex <= x or ey <= y: return False
    roi = gray[y:ey, x:ex]
    if roi.size == 0: return False
    b = cv2.GaussianBlur(roi, (7,7), 2)
    c = cv2.HoughCircles(b, cv2.HOUGH_GRADIENT, 1.2, 20,
                         param1=70, param2=10, minRadius=min_r, maxRadius=max_r)
    return c is not None

def clamp_bbox(bbox, wp, hp):
    x, y, w, h = bbox
    x, y = max(0, int(x)), max(0, int(y))
    w = min(wp-x, int(w))
    h = min(hp-y, int(h))
    return (x, y, max(1,w), max(1,h))

def bbox_from_center(cx, cy, r, scale=1.6):
    half = r * scale
    return (cx-half, cy-half, half*2, half*2)

def make_tracker():
    try: return cv2.TrackerCSRT_create()
    except AttributeError: return cv2.TrackerKCF_create()

@app.get("/health")
def health(): return {"status":"ok","version":"0.4.0"}

@app.post("/analyze")
async def analyze(video: UploadFile=File(...), params: str=Form("{}"), api_key: str=Form("")):
    if VALID_KEY and api_key != VALID_KEY:
        raise HTTPException(401, "Invalid API key")
    tmp = tempfile.mktemp(suffix=".mp4")
    try:
        data = await video.read()
        if len(data) > 600*1024*1024:
            raise HTTPException(400, "Video too large")
        with open(tmp,"wb") as f: f.write(data)
        del data

        rot = get_rotation(tmp)
        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened(): raise HTTPException(400, "Cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Frame 0: detect initial plate
        ret, f0 = cap.read()
        if not ret: raise HTTPException(400, "Cannot read first frame")
        f0 = rotate_frame(f0, rot)
        raw_h, raw_w = f0.shape[:2]

        scale = 0.5
        wp, hp = int(raw_w*scale), int(raw_h*scale)
        f0s = cv2.resize(f0, (wp, hp))
        g0  = cv2.cvtColor(f0s, cv2.COLOR_BGR2GRAY)
        del f0

        min_r = max(8,  int(hp * 0.06))
        max_r = min(wp//2, int(hp * 0.45))

        det = hough_detect(g0, min_r, max_r)
        if det is None: det = (wp/2, hp/2, min_r*2)
        cx0, cy0, r0 = det
        plate_r = r0

        # Verification radii (looser bounds for in-loop checks)
        vmin_r = int(plate_r * 0.5)
        vmax_r = int(plate_r * 1.5)

        bbox0 = clamp_bbox(bbox_from_center(cx0, cy0, r0, 1.6), wp, hp)
        tracker = make_tracker()
        tracker.init(f0s, bbox0)
        del f0s

        kal = Kalman2D()
        results = []
        last_cx, last_cy = cx0, cy0
        # Plausibility gate: allow up to 10x plate radius jump between ANY two frames
        # (with every-frame processing the per-frame velocity is ~1-3 plate-radii max)
        max_jump_sq  = (plate_r * 10) ** 2
        reinit_half  = int(plate_r * 8)   # large local search window

        # Process EVERY frame — accuracy over speed
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = rotate_frame(frame, rot)
            small = cv2.resize(frame, (wp, hp))
            t     = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            del frame

            ok, bbox = tracker.update(small)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            def reinit_window():
                """Hough in enlarged window around last known pos. Returns (kx,ky) or None."""
                nonlocal tracker, last_cx, last_cy
                sb = (last_cx - reinit_half, last_cy - reinit_half,
                      reinit_half * 2, reinit_half * 2)
                rd = hough_detect(gray, vmin_r, vmax_r, sb)
                if rd:
                    rx, ry, _ = rd
                    nb = clamp_bbox(bbox_from_center(rx, ry, plate_r, 1.6), wp, hp)
                    tracker = make_tracker()
                    tracker.init(small, nb)
                    kx2, ky2 = kal.update(rx, ry)
                    last_cx, last_cy = rx, ry
                    return kx2, ky2
                return None

            def reinit_global():
                """Full-frame Hough search — last resort."""
                nonlocal tracker, last_cx, last_cy
                rd = hough_detect(gray, vmin_r, vmax_r)
                if rd:
                    rx, ry, _ = rd
                    nb = clamp_bbox(bbox_from_center(rx, ry, plate_r, 1.6), wp, hp)
                    tracker = make_tracker()
                    tracker.init(small, nb)
                    kx2, ky2 = kal.update(rx, ry)
                    last_cx, last_cy = rx, ry
                    return kx2, ky2
                return None

            if ok:
                cx = bbox[0] + bbox[2] / 2
                cy = bbox[1] + bbox[3] / 2
                jump_sq = (cx - last_cx) ** 2 + (cy - last_cy) ** 2

                if jump_sq > max_jump_sq:
                    # CSRT jumped to a distant object — search near last known pos
                    res = reinit_window() or reinit_global()
                    kx, ky = res if res else kal.predict()
                else:
                    # CSRT stayed nearby — verify a barbell circle is actually inside its bbox
                    if circle_in_region(gray, bbox, vmin_r, vmax_r):
                        kx, ky = kal.update(cx, cy)
                        last_cx, last_cy = cx, cy
                    else:
                        # Tracker drifted to a non-circular object (rack, etc.)
                        res = reinit_window() or reinit_global()
                        kx, ky = res if res else kal.update(cx, cy)
            else:
                # CSRT completely lost — try to recover
                res = reinit_window() or reinit_global()
                kx, ky = res if res else kal.predict()

            del small

            results.append({"t": round(t, 4),
                            "x": round(kx / wp, 5),
                            "y": round(ky / hp, 5)})

        cap.release()
        return {"frames": results, "cap_w": raw_w, "cap_h": raw_h,
                "fps": fps, "rotation": rot}
    finally:
        try: os.unlink(tmp)
        except: pass
