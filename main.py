import os, cv2, numpy as np, tempfile, subprocess, json, asyncio
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="ForceTrack Bar Path API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def get_rotation(path):
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0","-show_entries",
             "stream_tags=rotate","-of","default=noprint_wrappers=1:nokey=1",path],
            capture_output=True, text=True, timeout=10)
        v = r.stdout.strip()
        return int(v) if v else 0
    except: return 0

def rotate_frame(frame, rot):
    if rot == 90:  return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rot == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if rot == 270: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

def hough_near(gray, cx, cy, plate_r, wp, hp, scale=2.5):
    sr = int(plate_r * scale)
    hx = max(0, int(cx)-sr); hy = max(0, int(cy)-sr)
    hw = min(wp-hx, sr*2);   hh = min(hp-hy, sr*2)
    if hw < 1 or hh < 1: return None
    roi = gray[hy:hy+hh, hx:hx+hw]
    b = cv2.GaussianBlur(roi, (9,9), 2)
    for p2 in [28, 22, 16, 10]:
        c = cv2.HoughCircles(b, cv2.HOUGH_GRADIENT, 1.2, hh,
            param1=80, param2=p2,
            minRadius=int(plate_r*0.65), maxRadius=int(plate_r*1.35))
        if c is not None:
            best = min(c[0], key=lambda v: (v[0]+hx-cx)**2+(v[1]+hy-cy)**2)
            if (best[0]+hx-cx)**2+(best[1]+hy-cy)**2 < (plate_r*2)**2:
                return float(best[0]+hx), float(best[1]+hy), float(best[2])
    return None

def hough_full(gray, min_r, max_r, hint_x, hint_y, pad):
    sx = max(0, int(hint_x-pad)); sy = max(0, int(hint_y-pad))
    h, w = gray.shape
    ex = min(w, int(hint_x+pad)); ey = min(h, int(hint_y+pad))
    roi = gray[sy:ey, sx:ex]
    b = cv2.GaussianBlur(roi, (9,9), 2)
    for p2 in [28, 22, 16, 10, 6]:
        c = cv2.HoughCircles(b, cv2.HOUGH_GRADIENT, 1.2, (ey-sy)//2,
            param1=80, param2=p2, minRadius=min_r, maxRadius=max_r)
        if c is not None:
            best = min(c[0], key=lambda v: (v[0]+sx-hint_x)**2+(v[1]+sy-hint_y)**2)
            return float(best[0]+sx), float(best[1]+sy), float(best[2])
    return None

def seed_features(gray, cx, cy, r):
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(r*0.9), 255, -1)
    pts = cv2.goodFeaturesToTrack(gray, 100, 0.01, 4, mask=mask, blockSize=7)
    if pts is not None and len(pts) >= 4:
        return pts
    grid = []
    step = max(2, r * 0.25)
    dy = -r*0.7
    while dy <= r*0.7:
        dx = -r*0.7
        while dx <= r*0.7:
            if dx*dx + dy*dy <= r*r*0.49:
                grid.append([[cx+dx, cy+dy]])
            dx += step
        dy += step
    return np.array(grid, dtype=np.float32) if grid else np.array([[[cx, cy]]], dtype=np.float32)

LK = dict(winSize=(25,25), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))

@app.get("/health")
def health(): return {"status":"ok","version":"1.1.0"}

@app.post("/analyze")
async def analyze(video: UploadFile=File(...), params: str=Form("{}"), api_key: str=Form("")):
    tmp = tempfile.mktemp(suffix=".mp4")
    try:
        data = await video.read()
        if len(data) > 600*1024*1024:
            raise HTTPException(400, "Video too large")
        with open(tmp,"wb") as f: f.write(data)
        del data
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, f"Save failed: {e}")

    async def stream():
        try:
            rot = get_rotation(tmp)
            cap = cv2.VideoCapture(tmp)
            if not cap.isOpened():
                yield json.dumps({"error":"Cannot open video"})+"\n"; return

            fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, f0 = cap.read()
            if not ret:
                yield json.dumps({"error":"Cannot read first frame"})+"\n"; return

            f0 = rotate_frame(f0, rot)
            raw_h, raw_w = f0.shape[:2]
            wp, hp = int(raw_w*0.5), int(raw_h*0.5)
            f0s  = cv2.resize(f0, (wp, hp))
            g0   = cv2.cvtColor(f0s, cv2.COLOR_BGR2GRAY)
            del f0, f0s

            try:   p = json.loads(params)
            except: p = {}
            hx  = float(p.get("start_x", 0.5)) * wp
            hy_ = float(p.get("start_y", 0.5)) * hp

            min_r = max(8,  int(hp*0.05))
            max_r = min(wp//2, int(hp*0.46))
            pad   = int(min(wp,hp)*0.3)

            det = hough_full(g0, min_r, max_r, hx, hy_, pad)
            if det is None:
                det = (hx, hy_, max(min_r*2, int(hp*0.08)))

            cx, cy, plate_r = det
            plate_r = max(min_r, plate_r)

            prev_gray = g0.copy()
            p0 = seed_features(g0, cx, cy, plate_r)
            del g0

            results   = []
            frames_since_hough = 0
            HOUGH_EVERY = 12
            skip = max(1, total//600)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fn = 0

            while True:
                ret, frame = cap.read()
                if not ret: break
                if fn % skip != 0:
                    fn += 1; continue

                frame = rotate_frame(frame, rot)
                small = cv2.resize(frame, (wp, hp))
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                del frame
                curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                del small

                p1, sf, _  = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **LK)
                p0r, sb, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1,  None, **LK)

                fb_err = np.abs(p0 - p0r).reshape(-1, 2).max(axis=1)
                good   = (sf.ravel()==1) & (sb.ravel()==1) & (fb_err < 1.5)

                if good.sum() >= 4:
                    good_new = p1[good]
                    good_old = p0[good]
                    dxs = (good_new[:,0,0] - good_old[:,0,0])
                    dys = (good_new[:,0,1] - good_old[:,0,1])
                    cx  = float(np.clip(cx + float(np.median(dxs)), plate_r, wp-plate_r))
                    cy  = float(np.clip(cy + float(np.median(dys)), plate_r, hp-plate_r))
                    p0  = good_new.reshape(-1,1,2)

                frames_since_hough += 1
                if frames_since_hough >= HOUGH_EVERY:
                    frames_since_hough = 0
                    rd = hough_near(curr_gray, cx, cy, plate_r, wp, hp, 2.5)
                    if rd:
                        ncx, ncy, _ = rd
                        if (ncx-cx)**2+(ncy-cy)**2 < (plate_r*1.8)**2:
                            cx, cy = ncx, ncy
                            new_pts = seed_features(curr_gray, cx, cy, plate_r)
                            if len(new_pts) >= 4:
                                p0 = new_pts

                results.append({"t":round(t,4),"x":round(cx/wp,5),"y":round(cy/hp,5)})
                prev_gray = curr_gray
                fn += 1

                if len(results) % 30 == 0:
                    pct = min(99, int(fn/max(total,1)*100))
                    yield json.dumps({"progress":pct})+"\n"
                    await asyncio.sleep(0)

            cap.release()
            yield json.dumps({"done":True,"frames":results,"cap_w":raw_w,"cap_h":raw_h,"fps":fps,"rotation":rot})+"\n"
        except Exception as e:
            yield json.dumps({"error":str(e)})+"\n"
        finally:
            try: os.unlink(tmp)
            except: pass

    return StreamingResponse(stream(), media_type="application/x-ndjson")
