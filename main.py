import os, cv2, numpy as np, tempfile, subprocess, json, asyncio
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="ForceTrack Bar Path API", version="1.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PLATE_DIAMETER_M = 0.450  # Standard Olympic plate — 10kg+ all 450mm
SCALE            = 1.0    # Full resolution — maximises Hough + LK accuracy
CLAHE            = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

def get_rotation(path):
    try:
        r = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
            "stream_tags=rotate","-of","default=noprint_wrappers=1:nokey=1",path],
            capture_output=True, text=True, timeout=10)
        v = r.stdout.strip(); return int(v) if v else 0
    except: return 0

def rotate_frame(frame, rot):
    if rot == 90:  return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rot == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if rot == 270: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

def enhance(gray):
    return CLAHE.apply(gray)

def hough_detect(gray, min_r, max_r, cx_hint, cy_hint, search_pad):
    h, w = gray.shape
    sx = max(0, int(cx_hint-search_pad)); sy = max(0, int(cy_hint-search_pad))
    ex = min(w, int(cx_hint+search_pad)); ey = min(h, int(cy_hint+search_pad))
    if ex-sx < 10 or ey-sy < 10: return None
    roi = enhance(gray[sy:ey, sx:ex])
    b = cv2.GaussianBlur(roi, (9,9), 2)
    for p2 in [30, 24, 18, 12, 8, 5]:
        c = cv2.HoughCircles(b, cv2.HOUGH_GRADIENT, 1.2, (ey-sy)//2,
            param1=80, param2=p2, minRadius=min_r, maxRadius=max_r)
        if c is not None:
            best = min(c[0], key=lambda v: (v[0]+sx-cx_hint)**2+(v[1]+sy-cy_hint)**2)
            return float(best[0]+sx), float(best[1]+sy), float(best[2])
    return None

def seed_pts(gray, cx, cy, r):
    eg = enhance(gray)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(max(4, r*0.88)), 255, -1)
    pts = cv2.goodFeaturesToTrack(eg, 150, 0.006, 3, mask=mask, blockSize=7)
    if pts is not None and len(pts) >= 6: return pts
    grid = []; step = max(2.0, r*0.22); dy = -r*0.75
    while dy <= r*0.75:
        dx = -r*0.75
        while dx <= r*0.75:
            if dx*dx+dy*dy <= r*r*0.56: grid.append([[cx+dx, cy+dy]])
            dx += step
        dy += step
    return np.array(grid, dtype=np.float32) if grid else np.array([[[cx,cy]]], dtype=np.float32)

class Kalman2D:
    """Position + velocity Kalman filter — smooths LK noise, rejects outlier frames."""
    def __init__(self, x0, y0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix  = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix   = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov    = np.eye(4, dtype=np.float32) * 0.05
        self.kf.measurementNoiseCov= np.eye(2, dtype=np.float32) * 2.0
        self.kf.statePre = np.array([[x0],[y0],[0.0],[0.0]], np.float32)
        self.kf.statePost= np.array([[x0],[y0],[0.0],[0.0]], np.float32)
    def update(self, x, y):
        pred = self.kf.predict()
        m = np.array([[x],[y]], np.float32)
        corr = self.kf.correct(m)
        return float(corr[0]), float(corr[1])
    def predict_only(self):
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

def smooth_coords(xs, ys, window=7):
    if len(xs) < window: return xs, ys
    w = window if window % 2 == 1 else window+1
    k = np.ones(w)/w
    xs_s = np.convolve(xs, k, mode='same'); ys_s = np.convolve(ys, k, mode='same')
    h = w//2
    xs_s[:h]=xs[:h]; xs_s[-h:]=xs[-h:]
    ys_s[:h]=ys[:h]; ys_s[-h:]=ys[-h:]
    return xs_s.tolist(), ys_s.tolist()

def detect_reps(frames, min_frames=8):
    if len(frames) < min_frames*2: return []
    ys=[f['y'] for f in frames]; ts=[f['t'] for f in frames]
    vel=[(ys[i]-ys[i-1])/max(ts[i]-ts[i-1],0.001) for i in range(1,len(ys))]
    vel=[vel[0]]+vel
    svel=np.convolve(vel,np.ones(5)/5,mode='same').tolist()
    reps=[]; direction=None; rep_start=0
    for i,v in enumerate(svel):
        if abs(v)<0.002: continue
        nd='down' if v>0 else 'up'
        if direction is None: direction=nd; rep_start=i; continue
        if nd!=direction:
            if i-rep_start>=min_frames: reps.append({'start':rep_start,'end':i,'phase':direction})
            direction=nd; rep_start=i
    merged=[]; i=0
    while i<len(reps)-1:
        if reps[i]['phase']=='down' and reps[i+1]['phase']=='up':
            merged.append({'start':reps[i]['start'],'end':reps[i+1]['end']}); i+=2
        else: i+=1
    return merged

LK = dict(winSize=(31,31), maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))

@app.get("/health")
def health(): return {"status":"ok","version":"1.4.0"}

@app.post("/analyze")
async def analyze(video: UploadFile=File(...), params: str=Form("{}"), api_key: str=Form("")):
    tmp = tempfile.mktemp(suffix=".mp4")
    try:
        data = await video.read()
        if len(data) > 600*1024*1024: raise HTTPException(400, "Video too large")
        with open(tmp,"wb") as f: f.write(data)
        del data
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, f"Save failed: {e}")

    async def stream():
        try:
            rot = get_rotation(tmp)
            cap = cv2.VideoCapture(tmp)
            if not cap.isOpened(): yield json.dumps({"error":"Cannot open video"})+"\n"; return
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, f0 = cap.read()
            if not ret: yield json.dumps({"error":"Cannot read first frame"})+"\n"; return
            f0 = rotate_frame(f0, rot)
            raw_h, raw_w = f0.shape[:2]
            wp = int(raw_w * SCALE); hp = int(raw_h * SCALE)
            f0s = cv2.resize(f0, (wp, hp)) if SCALE != 1.0 else f0
            g0  = cv2.cvtColor(f0s, cv2.COLOR_BGR2GRAY)
            del f0, f0s
            try: p = json.loads(params)
            except: p = {}
            tx = float(p.get("start_x", 0.5)) * wp
            ty = float(p.get("start_y", 0.5)) * hp
            min_r = max(10, int(hp*0.05)); max_r = min(wp//2, int(hp*0.47))
            pad   = int(min(wp,hp)*0.35)
            det = hough_detect(g0, min_r, max_r, tx, ty, pad)
            if det is None: det = hough_detect(g0, min_r, max_r, wp//2, hp//2, min(wp,hp)//2)
            if det is None: det = (tx, ty, max(min_r*2, int(hp*0.09)))
            cx, cy, plate_r = det; plate_r = max(min_r, plate_r)
            px_per_m = plate_r / (PLATE_DIAMETER_M / 2.0)
            prev_gray = enhance(g0.copy())
            p0 = seed_pts(g0, cx, cy, plate_r); del g0
            kal = Kalman2D(cx, cy)  # Kalman filter initialised at detected plate center
            results = [{"t":0.0,"x":round(cx/wp,5),"y":round(cy/hp,5)}]
            yield json.dumps({"frame":results[0]})+"\n"
            frames_since_hough = 0; HOUGH_EVERY = 10
            # 60fps support: budget by time not frame count
            # Process up to 900 "logical" frames — at 60fps skip every 2nd frame
            # at 30fps process every frame, at 120fps skip every 4th
            target_fps   = 30.0
            proc_every   = max(1, round(fps / target_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); fn=0; proc=0
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = rotate_frame(frame, rot)
                small = cv2.resize(frame, (wp, hp)) if SCALE != 1.0 else frame
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0; del frame
                curr_raw  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY); del small
                curr_gray = enhance(curr_raw)
                if fn % proc_every == 0:
                    # Bidirectional LK
                    p1, sf, _  = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **LK)
                    p0r, sb, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, p1, None, **LK)
                    fb = np.abs(p0-p0r).reshape(-1,2).max(axis=1)
                    good = (sf.ravel()==1)&(sb.ravel()==1)&(fb<1.5)
                    if good.sum() >= 4:
                        gn=p1[good]; go=p0[good]
                        raw_cx = cx + float(np.median(gn[:,0,0]-go[:,0,0]))
                        raw_cy = cy + float(np.median(gn[:,0,1]-go[:,0,1]))
                        raw_cx = float(np.clip(raw_cx, plate_r, wp-plate_r))
                        raw_cy = float(np.clip(raw_cy, plate_r, hp-plate_r))
                        # Kalman: blend LK measurement with physical prediction
                        cx, cy = kal.update(raw_cx, raw_cy)
                        p0 = gn.reshape(-1,1,2)
                    else:
                        # No good points — use Kalman prediction
                        cx, cy = kal.predict_only()
                    # Early re-anchor when tracking is weak
                    if good.sum() < 8:
                        rd = hough_detect(curr_raw, int(plate_r*0.65), int(plate_r*1.35), cx, cy, plate_r*2.2)
                        if rd:
                            ncx, ncy, _ = rd
                            if (ncx-cx)**2+(ncy-cy)**2 < (plate_r*1.6)**2:
                                cx, cy = kal.update(ncx, ncy)
                                np_=seed_pts(curr_raw, cx, cy, plate_r)
                                if len(np_)>=4: p0=np_; frames_since_hough=0
                    frames_since_hough+=1
                    if frames_since_hough >= HOUGH_EVERY:
                        frames_since_hough=0
                        rd = hough_detect(curr_raw, int(plate_r*0.65), int(plate_r*1.35), cx, cy, plate_r*2.8)
                        if rd:
                            ncx,ncy,_=rd
                            if (ncx-cx)**2+(ncy-cy)**2 < (plate_r*2.0)**2:
                                cx,cy=kal.update(ncx,ncy)
                                np_=seed_pts(curr_raw,cx,cy,plate_r)
                                if len(np_)>=4: p0=np_
                    frame_data={"t":round(t,4),"x":round(cx/wp,5),"y":round(cy/hp,5)}
                    results.append(frame_data)
                    yield json.dumps({"frame":frame_data})+"\n"
                    proc+=1
                    if proc%30==0: await asyncio.sleep(0)
                prev_gray=curr_gray; fn+=1
            cap.release()
            if results:
                xs=np.array([f['x'] for f in results]); ys=np.array([f['y'] for f in results])
                xs_s,ys_s=smooth_coords(xs.tolist(),ys.tolist(),window=7)
                for i,f in enumerate(results): f['x']=round(xs_s[i],5); f['y']=round(ys_s[i],5)
            reps=detect_reps(results)
            rep_metrics=[]
            for rep in reps:
                seg=results[rep['start']:rep['end']+1]
                if len(seg)<2: continue
                ys_r=[f['y'] for f in seg]; ts_r=[f['t'] for f in seg]
                rom_m=abs(max(ys_r)-min(ys_r))*hp/px_per_m
                vels_ms=[abs(ys_r[i]-ys_r[i-1])*hp/max(ts_r[i]-ts_r[i-1],0.001)/px_per_m for i in range(1,len(ys_r))]
                rep_metrics.append({"start":rep['start'],"end":rep['end'],
                    "rom_m":round(rom_m,3),"mean_ms":round(float(np.mean(vels_ms)),3),
                    "peak_ms":round(float(np.max(vels_ms)),3)})
            yield json.dumps({"done":True,"frames":results,"reps":rep_metrics,
                "cap_w":raw_w,"cap_h":raw_h,"fps":fps,"rotation":rot,
                "px_per_m":round(px_per_m,2)})+"\n"
        except Exception as e:
            yield json.dumps({"error":str(e)})+"\n"
        finally:
            try: os.unlink(tmp)
            except: pass

    return StreamingResponse(stream(), media_type="application/x-ndjson")
