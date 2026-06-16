import os, cv2, numpy as np, tempfile, subprocess, json
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ForceTrack Bar Path API", version="0.7.0")
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
    if rot == 90:  return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rot == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if rot == 270: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

def make_csrt():
    try:
        params = cv2.TrackerCSRT_Params()
        params.use_segmentation    = True
        params.use_channel_weights = True
        params.filter_lr           = 0.01
        params.padding             = 3.5
        params.histogram_lr        = 0.02
        params.num_hog_channels_used = 18
        return cv2.TrackerCSRT_create(params)
    except Exception:
        try:    return cv2.TrackerCSRT_create()
        except: return cv2.TrackerKCF_create()

LK_PARAMS = dict(winSize=(31,31), maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 50, 0.001))
GF_PARAMS = dict(maxCorners=80, qualityLevel=0.01, minDistance=3, blockSize=7)

def hough_detect(gray, min_r, max_r, search_box=None):
    if search_box is not None:
        sx,sy,sw,sh = [int(v) for v in search_box]
        h,w = gray.shape
        sx,sy = max(0,sx),max(0,sy)
        ex,ey = min(w,sx+sw),min(h,sy+sh)
        if ex<=sx or ey<=sy: return None
        roi = gray[sy:ey,sx:ex]; ox,oy = sx,sy
    else:
        roi,ox,oy = gray,0,0
    b = cv2.GaussianBlur(roi,(9,9),2)
    for p2 in [30,24,18,12]:
        c = cv2.HoughCircles(b,cv2.HOUGH_GRADIENT,1.2,40,
                             param1=80,param2=p2,minRadius=min_r,maxRadius=max_r)
        if c is not None:
            best = max(c[0],key=lambda x:x[2])
            return float(best[0]+ox),float(best[1]+oy),float(best[2])
    return None

def template_match(gray,tmpl,last_cx,last_cy,search_half,threshold=0.35):
    th,tw = tmpl.shape[:2]
    sx=max(0,int(last_cx-search_half)); sy=max(0,int(last_cy-search_half))
    ex=min(gray.shape[1],int(last_cx+search_half))
    ey=min(gray.shape[0],int(last_cy+search_half))
    if (ex-sx)<tw or (ey-sy)<th: return None
    roi=gray[sy:ey,sx:ex]
    res=cv2.matchTemplate(roi,tmpl,cv2.TM_CCOEFF_NORMED)
    _,max_val,_,max_loc=cv2.minMaxLoc(res)
    if max_val<threshold: return None
    return float(sx+max_loc[0]+tw//2),float(sy+max_loc[1]+th//2)

def lk_track(prev_gray,curr_gray,pts):
    if pts is None or len(pts)<4: return None,None,None
    new_pts,status,_ = cv2.calcOpticalFlowPyrLK(
        prev_gray,curr_gray,pts,None,**LK_PARAMS)
    if new_pts is None: return None,None,None
    mask = status.ravel()==1
    good = new_pts[mask]
    if len(good)<4: return None,None,None
    return float(np.median(good[:,0])),float(np.median(good[:,1])),good.reshape(-1,1,2)

def seed_lk(gray,cx,cy,plate_r):
    m = np.zeros_like(gray,dtype=np.uint8)
    cv2.ellipse(m,(int(cx),int(cy)),(int(plate_r),int(plate_r)),0,0,360,255,-1)
    return cv2.goodFeaturesToTrack(gray,mask=m,**GF_PARAMS)

def clamp_bbox(bbox,wp,hp):
    x,y,w,h=bbox
    x,y=max(0,int(x)),max(0,int(y))
    w=min(wp-x,int(w)); h=min(hp-y,int(h))
    return (x,y,max(1,w),max(1,h))

def bbox_from_center(cx,cy,r,scale=1.6):
    half=r*scale; return (cx-half,cy-half,half*2,half*2)

def get_actual_timestamps(path,total):
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","packet=pts_time","-of",
             "csv=print_section=0",path],
            capture_output=True,text=True,timeout=60)
        ts = []
        for line in r.stdout.strip().split('\n'):
            line=line.strip()
            if line and line!='N/A':
                try: ts.append(float(line))
                except: pass
        if len(ts)>0 and abs(len(ts)-total)/max(total,1)<0.1:
            return ts
    except: pass
    return None

@app.get("/health")
def health(): return {"status":"ok","version":"0.7.0"}

@app.post("/analyze")
async def analyze(video: UploadFile=File(...), params: str=Form("{}"), api_key: str=Form("")):
    if VALID_KEY and api_key!=VALID_KEY:
        raise HTTPException(401,"Invalid API key")
    p = {}
    try: p = json.loads(params)
    except: pass
    tmp = tempfile.mktemp(suffix=".mp4")
    try:
        data = await video.read()
        if len(data)>600*1024*1024:
            raise HTTPException(400,"Video too large (max 600MB)")
        with open(tmp,"wb") as f: f.write(data)
        del data
        rot = get_rotation(tmp)
        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened(): raise HTTPException(400,"Cannot open video")
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_ts = get_actual_timestamps(tmp, total)
        ret,f0 = cap.read()
        if not ret: raise HTTPException(400,"Cannot read first frame")
        f0 = rotate_frame(f0,rot)
        raw_h,raw_w = f0.shape[:2]
        wp,hp = raw_w,raw_h
        f0s = f0.copy()
        g0  = cv2.cvtColor(f0s,cv2.COLOR_BGR2GRAY)
        del f0
        min_r = max(8,int(hp*0.05))
        max_r = min(wp//2,int(hp*0.45))
        if 'start_x' in p and 'start_y' in p:
            cx0 = float(p['start_x']) * wp
            cy0 = float(p['start_y']) * hp
            search_r = int(min(wp,hp)*0.15)
            sb = (cx0-search_r,cy0-search_r,search_r*2,search_r*2)
            det = hough_detect(g0,min_r,max_r,sb)
            r0 = det[2] if det else int(min(wp,hp)*0.06)
        else:
            det = hough_detect(g0,min_r,max_r)
            if det is None: det=(wp/2,hp/2,min_r*2)
            cx0,cy0,r0 = det
        plate_r = r0
        pad=int(plate_r*1.8)
        tx1,ty1=max(0,int(cx0-pad)),max(0,int(cy0-pad))
        tx2,ty2=min(wp,int(cx0+pad)),min(hp,int(cy0+pad))
        plate_tmpl=g0[ty1:ty2,tx1:tx2].copy()
        lk_pts=seed_lk(g0,cx0,cy0,plate_r)
        bbox0=clamp_bbox(bbox_from_center(cx0,cy0,r0,1.6),wp,hp)
        tracker=make_csrt()
        tracker.init(f0s,bbox0)
        del f0s
        results=[]; last_cx,last_cy=cx0,cy0; prev_gray=g0.copy()
        max_jump_sq   =(plate_r*3.0)**2
        csrt_agree_sq =(plate_r*2.0)**2
        reinit_half   =int(plate_r*6)
        tmpl_half     =int(plate_r*8)
        LK_REFRESH    =30
        consecutive_bad=0
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        frame_idx=0
        while True:
            if actual_ts and frame_idx < len(actual_ts):
                t = actual_ts[frame_idx]
            else:
                t = frame_idx / fps
            ret,frame=cap.read()
            if not ret: break
            frame=rotate_frame(frame,rot)
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if frame_idx>0 and frame_idx%LK_REFRESH==0:
                fresh=seed_lk(gray,last_cx,last_cy,plate_r)
                if fresh is not None and len(fresh)>=4:
                    lk_pts=fresh
            lk_cx,lk_cy,lk_pts_new=lk_track(prev_gray,gray,lk_pts)
            ok,bbox=tracker.update(frame)
            def _reinit(rx,ry):
                nonlocal tracker,last_cx,last_cy,lk_pts,consecutive_bad
                nb=clamp_bbox(bbox_from_center(rx,ry,plate_r,1.6),wp,hp)
                tracker=make_csrt(); tracker.init(frame,nb)
                lk_pts=seed_lk(gray,rx,ry,plate_r)
                last_cx,last_cy=rx,ry; consecutive_bad=0
                return rx,ry
            def try_recover():
                sb=(last_cx-reinit_half,last_cy-reinit_half,
                    reinit_half*2,reinit_half*2)
                rd=hough_detect(gray,int(plate_r*0.6),int(plate_r*1.5),sb)
                if rd: return _reinit(rd[0],rd[1])
                tm=template_match(gray,plate_tmpl,last_cx,last_cy,tmpl_half)
                if tm: return _reinit(tm[0],tm[1])
                rd2=hough_detect(gray,int(plate_r*0.6),int(plate_r*1.5))
                if rd2:
                    rx,ry,_=rd2
                    if (rx-last_cx)**2+(ry-last_cy)**2<(plate_r*12)**2:
                        return _reinit(rx,ry)
                return None
            if lk_cx is not None:
                lk_jump=(lk_cx-last_cx)**2+(lk_cy-last_cy)**2
                if lk_jump>max_jump_sq:
                    consecutive_bad+=1
                    if consecutive_bad>=2:
                        res=try_recover()
                        cx,cy=res if res else (last_cx,last_cy)
                    else:
                        cx,cy=last_cx,last_cy
                else:
                    consecutive_bad=0
                    if ok:
                        csrt_cx=bbox[0]+bbox[2]/2; csrt_cy=bbox[1]+bbox[3]/2
                        csrt_jump=(csrt_cx-last_cx)**2+(csrt_cy-last_cy)**2
                        lk_vs_csrt=(lk_cx-csrt_cx)**2+(lk_cy-csrt_cy)**2
                        if csrt_jump<max_jump_sq and lk_vs_csrt<csrt_agree_sq:
                            cx=lk_cx*0.7+csrt_cx*0.3
                            cy=lk_cy*0.7+csrt_cy*0.3
                        else:
                            cx,cy=lk_cx,lk_cy
                    else:
                        cx,cy=lk_cx,lk_cy
                    last_cx,last_cy=lk_cx,lk_cy
            else:
                consecutive_bad+=1
                if ok:
                    cx=bbox[0]+bbox[2]/2; cy=bbox[1]+bbox[3]/2
                    if (cx-last_cx)**2+(cy-last_cy)**2>max_jump_sq:
                        if consecutive_bad>=2:
                            res=try_recover(); cx,cy=res if res else (last_cx,last_cy)
                        else:
                            cx,cy=last_cx,last_cy
                    else:
                        consecutive_bad=0; last_cx,last_cy=cx,cy
                else:
                    if consecutive_bad>=2:
                        res=try_recover(); cx,cy=res if res else (last_cx,last_cy)
                    else:
                        cx,cy=last_cx,last_cy
            if lk_pts_new is not None: lk_pts=lk_pts_new
            prev_gray=gray.copy(); del frame
            results.append({"t":round(t,6),"x":round(cx/wp,5),"y":round(cy/hp,5)})
            frame_idx+=1
        cap.release()
        return {"frames":results,"cap_w":raw_w,"cap_h":raw_h,
                "fps":fps,"rotation":rot,"frame_count":frame_idx}
    finally:
        try: os.unlink(tmp)
        except: pass
