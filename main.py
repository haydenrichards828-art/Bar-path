import os, cv2, numpy as np, tempfile, subprocess, json, asyncio
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="ForceTrack Bar Path API", version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PLATE_DIAMETER_M = 0.450
# Lighter CLAHE — aggressive enhancement creates false features in background
CLAHE       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
FB_THRESHOLD = 2.0
TARGET_RES   = 1920

def get_rotation(path):
    try:
        r = subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
            "stream_tags=rotate","-of","default=noprint_wrappers=1:nokey=1",path],
            capture_output=True, text=True, timeout=10)
        v = r.stdout.strip(); return int(v) if v else 0
    except: return 0

def normalise_video(path):
    try:
        probe = subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
            "-show_entries","stream=width,height,codec_name","-of","json",path],
            capture_output=True, text=True, timeout=10)
        info = json.loads(probe.stdout)
        stream = info.get("streams",[{}])[0]
        w = int(stream.get("width",0)); h = int(stream.get("height",0))
        codec = stream.get("codec_name","")
        heavy = codec not in ("h264","hevc","vp9","av1","vp8")
        if max(w,h) <= TARGET_RES and not heavy: return path
        out = path + "_norm.mp4"
        scale = f"scale='if(gt(iw,ih),{TARGET_RES},-2)':'if(gt(iw,ih),-2,{TARGET_RES})'"
        r2 = subprocess.run(["ffmpeg","-i",path,"-vf",scale,
            "-c:v","libx264","-crf","20","-preset","fast","-an","-y",out],
            capture_output=True, timeout=180)
        if r2.returncode == 0:
            os.unlink(path); return out
    except: pass
    return path

def rotate_frame(frame, rot):
    if rot == 90:  return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rot == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if rot == 270: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

def enhance(gray):
    return CLAHE.apply(gray)

def hough_detect(gray, min_r, max_r, cx_hint, cy_hint, search_pad):
    h, w = gray.shape
    sx=max(0,int(cx_hint-search_pad)); sy=max(0,int(cy_hint-search_pad))
    ex=min(w,int(cx_hint+search_pad)); ey=min(h,int(cy_hint+search_pad))
    if ex-sx<10 or ey-sy<10: return None
    roi=enhance(gray[sy:ey,sx:ex]); b=cv2.GaussianBlur(roi,(9,9),2)
    for p2 in [30,24,18,12,8,5]:
        c=cv2.HoughCircles(b,cv2.HOUGH_GRADIENT,1.2,(ey-sy)//2,param1=80,param2=p2,minRadius=min_r,maxRadius=max_r)
        if c is not None:
            best=min(c[0],key=lambda v:(v[0]+sx-cx_hint)**2+(v[1]+sy-cy_hint)**2)
            return float(best[0]+sx),float(best[1]+sy),float(best[2])
    return None

def hough_recover(gray, cx, cy, plate_r):
    min_r=int(plate_r*0.5); max_r=int(plate_r*1.6)
    for scale in [2.5,4.0,6.0,9.0]:
        result=hough_detect(gray,min_r,max_r,cx,cy,plate_r*scale)
        if result:
            _,_,rr=result
            if 0.45*plate_r<rr<1.8*plate_r: return result
    return None

def refine_center_radial(gray, cx, cy, r, n_rays=16):
    h,w=gray.shape; eg=enhance(gray); edge_pts=[]; search=max(4,int(r*0.18))
    for i in range(n_rays):
        angle=np.radians(i*360.0/n_rays); dx=np.cos(angle); dy=np.sin(angle)
        ex_=cx+r*0.95*dx; ey_=cy+r*0.95*dy
        best_grad=0; best_x=None; best_y=None
        for d in range(-search,search+1):
            px=int(round(ex_+d*dx)); py=int(round(ey_+d*dy))
            if 2<=px<w-2 and 2<=py<h-2:
                gx=float(eg[py,px+1])-float(eg[py,px-1])
                gy=float(eg[py+1,px])-float(eg[py-1,px])
                g=abs(gx*dx+gy*dy)
                if g>best_grad: best_grad=g; best_x=px; best_y=py
        if best_x is not None and best_grad>8: edge_pts.append((best_x,best_y))
    if len(edge_pts)<6: return cx,cy
    pts=np.array(edge_pts,dtype=np.float64); xs,ys=pts[:,0],pts[:,1]
    A=np.column_stack([xs,ys,np.ones(len(xs))]); b2=-(xs**2+ys**2)
    try:
        res,_,_,_=np.linalg.lstsq(A,b2,rcond=None); D,E,_=res
        ncx,ncy=-D/2.0,-E/2.0
        if (ncx-cx)**2+(ncy-cy)**2<(r*0.3)**2: return float(ncx),float(ncy)
    except: pass
    return cx,cy

def seed_pts(gray, cx, cy, r):
    """Interior features only — NO rim points. Rim points sit at the
    plate-background boundary and contaminate the displacement median."""
    eg=enhance(gray)
    mask=np.zeros(gray.shape,np.uint8)
    # Use 0.80x radius so all features are well inside the plate
    cv2.circle(mask,(int(cx),int(cy)),int(max(4,r*0.80)),255,-1)
    pts=cv2.goodFeaturesToTrack(eg,150,0.01,3,mask=mask,blockSize=7)
    if pts is not None and len(pts)>=8: return pts
    # Fallback grid
    grid=[]; step=max(2.0,r*0.25); dy=-r*0.7
    while dy<=r*0.7:
        dx=-r*0.7
        while dx<=r*0.7:
            if dx*dx+dy*dy<=r*r*0.49: grid.append([[cx+dx,cy+dy]])
            dx+=step
        dy+=step
    return np.array(grid,dtype=np.float32) if grid else np.array([[[cx,cy]]],dtype=np.float32)

def filter_in_plate(p0, cx, cy, plate_r, margin=1.1):
    """Discard any features that have drifted outside the plate circle.
    This is the critical step that prevents background contamination."""
    pts=p0.reshape(-1,2)
    dists=np.sqrt((pts[:,0]-cx)**2+(pts[:,1]-cy)**2)
    keep=dists<plate_r*margin
    if keep.sum()>=4: return p0[keep]
    return p0  # keep all if filter would leave too few

def smooth_coords(xs,ys,window=5):
    if len(xs)<window: return xs,ys
    w=window if window%2==1 else window+1; k=np.ones(w)/w
    xs_s=np.convolve(xs,k,mode='same'); ys_s=np.convolve(ys,k,mode='same')
    h=w//2; xs_s[:h]=xs[:h]; xs_s[-h:]=xs[-h:]; ys_s[:h]=ys[:h]; ys_s[-h:]=ys[-h:]
    return xs_s.tolist(),ys_s.tolist()

def detect_reps(frames,min_frames=8):
    if len(frames)<min_frames*2: return []
    ys=[f['y'] for f in frames]; ts=[f['t'] for f in frames]
    vel=[(ys[i]-ys[i-1])/max(ts[i]-ts[i-1],0.001) for i in range(1,len(ys))]
    vel=[vel[0]]+vel; svel=np.convolve(vel,np.ones(5)/5,mode='same').tolist()
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

LK=dict(winSize=(21,21),maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,0.01))

@app.get("/health")
def health(): return {"status":"ok","version":"2.1.0"}

@app.post("/analyze")
async def analyze(video: UploadFile=File(...), params: str=Form("{}"), api_key: str=Form("")):
    tmp=tempfile.mktemp(suffix=".mp4")
    try:
        data=await video.read()
        if len(data)>600*1024*1024: raise HTTPException(400,"Video too large")
        with open(tmp,"wb") as f: f.write(data)
        del data
    except HTTPException: raise
    except Exception as e: raise HTTPException(500,f"Save failed: {e}")

    async def stream():
        work=tmp
        try:
            work=normalise_video(tmp)
            rot=get_rotation(work)
            cap=cv2.VideoCapture(work)
            if not cap.isOpened(): yield json.dumps({"error":"Cannot open video"})+"\n"; return
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            yield json.dumps({"meta":{"total_frames":total,"fps":fps}})+"\n"
            ret,f0=cap.read()
            if not ret: yield json.dumps({"error":"Cannot read first frame"})+"\n"; return
            f0=rotate_frame(f0,rot); raw_h,raw_w=f0.shape[:2]; wp,hp=raw_w,raw_h
            g0=cv2.cvtColor(f0,cv2.COLOR_BGR2GRAY); del f0
            try: p=json.loads(params)
            except: p={}
            tx=float(p.get("start_x",0.5))*wp; ty=float(p.get("start_y",0.5))*hp
            min_r=max(10,int(hp*0.05)); max_r=min(wp//2,int(hp*0.47))
            det=hough_detect(g0,min_r,max_r,tx,ty,int(min(wp,hp)*0.35))
            if det is None: det=hough_detect(g0,min_r,max_r,wp//2,hp//2,min(wp,hp)//2)
            if det is None: det=(tx,ty,max(min_r*2,int(hp*0.09)))
            cx,cy,plate_r=det; plate_r=max(min_r,plate_r)
            cx,cy=refine_center_radial(g0,cx,cy,plate_r)
            px_per_m=plate_r/(PLATE_DIAMETER_M/2.0)
            prev_gray=enhance(g0.copy()); p0=seed_pts(g0,cx,cy,plate_r); del g0
            results=[{"t":0.0,"x":round(cx/wp,5),"y":round(cy/hp,5)}]
            yield json.dumps({"frame":results[0]})+"\n"
            frames_since_hough=0; HOUGH_EVERY=10; MIN_FEATURES=30
            cap.set(cv2.CAP_PROP_POS_FRAMES,0); fn=0; last_t=0.0

            while True:
                ret,frame=cap.read()
                if not ret: break
                frame=rotate_frame(frame,rot)
                raw_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); del frame
                msec_t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
                t=msec_t if msec_t>last_t else fn/fps
                curr_gray=enhance(raw_gray)

                p1,sf,_=cv2.calcOpticalFlowPyrLK(prev_gray,curr_gray,p0,None,**LK)
                p0r,sb,_=cv2.calcOpticalFlowPyrLK(curr_gray,prev_gray,p1,None,**LK)
                fb=np.abs(p0-p0r).reshape(-1,2).max(axis=1)
                good=(sf.ravel()==1)&(sb.ravel()==1)&(fb<FB_THRESHOLD)

                if good.sum()>=4:
                    gn=p1[good]; go=p0[good]
                    raw_cx=float(np.clip(cx+float(np.median(gn[:,0,0]-go[:,0,0])),plate_r,wp-plate_r))
                    raw_cy=float(np.clip(cy+float(np.median(gn[:,0,1]-go[:,0,1])),plate_r,hp-plate_r))
                    cx,cy=refine_center_radial(raw_gray,raw_cx,raw_cy,plate_r)
                    p0=gn.reshape(-1,1,2)
                    # ── CRITICAL: discard features that drifted onto background ──
                    p0=filter_in_plate(p0,cx,cy,plate_r,margin=1.1)
                else:
                    rd=hough_recover(raw_gray,cx,cy,plate_r)
                    if rd:
                        ncx,ncy,_=rd
                        cx,cy=refine_center_radial(raw_gray,ncx,ncy,plate_r)
                        p0=seed_pts(raw_gray,cx,cy,plate_r); frames_since_hough=0

                # Replenish if needed after drift filter
                if len(p0)<MIN_FEATURES:
                    new_pts=seed_pts(raw_gray,cx,cy,plate_r)
                    if len(new_pts)>len(p0): p0=new_pts

                frames_since_hough+=1
                if frames_since_hough>=HOUGH_EVERY:
                    frames_since_hough=0
                    rd=hough_detect(raw_gray,int(plate_r*0.65),int(plate_r*1.35),cx,cy,plate_r*2.8)
                    if rd:
                        ncx,ncy,rr=rd
                        if 0.55*plate_r<rr<1.7*plate_r and (ncx-cx)**2+(ncy-cy)**2<(plate_r*1.5)**2:
                            ncx,ncy=refine_center_radial(raw_gray,ncx,ncy,plate_r)
                            cx,cy=ncx,ncy
                            plate_r=plate_r*0.92+rr*0.08
                            px_per_m=plate_r/(PLATE_DIAMETER_M/2.0)
                            p0=seed_pts(raw_gray,cx,cy,plate_r)

                del raw_gray
                frame_data={"t":round(t,4),"x":round(cx/wp,5),"y":round(cy/hp,5)}
                results.append(frame_data); yield json.dumps({"frame":frame_data})+"\n"
                prev_gray=curr_gray; fn+=1; last_t=t
                if fn%30==0: await asyncio.sleep(0)

            cap.release()
            if results:
                xs=np.array([f['x'] for f in results]); ys=np.array([f['y'] for f in results])
                xs_s,ys_s=smooth_coords(xs.tolist(),ys.tolist(),window=5)
                for i,f in enumerate(results): f['x']=round(xs_s[i],5); f['y']=round(ys_s[i],5)
            reps=detect_reps(results); rep_metrics=[]
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
            for f in set([tmp,work]):
                try: os.unlink(f)
                except: pass

    return StreamingResponse(stream(),media_type="application/x-ndjson")
