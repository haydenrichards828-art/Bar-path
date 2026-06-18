import os, cv2, numpy as np, tempfile, subprocess, json, asyncio
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="ForceTrack Bar Path API", version="3.0.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PLATE_DIAMETER_M = 0.450
TARGET_RES       = 1920

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
        w=int(stream.get("width",0)); h=int(stream.get("height",0))
        codec=stream.get("codec_name","")
        if max(w,h)<=TARGET_RES and codec in ("h264","hevc","vp9","av1","vp8"): return path
        out=path+"_norm.mp4"
        scale=f"scale='if(gt(iw,ih),{TARGET_RES},-2)':'if(gt(iw,ih),-2,{TARGET_RES})'"
        r2=subprocess.run(["ffmpeg","-i",path,"-vf",scale,
            "-c:v","libx264","-crf","20","-preset","fast","-an","-y",out],
            capture_output=True, timeout=180)
        if r2.returncode==0: os.unlink(path); return out
    except: pass
    return path

def rotate_frame(frame, rot):
    if rot==90:  return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rot==180: return cv2.rotate(frame, cv2.ROTATE_180)
    if rot==270: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

def prep(gray):
    return cv2.equalizeHist(gray)

def hough_find(gray, min_r, max_r, cx_hint, cy_hint, pad):
    h,w=gray.shape
    sx=max(0,int(cx_hint-pad)); sy=max(0,int(cy_hint-pad))
    ex=min(w,int(cx_hint+pad)); ey=min(h,int(cy_hint+pad))
    if ex-sx<10 or ey-sy<10: return None
    roi=gray[sy:ey,sx:ex]; b=cv2.GaussianBlur(roi,(9,9),2)
    for p2 in [30,24,18,12,8,5]:
        c=cv2.HoughCircles(b,cv2.HOUGH_GRADIENT,1.2,(ey-sy)//2,param1=80,param2=p2,minRadius=min_r,maxRadius=max_r)
        if c is not None:
            best=min(c[0],key=lambda v:(v[0]+sx-cx_hint)**2+(v[1]+sy-cy_hint)**2)
            return float(best[0]+sx),float(best[1]+sy),float(best[2])
    return None

def make_template(proc_gray, cx, cy, tpad):
    h,w=proc_gray.shape
    x0=max(0,int(cx)-tpad); y0=max(0,int(cy)-tpad)
    x1=min(w,int(cx)+tpad); y1=min(h,int(cy)+tpad)
    if x1-x0<10 or y1-y0<10: return None,None,tpad
    patch=proc_gray[y0:y1,x0:x1].copy()
    actual_hx=int(cx)-x0; actual_hy=int(cy)-y0
    mask=np.zeros(patch.shape,dtype=np.uint8)
    r=int(min(actual_hx,actual_hy)*0.90)
    cv2.circle(mask,(actual_hx,actual_hy),max(4,r),255,-1)
    return patch,mask,(actual_hx,actual_hy)

def template_match(proc_gray, tmpl, tmpl_mask, half_wh, cx, cy, search_r):
    """SQDIFF_NORMED with mask — the reliable, well-supported masked-matching
    combo in OpenCV. Lower score = better match (0 = perfect). Unlike CCORR_NORMED,
    SQDIFF is shape/texture-sensitive and won't be fooled by flat dark regions
    (floor, shadows, rack uprights) that happen to be similar average brightness."""
    if tmpl is None: return None
    th,tw=tmpl.shape; h,w=proc_gray.shape
    hx,hy=half_wh
    sx=max(0,int(cx)-int(search_r)-hx)
    sy=max(0,int(cy)-int(search_r)-hy)
    ex=min(w-tw,int(cx)+int(search_r)-hx+1)
    ey=min(h-th,int(cy)+int(search_r)-hy+1)
    if ex<=sx or ey<=sy: return None
    region=proc_gray[sy:ey+th, sx:ex+tw]
    if region.shape[0]<th or region.shape[1]<tw: return None
    result=cv2.matchTemplate(region,tmpl,cv2.TM_SQDIFF_NORMED,mask=tmpl_mask)
    min_val,_,min_loc,_=cv2.minMaxLoc(result)
    if min_val>0.35: return None  # lower=better; reject poor matches
    ncx=float(sx+min_loc[0]+hx)
    ncy=float(sy+min_loc[1]+hy)
    confidence=1.0-min_val
    return ncx,ncy,confidence

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

@app.get("/health")
def health(): return {"status":"ok","version":"3.0.3"}

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
            g0=cv2.cvtColor(f0,cv2.COLOR_BGR2GRAY); p0=prep(g0); del f0
            try: p=json.loads(params)
            except: p={}
            tx=float(p.get("start_x",0.5))*wp; ty=float(p.get("start_y",0.5))*hp
            min_r=max(10,int(hp*0.05)); max_r=min(wp//2,int(hp*0.47))
            det=hough_find(p0,min_r,max_r,tx,ty,int(min(wp,hp)*0.35))
            if det is None: det=hough_find(p0,min_r,max_r,wp//2,hp//2,min(wp,hp)//2)
            if det is None: det=(tx,ty,max(min_r*2,int(hp*0.09)))
            cx,cy,plate_r=det; plate_r=max(min_r,plate_r)
            px_per_m=plate_r/(PLATE_DIAMETER_M/2.0)
            tpad=int(plate_r*0.9)
            tmpl,tmpl_mask,half_wh=make_template(p0,cx,cy,tpad)
            if tmpl is None: yield json.dumps({"error":"Could not build template"})+"\n"; return
            del g0
            results=[{"t":0.0,"x":round(cx/wp,5),"y":round(cy/hp,5)}]
            yield json.dumps({"frame":results[0]})+"\n"
            cap.set(cv2.CAP_PROP_POS_FRAMES,0); fn=0; last_t=0.0
            frames_since_update=0; UPDATE_EVERY=8

            while True:
                ret,frame=cap.read()
                if not ret: break
                frame=rotate_frame(frame,rot)
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); del frame
                pg=prep(gray)
                msec_t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
                t=msec_t if msec_t>last_t else fn/fps

                match=template_match(pg,tmpl,tmpl_mask,half_wh,cx,cy,plate_r*3.0)
                if match is None:
                    match=template_match(pg,tmpl,tmpl_mask,half_wh,cx,cy,plate_r*6.0)

                if match is not None:
                    ncx,ncy,conf=match
                    cx,cy=ncx,ncy
                    frames_since_update+=1
                    if frames_since_update>=UPDATE_EVERY and conf>0.70:
                        frames_since_update=0
                        new_tmpl,new_mask,_=make_template(pg,cx,cy,tpad)
                        if new_tmpl is not None and new_tmpl.shape==tmpl.shape:
                            tmpl=cv2.addWeighted(tmpl,0.4,new_tmpl,0.6,0)

                del pg,gray
                frame_data={"t":round(t,4),"x":round(cx/wp,5),"y":round(cy/hp,5)}
                results.append(frame_data); yield json.dumps({"frame":frame_data})+"\n"
                fn+=1; last_t=t
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
