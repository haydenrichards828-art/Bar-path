import os, cv2, numpy as np, tempfile, subprocess, json, asyncio
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="ForceTrack Bar Path API", version="7.3.3")
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

def hough_find(gray, min_r, max_r, cx_hint, cy_hint, pad):
    h,w=gray.shape
    sx=max(0,int(cx_hint-pad)); sy=max(0,int(cy_hint-pad))
    ex=min(w,int(cx_hint+pad)); ey=min(h,int(cy_hint+pad))
    if ex-sx<10 or ey-sy<10: return None
    eq=cv2.equalizeHist(gray[sy:ey,sx:ex]); b=cv2.GaussianBlur(eq,(9,9),2)
    for p2 in [30,24,18,12,8,5]:
        c=cv2.HoughCircles(b,cv2.HOUGH_GRADIENT,1.2,(ey-sy)//2,param1=80,param2=p2,minRadius=min_r,maxRadius=max_r)
        if c is not None:
            best=min(c[0],key=lambda v:(v[0]+sx-cx_hint)**2+(v[1]+sy-cy_hint)**2)
            return float(best[0]+sx),float(best[1]+sy),float(best[2])
    return None

def build_hist(bgr_frame, cx, cy, r):
    h,w=bgr_frame.shape[:2]
    x0=max(0,int(cx-r)); y0=max(0,int(cy-r))
    x1=min(w,int(cx+r)); y1=min(h,int(cy+r))
    roi=bgr_frame[y0:y1,x0:x1]
    if roi.size==0: return None
    roi_h,roi_w=roi.shape[:2]
    # Circular mask — only sample pixels inside the plate disc, not square ROI corners
    Ygrid,Xgrid=np.ogrid[:roi_h,:roi_w]
    circ_mask=((Xgrid-(cx-x0))**2+(Ygrid-(cy-y0))**2<=r**2).astype(np.uint8)*255
    hsv_roi=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    # S>40, V>40 excludes background grays/whites and deep shadow; keeps plate rubber+hub
    hsv_mask=cv2.inRange(hsv_roi,np.array((0.,40.,40.)),np.array((180.,255.,255.)))
    mask=cv2.bitwise_and(circ_mask,hsv_mask)
    hist=cv2.calcHist([hsv_roi],[0,1],mask,[30,32],[0,180,0,256])
    cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX)
    return hist,(x0,y0,x1-x0,y1-y0)

def plate_bp_score(bgr_frame, hist, cx, cy, r):
    """Mean back-projection score inside the circular region. Higher = better plate match."""
    fh, fw = bgr_frame.shape[:2]
    x0, y0 = max(0, int(cx-r)), max(0, int(cy-r))
    x1, y1 = min(fw, int(cx+r)), min(fh, int(cy+r))
    if x1-x0 < 8 or y1-y0 < 8: return 0.0
    hsv = cv2.cvtColor(bgr_frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    bp  = cv2.calcBackProject([hsv], [0,1], hist, [0,180,0,256], 1)
    return float(bp.mean())

TERM_CRIT=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,15,1)

def camshift_step(bgr_frame, hist, track_window):
    hsv=cv2.cvtColor(bgr_frame,cv2.COLOR_BGR2HSV)
    dst=cv2.calcBackProject([hsv],[0,1],hist,[0,180,0,256],1)
    ret,new_window=cv2.CamShift(dst,track_window,TERM_CRIT)
    x,y,w,h=new_window
    if w<4 or h<4: return None,track_window
    cx=x+w/2.0; cy=y+h/2.0
    return (cx,cy),new_window

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
def health(): return {"status":"ok","version":"7.3.3"}

@app.post("/analyze")
async def analyze(video: UploadFile=File(...), params: str=Form("{}"), api_key: str=Form("")):
    ct=video.content_type or ""
    ext=".webm" if "webm" in ct else ".mp4"
    tmp=tempfile.mktemp(suffix=ext)
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
            # Disable OpenCV auto-rotation so rotate_frame() is the single source of truth.
            # Without this, some builds apply rotation from metadata in cap.read() AND
            # rotate_frame() applies it again → double-rotation for iOS MOV files.
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            if not cap.isOpened(): yield json.dumps({"error":"Cannot open video"})+"\n"; return
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
            total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            yield json.dumps({"meta":{"total_frames":total,"fps":fps}})+"\n"
            ret,f0=cap.read()
            if not ret: yield json.dumps({"error":"Cannot read first frame"})+"\n"; return
            f0=rotate_frame(f0,rot); raw_h,raw_w=f0.shape[:2]; wp,hp=raw_w,raw_h
            g0=cv2.cvtColor(f0,cv2.COLOR_BGR2GRAY)
            try: p=json.loads(params)
            except: p={}
            tx=float(p.get("start_x",0.5))*wp; ty=float(p.get("start_y",0.5))*hp
            # Radius range tuned to a single bumper plate: 5–30% of the shorter frame side
            short_side=min(wp,hp)
            min_r=max(10,int(short_side*0.05)); max_r=int(short_side*0.30)
            det=hough_find(g0,min_r,max_r,tx,ty,int(short_side*0.35))
            if det is None: det=hough_find(g0,min_r,max_r,wp//2,hp//2,min(wp,hp)//2)
            if det is None: det=(tx,ty,max(min_r*2,int(hp*0.09)))
            cx,cy,plate_r=det; plate_r=max(min_r,plate_r)
            px_per_m=plate_r/(PLATE_DIAMETER_M/2.0)
            # Pre-lift lock: suppress Hough oscillation between nearby rings while the bar
            # is stationary. Any per-frame jump larger than ~1.7 m/s equivalent must persist
            # for LOCK_DEBOUNCE consecutive frames before being accepted. Single-frame jumps
            # (A→B→A oscillation) are rejected; sustained movement (bar genuinely lifting)
            # passes. Lock deactivates permanently once real movement is confirmed.
            lock_jump_thresh = plate_r * 0.25 * (30.0 / fps)  # fps-scaled ~1.7 m/s
            lock_debounce    = 3   # frames of consistency required to accept a large jump
            lock_active      = True
            lock_candidate   = None  # (cx, cy) of candidate being debounced
            lock_streak      = 0
            lock_origin_cx, lock_origin_cy = cx, cy  # initial detection position

            hist_data=build_hist(f0,cx,cy,plate_r)
            if hist_data is None: yield json.dumps({"error":"Could not build colour histogram"})+"\n"; return
            hist,_=hist_data
            # Appearance reference: plate score at frame 0 to gate Hough candidates.
            # Threshold = max absolute floor or 15% of reference, whichever is larger.
            ref_score=plate_bp_score(f0,hist,cx,cy,plate_r)
            min_score=max(12.0,ref_score*0.15)
            # Gate is only meaningful when the plate has enough visible colour for the
            # histogram to be discriminative. Fully dark plates (black rubber/plastic
            # with no coloured hub) produce ref_score near 0 because every pixel is
            # filtered out by the S>40/V>40 mask. In that case disabling the gate is
            # correct: it would only produce false rejections of the real plate.
            gate_enabled=ref_score>=20.0
            del f0,g0

            results=[]
            cap.set(cv2.CAP_PROP_POS_FRAMES,0); fn=0; last_t=0.0
            vx,vy=0.0,0.0; vel_hist=[]; bad_streak=0

            while True:
                ret,frame=cap.read()
                if not ret: break
                frame=rotate_frame(frame,rot)
                msec_t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
                t=msec_t if msec_t>last_t else fn/fps

                # Velocity-predicted position for this frame
                pred_cx=max(plate_r,min(wp-plate_r,cx+vx))
                pred_cy=max(plate_r,min(hp-plate_r,cy+vy))
                speed=(vx**2+vy**2)**0.5

                # Primary: Hough every frame, searched around the predicted position.
                # Search pad grows with speed but is capped at 3× plate radius so the
                # search window never covers most of the frame and attracts wrong circles.
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                search_pad=min(plate_r*3.0, plate_r*1.5+speed*2.5)
                found=hough_find(gray,int(plate_r*0.75),int(plate_r*1.25),
                                 pred_cx,pred_cy,search_pad)

                # Appearance gate: only applied when (a) the plate histogram is
                # discriminative (gate_enabled) and (b) Hough's result is suspiciously
                # far from the predicted position (> 30% of plate radius). Close results
                # are trusted unconditionally; far ones are verified against the plate
                # histogram to block wrong-circle grabs (lifter's body, equipment).
                if found and gate_enabled:
                    new_cx,new_cy,_=found
                    dist_from_pred=((new_cx-pred_cx)**2+(new_cy-pred_cy)**2)**0.5
                    if dist_from_pred>plate_r*0.3:
                        if plate_bp_score(frame,hist,new_cx,new_cy,plate_r)<min_score:
                            found=None

                # Pre-lift lock: while bar hasn't started moving, debounce large Hough
                # jumps (A→B→A oscillation between two rings). A jump exceeding
                # lock_jump_thresh must appear in the same location for lock_debounce
                # consecutive frames before being accepted as a real position change.
                # Travel exit: once the plate has moved >1.5× plate_r from the initial
                # detection, the bar is genuinely lifting — oscillation is impossible at
                # that distance, so the lock is released immediately.
                if found and lock_active:
                    nc_x,nc_y,_=found
                    jd=((nc_x-cx)**2+(nc_y-cy)**2)**0.5
                    travel=((nc_x-lock_origin_cx)**2+(nc_y-lock_origin_cy)**2)**0.5
                    if travel>plate_r*1.5:
                        lock_active=False       # bar has left the pre-lift zone
                    elif jd<=lock_jump_thresh:
                        lock_candidate=None; lock_streak=0   # small step, accepted as-is
                    elif (lock_candidate is not None and
                          ((nc_x-lock_candidate[0])**2+(nc_y-lock_candidate[1])**2)**0.5<lock_jump_thresh):
                        lock_streak+=1
                        if lock_streak>=lock_debounce:
                            lock_active=False   # confirmed sustained movement, release lock
                        else:
                            found=None          # not yet confirmed, suppress this frame
                    else:
                        lock_candidate=(nc_x,nc_y); lock_streak=1; found=None

                if found:
                    new_cx,new_cy,_=found
                    vel_hist.append((new_cx-cx,new_cy-cy))
                    if len(vel_hist)>3: vel_hist.pop(0)
                    vx=sum(v[0] for v in vel_hist)/len(vel_hist)
                    vy=sum(v[1] for v in vel_hist)/len(vel_hist)
                    cx,cy=new_cx,new_cy; bad_streak=0
                else:
                    # Fallback: CamShift anchored to predicted position.
                    # Use the same search region as Hough would have so it can reach a
                    # fast-moving plate that Hough missed. Velocity is NOT updated from
                    # CamShift to prevent drift compounding.
                    bad_streak+=1
                    cs_pad=int(search_pad)
                    tw_x=max(0,int(pred_cx-cs_pad)); tw_y=max(0,int(pred_cy-cs_pad))
                    tw_w=min(wp-tw_x,cs_pad*2);      tw_h=min(hp-tw_y,cs_pad*2)
                    pos,_=camshift_step(frame,hist,(tw_x,tw_y,tw_w,tw_h))
                    if pos:
                        if lock_active and ((pos[0]-cx)**2+(pos[1]-cy)**2)**0.5>lock_jump_thresh:
                            pos=None  # reject CamShift jump during pre-lift lock
                    if pos:
                        cx,cy=pos
                    else:
                        # Decay velocity slowly (0.97/frame ≈ −3%/frame) so the prediction
                        # stays near the plate during brief detection gaps.
                        vx*=0.97; vy*=0.97
                        cx,cy=pred_cx,pred_cy

                frame_data={"t":round(t,4),"x":round(cx/wp,5),"y":round(cy/hp,5)}
                results.append(frame_data); yield json.dumps({"frame":frame_data})+"\n"
                fn+=1; last_t=t
                if fn%30==0: await asyncio.sleep(0)

            cap.release()
            if results:
                pass  # smooth_coords removed: caused 12-17px amplitude compression and
                      # 1-2 frame temporal displacement at turnarounds, making the dot
                      # appear to reverse before the bar does. Hough jitter at rest is
                      # only ~7px (imperceptible), so smoothing is net harmful.
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
