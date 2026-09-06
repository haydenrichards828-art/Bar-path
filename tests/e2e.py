"""The real client flow over HTTP: shrink the clip, tile eleven full resolution
stills into one JPEG, post both, compare against posting the whole thing."""
import sys, os, json, subprocess, time, urllib.request, cv2
sys.path.insert(0,'/home/claude/ftx')
from taps import TAPS
S="/mnt/user-data/uploads/Training Footage/testset"
URL="http://127.0.0.1:8123/analyze"

def post(video, params, seed=None):
    B=b"--X"; parts=[]
    def field(n,v): parts.append((f'Content-Disposition: form-data; name="{n}"\r\n\r\n'+v).encode())
    field("params", json.dumps(params)); field("api_key","testkey")
    parts.append(b'Content-Disposition: form-data; name="video"; filename="c.mp4"\r\n'
                 b'Content-Type: video/mp4\r\n\r\n'+open(video,"rb").read())
    if seed:
        parts.append(b'Content-Disposition: form-data; name="seed"; filename="s.jpg"\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n'+open(seed,"rb").read())
    body=b"".join(b"--X\r\n"+p+b"\r\n" for p in parts)+b"--X--\r\n"
    req=urllib.request.Request(URL,data=body,headers={"Content-Type":"multipart/form-data; boundary=X"})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=300) as r: raw=r.read().decode()
    return time.time()-t0, len(body), [json.loads(l) for l in raw.splitlines() if l.strip()]

def rom(lines):
    for m in lines:
        if m.get("done"):
            return [round(x["rom_m"]*100,1) for x in m["reps"] if x.get("measured",True)], m
        if "error" in m: return None, m
    return None, None

print(f"{'clip':24s} {'what was posted':>22s} {'MB':>6s} {'wall':>6s}  verdict   reps")
for stem in ("sq-side-mixed-colour-02","dl-mixed-diameters-03","sq-bright-gym-close-07","dl-side-clean-01"):
    full=f"{S}/{stem}.mp4"
    cap=cv2.VideoCapture(full); W=cap.get(3); H=cap.get(4)
    stills=[]
    for _ in range(11):
        ok,f=cap.read()
        if not ok: break
        stills.append(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY))
    cap.release()
    import numpy as np
    cv2.imwrite("/tmp/seed.jpg", np.vstack(stills), [cv2.IMWRITE_JPEG_QUALITY,80])
    subprocess.run(["ffmpeg","-y","-i",full,"-vf","scale=-2:900,fps=15","-c:v","libx264",
                    "-crf","26","-preset","veryfast","-an","/tmp/small.mp4"],
                   capture_output=True, check=True)
    p={"start_x":TAPS[stem][0]/W,"start_y":TAPS[stem][1]/H}
    for lab, vid, sd in (("the whole clip", full, None),
                         ("900w15 + seed strip", "/tmp/small.mp4", "/tmp/seed.jpg")):
        el, nb, lines = post(vid, p, sd)
        r, m = rom(lines)
        v = (m or {}).get("verdict","-")
        print(f"{stem:24s} {lab:>22s} {nb/1e6:6.2f} {el:5.1f}s  {v:>7s}   "
              + (", ".join(str(x) for x in r) if r else str((m or {}).get("error"))[:40]), flush=True)
    os.unlink("/tmp/small.mp4"); os.unlink("/tmp/seed.jpg")
