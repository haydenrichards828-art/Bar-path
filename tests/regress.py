"""Post all 20 clips at the HTTP API and check the service behaves."""
import sys, os, json, time, urllib.request, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taps import TAPS
from truth import TRUE_R, MUST_REFUSE
SRC = os.environ.get("TESTSET", "/mnt/user-data/uploads/Training Footage/testset")
URL="http://127.0.0.1:8123/analyze"

def post(stem):
    path=f"{SRC}/{stem}.mp4"
    import cv2
    cap=cv2.VideoCapture(path); W=cap.get(3); H=cap.get(4); cap.release()
    t=TAPS[stem]
    params=json.dumps({"start_x":t[0]/W,"start_y":t[1]/H})
    b=b"--X\r\n"
    parts=[]
    def field(n,v): parts.append((f'Content-Disposition: form-data; name="{n}"\r\n\r\n'+v).encode())
    field("params",params); field("api_key",os.environ.get("K","testkey"))
    with open(path,"rb") as f: vid=f.read()
    parts.append(b'Content-Disposition: form-data; name="video"; filename="c.mp4"\r\nContent-Type: video/mp4\r\n\r\n'+vid)
    body=b"".join(b"--X\r\n"+p+b"\r\n" for p in parts)+b"--X--\r\n"
    req=urllib.request.Request(URL,data=body,headers={"Content-Type":"multipart/form-data; boundary=X"})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=300) as r:
        raw=r.read().decode()
    el=time.time()-t0
    lines=[json.loads(l) for l in raw.splitlines() if l.strip()]
    return el, lines

ok=fail=0
print(f"{'clip':26s} {'wall':>6s} {'srv':>6s} {'verdict':>8s} {'r':>5s}/{'true':>4s} {'trk':>9s} {'reps':>4s}  outcome")
for stem in sorted(TAPS):
    try:
        el,lines=post(stem)
    except Exception as e:
        print(f"{stem:26s} EXC {type(e).__name__}: {e}"); fail+=1; continue
    fin=[l for l in lines if l.get("done") or "error" in l]
    if not fin: print(f"{stem:26s} no terminal message"); fail+=1; continue
    m=fin[-1]
    must = stem in MUST_REFUSE
    if "error" in m:
        good = must
        print(f"{stem:26s} {el:6.2f} {'':>6s} {m.get('verdict','-'):>8s} {'':>10s} {'':>9s} {'':>4s}  "
              + ("correctly refused" if good else "WRONGLY REFUSED"))
        ok+=good; fail+= not good; continue
    t=TRUE_R.get(stem)
    r=m["seed"]["r"]; err=abs(r-t)/t*100 if t else None
    good = (not must) and (err is None or err<=12 or m["verdict"] in ("check","choose"))
    print(f"{stem:26s} {el:6.2f} {m['server_s']:6.2f} {m['verdict']:>8s} {r:5.0f}/{t or 0:4d} "
          f"{m['tracked']:4d}/{m['analysed']:<4d} {len(m['reps']):4d}  "
          + ("SHOULD HAVE REFUSED" if must else (f"r off {err:.0f}%" if err is not None else "")))
    ok+=good; fail+= not good
print(f"\n{ok} as expected, {fail} not")
