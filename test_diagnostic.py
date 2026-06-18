"""
Diagnostic test for bar-path tracker v6 (Hough-primary, velocity-predicted).
Usage: python test_diagnostic.py <video_path> <start_x> <start_y> [out_tag]
Saves annotated frames to ./diag_frames/<out_tag>/ and a CSV of all coords.
"""
import sys, os, csv, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from main import hough_find, build_hist, camshift_step, plate_bp_score

VIDEO   = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\hayde\OneDrive\Pictures\IMG_3108.MOV"
SX      = float(sys.argv[2]) if len(sys.argv) > 2 else 0.397
SY      = float(sys.argv[3]) if len(sys.argv) > 3 else 0.238
TAG     = sys.argv[4] if len(sys.argv) > 4 else "run"
OUT     = os.path.join(os.path.dirname(__file__), "diag_frames", TAG)
os.makedirs(OUT, exist_ok=True)

def save_frame(frame, cx, cy, plate_r, fn, label="", color=(0,0,255)):
    scale = 540.0 / frame.shape[1]
    vis = cv2.resize(frame, None, fx=scale, fy=scale)
    scx,scy,sr = int(cx*scale), int(cy*scale), max(2,int(plate_r*scale))
    cv2.circle(vis, (scx,scy), sr, color, 2)
    cv2.circle(vis, (scx,scy), 4, (0,255,0), -1)
    cv2.putText(vis, f"f{fn} {label}", (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(vis, f"f{fn} {label}", (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1)
    cv2.putText(vis, f"({cx/frame.shape[1]:.3f},{cy/frame.shape[0]:.3f})", (8,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
    path = os.path.join(OUT, f"frame_{fn:05d}_{label}.jpg")
    cv2.imwrite(path, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"ERROR: Cannot open {VIDEO}"); sys.exit(1)

fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, f0 = cap.read()
if not ret:
    print("ERROR: Cannot read first frame"); sys.exit(1)

h,w = f0.shape[:2]
tx,ty = SX*w, SY*h
print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames ({total/fps:.1f}s)")
print(f"Tap: ({tx:.0f},{ty:.0f})")

short_side = min(w,h)
min_r = max(10, int(short_side*0.05))
max_r = int(short_side*0.30)
g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
det = hough_find(g0, min_r, max_r, tx, ty, int(short_side*0.35))
if det is None:
    det = hough_find(g0, min_r, max_r, w//2, h//2, short_side//2)
if det is None:
    print("WARNING: Hough failed frame 0 — using tap as fallback")
    det = (tx, ty, max(min_r*2, int(h*0.09)))

cx,cy,plate_r = det
plate_r = max(min_r, plate_r)
print(f"Frame 0 plate: ({cx:.0f},{cy:.0f}) r={plate_r:.0f}px")

hist_data = build_hist(f0, cx, cy, plate_r)
if hist_data is None:
    print("ERROR: Could not build histogram"); sys.exit(1)
hist,_ = hist_data
ref_score = plate_bp_score(f0, hist, cx, cy, plate_r)
min_score = max(12.0, ref_score * 0.15)
gate_enabled = ref_score >= 20.0
print(f"Appearance ref_score={ref_score:.1f}  min_score={min_score:.1f}  gate_enabled={gate_enabled}")
save_frame(f0, cx, cy, plate_r, 0, "INIT")

# --- Tracking loop (mirrors main.py v7 exactly) ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
vx,vy = 0.0, 0.0
vel_hist = []
bad_streak = 0
fn = 0
results = [{"fn":0,"cx":cx,"cy":cy,"nx":cx/w,"ny":cy/h,"speed":0.0,"source":"init"}]

SNAP_EVERY = max(1, total // 40)  # ~40 snapshots across the video

prev_cx,prev_cy = cx,cy

while True:
    ret,frame = cap.read()
    if not ret: break
    fn += 1

    pred_cx = max(plate_r, min(w-plate_r, cx+vx))
    pred_cy = max(plate_r, min(h-plate_r, cy+vy))
    speed = (vx**2+vy**2)**0.5
    search_pad = min(plate_r*3.0, plate_r*1.5 + speed*2.5)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = hough_find(gray, int(plate_r*0.75), int(plate_r*1.25),
                       pred_cx, pred_cy, search_pad)

    # Appearance gate: only when histogram is discriminative (gate_enabled) and
    # Hough result is far from prediction
    if found and gate_enabled:
        fcx, fcy, _ = found
        dist_from_pred = ((fcx-pred_cx)**2+(fcy-pred_cy)**2)**0.5
        if dist_from_pred > plate_r * 0.3:
            if plate_bp_score(frame, hist, fcx, fcy, plate_r) < min_score:
                found = None

    source = ""
    color = (0,0,255)
    if found:
        new_cx,new_cy,_ = found
        vel_hist.append((new_cx-cx, new_cy-cy))
        if len(vel_hist)>3: vel_hist.pop(0)
        vx = sum(v[0] for v in vel_hist)/len(vel_hist)
        vy = sum(v[1] for v in vel_hist)/len(vel_hist)
        cx,cy = new_cx,new_cy; bad_streak=0
        source="hough"
        color=(255,128,0)
    else:
        bad_streak += 1
        cs_pad = int(search_pad)
        tw_x = max(0, int(pred_cx-cs_pad)); tw_y = max(0, int(pred_cy-cs_pad))
        tw_w = min(w-tw_x, cs_pad*2);       tw_h = min(h-tw_y, cs_pad*2)
        pos,_ = camshift_step(frame, hist, (tw_x, tw_y, tw_w, tw_h))
        if pos:
            cx,cy = pos; source="camshift"
            color=(0,165,255)
        else:
            vx*=0.97; vy*=0.97
            cx,cy=pred_cx,pred_cy; source="predict"
            color=(0,0,200)

    frame_speed = ((cx-prev_cx)**2+(cy-prev_cy)**2)**0.5
    results.append({"fn":fn,"cx":cx,"cy":cy,"nx":cx/w,"ny":cy/h,
                    "speed":round(frame_speed,1),"source":source})
    prev_cx,prev_cy = cx,cy

    # Save snaps: every SNAP_EVERY frames + any frame where source != hough
    should_save = (fn % SNAP_EVERY == 0) or (source != "hough")
    if should_save:
        save_frame(frame, cx, cy, plate_r, fn, source or "snap", color)

cap.release()

# Write CSV
csv_path = os.path.join(os.path.dirname(__file__), f"diag_coords_{TAG}.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fn","cx","cy","nx","ny","speed","source"])
    writer.writeheader()
    writer.writerows(results)

# Summary
nys = np.array([r["ny"] for r in results])
speeds = np.array([r["speed"] for r in results])
sources = [r["source"] for r in results]
hough_pct = 100*sources.count("hough")/max(1,len(sources))
fallback_pct = 100*(sources.count("camshift")+sources.count("predict"))/max(1,len(sources))

print(f"\n=== Tracking Summary ({TAG}) ===")
print(f"Frames: {fn}")
print(f"Y range: {nys.min():.3f} - {nys.max():.3f}  span={nys.max()-nys.min():.3f}")
print(f"Source breakdown: hough={hough_pct:.1f}%  fallback={fallback_pct:.1f}%")
print(f"Speed: mean={speeds.mean():.1f}px/frame  max={speeds.max():.1f}px/frame")

# Fastest frames (top 20 by per-frame displacement)
fast_frames = sorted(results, key=lambda r: r["speed"], reverse=True)[:20]
print(f"\nTop 20 fastest frames (where tracking is hardest):")
for r in fast_frames:
    print(f"  f{r['fn']:4d}: speed={r['speed']:5.1f}px/frame  source={r['source']:8s}  y={r['ny']:.3f}")

non_hough = [r for r in results if r["source"] not in ("hough","init")]
print(f"\nFallback frames (not Hough): {len(non_hough)}")
for r in non_hough[:30]:
    print(f"  f{r['fn']}: {r['source']} at y={r['ny']:.3f} speed={r['speed']:.1f}px")

print(f"\nSaved frames to {OUT}/")
print(f"CSV: {csv_path}")
