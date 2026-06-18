"""
Deep diagnostic for v6 tracker. Answers four specific questions:
1. Is velocity prediction actually applied before the Hough search each frame?
2. What is the prediction error (|pred - actual|) vs search_pad in fast frames?
3. Is search_pad wide enough to cover the fastest displacements?
4. In frames where Hough fails, is the plate inside or outside the search window?

Usage: python test_deep_diagnostic.py <video> <sx> <sy> [tag]
Saves annotated frames for the fastest segments with search window drawn.
"""
import sys, os, csv, math, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

# ---------- Inline reimplementation of hough_find that returns the p2 used ----------
import cv2 as _cv2
def hough_find_verbose(gray, min_r, max_r, cx_hint, cy_hint, pad):
    """Returns (cx, cy, r, p2_used, crop_shape) or None."""
    h, w = gray.shape
    sx = max(0, int(cx_hint - pad)); sy = max(0, int(cy_hint - pad))
    ex = min(w, int(cx_hint + pad)); ey = min(h, int(cy_hint + pad))
    if ex - sx < 10 or ey - sy < 10:
        return None
    crop_h, crop_w = ey - sy, ex - sx
    eq = _cv2.equalizeHist(gray[sy:ey, sx:ex])
    b = _cv2.GaussianBlur(eq, (9, 9), 2)
    for p2 in [30, 24, 18, 12, 8, 5]:
        c = _cv2.HoughCircles(b, _cv2.HOUGH_GRADIENT, 1.2, (ey - sy) // 2,
                              param1=80, param2=p2, minRadius=min_r, maxRadius=max_r)
        if c is not None:
            best = min(c[0], key=lambda v: (v[0] + sx - cx_hint)**2 + (v[1] + sy - cy_hint)**2)
            return float(best[0] + sx), float(best[1] + sy), float(best[2]), p2, (crop_h, crop_w)
    return None

from main import build_hist, camshift_step

VIDEO = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\hayde\OneDrive\Pictures\IMG_3108.MOV"
SX    = float(sys.argv[2]) if len(sys.argv) > 2 else 0.397
SY    = float(sys.argv[3]) if len(sys.argv) > 3 else 0.238
TAG   = sys.argv[4] if len(sys.argv) > 4 else "deep"
OUT   = os.path.join(os.path.dirname(__file__), "diag_frames", TAG)
os.makedirs(OUT, exist_ok=True)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"ERROR: Cannot open {VIDEO}"); sys.exit(1)

fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, f0 = cap.read()
if not ret:
    print("ERROR: Cannot read first frame"); sys.exit(1)

h, w = f0.shape[:2]
tx, ty = SX * w, SY * h
print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames ({total/fps:.1f}s)")

short_side = min(w, h)
min_r = max(10, int(short_side * 0.05))
max_r = int(short_side * 0.30)
g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
det = hough_find_verbose(g0, min_r, max_r, tx, ty, int(short_side * 0.35))
if det is None:
    det_raw = hough_find_verbose(g0, min_r, max_r, w//2, h//2, short_side//2)
    det = det_raw
if det is None:
    print("Hough failed frame 0 — using tap fallback")
    cx, cy, plate_r = tx, ty, max(min_r * 2, int(h * 0.09))
else:
    cx, cy, plate_r, p2_f0, crop_f0 = det
    plate_r = max(min_r, plate_r)
    print(f"Frame 0 plate: ({cx:.0f},{cy:.0f}) r={plate_r:.0f}  p2={p2_f0}  crop={crop_f0}")

hist_data = build_hist(f0, cx, cy, plate_r)
if hist_data is None:
    print("ERROR: Could not build histogram"); sys.exit(1)
hist, _ = hist_data

# ---------- Tracking loop — exact mirror of main.py v6, with full logging ----------
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
vx, vy = 0.0, 0.0
vel_hist = []
fn = 0

log = []  # one dict per frame

while True:
    ret, frame = cap.read()
    if not ret: break
    fn += 1

    # Snapshot state BEFORE this frame's measurement
    pre_cx, pre_cy = cx, cy
    pre_vx, pre_vy = vx, vy

    pred_cx = max(plate_r, min(w - plate_r, cx + vx))
    pred_cy = max(plate_r, min(h - plate_r, cy + vy))
    speed   = math.hypot(vx, vy)
    search_pad = plate_r * 1.5 + speed * 2.5

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = hough_find_verbose(gray, int(plate_r * 0.75), int(plate_r * 1.25),
                                pred_cx, pred_cy, search_pad)

    if result is not None:
        new_cx, new_cy, _, p2_used, crop_shape = result
        pred_error = math.hypot(new_cx - pred_cx, new_cy - pred_cy)
        actual_disp = math.hypot(new_cx - pre_cx, new_cy - pre_cy)
        vel_hist.append((new_cx - pre_cx, new_cy - pre_cy))
        if len(vel_hist) > 3: vel_hist.pop(0)
        vx = sum(v[0] for v in vel_hist) / len(vel_hist)
        vy = sum(v[1] for v in vel_hist) / len(vel_hist)
        cx, cy = new_cx, new_cy
        source = "hough"
        p2_log = p2_used
    else:
        pred_error = None
        actual_disp = None
        p2_log = None
        tw = (max(0, int(pred_cx - plate_r)), max(0, int(pred_cy - plate_r)),
              int(plate_r * 2), int(plate_r * 2))
        pos, _ = camshift_step(frame, hist, tw)
        if pos:
            cx, cy = pos; source = "camshift"
        else:
            vx *= 0.85; vy *= 0.85
            cx, cy = pred_cx, pred_cy; source = "predict"

    log.append({
        "fn": fn,
        "pre_cx": round(pre_cx, 1), "pre_cy": round(pre_cy, 1),
        "pre_vx": round(pre_vx, 2), "pre_vy": round(pre_vy, 2),
        "pred_cx": round(pred_cx, 1), "pred_cy": round(pred_cy, 1),
        "search_pad": round(search_pad, 1),
        "cx": round(cx, 1), "cy": round(cy, 1),
        "actual_disp": round(actual_disp, 1) if actual_disp is not None else None,
        "pred_error": round(pred_error, 1) if pred_error is not None else None,
        "search_margin": round(search_pad - pred_error, 1) if pred_error is not None else None,
        "source": source,
        "p2": p2_log,
        "nx": round(cx / w, 5), "ny": round(cy / h, 5),
    })

cap.release()

# ---------- Write CSV ----------
csv_path = os.path.join(os.path.dirname(__file__), f"deep_log_{TAG}.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=log[0].keys())
    writer.writeheader()
    writer.writerows(log)
print(f"\nWrote {len(log)} rows to {csv_path}")

# ---------- Analysis ----------
hough_rows   = [r for r in log if r["source"] == "hough"]
fail_rows    = [r for r in log if r["source"] != "hough"]
actual_disps = [r["actual_disp"] for r in hough_rows if r["actual_disp"] is not None]
pred_errors  = [r["pred_error"] for r in hough_rows if r["pred_error"] is not None]
search_pads  = [r["search_pad"] for r in log]
margins      = [r["search_margin"] for r in hough_rows if r["search_margin"] is not None]
p2_counts    = {}
for r in hough_rows:
    p2_counts[r["p2"]] = p2_counts.get(r["p2"], 0) + 1

print(f"\n=== Summary ===")
print(f"Total frames: {len(log)}  |  Hough: {len(hough_rows)}  |  Fallback: {len(fail_rows)}")
print(f"\nActual frame-to-frame displacement (px, Hough frames only):")
print(f"  mean={np.mean(actual_disps):.1f}  max={np.max(actual_disps):.1f}  p95={np.percentile(actual_disps,95):.1f}  p99={np.percentile(actual_disps,99):.1f}")
print(f"\nPrediction error (|pred_pos - actual_hough_pos|, px):")
print(f"  mean={np.mean(pred_errors):.1f}  max={np.max(pred_errors):.1f}  p95={np.percentile(pred_errors,95):.1f}  p99={np.percentile(pred_errors,99):.1f}")
print(f"\nSearch pad (distance from predicted pos to edge of search window, px):")
print(f"  mean={np.mean(search_pads):.1f}  min={np.min(search_pads):.1f}  max={np.max(search_pads):.1f}")
print(f"\nMargin = search_pad - pred_error (must be >0 for plate to be inside search):")
print(f"  mean={np.mean(margins):.1f}  min={np.min(margins):.1f}  (negative = plate outside search window)")
print(f"\nHough p2 sensitivity distribution (lower=more permissive):")
for p2 in sorted(p2_counts):
    pct = 100 * p2_counts[p2] / len(hough_rows)
    print(f"  p2={p2:2d}: {p2_counts[p2]:4d} frames ({pct:.1f}%)")

print(f"\nFallback frames: {len(fail_rows)}")
for r in fail_rows[:20]:
    print(f"  f{r['fn']:4d}: {r['source']:8s}  pred=({r['pred_cx']:.0f},{r['pred_cy']:.0f})  "
          f"landed=({r['cx']:.0f},{r['cy']:.0f})  vx={r['pre_vx']:.1f} vy={r['pre_vy']:.1f}")

# ---------- Top-30 fastest frames + their prediction errors ----------
print(f"\nTop 30 fastest frames (by actual_disp) — key diagnostic:")
sorted_fast = sorted(hough_rows, key=lambda r: r["actual_disp"] or 0, reverse=True)[:30]
print(f"  {'fn':>5}  {'disp':>6}  {'pred_err':>8}  {'pad':>6}  {'margin':>7}  {'p2':>3}  source")
for r in sorted_fast:
    print(f"  {r['fn']:5d}  {r['actual_disp']:6.1f}  {r['pred_error']:8.1f}  "
          f"{r['search_pad']:6.1f}  {r['search_margin']:7.1f}  {r['p2']:3d}  {r['source']}")

# ---------- Save annotated frames for fastest 20 + all fallback frames ----------
# We need to re-run tracking and capture the frames at specific fn values
targets_fast = {r["fn"] for r in sorted_fast[:20]}
targets_fail = {r["fn"] for r in fail_rows}
targets_all  = targets_fast | targets_fail
log_by_fn    = {r["fn"]: r for r in log}

if targets_all:
    print(f"\nSaving {len(targets_all)} annotated frames (fastest + fallback)...")
    cap2 = cv2.VideoCapture(VIDEO)
    fn2 = 0
    while True:
        ret2, frame2 = cap2.read()
        if not ret2: break
        fn2 += 1
        if fn2 not in targets_all: continue

        r = log_by_fn[fn2]
        scale = 540.0 / frame2.shape[1]
        vis = cv2.resize(frame2, None, fx=scale, fy=scale)

        # Draw search window (search_pad radius around pred, green dashed circle)
        s_pad_sc = int(r["search_pad"] * scale)
        s_cx_sc  = int(r["pred_cx"] * scale)
        s_cy_sc  = int(r["pred_cy"] * scale)
        cv2.circle(vis, (s_cx_sc, s_cy_sc), s_pad_sc, (0, 200, 0), 1)  # search window
        cv2.circle(vis, (s_cx_sc, s_cy_sc), 4, (0, 200, 0), -1)          # predicted pos

        # Draw tracked position
        t_cx_sc = int(r["cx"] * scale)
        t_cy_sc = int(r["cy"] * scale)
        plate_r_sc = max(2, int(plate_r * scale))
        color = (0, 0, 255) if r["source"] == "hough" else (0, 165, 255)
        cv2.circle(vis, (t_cx_sc, t_cy_sc), plate_r_sc, color, 2)
        cv2.circle(vis, (t_cx_sc, t_cy_sc), 4, (0, 255, 0), -1)

        # Text
        disp_str = f"disp={r['actual_disp']}px" if r['actual_disp'] is not None else "FALLBACK"
        err_str  = f"perr={r['pred_error']}px" if r['pred_error'] is not None else f"src={r['source']}"
        pad_str  = f"pad={r['search_pad']:.0f}px"
        cv2.putText(vis, f"f{fn2} {disp_str}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(vis, f"f{fn2} {disp_str}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(vis, err_str, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
        cv2.putText(vis, pad_str, (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

        fname = f"{'FAIL' if r['source'] != 'hough' else 'fast'}_f{fn2}.jpg"
        cv2.imwrite(os.path.join(OUT, fname), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cap2.release()
    print(f"Saved to {OUT}/")

print(f"\nKEY DIAGNOSTIC QUESTION:")
print(f"  Max actual displacement: {np.max(actual_disps):.1f}px/frame")
print(f"  Max prediction error at those frames: {np.max(pred_errors):.1f}px")
min_pad = np.min(search_pads)
print(f"  Min search_pad used: {min_pad:.1f}px  (= plate_r*1.5 = {plate_r*1.5:.0f} at rest)")
print(f"  Margin (search_pad - pred_error) was NEGATIVE on {sum(1 for m in margins if m < 0)} frames")
print(f"  => Plate was OUTSIDE search window on those frames (search bottleneck confirmed)")
print(f"  => Plate was INSIDE search window on all frames (detection failure, not radius)")
