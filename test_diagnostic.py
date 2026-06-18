"""
Diagnostic test for bar-path tracker.
Usage: python test_diagnostic.py <video_path> <start_x> <start_y>
Saves annotated frames to ./diag_frames/ and a CSV of all coordinates.
No live window needed — works headlessly.
"""
import sys, os, csv, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from main import hough_find, build_hist, camshift_step

VIDEO  = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\hayde\OneDrive\Pictures\IMG_3108.MOV"
SX     = float(sys.argv[2]) if len(sys.argv) > 2 else 0.397
SY     = float(sys.argv[3]) if len(sys.argv) > 3 else 0.238
OUT    = os.path.join(os.path.dirname(__file__), "diag_frames")
os.makedirs(OUT, exist_ok=True)

# Save a frame with tracking annotation drawn on it
def save_frame(frame, cx, cy, plate_r, fn, label="", color=(0, 0, 255)):
    vis = frame.copy()
    # Downscale for manageable file sizes (save at 540px wide)
    scale = 540.0 / vis.shape[1]
    scx, scy, sr = int(cx * scale), int(cy * scale), max(2, int(plate_r * scale))
    vis = cv2.resize(vis, None, fx=scale, fy=scale)
    cv2.circle(vis, (scx, scy), sr, color, 2)
    cv2.circle(vis, (scx, scy), 4, (0, 255, 0), -1)
    cv2.putText(vis, f"f{fn} {label}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(vis, f"f{fn} {label}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 1)
    cv2.putText(vis, f"({cx/frame.shape[1]:.3f}, {cy/frame.shape[0]:.3f})", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
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

h, w = f0.shape[:2]
tx, ty = SX * w, SY * h
print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames ({total/fps:.1f}s)")
print(f"Tap pixel: ({tx:.0f}, {ty:.0f}) = normalized ({SX}, {SY})")

min_r = max(10, int(h * 0.05))
max_r = min(w // 2, int(h * 0.47))
print(f"Hough radius search: {min_r}–{max_r}px")

det = hough_find(cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY), min_r, max_r, tx, ty, int(min(w,h)*0.35))
if det is None:
    det = hough_find(cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY), min_r, max_r, w//2, h//2, min(w,h)//2)
if det is None:
    print("WARNING: Hough failed frame 0 — using tap point as fallback")
    det = (tx, ty, max(min_r * 2, int(h * 0.09)))

cx, cy, plate_r = det
plate_r = max(min_r, plate_r)
print(f"Frame 0 plate: center=({cx:.0f},{cy:.0f}) r={plate_r:.0f}px ({plate_r/h*100:.1f}% of height)")

hist_data = build_hist(f0, cx, cy, plate_r)
if hist_data is None:
    print("ERROR: Could not build histogram"); sys.exit(1)
hist, track_window = hist_data

# Check how many pixels survived the mask (circular + HSV)
roi_x0, roi_y0 = max(0, int(cx - plate_r)), max(0, int(cy - plate_r))
roi_x1, roi_y1 = min(w, int(cx + plate_r)), min(h, int(cy + plate_r))
roi = f0[roi_y0:roi_y1, roi_x0:roi_x1]
roi_h2, roi_w2 = roi.shape[:2]
Ygrid2, Xgrid2 = np.ogrid[:roi_h2, :roi_w2]
circ_mask2 = ((Xgrid2 - (cx - roi_x0))**2 + (Ygrid2 - (cy - roi_y0))**2 <= plate_r**2).astype(np.uint8) * 255
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
hsv_mask2 = cv2.inRange(hsv_roi, np.array((0., 40., 40.)), np.array((180., 255., 255.)))
combined = cv2.bitwise_and(circ_mask2, hsv_mask2)
circ_total = np.count_nonzero(circ_mask2)
valid_px = np.count_nonzero(combined)
valid_pct = 100.0 * valid_px / max(1, circ_total)
print(f"Circular mask pixels: {circ_total}, survived S>40,V>40 filter: {valid_px} ({valid_pct:.1f}%)")
if valid_pct < 5.0:
    print("  WARNING: <5% of plate disc pixels survived the mask — histogram may be unreliable")

save_frame(f0, cx, cy, plate_r, 0, "INIT")

# Tracking loop
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frames_since_hough = 0
HOUGH_EVERY = 6
bad_streak = 0
fn = 0
results = [{"fn": 0, "cx": cx, "cy": cy, "nx": cx/w, "ny": cy/h, "event": "init"}]

# For drift/freeze detection
prev_cx, prev_cy = cx, cy
freeze_count = 0
FREEZE_THRESH_PX = 3.0   # pixels — less than this = frozen
JUMP_THRESH_PX   = plate_r * 0.5  # half plate radius = suspicious jump

# Save every SNAP_EVERY frames plus anomaly frames
SNAP_EVERY = max(1, total // 30)  # ~30 snapshots across the video
saved_frames = []

while True:
    ret, frame = cap.read()
    if not ret: break
    fn += 1

    pos, track_window = camshift_step(frame, hist, track_window)
    if pos is not None:
        cx, cy = pos; bad_streak = 0
    else:
        bad_streak += 1

    frames_since_hough += 1
    hough_fired = False
    if frames_since_hough >= HOUGH_EVERY or bad_streak >= 2:
        frames_since_hough = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pad = plate_r * (1.8 if bad_streak < 2 else 5.0)
        rd = hough_find(gray, int(plate_r * 0.6), int(plate_r * 1.5), cx, cy, pad)
        if rd:
            ncx, ncy, rr = rd
            if 0.5 * plate_r < rr < 1.8 * plate_r:
                cx, cy = ncx, ncy; bad_streak = 0
                x0, y0 = max(0, int(cx - plate_r)), max(0, int(cy - plate_r))
                track_window = (x0, y0, int(plate_r * 2), int(plate_r * 2))
                new_hist = build_hist(frame, cx, cy, plate_r)
                if new_hist: hist, _ = new_hist
                hough_fired = True

    # Anomaly detection
    dist = np.hypot(cx - prev_cx, cy - prev_cy)
    event = ""
    anomaly_color = (0, 0, 255)  # red = normal

    if dist < FREEZE_THRESH_PX:
        freeze_count += 1
    else:
        freeze_count = 0

    if freeze_count == 10:
        event = "FREEZE"
        anomaly_color = (0, 165, 255)  # orange
        print(f"  [f{fn}] FREEZE detected — tracker stuck at ({cx:.0f},{cy:.0f}) for 10+ frames")
    elif dist > JUMP_THRESH_PX:
        event = f"JUMP{dist:.0f}px"
        anomaly_color = (0, 0, 255)  # red
        print(f"  [f{fn}] JUMP {dist:.0f}px  ({prev_cx:.0f},{prev_cy:.0f})->({cx:.0f},{cy:.0f})")

    if hough_fired and not event:
        event = "hough"
        anomaly_color = (255, 128, 0)  # blue

    results.append({"fn": fn, "cx": cx, "cy": cy, "nx": cx/w, "ny": cy/h, "event": event})

    # Save snapshot
    should_save = (fn % SNAP_EVERY == 0) or bool(event)
    if should_save:
        p = save_frame(frame, cx, cy, plate_r, fn, event or "snap", anomaly_color)
        saved_frames.append(p)

    prev_cx, prev_cy = cx, cy

cap.release()

# Write CSV
csv_path = os.path.join(os.path.dirname(__file__), "diag_coords.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fn", "cx", "cy", "nx", "ny", "event"])
    writer.writeheader()
    writer.writerows(results)

# Summary stats
nys = np.array([r["ny"] for r in results])
nxs = np.array([r["nx"] for r in results])
print(f"\n=== Tracking Summary ===")
print(f"Frames processed: {fn}")
print(f"Y range (normalized): {nys.min():.3f} – {nys.max():.3f}  (span={nys.max()-nys.min():.3f})")
print(f"X range (normalized): {nxs.min():.3f} – {nxs.max():.3f}  (span={nxs.max()-nxs.min():.3f})")
anomalies = [r for r in results if r["event"] and r["event"] not in ("init", "hough", "snap")]
print(f"Anomaly events (freeze/jump): {len(anomalies)}")
for a in anomalies[:20]:
    print(f"  f{a['fn']}: {a['event']} at ({a['nx']:.3f},{a['ny']:.3f})")
print(f"\nSaved {len(saved_frames)} annotated frames to {OUT}/")
print(f"Full coordinate log: {csv_path}")
