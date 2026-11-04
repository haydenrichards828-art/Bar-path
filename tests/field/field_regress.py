"""Field regression for the bar path tracker, on real phone clips.

Runs every clip in a folder through the SAME path the ForceTrack app uses:
the phone shrink before upload (src/lib/video-shrink.ts: short edge 720, H.264
2.5 Mbps, all frames), then src/lib/bar-path/engine.ts prepareAnalysisUpload
and main.py /analyze:

    video : the clip re-encoded at most 900 px wide (never upscaled), 15 fps,
            H.264 around 900 kbps, upright, no audio
    seed  : the first 5 consecutive full resolution upright frames, stacked
            into one JPEG at quality 95
    params: only start_x and start_y, so step = 2 (7.5 analysed frames a second)

then checks the answer against expectations.json and writes, per clip, the raw
result JSON and an overlay strip (tap, seed circle, path, and the tracked point
on 7 frames) so a person can see every failure.

    python3 tests/field/field_regress.py <clips dir> [--expect tests/field/expectations.json]
            [--out <dir>] [--only 2983 3023] [--step 1] [--full]

--full skips the app emulation and analyses the original clip (60 fps, rotation
from the file), which is the reference the emulated run should agree with.

A clip passes only if every check below holds. Exit code is the number failing.

  verdict   expect "ok"     -> verdict ok or given
            expect "ask"    -> check or choose, OR ok when every other check passes
            expect "refuse" -> refuse, check or choose (never a silent ok)
  reps      exact count, when expectations give one (no missed, no ghost reps)
  jumps     no step between consecutive path points that the bar cannot make:
            up faster than 2.5 m/s, sideways faster than 1.2 m/s, or down
            faster than a 0.8 m free fall (the tracker leaving the plate)
  rom       every rep marked measured is inside the lift's plausible range
  mcv       mean concentric velocity of every measured rep is 0.05 to 1.6 m/s
  time      server analysis time is reported, not gated (see SPEED below)
"""
import argparse, glob, json, math, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root

import cv2, numpy as np
from bp.pipeline import analyse, PLATE_MM

ROM = {"dl": (0.40, 0.80), "sq": (0.40, 0.90), "bench": (0.20, 0.60)}


def rotation_of(path):
    """Clockwise degrees to turn the stored frame upright, as main.py should
    read it. ffmpeg < 5 exports the display matrix as a `rotate` tag with the
    OPPOSITE sign to the side data (tag 90 == side data -90), so read the side
    data only."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
                        "-of", "json", path], capture_output=True, text=True)
    try:
        s = json.loads(r.stdout)["streams"][0]
        for sd in s.get("side_data_list", []) or []:
            if "rotation" in sd:
                return int(float(sd["rotation"])) % 360
    except Exception:
        pass
    return 0


def shrink_720(src, work):
    """T-9300's phone shrink (src/lib/video-shrink.ts) before the set video is
    uploaded: short edge 720, H.264 at 2.5 Mbps, every frame kept. What is in
    R2, and therefore what the bar path analyser starts from, is this file."""
    stem = os.path.splitext(os.path.basename(src))[0]
    out = os.path.join(work, stem + ".r2.mp4")
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src,
                        "-vf", "scale=w='if(gt(iw,ih),-2,min(720,iw))':h='if(gt(iw,ih),min(720,ih),-2)'",
                        "-c:v", "libx264", "-preset", "veryfast", "-b:v", "2500k", "-pix_fmt", "yuv420p",
                        "-an", out], check=True)
    return out


def emulate_app(src, work):
    """prepareAnalysisUpload, in ffmpeg and OpenCV."""
    stem = os.path.splitext(os.path.basename(src))[0]
    small = os.path.join(work, stem + ".upload.mp4")
    seedp = os.path.join(work, stem + ".seed.jpg")
    if not (os.path.exists(small) and os.path.exists(seedp)):
        cap = cv2.VideoCapture(src)                 # OpenCV >= 4.5 rotates upright
        fr = []
        while len(fr) < 5:
            ok, f = cap.read()
            if not ok:
                break
            fr.append(f)
        cap.release()
        cv2.imwrite(seedp, np.vstack(fr), [cv2.IMWRITE_JPEG_QUALITY, 95])
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src,
                        "-vf", "scale=w=min(900\\,iw):h=-2,fps=15",
                        "-c:v", "libx264", "-preset", "veryfast", "-b:v", "900k",
                        "-maxrate", "1200k", "-bufsize", "1800k", "-pix_fmt", "yuv420p",
                        "-an", small], check=True)
    arr = cv2.imdecode(np.fromfile(seedp, np.uint8), cv2.IMREAD_GRAYSCALE)
    h = arr.shape[0] // 5
    return small, [arr[i * h:(i + 1) * h] for i in range(5)]


def concentric(fr, rep, H, ppm):
    """Mean and peak velocity (m/s) of the concentric, bar rising, leg of a rep,
    from frames_src. Image y grows downwards, so rising is y decreasing. The
    leg is the bottom of the rep to the highest point after it: for a squat or
    bench that is bottom then top, for a deadlift the floor then lockout."""
    seg = fr[rep["start"]:rep["end"] + 1]
    if len(seg) < 3 or not ppm:
        return None, None
    ys = np.array([p["y"] for p in seg]) * H / ppm      # metres, down positive
    ts = np.array([p["t"] for p in seg])
    b1 = int(np.argmax(ys)); t1 = b1 + int(np.argmin(ys[b1:]))       # bottom, then top after it
    t2 = int(np.argmin(ys)); b2 = int(np.argmax(ys[:t2 + 1]))        # top, and the bottom before it
    a, b = max([(b1, t1), (b2, t2)], key=lambda ab: ys[ab[0]] - ys[ab[1]])
    if b <= a or ts[b] <= ts[a]:
        return None, None
    # the leg starts when the bar leaves the bottom and ends when it reaches
    # the top: a rest on the floor or a pause at lockout is not concentric time
    bot, top = ys[a], ys[b]
    while a + 1 < b and ys[a + 1] >= bot - 0.01:
        a += 1
    while b - 1 > a and ys[b - 1] <= top + 0.01:
        b -= 1
    mcv = (ys[a] - ys[b]) / (ts[b] - ts[a])
    w = max(1, int(round(0.05 * (len(ts) - 1) / max(ts[-1] - ts[0], 1e-6))))
    vs = [(ys[i - w] - ys[i]) / max(ts[i] - ts[i - w], 1e-3) for i in range(a + w, b + 1)]
    return float(mcv), (float(max(vs)) if vs else None)


def render(clip, r, out_jpg, tap):
    """Eight frames across the clip with the tracked point drawn. Reads only
    the frames it draws: a full resolution 60 fps clip held in memory whole
    is several gigabytes."""
    cap = cv2.VideoCapture(clip); fps = cap.get(cv2.CAP_PROP_FPS) or 15
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, f0 = cap.read()
    if not ok:
        cap.release(); return
    H, W = f0.shape[:2]
    pts = r.get("frames_src") or []
    sr = r.get("seed") or {}
    rr = sr.get("r", 0) * (W / float(r.get("src_w") or W))
    def at(t):
        if not pts:
            return None
        b = min(pts, key=lambda p: abs(p["t"] - t))
        return b if abs(b["t"] - t) < 0.3 else None
    tiles = []
    f0 = f0.copy()
    cv2.drawMarker(f0, (int(tap[0] * W), int(tap[1] * H)), (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
    if sr:
        k = W / float(r.get("src_w") or W)
        cv2.circle(f0, (int(sr["cx"] * k), int(sr["cy"] * k)), int(rr), (255, 0, 255), 3)
    poly = np.array([[int(p["x"] * W), int(p["y"] * H)] for p in pts], np.int32)
    if len(poly) > 1:
        cv2.polylines(f0, [poly], False, (0, 255, 0), 3)
    tiles.append(f0)
    n = n or int((pts[-1]["t"] if pts else 1) * fps)
    for k in range(1, 8):
        i = min(max(0, n - 2), int(n * k / 8))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            continue
        f = f.copy()
        p = at(i / fps)
        if p:
            c = (int(p["x"] * W), int(p["y"] * H))
            cv2.circle(f, c, int(rr), (0, 0, 255), 3); cv2.circle(f, c, 8, (0, 0, 255), -1)
        else:
            cv2.putText(f, "no point", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.putText(f, f"{i / fps:.1f}s", (20, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        tiles.append(f)
    cap.release()
    tiles = [cv2.resize(t, (225, int(225 * H / W))) for t in tiles]
    cv2.imwrite(out_jpg, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])


def check(stem, exp, r):
    fails = []
    v = r.get("verdict")
    want = exp.get("expect", "ok")
    lift = exp.get("lift", "dl")
    fr = r.get("frames_src") or []
    reps = r.get("rep_metrics") or []
    H = r.get("src_h") or 1; W = r.get("src_w") or 1
    ppm = r.get("px_per_m") or 0
    # jumps: physically impossible steps between consecutive path points.
    # Upward faster than 2.5 m/s, sideways faster than 1.0 m/s, or downward
    # faster than a free fall from 0.8 m (4.0 m/s) is the tracker leaving the
    # plate, never the bar. A dropped deadlift can fall at 3 m/s, so downward
    # gets the free fall bound rather than the lifting one.
    worst = 0.0; jumped = False
    if ppm and len(fr) > 1:
        for a, b in zip(fr, fr[1:]):
            dt = b["t"] - a["t"]
            if dt <= 0:
                continue
            vx = abs(b["x"] - a["x"]) * W / ppm / dt
            vy = (b["y"] - a["y"]) * H / ppm / dt          # + is down
            worst = max(worst, math.hypot(vx, vy))
            # sideways 1.2 m/s, the tracker's own SIDE_MAX. Was 1.0: racking a
            # bench (2919, 2836, original files) peaks at 1.0 to 1.05 m/s over
            # one 1/15 s step, checked frame by frame with the plate centred.
            if vx > 1.2 or -vy > 2.5 or vy > 4.0:
                jumped = True
    # reps
    nrep = len(reps)
    if exp.get("reps") is not None and r.get("ok") and nrep != exp["reps"]:
        fails.append(f"reps {nrep} != {exp['reps']}")
    if exp.get("reps") is not None and not r.get("ok") and exp["reps"] > 0 and want == "ok":
        fails.append("refused a clip that should measure")
    if jumped:
        fails.append(f"path jumps at {worst:.1f} m/s")
    lo, hi = ROM.get(lift, (0.2, 0.9))
    mcvs = []
    for rp in reps:
        if rp.get("measured") and not (lo <= rp["rom_m"] <= hi):
            fails.append(f"rom {rp['rom_m']} outside {lo}-{hi}")
        mcv, pcv = concentric(fr, rp, H, ppm)
        # the pipeline's own mean concentric velocity when it gives one (v9),
        # else the harness's estimate from the path
        if rp.get("mean_concentric_ms") is not None:
            mcv, pcv = rp["mean_concentric_ms"], rp.get("peak_concentric_ms")
        mcvs.append((round(mcv, 3) if mcv else None, round(pcv, 3) if pcv else None))
        if rp.get("measured") and mcv is not None and not (0.05 <= mcv <= 1.6):
            fails.append(f"mcv {mcv:.2f} m/s implausible")
    # verdict
    bad = bool(fails)
    if want == "ok" and v not in ("ok", "given"):
        fails.append(f"verdict {v}, expected ok")
    elif want == "ask" and v in ("ok", "given") and bad:
        fails.append("silently wrong: verdict ok on a clip it got wrong")
    elif want == "refuse" and v in ("ok", "given"):
        fails.append("silently measured a clip it should refuse or ask about")
    if want == "ok" and v in ("ok", "given") and bad:
        fails.append("SILENTLY WRONG")
    return fails, {"verdict": v, "reps": nrep, "rom": [rp["rom_m"] for rp in reps],
                   "mcv_pcv": mcvs, "worst_step_ms": round(worst, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips")
    ap.add_argument("--expect", default=os.path.join(HERE, "expectations.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--full", action="store_true", help="analyse the original clip as is (reference)")
    ap.add_argument("--resume", action="store_true", help="reuse results already in --out (re-check them, do not re-run)")
    ap.add_argument("--budget", type=float, default=0, help="stop starting new clips after this many seconds")
    ap.add_argument("--no-shrink", action="store_true", help="skip the 720p phone shrink (older clips kept their original)")
    a = ap.parse_args()
    exps = json.load(open(a.expect))
    out = a.out or os.path.join(a.clips, "_field_results")
    work = os.path.join(out, "app_upload"); os.makedirs(work, exist_ok=True)
    nfail = 0; rows = []; T0 = time.time()
    for stem, exp in sorted(exps.items()):
        if stem.startswith("_"):
            continue
        if a.only and not any(o in stem for o in a.only):
            continue
        src = next(iter(glob.glob(os.path.join(a.clips, stem + ".*"))), None)
        if not src:
            print(f"{stem}: clip missing"); continue
        jp = os.path.join(out, stem + ".json")
        if a.resume and os.path.exists(jp):
            r = json.load(open(jp)); secs = r.get("_secs", 0)
            fails, info = check(stem, exp, r)
            nfail += bool(fails); rows.append((stem, info, secs, fails))
            print(f"{'FAIL' if fails else 'pass'} {stem[:44]:44s} {str(info['verdict']):7s} reps={info['reps']:<2d} "
                  f"rom={info['rom']} mcv/pcv={info['mcv_pcv']} worst={info['worst_step_ms']} {secs:.1f}s"
                  + (("  <- " + "; ".join(fails)) if fails else ""), flush=True)
            continue
        if a.budget and time.time() - T0 > a.budget:
            print(f"{stem}: budget used, run again with --resume"); continue
        t0 = time.time()
        if a.full:
            clip = src
            r = analyse(src, tap_norm=tuple(exp["tap"]), rot=rotation_of(src), plate_mm=exp.get("plate_mm", PLATE_MM))
        else:
            base = src if a.no_shrink else shrink_720(src, work)
            clip, seed = emulate_app(base, work)
            t1 = time.time()
            r = analyse(clip, tap_norm=tuple(exp["tap"]), rot=0, seed_images=seed, width=None,
                        step=a.step, plate_mm=exp.get("plate_mm", PLATE_MM))
            t0 = t1
        secs = time.time() - t0
        r.pop("path", None); r.pop("conf", None); r["_secs"] = round(secs, 2)
        json.dump(r, open(jp, "w"), default=str)
        try:
            render(clip, r, os.path.join(out, stem + ".jpg"), exp["tap"])
        except Exception as e:
            print(f"  render failed: {e}")
        fails, info = check(stem, exp, r)
        nfail += bool(fails)
        rows.append((stem, info, secs, fails))
        print(f"{'FAIL' if fails else 'pass'} {stem[:44]:44s} {str(info['verdict']):7s} reps={info['reps']:<2d} "
              f"rom={info['rom']} mcv/pcv={info['mcv_pcv']} worst={info['worst_step_ms']} {secs:.1f}s"
              + (("  <- " + "; ".join(fails)) if fails else ""), flush=True)
    print(f"\n{len(rows) - nfail} pass, {nfail} fail   median analysis {np.median([x[2] for x in rows]) if rows else 0:.1f}s")
    sys.exit(nfail)


if __name__ == "__main__":
    main()
