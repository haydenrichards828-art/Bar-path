"""ForceTrack bar path: the whole analysis, end to end.

v10 (Nov 2026): the clip itself now says which circle is the plate. Line the
frames up on the tracked point and the plate face stands still while the gym
behind it moves, so the plate's rim is the outermost circle that is both an
edge and the boundary of what moves with the bar (dp.motion_profile). A hub is
never that boundary, nor is a circle in the background. "ok" now needs that
confirmation. When the first track's circle turns out to be the wrong one
(a hub, a circle reaching into the rack, a circle that never moved), analyse()
tracks again from a better one, reusing the decoded frames, and keeps the pass
the clip confirms. A tap the two plate finders would refuse gets a third
finder (ringfind: whole circles containing the tap, partly out of shot
allowed) before it is refused. Template correlation is computed for every
candidate (it was read off a map filled only around its own peaks, so circle
candidates scored 0 and 3106 lost the bottom of a squat), and a gap of 0.8 s
mid set is a doubt, because a rep fits in it.

v9 (Nov 2026): the tracker is bp/dp.py (every frame's candidates scored for
plate evidence, one physically possible path for the whole clip), the seed
circle is checked against the whole path (bp/dp.rim_check), reps are counted
without bumper bounces and measured from their first leg, each rep reports
mean and peak CONCENTRIC velocity, and the track itself can turn an "ok"
from the plate finder into a "check" (track_doubts), or confirm an unsure
seed (confirmed_by_track). Measured on 47 of Hayden's clips through the app's
real upload chain: tests/field.

Latency
-------
Decoding is the floor. On this box H.264 1080p decodes at about 280 fps and
nothing in the analysis is slower than that, so the only way to hide the
analysis is to run it WHILE the decoder is still working. One thread pulls
frames; the caller's thread seeds, then tracks forward frame by frame as they
arrive. The reverse pass is the only part that must wait for the end of the
clip, and it runs over frames already resized and held in memory.

Resolution
----------
Two different jobs want two different frames.

  * The plate finder wants FULL resolution and CONSECUTIVE frames. It fits a
    rim gradient, so a 225 px plate shrunk to 63 px has nothing left to fit;
    and it takes a median across five frames a couple of frames apart on the
    assumption that the bar has barely moved between them. Feed it stepped
    frames and the median is taken across half a second of travel, which is
    how a working clip turns into a refusal.
  * The tracker wants SMALL and STEPPED. It downsamples to a 56 px template
    anyway, and every 8th frame at 59 fps is still 7.4 Hz — plenty for a
    barbell.

So the decoder emits both: eleven consecutive full-resolution frames at the
head of the clip for the seed, and every 8th frame at 540 px wide for the
track. The extra cost is eleven colour conversions.
"""
import math, queue, threading, time
import cv2, numpy as np

import v7
import dp as _dp
import snap as _snap
import ellipse as _el
import ringfind as _rf
from plate import find_plate

SEED_N      = 11       # consecutive full-res frames the plate finder gets
SCALE_EVERY = 6        # measure the plate on every Nth analysed frame
STEP        = 8        # analyse every Nth source frame
WIDTH       = 540      # tracking width in px
PLATE_R_M   = v7.PLATE_R_M
PLATE_MM    = 450.0    # competition bumpers are 450 mm at every weight, and so
                       # are 20 and 25 kg iron plates, which is what a working
                       # set is loaded with. It is wrong for a light bar loaded
                       # with small iron, so it is a parameter, not a constant:
                       # every distance scales linearly with it, which means the
                       # app can let the coach correct it afterwards and every
                       # number updates without re-analysing anything.
MIN_REP_F   = 8
MIN_REP_ROM = 0.03
MIN_REP_M   = 0.10     # under 10 cm of bar travel is not a rep
SANE_MAX_MS = 2.0      # above this the tracker is wrong, not the lifter fast
SANE_MAX_M  = 1.00     # and no squat, bench or deadlift moves the bar a metre
VEL_WINDOW_S= 0.05     # velocity is measured over this window, not frame to frame
QUAD        = True     # parabolic resample (False = the old straight line)
TRACKER     = "dp"     # "dp" (v9, whole clip path with plate evidence) or "v8"
WORK_MAX    = 480      # px, longest side of a working frame
N_WORKERS   = max(2, min(6, (__import__("os").cpu_count() or 2)))
HUB_ASK     = 15.0     # seed interior this much brighter than its surround: a hub
MCV_MAX     = 1.3      # m/s mean concentric; faster is the tracker, not the lifter
MR_LO, MR_HI = 0.88, 1.15  # a motion rim this close to the seed's size IS the seed's ring
RF_HIT_REFUSED = 0.65  # the same, when the alternative is refusing the clip
RF_HIT      = 0.75     # ringfind: fraction of in-frame rays on the circle to offer it
MAX_LOOKS   = 2        # extra tracking passes from a better circle, at most
GAP_ASK     = 0.8      # s; the bar lost this long mid set could hide a rep


# ---------------------------------------------------------------- decoding
# T-9378: a clip the server fetches itself is the phone's 720p file at its own
# frame rate. The app chain analyses 7.5 frames a second (15 fps, step 2), and
# the tracker's limits are per analysed frame, so this keeps that rate.
ANALYSED_HZ = 7.5


def direct_step(fps):
    """Analyse every Nth frame of a clip at this frame rate: 30 fps 4, 60 fps 8."""
    return max(1, int(round((fps or 30.0) / ANALYSED_HZ)))


def _rot(frame, rot):
    if rot == 90:  return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rot == 180: return cv2.rotate(frame, cv2.ROTATE_180)
    if rot == 270: return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def _decode_into(path, q, step, width, stop, seed_n, rot=0, scale_every=SCALE_EVERY):
    cap = cv2.VideoCapture(path)
    try:
        # rotate_frame is the single source of truth for orientation, so switch
        # off OpenCV's own metadata rotation or iOS clips get turned twice.
        try: cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        except Exception: pass
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        q.put(("meta", fps, total))
        n = 0; kept = 0; sc = None
        while not stop.is_set():
            if not cap.grab():
                break
            want_seed = n < seed_n
            want_trk = (n % step == 0)
            if want_seed or want_trk:
                ok, f = cap.retrieve()
                if not ok:
                    break
                if rot:
                    f = _rot(f, rot)
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                if sc is None:
                    sc = (width / g.shape[1]) if width else 1.0
                if want_seed:
                    q.put(("s", g))
                if want_trk:
                    small = cv2.resize(g, None, fx=sc, fy=sc,
                                       interpolation=cv2.INTER_AREA) if sc != 1.0 else g
                    # every Nth analysed frame also travels at full resolution,
                    # so the plate can be re-measured as the bar changes height.
                    full = g if (kept % scale_every == 0) else None
                    q.put(("f", kept, small, sc, full)); kept += 1
            n += 1
    except Exception as e:                       # truncated file, missing moov
        q.put(("err", repr(e)))
    finally:
        cap.release()
        q.put(("end", n))


# ---------------------------------------------------------------- reps
def turning_points(y, prom):
    """Indices where the trace turns round, ignoring wobbles smaller than
    `prom`. A zigzag filter: hold the running high and low since the last
    pivot, and commit a pivot only once the trace has retraced `prom` from it.

    This replaces counting runs of consistent velocity, which had three faults
    on real footage. It never closed the last run, so a clip that ended on the
    way down lost its final rep outright. A bar resting on the floor produced
    'up' and 'down' runs of zero amplitude out of sub-pixel jitter, which then
    consumed the pairing. And it could only pair adjacent runs, so one spurious
    run put every rep after it out of phase."""
    n = len(y)
    if n < 3:
        return []
    out = []
    hi = lo = 0
    d = 0                       # +1 rising (next pivot is a high), -1 falling
    for i in range(1, n):
        if y[i] > y[hi]: hi = i
        if y[i] < y[lo]: lo = i
        if d != 1 and y[i] - y[lo] >= prom:
            out.append(lo); d = 1; hi = lo = i
        elif d != -1 and y[hi] - y[i] >= prom:
            out.append(hi); d = -1; hi = lo = i
    out.append(hi if d == 1 else lo)
    return [p for k, p in enumerate(out) if k == 0 or p != out[k - 1]]


def detect_reps(ys, ts, min_rom=MIN_REP_M, min_s=0.25):
    """A rep is one full excursion: out to the far end of the range and back.

    Works the same for a squat (stand, bottom, stand) and a deadlift (floor,
    lockout, floor) because it counts turning points rather than assuming which
    way the lift starts. Also survives a clip that stops before the bar gets
    home: the range is taken across the whole excursion, so an unfinished
    return leg costs nothing."""
    if len(ys) < 6:
        return []
    span = max(ys) - min(ys)
    if span <= 0:
        return []
    prom = max(min_rom, span * 0.25)
    tp = turning_points(ys, prom)
    reps = []
    k = 0
    while k + 2 < len(tp):
        a, b, c = tp[k], tp[k + 1], tp[k + 2]
        seg = ys[a:c + 1]
        if (max(seg) - min(seg)) >= min_rom and (ts[c] - ts[a]) >= min_s:
            reps.append({"start": a, "end": c, "peak": b})
            k += 2
        else:
            k += 1
    return reps


def _quad_interp(ai, vals, src_i, step):
    """Resample onto every source frame with a LOCAL PARABOLA, not a straight
    line between neighbours.

    A barbell turns round under roughly constant acceleration, so between two
    analysed frames its path is a parabola. Joining the samples with straight
    lines therefore cuts the corner at the top and bottom of every rep, and the
    top and bottom are exactly what range of motion is measured between. The
    miss is a*dt^2/4 at worst -- for a squat reversing at about 5 m/s^2 sampled
    at 7.4 Hz that is 2 cm per turnaround, so up to 4 cm off a rep, always in
    the direction of reading SHORT.

    A parabola through the three nearest samples is exact for constant
    acceleration and costs nothing. It is only used where those three samples
    are genuinely consecutive: across a gap the spacing is wrong and a parabola
    would extrapolate wildly, so there it falls back to a straight line."""
    n = len(ai)
    lin = np.interp(src_i, ai, vals)
    if n < 3:
        return lin
    k = np.clip(np.searchsorted(ai, src_i), 1, n - 2)
    a0, a1, a2 = ai[k - 1], ai[k], ai[k + 1]
    v0, v1, v2 = vals[k - 1], vals[k], vals[k + 1]
    t = src_i.astype(float)
    q = (v0 * (t - a1) * (t - a2) / ((a0 - a1) * (a0 - a2))
         + v1 * (t - a0) * (t - a2) / ((a1 - a0) * (a1 - a2))
         + v2 * (t - a0) * (t - a1) / ((a2 - a0) * (a2 - a1)))
    ok = (a1 - a0 == step) & (a2 - a1 == step)
    return np.where(ok, q, lin)


def _size_curve(ys, hs, base, bins=4):
    """Robust line of plate size against bar height.

    Fitted through BIN MEDIANS, not raw frames, and that detail is the whole
    thing. A clip is not evenly spread over bar height: a deadlift spends ten
    of its twenty seconds with the bar on the floor, so a straight fit is
    dominated by floor frames and outlier trimming then throws away the handful
    of lockout frames that carry the signal. Binning by height gives each part
    of the lift one vote.

    Returns (slope, intercept, fitted); `fitted` is False when the answer is a
    constant, because a bad extrapolation is worse than no correction at all."""
    if len(ys) < 4:
        return 0.0, (float(np.median(hs)) if hs else float(base)), False
    y = np.array(ys, float); h = np.array(hs, float)
    med_h = float(np.median(h))
    span = float(y.max() - y.min())
    if span < 8.0:
        return 0.0, med_h, False              # the bar never moved: nothing to fit
    edges = np.linspace(y.min(), y.max() + 1e-6, bins + 1)
    by, bh, bw = [], [], []
    for k in range(bins):
        m = (y >= edges[k]) & (y < edges[k + 1])
        if m.sum() < 1:
            continue
        by.append(float(np.median(y[m]))); bh.append(float(np.median(h[m])))
        bw.append(float(m.sum()))
    if len(by) < 3 or (max(by) - min(by)) < span * 0.5:
        return 0.0, med_h, False
    m_, c_ = (float(v) for v in np.polyfit(by, bh, 1, w=np.sqrt(bw)))
    ends = [m_ * float(y.min()) + c_, m_ * float(y.max()) + c_]
    # an extrapolation that changes the plate by more than a third is not the
    # camera geometry, it is a bad fit
    if med_h <= 0 or any(e < med_h * 0.72 or e > med_h * 1.38 for e in ends):
        return 0.0, med_h, False
    return m_, c_, True


# ---------------------------------------------------------------- entry
def analyse(path, tap=None, step=STEP, width=WIDTH, radius_px=None, seed_hint=None,
            on_point=None, on_stage=None, timeout=180.0, rot=0, tap_norm=None,
            seed_images=None, scale_images=None, plate_mm=PLATE_MM):
    """See _analyse_once. This adds the second look: when the first track says
    its own circle was the wrong one (dp.motion_profile found the edge of the
    moving plate somewhere else, or found nothing and the plate finder had a
    second circle), track once more from the better circle, reusing the frames
    already decoded, and keep whichever pass the clip itself confirms."""
    kw = dict(tap=tap, step=step, width=width, radius_px=radius_px, seed_hint=seed_hint,
              on_stage=on_stage, timeout=timeout, rot=rot, tap_norm=tap_norm,
              seed_images=seed_images, scale_images=scale_images, plate_mm=plate_mm)
    t0 = time.time()
    tape = []
    r1 = _analyse_once(path, on_point=on_point, _tape_out=tape, **kw)
    still = r1.pop("_still", None)
    first = r1.pop("_retry", None)
    if TRACKER != "dp" or not r1.get("ok") or seed_hint is not None or radius_px is not None:
        return r1
    best, looks = r1, []
    tried = [r1.get("seed")]
    todo = [first] if first else []
    rings_done = False

    def confirmed(r):
        """The pass's circle is the moving plate's edge: its own ring, or a
        ring the pass already rescaled to and then confirmed as "ok"."""
        m = (r.get("track") or {}).get("motion_rim")
        return bool(m) and (MR_LO <= m[0] <= MR_HI or r.get("verdict") == "ok")

    def near(h):
        for t in tried:
            if t and math.hypot(h["cx"] - t["cx"], h["cy"] - t["cy"]) < 0.2 * t["r"] \
                    and abs(h["r"] - t["r"]) < 0.12 * t["r"]:
                return True
        return False

    while len(looks) < MAX_LOOKS:
        if not todo:
            if confirmed(best) or rings_done or still is None:
                break
            # nothing else to try from the first look: whole circles at the tap
            rings_done = True
            tap_px = tap or ((tap_norm or (0.5, 0.5))[0] * still.shape[1],
                             (tap_norm or (0.5, 0.5))[1] * still.shape[0])
            try:
                rc = [c for c in _rf.candidates(still, tap_px) if c["hit"] >= RF_HIT and c["cover"] >= 0.5]
            except Exception:
                rc = []
            rc.sort(key=lambda c: -c["hit"])
            todo = [{"why": "a whole circle at the tap", "hint": {"cx": c["cx"], "cy": c["cy"], "r": c["r"]}}
                    for c in rc]
            continue
        retry = todo.pop(0)
        h = dict(retry["hint"])
        if retry.get("refine") and still is not None:
            # the size came from a coarse ring search; settle it on the still
            try:
                cx, cy, rr, sco = _snap.snap(still, (h["cx"], h["cy"]), h["r"], lock=True)
                if 0.85 * h["r"] <= rr <= 1.25 * h["r"] and math.hypot(cx - h["cx"], cy - h["cy"]) < 0.3 * h["r"]:
                    h = {"cx": cx, "cy": cy, "r": rr}
            except Exception:
                pass
        if near(h):
            continue
        tried.append(h)
        kw2 = dict(kw); kw2["seed_hint"] = h; kw2["radius_px"] = None
        try:
            r2 = _analyse_once(path, on_point=None, _tape_in=tape, _look=len(looks) + 1, **kw2)
        except Exception:
            break
        looks.append(retry["why"])
        r2.pop("_still", None)
        nxt = r2.pop("_retry", None)
        if nxt:
            todo.insert(0, nxt)
        t2 = r2.get("track") or {}
        if not r2.get("ok") or not t2.get("motion_rim"):
            continue
        if not r2.get("doubts") and confirmed_by_track(t2, r2.get("rep_metrics") or []):
            r2["verdict"] = "ok"
        else:
            r2["verdict"] = "check"
        r2["seed_verdict"] = r1.get("seed_verdict", r1.get("verdict"))
        r2["alternate"] = r1.get("alternate")
        r2["first_seed"] = r1.get("seed")
        r2["t_seed"] = r1.get("t_seed", 0.0)
        if (confirmed(r2) and not (best.get("verdict") == "ok" and r2["verdict"] != "ok")) \
                or not confirmed(best):
            best = r2
        if confirmed(r2) and r2["verdict"] == "ok":
            break
    tape.clear()
    if looks:
        best["second_look"] = {"why": looks, "kept": "first" if best is r1 else "later"}
        best["seconds"] = time.time() - t0
    return best


def _analyse_once(path, tap=None, step=STEP, width=WIDTH, radius_px=None, seed_hint=None,
                  on_point=None, on_stage=None, timeout=180.0, rot=0, tap_norm=None,
                  seed_images=None, scale_images=None, plate_mm=PLATE_MM,
                  _tape_out=None, _tape_in=None, _look=0):
    """tap: (x, y) in SOURCE pixels, where the coach tapped the plate.
    radius_px: source pixels, when the coach has already sized the circle —
               skips the search entirely.
    seed_hint: {'cx','cy','r'} in source pixels, same thing after a 'choose'.
    on_point(i, x, y, conf): fires during the forward pass so the UI can draw
               the path live. x, y are in source pixels.
    Returns a dict. 'verdict' is the plate finder's: only 'ok' and 'given'
    mean the seed can be trusted without asking the coach."""
    t_all = time.time()
    plate_r_m = float(plate_mm) / 2000.0
    stop = threading.Event()
    if _tape_in is not None:
        # second look: replay what the decoder produced the first time
        q = queue.Queue()
        for m_ in _tape_in:
            q.put(m_)
    else:
        q = queue.Queue(maxsize=96)
        threading.Thread(target=_decode_into,
                         args=(path, q, step, width, stop, SEED_N, rot, SCALE_EVERY),
                         daemon=True).start()
    stage = on_stage or (lambda *a: None)

    fps = 30.0; sc = None; src_n = 0; err = None; src_total = 0
    src_w = src_h = 0
    small = []                # stepped, working resolution — for the tracker
    big = []                  # consecutive, full resolution — for the seed
    fullq = {}                # analysed index -> full-res frame, awaiting a crop
    crops = []                # (analysed index, crop, x0, y0) kept for measuring
    done = False

    def pump(block=True):
        """Drain one message. Returns False at end of stream."""
        nonlocal fps, sc, src_n, err, done, src_total, src_w, src_h
        if done:
            return False
        m = q.get(timeout=timeout)
        if _tape_out is not None:
            _tape_out.append(m)
        k = m[0]
        if k == "f":
            sc = m[3]; small.append(m[2])
            if m[4] is not None:
                fullq[m[1]] = m[4]
        elif k == "s":
            big.append(m[1])
            if not src_w:
                src_h, src_w = m[1].shape[:2]
        elif k == "meta":
            fps = m[1]; src_total = m[2]
        elif k == "err":
            err = m[1]
        else:
            src_n = m[1]; done = True; return False
        return True

    # ---- seed. Runs as soon as eleven consecutive frames exist, i.e. long
    #      before the decoder has finished the clip.
    #
    # seed_images lets the caller supply them from OUTSIDE the clip. That is how
    # the upload shrinks: the phone sends a 540 wide video for the tracker plus
    # eleven full resolution stills for the plate finder, about 1.6 MB instead
    # of the 80 MB it sends today, and the plate finder still gets the detail it
    # needs. When they are supplied, the working scale is the clip against the
    # stills rather than the clip against itself.
    ext_seed = bool(seed_images)
    if ext_seed:
        big = list(seed_images)
        while sc is None and not done:
            pump()
    else:
        while len(big) < SEED_N and not done:
            pump()
    if not big:
        stop.set()
        return {"ok": False, "reason": "no frames decoded", "error": err,
                "seconds": time.time() - t_all}
    while sc is None and not done:
        pump()
    sc = sc or 1.0
    if ext_seed:
        # the seed answer is in STILL pixels; the tracker works in clip pixels
        clip_w = small[0].shape[1] if small else big[0].shape[1]
        sc = clip_w / float(big[0].shape[1])
        src_w, src_h = big[0].shape[1], big[0].shape[0]
    if tap is None:
        fx, fy = (tap_norm or (0.5, 0.5))
        tap = (fx * src_w, fy * src_h)

    t0 = time.time()
    alt = None
    if radius_px or seed_hint:
        # A supplied radius fixes the SIZE, not the CENTRE. The coach taps
        # somewhere on the plate, not on its axis, and the tap was 6 px from
        # the hub on one clip and 60 px from it on another. Tracking from an
        # off-centre template biases every point in the path by that offset,
        # so lock the radius and re-centre before anything else happens.
        h = seed_hint or {"cx": tap[0], "cy": tap[1], "r": radius_px}
        cx, cy, rr = float(h["cx"]), float(h["cy"]), float(h["r"])
        sco = 100.0
        if seed_hint is None:
            # A radius on its own fixes the SIZE, not the CENTRE: the coach
            # tapped somewhere on the plate, not on its axis, and tracking from
            # an off-centre template biases every point in the path by that
            # offset. Try to re-centre — but only accept the result if it is
            # both convincing and the same size, because locked-radius snapping
            # has been seen to wander onto a neighbouring plate and come back
            # with r=271 (score 26) for a plate of 227.
            try:
                ncx, ncy, nr, nsco, _cut, _cf = _snap.snap_multi(
                    big, 0, (cx, cy), hint_r=rr, lock=True)
                if nsco >= 60.0 and abs(nr - rr) / rr <= 0.10:
                    cx, cy, rr, sco = ncx, ncy, nr, float(nsco)
            except Exception:
                pass
        seed = {"cx": cx * sc, "cy": cy * sc, "r": rr * sc, "score": float(sco)}
        verdict = "given"
    else:
        res = find_plate(big, 0, (float(tap[0]), float(tap[1])))
        verdict = res["verdict"]
        p0 = res["primary"]
        seed = {"cx": p0["cx"] * sc, "cy": p0["cy"] * sc, "r": p0["r"] * sc,
                "score": float(p0.get("score", 0.0))}
        a = res.get("alternate")
        if a:
            alt = {"cx": float(a["cx"]), "cy": float(a["cy"]), "r": float(a["r"]),
                   "score": float(a.get("score", 0.0))}
        if verdict == "refuse" and TRACKER == "dp":
            # Neither estimator would stand behind a circle. Before refusing,
            # look for whole circles containing the tap (ringfind), which is
            # how a plate seen from below with the tap on its sleeve end
            # (2708), or one partly out of shot, is still found. Whatever it
            # finds is only provisional: the clip must then show it to be the
            # edge of the thing moving with the bar before anyone sees "ok".
            try:
                rc = [c for c in _rf.candidates(big[0], (float(tap[0]), float(tap[1])))
                      if c["hit"] >= RF_HIT_REFUSED and c["cover"] >= 0.5]
            except Exception:
                rc = []
            if rc:
                c = max(rc, key=lambda c: c["hit"] * min(1.0, c["cover"] / 0.8))
                seed = {"cx": c["cx"] * sc, "cy": c["cy"] * sc, "r": c["r"] * sc,
                        "score": round(100.0 * c["hit"], 1)}
                verdict = "check"
                alt = None
    t_seed = time.time() - t0
    still = big[0] if big else None
    # is the seed circle brighter inside than just outside it? That is a hub
    # (a chrome boss or a coloured insert in a dark plate), not a plate's rim.
    # Measured on the 47 field clips: the hubs read +10 to +40, real rims -80
    # to +5. Kept for track_doubts, which asks when no rim was found outside it.
    try:
        seed_hub = float(_snap._hubness(big[0], seed["cx"] / sc, seed["cy"] / sc, seed["r"] / sc))
    except Exception:
        seed_hub = 0.0
    big.clear()
    stage("seed", t_seed, verdict)
    if verdict == "refuse":
        stop.set()
        return {"ok": False, "reason": "no plate found at that tap", "verdict": verdict,
                "seconds": time.time() - t_all}

    # ---- forward pass, fed straight off the decoder
    R = seed["r"]
    tsc = min(1.0, v7.TARGET_D / (2.0 * R))
    # A small seed circle (a hub, or a plate far from the camera) would make
    # the working frames huge: 2843 seeded on a 33 px hub and tracked at 610 x
    # 1090, 20 seconds for a 36 second clip. Cap the working frame instead and
    # let the template be smaller; a plate 30 px across is still found.
    if small and TRACKER == "dp":
        long_side = max(small[0].shape[:2]) * tsc
        if long_side > WORK_MAX:
            tsc *= WORK_MAX / long_side
    fine = v7.FINE_MULT
    Rw = R * tsc
    efps = fps / step
    cap_px = v7.BAR_MAX_MS / efps * (Rw / plate_r_m)
    GW, GF = [], []

    def pairs():
        i = 0
        while True:
            while i >= len(small):
                if not pump():
                    return
            g = small[i]; i += 1
            gf = cv2.resize(g, None, fx=tsc * fine, fy=tsc * fine, interpolation=cv2.INTER_AREA)
            gw = cv2.resize(gf, None, fx=1.0 / fine, fy=1.0 / fine, interpolation=cv2.INTER_AREA)
            GW.append(gw); GF.append(gf)
            yield gw, gf

    gen = pairs()
    first = next(gen, None)
    if first is None:
        stop.set()
        return {"ok": False, "reason": "no frames", "seconds": time.time() - t_all}
    tf, thb = v7.build(GW[0], (seed["cx"] * tsc, seed["cy"] * tsc), Rw, fine)

    def chain():
        yield first
        for x in gen:
            yield x

    def take_crop(i, p):
        g = fullq.get(i)
        if g is None or p is None:
            return
        X = p[0] / tsc / sc; Y = p[1] / tsc / sc          # source pixels
        half = int(R / sc * 1.45)
        x0 = max(0, int(X) - half); y0 = max(0, int(Y) - half)
        x1 = min(g.shape[1], int(X) + half); y1 = min(g.shape[0], int(Y) + half)
        if x1 - x0 > 40 and y1 - y0 > 40:
            crops.append((i, g[y0:y1, x0:x1].copy(), x0, y0, X, Y))

    path_info = None
    if TRACKER == "dp":
        # ---- v9: candidates with plate evidence on every frame, as the frames
        #      arrive; one physically possible path for the whole clip at the end
        t0 = time.time()
        seed_w = (seed["cx"] * tsc, seed["cy"] * tsc)
        gx0, gy0 = _dp._grad(GW[0])
        circ0 = _dp.circle_score(gx0, gy0, seed_w[0], seed_w[1], Rw) or 1.0
        prof0 = _dp.ring_profile(cv2.GaussianBlur(GW[0].astype(np.float32), (0, 0), 1.0),
                                 seed_w[0], seed_w[1], Rw)
        DF = []
        last = seed_w
        fb0 = cv2.GaussianBlur(GW[0].astype(np.float32), (0, 0), 1.0)
        seed_patch = _dp.patch(fb0, seed_w[0], seed_w[1], Rw)
        c0 = _dp.crop(GW[0], seed_w[0], seed_w[1], Rw)
        carry = [(seed_w[0], seed_w[1], c0)] if c0 is not None else []
        near_rad = max(cap_px * 2.0, Rw * 0.6)
        # The work for a frame that does not depend on the frame before it
        # (gradient, whole frame template peaks, circle votes) runs on a pool
        # of threads a few frames ahead of the sequential part (following the
        # previous frame's candidates forward, scoring). OpenCV releases the
        # interpreter lock, so this uses every core Railway gives us.
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=N_WORKERS)
        ahead = deque()
        src = chain()
        k = 0
        try:
            while True:
                while len(ahead) < N_WORKERS * 3:
                    nxt = next(src, None)
                    if nxt is None:
                        break
                    ahead.append(pool.submit(_dp.prepare, nxt[0], nxt[1], tf, Rw))
                if not ahead:
                    break
                ctx = ahead.popleft().result()
                fr_ = _dp.finish(ctx, Rw, circ0, prof0, carry=carry, carry_rad=near_rad)
                DF.append(fr_)
                gw = ctx.gw
                carry = []
                for j_ in np.argsort(fr_.unary)[:_dp.N_CARRY]:
                    cc = _dp.crop(gw, fr_.xy[j_, 0], fr_.xy[j_, 1], Rw)
                    if cc is not None:
                        carry.append((fr_.xy[j_, 0], fr_.xy[j_, 1], cc))
                # a provisional point for the live drawing: the best candidate a
                # barbell could have reached. The answer itself comes from solve().
                pick = None
                if len(fr_.xy):
                    dd = np.hypot(fr_.xy[:, 0] - last[0], fr_.xy[:, 1] - last[1])
                    okk = dd <= near_rad
                    if okk.any():
                        j = int(np.argmin(np.where(okk, fr_.unary, 1e9)))
                        pick = (float(fr_.xy[j, 0]), float(fr_.xy[j, 1]))
                        last = pick
                if on_point:
                    on_point(k, None if pick is None else pick[0] / tsc / sc,
                             None if pick is None else pick[1] / tsc / sc,
                             0.0 if pick is None else float(fr_.ncc[j]))
                k += 1
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        t_fwd = time.time() - t0
        stage("forward", t_fwd, len(DF))
        t0 = time.time()
        dpath = _dp.solve(DF, seed_w, Rw, efps, seed_patch=seed_patch)
        dpath = _dp.refine_path(dpath, GF, tf, fine)
        # is the seed circle the rim? asked of the whole path now it is known
        rim_ratio, rim_score, _prof = _dp.rim_check(GW, dpath, Rw)
        mprof = _dp.motion_profile(GW, dpath, Rw)
        mrim = _dp.motion_rim(mprof)
        resolved = False
        if mrim is not None:
            q = mrim[0]
            if MR_LO <= q <= MR_HI:
                # the circle that moves with the bar is the seed's own: the
                # ring a static shape check might prefer is not the plate's
                rim_ratio = 1.0
            else:
                rim_ratio = q
        merged = [None if p is None else (p[0], p[1]) for p in dpath]
        conf = [0.0 if p is None else p[2] for p in dpath]
        circ = [None if p is None else p[3] for p in dpath]
        pts = [None if p is None else (p[0] / tsc, p[1] / tsc) for p in merged]
        out, gaps = _dp.bridge(pts, v7.BRIDGE_MAX)
        if rim_ratio != 1.0:
            R = R * rim_ratio
            seed["r"] = R
            stage("rim", rim_ratio, rim_score)
        for i, p in enumerate(merged):
            take_crop(i, p)
        fullq.clear()
        t_rev = time.time() - t0
        stage("reverse", t_rev, len(out))
        vis = [p for p in dpath if p is not None]
        path_info = {"visited": len(vis), "frames": len(dpath),
                     "rim_ratio": rim_ratio, "rim_score": rim_score,
                     "rim_profile": None if _prof is None else
                         [[round(float(q), 2), round(float(v), 2)] for q, v in zip(_dp.RIM_RATIOS, _prof)],
                     "motion_rim": mrim, "resolved": resolved,
                     "gaps_s": _gaps_s(gaps, dpath, efps),
                     "motion_profile": None if mprof is None else
                         [[round(float(q), 2), round(float(a), 2), round(float(b), 2), round(float(c), 2)]
                          for q, a, b, c in zip(*mprof[:4])],
                     "motion_spread": None if mprof is None else round(mprof[4], 2),
                     "ncc_med": float(np.median([p[2] for p in vis])) if vis else 0.0,
                     "circ_med": float(np.median([p[3] for p in vis])) if vis else 0.0,
                     "circ_low": float(np.mean([p[3] < 0.45 for p in vis])) if vis else 1.0,
                     "shape_med": float(np.median([p[4] for p in vis])) if vis else 0.0,
                     "shape_low": float(np.mean([p[4] < 0.3 for p in vis])) if vis else 1.0}
    else:
        def sink(i, p, c):
            take_crop(i, p)
            fullq.pop(i, None)
            if on_point:
                on_point(i, None if p is None else p[0] / tsc / sc,
                         None if p is None else p[1] / tsc / sc, float(c))
        t0 = time.time()
        gw_it, gf_it = _split(chain())
        fwd, cf = v7.pass_dir(gw_it, gf_it, tf, thb, Rw, cap_px, fine,
                              start=(seed["cx"] * tsc, seed["cy"] * tsc),
                              hw=GW[0].shape, sink=sink)
        t_fwd = time.time() - t0
        stage("forward", t_fwd, len(fwd))

        # ---- reverse pass, over frames already in memory
        t0 = time.time()
        tail = None
        for i in range(len(fwd) - 1, -1, -1):
            if fwd[i] is not None and cf[i] > 0.35:
                tail = i; break
        if tail is None:
            rev, cr = [None] * len(GW), [0.0] * len(GW)
        else:
            rs, cs = v7.pass_dir(GW[tail::-1], GF[tail::-1], tf, thb, Rw, cap_px, fine,
                                 start=fwd[tail])
            rev = rs[::-1] + [None] * (len(GW) - tail - 1)
            cr = cs[::-1] + [0.0] * (len(GW) - tail - 1)
        merged, conf = [], []
        for a, b, ca, cb in zip(fwd, rev, cf, cr):
            if a is None and b is None: merged.append(None); conf.append(0.0)
            elif a is None: merged.append(b); conf.append(cb)
            elif b is None: merged.append(a); conf.append(ca)
            elif math.hypot(a[0] - b[0], a[1] - b[1]) < Rw * 0.6:
                w = ca + cb + 1e-9
                merged.append(((a[0] * ca + b[0] * cb) / w, (a[1] * ca + b[1] * cb) / w))
                conf.append(max(ca, cb))
            else:
                merged.append(a if ca >= cb else b); conf.append(max(ca, cb))
        pts = [None if p is None else (p[0] / tsc, p[1] / tsc) for p in merged]
        out, gaps = v7.finish(pts, conf)
        t_rev = time.time() - t0
        stage("reverse", t_rev, len(out))

    # ---- reps and real-world numbers
    Hf = GW[0].shape[0] / tsc
    idx = [i for i, p in enumerate(out) if p]
    ys = [out[i][1] / Hf for i in idx]
    ts = [i / efps for i in idx]
    reps = detect_reps(ys, ts)
    ppm = R / PLATE_R_M                     # working px per metre
    out_reps = []
    for rp in reps:
        seg = [out[idx[k]] for k in range(rp["start"], rp["end"] + 1)]
        yy = [p[1] for p in seg]; xx = [p[0] for p in seg]
        out_reps.append({"start_s": round(ts[rp["start"]], 2),
                         "end_s": round(ts[rp["end"]], 2),
                         "rom_cm": round(float(max(yy) - min(yy)) / ppm * 100, 1),
                         "drift_cm": round(float(max(xx) - min(xx)) / ppm * 100, 1)})

    # ---- how big is the plate, as a function of where the bar is?
    #
    # A single radius measured at the start is wrong for the rest of the lift.
    # On a deadlift filmed from near floor height the plate reads 205 px tall
    # with the bar on the ground and 190 px at lockout: the bar has risen above
    # the lens and is now being looked at from below. Keep using the floor
    # figure and the scale is 8% too generous at the top, so range of motion
    # comes back 8% short — about 4 cm on a 50 cm pull.
    #
    # So measure the plate at a couple of dozen points through the clip and fit
    # the half height against bar height. One frame on its own is far too noisy
    # for this (the same lockout measured 172 to 239 px across nine frames); the
    # line through all of them is stable.
    meas = []
    if scale_images:
        # Full resolution stills taken through the clip, for the case where the
        # video itself was shrunk before upload. A 540 wide clip puts a 225 mm
        # plate at about 100 px across, which is not enough to measure an 8%
        # change in its height: on a deadlift that cost 3 cm of range. The
        # stills cost about a megabyte and put the measurement back.
        H_full = float(big[0].shape[0]) if ext_seed and big else None
        for t_s, img in scale_images:
            k = int(round(t_s * fps))
            if k < 0 or k >= len(out) * step:
                continue
            j = k // step
            if j >= len(out) or out[j] is None:
                continue
            X = out[j][0] / tsc / sc
            Y = out[j][1] / tsc / sc                  # still pixels: seed scale
            try:
                e = _el.measure(img, X, Y, R / sc)
            except Exception:
                e = None
            if e and e[2] >= 24:
                meas.append((float(Y), float(e[0]), float(e[1])))
    if not meas:
        for (_i, crop, x0, y0, X, Y) in crops:
            try:
                e = _el.measure(crop, X - x0, Y - y0, R / sc)
            except Exception:
                e = None
            if e and e[2] >= 24:
                meas.append((float(Y), float(e[0]), float(e[1])))
    base = R / sc
    hm, hc, h_fit = _size_curve([m[0] for m in meas], [m[2] for m in meas], base)
    wm, wc, w_fit = _size_curve([m[0] for m in meas], [m[1] for m in meas], base)
    off_deg = 0.0
    if meas:
        off_deg = _el.off_square_deg(float(np.median([m[1] for m in meas])),
                                     float(np.median([m[2] for m in meas])))
    stage("scale", len(meas), h_fit)

    # ---- resample to the source frame rate so the caller gets one point per
    #      video frame, which is what the overlay draws against.
    n_src = src_n or (len(GW) * step)
    idx_an = [i * step for i in range(len(out))]
    fr_src = []; mY = []; mX = []; inv = []
    ppm_src = base / plate_r_m
    if idx_an:
        xs = np.array([p[0] / sc if p else np.nan for p in out])
        ys_ = np.array([p[1] / sc if p else np.nan for p in out])
        gi = ~np.isnan(xs)
        if gi.sum() >= 2:
            src_i = np.arange(n_src)
            ai = np.array(idx_an)[gi]
            if QUAD:
                xi = _quad_interp(ai, xs[gi], src_i, step)
                yi = _quad_interp(ai, ys_[gi], src_i, step)
            else:
                xi = np.interp(src_i, ai, xs[gi])
                yi = np.interp(src_i, ai, ys_[gi])
            lo, hi = ai[0], ai[-1]
            # Which source frames sit inside a stretch the tracker never
            # measured? Interpolating across one missed analysed frame is
            # harmless; interpolating across ten invents a straight line
            # through whatever the bar really did, and the velocity that comes
            # out of it is fiction. Mark them so the numbers can say so.
            invented = np.zeros(n_src, bool)
            for a_, b_ in zip(ai[:-1], ai[1:]):
                if b_ - a_ > step * 2:
                    invented[int(a_) + 1:int(b_)] = True
            W_ = (GW[0].shape[1] / tsc) / sc
            H_ = (GW[0].shape[0] / tsc) / sc
            # metres of real bar travel, integrated along the path so a scale
            # that changes with height is handled exactly rather than averaged
            hy = np.maximum(4.0, hm * yi + hc)
            wx = np.maximum(4.0, wm * yi + wc)
            Ym = np.zeros(n_src); Xm = np.zeros(n_src)
            for k in range(1, n_src):
                Ym[k] = Ym[k - 1] + (yi[k] - yi[k - 1]) * plate_r_m / (0.5 * (hy[k] + hy[k - 1]))
                Xm[k] = Xm[k - 1] + (xi[k] - xi[k - 1]) * plate_r_m / (0.5 * (wx[k] + wx[k - 1]))
            ppm_src = float(np.median(hy)) / plate_r_m
            for k in range(n_src):
                if k < lo or k > hi:
                    continue
                fr_src.append({"t": round(float(k / fps), 4),
                               "x": round(float(xi[k]) / W_, 5),
                               "y": round(float(yi[k]) / H_, 5)})
                mY.append(float(Ym[k])); mX.append(float(Xm[k]))
                inv.append(bool(invented[k]))

    rep_metrics = []
    if fr_src:
        tv_all = [f["t"] for f in fr_src]
        rs = detect_reps(mY, tv_all, min_rom=MIN_REP_M)
        rs += final_pull(mY, tv_all, rs)
        for rp in rs:
            a, b = rp["start"], rp["end"]
            yv = mY[a:b + 1]; xv = mX[a:b + 1]; tv = tv_all[a:b + 1]
            # Velocity over a ~50 ms window, not frame to frame.
            #
            # Frame-to-frame differencing at 59 fps is a noise amplifier: one
            # pixel of jitter is 6 cm/s, and a single bad frame produced a
            # "peak" of 4.82 m/s on a deadlift, which then got the whole rep
            # refused as impossible. It also made peak velocity depend on an
            # internal sampling constant -- the same lift read 1.56 m/s
            # analysed every 8th frame and 4.82 m/s analysed every 2nd, which
            # is not a number anyone can coach off.
            #
            # 50 ms is also what the peak velocity of a barbell actually means:
            # nobody cares what it did for one sixtieth of a second.
            w = max(1, int(round(len(tv) / max(tv[-1] - tv[0], 1e-6) * VEL_WINDOW_S)))
            vs = [abs(yv[i] - yv[i - w]) / max(tv[i] - tv[i - w], 1e-3)
                  for i in range(w, len(yv))]
            peak = float(np.max(vs)) if vs else 0.0
            guessed = float(np.mean(inv[a:b + 1])) if inv else 0.0
            con = concentric(yv, tv, rp["peak"] - a, w)
            if rp.get("partial") and con is None and rp["peak"] - a >= 3:
                con = concentric(yv + [yv[-1]] * 3, tv + [tv[-1] + 0.01 * k for k in (1, 2, 3)],
                                 rp["peak"] - a, w)
            # Range of motion is the FIRST leg of the rep: top to bottom on a
            # squat or bench, floor to lockout on a deadlift. Max minus min
            # over the whole window also counted the walk into the rack after
            # the last squat (3104: 0.76 and 0.79 m against about 0.70).
            rom = abs(float(yv[rp["peak"] - a]) - float(yv[0]))
            if a <= 1:
                # the clip starts inside this rep (2984 starts with the bar
                # already off the floor), so its first leg is partial and the
                # second leg is the only whole one
                rom = max(rom, abs(float(yv[rp["peak"] - a]) - float(yv[-1])))
            rom_window = float(max(yv) - min(yv))
            why = []
            # Range of motion and peak velocity are separate claims and they
            # fail separately. A single jumpy frame wrecks a peak velocity and
            # leaves the range untouched -- on one deadlift rep the range came
            # out 60.8 cm and 62.2 cm at two sampling rates while the "peak"
            # read 2.96 and 1.41. Refusing the whole rep over the velocity
            # threw away a range that was fine. So say which number is good.
            vel_why = []
            pc = con["peak_ms"] if con else peak
            if pc > SANE_MAX_MS:
                vel_why.append("bar speed above %.1f m/s, which no barbell does "
                               "in these lifts" % SANE_MAX_MS)
            if con and con["mean_ms"] > MCV_MAX:
                vel_why.append("mean concentric speed above %.1f m/s, faster than a "
                               "loaded bar is lifted" % MCV_MAX)
            if rom > SANE_MAX_M:
                why.append("range above %.0f cm, which no barbell does in these "
                           "lifts" % (SANE_MAX_M * 100))
            if guessed > 0.15:
                why.append("bar lost for part of the rep")
            rep_metrics.append({"start": a, "end": b, "peak": rp["peak"],
                                "rom_m": round(rom, 3),
                                "rom_window_m": round(rom_window, 3),
                                "drift_m": round(float(max(xv) - min(xv)), 3),
                                "mean_ms": round(float(np.mean(vs)) if vs else 0.0, 3),
                                "peak_ms": round(peak, 3),
                                "mean_concentric_ms": None if not con else round(con["mean_ms"], 3),
                                "peak_concentric_ms": None if not con else round(con["peak_ms"], 3),
                                "concentric_s": None if not con else round(con["secs"], 2),
                                "concentric_start_s": None if not con else round(con["t0"], 3),
                                "measured": not why,
                                "velocity_ok": not (why or vel_why),
                                "unmeasured_because": ", ".join(why) or None,
                                "velocity_unreliable_because":
                                    ", ".join(why + vel_why) or None})
        rep_metrics = drop_bounces(rep_metrics, mY, tv_all)

    # ---- does the track itself hold up? The verdict used to be the plate
    #      finder's alone, so a path that left the plate still answered "ok".
    doubts = track_doubts(fr_src, rep_metrics, ppm_src, path_info,
                          src_w, src_h, W_ if fr_src else 1.0, H_ if fr_src else 1.0)
    motion_ok = bool((path_info or {}).get("motion_rim"))
    if (seed_hub >= HUB_ASK and (path_info or {}).get("rim_ratio", 1.0) == 1.0
            and not motion_ok):
        doubts.append("the circle looks like the plate's hub, not its rim, and the "
                      "rim could not be found (cut off by the frame?)")
    seed_verdict = verdict
    if TRACKER == "dp" and not motion_ok and verdict == "ok":
        # Five stills agreed on a circle, but the clip did not show it to be
        # the edge of the thing moving with the bar. Not wrong, not proven.
        doubts.append("the plate's edge could not be confirmed from the bar's movement")
    if doubts and verdict in ("ok", "given"):
        verdict = "check"
    elif not doubts and verdict in ("check", "choose") and confirmed_by_track(path_info, rep_metrics):
        # The plate finder was unsure from five frames at the tap; the whole
        # clip has since settled it. The circle it offered first is round, of
        # that size, around the path, in every part of the set, the path never
        # does anything a barbell cannot, and every rep measures. Asking the
        # coach to confirm a circle the clip has already confirmed is the
        # question that made v8 ask on 16 of 47 clips.
        verdict = "ok"

    stop.set()
    # ---- a second look, from a better circle? (analyse() decides)
    retry = None
    if TRACKER == "dp" and path_info is not None and (
            (_look == 0 and seed_hint is None and radius_px is None) or 1 <= _look < MAX_LOOKS):
        m_ = path_info.get("motion_rim")
        p0 = next((p for p in dpath if p is not None), None)
        if m_ is not None and m_[0] < MR_LO and p0 is not None:
            # the edge of the moving plate is well inside the seed circle: the
            # template carried the background with it (3459 reached into the
            # rack and stayed there while the lifter walked out)
            retry = {"why": "seed circle larger than the moving plate", "refine": True,
                     "hint": {"cx": p0[0] / tsc / sc, "cy": p0[1] / tsc / sc,
                              "r": float(seed["r"]) / sc}}
        elif m_ is not None and m_[0] > MR_HI and doubts and p0 is not None:
            # The seed was an inner ring (a hub, an insert). Its path is right
            # wherever the hub was found, but the template was the hub and
            # every speed limit was judged against a plate that small, so the
            # path lost whatever the bar did faster: 2919 lost the bottom of
            # every bench rep. Track again with the plate's own size.
            retry = {"why": "seed circle was the plate's hub", "refine": True,
                     "hint": {"cx": p0[0] / tsc / sc, "cy": p0[1] / tsc / sc,
                              "r": float(seed["r"]) / sc}}
        elif m_ is None and alt is not None and verdict in ("choose", "check"):
            retry = {"why": "the first circle's edge did not move with the bar",
                     "hint": {"cx": alt["cx"], "cy": alt["cy"], "r": alt["r"]}}
    total = time.time() - t_all
    vid_s = src_n / fps if src_n else len(GW) * step / fps
    return {"ok": True, "_retry": retry, "_still": still, "frames_src": fr_src, "rep_metrics": rep_metrics,
            "src_w": src_w, "src_h": src_h, "src_frames": n_src,
            "src_total": src_total, "src_fps": fps, "verdict": verdict, "alternate": alt,
            "seed_verdict": seed_verdict, "doubts": doubts, "track": path_info,
            "tracker": TRACKER,
            "seed": {"cx": float(seed["cx"]) / sc, "cy": float(seed["cy"]) / sc,
                     "r": float(R) / sc, "score": seed["score"]},
            "path": [None if p is None else (float(p[0]) / sc, float(p[1]) / sc) for p in out],
            "conf": [float(c) for c in conf], "gaps": [list(g) for g in gaps],
            "fps": efps, "scale": sc, "px_per_m": float(ppm_src), "reps": out_reps,
            "off_square_deg": round(float(off_deg), 1),
            "plate_measurements": len(meas),
            "plate_mm": float(plate_mm),
            "scale_model": ("height-dependent" if h_fit else
                            ("measured constant" if meas else "seed radius")),
            "plate_half_height_px": [round(float(hc), 1), round(float(hm), 5)],
            "tracked": int(sum(1 for p in out if p)), "frames": len(GW),
            "seconds": total, "video_s": vid_s,
            "realtime_x": (vid_s / total) if total else 0.0,
            "t_seed": t_seed, "t_forward": t_fwd, "t_reverse": t_rev, "error": err}


def concentric(yv, tv, k_peak, w, onset_ms=0.05):
    """The rising leg of one rep, trimmed to the bar actually moving.

    yv is bar height in metres measured DOWNWARDS (image convention), so rising
    is yv falling. A squat or bench starts high, so its rising leg is the
    second half (peak to end); a deadlift starts on the floor, so it is the
    first half (start to peak). The leg is trimmed to the bar moving (see
    below), so the seconds on the floor before a pull and a pause at lockout
    count for nothing (2860 rep 6 read 0.08 m/s with the rest left in).
    Returns mean and peak velocity in m/s, or None."""
    n = len(yv)
    if n < 4 or k_peak <= 0 or k_peak >= n - 1:
        return None
    y = np.asarray(yv, float); t = np.asarray(tv, float)
    if y[k_peak] < y[0]:           # the far end is higher: the first leg rises
        a, b = 0, k_peak
    else:                          # the far end is lower: the second leg rises
        a, b = k_peak, n - 1
    if b - a < 3:
        return None
    seg = y[a:b + 1]; ts = t[a:b + 1]
    # bottom and top of the leg, then velocity on the window
    lo = int(np.argmax(seg)); hi = lo + int(np.argmin(seg[lo:]))
    if hi - lo < 2:
        return None
    ww = max(1, w)
    v = np.zeros(len(seg))
    for i in range(len(seg)):
        i0 = max(0, i - ww); i1 = min(len(seg) - 1, i + ww)
        dt = ts[i1] - ts[i0]
        v[i] = (seg[i0] - seg[i1]) / dt if dt > 0 else 0.0     # + is rising
    # the fastest point OF THE MAIN RISE, between 20% and 80% of the way up:
    # a bumper bouncing off the floor before the pull is faster than the pull
    # itself and used to be taken as the rep (2857 rep 9)
    b0, t0_ = float(seg[lo]), float(seg[hi])
    span = b0 - t0_
    mid = [i for i in range(lo, hi + 1) if 0.2 * span <= b0 - seg[i] <= 0.8 * span]
    kmax = max(mid, key=lambda i: v[i]) if mid else lo + int(np.argmax(v[lo:hi + 1]))
    # trim by position, not by speed. A speed threshold ended the leg at a
    # deadlift's sticking point; a "within 1 cm of the top" rule let a lifter
    # settling into lockout stretch the leg by a second (2843 rep 1 read 0.21
    # m/s against 0.6 for the other five). So the leg runs from 3% of the rise
    # to 97% of it (at least 1 cm each end), which keeps the sticking point
    # and drops the creep at either end.
    # the bottom and top are where the bar was AT REST either side of its
    # fastest point, not the extremes of the window: a dropped deadlift
    # bottoms out 2 cm below where it then sits (2860 rep 6 read 0.08 m/s
    # because the leg started at the bounce, seven seconds before the pull)
    def rest(k, d):
        i = k
        while 0 < i < len(seg) - 1:
            j = i
            while 0 <= j + d < len(seg) and abs(ts[j + d] - ts[i]) < 0.2:
                j += d
            if j == i or abs(float(seg[j] - seg[i])) < 0.01:
                return i
            i += d
        return i
    lo = rest(kmax, -1); hi = rest(kmax, +1)
    if hi - lo < 2:
        return None
    bot, top = float(seg[lo]), float(seg[hi])
    total = bot - top
    if total <= 0.03:
        return None
    m = max(0.01, 0.03 * total)
    s0 = lo
    while s0 < hi and bot - seg[s0] < m:
        s0 += 1
    e0 = hi
    while e0 > s0 and seg[e0] - top < m:
        e0 -= 1
    e0 = min(hi, e0 + 1)
    s0 = max(lo, s0 - 1)
    dt = ts[e0] - ts[s0]
    if dt <= 0.05:
        return None
    rise = float(seg[s0] - seg[e0])
    if rise <= 0.02:
        return None
    return {"mean_ms": rise / dt, "peak_ms": float(v[s0:e0 + 1].max()),
            "secs": float(dt), "t0": float(ts[s0])}


def final_pull(mY, tv, reps):
    """A clip that stops at the top of a deadlift (3053 ends at lockout) has
    lifted the bar and never put it down, so no rep closes. The pull is still
    a complete concentric and the coach filmed it to see it. Counted when the
    last leg rises at least 25 cm (and at least half the clip's biggest rep)
    and the bar is still at least 60% of the way up when the clip ends."""
    if len(mY) < 6:
        return []
    start = reps[-1]["end"] if reps else 0
    seg = np.asarray(mY[start:], float)
    if len(seg) < 6:
        return []
    lo = int(np.argmax(seg))                       # lowest point (y grows down)
    if lo >= len(seg) - 3:
        return []
    top = lo + int(np.argmin(seg[lo:]))
    rise = float(seg[lo] - seg[top])
    need = 0.25
    if reps:
        need = max(need, 0.5 * max(float(max(mY[r["start"]:r["end"] + 1]) - min(mY[r["start"]:r["end"] + 1]))
                                    for r in reps))
    if rise < need or float(seg[-1] - seg[top]) > 0.4 * rise:
        return []
    # shaped like a rep whose far end is the last point
    return [{"start": start + lo, "peak": start + top, "end": len(mY) - 1, "partial": True}]


def drop_bounces(reps, mY, tv):
    """A bumper plate dropped from lockout bounces 15 to 20 cm in under half a
    second, and a pivot detector counts that as a rep (2819 counted 3 for 2,
    2856 5 for 4). A rep is kept only if its range is at least half the
    biggest rep in the clip, so a bounce, a re-grip or a hitch off the floor
    after the set never becomes a rep."""
    if len(reps) < 2:
        return reps
    big = max(r["rom_m"] for r in reps)
    return [r for r in reps if r["rom_m"] >= 0.5 * big]


def _gaps_s(gaps, path, efps):
    """[start s, length s, interior] for each stretch the path did not visit
    (bridged or not). Interior means visited frames on both sides."""
    vis = [i for i, p in enumerate(path) if p is not None]
    if not vis:
        return []
    first, last = vis[0], vis[-1]
    out = []
    i = 0
    while i < len(path):
        if path[i] is None:
            j = i
            while j < len(path) and path[j] is None:
                j += 1
            out.append([round(i / efps, 2), round((j - i) / efps, 2), bool(i > first and j <= last)])
            i = j
        else:
            i += 1
    return out


def confirmed_by_track(info, reps):
    """The plate finder was unsure from five stills; has the whole clip
    settled it? Only if the circle the path carries is the edge of the thing
    moving with the bar (dp.motion_profile), the path stays on it for the set,
    and every rep measures to a sane range. A circle whose size the motion
    check had to correct must pass a higher bar."""
    if not info or not reps:
        return False
    m = info.get("motion_rim")
    if not m:
        return False
    q, ring, move = m
    moved = not (MR_LO <= q <= MR_HI)
    if ring < (0.60 if moved else _dp.MP_RING) or move < (0.60 if moved else _dp.MP_MOVE):
        return False
    if info.get("frames") and info["visited"] < 0.85 * info["frames"]:
        return False
    if not all(r.get("measured") for r in reps):
        return False
    rom = float(np.median([r["rom_m"] for r in reps]))
    return 0.15 <= rom <= 0.95


def track_doubts(fr_src, reps, ppm, info, src_w, src_h, W_, H_):
    """Reasons to show the coach the path before trusting its numbers."""
    out = []
    if not fr_src:
        return ["no path"]
    if info:
        if info.get("frames") and info["visited"] < 0.6 * info["frames"]:
            out.append("the plate was not followed for %d%% of the clip"
                       % round(100 * (1 - info["visited"] / info["frames"])))
        if info.get("shape_low", 0) > 0.35:
            out.append("much of the path is not on a round plate")
        lost = [g for g in info.get("gaps_s") or [] if g[2] and g[1] >= GAP_ASK]
        if lost:
            g = max(lost, key=lambda g: g[1])
            out.append("the bar was lost for %.1f s at %.1f s, long enough to hide a rep"
                       % (g[1], g[0]))
    # physically impossible steps between consecutive points
    if ppm and len(fr_src) > 2:
        x = np.array([p["x"] for p in fr_src]) * W_ / ppm
        y = np.array([p["y"] for p in fr_src]) * H_ / ppm
        t = np.array([p["t"] for p in fr_src])
        dt = np.diff(t)
        okd = dt > 0
        vx = np.abs(np.diff(x))[okd] / dt[okd]; vy = np.diff(y)[okd] / dt[okd]
        if len(vx) and (vx.max() > 1.2 or (-vy).max() > 2.5 or vy.max() > 4.5):
            out.append("the path jumps faster than a barbell moves")
    bad = [r for r in reps if not r.get("measured")]
    if bad:
        out.append("%d of %d reps could not be measured (%s)"
                   % (len(bad), len(reps), bad[0].get("unmeasured_because")))
    slow = [r for r in reps if r.get("measured") and not r.get("velocity_ok")]
    if slow:
        out.append("%d of %d reps have a speed no loaded bar reaches (%s)"
                   % (len(slow), len(reps), slow[0].get("velocity_unreliable_because")))
    roms = [r["rom_m"] for r in reps if r.get("measured")]
    if roms and float(np.median(roms)) > 0.95:
        out.append("every rep reads longer than a barbell travels, so the circle is "
                   "probably the hub rather than the plate")
    return out


def _split(pairs):
    """pass_dir does `zip(GW, GF)`, which pulls one from GW then one from GF.
    So a single generator can feed both: the GW side produces the next pair and
    parks its GF half, the GF side hands that half straight back. One frame in
    flight, never a buffered clip."""
    box = []
    def gw():
        for p in pairs:
            box.append(p[1]); yield p[0]
    return gw(), _Lockstep(box)


class _Lockstep:
    def __init__(self, box): self.box = box
    def __iter__(self): return self
    def __next__(self):
        if not self.box:
            raise StopIteration
        return self.box.pop(0)
