"""ForceTrack bar path: the whole analysis, end to end.

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
import snap as _snap
import ellipse as _el
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


# ---------------------------------------------------------------- decoding
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
    q = queue.Queue(maxsize=96)
    stop = threading.Event()
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
    t_seed = time.time() - t0
    big.clear()
    stage("seed", t_seed, verdict)
    if verdict == "refuse":
        stop.set()
        return {"ok": False, "reason": "no plate found at that tap", "verdict": verdict,
                "seconds": time.time() - t_all}

    # ---- forward pass, fed straight off the decoder
    R = seed["r"]
    tsc = min(1.0, v7.TARGET_D / (2.0 * R))
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
            gw = cv2.resize(g, None, fx=tsc, fy=tsc, interpolation=cv2.INTER_AREA)
            gf = cv2.resize(g, None, fx=tsc * fine, fy=tsc * fine, interpolation=cv2.INTER_AREA)
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

    def sink(i, p, c):
        if p is not None and i in fullq:
            g = fullq.pop(i)
            X = p[0] / tsc / sc; Y = p[1] / tsc / sc          # source pixels
            half = int(R / sc * 1.45)
            x0 = max(0, int(X) - half); y0 = max(0, int(Y) - half)
            x1 = min(g.shape[1], int(X) + half); y1 = min(g.shape[0], int(Y) + half)
            if x1 - x0 > 40 and y1 - y0 > 40:
                crops.append((i, g[y0:y1, x0:x1].copy(), x0, y0, X, Y))
        elif i in fullq:
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
            # A barbell in these lifts does not exceed about 2 m/s. A rep that
            # says otherwise is not a fast rep, it is a rep the tracker got
            # wrong, and so is its range of motion. Say so rather than print it.
            why = []
            rom = float(max(yv) - min(yv))
            # Range of motion and peak velocity are separate claims and they
            # fail separately. A single jumpy frame wrecks a peak velocity and
            # leaves the range untouched -- on one deadlift rep the range came
            # out 60.8 cm and 62.2 cm at two sampling rates while the "peak"
            # read 2.96 and 1.41. Refusing the whole rep over the velocity
            # threw away a range that was fine. So say which number is good.
            vel_why = []
            if peak > SANE_MAX_MS:
                vel_why.append("bar speed above %.1f m/s, which no barbell does "
                               "in these lifts" % SANE_MAX_MS)
            if rom > SANE_MAX_M:
                why.append("range above %.0f cm, which no barbell does in these "
                           "lifts" % (SANE_MAX_M * 100))
            if guessed > 0.15:
                why.append("bar lost for part of the rep")
            why += vel_why and [] or []
            rep_metrics.append({"start": a, "end": b,
                                "rom_m": round(rom, 3),
                                "drift_m": round(float(max(xv) - min(xv)), 3),
                                "mean_ms": round(float(np.mean(vs)) if vs else 0.0, 3),
                                "peak_ms": round(peak, 3),
                                "measured": not why,
                                "velocity_ok": not (why or vel_why),
                                "unmeasured_because": ", ".join(why) or None,
                                "velocity_unreliable_because":
                                    ", ".join(why + vel_why) or None})

    stop.set()
    total = time.time() - t_all
    vid_s = src_n / fps if src_n else len(GW) * step / fps
    return {"ok": True, "frames_src": fr_src, "rep_metrics": rep_metrics,
            "src_w": src_w, "src_h": src_h, "src_frames": n_src,
            "src_total": src_total, "src_fps": fps, "verdict": verdict, "alternate": alt,
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
