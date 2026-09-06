"""Bar path tracker v6.

v3 was flawless on five of six hostile clips and failed hard on the sixth. The
failure is worth stating precisely, because the fix follows from it:

    When the bar left the top of the frame, the tracker walked across the gym
    and settled on a STATIONARY PLATE ON A RACK — and reported 0.88 confidence
    the whole time it was wrong.

Two things went wrong, and both are structural rather than a tuning miss.

  RATCHETING. The motion prior penalises distance from the LAST ACCEPTED
  position. That makes one 600px jump impossible but six 100px steps free, and
  once the target is absent there is nothing to hold the chain in place. A soft
  penalty cannot stop a walk; only a hard gate can.

  A STATIONARY DISTRACTOR IS THE PERFECT PREY FOR A MOTION PRIOR. Something
  that never moves matches "predicted = previous" on every single frame, for
  free, forever. The rack was not tracked despite the prior — it was tracked
  BECAUSE of it.

  And confidence did not catch it. Correlation answers "does this look like the
  template", not "is this the object I was asked to follow". A second identical
  plate scores just as well. No confidence threshold can separate them, so the
  gate has to be physical, not photometric.

v4 therefore adds:

  A PHYSICAL DISPLACEMENT CAP. A loaded barbell has a top speed. Given the
  plate radius in pixels we know the pixels-per-metre, so we know how far the
  bar can possibly move between two analysed frames. The cap is set at 4 m/s —
  roughly triple a hard squat and above a competition jerk — so it never bites
  on real lifting, but it makes a jump across the gym impossible.

  FREEZE ON LOSS, AND ADMIT IT. If nothing lies within the cap, v4 emits None
  and does NOT advance the anchor or the velocity. A frozen anchor cannot
  ratchet. The bar is then picked up again when it returns to where it left.

  A SEARCH RADIUS THAT GROWS BY PHYSICS, NOT BY HOPE. While frozen the radius
  grows one frame's travel per frame, because that is the exact bound on where
  the bar can be. It stops growing after a few frames: an unbounded radius
  eventually covers the whole gym and re-opens the very hole it closed.

  THE BACKWARD PASS IS ANCHORED BY THE FORWARD PASS. Starting the reverse pass
  at the last frame is only safe if the bar is visible there. v4 starts it from
  the last frame the forward pass was confident about instead.

  THE LOCAL MATCH IS ALWAYS A CANDIDATE. This is the deepest of the fixes.
  v4 built its candidate list from the three strongest correlation peaks in the
  whole frame. When an arm swept across the bar, the bar's peak fell to ~0.5
  while three rack plates sat at 0.88 — so the bar was not on the list at all,
  the tracker saw nothing within the cap, and it froze even though the bar was
  right there and perfectly visible. The rescue path made it worse: the hub
  template was only searched when the best score was low, and the best score was
  measured ACROSS THE WHOLE FRAME, so a distractor scoring 0.88 suppressed the
  very rescue that the occluded bar needed.

  v5 always adds one more candidate: a correlation measured in a window around
  where the bar should be. It is still an absolute measurement made in that
  frame alone — nothing is integrated — but it means a weakly-visible target can
  never be crowded off the list by strong distractors elsewhere. The hub
  template is now gated on the LOCAL score, which is the number that was
  supposed to be asked about all along.

  A MATCH HAS TO BE GOOD ENOUGH TO BE BELIEVED. Always supplying a local
  candidate fixed the crowding-out problem and created a new one: a local match
  is inside the cap by construction, so there was always something to accept and
  the tracker could never freeze at all. When the bar was 90% off the top of the
  frame it accepted a rack upright at 0.50 and walked down it. So an accepted
  match must also clear a quality bar — and that bar is RELATIVE to how well
  this clip has been matching, not an absolute number. A fixed threshold was
  what made the original Hough attempt useless: 0.35 rejected 599 frames out of
  600 on footage whose median peak was 0.29. The gate here is a fraction of the
  running median of accepted scores, so it adapts to the clip's own contrast,
  compression and lighting.

  GAPS ARE REPORTED, NOT PAPERED OVER. A long stretch with no measurement means
  the bar was not in the frame. Drawing a smooth interpolation across it would
  be an invented path, so finish() bridges short gaps and hands back the long
  ones for the caller to show the coach.
"""
import cv2, numpy as np, math, time

TARGET_D = 56.0
FINE_MULT = 2.0
SCALES = (0.87, 1.0, 1.15)
PLATE_R_M = 0.225        # standard 450mm competition plate
BAR_MAX_MS = 2.8         # a hard physical ceiling on bar speed.
#
# 4.0 was chosen as "generous" against a competition jerk. On real gym footage
# it was too generous by half: on a bench clip the tracker leapt 55 cm between
# two analysed frames 136 ms apart, which is 4.04 m/s, and squeaked under the
# old cap. It had jumped onto a plate stored on the rack behind the lifter, and
# then reported seven reps of 55 cm at 3.6 m/s -- for a bench press, where the
# bar covers about 20 cm and never exceeds 1 m/s.
#
# 2.8 is still above anything a barbell does in these lifts (a jerk drive peaks
# near 2 m/s, a competition squat or bench far below that) and it kills the
# jump. Tried 2.2 as well: no better on the clips that were already right, and
# worse on one, where the tighter gate made the tracker drop the bar entirely.
EDGE_LIM   = 1.5         # search radius, in frames of travel, for a bar re-entering the shot
EDGE_QUAL  = 0.85        # re-entering the shot demands a much better match than merely continuing
QUAL       = 0.68        # accepted score must clear this fraction of the running median
GROW_MAX   = 4           # frames of travel the search radius may grow to while frozen
BRIDGE_MAX = 3           # analysed frames; longer gaps are reported, never invented


def _mk(img, s, fine):
    sx, sy = s if isinstance(s, tuple) else (s, s)
    return (cv2.resize(img, None, fx=sx, fy=sy, interpolation=cv2.INTER_AREA),
            cv2.resize(img, None, fx=sx * fine, fy=sy * fine, interpolation=cv2.INTER_AREA))


def build(g0, seed, R, fine=FINE_MULT):
    x, y = seed
    H, W = g0.shape
    r = int(round(R))
    full = g0[max(0, int(y - r)):min(H, int(y + r)), max(0, int(x - r)):min(W, int(x + r))]
    hr = max(3, int(R * 0.55))
    hub = g0[max(0, int(y - hr)):min(H, int(y + hr)), max(0, int(x - hr)):min(W, int(x + hr))]
    if full.size == 0 or hub.size == 0:
        raise ValueError('seed outside frame')
    return ([_mk(full, s, fine) for s in SCALES], [_mk(hub, s, fine) for s in SCALES])


def _peaks(res, tpl, k, floor=0.04):
    out, work = [], res.copy()
    for _ in range(k):
        _, mx, _, loc = cv2.minMaxLoc(work)
        if mx < floor:
            break
        work[max(0, loc[1] - 12):loc[1] + 13, max(0, loc[0] - 12):loc[0] + 13] = -1
        out.append((mx, loc[0] + tpl.shape[1] / 2, loc[1] + tpl.shape[0] / 2))
    return out


def _subpix(r2, l2):
    x, y = l2
    xi, yi = float(x), float(y)
    if 0 < x < r2.shape[1] - 1:
        dn = 2 * r2[y, x] - r2[y, x - 1] - r2[y, x + 1]
        if abs(dn) > 1e-6:
            xi += float(np.clip((r2[y, x + 1] - r2[y, x - 1]) / (2 * dn), -1, 1))
    if 0 < y < r2.shape[0] - 1:
        dn = 2 * r2[y, x] - r2[y - 1, x] - r2[y + 1, x]
        if abs(dn) > 1e-6:
            yi += float(np.clip((r2[y + 1, x] - r2[y - 1, x]) / (2 * dn), -1, 1))
    return xi, yi


def _local(gw, gf, tpls, R, fine, pred, rad):
    """Correlate in a window around the prediction. Returns the peak whatever its
    global rank. This is still an absolute measurement in this frame — the
    window only says where to look, never what the answer is."""
    best = None
    x0 = int(max(0, pred[0] - rad - R)); y0 = int(max(0, pred[1] - rad - R))
    x1 = int(min(gw.shape[1], pred[0] + rad + R)); y1 = int(min(gw.shape[0], pred[1] + rad + R))
    win = gw[y0:y1, x0:x1]
    for t_w, t_f in tpls:
        if t_w.shape[0] >= win.shape[0] or t_w.shape[1] >= win.shape[1]:
            continue
        res = cv2.matchTemplate(win, t_w, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if best is None or mx > best[0]:
            best = (mx, x0 + loc[0] + t_w.shape[1] / 2, y0 + loc[1] + t_w.shape[0] / 2, t_f)
    return best


def _refine(gf, cx, cy, t_f, fine):
    fx, fy = cx * fine, cy * fine
    pad = int(t_f.shape[0] * 0.35) + 6
    x0 = int(max(0, fx - t_f.shape[1] / 2 - pad)); y0 = int(max(0, fy - t_f.shape[0] / 2 - pad))
    x1 = int(min(gf.shape[1], fx + t_f.shape[1] / 2 + pad))
    y1 = int(min(gf.shape[0], fy + t_f.shape[0] / 2 + pad))
    roi = gf[y0:y1, x0:x1]
    if roi.shape[0] <= t_f.shape[0] or roi.shape[1] <= t_f.shape[1]:
        return None
    r2 = cv2.matchTemplate(roi, t_f, cv2.TM_CCOEFF_NORMED)
    _, m2, _, l2 = cv2.minMaxLoc(r2)
    xi, yi = _subpix(r2, l2)
    return m2, (x0 + xi + t_f.shape[1] / 2) / fine, (y0 + yi + t_f.shape[0] / 2) / fine


def _candidates(gw, gf, tpl_full, tpl_hub, R, fine, pred=None, rad=None,
                topk=4, hub_gate=0.55):
    """Absolute measurement only — no motion prior enters the scoring here.
    Returns (weight, raw, x, y, is_local)."""
    out = []

    # (1) the local match, always present when we have a prediction
    local_score = 1.0
    if pred is not None:
        lb = _local(gw, gf, tpl_full, R, fine, pred, rad)
        if lb is not None:
            local_score = lb[0]
            r = _refine(gf, lb[1], lb[2], lb[3], fine)
            out.append((r[0], r[0], r[1], r[2], True) if r else (lb[0], lb[0], lb[1], lb[2], True))
        # the hub set is the answer to OCCLUSION, so it is gated on how the bar
        # is doing where the bar is — not on the best score somewhere else
        if local_score < hub_gate:
            hb = _local(gw, gf, tpl_hub, R * 0.55, fine, pred, rad)
            if hb is not None:
                r = _refine(gf, hb[1], hb[2], hb[3], fine)
                sc_ = (r[0] if r else hb[0]) * 0.85
                pos = (r[1], r[2]) if r else (hb[1], hb[2])
                out.append((sc_, r[0] if r else hb[0], pos[0], pos[1], True))

    # (2) full-frame peaks, so the tracker can re-anchor absolutely and never
    #     depends on its own history to find the plate
    glob = []
    for t_w, t_f in tpl_full:
        if t_w.shape[0] >= gw.shape[0] or t_w.shape[1] >= gw.shape[1]:
            continue
        res = cv2.matchTemplate(gw, t_w, cv2.TM_CCOEFF_NORMED)
        for mx, cx, cy in _peaks(res, t_w, topk):
            glob.append((mx, mx, cx, cy, t_f))
    if pred is None and not glob:
        for t_w, t_f in tpl_hub:
            if t_w.shape[0] >= gw.shape[0] or t_w.shape[1] >= gw.shape[1]:
                continue
            res = cv2.matchTemplate(gw, t_w, cv2.TM_CCOEFF_NORMED)
            for mx, cx, cy in _peaks(res, t_w, topk):
                glob.append((mx * 0.75, mx, cx, cy, t_f))
    glob.sort(key=lambda c: -c[0])
    uniq = []
    for c in glob:
        if all(math.hypot(c[2] - u[2], c[3] - u[3]) > R * 0.5 for u in uniq):
            uniq.append(c)
        if len(uniq) >= 3:
            break
    for w, raw, cx, cy, t_f in uniq:
        if any(math.hypot(cx - o[2], cy - o[3]) <= R * 0.5 for o in out):
            continue
        r = _refine(gf, cx, cy, t_f, fine)
        out.append((r[0], r[0], r[1], r[2], False) if r else (w, raw, cx, cy, False))
    return out


def pass_dir(GW, GF, tpl_full, tpl_hub, R, cap, fine=FINE_MULT, lam=0.010, start=None,
             grow_max=GROW_MAX, qual=QUAL, warm=5, edge_lim=None, hw=None, sink=None):
    """Growth is switched off when the bar was last seen AT THE FRAME EDGE: that
    is the signature of the bar leaving the shot, and a bar that leaves through
    the top comes back through the top, near where it left. Growing the radius
    there only opens the door to a distractor — which is precisely how an
    earlier version ended up following a stationary rack plate for 250 frames."""
    H, W = hw if hw is not None else GW[0].shape
    preds, conf = [], []
    anchor = None if start is None else np.array(start, float)
    vel = np.zeros(2)
    frozen = 0
    hist = []
    for gw, gf in zip(GW, GF):
        if anchor is None:
            cands = _candidates(gw, gf, tpl_full, tpl_hub, R, fine)
            if not cands:
                preds.append(None); conf.append(0.0)
                if sink: sink(len(preds) - 1, None, 0.0)
                continue
            c = max(cands, key=lambda c: c[0])
            anchor = np.array([c[2], c[3]]); vel = np.zeros(2); frozen = 0
            hist.append(c[1])
            preds.append((c[2], c[3])); conf.append(c[1])
            if sink: sink(len(preds) - 1, (c[2], c[3]), c[1])
            continue

        at_edge = (anchor[0] < R * 1.1 or anchor[1] < R * 1.1 or
                   anchor[0] > W - R * 1.1 or anchor[1] > H - R * 1.1)
        if at_edge and frozen > 0:
            # The bar left the shot. It comes back through the edge it left, at
            # the place it left — so search a tight circle around the ANCHOR and
            # do not extrapolate the velocity it had on the way out, which points
            # further off-screen and drags the search window with it.
            pred = anchor.copy()
            lim = cap * (edge_lim if edge_lim else EDGE_LIM)
        else:
            pred = anchor + vel
            lim = cap * (2.0 if at_edge else min(1 + frozen, grow_max))
        cands = _candidates(gw, gf, tpl_full, tpl_hub, R, fine, pred=pred, rad=lim)
        # Resuming after the bar left the SHOT is a different claim from
        # continuing an ongoing track, and it deserves a higher standard of
        # evidence. A bar coming back into frame produces a strong, unambiguous
        # match once enough of it is visible; a half-hearted 0.6 on some gym
        # furniture is not the bar returning, and letting it through is how the
        # tracker used to end up following a rack for the rest of the clip.
        q = (EDGE_QUAL if (at_edge and frozen > 0) else qual)
        floor = q * float(np.median(hist)) if len(hist) >= warm else 0.0
        inside = [c for c in cands
                  if math.hypot(c[2] - pred[0], c[3] - pred[1]) <= lim and c[1] >= floor]
        if inside:
            best = max(inside, key=lambda c: c[0] - lam * math.hypot(c[2] - pred[0], c[3] - pred[1]))
            cur = np.array([best[2], best[3]])
            vel = 0.6 * vel + 0.4 * (cur - anchor) if frozen == 0 else np.zeros(2)
            anchor = cur; frozen = 0
            hist.append(best[1])
            if len(hist) > 90: hist.pop(0)
            preds.append((best[2], best[3])); conf.append(best[1])
            if sink: sink(len(preds) - 1, (best[2], best[3]), best[1])
        else:
            preds.append(None); conf.append(0.0)
            if sink: sink(len(preds) - 1, None, 0.0)
            vel = vel * 0.5
            frozen += 1
    return preds, conf


def track(GRAY, seed, R, fps=30.0, target_d=TARGET_D, fine=FINE_MULT, every=1,
          max_ms=BAR_MAX_MS, qual=QUAL):
    """GRAY: full-resolution grayscale frames. seed, R: full-resolution pixels.
    Returns (positions per input frame, confidence, seconds, gaps)."""
    t0 = time.time()
    sc = min(1.0, target_d / (2.0 * R))
    idx = list(range(0, len(GRAY), every))
    GW = [cv2.resize(GRAY[i], None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) for i in idx]
    GF = [cv2.resize(GRAY[i], None, fx=sc * fine, fy=sc * fine, interpolation=cv2.INTER_AREA) for i in idx]
    Rw = R * sc
    # pixels-per-metre from the plate itself, so the cap is in real units
    px_per_m = Rw / PLATE_R_M
    cap = max_ms / fps * every * px_per_m

    tf, th = build(GW[0], (seed[0] * sc, seed[1] * sc), Rw, fine)
    fwd, cf = pass_dir(GW, GF, tf, th, Rw, cap, fine, start=(seed[0] * sc, seed[1] * sc), qual=qual)

    # Anchor the reverse pass on the forward pass's last confident frame rather
    # than on the last frame, which may not contain the bar at all.
    tail = None
    for i in range(len(fwd) - 1, -1, -1):
        if fwd[i] is not None and cf[i] > 0.35:
            tail = i; break
    if tail is None:
        rev, cr = [None] * len(GW), [0.0] * len(GW)
    else:
        rev_s, cr_s = pass_dir(GW[tail::-1], GF[tail::-1], tf, th, Rw, cap, fine, start=fwd[tail], qual=qual)
        rev = rev_s[::-1] + [None] * (len(GW) - tail - 1)
        cr = cr_s[::-1] + [0.0] * (len(GW) - tail - 1)

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

    out = [None if p is None else (p[0] / sc, p[1] / sc) for p in merged]
    if every > 1:
        out, conf = _upsample(out, conf, idx, len(GRAY), every)
    return out, conf, time.time() - t0


def _upsample(vals, conf, idx, n, every):
    xs = np.array([v[0] if v else np.nan for v in vals])
    ys = np.array([v[1] if v else np.nan for v in vals])
    cs = np.array(conf, float)
    good = ~np.isnan(xs)
    if good.sum() < 2:
        return [None] * n, [0.0] * n
    src = np.array(idx)[good]
    fx = np.interp(np.arange(n), src, xs[good])
    fy = np.interp(np.arange(n), src, ys[good])
    fc = np.interp(np.arange(n), src, cs[good])
    out = [(float(a), float(b)) for a, b in zip(fx, fy)]
    # do not invent positions inside a gap wider than BRIDGE_MAX analysed frames
    holes = np.where(~good)[0]
    for h in holes:
        run = 1
        k = h
        while k + 1 < len(good) and not good[k + 1]:
            run += 1; k += 1
    i = 0
    g = list(good)
    while i < len(g):
        if not g[i]:
            j = i
            while j < len(g) and not g[j]: j += 1
            if j - i > BRIDGE_MAX:
                lo = idx[i]; hi = idx[j - 1] + every
                for m in range(max(0, lo), min(n, hi)):
                    out[m] = None; fc[m] = 0.0
            i = j
        else:
            i += 1
    return out, [float(c) for c in fc]


ABS_OK = 0.50            # a match this good is kept whatever the clip's median
# Swept 0.60 / 0.50 / 0.40 on the real clips. 0.60 still lost nine frames at
# one lockout; 0.40 let a bad track through on a clip that was already wrong.
# At 0.50 two deadlifts went from 117/141 and 132/146 frames to complete, with
# no gaps at all, and one of them went from reporting a single 48 cm rep to two
# reps of 60 cm -- because the frames being deleted were the lockouts, which
# are the top of the range.


def finish(preds, conf, k=0.75, bridge_max=BRIDGE_MAX, abs_ok=ABS_OK):
    """Drop low-confidence frames and bridge SHORT holes. Long holes are handed
    back so the caller can tell the coach the bar was not in frame, instead of
    drawing a path that was never measured.

    The relative test alone was wrong. Judging every frame against 0.75 of the
    clip's MEDIAN score assumes the plate looks equally matchable throughout,
    and it does not: the plate changes shape as the bar rises past the camera,
    so the frames at the top of the lift score lower than the rest. On a
    deadlift that quietly deleted thirteen measured frames at each lockout --
    the exact frames that set the range of motion -- and reported them as the
    bar having left the shot. A frame that matches well in absolute terms is
    kept now, whatever the rest of the clip managed."""
    live = [c for c in conf if c > 0]
    med = float(np.median(live)) if live else 0.0
    out = [p if (p is not None and (conf[i] >= med * k or conf[i] >= abs_ok)) else None
           for i, p in enumerate(preds)]
    gaps = []
    i = 0
    while i < len(out):
        if out[i] is None:
            j = i
            while j < len(out) and out[j] is None: j += 1
            a = out[i - 1] if i > 0 else None
            b = out[j] if j < len(out) else None
            if (j - i) <= bridge_max and a and b:
                for m in range(i, j):
                    f = (m - i + 1) / (j - i + 1)
                    out[m] = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            else:
                gaps.append((i, j))
            i = j
        else:
            i += 1
    return out, gaps
