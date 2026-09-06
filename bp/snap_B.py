"""Snap a coach's tap to the actual plate — position AND radius.

Why this exists: the benchmark showed the tracker does not much care where the
coach taps (an 11px miss costs nothing, because a constant offset shifts the
whole path without distorting it) but it cares a great deal about the RADIUS.
A seed given r=178 for a 230px plate dropped 45 frames; the same clip seeded at
the right size dropped none. The radius sets the template, the working
resolution and the physical speed cap all at once, and a coach tapping a phone
cannot be asked to supply it. So we measure it.

Not Hough: Hough votes on binarised edge pixels, and a black bumper plate
against a dark rack produces very few. This accumulates the RADIAL COMPONENT OF
THE IMAGE GRADIENT around each candidate circle, which uses the gradient at full
strength wherever it exists and fades gracefully where it does not. The sign is
taken as absolute, so a dark plate on a bright wall and a bright plate on a dark
floor both work without being told which.

The one real subtlety: a bumper plate contains SEVERAL true concentric circles —
the coloured hub ring, the insert, the outer rim. All are correct answers to
"find a circle", and on real footage the hub usually WINS, because bright yellow
against black rubber is a far crisper edge than black rubber against a dim gym.
Picking it would put the pixels-per-metre scale out by a factor of nearly three,
making the tracker's physical speed cap far too tight and causing false losses.

Simply preferring the largest well-scoring circle does not work either: the hub
outscores the rim by more than the margin, so the rim never qualifies. What does
work is the structure itself. Those circles are CONCENTRIC. So: find the
strongest circle wherever it is, lock its centre, then sweep outwards from that
fixed centre and take the outermost radius that still shows a clear edge. A
spurious large circle in the background will not happen to be centred on the
plate's hub, so fixing the centre first makes a much lower acceptance threshold
safe.
"""
import cv2, numpy as np, math

KEEP  = 0.72         # a radius is "good" if it scores this fraction of the best
OUTER = 0.34         # once the centre is locked, a concentric edge this strong counts
HUB_DARK = -10.0     # interior this much darker than surround => already the rim
CONF_MIN = 75.0      # below this we did not find a plate, say so
OUT_MAX = 3.6        # the rim is at most this many hub-radii out


def _prep(gray, tap, rmax_guess):
    H, W = gray.shape
    pad = int(rmax_guess * 1.3) + 10
    x0, y0 = int(max(0, tap[0] - pad)), int(max(0, tap[1] - pad))
    x1, y1 = int(min(W, tap[0] + pad)), int(min(H, tap[1] + pad))
    win = gray[y0:y1, x0:x1]
    sc = min(1.0, 150.0 / max(20.0, rmax_guess))
    w = cv2.GaussianBlur(cv2.resize(win, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                         .astype(np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(w, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(w, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy) + 1e-6
    return w, gx / mag, gy / mag, mag, sc, x0, y0


def _polish(score, bx, by, r, sc, x0, y0):
    """Joint centre+radius refinement only, used when the radius is given."""
    for step in (r * 0.04, r * 0.015, r * 0.006):
        cand = (bx, by, r, score(bx, by, r))
        for dx in np.arange(-2, 2.01, 1.0):
            for dy in np.arange(-2, 2.01, 1.0):
                for dr in np.arange(-2, 2.01, 1.0):
                    rr = r + dr * step
                    if rr < 8: continue
                    v = score(bx + dx * step, by + dy * step, rr)
                    if v > cand[3]:
                        cand = (bx + dx * step, by + dy * step, rr, v)
        bx, by, r, s = cand
    return (bx / sc + x0, by / sc + y0, r / sc, s)


def snap(gray, tap, hint_r=None, n_ang=96, passes=2, lock=False):
    """Return (cx, cy, r, score) in full-resolution pixels.

    Run twice by default, feeding the first pass's centre back in as the tap. A
    tap far off centre can land the first pass on an inner feature — the chrome
    collar, say — and a second pass from the corrected centre recovers the rim.
    It costs about 70ms and only happens once, when the coach taps."""
    if lock:
        return _snap1(gray, tap, hint_r, n_ang, lock=True)
    res = _snap1(gray, tap, hint_r, n_ang)
    for _ in range(max(0, passes - 1)):
        nxt = _snap1(gray, (res[0], res[1]), res[2], n_ang)
        # FIX (6 Sep, found on Hayden's 20-clip real-footage set): the second
        # pass exists to GROW the radius from an inner feature out to the rim.
        # Accepting it purely on score let it do the opposite. A bumper hub is
        # bright metal on black rubber and outscores the rim against a dim gym
        # every time, so `nxt[3] > res[3]` handed the hub back on 5 of 20 clips
        # and put pixels-per-metre out by 3-4x. Growth is always welcome; a
        # shrink now has to stay near the current radius to be accepted.
        if nxt[2] > res[2] * 1.05:
            res = nxt
        elif nxt[3] > res[3] and nxt[2] > res[2] * 0.85:
            res = nxt
    return res


def _hubness(gray, cx, cy, r):
    """Is this circle a bright hub inside a dark plate, or the plate rim itself?

    Measured across the twenty-clip real-footage set: a hub (chrome or coloured
    insert ringed by black rubber) reads -7 to +27, a genuine rim reads -1 to
    -57. So a clearly darker interior than surround means we already have the
    rim and must NOT sweep outward — which is what was dragging the bench clips
    off the bar and into the loaded rack behind the lifter."""
    H, W = gray.shape
    pad = int(r * 2.2) + 6
    x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
    x1, y1 = min(W, int(cx + pad)), min(H, int(cy + pad))
    p = gray[y0:y1, x0:x1].astype(np.float32)
    rh, rw = p.shape
    Y, X = np.ogrid[:rh, :rw]
    d = np.hypot(X - (cx - x0), Y - (cy - y0))
    inn = p[d <= r * 0.80]
    out = p[(d >= r * 1.25) & (d <= r * 2.0)]
    if inn.size < 80 or out.size < 80:
        return 0.0                      # cannot tell - allow the sweep
    return float(np.median(inn) - np.median(out))


def _snap1(gray, tap, hint_r=None, n_ang=96, lock=False):
    H, W = gray.shape
    small = min(H, W)
    rlo = max(12.0, small * 0.04)
    rhi = small * 0.46
    if hint_r and lock:
        # FIX 3 (6 Sep). The coach has given us the radius directly (tap the
        # centre, drag to the rim), so this is refinement, not search. The old
        # hint band (0.35x to 2.6x) still contained the hub, and on real gym
        # footage the hub outscores the rim every time, so a CORRECT hint could
        # still be dragged down onto the hub — measured on this set, a supplied
        # 136 came back as 53. Locking the band around the hint makes the hub
        # unreachable and leaves only sub-pixel refinement to do.
        rlo = max(12.0, hint_r * 0.78)
        rhi = min(small * 0.5, hint_r * 1.28)
    elif hint_r:
        rlo = max(rlo, hint_r * 0.35)
        rhi = min(rhi, hint_r * 2.6)
    w, ux, uy, mag, sc, x0, y0 = _prep(gray, tap, rhi)
    Hs, Ws = w.shape
    ang = np.linspace(0, 2 * math.pi, n_ang, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)

    def score(cx, cy, r):
        px = cx + r * ca; py = cy + r * sa
        ok = (px >= 1) & (px < Ws - 1) & (py >= 1) & (py < Hs - 1)
        if ok.sum() < n_ang * 0.5:
            return -1.0
        xi = px[ok].astype(np.int32); yi = py[ok].astype(np.int32)
        rad = np.abs(ux[yi, xi] * ca[ok] + uy[yi, xi] * sa[ok]) * mag[yi, xi]
        # median, not mean: a lifter's body occluding a third of the rim should
        # not be able to veto the right circle
        return float(np.median(rad))

    cx0, cy0 = (tap[0] - x0) * sc, (tap[1] - y0) * sc
    radii = np.geomspace(rlo * sc, rhi * sc, 44)
    prof = []
    for r in radii:
        st = max(1.0, r * 0.09)
        best = -1.0; bc = (cx0, cy0)
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                s = score(cx0 + dx * st, cy0 + dy * st, r)
                if s > best: best, bc = s, (cx0 + dx * st, cy0 + dy * st)
        prof.append((r, best, bc))

    top = max(p[1] for p in prof)
    if top <= 0:
        return (tap[0], tap[1], hint_r or small * 0.12, 0.0)
    r, s, (bx, by) = max(prof, key=lambda p: p[1])

    # Lock the centre on the strongest circle, then walk OUTWARD looking for the
    # rim. Everything concentric with the hub belongs to the same plate, so a
    # much lower bar is safe here than in the open search above.
    for step in (r * 0.05, r * 0.02):
        cand = (bx, by, r, s)
        for dx in np.arange(-2, 2.01, 1.0):
            for dy in np.arange(-2, 2.01, 1.0):
                v = score(bx + dx * step, by + dy * step, r)
                if v > cand[3]: cand = (bx + dx * step, by + dy * step, r, v)
        bx, by, r, s = cand

    # FIX 2 (6 Sep, real-footage set): the outward sweep used to run all the way
    # to rhi. In a gym the background is full of plates, racks and mirrors, so
    # there is nearly always SOME concentric edge response out there at 40-60%
    # of peak, which clears OUTER and drags the fit off the bar. Both clean
    # bench clips ran away to r=463 and r=375 from true plates of 107 and 113.
    # A plate's rim is at most about 3-4x its hub, so cap the sweep.
    if lock:
        return _polish(score, bx, by, r, sc, x0, y0)
    if _hubness(gray, bx / sc + x0, by / sc + y0, r / sc) <= HUB_DARK:
        return _polish(score, bx, by, r, sc, x0, y0)   # already the rim

    rs = np.geomspace(r, min(rhi * sc, r * OUT_MAX), 40)
    prof2 = [(rr, score(bx, by, rr)) for rr in rs]
    base = max(p[1] for p in prof2)
    outer = r
    for i in range(1, len(prof2) - 1):
        rr, v = prof2[i]
        if v >= OUTER * base and v >= prof2[i - 1][1] and v >= prof2[i + 1][1]:
            outer = rr
    r = outer

    # final joint refinement of centre and radius
    for step in (r * 0.04, r * 0.015, r * 0.006):
        cand = (bx, by, r, score(bx, by, r))
        for dx in np.arange(-2, 2.01, 1.0):
            for dy in np.arange(-2, 2.01, 1.0):
                for dr in np.arange(-2, 2.01, 1.0):
                    rr = r + dr * step
                    if rr < 8: continue
                    v = score(bx + dx * step, by + dy * step, rr)
                    if v > cand[3]:
                        cand = (bx + dx * step, by + dy * step, rr, v)
        bx, by, r, s = cand
    return (bx / sc + x0, by / sc + y0, r / sc, s)


def snap_multi(frames, seed_frame, tap, spread=None, k=5, hint_r=None, lock=False):
    """Measure the plate on SEVERAL frames and take the median radius.

    A single frame is a fragile place to measure from. On a clip where the plate
    is cut by the frame edge and a lens flare crosses the shot, snapping one
    frame at a time returned radii of 175, 314, 348, 449 and 465 px for the same
    plate. The plate's size barely changes across a clip, so measuring it five
    times and taking the median turns a coin-flip into a reliable number — and
    the tracker cares more about the radius than about anything else the tap
    provides.

    Returns (cx, cy, r, score, cut) where `cut` is True if the fitted circle
    leaves the frame — the caller should tell the coach the framing is too
    tight rather than trusting the measurement.
    """
    n = len(frames)
    seed_frame = max(0, min(seed_frame, n - 1))
    # The sampled frames must be CLOSE. The tap is only valid where the coach
    # put it, and a barbell moves: sampling nine frames away meant snapping a
    # frame where the plate had left the tap behind, which returned whatever
    # circle happened to be there and poisoned the median. Two frames at 59fps
    # is 34ms and about ten pixels of travel.
    if spread is None:
        spread = 2
    offs = [0]
    for j in range(1, k):
        offs.append(spread * ((j + 1) // 2) * (1 if j % 2 else -1))
    hits = []
    ref = tap
    for o in offs:
        f = seed_frame + o
        if not (0 <= f < n):
            continue
        try:
            cx, cy, r, s = snap(frames[f], ref, hint_r, lock=lock)
        except Exception:
            continue
        hits.append((cx, cy, r, s, f))
        if o == 0:
            ref = (cx, cy)          # later frames start from the corrected centre
    if not hits:
        return (tap[0], tap[1], hint_r or min(frames[0].shape) * 0.12, 0.0, False)
    rs = sorted(h[2] for h in hits)
    rmed = rs[len(rs) // 2]
    # re-centre on the seed frame with the agreed radius, so the centre comes
    # from the frame the coach actually looked at
    cx, cy, r, s = snap(frames[seed_frame], tap, rmed, lock=lock)
    if abs(r - rmed) / rmed > 0.25:
        cx, cy, r = cx, cy, rmed
    H, W = frames[seed_frame].shape
    cut = (cx - r < 0) or (cy - r < 0) or (cx + r > W) or (cy + r > H)
    # FIX 5 (6 Sep, real-footage set): a low edge score means we did not really
    # find a plate. Measured on 20 clips, good fits score 85-192 and every bad
    # one scores under 45 — a bare barbell with no plate at all scored 17.8 and
    # was still handed back as a confident 228px circle. Silently wrong is the
    # worst outcome for a coaching tool, so say so instead.
    confident = s >= CONF_MIN
    return (cx, cy, r, s, cut, confident)
