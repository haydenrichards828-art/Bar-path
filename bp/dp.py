"""Bar path tracker v9: every frame's candidates, one path chosen for the
whole clip.

What v8 got wrong, measured on 47 of Hayden's clips (4 Nov 2026)
-----------------------------------------------------------------
v8 tracked greedily, one frame at a time, and accepted the best template
match inside a search window. Three failures followed from that, and all
three answered "ok":

  * It took the best match it could see, whether or not it was a plate. At a
    bench unrack the bar moves sideways out of the hooks, the window missed
    it, and the best match left in the window was the lifter's knee. At a
    deadlift lockout it was the lifter's chest. Nothing asked whether the
    thing being followed had a round rim of the plate's size.
  * Once it was off the plate, it never came back. Each frame trusted the one
    before, so a single bad step became the rest of the set.
  * A loss in the middle of a rep could only be repaired by the reverse pass,
    which made the same greedy choices from the other end.

v9 does three things instead:

  1. PLATE EVIDENCE ON EVERY CANDIDATE. Each frame offers a handful of
     candidates (template peaks over the whole frame, plus the best match near
     where the bar was). Every one is scored twice: how much it looks like the
     plate (template correlation), and whether there is a round edge at the
     plate's radius around it (median radial gradient at the seed radius,
     against the same measurement on the seed). A knee or a chest has no rim
     of that size.
  2. ONE PATH FOR THE WHOLE CLIP. The path is chosen by dynamic programming
     over all frames at once: the cheapest chain of candidates from the tapped
     plate to the end, where each step must be physically possible for a
     barbell and every skipped frame costs something. A frame where nothing
     convincing is found is skipped, not filled with the least bad thing. The
     chain can therefore pass over a bad second and land back on the plate.
  3. SPEED LIMITS IN METRES, SEPARATELY FOR EACH DIRECTION. Up to 2.5 m/s up,
     a free fall down (a dropped deadlift), and much less sideways, except for
     the first seconds of a set when a bench bar leaves the hooks.
"""
import math
import cv2
import numpy as np

import v7

PLATE_R_M = v7.PLATE_R_M

K_GLOBAL   = 5        # template peaks kept per scale over the whole frame
UP_MAX     = 2.6      # m/s, bar rising
DOWN_MAX   = 4.5      # m/s, a dropped deadlift falls about this fast
SIDE_MAX   = 1.2      # m/s sideways, between two consecutive frames
SIDE_SKIP_M = 0.2     # and never more than this in all across one skip
SIDE_SKIP  = 0.4      # m/s sideways across frames the path did not visit: an
                      # unseen bar is near where it was last seen. Without this
                      # the path jumped 0.7 m across a bench (2836) onto a red
                      # plate stored on the rack and sat on it for 30 seconds
JITTER_R   = 0.25     # always allow this many plate radii of movement per step
SKIP_COST  = 1.8      # per frame the path does not visit
MAX_SKIP_S = 3.0      # longest stretch the path may jump over
W_NCC      = 0.6      # unary = W_NCC (1 - ncc) + W_CIRC (1 - circ) + W_SHAPE (1 - shape)
W_CIRC     = 0.3      #         + W_ID id
W_SHAPE    = 1.0
W_ID       = 1.0
W_PAIR     = 1.2      # per step, (1 - correlation of the plate's inside between the two frames)
W_DIST     = 0.6      # per metre the bar moves up or down between two visited frames
W_SIDE     = 2.5      # per metre it moves sideways
N_CARRY    = 3        # candidates of the previous frame followed into this one
MIN_NCC    = 0.15
N_ANG      = 64
#
# Measured on 2858, 2861, 3110, 3254 and 3458 against the v8 path where it was
# right: the plate's candidates score shape 0.33 to 0.67 at the 5th percentile
# and about 0.8 at the median; everything else scores 0.10 to 0.42 at the 95th.
# Rim contrast (circ) separates less well (it falls to 0.2 to 0.4 when the plate
# passes in front of dark clothing), and template correlation least of all.


_GCACHE = [None, None]


def _grad(gw):
    """Smoothed frame and its gradient, computed once per frame."""
    if _GCACHE[0] is gw:
        return _GCACHE[1][1], _GCACHE[1][2]
    f = cv2.GaussianBlur(gw.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    _GCACHE[0] = gw; _GCACHE[1] = (f, gx, gy)
    return gx, gy


def _blur(gw):
    _grad(gw)
    return _GCACHE[1][0]


_ANG = np.linspace(0, 2 * math.pi, N_ANG, endpoint=False)
_CA, _SA = np.cos(_ANG), np.sin(_ANG)


def circle_score(gx, gy, cx, cy, R):
    """Median |radial gradient| on the circle of radius R (best of three radii
    near R) around (cx, cy). In-frame samples only; None if under half the rim
    is in frame."""
    H, W = gx.shape
    rr = np.array([0.92, 1.0, 1.08])[:, None] * R
    px = cx + rr * _CA[None, :]; py = cy + rr * _SA[None, :]
    ok = (px >= 1) & (px < W - 1) & (py >= 1) & (py < H - 1)
    if ok[1].sum() < N_ANG * 0.5:
        return None
    xi = np.clip(px, 0, W - 1).astype(np.int32); yi = np.clip(py, 0, H - 1).astype(np.int32)
    rad = np.abs(gx[yi, xi] * _CA[None, :] + gy[yi, xi] * _SA[None, :])
    if ok.all():
        return float(np.median(rad, axis=1).max())
    return float(max(np.median(rad[k][ok[k]]) for k in range(3) if ok[k].any()))


def vote_peaks(gw, R, k=6):
    """Fixed radius circle votes: every strong edge votes for the points R away
    along its gradient, both ways. A plate's rim is round whatever way the
    plate has turned, so this finds it when the template does not: a bar that
    rolls in the hands spins the plate's lettering through 90 degrees in one
    pull, which is what template correlation never survived. Returns a list of
    (x, y, votes) in working pixels."""
    H, W = gw.shape
    gx, gy = _grad(gw)
    mag = np.hypot(gx, gy)
    flat = mag.ravel()
    thr = max(8.0, float(np.partition(flat, int(len(flat) * 0.8))[int(len(flat) * 0.8)]))
    ys, xs = np.nonzero(mag > thr)
    if len(xs) == 0:
        return []
    ux = gx[ys, xs] / mag[ys, xs]; uy = gy[ys, xs] / mag[ys, xs]
    kk = (np.array([0.92, 1.0, 1.08, -0.92, -1.0, -1.08], np.float32) * R)[:, None]
    vx = np.rint(xs[None, :] + kk * ux[None, :]).astype(np.int32).ravel()
    vy = np.rint(ys[None, :] + kk * uy[None, :]).astype(np.int32).ravel()
    ok = (vx >= 0) & (vx < W) & (vy >= 0) & (vy < H)
    acc = np.bincount(vy[ok] * W + vx[ok], minlength=H * W).reshape(H, W).astype(np.float32)
    acc = cv2.GaussianBlur(acc, (0, 0), max(1.0, R * 0.08))
    out = []
    sup = max(3, int(R * 0.5))
    work = acc.copy()
    for _ in range(k):
        _, mx, _, loc = cv2.minMaxLoc(work)
        if mx <= 0:
            break
        work[max(0, loc[1] - sup):loc[1] + sup + 1, max(0, loc[0] - sup):loc[0] + sup + 1] = 0
        out.append((float(loc[0]), float(loc[1]), float(mx)))
    return out


def candidates(gw, gf, tpl_full, R, fine, near=None, near_rad=None, carry=None):
    """Template peaks over the whole frame, the best template match in a
    window around `near`, the strongest fixed radius circles, and the previous
    frame's best candidates followed into this frame by their own appearance
    (carry: list of (x, y, patch) with patch cut from the previous frame).
    Returns a list of (x, y, ncc against the seed) in working pixels."""
    raw = []
    pad = int(R)
    gwp = cv2.copyMakeBorder(gw, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    maps = []
    # the whole frame is searched at the seed's size only; the plate's
    # apparent size changes by a few per cent through a lift, which the other
    # two sizes cover in the window near the bar and in the refinement
    for si, (t_w, t_f) in enumerate(tpl_full):
        if t_w.shape[0] >= gwp.shape[0] or t_w.shape[1] >= gwp.shape[1]:
            continue
        if si != len(tpl_full) // 2:
            if near is None:
                continue
            x0 = int(max(0, near[0] + pad - near_rad - t_w.shape[1]))
            y0 = int(max(0, near[1] + pad - near_rad - t_w.shape[0]))
            x1 = int(min(gwp.shape[1], near[0] + pad + near_rad + t_w.shape[1]))
            y1 = int(min(gwp.shape[0], near[1] + pad + near_rad + t_w.shape[0]))
            win = gwp[y0:y1, x0:x1]
            if win.shape[0] <= t_w.shape[0] or win.shape[1] <= t_w.shape[1]:
                continue
            res = cv2.matchTemplate(win, t_w, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            raw.append((x0 + loc[0] + t_w.shape[1] / 2 - pad, y0 + loc[1] + t_w.shape[0] / 2 - pad, mx, t_f))
            continue
        # the whole frame at half size (four times cheaper, and a plate 28 px
        # across is still unmistakable), each peak then placed at full size
        hp = cv2.resize(gwp, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        ht = cv2.resize(t_w, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        if ht.shape[0] >= hp.shape[0] or ht.shape[1] >= hp.shape[1] or ht.shape[0] < 6:
            continue
        hres = cv2.matchTemplate(hp, ht, cv2.TM_CCOEFF_NORMED)
        work = hres.copy()
        sup = max(2, int(R * 0.22))
        res = np.full((gwp.shape[0] - t_w.shape[0] + 1, gwp.shape[1] - t_w.shape[1] + 1), -1.0, np.float32)
        for _ in range(K_GLOBAL):
            _, mx, _, loc = cv2.minMaxLoc(work)
            if mx < MIN_NCC:
                break
            work[max(0, loc[1] - sup):loc[1] + sup + 1, max(0, loc[0] - sup):loc[0] + sup + 1] = -1
            # place it at full size in a small window
            fx0 = max(0, loc[0] * 2 - 3); fy0 = max(0, loc[1] * 2 - 3)
            fx1 = min(res.shape[1], loc[0] * 2 + 4); fy1 = min(res.shape[0], loc[1] * 2 + 4)
            win = gwp[fy0:fy1 + t_w.shape[0] - 1, fx0:fx1 + t_w.shape[1] - 1]
            if win.shape[0] < t_w.shape[0] or win.shape[1] < t_w.shape[1]:
                continue
            r2 = cv2.matchTemplate(win, t_w, cv2.TM_CCOEFF_NORMED)
            res[fy0:fy0 + r2.shape[0], fx0:fx0 + r2.shape[1]] = r2
            _, m2, _, l2 = cv2.minMaxLoc(r2)
            raw.append((fx0 + l2[0] + t_w.shape[1] / 2 - pad, fy0 + l2[1] + t_w.shape[0] / 2 - pad, m2, t_f))
        maps.append((res, t_w, t_f))
        if near is not None:
            x0 = int(max(0, near[0] + pad - near_rad - t_w.shape[1] / 2))
            y0 = int(max(0, near[1] + pad - near_rad - t_w.shape[0] / 2))
            x1 = int(min(res.shape[1], near[0] + pad + near_rad - t_w.shape[1] / 2 + 1))
            y1 = int(min(res.shape[0], near[1] + pad + near_rad - t_w.shape[0] / 2 + 1))
            if x1 > x0 and y1 > y0:
                sub = res[y0:y1, x0:x1]
                _, mx, _, loc = cv2.minMaxLoc(sub)
                raw.append((x0 + loc[0] + t_w.shape[1] / 2 - pad,
                            y0 + loc[1] + t_w.shape[0] / 2 - pad, mx, t_f))
    raw.sort(key=lambda c: -c[2])
    out = []
    for x, y, s, t_f in raw:
        if any(math.hypot(x - o[0], y - o[1]) < R * 0.4 for o in out):
            continue
        out.append((x, y, float(s)))
        if len(out) >= 8:
            break
    extra = list(vote_peaks(gw, R))
    # the previous frame's candidates, followed by what they looked like a
    # frame ago rather than at the seed: a plate passing in front of the
    # lifter's legs at lockout looks little like the plate on the floor at the
    # tap, and a great deal like itself 130 ms earlier
    for x, y, tpl in (carry or []):
        h, w = tpl.shape
        rad = near_rad or R
        x0 = int(max(0, x - rad - w / 2)); y0 = int(max(0, y - rad - h / 2))
        x1 = int(min(gw.shape[1], x + rad + w / 2)); y1 = int(min(gw.shape[0], y + rad + h / 2))
        win = gw[y0:y1, x0:x1]
        if win.shape[0] <= h or win.shape[1] <= w:
            continue
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx >= 0.4:
            extra.append((x0 + loc[0] + w / 2.0, y0 + loc[1] + h / 2.0, mx))
    # circles and carried positions: their correlation with the seed is read
    # off the maps
    for x, y, _v in extra:
        if any(math.hypot(x - o[0], y - o[1]) < R * 0.3 for o in out):
            continue
        best = 0.0
        for res, t_w, _tf in maps:
            cx = int(round(x + pad - t_w.shape[1] / 2)); cy = int(round(y + pad - t_w.shape[0] / 2))
            x0, y0 = max(0, cx - 2), max(0, cy - 2)
            x1, y1 = min(res.shape[1], cx + 3), min(res.shape[0], cy + 3)
            if x1 > x0 and y1 > y0:
                best = max(best, float(res[y0:y1, x0:x1].max()))
        out.append((x, y, best))
    return out


_SA48 = np.linspace(0, 2 * math.pi, 48, endpoint=False)
_SR = np.linspace(0.8, 1.25, 16)


def shape_score(f, cx, cy, R):
    """Fraction of rays from (cx, cy) whose strongest brightness step between
    0.8R and 1.25R sits within 12% of R (lettering inside the plate sits
    inside that band, so it cannot win). A plate's rim puts the step at R on
    almost every ray whatever its contrast; a knee, a chest or a rack upright
    puts it anywhere. f: the blurred float working frame."""
    H, W = f.shape
    xs = cx + np.outer(np.cos(_SA48), _SR * R)
    ys = cy + np.outer(np.sin(_SA48), _SR * R)
    ok = (xs.min(1) >= 1) & (ys.min(1) >= 1) & (xs.max(1) <= W - 2) & (ys.max(1) <= H - 2)
    if ok.sum() < 20:
        return 0.0
    v = cv2.remap(f, xs[ok].astype(np.float32), ys[ok].astype(np.float32), cv2.INTER_LINEAR)
    g = np.abs(np.gradient(v, axis=1))
    j = np.argmax(g, axis=1)
    m = g[np.arange(len(j)), j]
    hit = (m > 1.5) & (np.abs(_SR[j] - 1.0) <= 0.12)
    return float(hit.mean())


_RF = np.array([0.2, 0.35, 0.5, 0.65, 0.8, 0.92])
_RA = np.linspace(0, 2 * math.pi, 32, endpoint=False)


def ring_profile(f, cx, cy, R):
    """Mean and spread of brightness on six rings inside the plate. It does not
    change when the plate spins, and it tells one plate from another: a blue
    bumper with white lettering and a black iron plate on the rack behind it
    have the same rim but nothing else in common."""
    H, W = f.shape
    xs = (cx + np.outer(_RF * R, np.cos(_RA))).astype(np.float32)
    ys = (cy + np.outer(_RF * R, np.sin(_RA))).astype(np.float32)
    v = cv2.remap(f, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.concatenate([v.mean(1), v.std(1)])


def profile_dist(p, p0):
    if p is None or p0 is None:
        return 0.5
    m, s = p[:6], p[6:]
    m0, s0 = p0[:6], p0[6:]
    # brightness can drift with exposure and the plate's tilt to the light, so
    # compare shapes of the profiles (offset removed) more than levels
    dm = np.mean(np.abs((m - m.mean()) - (m0 - m0.mean()))) / (np.std(m0) + 12.0)
    dl = abs(m.mean() - m0.mean()) / (abs(m0.mean()) * 0.5 + 25.0)
    ds = np.mean(np.abs(s - s0)) / (np.mean(s0) + 8.0)
    return float(min(1.5, 0.5 * dm + 0.3 * dl + 0.4 * ds))


_PG = np.linspace(-0.9, 0.9, 20)
_PX, _PY = np.meshgrid(_PG, _PG)
_PM = (_PX ** 2 + _PY ** 2) <= 0.81


def patch(f, cx, cy, R):
    """The inside of the plate, 20 x 20 samples inside 0.9R, zero mean and
    unit length, so a dot product of two is their correlation."""
    xs = (cx + _PX[_PM] * R).astype(np.float32).reshape(1, -1)
    ys = (cy + _PY[_PM] * R).astype(np.float32).reshape(1, -1)
    v = cv2.remap(f, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
    v = v - v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else v


def crop(gw, cx, cy, R):
    """A square cut around (cx, cy), to follow into the next frame."""
    r = int(round(R * 0.95))
    x0, y0 = int(round(cx)) - r, int(round(cy)) - r
    if x0 < 0 or y0 < 0 or x0 + 2 * r > gw.shape[1] or y0 + 2 * r > gw.shape[0]:
        return None
    return gw[y0:y0 + 2 * r, x0:x0 + 2 * r].copy()


class Frame:
    pass


def score_frame(gw, cands, R, circ0, prof0=None):
    gx, gy = _grad(gw)
    fb = _blur(gw)
    f = Frame()
    xy, ncc, circ, shp, idd = [], [], [], [], []
    for x, y, s in cands:
        c = circle_score(gx, gy, x, y, R)
        cn = 0.0 if c is None else min(1.2, c / max(circ0, 1e-6))
        xy.append((x, y)); ncc.append(s); circ.append(cn); shp.append(shape_score(fb, x, y, R))
        idd.append(profile_dist(ring_profile(fb, x, y, R), prof0) if prof0 is not None else 0.0)
    f.xy = np.array(xy, float).reshape(-1, 2)
    f.ncc = np.array(ncc, float)
    f.circ = np.array(circ, float)
    f.shape = np.array(shp, float)
    f.id = np.array(idd, float)
    f.patch = (np.array([patch(fb, x, y, R) for x, y, _s in cands], np.float32)
               if cands else np.zeros((0, int(_PM.sum())), np.float32))
    f.unary = (W_NCC * (1.0 - np.clip(f.ncc, 0, 1)) + W_CIRC * np.clip(1.0 - f.circ, 0, 1)
               + W_SHAPE * (1.0 - f.shape) + W_ID * f.id)
    return f


def solve(frames, seed_xy, R, efps, side_free_s=4.0, plate_r_m=PLATE_R_M, seed_patch=None):
    """Cheapest physically possible chain of candidates, from the seed.

    frames: list of Frame (frame 0 is the seed frame). Returns a list, one per
    frame, of (x, y, ncc, circ) or None where the path skips the frame."""
    T = len(frames)
    ppm = R / plate_r_m
    G = max(1, int(round(MAX_SKIP_S * efps)))
    INF = 1e18
    cost = [None] * T
    back = [None] * T
    # frame 0: the seed itself, plus any candidate on top of it
    f0 = frames[0]
    seed = np.array(seed_xy, float)
    f0.xy = np.vstack([seed[None, :], f0.xy]) if len(f0.xy) else seed[None, :]
    f0.ncc = np.concatenate([[1.0], f0.ncc]); f0.circ = np.concatenate([[1.0], f0.circ])
    f0.shape = np.concatenate([[1.0], f0.shape]); f0.id = np.concatenate([[0.0], f0.id])
    if seed_patch is not None:
        f0.patch = np.vstack([seed_patch[None, :], f0.patch]) if len(f0.patch) else seed_patch[None, :]
    f0.unary = np.concatenate([[0.0], f0.unary])
    c0 = np.full(len(f0.xy), INF); c0[0] = 0.0
    cost[0] = c0; back[0] = np.full((len(f0.xy), 2), -1, int)
    jit = JITTER_R * R
    for t in range(1, T):
        ft = frames[t]
        n = len(ft.xy)
        if n == 0:
            cost[t] = np.zeros(0); back[t] = np.zeros((0, 2), int); continue
        best = np.full(n, INF); bk = np.full((n, 2), -1, int)
        for k in range(1, min(G, t) + 1):
            p = t - k
            cp = cost[p]
            if cp is None or len(cp) == 0:
                continue
            live = cp < INF / 2
            if not live.any():
                continue
            dt = k / efps
            side = SIDE_MAX if k == 1 else SIDE_SKIP
            d = ft.xy[None, :, :] - frames[p].xy[live][:, None, :]      # (m, n, 2)
            dx = np.abs(d[..., 0]); dy = d[..., 1]                      # +dy is down
            side_lim = side * dt * ppm if k == 1 else min(side * dt, SIDE_SKIP_M) * ppm
            ok = ((dx <= side_lim + jit) &
                  (-dy <= UP_MAX * dt * ppm + jit) &
                  (dy <= DOWN_MAX * dt * ppm + jit))
            v = np.hypot(dx, dy) / ppm / dt
            pc = frames[p].patch[live] @ ft.patch.T                     # (m, n)
            tr = (0.03 * v * v + (W_DIST * np.abs(dy) + W_SIDE * dx) / ppm + SKIP_COST * (k - 1)
                  + W_PAIR * np.clip(1.0 - pc, 0.0, 1.5))
            tot = cp[live][:, None] + tr + ft.unary[None, :]
            tot = np.where(ok, tot, INF)
            arg = np.argmin(tot, axis=0)
            val = tot[arg, np.arange(n)]
            src = np.flatnonzero(live)[arg]
            better = val < best
            best = np.where(better, val, best)
            bk[better, 0] = p; bk[better, 1] = src[better]
        cost[t] = best; back[t] = bk
    # end anywhere, paying for the frames after it
    endt, endi, endc = 0, 0, cost[0][0] + SKIP_COST * (T - 1)
    for t in range(T):
        if cost[t] is None or len(cost[t]) == 0:
            continue
        i = int(np.argmin(cost[t]))
        c = cost[t][i] + SKIP_COST * (T - 1 - t)
        if c < endc:
            endt, endi, endc = t, i, c
    path = [None] * T
    t, i = endt, endi
    while t >= 0 and i >= 0:
        f = frames[t]
        path[t] = (float(f.xy[i, 0]), float(f.xy[i, 1]), float(f.ncc[i]), float(f.circ[i]),
                   float(f.shape[i]))
        p, j = back[t][i]
        t, i = int(p), int(j)
        if t < 0:
            break
    return path


RIM_RATIOS = np.round(np.arange(0.45, 4.65, 0.05), 3)
RIM_OUTER  = 0.70     # a circle this consistent outside the seed's is the rim
RIM_SEED   = 0.45     # the seed's own circle below this was not a plate rim
RIM_INNER  = 0.60


def rim_check(GW, path, R, n=24):
    """Is the seed circle the plate's rim? Asked of the whole path, not of one
    still.

    The seed finder looks at five frames from the first fifth of a second, and
    on a plate with a bright hub the hub is the crisper circle: 2843 seeded on
    a yellow hub a third of the plate's size and reported 1.9 m squats. Once
    the path is known the question is easy, because the rim is the one circle
    of the plate's size that is there in EVERY frame, around the bar, whatever
    is behind it: the gym behind the plate changes from frame to frame as the
    bar moves, so a circle in the background cannot keep scoring. The rim is
    therefore the outermost radius whose shape score, median across frames
    spread along the path, stays high. If the seed's own radius scores low and
    a smaller one scores high, the seed was too big (3459 reached past the
    plate into the rack).

    Returns (ratio to apply to R, median shape at that ratio, profile)."""
    idx = [i for i, p in enumerate(path) if p is not None and i < len(GW)]
    if len(idx) < 8:
        return 1.0, None, None
    pick = idx[::max(1, len(idx) // n)][:n]
    M = []
    for i in pick:
        f = _blur(GW[i]) if _GCACHE[0] is GW[i] else cv2.GaussianBlur(GW[i].astype(np.float32), (0, 0), 1.0)
        M.append([shape_score(f, path[i][0], path[i][1], R * q) for q in RIM_RATIOS])
    med = np.median(np.array(M), axis=0)
    def at(q):
        return float(med[int(np.argmin(np.abs(RIM_RATIOS - q)))])
    s1 = max(at(0.95), at(1.0), at(1.05))
    outer = None
    for k in range(len(RIM_RATIOS) - 2, 0, -1):
        q = RIM_RATIOS[k]
        if q < 1.4:
            break
        if med[k] >= RIM_OUTER and med[k] >= med[k - 1] and med[k] >= med[k + 1]:
            outer = (float(q), float(med[k])); break
    if outer:
        return outer[0], outer[1], med
    if s1 < RIM_SEED:
        for k in range(len(RIM_RATIOS) - 1, -1, -1):
            q = RIM_RATIOS[k]
            if q > 0.92 or q < 0.45:
                continue
            if med[k] >= RIM_INNER and (k == 0 or med[k] >= med[k - 1]) and (k + 1 >= len(med) or med[k] >= med[k + 1]):
                return float(q), float(med[k]), med
    return 1.0, s1, med


def refine_path(path, GF, tpl_full, fine):
    """Sub-pixel positions for the frames the path visits, at twice the working
    resolution. Done once per frame on the chosen point, not on every
    candidate, because it is the one expensive correlation in the tracker."""
    t_f = tpl_full[len(tpl_full) // 2][1]
    out = []
    for i, p in enumerate(path):
        if p is None or i >= len(GF):
            out.append(p); continue
        r = v7._refine(GF[i], p[0], p[1], t_f, fine)
        if r is not None and math.hypot(r[1] - p[0], r[2] - p[1]) <= 2.0 and r[0] >= p[2] - 0.05:
            out.append((r[1], r[2]) + tuple(p[2:]))
        else:
            out.append(p)
    return out


def bridge(path, max_gap):
    """Fill short holes by straight lines; return (points, gaps)."""
    out = [None if p is None else (p[0], p[1]) for p in path]
    gaps = []
    i = 0
    while i < len(out):
        if out[i] is None:
            j = i
            while j < len(out) and out[j] is None:
                j += 1
            a = out[i - 1] if i > 0 else None
            b = out[j] if j < len(out) else None
            if (j - i) <= max_gap and a and b:
                for m in range(i, j):
                    f = (m - i + 1) / (j - i + 1)
                    out[m] = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            else:
                gaps.append((i, j))
            i = j
        else:
            i += 1
    return out, gaps
