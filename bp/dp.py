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
import warnings
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


def _fgrad(gw):
    """Smoothed frame and its gradient."""
    f = cv2.GaussianBlur(gw.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return f, gx, gy


def _grad(gw):
    f, gx, gy = _fgrad(gw)
    return gx, gy


def _blur(gw):
    return cv2.GaussianBlur(gw.astype(np.float32), (0, 0), 1.0)


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


def vote_peaks(gw, R, k=6, grad=None):
    """Fixed radius circle votes: every strong edge votes for the points R away
    along its gradient, both ways. A plate's rim is round whatever way the
    plate has turned, so this finds it when the template does not: a bar that
    rolls in the hands spins the plate's lettering through 90 degrees in one
    pull, which is what template correlation never survived. Returns a list of
    (x, y, votes) in working pixels."""
    H, W = gw.shape
    gx, gy = grad if grad is not None else _grad(gw)
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


def candidates(gw, gf, tpl_full, R, fine=None, near=None, near_rad=None, carry=None,
               grad=None, want_maps=False):
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
        maps.append((res, t_w, t_f, gwp))
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
    extra = list(vote_peaks(gw, R, grad=grad))
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
        out.append((x, y, _ncc_at(maps, x, y, pad)))
    return (out, maps) if want_maps else out


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


class Ctx:
    """What one frame needs from the parallel stage."""
    pass


def prepare(gw, gf, tpl_full, R):
    """The part of a frame's work that needs nothing from the frame before it:
    the smoothed frame and its gradient, template peaks over the whole frame,
    and the circle votes. Safe to run on several frames at once (OpenCV and the
    big numpy calls release the interpreter lock)."""
    c = Ctx()
    c.gw = gw; c.gf = gf
    c.fb, c.gx, c.gy = _fgrad(gw)
    c.cands, c.maps = candidates(gw, gf, tpl_full, R, grad=(c.gx, c.gy), want_maps=True)
    return c


def finish(c, R, circ0, prof0, carry=None, carry_rad=None):
    """The sequential part: follow the previous frame's best candidates into
    this one, then score everything at once."""
    out = list(c.cands)
    pad = int(R)
    for x, y, tpl in (carry or []):
        h, w = tpl.shape
        rad = carry_rad or R
        x0 = int(max(0, x - rad - w / 2)); y0 = int(max(0, y - rad - h / 2))
        x1 = int(min(c.gw.shape[1], x + rad + w / 2)); y1 = int(min(c.gw.shape[0], y + rad + h / 2))
        win = c.gw[y0:y1, x0:x1]
        if win.shape[0] <= h or win.shape[1] <= w:
            continue
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx < 0.4:
            continue
        px, py = x0 + loc[0] + w / 2.0, y0 + loc[1] + h / 2.0
        if any(math.hypot(px - o[0], py - o[1]) < R * 0.3 for o in out):
            continue
        out.append((px, py, _ncc_at(c.maps, px, py, pad)))
    return score_frame(c.gw, out, R, circ0, prof0, fgr=(c.fb, c.gx, c.gy))


def _ncc_at(maps, x, y, pad):
    """Correlation with the seed template at (x, y), best over a 5 x 5 px
    neighbourhood. The whole frame map is only filled in around its own peaks
    (the search runs at half size), so a candidate that came from the circle
    votes or from the previous frame is correlated here, on demand. Reading the
    unfilled map instead gave every such candidate a correlation of 0, which
    is how 3106 lost the bottom of a squat: the plate there was a perfect
    circle that the template, built against a pale wall, did not rank."""
    best = 0.0
    for m in maps:
        res, t_w = m[0], m[1]
        cx = int(round(x + pad - t_w.shape[1] / 2)); cy = int(round(y + pad - t_w.shape[0] / 2))
        x0, y0 = max(0, cx - 2), max(0, cy - 2)
        x1, y1 = min(res.shape[1], cx + 3), min(res.shape[0], cy + 3)
        if x1 <= x0 or y1 <= y0:
            continue
        if len(m) > 3 and res[y0:y1, x0:x1].min() <= -1.0:
            gwp = m[3]
            win = gwp[y0:y1 + t_w.shape[0] - 1, x0:x1 + t_w.shape[1] - 1]
            if win.shape[0] >= t_w.shape[0] and win.shape[1] >= t_w.shape[1]:
                r2 = cv2.matchTemplate(win, t_w, cv2.TM_CCOEFF_NORMED)
                res[y0:y0 + r2.shape[0], x0:x0 + r2.shape[1]] = r2
        best = max(best, float(res[y0:y1, x0:x1].max()))
    return best


def _circle_v(gx, gy, X, Y, R):
    """circle_score for many centres at once (None -> nan)."""
    H, W = gx.shape
    rr = np.array([0.92, 1.0, 1.08])[None, :, None] * R
    px = X[:, None, None] + rr * _CA[None, None, :]
    py = Y[:, None, None] + rr * _SA[None, None, :]
    ok = (px >= 1) & (px < W - 1) & (py >= 1) & (py < H - 1)
    xi = np.clip(px, 0, W - 1).astype(np.int32); yi = np.clip(py, 0, H - 1).astype(np.int32)
    rad = np.abs(gx[yi, xi] * _CA[None, None, :] + gy[yi, xi] * _SA[None, None, :])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.where(ok, rad, np.nan), axis=2)        # (n, 3)
        out = np.nanmax(np.where(ok.any(2), med, np.nan), axis=1)
    out[ok[:, 1, :].sum(1) < N_ANG * 0.5] = np.nan
    return out


def _shape_v(f, X, Y, R):
    """shape_score for many centres at once."""
    H, W = f.shape
    n = len(X)
    xs = X[:, None, None] + (np.cos(_SA48)[:, None] * (_SR * R)[None, :])[None]
    ys = Y[:, None, None] + (np.sin(_SA48)[:, None] * (_SR * R)[None, :])[None]
    ok = (xs.min(2) >= 1) & (ys.min(2) >= 1) & (xs.max(2) <= W - 2) & (ys.max(2) <= H - 2)
    v = cv2.remap(f, xs.reshape(n * 48, -1).astype(np.float32), ys.reshape(n * 48, -1).astype(np.float32),
                  cv2.INTER_LINEAR).reshape(n, 48, -1)
    g = np.abs(np.gradient(v, axis=2))
    j = np.argmax(g, axis=2)
    m = np.take_along_axis(g, j[..., None], axis=2)[..., 0]
    hit = (m > 1.5) & (np.abs(_SR[j] - 1.0) <= 0.12) & ok
    cnt = ok.sum(1)
    return np.where(cnt >= 20, hit.sum(1) / np.maximum(cnt, 1), 0.0)


def _shape_radii(f, cx, cy, Rs):
    """shape_score at one centre for many radii, in one remap."""
    H, W = f.shape
    nr = len(Rs)
    rad = Rs[:, None, None] * _SR[None, None, :]                       # (nr, 1, 16)
    xs = cx + np.cos(_SA48)[None, :, None] * rad                       # (nr, 48, 16)
    ys = cy + np.sin(_SA48)[None, :, None] * rad
    ok = (xs.min(2) >= 1) & (ys.min(2) >= 1) & (xs.max(2) <= W - 2) & (ys.max(2) <= H - 2)
    v = cv2.remap(f, xs.reshape(nr * 48, -1).astype(np.float32), ys.reshape(nr * 48, -1).astype(np.float32),
                  cv2.INTER_LINEAR).reshape(nr, 48, -1)
    g = np.abs(np.gradient(v, axis=2))
    j = np.argmax(g, axis=2)
    m = np.take_along_axis(g, j[..., None], axis=2)[..., 0]
    hit = (m > 1.5) & (np.abs(_SR[j] - 1.0) <= 0.12) & ok
    cnt = ok.sum(1)
    return np.where(cnt >= 20, hit.sum(1) / np.maximum(cnt, 1), 0.0)


def _ring_v(f, X, Y, R):
    n = len(X)
    xs = (X[:, None, None] + (np.outer(_RF * R, np.cos(_RA)))[None]).astype(np.float32)
    ys = (Y[:, None, None] + (np.outer(_RF * R, np.sin(_RA)))[None]).astype(np.float32)
    v = cv2.remap(f, xs.reshape(n * 6, -1), ys.reshape(n * 6, -1), cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REPLICATE).reshape(n, 6, -1)
    return np.concatenate([v.mean(2), v.std(2)], axis=1)


def _profile_dist_v(P, p0):
    m, s_ = P[:, :6], P[:, 6:]
    m0, s0 = p0[:6], p0[6:]
    dm = np.mean(np.abs((m - m.mean(1, keepdims=True)) - (m0 - m0.mean())), axis=1) / (np.std(m0) + 12.0)
    dl = np.abs(m.mean(1) - m0.mean()) / (abs(m0.mean()) * 0.5 + 25.0)
    ds = np.mean(np.abs(s_ - s0), axis=1) / (np.mean(s0) + 8.0)
    return np.minimum(1.5, 0.5 * dm + 0.3 * dl + 0.4 * ds)


def _patch_v(f, X, Y, R):
    xs = (X[:, None] + (_PX[_PM] * R)[None, :]).astype(np.float32)
    ys = (Y[:, None] + (_PY[_PM] * R)[None, :]).astype(np.float32)
    v = cv2.remap(f, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    v = v - v.mean(1, keepdims=True)
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(nrm > 1e-6, v / np.maximum(nrm, 1e-6), v).astype(np.float32)


def score_frame(gw, cands, R, circ0, prof0=None, fgr=None):
    """Every candidate of one frame scored at once (one remap per measure),
    which took the scoring from about 6 ms a frame to under 1."""
    fb, gx, gy = fgr if fgr is not None else _fgrad(gw)
    f = Frame()
    n = len(cands)
    if n == 0:
        f.xy = np.zeros((0, 2)); f.ncc = f.circ = f.shape = f.id = f.unary = np.zeros(0)
        f.patch = np.zeros((0, int(_PM.sum())), np.float32)
        return f
    f.xy = np.array([(c[0], c[1]) for c in cands], float)
    f.ncc = np.array([c[2] for c in cands], float)
    X, Y = f.xy[:, 0], f.xy[:, 1]
    c = _circle_v(gx, gy, X, Y, R)
    f.circ = np.where(np.isnan(c), 0.0, np.minimum(1.2, c / max(circ0, 1e-6)))
    f.shape = _shape_v(fb, X, Y, R)
    f.id = _profile_dist_v(_ring_v(fb, X, Y, R), prof0) if prof0 is not None else np.zeros(n)
    f.patch = _patch_v(fb, X, Y, R)
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
    # All the frames a step may come from (the last G) are scored in one go:
    # their live candidates are stacked, nearest frame first, so ties go to the
    # shortest step exactly as the frame by frame loop this replaces did.
    for t in range(1, T):
        ft = frames[t]
        n = len(ft.xy)
        if n == 0:
            cost[t] = np.zeros(0); back[t] = np.zeros((0, 2), int); continue
        XY, CP, PT, KK, PF, PI = [], [], [], [], [], []
        for k in range(1, min(G, t) + 1):
            p = t - k
            cp = cost[p]
            if cp is None or len(cp) == 0:
                continue
            live = np.flatnonzero(cp < INF / 2)
            if len(live) == 0:
                continue
            XY.append(frames[p].xy[live]); CP.append(cp[live]); PT.append(frames[p].patch[live])
            KK.append(np.full(len(live), k)); PF.append(np.full(len(live), p)); PI.append(live)
        if not XY:
            cost[t] = np.full(n, INF); back[t] = np.full((n, 2), -1, int); continue
        XY = np.concatenate(XY); CP = np.concatenate(CP); PT = np.concatenate(PT)
        KK = np.concatenate(KK).astype(float); PF = np.concatenate(PF); PI = np.concatenate(PI)
        dt = (KK / efps)[:, None]
        d = ft.xy[None, :, :] - XY[:, None, :]                          # (M, n, 2)
        dx = np.abs(d[..., 0]); dy = d[..., 1]                          # +dy is down
        one = (KK == 1)[:, None]
        side_lim = np.where(one, SIDE_MAX * dt * ppm, np.minimum(SIDE_SKIP * dt, SIDE_SKIP_M) * ppm)
        ok = ((dx <= side_lim + jit) &
              (-dy <= UP_MAX * dt * ppm + jit) &
              (dy <= DOWN_MAX * dt * ppm + jit))
        v = np.hypot(dx, dy) / ppm / dt
        pc = PT @ ft.patch.T                                            # (M, n)
        tr = (0.03 * v * v + (W_DIST * np.abs(dy) + W_SIDE * dx) / ppm
              + SKIP_COST * (KK[:, None] - 1) + W_PAIR * np.clip(1.0 - pc, 0.0, 1.5))
        tot = np.where(ok, CP[:, None] + tr + ft.unary[None, :], INF)
        arg = np.argmin(tot, axis=0)
        val = tot[arg, np.arange(n)]
        bk = np.full((n, 2), -1, int)
        good = val < INF / 2
        bk[good, 0] = PF[arg[good]]; bk[good, 1] = PI[arg[good]]
        cost[t] = np.where(good, val, INF); back[t] = bk
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
        f = _blur(GW[i])
        M.append(_shape_radii(f, path[i][0], path[i][1], R * RIM_RATIOS))
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


MP_Q    = np.round(np.arange(0.40, 3.05, 0.05), 3)   # ring sizes tried, x seed radius
MP_ANG  = 2 * math.pi * np.arange(72) / 72
MP_IN   = np.array([0.80, 0.84, 0.88, 0.92])          # just inside a ring, x ring
MP_OUT  = np.array([1.08, 1.12, 1.16, 1.20])          # just outside it
MP_RING = 0.55     # shape score, median across frames: a ring is there
MP_MOVE = 0.50     # fraction of directions where the ring is a motion edge
MP_N    = 32


def _spread(path, idx, n, R):
    """Frames whose bar positions are as far apart as possible (greedy
    farthest point). A deadlift spends most of its clip on the floor, so
    frames evenly spaced in time are mostly the same picture."""
    P = np.array([[path[i][0], path[i][1]] for i in idx], float)
    pick = [0]
    d = np.hypot(P[:, 0] - P[0, 0], P[:, 1] - P[0, 1])
    while len(pick) < min(n, len(idx)):
        j = int(np.argmax(d))
        if d[j] < R * 0.04:
            break
        pick.append(j)
        d = np.minimum(d, np.hypot(P[:, 0] - P[j, 0], P[:, 1] - P[j, 1]))
    if len(pick) < n:                                  # top up evenly in time
        for j in np.linspace(0, len(idx) - 1, n).astype(int):
            if len(pick) >= n:
                break
            if j not in pick:
                pick.append(int(j))
    return sorted(idx[j] for j in pick)


def motion_profile(GW, path, R, n=MP_N):
    """Which circle around the tracked point is the edge of the thing that
    MOVES WITH THE BAR?

    Line up the frames on the tracked point. Everything fixed to the bar (the
    whole plate face, hub, lettering and rim) then stands still from frame to
    frame, and everything else (the gym behind, which the plate passes in front
    of) changes. So just inside the rim the picture is the same in every
    frame, and just outside it is not. Inside a hub, both sides of the hub's
    edge are plate, so a hub is never a motion edge, however crisp it is; and
    a circle in the background is background on both sides. That is the
    question the seed finder cannot answer from five stills: it is how 2919
    (seeded on a green plate's yellow insert) and 3459 (seeded on a circle
    reaching into the rack) give themselves away.

    For each ring size q (x R): `ring` is the median shape score across frames
    (is there a circle edge there), `move` the fraction of directions where
    the brightness just outside the ring varies across frames much more than
    just inside it, `cover` the fraction of directions in frame.
    Returns (MP_Q, ring, move, cover, moved) or None; `moved` is how far the
    chosen frames spread, in R."""
    idx = [i for i, p in enumerate(path) if p is not None and i < len(GW)]
    if len(idx) < 10:
        return None
    pick = _spread(path, idx, n, R)
    P = np.array([[path[i][0], path[i][1]] for i in pick], float)
    moved = float(np.hypot(P[:, 0] - P[:, 0].mean(), P[:, 1] - P[:, 1].mean()).max() / R)
    nq = len(MP_Q)
    rin = (np.outer(MP_Q, MP_IN).ravel() * R)
    rout = (np.outer(MP_Q, MP_OUT).ravel() * R)
    ca, sa = np.cos(MP_ANG), np.sin(MP_ANG)
    SI, SO, SH = [], [], []
    for i in pick:
        f = _blur(GW[i]); H, W = f.shape
        cx, cy = path[i][0], path[i][1]
        for rr, acc in ((rin, SI), (rout, SO)):
            X = (cx + np.outer(ca, rr)).astype(np.float32)
            Y = (cy + np.outer(sa, rr)).astype(np.float32)
            v = cv2.remap(f, X, Y, cv2.INTER_LINEAR)
            v[(X < 0) | (Y < 0) | (X > W - 1) | (Y > H - 1)] = np.nan
            acc.append(v.reshape(len(ca), nq, 4))
        SH.append(_shape_radii(f, cx, cy, R * MP_Q))
    SI = np.array(SI); SO = np.array(SO)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        si = np.nanmedian(np.nanstd(SI, 0), -1)            # directions x rings
        so = np.nanmedian(np.nanstd(SO, 0), -1)
    valid = (np.sum(~np.isnan(SI[..., 0]), 0) >= 8) & (np.sum(~np.isnan(SO[..., 0]), 0) >= 8)
    jump = valid & (so > 1.5 * si + 3.0)
    move = jump.sum(0) / np.maximum(1, valid.sum(0))
    cover = valid.mean(0)
    ring = np.median(np.array(SH), 0)
    return MP_Q, ring, move, cover, moved


def motion_rim(prof):
    """Read motion_profile: the rim is the OUTERMOST ring that is both a circle
    edge and a motion edge. Returns (q, ring, move) for it, or None."""
    if prof is None:
        return None
    Q, ring, move, cover, moved = prof
    best = None
    for i in range(len(Q)):
        if ring[i] < MP_RING:
            continue
        if (i > 0 and ring[i] < ring[i - 1]) or (i + 1 < len(Q) and ring[i] < ring[i + 1]):
            continue
        # the jump in what moves can sit one step inside the edge: motion
        # blur on a fast rep smears the outer rim (2960, original file)
        mv = max(move[i], move[i - 1]) if i > 0 else move[i]
        if mv >= MP_MOVE and cover[i] >= 0.5:
            best = (float(Q[i]), float(ring[i]), float(mv))
    return best


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
