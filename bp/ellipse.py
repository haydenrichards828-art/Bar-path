"""Measure the plate as an ELLIPSE, not a circle.

A plate is a circle, so any ovalness in the picture is the camera not being
square to the bar, and the amount of it is a number the coach can act on.
It also matters for the maths: the scale that turns pixels into centimetres of
bar travel is the plate's VERTICAL half height, never an average radius.

Why the vertical half height and nothing else:

  Camera round in front of or behind the lifter. The plate squashes sideways,
  its height is untouched, and so is the bar's vertical travel. A circle fit
  lands between the two half axes, so it is SMALLER than the true half height,
  the scale comes out short, and every centimetre reads too long.

  Camera too high or too low. The plate squashes vertically AND the bar's
  vertical travel foreshortens by exactly the same cosine, so those cancel out.
  But a circle fit now lands ABOVE the true half height, the scale comes out
  long, and range of motion reads short.

Both are fixed by scaling off the half height, and only that.
"""
import math
import cv2
import numpy as np

N_ANG = 180
BAND  = (0.84, 1.16)     # search for the rim within this much of the hint
N_RAD = 48
MIN_G = 1.5              # a gradient weaker than this is not a rim


def rim_points(gray, cx, cy, r, n_ang=N_ANG):
    """The radius of the strongest brightness step, one per angle."""
    rs = np.linspace(r * BAND[0], r * BAND[1], N_RAD)
    pts = []
    for k in range(n_ang):
        a = 2 * math.pi * k / n_ang
        ca, sa = math.cos(a), math.sin(a)
        xs = cx + rs * ca
        ys = cy + rs * sa
        ok = (xs >= 1) & (ys >= 1) & (xs < gray.shape[1] - 1) & (ys < gray.shape[0] - 1)
        if ok.sum() < 20:
            continue
        v = cv2.remap(gray, xs[ok].astype(np.float32).reshape(-1, 1),
                      ys[ok].astype(np.float32).reshape(-1, 1),
                      cv2.INTER_LINEAR).ravel().astype(np.float32)
        g = np.abs(np.gradient(v))
        j = int(np.argmax(g))
        if g[j] < MIN_G or j <= 1 or j >= len(g) - 2:
            continue        # a peak sitting on the edge of the band is not a rim
        pts.append((rs[ok][j] * ca, rs[ok][j] * sa, g[j]))
    return pts


def fit(pts, rounds=3):
    """Least squares (x/a)^2 + (y/b)^2 = 1 about the given centre, axes aligned
    to the frame, weighted by edge strength and trimmed for outliers.
    Returns (half_width, half_height, points_kept, points_offered)."""
    if len(pts) < 24:
        return None
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    w = np.array([p[2] for p in pts]); w = w / w.max()
    keep = np.ones(len(x), bool)
    out = None
    for _ in range(rounds):
        A = np.stack([x[keep] ** 2, y[keep] ** 2], 1) * w[keep, None]
        b = np.ones(int(keep.sum())) * w[keep]
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None
        if sol[0] <= 0 or sol[1] <= 0:
            return None
        a_, b_ = 1 / math.sqrt(sol[0]), 1 / math.sqrt(sol[1])
        res = np.abs((x / a_) ** 2 + (y / b_) ** 2 - 1.0)
        med = float(np.median(res[keep]))
        mad = float(np.median(np.abs(res[keep] - med))) + 1e-9
        nk = res < med + 2.5 * mad
        out = (a_, b_, int(keep.sum()), len(x))
        if nk.sum() < 24:
            break
        keep = nk
    return out


def measure(gray, cx, cy, r):
    """(half_width, half_height, kept, offered) at full resolution, or None."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    return fit(rim_points(g, cx, cy, r))


def off_square_deg(a, b):
    """How far the camera is from square to the bar, in degrees."""
    if not a or not b:
        return 0.0
    ratio = min(a, b) / max(a, b)
    return math.degrees(math.acos(max(0.0, min(1.0, ratio))))
