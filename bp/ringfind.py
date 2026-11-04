"""A third way to find the plate at the tap: whole circles, counted by rim.

The two estimators in plate.py look for the circle with the strongest edge
near the tap and then argue about which ring of the plate it is. Both search
centres close to the tap, so a tap on the sleeve end of a plate seen from
below (2708: the sleeve end sits a fifth of a radius above the plate's
centre) or a plate cut by the frame (2710, 5 px from the left edge) leaves
them with nothing they will stand behind, and the clip is refused.

This one asks a plainer question of every circle that contains the tap: on
how many of 48 rays out from its centre is the strongest brightness step
right on the circle? A plate's rim puts it there on almost every ray, however
weak its contrast; a background shape puts it there on a few. Rays that
leave the frame do not count against it, so a plate partly out of shot is
still found, and how much of it is in shot is reported, so the caller can
refuse when less than half the rim is visible.

It does not decide between the rings of one plate (hub, insert, rim); it
returns them all, and the tracker's motion check (dp.motion_profile) says
which one is the edge of the thing moving with the bar.
"""
import math
import cv2
import numpy as np

NA = 48
_A = 2 * math.pi * np.arange(NA) / NA
_CA, _SA = np.cos(_A), np.sin(_A)
_QB = np.linspace(0.84, 1.16, 13)            # radial samples across the band
_MID = (len(_QB) - 1) / 2.0 - 0.5            # index of the step at the circle
SHORT = 240.0                                # work with the short side this long
MIN_STEP = 2.0                               # grey levels; weaker is not an edge


def _score(b, cx, cy, r):
    """cx, cy, r: arrays. Returns (hits over all rays, rays in frame) / NA."""
    n = len(cx)
    rr = r[:, None, None] * _QB[None, None, :]
    X = cx[:, None, None] + rr * _CA[None, :, None]
    Y = cy[:, None, None] + rr * _SA[None, :, None]
    H, W = b.shape
    inside = ((X >= 1) & (Y >= 1) & (X < W - 2) & (Y < H - 2)).all(axis=2)
    v = cv2.remap(b, X.reshape(n, -1).astype(np.float32), Y.reshape(n, -1).astype(np.float32),
                  cv2.INTER_LINEAR).reshape(X.shape)
    d = np.abs(np.diff(v, axis=2))
    j = np.argmax(d, axis=2)
    peak = np.take_along_axis(d, j[..., None], 2)[..., 0]
    hit = (np.abs(j - _MID) <= 1.6) & (peak >= MIN_STEP) & inside
    return hit.sum(1) / NA, inside.sum(1) / NA


def rings(gray, tap, rmin_frac=0.035, rmax_frac=0.62):
    """Every ring size with its best circle containing the tap.
    Returns a list of dicts {cx, cy, r, hit, cover} in the caller's pixels,
    one per radius, smallest first."""
    H, W = gray.shape
    k = min(1.0, SHORT / float(min(H, W)))
    g = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else gray
    b = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    tx, ty = tap[0] * k, tap[1] * k
    short = min(b.shape)
    out = []
    for r in np.geomspace(max(6.0, short * rmin_frac), short * rmax_frac, 36):
        st = max(1.0, r * 0.07)
        n = int(math.ceil(0.95 * r / st))
        o = np.arange(-n, n + 1) * st
        CX, CY = np.meshgrid(tx + o, ty + o)
        CX = CX.ravel(); CY = CY.ravel()
        keep = np.hypot(CX - tx, CY - ty) <= 0.95 * r
        CX, CY = CX[keep], CY[keep]
        hit, cov = _score(b, CX, CY, np.full(len(CX), r))
        # rank by hits among rays in frame, but only circles at least half in
        # frame can win
        rel = np.where(cov >= 0.5, hit / np.maximum(cov, 1e-6), 0.0)
        i = int(np.argmax(rel))
        out.append({"cx": float(CX[i] / k), "cy": float(CY[i] / k), "r": float(r / k),
                    "hit": float(rel[i]), "cover": float(cov[i])})
    return out


def refine(gray, c, steps=(0.04, 0.015)):
    """Centre and radius polish of one ring, at up to twice the search scale."""
    H, W = gray.shape
    k = min(1.0, 2 * SHORT / float(min(H, W)))
    g = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else gray
    b = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    cx, cy, r = c["cx"] * k, c["cy"] * k, c["r"] * k
    best = None
    for s in steps:
        o = np.array([-2, -1, 0, 1, 2]) * r * s
        DX, DY, DR = np.meshgrid(o, o, o)
        X = cx + DX.ravel(); Y = cy + DY.ravel(); Rr = r + DR.ravel()
        hit, cov = _score(b, X, Y, Rr)
        rel = np.where(cov >= 0.5, hit / np.maximum(cov, 1e-6), 0.0)
        i = int(np.argmax(rel))
        cx, cy, r = float(X[i]), float(Y[i]), float(Rr[i])
        best = (float(rel[i]), float(cov[i]))
    return {"cx": cx / k, "cy": cy / k, "r": r / k, "hit": best[0], "cover": best[1]}


def candidates(gray, tap, min_hit=0.55):
    """Rings of the plate at the tap, smallest first: local maxima of the
    rim score over radius that clear `min_hit` (fraction of in-frame rays)."""
    rs = rings(gray, tap)
    out = []
    for i, c in enumerate(rs):
        if c["hit"] < min_hit:
            continue
        if i > 0 and c["hit"] < rs[i - 1]["hit"]:
            continue
        if i + 1 < len(rs) and c["hit"] < rs[i + 1]["hit"]:
            continue
        out.append(refine(gray, c))
    return out
