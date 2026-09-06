"""Find the plate, and be honest about how sure we are.

Built against Hayden's twenty-clip real-footage set, 6 Sep 2026.

Two independent estimators disagree in useful ways:

  A  the original search, biased toward the largest circle scoring near the
     peak. Handles bumpers with a bright hub. Reaches past the plate into rack
     clutter on bench.
  B  the score peak, with the outward hub-to-rim sweep gated on whether the
     interior is darker than its surround. Nails bench. Under-reads on plates
     whose centre boss is itself a strong circle.

Neither wins outright. But measured across the twenty clips:

  they AGREE (within 12%)  ->  7 of 9 correct
  they DISAGREE            ->  one of the two is correct 6 times out of 6

So disagreement is not a failure, it is a reliable signal that the coach should
be asked which of two circles is their plate. That is one tap, and it converts
the hardest remaining cases from silent errors into a question.
"""
import importlib.util, os

_HERE = os.path.dirname(os.path.abspath(__file__))

def _load(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(_HERE, fn))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

_A = _load("_snap_a", "snap_A.py")
_B = _load("_snap_b", "snap_B.py")

AGREE_TOL = 0.12
CONF_MIN   = 75.0
SCORE_GAP  = 4.0     # this much disparity means they agree by accident
REFUSE_MIN = 50.0    # below this on BOTH estimators there is no plate here
                     # Was min(): one weak estimator vetoed three clips the
                     # other had right. A veto needs both to be weak.


def find_plate(frames, seed_frame, tap):
    """Returns a dict the caller can act on without guessing.

    verdict is one of:
      'ok'      measured, both estimators agree and the fit is strong
      'check'   measured, but show the coach the circle and let them confirm
      'choose'  two plausible circles - ask the coach which one is the plate
      'refuse'  no plate we can measure (cut by the frame, or not there at all)
    """
    H, W = frames[seed_frame].shape
    a = _A.snap_multi(frames, seed_frame, tap)
    b = _B.snap_multi(frames, seed_frame, tap)
    ax, ay, ar, asc, acut = a[0], a[1], a[2], a[3], a[4]
    bx, by, br, bsc, bcut = b[0], b[1], b[2], b[3], b[4]

    agree = abs(ar - br) / max(ar, br) <= AGREE_TOL

    cands = [{"cx": ax, "cy": ay, "r": ar, "score": asc, "cut": acut},
             {"cx": bx, "cy": by, "r": br, "score": bsc, "cut": bcut}]
    live = [c for c in cands if not c["cut"]]

    if max(asc, bsc) < REFUSE_MIN or not live:
        # Nothing that looks like a plate (a bare barbell scored 23), or every
        # candidate we found is cut off by the frame edge. Either way there is
        # no measurement to be had and saying so beats inventing one.
        verdict = "refuse"
    elif len(live) == 1:
        cut_one = [c for c in cands if c["cut"]][0]
        keep = live[0]
        edge = min(keep["cx"], keep["cy"], W - keep["cx"], H - keep["cy"])
        if cut_one["r"] > keep["r"] * 1.8 and edge < keep["r"] * 1.5:
            # A small circle hugging the frame edge, with a much larger cut
            # candidate on the same spot, is the HUB of a plate whose rim is
            # outside the picture. reject-plate-cut-10 is exactly this and used
            # to sail through as a confident 76px measurement.
            verdict = "refuse"
        else:
            cands = [keep, cut_one]
            verdict = "check"
        primary, alternate = cands[0], cands[1]
    elif not agree:
        verdict = "choose"
    elif max(asc, bsc) < CONF_MIN or max(asc, bsc) > min(asc, bsc) * SCORE_GAP:
        # Same radius from both, but one of them barely sees an edge there.
        # dl-dark-lowcontrast-05 agreed on 220 with scores of 11.7 and 136.8,
        # and 220 was 18% wrong. Agreement is only evidence when both are sure.
        verdict = "check"
    else:
        verdict = "ok"
    primary, alternate = cands[0], cands[1]

    return {
        "verdict": verdict,
        "primary":   primary,
        "alternate": alternate,
        "agree": agree,
    }
