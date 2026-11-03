"""get_rotation reads both ways a clip records its orientation, with the
right sign for each. Run: python tests/test_rotation.py"""
import json, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main

def probe(side=None, tag=None):
    st = {}
    if side is not None:
        st["side_data_list"] = [{"side_data_type": "Display Matrix", "rotation": side}]
    if tag is not None:
        st["tags"] = {"rotate": str(tag)}
    out = json.dumps({"streams": [st]})
    return lambda *a, **k: types.SimpleNamespace(stdout=out, returncode=0)

cases = [
    (dict(side=-90), 270),    # iPhone portrait, ffmpeg 5+: turn 90 clockwise
    (dict(tag=90), 270),      # the same clip on ffmpeg 4: tag 90 = 90 clockwise
    (dict(side=90), 90),
    (dict(tag=270), 90),
    (dict(side=180), 180),
    (dict(tag=180), 180),
    (dict(side=-90, tag=90), 270),
    (dict(), 0),
]
bad = 0
real = main.subprocess.run
for kw, want in cases:
    main.subprocess.run = probe(**kw)
    got = main.get_rotation("x.mp4")
    if got != want:
        bad += 1; print("FAIL", kw, "got", got, "want", want)
main.subprocess.run = real
print("rotation:", len(cases) - bad, "of", len(cases), "pass")
sys.exit(1 if bad else 0)
