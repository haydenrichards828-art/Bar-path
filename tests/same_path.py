"""Does a change to the service leave the bar path exactly as it was?

Runs ONE real clip through the same two calls the annotation editor makes —
/analyze with a shrunk clip plus a five-frame seed strip, then /refine with the
circle the analysis returned — and writes what came back with the timing fields
removed. Run it against the old build and the new one and diff the two files:
any change to a frame, a rep or the seed shows up as a difference.

    # once: make the upload the editor would make (deterministic, reused by both runs)
    python3 tests/same_path.py prepare clip.webm 0.281 0.678 out/
    # then, against each build, listening on :8123 with the same key
    K=<key> python3 tests/same_path.py run out/ before.json
    K=<key> python3 tests/same_path.py run out/ after.json
    cmp before.json after.json

The key is sent the way netlify/functions/barpath.ts sends it, in the
x-barpath-key header, and ALSO as the old api_key form field, so the same
harness drives a build from before the header existed.

`prepare` approximates prepareAnalysisUpload in src/lib/bar-path/engine.ts:
900 px wide at 15 fps for the tracker, and the first five full resolution
frames stacked into one JPEG at quality 95 for the plate finder. It does not
have to match the browser byte for byte; it has to be the SAME bytes for both
builds, which is why it is a separate step.
"""
import json, os, subprocess, sys, urllib.request
import cv2, numpy as np

URL = os.environ.get("URL", "http://127.0.0.1:8123")
# Wall clock and per-run identifiers. Everything else must match exactly.
VOLATILE = {"seconds", "server_s", "job"}


def prepare(src, tx, ty, out):
    os.makedirs(out, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-vf", "scale=900:-2,fps=15",
                    "-c:v", "libx264", "-b:v", "900k", "-an", f"{out}/video.mp4"], check=True)
    cap = cv2.VideoCapture(src)
    frames = []
    while len(frames) < 5:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    cv2.imwrite(f"{out}/seed.jpg", np.vstack(frames), [cv2.IMWRITE_JPEG_QUALITY, 95])
    json.dump({"start_x": float(tx), "start_y": float(ty)}, open(f"{out}/params.json", "w"))


def post(path, fields, files=()):
    parts = [(f'Content-Disposition: form-data; name="{k}"\r\n\r\n{v}').encode() for k, v in fields.items()]
    for name, fn, ct in files:
        parts.append((f'Content-Disposition: form-data; name="{name}"; filename="{os.path.basename(fn)}"\r\n'
                      f'Content-Type: {ct}\r\n\r\n').encode() + open(fn, "rb").read())
    body = b"".join(b"--X\r\n" + p + b"\r\n" for p in parts) + b"--X--\r\n"
    key = os.environ.get("K", "")
    req = urllib.request.Request(URL + path, data=body, headers={
        "Content-Type": "multipart/form-data; boundary=X", "x-barpath-key": key})
    with urllib.request.urlopen(req, timeout=300) as r:
        return [json.loads(l) for l in r.read().decode().splitlines() if l.strip()]


def terminal(lines):
    end = [l for l in lines if l.get("done") or "error" in l]
    if not end:
        raise SystemExit("no terminal message")
    return end[-1]


def run(out, dest):
    key = os.environ.get("K", "")
    params = open(f"{out}/params.json").read()
    a = post("/analyze", {"params": params, "api_key": key},
             [("video", f"{out}/video.mp4", "video/mp4"), ("seed", f"{out}/seed.jpg", "image/jpeg")])
    first = terminal(a)
    result = {"meta": a[0].get("meta"), "progress_lines": sum(1 for l in a if "frame" in l),
              "analyze": {k: v for k, v in first.items() if k not in VOLATILE}}
    if first.get("done") and first.get("seed") and first.get("job"):
        s = first["seed"]
        r = post("/refine", {"job": first["job"], "api_key": key,
                             "circle": json.dumps({"cx": s["cx"], "cy": s["cy"], "r": s["r"]})})
        result["refine"] = {k: v for k, v in terminal(r).items() if k not in VOLATILE}
    json.dump(result, open(dest, "w"), indent=1, sort_keys=True)
    an = result["analyze"]
    print(f"{dest}: verdict={an.get('verdict')} frames={len(an.get('frames') or [])} "
          f"reps={len(an.get('reps') or [])} refine={'yes' if 'refine' in result else 'no'}")


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare(*sys.argv[2:6])
    else:
        run(*sys.argv[2:4])
