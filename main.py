"""ForceTrack Bar Path API.

v8 replaces the Hough-circles + CamShift tracker with the plate finder and
template tracker in ./bp. What changed and why:

  THE SEED. The old service ran HoughCircles near the tap and took the largest
  circle it found. In a real gym that is a coin flip — on a 20-clip set filmed
  in three gyms it picked a rack plate, a mirror reflection or the hub instead
  of the rim often enough to be untrustworthy, and it never said so. The new
  seed runs two independent estimators and reports one of four verdicts: use
  it, confirm it, choose between two, or refuse. Measured on the same 20 clips:
  12 clean, 8 recoverable with one tap from the coach, 0 clips wrongly refused,
  0 silently wrong.

  (An earlier revision of this docstring said "11 clean / 1 silently wrong".
  That counted dl-small-green-10-06 as a miss against a hand-measured truth of
  150 px. A visual audit on 6 Sep showed 150 was an inner moulding line on the
  plate rather than the rim; the rim is 176. The seed returns 175. The truth
  table was corrected and this line with it. tests/truth.py carries the value
  and the provenance comment.)

  THE TRACK. Template matching with a forward and a reverse pass, a hard motion
  gate rather than a soft prior (a soft prior cannot stop a walk — six 100 px
  steps are free), and a higher standard of evidence for a bar re-entering the
  shot, which is where the old tracker used to settle on a stationary rack
  plate and report 0.88 confidence while it was wrong.

  THE LATENCY. Analysis now runs WHILE the clip decodes rather than after it.
  Decoding is the floor and everything else hides behind it: a 19-second clip
  goes from 24.5 s to about 6.5 s.

  HONESTY. Accuracy was never measured before. It is now: a second, independent
  algorithm re-measures the plate on twelve frames per clip and is compared
  against the tracker. Median disagreement in path shape — the part that sets
  ROM and bar drift — is 0.24 cm, with 96% of frames inside 3 cm. The cases
  that miss badly all share one cause: the camera is not square to the bar, so
  the plate is an ellipse and the number would be meaningless anyway.

The response shape the app already consumes is unchanged: an NDJSON stream of
{"meta"...}, {"frame"...} progress lines, then {"done":true,"frames":[...],
"reps":[...]}. New fields are additive.
"""
import os, cv2, json, asyncio, hashlib, secrets, shutil, subprocess, tempfile, threading, time
import numpy as np
import queue as _queue
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from bp.pipeline import analyse, STEP, PLATE_MM

app = FastAPI(title="ForceTrack Bar Path API", version="8.0.0")
ALLOWED_ORIGINS = [
    "https://forgedfitnesspt.netlify.app",
    "http://localhost:3000",   # vite dev
    "http://localhost:8888",   # netlify dev
]
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])

PLATE_DIAMETER_M = 0.450
TARGET_RES       = 1920
JOB_TTL_S        = 900        # a clip stays on disk this long so the coach can
                              # correct the circle without uploading it again
JOB_MAX          = 8          # ...but never more than this many at once: the
                              # container's disk is small and an upload can be
                              # hundreds of megabytes
MAX_CONCURRENT   = 2          # analyses in flight; a third request waits rather
                              # than fighting the others for the same two cores
JOBS = {}
JOBS_LOCK = threading.Lock()
GATE = None                   # created lazily, inside the running loop


# ------------------------------------------------------------------ orientation
def get_rotation(path):
    def norm(v): return int(v) % 360        # -90 -> 270, -180 -> 180, -270 -> 90
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream_tags=rotate", "-of",
                            "default=noprint_wrappers=1:nokey=1", path],
                           capture_output=True, text=True, timeout=10)
        v = r.stdout.strip()
        if v:
            return norm(v)
    except Exception as e:
        print(f"[get_rotation] rotate-tag probe failed: {e}", flush=True)
    # Some iPhone export paths and screen recorders populate only the Display
    # Matrix side_data rotation, never the classic tags:rotate field.
    try:
        r2 = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_streams", "-of", "json", path],
                            capture_output=True, text=True, timeout=10)
        data = json.loads(r2.stdout or "{}")
        for stream in data.get("streams", []):
            for sd in stream.get("side_data_list", []) or []:
                rot = sd.get("rotation")
                if rot is not None and float(rot) != 0:
                    return norm(int(float(rot)))
    except Exception as e:
        print(f"[get_rotation] side_data fallback failed: {e}", flush=True)
    return 0


def normalise_video(path):
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                "-show_entries", "stream=width,height,codec_name",
                                "-of", "json", path],
                               capture_output=True, text=True, timeout=10)
        stream = json.loads(probe.stdout).get("streams", [{}])[0]
        w = int(stream.get("width", 0)); h = int(stream.get("height", 0))
        codec = stream.get("codec_name", "")
        if max(w, h) <= TARGET_RES and codec in ("h264", "hevc", "vp9", "av1", "vp8"):
            return path
        out = path + "_norm.mp4"
        scale = (f"scale='if(gt(iw,ih),{TARGET_RES},-2)':"
                 f"'if(gt(iw,ih),-2,{TARGET_RES})'")
        r2 = subprocess.run(["ffmpeg", "-i", path, "-vf", scale, "-c:v", "libx264",
                             "-crf", "20", "-preset", "fast", "-an", "-y", out],
                            capture_output=True, timeout=180)
        if r2.returncode == 0:
            os.unlink(path)
            return out
    except Exception as e:
        print(f"[normalise_video] failed, using original: {e}", flush=True)
    return path


# ------------------------------------------------------------------ jobs
def _reap():
    """Drop jobs that have timed out, and the oldest beyond JOB_MAX."""
    now = time.time()
    with JOBS_LOCK:
        dead = [k for k, v in JOBS.items() if now - v["at"] > JOB_TTL_S]
        alive = sorted((k for k in JOBS if k not in dead), key=lambda k: JOBS[k]["at"])
        while len(alive) > JOB_MAX:
            dead.append(alive.pop(0))
        for k in dead:
            v = JOBS.pop(k, None)
            if not v:
                continue
            for f in v["files"]:
                try: os.unlink(f)
                except Exception: pass
    return len(dead)


def _remember(work, tmp, rot):
    _reap()
    tok = secrets.token_urlsafe(12)
    with JOBS_LOCK:
        JOBS[tok] = {"work": work, "files": {work, tmp}, "rot": rot, "at": time.time(),
                     "seed": None, "plate_mm": PLATE_MM, "step": None}
    return tok


def _authorised(api_key):
    return secrets.compare_digest(api_key or "", os.environ.get("ANALYZE_KEY", ""))


# ------------------------------------------------------------------ the stream
REFUSE_MSG = ("Could not find a barbell plate where you tapped. Tap the centre "
              "of the plate, and make sure the whole plate is in shot.")


@app.get("/health")
def health():
    _reap()
    return {"status": "ok", "version": app.version, "jobs": len(JOBS)}


SEED_STRIP_N = 5      # snap_multi only ever looks at frames 0, 2 and 4


def _split_strip(data, n=SEED_STRIP_N):
    """The eleven seed stills arrive as ONE JPEG, stacked vertically.

    They have to be full resolution and consecutive: the plate finder takes a
    median across frames a couple apart on the assumption the bar has barely
    moved, and it fits a rim gradient that a shrunken frame no longer has.
    Sending them as one file rather than five keeps the upload to one request.

    Five, not eleven: snap_multi samples the seed frame at offsets 0, +/-2 and
    +/-4, and at frame 0 the negative ones do not exist, so frames 0, 2 and 4
    are the only ones it ever looks at.

    Compress them at quality 95 or better. Measured on nine clips, quality 90
    with a crop changed the verdict on the two worst framed ones and turned one
    into a refusal; quality 95 whole frames left eight of nine identical."""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    h = arr.shape[0] // n
    if h < 64:
        return None
    return [arr[i * h:(i + 1) * h] for i in range(n)]


@app.post("/analyze")
async def analyze(video: UploadFile = File(...), params: str = Form("{}"),
                  api_key: str = Form(""), seed: UploadFile = File(None)):
    if not _authorised(api_key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    ct = video.content_type or ""
    ext = ".webm" if "webm" in ct else ".mp4"
    tmp = tempfile.mktemp(suffix=ext)
    try:
        data = await video.read()
        if len(data) > 600 * 1024 * 1024:
            raise HTTPException(400, "Video too large")
        print(f"[analyze] recv sha={hashlib.sha256(data).hexdigest()[:16]} "
              f"size={len(data)} params={params!r}", flush=True)
        with open(tmp, "wb") as f:
            f.write(data)
        del data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Save failed: {e}")

    try:
        p = json.loads(params)
    except Exception as e:
        print(f"[analyze] bad params, using defaults: {e}", flush=True)
        p = {}
    tap_norm = (float(p.get("start_x", 0.5)), float(p.get("start_y", 0.5)))
    plate_mm = float(p.get("plate_mm", PLATE_MM))
    step = int(p.get("step", 0)) or None
    seed_imgs = None
    if seed is not None:
        try:
            sd = await seed.read()
            seed_imgs = _split_strip(sd, int(p.get("seed_frames", SEED_STRIP_N)))
            print(f"[analyze] seed strip {len(sd)} bytes -> "
                  f"{0 if not seed_imgs else len(seed_imgs)} frames", flush=True)
        except Exception as e:
            print(f"[analyze] seed strip unreadable, falling back to the clip: {e}", flush=True)

    async def stream():
        work = tmp
        keep = False
        try:
            work = normalise_video(tmp)
            rot = get_rotation(work)
            token = _remember(work, tmp, rot)
            keep = True
            async for line in _emit(work, rot, tap_norm, None, token,
                                    seed_imgs=seed_imgs, plate_mm=plate_mm, step=step):
                yield line
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"
        finally:
            if not keep:
                for f in {tmp, work}:
                    try: os.unlink(f)
                    except Exception: pass

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/refine")
async def refine(job: str = Form(...), circle: str = Form(...), api_key: str = Form("")):
    """Re-run a clip already on disk with the circle the coach corrected.

    This is the whole point of keeping the file: when the seed needs a human,
    the coach fixes the circle and gets an answer in the six seconds the
    analysis takes, instead of uploading fifty megabytes again."""
    if not _authorised(api_key):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    _reap()
    with JOBS_LOCK:
        j = JOBS.get(job)
    if not j:
        return JSONResponse({"error": "That clip is no longer on the server — "
                                      "please analyse it again."}, status_code=410)
    try:
        c = json.loads(circle)
        hint = {"cx": float(c["cx"]), "cy": float(c["cy"]), "r": float(c["r"])}
        if hint["r"] <= 2:
            raise ValueError("radius too small")
    except Exception as e:
        return JSONResponse({"error": f"Bad circle: {e}"}, status_code=400)

    async def stream():
        try:
            async for line in _emit(j["work"], j["rot"], None, hint, job,
                                    seed_imgs=j.get("seed"),
                                    plate_mm=j.get("plate_mm", PLATE_MM),
                                    step=j.get("step")):
                yield line
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


async def _emit(work, rot, tap_norm, seed_hint, token, seed_imgs=None,
                plate_mm=PLATE_MM, step=None):
    """Bridge the blocking pipeline into an async NDJSON stream."""
    global GATE
    if GATE is None:
        GATE = asyncio.Semaphore(MAX_CONCURRENT)
    async with GATE:
        async for line in _emit_inner(work, rot, tap_norm, seed_hint, token,
                                      seed_imgs, plate_mm, step):
            yield line


async def _emit_inner(work, rot, tap_norm, seed_hint, token, seed_imgs=None,
                      plate_mm=PLATE_MM, step=None):
    out = _queue.Queue()
    box = {}

    def on_point(i, x, y, c):
        out.put(("p", i))

    def run():
        try:
            kw = {}
            if seed_imgs:
                # the clip was shrunk before upload, so the tracker works in
                # clip pixels while the plate was measured in still pixels
                kw = dict(seed_images=seed_imgs, width=None, step=step or 2)
            elif step:
                kw = dict(step=step)
            box["r"] = analyse(work, tap_norm=tap_norm, rot=rot, seed_hint=seed_hint,
                               on_point=on_point, plate_mm=plate_mm, **kw)
        except Exception as e:
            box["e"] = e
        finally:
            out.put(("end", None))

    # The progress bar counts the messages below, so total_frames must be the
    # number of ANALYSED frames, not the clip's own frame count. Probing for it
    # costs about ten milliseconds and has to happen before the first tick.
    src_total, src_fps = 0, 30.0
    try:
        pc = cv2.VideoCapture(work)
        src_total = int(pc.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_fps = pc.get(cv2.CAP_PROP_FPS) or 30.0
        pc.release()
    except Exception as e:
        print(f"[analyze] frame-count probe failed: {e}", flush=True)
    eff_step = (step or (2 if seed_imgs else STEP))
    analysed_total = max(1, -(-src_total // eff_step)) if src_total else 0
    yield json.dumps({"meta": {"total_frames": analysed_total, "fps": src_fps,
                               "source_frames": src_total, "step": eff_step}}) + "\n"

    started = time.time()
    threading.Thread(target=run, daemon=True).start()

    while True:
        try:
            kind, val = out.get_nowait()
        except _queue.Empty:
            await asyncio.sleep(0.02)
            continue
        if kind == "end":
            break
        yield json.dumps({"frame": {"i": val}}) + "\n"

    if "e" in box:
        print(f"[analyze] pipeline error: {box['e']!r}", flush=True)
        yield json.dumps({"error": str(box["e"])}) + "\n"
        return
    r = box.get("r") or {}
    if not r.get("ok"):
        yield json.dumps({"error": REFUSE_MSG if r.get("verdict") == "refuse"
                          else (r.get("reason") or "Analysis failed"),
                          "verdict": r.get("verdict")}) + "\n"
        return
    print(f"[analyze] verdict={r['verdict']} seed={r['seed']} "
          f"{r['seconds']:.2f}s {r['realtime_x']:.2f}xRT tracked={r['tracked']}/{r['frames']}",
          flush=True)
    yield json.dumps({
        "done": True,
        "frames": r["frames_src"],
        "reps": r["rep_metrics"],
        "cap_w": r["src_w"], "cap_h": r["src_h"], "fps": r["src_fps"],
        "rotation": rot, "px_per_m": round(r["px_per_m"], 2),
        # --- new, additive ---
        "verdict": r["verdict"],
        "plate_mm": r.get("plate_mm"),
        "off_square_deg": r.get("off_square_deg"),
        "scale_model": r.get("scale_model"),
        "seed": r["seed"],
        "alternate": r.get("alternate"),
        "job": token,
        "tracked": r["tracked"], "analysed": r["frames"],
        "gaps": len(r.get("gaps") or []),
        "seconds": round(r["seconds"], 2),
        "server_s": round(time.time() - started, 2),
    }) + "\n"
