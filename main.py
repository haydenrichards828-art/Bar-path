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
import urllib.error, urllib.parse, urllib.request
import numpy as np
import queue as _queue
from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from bp.pipeline import analyse, direct_step, STEP, PLATE_MM

# No /docs, /redoc or /openapi.json: the only caller is the app's own proxy, and
# a public schema is a map for everyone else. No CORS either, for the same
# reason — since T-9056 no browser calls this service directly, so a browser
# that tries is refused by its own same-origin rule as well as by the key.
app = FastAPI(title="ForceTrack Bar Path API", version="10.0.0",
              docs_url=None, redoc_url=None, openapi_url=None)

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
    """Degrees, in bp.pipeline._rot's terms (90 = turn anticlockwise), that
    make the stored frame upright.

    Two places carry the answer and they use OPPOSITE signs. The display
    matrix side data (what ffmpeg 5 and later report, and what iPhones write)
    says -90 for a portrait clip, meaning turn it 90 clockwise, which is 270
    here. The legacy `rotate` tag (ffmpeg 4 and older, and some exports) says
    90 for the same clip, also meaning 90 clockwise, so it must be negated.
    This used to read the tag first and pass it straight through, which turns
    a portrait clip upside down wherever the tag exists. Side data first now,
    and the tag negated."""
    def norm(v): return int(round(float(v))) % 360   # -90 -> 270, -180 -> 180, -270 -> 90
    try:
        r2 = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_streams", "-of", "json", path],
                            capture_output=True, text=True, timeout=10)
        data = json.loads(r2.stdout or "{}")
        for stream in data.get("streams", []):
            for sd in stream.get("side_data_list", []) or []:
                rot = sd.get("rotation")
                if rot is not None and float(rot) != 0:
                    return norm(rot)
            tag = (stream.get("tags") or {}).get("rotate")
            if tag not in (None, "", "0"):
                return norm(-float(tag))
    except Exception as e:
        print(f"[get_rotation] probe failed: {e}", flush=True)
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


def _remember(work, tmp, rot, coach=None, step=None, direct=False, plate_mm=PLATE_MM):
    _reap()
    tok = secrets.token_urlsafe(12)
    with JOBS_LOCK:
        JOBS[tok] = {"work": work, "files": {work, tmp}, "rot": rot, "at": time.time(),
                     "seed": None, "plate_mm": plate_mm, "step": step,
                     "coach": coach, "direct": direct, "result": None}
    return tok


# ------------------------------------------------------------------ the gate
# Everything below runs BEFORE FastAPI reads a byte of the upload, so a caller
# without the key costs the service one header comparison, not a 600 MB read.
#
# Until 8.1.0 the key was the `api_key` form field, compared against
# ANALYZE_KEY with a default of "". With ANALYZE_KEY unset, "" == "" and every
# caller was let in: measured on 28 Oct, a direct POST to /analyze with no key
# at all was analysed. Now:
#   - the key is a HEADER, x-barpath-key, checked before the body is read;
#   - an unset or short ANALYZE_KEY refuses everything (503) instead of
#     accepting everything. The old key was public in the app's JavaScript
#     until T-9056 and was short; the length rule is what makes "accept the new
#     key only" true without the old value appearing anywhere.
KEY_ENV       = "ANALYZE_KEY"
KEY_MIN_LEN   = 32
MAX_BODY      = 10 * 1024 * 1024   # the proxy refuses over 6 MB; this is the
                                   # service's own ceiling, with room for the
                                   # multipart framing the proxy adds back
MAX_REQUEST_S = 150                # the editor gives up at 180 s
RATE_WINDOW_S = 600
RATE_MAX      = 30                 # per coach per 10 min, /analyze and /refine
                                   # together. An analysis and a few corrections
                                   # is under ten; thirty is already a loop.
GUARDED       = {"/analyze", "/refine", "/analyze-url", "/result"}
UNCOUNTED     = {"/result"}            # collecting an answer is not an analysis
_RATE = {}
_RATE_LOCK = threading.Lock()


def _key():
    k = os.environ.get(KEY_ENV, "")
    return k if len(k) >= KEY_MIN_LEN else None


def _over_rate(coach, now=None):
    """Seconds to wait, or 0. This process is the only replica, so unlike the
    proxy's per-instance brake this count is the real one across every
    Netlify instance."""
    now = now or time.time()
    with _RATE_LOCK:
        if len(_RATE) > 5000:
            for k in [k for k, v in _RATE.items() if now - v[0] > RATE_WINDOW_S]:
                del _RATE[k]
        start, used = _RATE.get(coach, (now, 0))
        if now - start > RATE_WINDOW_S:
            start, used = now, 0
        if used >= RATE_MAX:
            return max(1, int(start + RATE_WINDOW_S - now))
        _RATE[coach] = (start, used + 1)
        return 0


async def _reply(send, status, body, extra=()):
    data = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(data)).encode()), *extra]})
    await send({"type": "http.response.body", "body": data})


class Gate:
    """Key, size and rate, in that order, as plain ASGI so it runs first."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] not in GUARDED:
            return await self.app(scope, receive, send)
        h = {k.decode("latin-1").lower(): v for k, v in scope["headers"]}
        key = _key()
        if key is None:
            print(f"[gate] {KEY_ENV} unset or shorter than {KEY_MIN_LEN}; refusing", flush=True)
            return await _reply(send, 503, {"error": "Bar path analysis is not configured on the server."})
        if not secrets.compare_digest(h.get("x-barpath-key", b""), key.encode()):
            return await _reply(send, 401, {"error": "unauthorized"})
        try:
            declared = int(h.get("content-length", b"0"))
        except ValueError:
            return await _reply(send, 400, {"error": "Bad Content-Length"})
        if declared > MAX_BODY:
            return await _reply(send, 413, {"error": "Upload too large"})
        coach = h.get("x-barpath-coach", b"").decode("latin-1")[:64] or "unknown"
        wait = 0 if scope["path"] in UNCOUNTED else _over_rate(coach)
        if wait:
            return await _reply(send, 429, {"error": "Too many analyses — try again shortly."},
                                [(b"retry-after", str(wait).encode())])
        # The body is read here, under the cap, so an upload with no
        # Content-Length (chunked) meets the same limit as one that declares it.
        chunks, size = [], 0
        while True:
            m = await receive()
            if m["type"] == "http.disconnect":
                return
            size += len(m.get("body", b""))
            if size > MAX_BODY:
                return await _reply(send, 413, {"error": "Upload too large"})
            chunks.append(m.get("body", b""))
            if not m.get("more_body"):
                break
        body, replayed = b"".join(chunks), False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()       # the disconnect the stream listens for

        await self.app(scope, replay, send)


app.add_middleware(Gate)


# ------------------------------------------------------------------ the stream
REFUSE_MSG = ("Could not find a barbell plate where you tapped. Tap the centre "
              "of the plate, and make sure the whole plate is in shot.")


@app.get("/health")
def health():
    """Railway's deploy check (railway.json). 503 only when the image itself is
    broken, so a bad build never replaces a good one. A missing key is
    reported, not failed: failing the check would keep the previous, open
    build serving traffic, which is the exact thing this release closes."""
    _reap()
    tools = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
    body = {"status": "ok" if tools else "degraded", "version": app.version,
            "jobs": len(JOBS), "auth": "configured" if _key() else "missing"}
    return JSONResponse(body, status_code=200 if tools else 503)


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
                  seed: UploadFile = File(None)):
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
async def refine(job: str = Form(...), circle: str = Form(...)):
    """Re-run a clip already on disk with the circle the coach corrected.

    This is the whole point of keeping the file: when the seed needs a human,
    the coach fixes the circle and gets an answer in the six seconds the
    analysis takes, instead of uploading fifty megabytes again."""
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
            with JOBS_LOCK:
                j["result"] = None
            async for line in _emit(j["work"], j["rot"], None, hint, job,
                                    seed_imgs=j.get("seed"),
                                    plate_mm=j.get("plate_mm", PLATE_MM),
                                    step=j.get("step"), direct=j.get("direct", False)):
                yield line
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ------------------------------------------------------------------ the clip by URL
# T-9378 (B-0339 part B). The phone used to download the whole set video, play
# it through MediaRecorder in real time and upload the result: 30 to 40 seconds
# for a 20 second clip. Now the app's proxy checks the coach owns the set,
# presigns a short lived GET and posts only that URL and the tap, and this fetches
# the phone's own 720p H.264 from R2 straight into the analyser.
#
# A URL from outside is the classic way to make a server fetch something it
# should not, so: https only, the R2 host only, no userinfo, no other port, and
# NO REDIRECTS, because a redirect is a second URL nobody checked.
CLIP_HOST_ENV  = "CLIP_HOST"           # exact host, when set; else R2's own suffix
CLIP_HOST_R2   = ".r2.cloudflarestorage.com"
CLIP_TEST_ENV  = "CLIP_TEST_HTTP_HOST" # tests/analyze_url.py only; never set on Railway
MAX_CLIP_BYTES = 300 * 1024 * 1024     # the app takes set videos up to 100 MB;
                                       # 3 minutes of 720p at 2.5 Mbps is 56 MB
MAX_CLIP_S     = 300                   # past this it cannot finish in MAX_REQUEST_S
FETCH_TIMEOUT_S = 60
FETCH_CHUNK    = 1 << 20


class ClipRefused(Exception):
    pass


def clip_url_ok(url):
    """The reason this URL may not be fetched, or None."""
    try:
        u = urllib.parse.urlsplit(url)
        host, port = (u.hostname or "").lower(), u.port
    except ValueError:
        return "not a URL"
    if u.username or u.password:
        return "userinfo"
    test = os.environ.get(CLIP_TEST_ENV, "")
    if test and u.scheme == "http" and u.netloc == test:
        return None
    if u.scheme != "https":
        return "not https"
    if port not in (None, 443):
        return "port"
    want = os.environ.get(CLIP_HOST_ENV, "").lower()
    if not (host == want if want else (host.endswith(CLIP_HOST_R2) and len(host) > len(CLIP_HOST_R2))):
        return "host"
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def fetch_clip(url, dest, max_bytes=MAX_CLIP_BYTES, timeout=FETCH_TIMEOUT_S):
    """Stream the clip to `dest` under the size and time limits. Bytes written."""
    opener = urllib.request.build_opener(_NoRedirect)
    started = time.time()
    try:
        with opener.open(url, timeout=min(15, timeout)) as r, open(dest, "wb") as f:
            declared = int(r.headers.get("content-length") or 0)
            if declared > max_bytes:
                raise ClipRefused("too large")
            got = 0
            while True:
                chunk = r.read(FETCH_CHUNK)
                if not chunk:
                    return got
                got += len(chunk)
                if got > max_bytes:
                    raise ClipRefused("too large")
                if time.time() - started > timeout:
                    raise ClipRefused("too slow")
                f.write(chunk)
    except urllib.error.HTTPError as e:
        raise ClipRefused(f"status {e.code}")
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise ClipRefused(f"unreachable: {type(e).__name__}")


CLIP_MESSAGES = {
    "too large": "This video is too large to analyse. Trim it to the lift and try again.",
    "too long": "This video is too long to analyse. Trim it to the lift and try again.",
}
CLIP_FAILED = "Could not read this video from storage. Try again."


def clip_seconds(path):
    pc = cv2.VideoCapture(path)
    try:
        fps = pc.get(cv2.CAP_PROP_FPS) or 30.0
        return (pc.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps, fps
    finally:
        pc.release()


@app.post("/analyze-url")
async def analyze_url(request: Request):
    """The same NDJSON stream as /analyze, from a clip this service fetches. The
    first line names the job, so an answer outlives the connection: the proxy
    can be cut at its own time limit and the phone can lose signal, and the app
    collects the result from /result."""
    t0 = time.time()
    try:
        body = json.loads(await request.body())
        url = body["url"]
        tap_norm = (float(body.get("start_x", 0.5)), float(body.get("start_y", 0.5)))
        plate_mm = float(body.get("plate_mm", PLATE_MM))
        if not isinstance(url, str) or not all(0 <= v <= 1 for v in tap_norm) or plate_mm <= 0:
            raise ValueError("bad value")
    except Exception as e:
        return JSONResponse({"error": f"Bad request: {type(e).__name__}"}, status_code=400)
    why = clip_url_ok(url)
    if why:
        # The host alone: the rest of a presigned URL is a credential.
        print(f"[analyze-url] refused ({why}): {urllib.parse.urlsplit(url).hostname}", flush=True)
        return JSONResponse({"error": "That address is not allowed."}, status_code=400)
    coach = request.headers.get("x-barpath-coach", "")[:64] or "unknown"
    tmp = tempfile.mktemp(suffix=".mp4")

    async def stream():
        keep = False
        work = tmp
        try:
            size = await asyncio.to_thread(fetch_clip, url, tmp)
            fetched = time.time() - t0
            secs, fps = clip_seconds(tmp)
            if secs > MAX_CLIP_S:
                raise ClipRefused("too long")
            work = normalise_video(tmp)
            rot = get_rotation(work)
            step = direct_step(fps)
            token = _remember(work, tmp, rot, coach=coach, step=step, direct=True, plate_mm=plate_mm)
            keep = True
            print(f"[analyze-url] {token} {size} bytes {secs:.1f}s fetched in {fetched:.2f}s", flush=True)
            async for line in _emit(work, rot, tap_norm, None, token, plate_mm=plate_mm,
                                    step=step, direct=True, t0=t0, fetched=fetched):
                yield line
        except ClipRefused as e:
            print(f"[analyze-url] clip refused: {e}", flush=True)
            yield json.dumps({"error": CLIP_MESSAGES.get(str(e), CLIP_FAILED)}) + "\n"
        except Exception as e:
            print(f"[analyze-url] failed: {e!r}", flush=True)
            yield json.dumps({"error": str(e)}) + "\n"
        finally:
            if not keep:
                for f in {tmp, work}:
                    try: os.unlink(f)
                    except Exception: pass

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/result")
async def result(request: Request):
    """The last line of a job's analysis, once it has one. Only for the coach who
    started it: a job name is hard to guess, and this is the second lock."""
    try:
        job = json.loads(await request.body())["job"]
    except Exception:
        return JSONResponse({"error": "Bad request"}, status_code=400)
    coach = request.headers.get("x-barpath-coach", "")[:64] or "unknown"
    with JOBS_LOCK:
        j = JOBS.get(job) if isinstance(job, str) else None
        line = j.get("result") if j else None
    if not j or j.get("coach") != coach:
        return JSONResponse({"error": "That clip is no longer on the server — "
                                      "please analyse it again."}, status_code=410)
    if line is None:
        return JSONResponse({"pending": True})
    return StreamingResponse(iter([line]), media_type="application/x-ndjson")


async def _emit(work, rot, tap_norm, seed_hint, token, seed_imgs=None,
                plate_mm=PLATE_MM, step=None, direct=False, t0=None, fetched=None):
    """Bridge the blocking pipeline into an async NDJSON stream."""
    global GATE
    if GATE is None:
        GATE = asyncio.Semaphore(MAX_CONCURRENT)
    await GATE.acquire()
    held = {}
    try:
        async for line in _emit_inner(work, rot, tap_norm, seed_hint, token,
                                      seed_imgs, plate_mm, step, held, direct, t0, fetched):
            yield line
    finally:
        # The pipeline cannot be interrupted. If the stream ends while it is
        # still running — the time limit, or the coach closing the editor —
        # its slot stays taken until the thread really finishes, so abandoned
        # analyses cannot pile up past MAX_CONCURRENT.
        w = held.get("worker")
        if w is not None and w.is_alive():
            t = asyncio.get_running_loop().create_task(_release_when_done(w))
            _PENDING.add(t)
            t.add_done_callback(_PENDING.discard)
        else:
            GATE.release()


_PENDING = set()


async def _release_when_done(worker):
    try:
        await asyncio.to_thread(worker.join)
    finally:
        GATE.release()


def _final_line(box, rot, token, started, seconds_to=None):
    """The stream's last line, also kept on the job for /result."""
    if "e" in box:
        print(f"[analyze] pipeline error: {box['e']!r}", flush=True)
        return json.dumps({"error": str(box["e"])}) + "\n"
    r = box.get("r") or {}
    if not r.get("ok"):
        return json.dumps({"error": REFUSE_MSG if r.get("verdict") == "refuse"
                           else (r.get("reason") or "Analysis failed"),
                           "verdict": r.get("verdict")}) + "\n"
    print(f"[analyze] verdict={r['verdict']} seed={r['seed']} "
          f"{r['seconds']:.2f}s {r['realtime_x']:.2f}xRT tracked={r['tracked']}/{r['frames']}",
          flush=True)
    return json.dumps({
        "done": True,
        "frames": r["frames_src"],
        "reps": r["rep_metrics"],
        "cap_w": r["src_w"], "cap_h": r["src_h"], "fps": r["src_fps"],
        "rotation": rot, "px_per_m": round(r["px_per_m"], 2),
        # --- new, additive ---
        "verdict": r["verdict"],
        # v9: why the verdict is "check" when the plate finder said "ok"
        # (the path left the plate, a rep could not be measured, ...), and
        # the plate finder's own verdict before the track confirmed or doubted it
        "doubts": r.get("doubts") or [],
        "seed_verdict": r.get("seed_verdict"),
        # v10: tracked again from a better circle (why, and which pass was kept)
        "second_look": r.get("second_look"),
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
        **(seconds_to or {}),
    }) + "\n"


async def _emit_inner(work, rot, tap_norm, seed_hint, token, seed_imgs=None,
                      plate_mm=PLATE_MM, step=None, held=None, direct=False, t0=None, fetched=None):
    out = _queue.Queue()
    box = {}
    timing = {}                      # T-9378: filled only on the URL path
    finished = {}

    def on_point(i, x, y, c):
        if t0 is not None and "first_point_s" not in timing:
            timing["first_point_s"] = round(time.time() - t0, 2)
            print(f"[analyze-url] {token} first point at {timing['first_point_s']}s", flush=True)
        out.put(("p", (i, x, y, c)))

    def run():
        try:
            kw = {}
            if direct:
                # the phone's own file, seeded from its own frames at full
                # quality; the tracker works in its pixels at the same analysed
                # rate as the app chain (see bp.pipeline.direct_step)
                kw = dict(width=None, step=step)
            elif seed_imgs:
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
            if t0 is not None:
                timing["fetch_s"] = round(fetched or 0, 2)
                timing["done_s"] = round(time.time() - t0, 2)
                print(f"[analyze-url] {token} done at {timing['done_s']}s "
                      f"(fetch {timing['fetch_s']}s, first point {timing.get('first_point_s')}s)", flush=True)
            # Kept on the job first: the stream may already have gone (the
            # proxy's time limit, the phone's signal) and the worker is the one
            # thing that still knows the answer.
            finished["line"] = _final_line(box, rot, token, started, timing)
            with JOBS_LOCK:
                if token in JOBS:
                    JOBS[token]["result"] = finished["line"]
            out.put(("end", None))

    # The progress bar counts the messages below, so total_frames must be the
    # number of ANALYSED frames, not the clip's own frame count. Probing for it
    # costs about ten milliseconds and has to happen before the first tick.
    src_total, src_fps = 0, 30.0
    cap_w = cap_h = 0
    try:
        pc = cv2.VideoCapture(work)
        src_total = int(pc.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_fps = pc.get(cv2.CAP_PROP_FPS) or 30.0
        if direct:
            pc.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            cap_w, cap_h = int(pc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(pc.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if rot in (90, 270):
                cap_w, cap_h = cap_h, cap_w
        pc.release()
    except Exception as e:
        print(f"[analyze] frame-count probe failed: {e}", flush=True)
    eff_step = (step or (2 if seed_imgs else STEP))
    analysed_total = max(1, -(-src_total // eff_step)) if src_total else 0
    meta = {"total_frames": analysed_total, "fps": src_fps,
            "source_frames": src_total, "step": eff_step}
    if direct:
        meta.update(job=token, cap_w=cap_w, cap_h=cap_h)
    yield json.dumps({"meta": meta}) + "\n"

    started = time.time()
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    if held is not None:
        held["worker"] = worker

    while True:
        try:
            kind, val = out.get_nowait()
        except _queue.Empty:
            if time.time() - started > MAX_REQUEST_S:
                # The coach gets an answer now; the worker keeps its slot
                # until it ends (see _emit).
                print(f"[analyze] over {MAX_REQUEST_S}s, giving up", flush=True)
                yield json.dumps({"error": "The analysis took too long. Trim the "
                                           "video to the lift itself and try again."}) + "\n"
                return
            await asyncio.sleep(0.02)
            continue
        if kind == "end":
            break
        i, x, y, c = val
        frame = {"i": i}
        if direct:
            # the live drawing: a provisional point, in the clip's pixels. The
            # answer is the last line. Only this path: /analyze's lines stay
            # byte for byte what tests/same_path.py compares.
            frame.update(t=round(i * eff_step / src_fps, 3),
                         x=None if x is None else round(x, 1),
                         y=None if y is None else round(y, 1))
        yield json.dumps({"frame": frame}) + "\n"

    yield finished["line"]
