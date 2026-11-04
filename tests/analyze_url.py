"""T-9378: /analyze-url and /result. Run against a build on :8123 started with
ANALYZE_KEY=$K (32+ characters) and CLIP_TEST_HTTP_HOST=127.0.0.1:8124:

    K=<key> python3 tests/analyze_url.py clip.mp4 [three-minutes.mp4]

It serves the clip itself on :8124, because what is being tested is the fetch:
every address that must never be fetched, the limits, the stream, an answer that
outlives the connection, and a correction on the same job. CLIP_TEST_HTTP_HOST
is the only way an http address passes, and it is never set on Railway."""
import http.client, http.server, json, os, sys, threading, time

K = os.environ["K"]
HOST, PORT, CLIP_PORT = "127.0.0.1", int(os.environ.get("PORT", 8123)), 8124
CLIP, LONG = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
fails = 0
fetched = []


class Clips(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        fetched.append(self.path)
        if self.path == "/redirect":
            self.send_response(302); self.send_header("Location", f"http://{HOST}:{CLIP_PORT}/clip"); self.end_headers()
        elif self.path == "/missing":
            self.send_response(403); self.send_header("Content-Length", "0"); self.end_headers()
        elif self.path == "/huge":
            self.send_response(200); self.send_header("Content-Length", str(400 * 1024 * 1024)); self.end_headers()
        else:
            data = open(LONG if self.path == "/long" else CLIP, "rb").read()
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
            try: self.wfile.write(data)
            except OSError: pass


threading.Thread(target=http.server.ThreadingHTTPServer((HOST, CLIP_PORT), Clips).serve_forever, daemon=True).start()


def post(path, body, coach="t9378", key=K):
    c = http.client.HTTPConnection(HOST, PORT, timeout=200)
    h = {"content-type": "application/json", "x-barpath-coach": coach}
    if key:
        h["x-barpath-key"] = key
    c.request("POST", path, json.dumps(body), h)
    return c, c.getresponse()


def lines(r):
    return [json.loads(l) for l in r.read().decode().splitlines() if l.strip()]


def check(name, ok, extra=""):
    global fails
    print(("ok   " if ok else "FAIL ") + name + (f"  {extra}" if extra else ""), flush=True)
    fails += not ok


clip_url = lambda p="/clip": f"http://{HOST}:{CLIP_PORT}{p}"
tap = {"start_x": float(os.environ.get("TAP_X", 0.5)), "start_y": float(os.environ.get("TAP_Y", 0.5))}

_, r = post("/analyze-url", {"url": clip_url(), **tap}, key=None); r.read()
check("no key: 401", r.status == 401)

# Every one of these must be refused BEFORE anything is fetched.
for name, url in [
    ("another host", "https://example.com/clip.mp4"),
    ("plain http to R2", "http://acct.r2.cloudflarestorage.com/b/k"),
    ("a look-alike host", "https://acct.r2.cloudflarestorage.com.evil.example/b/k"),
    ("the bare suffix", "https://.r2.cloudflarestorage.com/b/k"),
    ("userinfo", "https://acct.r2.cloudflarestorage.com@evil.example/b/k"),
    ("userinfo on the real host", "https://me:pw@acct.r2.cloudflarestorage.com/b/k"),
    ("another port", "https://acct.r2.cloudflarestorage.com:8443/b/k"),
    ("the metadata address", "http://169.254.169.254/latest/meta-data/"),
    ("localhost", "http://localhost:8123/health"),
    ("a file", "file:///etc/passwd"),
    ("not a string", 5),
    ("a broken address", "https://[bad/x"),
    ("a broken port", "https://acct.r2.cloudflarestorage.com:port/b/k"),
]:
    _, r = post("/analyze-url", {"url": url, **tap}); r.read()
    check(f"refused, {name}: 400", r.status == 400, str(r.status))
_, r = post("/analyze-url", {"url": clip_url(), "start_x": 5, "start_y": 0.5}); r.read()
check("a tap outside the frame: 400", r.status == 400)
check("none of those fetched anything", not fetched, str(fetched))

_, r = post("/analyze-url", {"url": clip_url("/redirect"), **tap}); got = lines(r)
check("a redirect is not followed", "error" in got[-1] and fetched == ["/redirect"], str(fetched))
_, r = post("/analyze-url", {"url": clip_url("/missing"), **tap}); got = lines(r)
check("a clip storage refuses: a sentence", "Could not read" in got[-1].get("error", ""))
_, r = post("/analyze-url", {"url": clip_url("/huge"), **tap}); got = lines(r)
check("over the size limit: a sentence", "too large" in got[-1].get("error", ""))

# The stream.
t0 = time.time()
_, r = post("/analyze-url", {"url": clip_url(), **tap}, coach="a")
got, first_point = [], None
for raw in r:
    if not raw.strip():
        continue
    m = json.loads(raw)
    got.append(m)
    if first_point is None and "frame" in m and m["frame"].get("x") is not None:
        first_point = time.time() - t0
done = got[-1]
check("the first line names the job and the size", bool(got[0]["meta"].get("job")) and got[0]["meta"]["cap_w"] > 0)
pts = [m["frame"] for m in got if "frame" in m and m["frame"].get("x") is not None]
check("points carry where and when", len(pts) > 10 and all("t" in p and "y" in p for p in pts[:5]), str(len(pts)))
check("the last line is the answer", done.get("done") is True or "error" in done, str({k: done.get(k) for k in ("verdict", "error")}))
if done.get("done"):
    check("timings on the answer", all(k in done for k in ("fetch_s", "first_point_s", "done_s")))
    print(f"     first point {first_point:.2f}s, done {time.time() - t0:.2f}s, "
          f"verdict {done['verdict']}, reps {len(done['reps'])}, frames {len(done['frames'])}, "
          f"{done['cap_w']}x{done['cap_h']}, fetch {done['fetch_s']}s", flush=True)
    # /result gives the same line, to the same coach only.
    _, r = post("/result", {"job": done["job"]}, coach="a"); again = lines(r)
    check("/result: the same answer", again and again[-1] == done)
    _, r = post("/result", {"job": done["job"]}, coach="b"); r.read()
    check("/result: not to another coach", r.status == 410, str(r.status))
    _, r = post("/result", {"job": "nope"}, coach="a"); r.read()
    check("/result: an unknown job", r.status == 410)
    n = 0
    for _ in range(40):
        _, r = post("/result", {"job": done["job"]}, coach="a"); r.read(); n += r.status == 200
    check("/result does not count against the limit", n == 40, str(n))
    # A correction on the same job, from the clip already on disk.
    c = done["seed"]
    b = f"job={done['job']}&circle=" + json.dumps({"cx": c["cx"], "cy": c["cy"], "r": c["r"]})
    k = http.client.HTTPConnection(HOST, PORT, timeout=200)
    k.request("POST", "/refine", b, {"content-type": "application/x-www-form-urlencoded", "x-barpath-key": K, "x-barpath-coach": "a"})
    rf = lines(k.getresponse())
    check("/refine works on a fetched clip", rf[-1].get("done") is True and rf[-1]["cap_w"] == done["cap_w"], str(rf[-1].get("error")))

# The connection goes (the proxy's time limit, the phone's signal). The answer does not.
c, r = post("/analyze-url", {"url": clip_url(), **tap}, coach="c")
job = json.loads(r.readline())["meta"]["job"]
r.readline()
c.close()
ans, t1 = None, time.time()
while time.time() - t1 < 120:
    k, r = post("/result", {"job": job}, coach="c")
    body = lines(r)
    if body and not body[-1].get("pending"):
        ans = body[-1]; break
    time.sleep(0.5)
check("the connection dropped: the answer is still collected", bool(ans) and ("done" in ans or "error" in ans))

if LONG:
    t0 = time.time()
    _, r = post("/analyze-url", {"url": clip_url("/long"), **tap}, coach="d")
    got = lines(r)
    d = got[-1]
    secs = got[0]["meta"]["source_frames"] / got[0]["meta"]["fps"]
    check(f"a {secs:.0f} s clip of {os.path.getsize(LONG) >> 20} MB is analysed", d.get("done") is True and d.get("error") is None,
          f"{time.time() - t0:.1f}s, verdict {d.get('verdict')}, error {d.get('error')}")

print(f"\n{'PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
