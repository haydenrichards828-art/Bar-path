"""T-9378: tap to first point and tap to done through /analyze-url, per clip.
A build on :8123 started with ANALYZE_KEY=$K and CLIP_TEST_HTTP_HOST=127.0.0.1:8124,
held to Railway's two cores with `taskset -c 0,1`:

    K=<key> python3 tests/load_time.py <dir of 720p .r2.mp4 files> [--secs 15 25]

The clip is served from this machine, so the fetch costs nothing here: on Railway
add its time to R2. That is what these numbers do not say. The numbers on the
deployed service come from real taps: route_timings, /bar-path/first-point and
/bar-path/done."""
import argparse, glob, http.client, http.server, json, os, statistics as st, subprocess, sys, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--secs", nargs=2, type=float, default=[15, 25])
ap.add_argument("--expect", default=os.path.join(os.path.dirname(__file__), "field", "expectations.json"))
a = ap.parse_args()
K = os.environ["K"]
exps = json.load(open(a.expect))
root = os.path.abspath(a.dir)


class Clips(http.server.BaseHTTPRequestHandler):
    def log_message(self, *x): pass

    def do_GET(self):
        data = open(os.path.join(root, os.path.basename(self.path)), "rb").read()
        self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)


threading.Thread(target=http.server.ThreadingHTTPServer(("127.0.0.1", 8124), Clips).serve_forever, daemon=True).start()


def secs(p):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p]))


rows = []
for path in sorted(glob.glob(os.path.join(root, "*.r2.mp4"))):
    stem = os.path.basename(path)[:-len(".r2.mp4")]
    if stem not in exps or not a.secs[0] <= secs(path) <= a.secs[1]:
        continue
    tap = exps[stem]["tap"]
    c = http.client.HTTPConnection("127.0.0.1", 8123, timeout=200)
    t0 = time.time()
    c.request("POST", "/analyze-url", json.dumps({"url": f"http://127.0.0.1:8124/{os.path.basename(path)}", "start_x": tap[0], "start_y": tap[1]}),
              {"content-type": "application/json", "x-barpath-key": K, "x-barpath-coach": f"load{len(rows) // 25}"})
    r = c.getresponse()
    first, last = None, {}
    for raw in r:
        m = json.loads(raw) if raw.strip() else {}
        if first is None and (m.get("frame") or {}).get("x") is not None:
            first = time.time() - t0
        last = m or last
    done = time.time() - t0
    rows.append((stem, secs(path), first, done, last.get("verdict") or last.get("error", "?")[:30]))
    print(f"{stem:42} {rows[-1][1]:5.1f}s  first {first if first is None else round(first, 2)}  done {done:5.2f}  {rows[-1][4]}", flush=True)

fs = sorted(x[2] for x in rows if x[2] is not None)
ds = sorted(x[3] for x in rows)
p95 = lambda v: v[min(len(v) - 1, int(round(len(v) * 0.95 + 0.49)) - 1)]
print(f"\n{len(rows)} clips of {a.secs[0]:.0f} to {a.secs[1]:.0f} s: first point median {st.median(fs):.2f} s, p95 {p95(fs):.2f} s, "
      f"done median {st.median(ds):.2f} s, p95 {p95(ds):.2f} s")
