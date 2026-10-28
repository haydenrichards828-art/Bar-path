"""The gate in front of /analyze and /refine. Run against a build on :8123
started with ANALYZE_KEY=$K (32+ characters):

    K=<key> python3 tests/gate.py

Every check here is a request that must be refused BEFORE any work is done,
plus the ones that must still get through."""
import http.client, json, os, sys

K = os.environ["K"]
HOST, PORT = "127.0.0.1", int(os.environ.get("PORT", 8123))
fails = 0


def req(method, path, headers=None, body=b"", chunked=False):
    c = http.client.HTTPConnection(HOST, PORT, timeout=30)
    c.putrequest(method, path)
    for k, v in (headers or {}).items():
        c.putheader(k, v)
    if chunked:
        c.putheader("Transfer-Encoding", "chunked")
        c.endheaders()
        try:
            for i in range(0, len(body), 1 << 20):
                part = body[i:i + (1 << 20)]
                c.send(b"%x\r\n" % len(part) + part + b"\r\n")
            c.send(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
    else:
        if "Content-Length" not in (headers or {}):
            c.putheader("Content-Length", str(len(body)))
        c.endheaders()
        try:
            if body:
                c.send(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
    r = c.getresponse()
    out = r.status, r.read()[:200], dict(r.getheaders())
    c.close()
    return out


def check(name, got, want):
    global fails
    ok = got == want
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f" (want {want})"))


form = {"Content-Type": "multipart/form-data; boundary=X"}
refine = (b'--X\r\nContent-Disposition: form-data; name="job"\r\n\r\nnope\r\n'
          b'--X\r\nContent-Disposition: form-data; name="circle"\r\n\r\n{}\r\n--X--\r\n')

check("analyze, no key", req("POST", "/analyze", form, refine)[0], 401)
check("refine, no key", req("POST", "/refine", form, refine)[0], 401)
check("old api_key form field only",
      req("POST", "/refine", form, refine.replace(b"--X--", b'--X\r\nContent-Disposition: form-data; name="api_key"\r\n\r\n' + K.encode() + b"\r\n--X--"))[0], 401)
check("wrong key", req("POST", "/refine", {**form, "x-barpath-key": K[:-1] + "x"}, refine)[0], 401)
check("no key, declares 1 GB (refused unread)",
      req("POST", "/analyze", {**form, "Content-Length": str(1 << 30)})[0], 401)
check("browser preflight", req("OPTIONS", "/analyze", {"Origin": "https://evil.example",
      "Access-Control-Request-Method": "POST"})[0], 401)
check("schema hidden", req("GET", "/openapi.json")[0], 404)
check("docs hidden", req("GET", "/docs")[0], 404)
st, body, _ = req("GET", "/health")
check("health", (st, json.loads(body)["auth"]), (200, "configured"))

key = {**form, "x-barpath-key": K, "x-barpath-coach": "gate-test"}
check("key, declares 11 MB", req("POST", "/analyze", {**key, "Content-Length": str(11 << 20)})[0], 413)
check("key, chunked 11 MB", req("POST", "/analyze", key, b"0" * (11 << 20), chunked=True)[0], 413)
check("key, gets through to the handler", req("POST", "/refine", key, refine)[0], 410)

codes = [req("POST", "/refine", {**key, "x-barpath-coach": "flood"}, refine)[0] for _ in range(31)]
check("30 per coach then 429", (codes[:30].count(410), codes[30]), (30, 429))
check("another coach unaffected", req("POST", "/refine", {**key, "x-barpath-coach": "other"}, refine)[0], 410)

sys.exit(1 if fails else 0)
