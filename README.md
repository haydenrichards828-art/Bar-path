# ForceTrack Bar Path API

FastAPI + OpenCV service that tracks a barbell plate through a lift video. Its
only caller is the ForceTrack app's Netlify function `netlify/functions/barpath.ts`,
which checks the coach's session, adds the key and streams the answer back.
Browsers never call this service directly.

## Endpoints

- `POST /analyze`, `POST /refine` — need the `x-barpath-key` header. Refused
  before the upload is read: no or wrong key 401, unset key 503, over 10 MB 413,
  over 30 calls per coach (`x-barpath-coach`) in 10 minutes 429. An analysis is
  stopped after 150 s.
- `GET /health` — `{"status","version","jobs","auth"}`. Railway's deploy check
  (`railway.json`). `auth: "missing"` means ANALYZE_KEY is unset or too short.

## Environment (Railway → service → Variables)

- `ANALYZE_KEY` — at least 32 characters. Must equal `BARPATH_API_KEY` in
  Netlify. Make one with `openssl rand -base64 48`. Never put it in a `VITE_`
  variable: those are compiled into the public JavaScript.

## Deploy

Railway builds the Dockerfile from `main` of this repo.

## Before changing a dependency or anything in bp/

Run `tests/same_path.py` against the old and new build and `cmp` the outputs,
and `tests/regress.py` on the 20-clip set. See `tests/README.md`.
