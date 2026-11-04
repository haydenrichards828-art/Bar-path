# Running the regression gate

    # ANALYZE_KEY must be 32+ characters (8.1.0); the harness sends it as x-barpath-key
    export K=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
    ANALYZE_KEY=$K python3 -m uvicorn main:app --host 127.0.0.1 --port 8123
    TESTSET="C:\Users\hayde\Downloads\Training Footage\testset" python3 tests/regress.py

Two quicker checks, both against the same server:

    python3 tests/gate.py            # the key, size and rate gate (14 checks)
    python3 tests/same_path.py ...   # one real clip, before vs after a change — see its docstring

`regress.py` imports `taps.py` and `truth.py` from this folder. Both ship here.
Do NOT let it pick up a `truth.py` from anywhere else on the machine — an older
copy exists under `C:\Users\hayde\Bar-path` and carries a superseded value for
`dl-small-green-10-06` (150, an inner moulding line on the plate rather than the
rim). The correct value is 176, confirmed by visual audit on 6 Sep, and it is the
one in this folder.

Expected output: `20 as expected, 0 not`.
