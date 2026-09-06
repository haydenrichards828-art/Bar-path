# Running the regression gate

    ANALYZE_KEY=whatever python3 -m uvicorn main:app --host 127.0.0.1 --port 8123
    TESTSET="C:\Users\hayde\Downloads\Training Footage\testset" python3 tests/regress.py

`regress.py` imports `taps.py` and `truth.py` from this folder. Both ship here.
Do NOT let it pick up a `truth.py` from anywhere else on the machine — an older
copy exists under `C:\Users\hayde\Bar-path` and carries a superseded value for
`dl-small-green-10-06` (150, an inner moulding line on the plate rather than the
rim). The correct value is 176, confirmed by visual audit on 6 Sep, and it is the
one in this folder.

Expected output: `20 as expected, 0 not`.
