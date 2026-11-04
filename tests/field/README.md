# Field regression: Hayden's own clips, sent the way the app sends them

    python tests/field/field_regress.py "<clips folder>" --out <results folder>

Clips: on Hayden's computer, `Downloads/bar path test videos/trimmed` (47 clips
cut from his Drive videos). `expectations.json` holds, per clip: the lift, the
tap (normalised x, y on the first frame), what the answer should be (`ok`,
`ask` = asking or a right ok are both fine, `refuse`), the rep count where it
was counted by eye, and why the clip is in the set.

Each clip goes through T-9300's phone shrink (720p short edge, H.264 2.5 Mbps),
then `prepareAnalysisUpload` (seed strip of 5 frames at JPEG q95, 900 px 15 fps
900 kbps re-encode), then `analyse(step=2)`, exactly as the server receives it.

Run it three ways before shipping a tracker change:

    field_regress.py CLIPS --out A                 # the app's chain
    field_regress.py CLIPS --out B --no-shrink     # original file into the re-encode
    field_regress.py CLIPS --out C --full          # full resolution, the old upload path

A clip fails if the verdict is wrong for its expectation, the rep count differs
from the counted one, the path moves faster than a barbell can (up 2.5 m/s,
down 4 m/s, sideways 1 m/s), a rep's range is outside the lift's range, or a
rep's mean concentric velocity is outside 0.05 to 1.6 m/s. "SILENTLY WRONG"
means it failed AND answered ok: that count must be zero in all three runs.

`--resume` re-checks saved results without re-running; `--only 2843 2919`
runs a few clips; overlays of every run are written next to the results.

Results, 4 Nov 2026 (47 clips):

| | right, answers ok | correctly asks or refuses | wrong, but asks | wrong, answers ok |
|---|---|---|---|---|
| v8 (main 1248a5b), app chain | 15 | 6 | 16 | 10 |
| v9, app chain | 32 | 6 | 9 | 0 |
| v9, original into the re-encode | 32 | 7 | 8 | 0 |
| v9, full resolution | 31 | 6 | 10 | 0 |

Between the two app encodes v9 changes answer class on 4 clips (v8: 13), and
mean concentric velocity agrees within 0.03 m/s on 87% of 95 reps.
