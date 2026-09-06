"""The circle a coach would have confirmed, per clip: (cx, cy, r) in source
pixels on frame 0. Centre from find_plate's better candidate, radius from the
hand measurement in truth.py. This is the seed the tracker gets in production,
where the coach taps and then signs off on the circle."""
CONFIRMED = {
  "bench-side-clean-01":     (418,  809, 107),
  "bench-side-occluded-02":  (580,  774, 113),
  "dl-angle-blue-bumper-02": (540, 1265, 200),
  "dl-dark-lowcontrast-05":  (430, 1663, 186),   # centre from a 220px fit: suspect
  "dl-iron-red-25-04":       (511, 1462, 216),
  "dl-mixed-diameters-03":   (472, 1249, 203),
  "dl-side-clean-01":        (506, 1229, 190),
  "dl-small-green-10-06":    (419, 1509, 176),   # rim confirmed by eye, 6 Sep
  "sq-bright-gym-close-07":  (347,  631, 138),
  "sq-bright-gym-wide-06":   (709,  857, 136),
  "sq-calibrated-mixed-04":  (191,  839, 167),
  "sq-iron-rusty-glare-03":  (304, 1144, 200),
  "sq-side-clean-01":        (546,  701, 180),
  "sq-side-mixed-colour-02": (391,  445, 227),
  "sq-small-green-close-05": (385,  790, 254),
}
