# Tile Grid Phase Experiment

Does the *position* of the DeepForest tiling grid, at a fixed overlap ratio,
change which tree crowns get detected? These scripts establish the raster,
pick a working window, and gate the question at a chosen resolution.

## Scripts

| Script | What it does |
| --- | --- |
| `check_raster.py` | Prints size, band count, dtype, CRS, nodata, pixel size, GSD, footprint and bounds for the orthomosaic. |
| `find_window.py` | Locates a 2000 px square window centred on the valid (alpha > 0) region of the mosaic and reports percent valid coverage. |
| `run_gate.py` | Reads the working window, writes it at native and half resolution, runs DeepForest on both, and reports detection count, score distribution and median box width. |
| `draw_boxes.py` | Overlays the boxes from `run_gate.py` onto the window PNGs for visual inspection. |
| `phase_sweep.py` | The experiment proper. Lays the tiling grid down 16 times at a 4x4 grid of sub-stride phase offsets, holding tile size and overlap ratio fixed, and reports how much the detection set moves. Not yet run. |
| `check_expanded_window.py` | Confirms the expanded read region used by `phase_sweep.py` is inside the raster and that its margin carries valid alpha on all four sides. Read only. |
| `check_tiler_vs_predict_tile.py` | Zero offset sanity check. Runs the hand rolled tiler and `predict_tile()` on the same bare window with identical settings and reports how far they agree. Must pass before the sweep is worth running. Not yet run. |
| `phase_matching.py` | Shared read only module. Reloads the per phase box CSVs and redoes the clustering at any threshold. Imports no model. |
| `check_match_sensitivity.py` | Redoes cross phase matching at IoU 0.3, 0.4 and 0.5 and reports distinct crowns, support histogram, and singleton count for each. Tests whether the singleton pile is a clustering artefact. |
| `draw_support.py` | Draws the core region at one phase with crowns coloured by support band, plus a second figure pooling singletons from all 16 phases. |
| `analyse_support.py` | Box geometry by support band, pattern of support at 4, 8 and 12 against an exact null, axis marginals, and per phase count spread with phase 0 separated. |
| `analyse_mechanism.py` | Mechanism tests: elongation axis against sensitivity axis, seam proximity at the samples found versus missed, and Miller spatial variance per cluster. |

Generated outputs (`*.png`, `*_boxes.csv`) are gitignored. Rerun the scripts to
regenerate them.

## Settled decisions

### Dataset

OFO Mission 000103, CC BY 4.0. Withheld from ML training and reserved by OFO
for model evaluation.

### Orthomosaic

- 11632 x 12458 px
- EPSG:32610
- 3.89 cm GSD
- 4 bands, band 4 is alpha
- no nodata value set

### Working window

`col_off 4820`, `row_off 5260`, `size 2000`, at native resolution.

### Experiment resolution

The experiment runs at **7.78 cm**, downsampled by a factor of 2 from native.
Justified by crown fragmentation at native resolution.

### Gate result at the working window

| Resolution | Detections | Median score | Median box width |
| --- | --- | --- | --- |
| native 3.89 cm | 1201 | 0.273 | 0.97 m |
| downsampled 7.78 cm | 311 | 0.348 | 2.10 m |

At 3.89 cm the median box is under a metre wide, narrower than a real crown at
this site, and confidence is low. Halving the resolution yields fewer,
higher confidence, crown sized boxes. 7.78 cm is therefore the operating point
for the phase sweep.

## Phase sweep design

`phase_sweep.py` isolates grid position as the only variable.

**Phases.** `PATCH_SIZE` 400, `PATCH_OVERLAP` 0.25, so stride is 300 px.
Offsets are swept at 0, 75, 150 and 225 px in both x and y, 16 positions
covering one stride in quarter steps. Overlap ratio is identical throughout.

**Tiling.** `predict_tile()` gives no control over grid offset, so tiles are
cut by hand and `predict_image()` is called per tile, with boxes translated
back to scored window coordinates and merged by one `torchvision.ops.nms` pass
at IoU 0.15. That is the same function and the same threshold DeepForest's own
`mosaic()` uses, since `predict_tile()` passes `iou_threshold=0.15` into it.

**Constant tile size.** Every tile is exactly 400 px under every phase.
DeepForest is sensitive to input size, and if tile size co varied with phase
the two effects could not be separated. The grid therefore runs over a canvas
larger than the scored region.

**Real pixel margin.** The margin is real imagery read from the orthomosaic,
not padded or reflected pixels.

- Expanded read, native: `col_off 4220`, `row_off 4660`, `size 3200`
- Downsampled by 2 to a 1600 px canvas at 7.78 cm
- 300 px of real margin on every side, at experiment resolution
- Alpha coverage verified at 100 percent on all four margin bands

An earlier version reflect padded the window instead. That was wrong. Tiles at
some phases reached up to 325 px into the scored region while still containing
synthetic pixels, far past the 100 px inset that was supposed to exclude them.
Depth by phase was 100 px at phase 0, 175, 250 and 325 px at phases 75, 150 and
225, asymmetric between the low and high edge because nonzero phases emit four
tiles per axis rather than five.

**Scored region.** The scored region is the original 2000 px native working
window, the inner 1000 px of the canvas at experiment resolution. The margin is
context only and is never scored. A box is kept when its centre falls inside
the scored region.

**Nominal inset.** `CORE_INSET` is 25 px, about 1.9 m. It is not a
contamination guard, the real pixel margin removed that need. It exists because
a crown sitting on the scored boundary can have its centre fall inside at one
phase and outside at the next, which would register as instability that is
really a boundary artefact. 25 px is roughly two median box radii at the gate's
2.10 m median width and costs under 10 percent of the scored area.

**Matching.** Boxes from different phases are the same crown at IoU >=
`MATCH_IOU`, default 0.5, clustered greedily from the highest scoring box
outward, at most one box per phase per cluster. Cluster support is then a clean
1 to 16 count, and the headline number is what fraction of crowns survive all
16 grid positions.

**Tunable constants.** `NMS_IOU`, `MATCH_IOU` and `CORE_INSET` are top level
constants in `phase_sweep.py`. `MATCH_IOU` can be varied to 0.3 or 0.4 without
touching any logic.

**Score floor.** `SCORE_THRESHOLD` is `None`, no extra filter. DeepForest's own
config `score_thresh` is 0.1, so 0.1 is the effective floor regardless.

## Before the sweep runs

`check_tiler_vs_predict_tile.py` must pass. It runs the hand rolled tiler at
offset 0 and `predict_tile()` on the same bare 1000 px window, with patch size,
overlap and NMS threshold held identical, and reports detection counts from
each plus how many boxes match at IoU 0.5. If the hand rolled path does not
reproduce `predict_tile()` at zero offset, every phase result afterwards is
untrustworthy.

The check deliberately uses no margin, because `predict_tile()` has no concept
of one. `phase_sweep.py` prints its own gate drift comparison at phase (0, 0)
instead.

## Sweep result, first run

Core region only. `phase_sweep.py` ran at `MATCH_IOU` 0.5. The matching was
then redone at 0.4 and 0.3 by `check_match_sensitivity.py`, which reads the
saved per phase CSVs and loads no model.

| match IoU | distinct crowns | found in all 16 | pct | found in exactly 1 | pct | found in 2 to 15 | pct | median support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 591 | 139 | 23.52 | 129 | 21.83 | 323 | 54.65 | 4.0 |
| 0.4 | 651 | 127 | 19.51 | 167 | 25.65 | 357 | 54.84 | 4.0 |
| 0.5 | 710 | 115 | 16.20 | 196 | 27.61 | 399 | 56.20 | 4.0 |

The support histogram is U shaped at every threshold: a pile at 1, a trough
from 5 to 14, a spike at 16, with a local peak at 4.

### Two facts that do not depend on the threshold

1. **The 2 to 15 band sits near 55 percent at every threshold**, 54.65, 54.84
   and 56.20 percent. Most crowns are neither fully stable nor one offs. They
   appear at some grid positions and not others.
2. **Median support is exactly 4.0 at every threshold.** The typical crown is
   found at a quarter of the grid positions.

Neither moves when the matching threshold moves, so neither is an artefact of
where the matching line was drawn.

### What is threshold sensitive

The singleton count is not stable. It runs 129, 167, 196 as the threshold
tightens from 0.3 to 0.5, and singletons account for 56.3 percent of the drop
in distinct crowns from 0.5 to 0.3. Some of the singleton pile at 0.5 is one
crown split across phases by a strict threshold.

**Quote IoU 0.3 as the conservative case.** At 0.3, 129 crowns, 21.83 percent,
are found at exactly one of 16 grid positions. That is the floor. The looser
the threshold, the harder it is to call a singleton an artefact.

The percentage found in all 16 is threshold sensitive in the other direction,
23.52 at 0.3 down to 16.20 at 0.5. The conservative claim there is 16.20
percent.

### Visual check

Read from `support_dx000_dy000_iou05.png`: the singletons sit on real crowns,
not on bare ground or shadow. They are flickering detections, not junk.

A second impression from that figure, that the unstable boxes are larger than
the stable ones, **was wrong and the measurements refuted it.** See the
correction under Support structure results. Unstable boxes are smaller and more
elongated. They look larger by eye because they are more elongated, and a few
long boxes spanning several crowns draw attention away from the many small
ones.

That observation is the reason for `analyse_support.py`. The working hypothesis
was that ambiguous multi crown groupings resolve differently depending on where
the seam falls, which predicted large malformed unstable boxes. The measured
geometry points the other way: unstable detections are small elongated slivers,
which points at crowns severed by a seam rather than at groupings resolving
differently. `analyse_mechanism.py` tests that directly.

Caveat on that figure: it drew phase (0, 0) only, so it showed 17 of 196
singletons, and from the 25 tile regime. `draw_support.py` now defaults to
`dx075_dy075` and also writes a pooled figure covering all 16 phases.

### Per phase count spread

The spread quoted from the first run, min 262, max 288, cv 0.0269, was taken
over all 16 phases. It therefore mixes two tiling regimes, see the next
section. `analyse_support.py` recomputes it over the 15 four tile phases with
phase (0, 0) quoted separately. **Quote the 15 phase spread as the
experiment's spread.**

## Phase 0 is a different tiling regime

**Do not pool phase (0, 0) with the other 15 phases without saying so.**

Phase 0 fits 5 tile origins per axis. Every other phase fits 4. The canvas is
1600 px, the patch is 400 px and the stride is 300 px, so origins run
`range(phase, 1201, 300)`. At phase 0 that yields 0, 300, 600, 900, 1200. At
phase 75 the fifth origin would be 1275, which overruns, so only four are
emitted. Same for 150 and 225.

That is 25 tiles at phase 0 against 16 elsewhere, and the extra tiles raise how
many times each scored pixel is seen.

| phase | tiles | mean tile coverage, scored | mean tile coverage, core |
| --- | --- | --- | --- |
| 0 | 25 | 1.9600 | 1.8726 |
| 75 | 16 | 1.6900 | 1.7313 |
| 150 | 16 | 1.6900 | 1.7313 |
| 225 | 16 | 1.6900 | 1.7313 |

Predicted excess coverage at phase 0 over the others is 16.0 percent in the
scored region and 8.2 percent in the core. Observed excess in detections is
14.7 percent scored, 344 against roughly 300, and 5.5 percent core, 288 against
roughly 273.

Both the direction and the rough magnitude track, and the excess shrinks from
scored to core in the prediction exactly as it does in the observation. That is
consistent with tile count being the cause. It is not a controlled test: the
tile count was not varied independently of the phase, so this is corroboration
rather than proof.

Consequences:

- Phase 0 sees more of the scored region more often, so it has more chances to
  detect a given crown and more chances to emit a false positive.
- Pooling it with the other 15 mixes two tiling regimes. Any per phase spread
  quoted across all 16, including the cv of 0.0269, carries that mixture.
- A clean version of the experiment would either hold tile count fixed across
  phases, for example by choosing a canvas size that admits the same count at
  every offset, or report phase 0 separately.

## Terminology

Following Miller, Dayoub, Milford and Sunderhauf, *Evaluating Merging
Strategies for Sampling-based Uncertainty Techniques in Object Detection*,
ICRA 2019, arXiv 1809.06006.

| term | meaning here |
| --- | --- |
| sample | one run of the detector over the scene, that is one grid phase. 16 samples. |
| observation | one detection box produced by one sample. |
| cluster | a set of observations from different samples judged to be the same underlying object. |
| support | the number of samples contributing an observation to a cluster, 1 to 16. |

Our clustering is **BSAS** in their taxonomy, Basic Sequential Algorithmic
Scheme, with **intra sample exclusivity**: at most one observation per sample
per cluster. That is their best performing configuration. Their semantic
affinity component does not apply, this problem has one class.

Their spatial affinity threshold is IoU 0.95, appropriate for MC Dropout
samples over an identical image where boxes are near coincident. Our samples
come from different tilings, so boxes legitimately shift, which is why our
threshold is far lower. The 0.3 to 0.5 sweep is the response to their finding
that results are sensitive to that threshold.

Variable names in the code are unchanged. This is documentation only.

## Related work

- **Miller, Dayoub, Milford and Sunderhauf, ICRA 2019, arXiv 1809.06006.**
  Merging strategies for sampling based uncertainty in object detection.
  Source of the clustering taxonomy, the intra sample exclusivity rule, and
  the spatial variance measure used here.
- **Zhang and Wang, 2016, arXiv 1611.06467,** *On The Stability of Video
  Detection and Tracking*. Showed that the stability metric has low
  correlation with the accuracy metric.
- **Tung et al., 2022, IEEE Multimedia, arXiv 2207.13890,** *Why Accuracy Is
  Not Enough: The Need for Consistency in Object Detection*. Reported
  consistency between 83.2 and 97.1 percent on video.

Those two establish that aggregate metrics hide instability. **Our
contribution is narrower: the source of the instability is an undisclosed
inference parameter that practitioners control and never report.**

## Support structure results

At `MATCH_IOU` 0.5, core region only.

### Geometry by support band

| band | median width | median area | median aspect | above aspect 2 |
| --- | --- | --- | --- | --- |
| support 1 | 1.59 m | 2.16 m2 | 1.72 | 43.88 percent |
| support 16 | 2.49 m | 6.36 m2 | 1.05 | 0.00 percent |

Rank correlation with support:

| quantity | rho | reading |
| --- | --- | --- |
| width | +0.3043 | stable crowns are LARGER |
| area | +0.3605 | stable crowns are LARGER |
| aspect | -0.4597 | stable crowns are LESS elongated |

**Correction on the record.** The hypothesis going in, taken from a visual
read of the first figure, was that unstable detections are larger and more
malformed. **The size half is refuted.** Stable detections are larger, not
smaller. The shape half is confirmed. Unstable detections are small and
elongated: slivers, which is what a crown severed by a tile seam should look
like.

### Support 4

| quantity | value |
| --- | --- |
| crowns | 86 |
| structured, not scattered | 74, 86.05 percent |
| null expectation for structured | 2.42 percent |
| all_dy_one_dx, sensitive to dx | 40 |
| all_dx_one_dy, sensitive to dy | 31 |
| median aspect | 2.23, the highest of any support level |
| median area | 1.67 m2, the smallest of any support level |

The null is exact, every 4 subset of the 4 by 4 grid enumerated. 86.05 percent
against 2.42 percent is not a chance pattern.

### Axis marginals

| quantity | value |
| --- | --- |
| mean distinct dx per crown | 2.4141 |
| mean distinct dy per crown | 2.4704 |
| sign test on discordant pairs, two sided p | 0.313 |

Joint table mass sits in the corners: 196 at (1, 1), 201 at (4, 4), 40 and 31
at the off corners, middle nearly empty.

### How to state this finding

**Not as anisotropy.** There is no evidence that one axis matters more overall.
The two single axis classes are roughly balanced, 40 against 31, and the axis
marginals show no global bias at p 0.313.

**State it as: each crown is sensitive to exactly one axis, and which axis
varies by crown.** The corner heavy joint table says the same thing. A crown
is either robust to both axes or hostage to one, with very little in between.

### Per phase count spread

| set | n_core |
| --- | --- |
| 15 four tile phases | mean 273.80, cv 0.0245 |
| phase (0, 0), 25 tiles | 288, 5.19 percent above the 15 phase mean |

**Quote cv 0.0245 over the 15 four tile phases as the experiment's spread.**
The earlier figure, cv 0.0269 over all 16, mixed two tiling regimes.

The observed 5.19 percent excess sits below the 8.2 percent predicted from
mean tile coverage in the core. Direction and rough magnitude still agree.

### Still to be produced

`analyse_mechanism.py` has not been run. Pending:

- elongation axis against sensitivity axis, with Fisher exact test
- seam distance and containment margin at the samples found against missed
- spatial variance per cluster, and its axis split

Also not recorded above: the 2 to 15 band row of the geometry table, which was
not in the output that was pasted back.
