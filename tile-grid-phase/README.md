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
| `draw_support.py` | Draws the core region at one phase with crowns coloured by support band, to see whether singletons are plausible trees or junk on bare ground. |

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

At `MATCH_IOU` 0.5, core region only.

| Quantity | Value |
| --- | --- |
| core count per phase | min 262, max 288, cv 0.0269 |
| distinct crowns | 710 |
| found in all 16 phases | 115, 16.2 percent |
| found in exactly 1 phase | 196, 27.6 percent |

The support histogram is U shaped: a pile at 1, a trough from 5 to 14, a spike
at 16.

Two open questions on that result, neither settled.

1. Whether the 196 singletons are a clustering artefact. If a crown shifts
   between phases and falls below IoU 0.5, one real crown splits into several
   one phase clusters, inflating both the 710 and the 196.
   `check_match_sensitivity.py` tests this at 0.3 and 0.4.
2. Whether the singletons are plausible trees that flicker or junk on bare
   ground and shadow. Those are different findings. `draw_support.py` is the
   visual check.

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
