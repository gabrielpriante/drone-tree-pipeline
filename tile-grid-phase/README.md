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
