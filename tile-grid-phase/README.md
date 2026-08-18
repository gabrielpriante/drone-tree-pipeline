# Tile Grid Phase Experiment

Does the *position* of the DeepForest tiling grid — at a fixed overlap ratio —
change which tree crowns get detected? These scripts establish the raster,
pick a working window, and gate the question at a chosen resolution.

## Scripts

| Script | What it does |
| --- | --- |
| `check_raster.py` | Prints size, band count, dtype, CRS, nodata, pixel size, GSD, footprint and bounds for the orthomosaic. |
| `find_window.py` | Locates a 2000 px square window centred on the valid (alpha > 0) region of the mosaic and reports percent-valid coverage. |
| `run_gate.py` | Reads the working window, writes it at native and half resolution, runs DeepForest on both, and reports detection count, score distribution and median box width. |
| `draw_boxes.py` | Overlays the boxes from `run_gate.py` onto the window PNGs for visual inspection. |
| `phase_sweep.py` | The experiment proper. Lays the tiling grid down 16 times at a 4x4 grid of sub-stride phase offsets, holding tile size and overlap ratio fixed, and reports how much the detection set moves. Not yet run. |

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

`col_off 4820`, `row_off 5260`, `size 2000` — at native resolution.

### Experiment resolution

The experiment runs at **7.78 cm**, downsampled by a factor of 2 from native.
Justified by crown fragmentation at native resolution.

### Gate result at the working window

| Resolution | Detections | Median score | Median box width |
| --- | --- | --- | --- |
| native 3.89 cm | 1201 | 0.273 | 0.97 m |
| downsampled 7.78 cm | 311 | 0.348 | 2.10 m |

At 3.89 cm the median box is under a metre wide — narrower than a real crown at
this site — and confidence is low. Halving the resolution yields fewer,
higher-confidence, crown-sized boxes. 7.78 cm is therefore the operating point
for the phase sweep.

## Phase sweep design

`phase_sweep.py` isolates grid position as the only variable:

- **Phases.** `PATCH_SIZE` 400, `PATCH_OVERLAP` 0.25, so stride is 300 px.
  Offsets are swept at 0, 75, 150 and 225 px in both x and y — 16 positions
  covering one stride in quarter steps. Overlap ratio is identical throughout.
- **Tiling.** `predict_tile()` gives no control over grid offset, so tiles are
  cut by hand and `predict_image()` is called per tile, with boxes translated
  back to window coordinates and merged by a single NMS pass at IoU 0.15
  (DeepForest's own mosaic default).
- **Constant tile size.** The window is reflect-padded by one stride before
  tiling so every tile is exactly 400 px under every phase. Input size must not
  co-vary with phase, or the two effects cannot be separated.
- **Core region.** The padded margin is synthetic, so cross-phase matching only
  considers boxes whose centre is at least `CORE_INSET` (100 px, ~7.8 m) from
  the window edge.
- **Matching.** Boxes from different phases are the same crown at IoU >= 0.5,
  clustered greedily from the highest-scoring box outward, at most one box per
  phase per cluster. Cluster support is then a clean 1..16 count, and the
  headline number is what fraction of crowns survive all 16 grid positions.
