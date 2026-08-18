"""
Tile grid phase sweep.

Question: at a fixed overlap ratio, does the *position* of the tiling grid
change which tree crowns DeepForest detects?

Method
------
The tiling grid is laid down 16 times, once per phase offset (dx, dy) drawn
from a 4 x 4 sub stride grid. Tile size and overlap ratio, and therefore
stride, are identical in every run. The only thing that changes is where the
grid starts.

DeepForest's predict_tile() does not expose a grid offset, so the tiler here
is hand rolled: tiles are cut at the offset grid, predict_image() is called
per tile, boxes are translated back into scored window coordinates, and a
single torchvision NMS pass merges them, matching what DeepForest's own
mosaic() does across windows.

Margin
------
Every tile must be exactly PATCH_SIZE px. DeepForest is sensitive to input
size, and if tile size co varied with phase the two effects could not be
separated. So the grid runs over a canvas larger than the scored window.

That margin is REAL imagery read from the orthomosaic, not reflected or
padded pixels. An earlier version reflect padded the window, which put
synthetic pixels inside tiles that reached up to 325 px into the scored
region at some phases. That is gone. Alpha coverage over the expanded read
was verified at 100 percent on all four sides by check_expanded_window.py.

Coordinate systems
------------------
Two resolutions are in play and the constants below are labelled for it.

    NATIVE      3.89 cm per px, the orthomosaic as stored
    EXPERIMENT  7.78 cm per px, NATIVE downsampled by 2

Read offsets and read sizes are NATIVE. Tiling, strides, phases, box
coordinates and the inset are EXPERIMENT. Anything ending in _NATIVE is
native, everything else is experiment resolution.

Scored region
-------------
The scored region is the original 2000 px NATIVE working window, which is the
inner 1000 px of the canvas at EXPERIMENT resolution. The margin is context
only and is never scored. Boxes are kept when their centre falls inside the
scored region.

Matching
--------
Cross phase matching is NOT implemented here. It lives in phase_matching.py,
which imports no model and can be rerun cheaply at other thresholds. There is
one implementation, used by this script and by every analysis script.

Outputs (all gitignored)
------------------------
phase_boxes_dx###_dy###.csv   per phase detections, scored window coordinates
phase_summary.csv             one row per phase: counts, score, box width
phase_stability.csv           one row per matched crown cluster
phase_stability_hist.csv      how many crowns were found in how many phases

Not run yet. Roughly 16 phases x 16 to 25 tiles, so 300 to 400 predict_image
calls.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from PIL import Image
from torchvision.ops import nms as tv_nms
import torch
from deepforest import main

# Cross phase matching lives in phase_matching.py, which imports no model and
# can therefore be rerun cheaply at other thresholds. There is exactly one
# implementation of the clustering, here and in every analysis script.
import phase_matching as pm

# =========================================================================
# CONSTANTS
# =========================================================================

PATH = r"C:\Users\gabpe\Downloads\000103_ortho-dsm-ptcloud.tif"

# --- resolution ----------------------------------------------------------
NATIVE_GSD_CM = 3.89
DOWNSAMPLE = 2                              # NATIVE -> EXPERIMENT
GSD_CM = NATIVE_GSD_CM * DOWNSAMPLE         # 7.78, EXPERIMENT

# --- scored working window, NATIVE px (settled, see README) --------------
WIN_COL_OFF_NATIVE = 4820
WIN_ROW_OFF_NATIVE = 5260
WIN_SIZE_NATIVE = 2000

# --- tiling, EXPERIMENT px ----------------------------------------------
PATCH_SIZE = 400
PATCH_OVERLAP = 0.25
STRIDE = int(PATCH_SIZE * (1 - PATCH_OVERLAP))          # 300, EXPERIMENT

# --- margin of real imagery ---------------------------------------------
MARGIN = STRIDE                                          # 300, EXPERIMENT
MARGIN_NATIVE = MARGIN * DOWNSAMPLE                      # 600, NATIVE

# --- expanded read, NATIVE px -------------------------------------------
EXP_COL_OFF_NATIVE = WIN_COL_OFF_NATIVE - MARGIN_NATIVE  # 4220, NATIVE
EXP_ROW_OFF_NATIVE = WIN_ROW_OFF_NATIVE - MARGIN_NATIVE  # 4660, NATIVE
EXP_SIZE_NATIVE = WIN_SIZE_NATIVE + 2 * MARGIN_NATIVE    # 3200, NATIVE

# --- derived, EXPERIMENT px ---------------------------------------------
WIN_SIZE = WIN_SIZE_NATIVE // DOWNSAMPLE                 # 1000, EXPERIMENT
CANVAS_SIZE = EXP_SIZE_NATIVE // DOWNSAMPLE              # 1600, EXPERIMENT

# --- phase sweep, EXPERIMENT px -----------------------------------------
PHASES_PER_AXIS = 4
PHASE_STEP = STRIDE // PHASES_PER_AXIS                   # 75
PHASE_OFFSETS = [i * PHASE_STEP for i in range(PHASES_PER_AXIS)]

# --- thresholds ----------------------------------------------------------
# Cross window merge. DeepForest predict_tile passes iou_threshold=0.15 into
# mosaic(), which calls torchvision.ops.nms. Same value, same function here.
NMS_IOU = 0.15

# Two boxes in different phases are treated as the same crown at or above
# this IoU. Vary to 0.3 or 0.4 without touching any logic.
MATCH_IOU = 0.5

# Additional score floor applied on top of DeepForest's own. Set to None for
# none. NOTE: DeepForest's config score_thresh is 0.1, so 0.1 is the real
# floor regardless of what this is set to. Setting this below 0.1 does
# nothing.
SCORE_THRESHOLD = None

# Nominal inset from the scored boundary, EXPERIMENT px. This is NOT a
# contamination guard any more, the real pixel margin removed that need. It
# exists because a crown sitting on the scored boundary can have its centre
# fall inside at one phase and outside at the next, which would register as
# instability that is really just a boundary artefact. 25 px is about 1.9 m,
# roughly two median box radii at the gate's 2.10 m median width, comfortably
# larger than any plausible centre jitter between phases, and it costs under
# 10 percent of the scored area (950 x 950 of 1000 x 1000).
CORE_INSET = 25

# --- gate reference, for drift detection only ---------------------------
# Recorded in the README from run_gate.py at the working window, 7.78 cm.
# Phase (0, 0) here will be NEAR these but not identical: the gate had no
# margin, so its edge tiles saw no context. Exact reproduction of predict_tile
# is what check_tiler_vs_predict_tile.py tests, on a bare window with no
# margin.
GATE_N = 311
GATE_MEDIAN_SCORE = 0.348
GATE_MEDIAN_WIDTH_M = 2.10

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

assert CANVAS_SIZE == WIN_SIZE + 2 * MARGIN, "canvas geometry inconsistent"

# phase_matching.py restates the geometry it needs so it can run standalone.
# Guard the two copies against drifting apart.
assert pm.WIN_SIZE == WIN_SIZE, "WIN_SIZE differs from phase_matching"
assert pm.CORE_INSET == CORE_INSET, "CORE_INSET differs from phase_matching"
assert pm.GSD_CM == GSD_CM, "GSD_CM differs from phase_matching"
assert pm.PHASE_OFFSETS == PHASE_OFFSETS, "phase grid differs from phase_matching"
assert pm.N_PHASES == PHASES_PER_AXIS ** 2, "phase count differs"
assert pm.PATCH_SIZE == PATCH_SIZE, "PATCH_SIZE differs from phase_matching"
assert pm.STRIDE == STRIDE, "STRIDE differs from phase_matching"
assert pm.MARGIN == MARGIN, "MARGIN differs from phase_matching"
assert pm.CANVAS_SIZE == CANVAS_SIZE, "CANVAS_SIZE differs from phase_matching"
# the tile grid itself is compared inside run_sweep(), once tile_origins exists


# =========================================================================
# geometry helpers
# =========================================================================

def merge_nms(coords, scores, iou_thr):
    """Cross window merge, identical call to DeepForest's mosaic()."""
    keep = tv_nms(
        boxes=torch.tensor(coords, dtype=torch.float32),
        scores=torch.tensor(scores, dtype=torch.float32),
        iou_threshold=iou_thr,
    ).numpy()
    return keep


def tile_origins(phase, canvas_size):
    """Grid origins on the canvas for one axis at a given phase.

    EXPERIMENT px. MARGIN is exactly one stride, so a canvas origin congruent
    to `phase` mod STRIDE puts the grid at phase `phase` in scored window
    coordinates. Only full size tiles are emitted, and because the margin is
    real imagery every one of them is real.
    """
    return list(range(phase, canvas_size - PATCH_SIZE + 1, STRIDE))


# =========================================================================
# load the canvas
# =========================================================================

def load_canvas():
    """Read the expanded region and downsample it.

    Read is NATIVE, result is EXPERIMENT. Deliberately mirrors run_gate.py:
    native read, then PIL bilinear resize. Resampling inside rasterio instead
    would give slightly different pixels and break comparability with the gate
    numbers in the README.
    """
    win = Window(
        EXP_COL_OFF_NATIVE, EXP_ROW_OFF_NATIVE,
        EXP_SIZE_NATIVE, EXP_SIZE_NATIVE,
    )
    with rasterio.open(PATH) as src:
        rgb = src.read([1, 2, 3], window=win)
    native = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)
    return np.array(
        Image.fromarray(native).resize(
            (CANVAS_SIZE, CANVAS_SIZE), Image.BILINEAR
        )
    )


# =========================================================================
# one phase
# =========================================================================

def run_phase(model, canvas, dx, dy):
    """Tile the canvas at phase (dx, dy), predict, stitch, merge.

    All coordinates EXPERIMENT px. Canvas coordinates are converted to scored
    window coordinates by subtracting MARGIN.
    """
    rows = []
    xs = tile_origins(dx, canvas.shape[1])
    ys = tile_origins(dy, canvas.shape[0])

    for y0 in ys:
        for x0 in xs:
            tile = canvas[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            # predict_image wants float32, channels last, values 0 to 255
            preds = model.predict_image(image=tile.astype(np.float32))
            if preds is None or len(preds) == 0:
                continue
            preds = preds.copy()
            # tile coords -> canvas coords -> scored window coords
            preds["xmin"] = preds["xmin"] + x0 - MARGIN
            preds["xmax"] = preds["xmax"] + x0 - MARGIN
            preds["ymin"] = preds["ymin"] + y0 - MARGIN
            preds["ymax"] = preds["ymax"] + y0 - MARGIN
            preds["tile_x"] = x0 - MARGIN
            preds["tile_y"] = y0 - MARGIN
            rows.append(preds)

    empty = pd.DataFrame(
        columns=["xmin", "ymin", "xmax", "ymax", "score", "label",
                 "tile_x", "tile_y", "cx", "cy", "width_px", "width_m",
                 "dx", "dy"]
    )
    if not rows:
        return empty

    boxes = pd.concat(rows, ignore_index=True)

    # keep boxes whose centre falls inside the scored window. The margin is
    # context only and is never scored.
    cx = (boxes["xmin"] + boxes["xmax"]) / 2
    cy = (boxes["ymin"] + boxes["ymax"]) / 2
    boxes = boxes[(cx >= 0) & (cx < WIN_SIZE) & (cy >= 0) & (cy < WIN_SIZE)]

    if SCORE_THRESHOLD is not None:
        boxes = boxes[boxes["score"] >= SCORE_THRESHOLD]
    if len(boxes) == 0:
        return empty

    coords = boxes[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)
    scores = boxes["score"].to_numpy(dtype=np.float32)
    boxes = boxes.iloc[merge_nms(coords, scores, NMS_IOU)].reset_index(drop=True)

    boxes["cx"] = (boxes["xmin"] + boxes["xmax"]) / 2
    boxes["cy"] = (boxes["ymin"] + boxes["ymax"]) / 2
    boxes["width_px"] = boxes["xmax"] - boxes["xmin"]
    boxes["width_m"] = boxes["width_px"] * GSD_CM / 100.0
    boxes["dx"] = dx
    boxes["dy"] = dy
    return boxes


# =========================================================================
# main
# =========================================================================

def run_sweep():
    print("expanded read NATIVE : col_off", EXP_COL_OFF_NATIVE,
          "row_off", EXP_ROW_OFF_NATIVE, "size", EXP_SIZE_NATIVE)
    print("canvas EXPERIMENT    :", CANVAS_SIZE, "px at",
          f"{GSD_CM:.2f} cm, margin", MARGIN, "px of real imagery")
    print("scored EXPERIMENT    :", WIN_SIZE, "px, inset", CORE_INSET)

    # the analysis scripts locate tile seams from phase_matching's copy of the
    # grid. Confirm it is the same grid this script actually cuts.
    for p in PHASE_OFFSETS:
        assert pm.scored_tile_origins(p) == [
            o - MARGIN for o in tile_origins(p, CANVAS_SIZE)
        ], f"phase_matching tile grid differs at phase {p}"

    canvas = load_canvas()
    assert canvas.shape[0] == CANVAS_SIZE and canvas.shape[1] == CANVAS_SIZE, \
        f"canvas is {canvas.shape}, expected {CANVAS_SIZE} square"

    model = main.deepforest()
    model.load_model("weecology/deepforest-tree")

    summaries = []
    pool = []

    for dy in PHASE_OFFSETS:
        for dx in PHASE_OFFSETS:
            tag = f"dx{dx:03d}_dy{dy:03d}"
            boxes = run_phase(model, canvas, dx, dy)
            boxes.to_csv(
                os.path.join(OUT_DIR, f"phase_boxes_{tag}.csv"), index=False
            )

            if len(boxes):
                core = boxes[
                    (boxes["cx"] >= CORE_INSET)
                    & (boxes["cx"] < WIN_SIZE - CORE_INSET)
                    & (boxes["cy"] >= CORE_INSET)
                    & (boxes["cy"] < WIN_SIZE - CORE_INSET)
                ]
            else:
                core = boxes

            summaries.append({
                "phase_id": tag,
                "dx": dx,
                "dy": dy,
                "n_tiles": len(tile_origins(dx, CANVAS_SIZE))
                           * len(tile_origins(dy, CANVAS_SIZE)),
                "n_scored": len(boxes),
                "n_core": len(core),
                "median_score": round(float(boxes["score"].median()), 4)
                                if len(boxes) else np.nan,
                "median_width_px": round(float(boxes["width_px"].median()), 2)
                                   if len(boxes) else np.nan,
                "median_width_m": round(float(boxes["width_m"].median()), 3)
                                  if len(boxes) else np.nan,
            })
            print(tag, "->", len(boxes), "scored,", len(core), "in core")

            if dx == 0 and dy == 0 and len(boxes):
                print("")
                print("--- gate drift check, phase (0, 0) vs README ---")
                print("detections   :", len(boxes), "vs gate", GATE_N)
                print("median score :",
                      round(float(boxes["score"].median()), 3),
                      "vs gate", GATE_MEDIAN_SCORE)
                print("median width :",
                      round(float(boxes["width_m"].median()), 2),
                      "m vs gate", GATE_MEDIAN_WIDTH_M, "m")
                print("expect near, not identical: the gate had no margin.")
                print("")

            if len(core):
                c = core.copy()
                c["phase_id"] = tag
                pool.append(c)

    summary = pd.DataFrame(summaries)
    summary.to_csv(os.path.join(OUT_DIR, "phase_summary.csv"), index=False)

    print("")
    print("=== per phase detection counts ===")
    print(summary[["phase_id", "n_scored", "n_core",
                   "median_score", "median_width_m"]].to_string(index=False))
    print("")
    print("core count spread: min", summary["n_core"].min(),
          "max", summary["n_core"].max(),
          "cv", round(summary["n_core"].std() / summary["n_core"].mean(), 4))

    if not pool:
        print("no core detections, nothing to match")
        return

    pooled = pd.concat(pool, ignore_index=True)
    n_phases = PHASES_PER_AXIS ** 2
    # one implementation of the clustering, shared with every analysis script
    clusters, _ = pm.cluster_across_phases(pooled, MATCH_IOU, n_phases)
    clusters.to_csv(os.path.join(OUT_DIR, "phase_stability.csv"), index=False)

    hist = pm.support_histogram(clusters, n_phases)
    hist.to_csv(os.path.join(OUT_DIR, "phase_stability_hist.csv"), index=False)

    print("")
    print("=== cross phase stability, core region only ===")
    print("match IoU:", MATCH_IOU)
    print("distinct crowns:", len(clusters))
    print("found in all", n_phases, "phases:",
          int(clusters["found_in_all"].sum()),
          f"({clusters['found_in_all'].mean():.1%})")
    print("found in only one phase:", int((clusters["n_phases"] == 1).sum()))
    print("")
    print(hist.to_string(index=False))


if __name__ == "__main__":
    run_sweep()
