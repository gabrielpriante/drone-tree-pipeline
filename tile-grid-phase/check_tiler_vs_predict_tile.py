"""
Zero offset sanity check: hand rolled tiler vs DeepForest predict_tile.

Why this exists
---------------
phase_sweep.py cannot use predict_tile, because predict_tile exposes no grid
offset. It cuts tiles itself, calls predict_image per tile, and merges. If
that hand rolled path does not reproduce predict_tile at offset 0, every
phase result afterwards is untrustworthy. This check has to pass before the
sweep is worth running.

What is held identical
----------------------
    patch size      PATCH_SIZE
    overlap ratio   PATCH_OVERLAP
    merge NMS       NMS_IOU, via torchvision.ops.nms in both paths
    image           byte identical, both read the same PNG
    model           same weights, same instance

What is deliberately NOT the phase_sweep setup
----------------------------------------------
No margin. This runs on the bare 1000 px scored window, the same image
run_gate.py used, because predict_tile has no concept of a margin. Adding one
would make the two paths incomparable. phase_sweep.py's real pixel margin is
tested by its own gate drift print, not here.

Tile origins
------------
On a bare window the last tile would overrun the edge, so it is clamped back
to size minus patch. That is what DeepForest's window generation does, and
matching it is the point of the check. Origins from both paths are printed so
a mismatch is visible rather than inferred.

Interpreting the result
-----------------------
    counts equal and nearly all boxes match at IoU 0.5
        the tiler is sound, proceed to the sweep
    counts equal but many boxes unmatched
        coordinate translation is wrong somewhere
    counts differ by a few
        likely NMS tie ordering, check the unmatched boxes by hand
    counts differ a lot, or origins differ
        window generation is wrong, fix before the sweep

Read only in the sense that it writes one PNG and prints. Runs the model
twice over the same window. Not run yet.
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

PATH = r"C:\Users\gabpe\Downloads\000103_ortho-dsm-ptcloud.tif"

# --- scored working window, NATIVE px (settled) --------------------------
WIN_COL_OFF_NATIVE = 4820
WIN_ROW_OFF_NATIVE = 5260
WIN_SIZE_NATIVE = 2000

# --- resolution ----------------------------------------------------------
NATIVE_GSD_CM = 3.89
DOWNSAMPLE = 2
GSD_CM = NATIVE_GSD_CM * DOWNSAMPLE                  # 7.78
WIN_SIZE = WIN_SIZE_NATIVE // DOWNSAMPLE             # 1000, EXPERIMENT

# --- tiling, EXPERIMENT px. Must match phase_sweep.py --------------------
PATCH_SIZE = 400
PATCH_OVERLAP = 0.25
STRIDE = int(PATCH_SIZE * (1 - PATCH_OVERLAP))       # 300

# --- thresholds. Must match phase_sweep.py -------------------------------
NMS_IOU = 0.15        # predict_tile iou_threshold, and our merge
MATCH_IOU = 0.5       # box A and box B are the same detection

# --- gate reference from the README --------------------------------------
GATE_N = 311
GATE_MEDIAN_SCORE = 0.348
GATE_MEDIAN_WIDTH_M = 2.10

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_PNG = os.path.join(OUT_DIR, "sanity_window.png")


# =========================================================================
# helpers
# =========================================================================

def iou_matrix(a, b):
    """(Na, 4) against (Nb, 4), returns (Na, Nb) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aa = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    ab = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = aa + ab - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(a, b, thr):
    """One to one greedy matching by descending IoU. Returns matched pairs."""
    m = iou_matrix(a, b)
    pairs = []
    if m.size == 0:
        return pairs
    flat = np.dstack(np.unravel_index(np.argsort(m, axis=None)[::-1], m.shape))[0]
    used_a, used_b = set(), set()
    for i, j in flat:
        if m[i, j] < thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        pairs.append((int(i), int(j), float(m[i, j])))
    return pairs


def bare_tile_origins(size, patch, stride):
    """Origins on a bare window, last tile clamped back to fit.

    DeepForest's window generation does the same: it does not emit a short
    final tile, it slides the last full tile back against the edge.
    """
    origins = list(range(0, size - patch + 1, stride))
    last = size - patch
    if origins[-1] != last:
        origins.append(last)
    return origins


def load_scored_window():
    """Exactly what run_gate.py did: NATIVE read, PIL bilinear to EXPERIMENT."""
    win = Window(
        WIN_COL_OFF_NATIVE, WIN_ROW_OFF_NATIVE,
        WIN_SIZE_NATIVE, WIN_SIZE_NATIVE,
    )
    with rasterio.open(PATH) as src:
        rgb = src.read([1, 2, 3], window=win)
    native = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)
    return np.array(
        Image.fromarray(native).resize((WIN_SIZE, WIN_SIZE), Image.BILINEAR)
    )


def hand_rolled(model, img):
    """The phase_sweep.py tiling path, at offset 0, on a bare window."""
    origins = bare_tile_origins(img.shape[0], PATCH_SIZE, STRIDE)
    rows = []
    for y0 in origins:
        for x0 in origins:
            tile = img[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            preds = model.predict_image(image=tile.astype(np.float32))
            if preds is None or len(preds) == 0:
                continue
            preds = preds.copy()
            preds["xmin"] = preds["xmin"] + x0
            preds["xmax"] = preds["xmax"] + x0
            preds["ymin"] = preds["ymin"] + y0
            preds["ymax"] = preds["ymax"] + y0
            rows.append(preds)

    if not rows:
        return pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score"]), origins

    boxes = pd.concat(rows, ignore_index=True)
    coords = boxes[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)
    scores = boxes["score"].to_numpy(dtype=np.float32)
    keep = tv_nms(
        boxes=torch.tensor(coords),
        scores=torch.tensor(scores),
        iou_threshold=NMS_IOU,
    ).numpy()
    return boxes.iloc[keep].reset_index(drop=True), origins


def describe(name, boxes):
    if len(boxes) == 0:
        print(name, ": 0 detections")
        return
    w = boxes["xmax"] - boxes["xmin"]
    print(f"{name:14s}: {len(boxes):5d} detections   "
          f"median score {boxes['score'].median():.3f}   "
          f"median width {w.median() * GSD_CM / 100.0:.2f} m")


# =========================================================================
# main
# =========================================================================

def run_check():
    print("scored window:", WIN_SIZE, "px at", f"{GSD_CM:.2f} cm, no margin")
    print("patch", PATCH_SIZE, "overlap", PATCH_OVERLAP,
          "stride", STRIDE, "nms iou", NMS_IOU)
    print("")

    img = load_scored_window()
    Image.fromarray(img).save(WINDOW_PNG)
    print("wrote", os.path.basename(WINDOW_PNG))

    model = main.deepforest()
    model.load_model("weecology/deepforest-tree")

    # --- path A, DeepForest ---------------------------------------------
    ref = model.predict_tile(
        path=WINDOW_PNG,
        patch_size=PATCH_SIZE,
        patch_overlap=PATCH_OVERLAP,
        iou_threshold=NMS_IOU,
    )
    if ref is None:
        ref = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score"])

    # --- path B, ours ----------------------------------------------------
    ours, origins = hand_rolled(model, img)

    print("")
    print("our tile origins per axis:", origins,
          f"({len(origins)} x {len(origins)} = {len(origins) ** 2} tiles)")
    print("")
    describe("predict_tile", ref)
    describe("hand rolled", ours)
    print("")
    print("count difference:", len(ours) - len(ref))

    # --- overlap ---------------------------------------------------------
    a = ref[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    b = ours[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    pairs = greedy_match(a, b, MATCH_IOU)

    print("")
    print("=== box level agreement at IoU", MATCH_IOU, "===")
    print("matched            :", len(pairs))
    print("predict_tile only  :", len(ref) - len(pairs))
    print("hand rolled only   :", len(ours) - len(pairs))
    if len(ref):
        print("recall of predict_tile:", f"{len(pairs) / len(ref):.1%}")
    if pairs:
        ious = np.array([p[2] for p in pairs])
        print("matched IoU median :", round(float(np.median(ious)), 4))
        print("matched IoU min    :", round(float(ious.min()), 4))

    # --- gate reference --------------------------------------------------
    print("")
    print("=== gate reference from README, run_gate.py at 7.78 cm ===")
    print("gate:", GATE_N, "detections, median score", GATE_MEDIAN_SCORE,
          ", median width", GATE_MEDIAN_WIDTH_M, "m")
    print("predict_tile here should match the gate closely. It is the same")
    print("call on the same window. A large gap means the resampling path")
    print("changed, not the tiler.")

    if len(pairs) and len(ref):
        ok = (abs(len(ours) - len(ref)) <= max(2, 0.02 * len(ref))
              and len(pairs) / len(ref) >= 0.95)
    else:
        ok = False
    print("")
    print("VERDICT:", "PASS, tiler reproduces predict_tile" if ok
          else "FAIL or borderline, inspect before running the sweep")


if __name__ == "__main__":
    run_check()
