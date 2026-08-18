"""
Tile grid phase sweep.

Question: at a fixed overlap ratio, does the *position* of the tiling grid
change which tree crowns DeepForest detects?

Method
------
The working window is read at the settled experiment resolution (7.78 cm,
downsampled by 2 from native). The tiling grid is then laid down 16 times,
once per phase offset (dx, dy) drawn from a 4 x 4 sub-stride grid. Tile size
and overlap ratio -- and therefore stride -- are identical in every run. The
only thing that changes is where the grid starts.

DeepForest's predict_tile() does not expose a grid offset, so the tiler here
is hand-rolled: tiles are cut at the offset grid, predict_image() is called
per tile, boxes are translated back into window coordinates, and a single NMS
pass merges them. To keep every tile exactly PATCH_SIZE px -- input size is
something DeepForest is sensitive to, and it must not co-vary with phase --
the window is reflect-padded by one stride on all sides before tiling. The
padded margin is fake imagery, so the cross-phase stability analysis is
restricted to a core region inset from the window edge.

Outputs (all gitignored)
------------------------
phase_boxes_dx###_dy###.csv   per-phase detections, window coordinates
phase_summary.csv             one row per phase: counts, score, box width
phase_stability.csv           one row per matched crown cluster
phase_stability_hist.csv      how many crowns were found in how many phases

Not run yet. Roughly 16 phases x ~25 tiles = ~400 predict_image calls.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from PIL import Image
from deepforest import main

# --- raster and working window (settled, see README) ---------------------
PATH = r"C:\Users\gabpe\Downloads\000103_ortho-dsm-ptcloud.tif"
COL_OFF = 4820
ROW_OFF = 5260
SIZE = 2000              # native px
DOWNSAMPLE = 2           # 3.89 cm -> 7.78 cm
GSD_CM = 3.89 * DOWNSAMPLE

# --- tiling ---------------------------------------------------------------
PATCH_SIZE = 400
PATCH_OVERLAP = 0.25
STRIDE = int(PATCH_SIZE * (1 - PATCH_OVERLAP))   # 300

# --- phase sweep ----------------------------------------------------------
PHASES_PER_AXIS = 4
PHASE_STEP = STRIDE // PHASES_PER_AXIS           # 75
PHASE_OFFSETS = [i * PHASE_STEP for i in range(PHASES_PER_AXIS)]  # 0,75,150,225

# --- post-processing ------------------------------------------------------
NMS_IOU = 0.15           # matches DeepForest's predict_tile mosaic default
SCORE_THRESHOLD = 0.0    # keep everything; filter downstream in analysis
MATCH_IOU = 0.5          # two boxes in different phases are the same crown
CORE_INSET = 100         # px of window edge excluded from stability analysis

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================================================
# geometry helpers
# ==========================================================================

def iou_one_to_many(box, others):
    """IoU of one [xmin, ymin, xmax, ymax] against an (N, 4) array."""
    if len(others) == 0:
        return np.zeros(0)
    ix1 = np.maximum(box[0], others[:, 0])
    iy1 = np.maximum(box[1], others[:, 1])
    ix2 = np.minimum(box[2], others[:, 2])
    iy2 = np.minimum(box[3], others[:, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / union, 0.0)


def nms(boxes, scores, iou_thr):
    """Plain greedy NMS. Returns indices to keep, highest score first."""
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou_one_to_many(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_thr]
    return keep


def tile_origins(phase, canvas_size):
    """Grid origins on the padded canvas for one axis at a given phase.

    PAD is exactly one stride, so a canvas origin congruent to `phase`
    mod STRIDE puts the grid at phase `phase` in window coordinates.
    Only full-size tiles are emitted.
    """
    return list(range(phase, canvas_size - PATCH_SIZE + 1, STRIDE))


# ==========================================================================
# load the working window at experiment resolution
# ==========================================================================

def load_window():
    """Read the working window and downsample it.

    Deliberately mirrors run_gate.py: read native, then PIL bilinear resize.
    Resampling in rasterio instead would give slightly different pixels and
    break comparability with the gate numbers in the README.
    """
    win = Window(COL_OFF, ROW_OFF, SIZE, SIZE)
    with rasterio.open(PATH) as src:
        rgb = src.read([1, 2, 3], window=win)
    native = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)
    small = SIZE // DOWNSAMPLE
    return np.array(Image.fromarray(native).resize((small, small), Image.BILINEAR))


# ==========================================================================
# one phase
# ==========================================================================

def run_phase(model, canvas, window_size, dx, dy, pad):
    """Tile the padded canvas at phase (dx, dy), predict, stitch, NMS."""
    rows = []
    xs = tile_origins(dx, canvas.shape[1])
    ys = tile_origins(dy, canvas.shape[0])

    for y0 in ys:
        for x0 in xs:
            tile = canvas[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE]
            preds = model.predict_image(image=tile.astype(np.float32))
            if preds is None or len(preds) == 0:
                continue
            preds = preds.copy()
            # tile coords -> canvas coords -> window coords
            preds["xmin"] = preds["xmin"] + x0 - pad
            preds["xmax"] = preds["xmax"] + x0 - pad
            preds["ymin"] = preds["ymin"] + y0 - pad
            preds["ymax"] = preds["ymax"] + y0 - pad
            preds["tile_x"] = x0 - pad
            preds["tile_y"] = y0 - pad
            rows.append(preds)

    if not rows:
        return pd.DataFrame(
            columns=["xmin", "ymin", "xmax", "ymax", "score", "label",
                     "tile_x", "tile_y"]
        )

    boxes = pd.concat(rows, ignore_index=True)

    # drop boxes whose centre falls in the reflected margin
    cx = (boxes["xmin"] + boxes["xmax"]) / 2
    cy = (boxes["ymin"] + boxes["ymax"]) / 2
    inside = (cx >= 0) & (cx < window_size) & (cy >= 0) & (cy < window_size)
    boxes = boxes[inside]

    if SCORE_THRESHOLD > 0:
        boxes = boxes[boxes["score"] >= SCORE_THRESHOLD]
    if len(boxes) == 0:
        return boxes.reset_index(drop=True)

    coords = boxes[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    keep = nms(coords, boxes["score"].to_numpy(dtype=float), NMS_IOU)
    boxes = boxes.iloc[keep].reset_index(drop=True)

    boxes["cx"] = (boxes["xmin"] + boxes["xmax"]) / 2
    boxes["cy"] = (boxes["ymin"] + boxes["ymax"]) / 2
    boxes["width_px"] = boxes["xmax"] - boxes["xmin"]
    boxes["width_m"] = boxes["width_px"] * GSD_CM / 100.0
    boxes["dx"] = dx
    boxes["dy"] = dy
    return boxes


# ==========================================================================
# cross-phase matching
# ==========================================================================

def cluster_across_phases(pool, n_phases):
    """Greedy single-pass clustering of boxes into crowns.

    Highest-scoring unassigned box seeds a cluster; the best-IoU unassigned
    box from each *other* phase joins it if IoU >= MATCH_IOU. At most one box
    per phase per cluster, so cluster support is a clean 1..n_phases count.
    """
    coords = pool[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    scores = pool["score"].to_numpy(dtype=float)
    phase_id = pool["phase_id"].to_numpy()

    assigned = np.zeros(len(pool), dtype=bool)
    clusters = []

    for i in np.argsort(scores)[::-1]:
        if assigned[i]:
            continue
        assigned[i] = True
        members = [i]
        used = {phase_id[i]}

        ious = iou_one_to_many(coords[i], coords)
        cand = np.where(~assigned & (ious >= MATCH_IOU))[0]
        for c in cand[np.argsort(ious[cand])[::-1]]:
            if phase_id[c] in used:
                continue
            assigned[c] = True
            used.add(phase_id[c])
            members.append(c)

        m = np.array(members)
        clusters.append({
            "n_phases": len(used),
            "found_in_all": len(used) == n_phases,
            "mean_score": float(scores[m].mean()),
            "max_score": float(scores[m].max()),
            "mean_xmin": float(coords[m, 0].mean()),
            "mean_ymin": float(coords[m, 1].mean()),
            "mean_xmax": float(coords[m, 2].mean()),
            "mean_ymax": float(coords[m, 3].mean()),
            "mean_width_m": float(
                (coords[m, 2] - coords[m, 0]).mean() * GSD_CM / 100.0
            ),
            "phases": ";".join(sorted(str(p) for p in used)),
        })

    out = pd.DataFrame(clusters)
    out.insert(0, "cluster_id", np.arange(len(out)))
    return out


# ==========================================================================
# main
# ==========================================================================

def make_main():
    print("loading window", COL_OFF, ROW_OFF, SIZE, "downsample", DOWNSAMPLE)
    img = load_window()
    window_size = img.shape[0]
    print("window at experiment resolution:", img.shape, f"({GSD_CM:.2f} cm)")

    pad = STRIDE
    canvas = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    print("padded canvas:", canvas.shape)

    model = main.deepforest()
    model.load_model("weecology/deepforest-tree")

    summaries = []
    pool = []

    for dy in PHASE_OFFSETS:
        for dx in PHASE_OFFSETS:
            tag = f"dx{dx:03d}_dy{dy:03d}"
            boxes = run_phase(model, canvas, window_size, dx, dy, pad)
            boxes.to_csv(
                os.path.join(OUT_DIR, f"phase_boxes_{tag}.csv"), index=False
            )

            core = boxes[
                (boxes["cx"] >= CORE_INSET)
                & (boxes["cx"] < window_size - CORE_INSET)
                & (boxes["cy"] >= CORE_INSET)
                & (boxes["cy"] < window_size - CORE_INSET)
            ] if len(boxes) else boxes

            summaries.append({
                "phase_id": tag,
                "dx": dx,
                "dy": dy,
                "n_tiles": len(tile_origins(dx, canvas.shape[1]))
                           * len(tile_origins(dy, canvas.shape[0])),
                "n_detections": len(boxes),
                "n_core": len(core),
                "median_score": round(float(boxes["score"].median()), 4)
                                if len(boxes) else np.nan,
                "median_width_px": round(float(boxes["width_px"].median()), 2)
                                   if len(boxes) else np.nan,
                "median_width_m": round(float(boxes["width_m"].median()), 3)
                                  if len(boxes) else np.nan,
            })
            print(tag, "->", len(boxes), "detections,", len(core), "in core")

            if len(core):
                c = core.copy()
                c["phase_id"] = tag
                pool.append(c)

    summary = pd.DataFrame(summaries)
    summary.to_csv(os.path.join(OUT_DIR, "phase_summary.csv"), index=False)

    print("")
    print("=== per-phase detection counts ===")
    print(summary[["phase_id", "n_detections", "n_core",
                   "median_score", "median_width_m"]].to_string(index=False))
    print("")
    print("count spread across phases: min", summary["n_core"].min(),
          "max", summary["n_core"].max(),
          "cv", round(summary["n_core"].std() / summary["n_core"].mean(), 4))

    if not pool:
        print("no core detections; nothing to match")
        return

    pooled = pd.concat(pool, ignore_index=True)
    n_phases = PHASES_PER_AXIS ** 2
    clusters = cluster_across_phases(pooled, n_phases)
    clusters.to_csv(os.path.join(OUT_DIR, "phase_stability.csv"), index=False)

    hist = (
        clusters["n_phases"].value_counts().sort_index()
        .rename_axis("n_phases").reset_index(name="n_crowns")
    )
    hist["fraction"] = (hist["n_crowns"] / len(clusters)).round(4)
    hist.to_csv(os.path.join(OUT_DIR, "phase_stability_hist.csv"), index=False)

    print("")
    print("=== cross-phase stability (core region only) ===")
    print("distinct crowns:", len(clusters))
    print("found in all", n_phases, "phases:",
          int(clusters["found_in_all"].sum()),
          f"({clusters['found_in_all'].mean():.1%})")
    print("found in only one phase:", int((clusters["n_phases"] == 1).sum()))
    print("")
    print(hist.to_string(index=False))


if __name__ == "__main__":
    make_main()
