"""
Shared cross phase matching, read only.

Terminology
-----------
Following Miller, Dayoub, Milford and Sunderhauf, "Evaluating Merging
Strategies for Sampling-based Uncertainty Techniques in Object Detection",
ICRA 2019, arXiv 1809.06006.

    sample        one run of the detector over the scene. Here, one grid
                  phase. 16 samples.
    observation   one detection box produced by one sample.
    cluster       a set of observations from different samples judged to be
                  the same underlying object.
    support       the number of samples contributing an observation to a
                  cluster. 1 to 16 here.

The clustering below is BSAS in that taxonomy, Basic Sequential Algorithmic
Scheme, with intra sample exclusivity: at most one observation per sample per
cluster. That is their best performing configuration. Their semantic affinity
component does not apply, this problem has one class.

Their spatial affinity threshold is IoU 0.95, appropriate for MC Dropout
samples over an identical image, where boxes are near coincident. Our samples
come from different tilings, so boxes legitimately shift, which is why
MATCH_IOU here is far lower. The 0.3 to 0.5 sweep in
check_match_sensitivity.py is the response to their finding that results are
sensitive to that threshold.

Variable names are unchanged. This is documentation only.

Loads the per phase box CSVs written by phase_sweep.py and redoes the
clustering. Imports no model and touches no raster, so it is cheap to rerun at
different thresholds.

IMPORTANT: the clustering logic below is a verbatim copy of
cluster_across_phases() in phase_sweep.py, extended only to return which
pooled boxes ended up in which cluster. It is duplicated rather than imported
because importing phase_sweep.py would pull in deepforest and torch.

That duplication can drift. To catch it, check_match_sensitivity.py asserts
that this module reproduces the recorded sweep result at MATCH_IOU 0.5. If
phase_sweep.py's matching is ever edited, edit this too and rerun that check.
"""

import os
import glob
import numpy as np
import pandas as pd

# --- must match phase_sweep.py -------------------------------------------
WIN_SIZE = 1000          # EXPERIMENT px
CORE_INSET = 25          # EXPERIMENT px
GSD_CM = 7.78
PHASES_PER_AXIS = 4
STRIDE = 300
PHASE_STEP = STRIDE // PHASES_PER_AXIS
PHASE_OFFSETS = [i * PHASE_STEP for i in range(PHASES_PER_AXIS)]
N_PHASES = PHASES_PER_AXIS ** 2

# tiling geometry, needed to locate tile seams. EXPERIMENT px.
PATCH_SIZE = 400
MARGIN = STRIDE          # real imagery margin, one stride each side
CANVAS_SIZE = WIN_SIZE + 2 * MARGIN


def scored_tile_origins(phase):
    """Tile origins on one axis, in SCORED WINDOW coordinates.

    Same grid phase_sweep.py cuts, shifted from canvas coordinates into
    scored window coordinates. Origins can be negative, because tiles reach
    into the real pixel margin.
    """
    return [o - MARGIN
            for o in range(phase, CANVAS_SIZE - PATCH_SIZE + 1, STRIDE)]


def tile_edges(phase):
    """Every tile boundary line on one axis, scored window coordinates.

    A tile spans [o, o + PATCH_SIZE), so it contributes two boundary lines.
    Duplicates removed and sorted.
    """
    edges = set()
    for o in scored_tile_origins(phase):
        edges.add(o)
        edges.add(o + PATCH_SIZE)
    return sorted(edges)

# --- recorded sweep result, for drift detection --------------------------
# From the run at MATCH_IOU 0.5.
RECORDED = {
    "match_iou": 0.5,
    "distinct_crowns": 710,
    "found_in_all": 115,
    "found_in_one": 196,
}


def phase_tag(dx, dy):
    return f"dx{dx:03d}_dy{dy:03d}"


def load_pool(out_dir, verbose=True):
    """Rebuild the pooled core box set exactly as phase_sweep.py built it.

    Order matters. phase_sweep.py appended phases in the order dy outer, dx
    inner, and clustering walks the pool by descending score, so a different
    pool order can produce a different clustering on tied scores. This
    reproduces that order.
    """
    frames = []
    missing = []
    for dy in PHASE_OFFSETS:
        for dx in PHASE_OFFSETS:
            tag = phase_tag(dx, dy)
            path = os.path.join(out_dir, f"phase_boxes_{tag}.csv")
            if not os.path.exists(path):
                missing.append(tag)
                continue
            boxes = pd.read_csv(path)
            if len(boxes) == 0:
                continue
            # same core filter as phase_sweep.py
            core = boxes[
                (boxes["cx"] >= CORE_INSET)
                & (boxes["cx"] < WIN_SIZE - CORE_INSET)
                & (boxes["cy"] >= CORE_INSET)
                & (boxes["cy"] < WIN_SIZE - CORE_INSET)
            ].copy()
            core["phase_id"] = tag
            frames.append(core)

    if missing:
        raise SystemExit(
            "missing phase CSVs: " + ", ".join(missing)
            + "\nrun phase_sweep.py first, or point OUT_DIR at the right folder"
        )
    if not frames:
        raise SystemExit("no boxes found in any phase CSV")

    pooled = pd.concat(frames, ignore_index=True)
    if verbose:
        found = glob.glob(os.path.join(out_dir, "phase_boxes_*.csv"))
        print("phase CSVs found :", len(found))
        print("pooled core boxes:", len(pooled))
    return pooled


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


def cluster_across_phases(pool, match_iou, n_phases=N_PHASES):
    """Greedy single pass clustering of boxes into crowns.

    Verbatim from phase_sweep.py apart from the match_iou argument and the
    returned box to cluster mapping.

    Returns (clusters DataFrame, cluster_of array aligned to pool rows).
    """
    coords = pool[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    scores = pool["score"].to_numpy(dtype=float)
    phase_id = pool["phase_id"].to_numpy()

    assigned = np.zeros(len(pool), dtype=bool)
    cluster_of = np.full(len(pool), -1, dtype=int)
    clusters = []

    for i in np.argsort(scores)[::-1]:
        if assigned[i]:
            continue
        cid = len(clusters)
        assigned[i] = True
        cluster_of[i] = cid
        members = [i]
        used = {phase_id[i]}

        ious = iou_one_to_many(coords[i], coords)
        cand = np.where(~assigned & (ious >= match_iou))[0]
        for c in cand[np.argsort(ious[cand])[::-1]]:
            if phase_id[c] in used:
                continue
            assigned[c] = True
            cluster_of[c] = cid
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
    return out, cluster_of


def attach_clusters(pool, cluster_of, clusters):
    """Return a copy of pool with cluster_id and support columns added."""
    support = clusters["n_phases"].to_numpy()
    out = pool.copy()
    out["cluster_id"] = cluster_of
    out["support"] = support[cluster_of]
    return out


def phase_index(dx, dy):
    """(dx, dy) offsets to 0 based grid indices (ix, iy)."""
    return PHASE_OFFSETS.index(int(dx)), PHASE_OFFSETS.index(int(dy))


def support_histogram(clusters, n_phases=N_PHASES):
    """Full 1 to n_phases histogram, zeros included."""
    counts = clusters["n_phases"].value_counts()
    rows = []
    total = len(clusters)
    for k in range(1, n_phases + 1):
        n = int(counts.get(k, 0))
        rows.append({
            "n_phases": k,
            "n_crowns": n,
            "fraction": round(n / total, 4) if total else 0.0,
        })
    return pd.DataFrame(rows)
