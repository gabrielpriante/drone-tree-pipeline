"""
Seam pinning check: are the single axis sensitive detections boxes whose EDGE
is a tile boundary, or boxes that merely sit near one?

Reads the per position CSVs and mechanism_seam_distance.csv. No model, no
raster, no inference.

Why this exists
---------------
analyse_mechanism.py measured distance to seam from the box CENTROID, not the
box edge, so it is not the trivial artefact where an edge on a boundary scores
zero by construction. But the numbers it produced are:

    median seam_found                              4.7736 px
    median half extent of the box, sensitive axis  4.621  px

Those agree to within 0.15 px. That is exactly what you get if the box edge
lies on the boundary and the centroid therefore sits half a box away from it.
For scale, a centroid dropped at random has a median distance to the nearest
boundary near 37.5 px, given boundaries at two residues per 300 px stride.

If the boxes are pinned to boundaries, the 4.77 px result is a property of the
measurement geometry rather than of the trees, and the seam entry in the
project notes is describing the wrong mechanism.

What does NOT depend on this check
----------------------------------
Two things survive whatever comes back.

  1. The sign test p value is a true statement about the data as measured.
     What this check can invalidate is the mechanism label attached to it, not
     the number.
  2. The refutation of seam severing does not rest on the centroid measure at
     all. It rests on containment being BETTER at the samples where the tree
     was missed, median 130.3292 against 90.6744. Crowns that were missed were
     more comfortably inside a tile, not less.

Unit of analysis
----------------
THE CLUSTER, n = 71. A crown sensitive to dx was found at four samples that
all share the same dx, so its four boxes have near identical left and right
edges. Those are not four independent observations. Per observation numbers
below are descriptive only. Every test statistic is computed over 71 clusters.

The statistic
-------------
For each cluster, take its observations at the samples where it was found. On
the SENSITIVE AXIS ONLY, compute the distance from each box edge to the
nearest tile boundary, and keep the smaller of the two edges. Median over the
cluster's observations gives one edge gap per cluster.

The null
--------
Not a theoretical distribution. Each box keeps its size and is placed
uniformly at random along the sensitive axis inside the core, then the same
edge gap is computed at the same offset. N_NULL iterations, each producing a
full set of 71 gaps and therefore one draw of each summary statistic.

A theoretical null would not control for the fact that small boxes sit closer
to things by chance, and these boxes are small.

Decision rule, FIXED BEFORE THE NUMBERS ARE SEEN
------------------------------------------------
    SEAM MANUFACTURED
        observed median edge gap below PIN_TOL, and observed share pinned
        above the null with empirical p below P_CRIT.
        Consequence: the 4.77 px centroid median is structural. The seam entry
        in the notes needs rewriting.

    NOT SEAM MANUFACTURED
        observed statistics sit inside the central 95 percent of the null.
        Consequence: the centroid measure stands on its own, and the finding
        is a real statement about where these trees sit relative to the grid.

    PARTIAL
        anything else. Report the SHARE pinned, not a verdict.

If the answer is PARTIAL, the pinned and unpinned subsets are separated and
their box geometry is compared: median width, height and aspect for each. If
the pinned subset is the small elongated fragments and the unpinned subset is
not, then two different phenomena are sharing one bucket and the paper has to
say so rather than quoting a single share.

Secondary diagnostics
---------------------
    which edge is pinned, the low coordinate edge or the high coordinate edge
    whether the pinned boundary is the START of a tile or the END of one

At stride 300 and patch 400 those two sets never collide: starts sit at
phase mod 300, ends at phase + 100 mod 300.

Outputs (gitignored)
--------------------
seam_pinning_clusters.csv   one row per cluster
seam_pinning_summary.csv    observed statistics, null quantiles, verdict

Not run yet. Cheap, no model load. The null loop is the only real cost.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd

import phase_matching as pm
from analyse_mechanism import (
    build_crowns,
    classify_pattern,
    sensitive_axis,
    STRUCTURED_SUPPORTS,
)

MATCH_IOU = 0.5

# --- decision rule, set before the numbers are seen ---------------------
PIN_TOL = 1.0        # px. an edge within this of a boundary counts as pinned
P_CRIT = 0.001       # empirical p below this counts as far above the null
N_NULL = 1000        # null iterations
NULL_SEED = 20260818  # fixed, so the verdict is reproducible

# --- drift guards --------------------------------------------------------
EXPECTED_N_CLUSTERS = 71
RECORDED_MEDIAN_SEAM_FOUND = 4.7736
RECORDED_MEDIAN_SEAM_MISSED = 48.5241

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================================
# geometry
# =========================================================================

def edge_arrays(offset):
    """Tile starts and ends on one axis, scored window coordinates.

    Returned separately so the secondary diagnostic can tell which kind of
    boundary an edge is pinned to. At stride 300 and patch 400 the two sets
    are disjoint.
    """
    starts = np.array(pm.scored_tile_origins(offset), dtype=float)
    ends = starts + pm.PATCH_SIZE
    assert not set(starts).intersection(set(ends)), \
        "tile starts and ends collide, the diagnostic below is meaningless"
    return starts, ends


def nearest_boundary(coord, starts, ends):
    """(gap, kind) for one coordinate. kind is 'start' or 'end'."""
    ds = np.min(np.abs(starts - coord))
    de = np.min(np.abs(ends - coord))
    return (float(ds), "start") if ds <= de else (float(de), "end")


def edge_gap(lo, hi, offset, starts=None, ends=None):
    """Smaller of the two box edges' distances to the nearest boundary.

    Returns (gap, which_edge, boundary_kind).
    """
    if starts is None:
        starts, ends = edge_arrays(offset)
    g_lo, k_lo = nearest_boundary(lo, starts, ends)
    g_hi, k_hi = nearest_boundary(hi, starts, ends)
    if g_lo <= g_hi:
        return g_lo, "low", k_lo
    return g_hi, "high", k_hi


# =========================================================================
# observed
# =========================================================================

def observed_table(pooled, crowns):
    """One row per single axis sensitive cluster."""
    sub = crowns[crowns["sensitive_axis"].notna()
                 & crowns["support"].isin(STRUCTURED_SUPPORTS)]
    rows = []
    for _, c in sub.iterrows():
        cid = c["cluster_id"]
        axis = c["sensitive_axis"]
        members = pooled[pooled["cluster_id"] == cid]

        # on the sensitive axis every member shares one offset, by definition
        # of the pattern. Assert it rather than assume it.
        offsets = sorted(set(members["dx" if axis == "dx" else "dy"]))
        assert len(offsets) == 1, (
            f"cluster {cid} is {c['pattern']} but has {len(offsets)} "
            f"offsets on its sensitive axis {axis}"
        )
        offset = int(offsets[0])
        starts, ends = edge_arrays(offset)

        gaps, which, kinds, extents = [], [], [], []
        for _, m in members.iterrows():
            if axis == "dx":
                lo, hi = float(m["xmin"]), float(m["xmax"])
            else:
                lo, hi = float(m["ymin"]), float(m["ymax"])
            g, w, k = edge_gap(lo, hi, offset, starts, ends)
            gaps.append(g)
            which.append(w)
            kinds.append(k)
            extents.append(hi - lo)

        w_px = (members["xmax"] - members["xmin"]).median()
        h_px = (members["ymax"] - members["ymin"]).median()
        rows.append({
            "cluster_id": cid,
            "support": int(c["support"]),
            "pattern": c["pattern"],
            "sensitive_axis": axis,
            "offset": offset,
            "n_obs": len(members),
            "edge_gap_px": float(np.median(gaps)),
            "edge_gap_min_px": float(np.min(gaps)),
            "which_edge": Counter(which).most_common(1)[0][0],
            "boundary_kind": Counter(kinds).most_common(1)[0][0],
            "extent_sensitive_px": float(np.median(extents)),
            "med_w_px": float(w_px),
            "med_h_px": float(h_px),
            "med_w_m": float(w_px) * pm.GSD_CM / 100.0,
            "med_h_m": float(h_px) * pm.GSD_CM / 100.0,
            "med_aspect": float(max(w_px, h_px) / max(min(w_px, h_px), 1e-9)),
        })
    return pd.DataFrame(rows)


# =========================================================================
# null
# =========================================================================

def null_draw(table, rng):
    """One randomised placement per cluster. Returns the 71 edge gaps.

    Size and offset held, position shuffled. Placement is uniform over the
    core region, the same region the clustering was restricted to.
    """
    lo_bound = pm.CORE_INSET
    hi_bound = pm.WIN_SIZE - pm.CORE_INSET
    gaps = np.empty(len(table))
    for i, (_, r) in enumerate(table.iterrows()):
        extent = r["extent_sensitive_px"]
        span = (hi_bound - extent) - lo_bound
        lo = lo_bound + (rng.random() * span if span > 0 else 0.0)
        g, _, _ = edge_gap(lo, lo + extent, int(r["offset"]))
        gaps[i] = g
    return gaps


def summarise(gaps):
    return {
        "median_edge_gap": float(np.median(gaps)),
        "share_pinned": float(np.mean(np.asarray(gaps) <= PIN_TOL)),
    }


# =========================================================================
# reporting
# =========================================================================

def compare_geometry(table):
    """Pinned against unpinned. Only meaningful when the answer is PARTIAL."""
    t = table.copy()
    t["pinned"] = t["edge_gap_px"] <= PIN_TOL
    g = (t.groupby("pinned")
          .agg(n_clusters=("cluster_id", "size"),
               med_w_m=("med_w_m", "median"),
               med_h_m=("med_h_m", "median"),
               med_aspect=("med_aspect", "median"),
               med_extent_sensitive_px=("extent_sensitive_px", "median"))
          .rename(index={True: "pinned", False: "not pinned"})
          .round(4))
    return g


def run():
    print("MATCH_IOU:", MATCH_IOU)
    print("decision rule fixed in advance:")
    print("  PIN_TOL", PIN_TOL, "px   P_CRIT", P_CRIT,
          "  N_NULL", N_NULL, "  seed", NULL_SEED)
    print("")

    pool = pm.load_pool(OUT_DIR)
    clusters, cluster_of = pm.cluster_across_phases(pool, MATCH_IOU)
    pooled = pm.attach_clusters(pool, cluster_of, clusters)
    crowns = build_crowns(pooled, clusters)
    print("")

    table = observed_table(pooled, crowns)
    print("single axis sensitive clusters:", len(table))
    assert len(table) == EXPECTED_N_CLUSTERS, (
        f"expected {EXPECTED_N_CLUSTERS} clusters, got {len(table)}. "
        "The clustering has changed since mechanism_seam_distance.csv was "
        "written. Resolve before reading anything below."
    )

    # --- drift guard against the recorded centroid result ----------------
    prev_path = os.path.join(OUT_DIR, "mechanism_seam_distance.csv")
    if os.path.exists(prev_path):
        prev = pd.read_csv(prev_path)
        m_found = float(prev["seam_found"].median())
        m_missed = float(prev["seam_missed"].median())
        ok = (abs(m_found - RECORDED_MEDIAN_SEAM_FOUND) < 0.01
              and abs(m_missed - RECORDED_MEDIAN_SEAM_MISSED) < 0.01)
        print("drift guard against mechanism_seam_distance.csv:",
              "PASS" if ok else "FAIL")
        print(f"  median seam_found  {m_found:.4f} "
              f"vs recorded {RECORDED_MEDIAN_SEAM_FOUND}")
        print(f"  median seam_missed {m_missed:.4f} "
              f"vs recorded {RECORDED_MEDIAN_SEAM_MISSED}")
        if not ok:
            print("  the file on disk does not match the recorded result.")
    else:
        print("mechanism_seam_distance.csv absent, drift guard skipped")
    print("")

    # --- observed --------------------------------------------------------
    obs = summarise(table["edge_gap_px"].to_numpy())
    print("=" * 66)
    print("OBSERVED")
    print("=" * 66)
    print(f"median edge gap : {obs['median_edge_gap']:.4f} px")
    print(f"share pinned    : {obs['share_pinned']:.4f} "
          f"({int(round(obs['share_pinned'] * len(table)))} of {len(table)})")
    print("")
    print("median box half extent on the sensitive axis :",
          round(float(table["extent_sensitive_px"].median()) / 2, 4), "px")
    print("(this is the value the centroid measure would return if every box")
    print(" edge sat exactly on a boundary)")
    print("")

    # --- null ------------------------------------------------------------
    rng = np.random.default_rng(NULL_SEED)
    null_med, null_share = [], []
    for _ in range(N_NULL):
        s = summarise(null_draw(table, rng))
        null_med.append(s["median_edge_gap"])
        null_share.append(s["share_pinned"])
    null_med = np.array(null_med)
    null_share = np.array(null_share)

    p_med = float(np.mean(null_med <= obs["median_edge_gap"]))
    p_share = float(np.mean(null_share >= obs["share_pinned"]))

    print("=" * 66)
    print("NULL, size and offset held, position shuffled")
    print("=" * 66)
    print(f"median edge gap : median {np.median(null_med):.4f}  "
          f"2.5th {np.quantile(null_med, 0.025):.4f}  "
          f"97.5th {np.quantile(null_med, 0.975):.4f}")
    print(f"share pinned    : median {np.median(null_share):.4f}  "
          f"2.5th {np.quantile(null_share, 0.025):.4f}  "
          f"97.5th {np.quantile(null_share, 0.975):.4f}")
    print("")
    print(f"empirical p, median edge gap at or below observed : {p_med:.5f}")
    print(f"empirical p, share pinned at or above observed    : {p_share:.5f}")
    print(f"(p of 0 means below 1 in {N_NULL})")
    print("")

    # --- verdict ---------------------------------------------------------
    inside_med = (np.quantile(null_med, 0.025) <= obs["median_edge_gap"]
                  <= np.quantile(null_med, 0.975))
    inside_share = (np.quantile(null_share, 0.025) <= obs["share_pinned"]
                    <= np.quantile(null_share, 0.975))

    if obs["median_edge_gap"] < PIN_TOL and p_share < P_CRIT:
        verdict = "SEAM MANUFACTURED"
        note = ("The 4.77 px centroid median is structural. Rewrite the seam "
                "entry in the project notes.")
    elif inside_med and inside_share:
        verdict = "NOT SEAM MANUFACTURED"
        note = ("The centroid measure stands on its own. The seam entry "
                "describes a real positional finding.")
    else:
        verdict = "PARTIAL"
        note = ("Report the share pinned, not a verdict. See the geometry "
                "split below before writing anything.")

    print("=" * 66)
    print("VERDICT:", verdict)
    print("=" * 66)
    print(note)
    print("")

    # --- secondary diagnostics ------------------------------------------
    print("--- which edge is pinned ---")
    print(table["which_edge"].value_counts().to_string())
    print("")
    print("--- boundary kind, tile start or tile end ---")
    print(table["boundary_kind"].value_counts().to_string())
    print("")
    print("--- split by sensitive axis ---")
    print(table.groupby("sensitive_axis")["edge_gap_px"]
          .describe()[["count", "25%", "50%", "75%"]].round(4).to_string())
    print("")

    # --- geometry split, the two phenomena question ---------------------
    geom = compare_geometry(table)
    print("--- geometry, pinned against not pinned ---")
    print(geom.to_string())
    print("")
    if verdict == "PARTIAL":
        print("The answer is PARTIAL, so read the table above carefully.")
        print("If the pinned rows are the small elongated boxes and the")
        print("unpinned rows are not, these are two phenomena in one bucket")
        print("and the paper has to separate them rather than quote a share.")
    else:
        print("Reported for completeness. The verdict was not PARTIAL, so")
        print("this split is descriptive rather than load bearing.")
    print("")

    # --- outputs ---------------------------------------------------------
    table.to_csv(os.path.join(OUT_DIR, "seam_pinning_clusters.csv"),
                 index=False)
    pd.DataFrame([{
        "match_iou": MATCH_IOU,
        "n_clusters": len(table),
        "pin_tol_px": PIN_TOL,
        "n_null": N_NULL,
        "null_seed": NULL_SEED,
        "obs_median_edge_gap": round(obs["median_edge_gap"], 4),
        "obs_share_pinned": round(obs["share_pinned"], 4),
        "null_median_edge_gap_med": round(float(np.median(null_med)), 4),
        "null_median_edge_gap_lo": round(float(np.quantile(null_med, 0.025)), 4),
        "null_median_edge_gap_hi": round(float(np.quantile(null_med, 0.975)), 4),
        "null_share_pinned_med": round(float(np.median(null_share)), 4),
        "null_share_pinned_lo": round(float(np.quantile(null_share, 0.025)), 4),
        "null_share_pinned_hi": round(float(np.quantile(null_share, 0.975)), 4),
        "p_median_edge_gap": round(p_med, 5),
        "p_share_pinned": round(p_share, 5),
        "verdict": verdict,
    }]).to_csv(os.path.join(OUT_DIR, "seam_pinning_summary.csv"), index=False)
    print("wrote seam_pinning_clusters.csv and seam_pinning_summary.csv")


if __name__ == "__main__":
    run()
