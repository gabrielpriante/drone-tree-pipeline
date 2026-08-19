"""
Seam pinning across every cluster, not just the single axis sensitive ones.

Reads the per position CSVs. No model, no raster, no inference.

Why this exists
---------------
check_seam_pinning.py found 63 of 71 single axis sensitive clusters with a box
edge sitting exactly on a grid boundary. Those 71 are support 4, 8 and 12 and
they are a subset. Nothing yet licenses saying the tiling fabricates detections
across the whole 2 to 15 band, and the contribution framing in the README must
not assert on a subset what was measured on a subset.

This runs the same test over all 710 clusters and reports the pinned share BY
SUPPORT LEVEL, so a result that concentrates at low support reads differently
from one that is uniform across the band.

The axis problem, and how it is handled
---------------------------------------
The 71 had one sensitive axis and, on that axis, exactly one offset. A general
cluster has neither: a support 10 cluster has ten observations at ten different
grid positions, so there is no single grid to measure the cluster against.

So the measurement moves down to the observation and the analysis stays at the
cluster:

    per observation   gap_x against that observation's own dx grid,
                      gap_y against its own dy grid, each the smaller of the
                      box's two edges on that axis
    observation gap   min(gap_x, gap_y)
    cluster gap       median over the cluster's observations

Taking the minimum over two axes inflates the pinned share. THE NULL IS BUILT
THE SAME WAY: each observation keeps its size and its own dx and dy, its
position is shuffled in two dimensions over the core, and the null takes the
same minimum over the same two axes. The inflation sits on both sides of the
comparison, so it cancels.

Bridge to the earlier result
----------------------------
The two axis statistic is not the one axis statistic. For the 71, this script
reports both, so the numbers reconcile instead of one silently replacing the
other.

Support 1 is reported as its OWN row and is never pooled into the 2 to 15 band.
If the singletons are largely pinned slivers then the instability finding and
the seam finding are one mechanism rather than two, and that is a claim about
support 1 specifically.

Decision rule, FIXED BEFORE THE NUMBERS ARE SEEN
------------------------------------------------
Same constants as check_seam_pinning.py. Per support level:

    PINNED at that level   observed share above the null 97.5th percentile
                           AND empirical p below P_CRIT
    NOT PINNED             observed share inside the null 95 percent band
    AMBIGUOUS              anything else

The headline the README needs: what share of the 399 clusters in the 2 to 15
band is pinned, and whether pinning is uniform across the band or concentrated
at low support.

Outputs (gitignored)
--------------------
seam_pinning_all_clusters.csv   one row per cluster, all 710
seam_pinning_all_by_support.csv one row per support level, with null and p

Not run yet at time of writing.
"""

import os

import numpy as np
import pandas as pd

import phase_matching as pm
from analyse_mechanism import build_crowns

MATCH_IOU = 0.5

# --- decision rule, set before the numbers are seen ---------------------
PIN_TOL = 1.0        # px. an edge within this of a boundary counts as pinned
P_CRIT = 0.001       # empirical p below this counts as above the null
N_NULL = 1000        # null iterations
NULL_SEED = 20260819
BLOCK = 100          # null iterations per vectorised block

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================================
# gaps
# =========================================================================

def edges_for(offset):
    starts = np.array(pm.scored_tile_origins(offset), dtype=float)
    return np.concatenate([starts, starts + pm.PATCH_SIZE])


def axis_gap(lo, hi, offset_codes, edge_table):
    """Smaller of the two edges' distance to the nearest boundary.

    lo and hi are (..., n_obs). offset_codes is (n_obs,) indexing edge_table.
    Vectorised over whatever leading axes lo and hi carry.
    """
    out = np.empty(lo.shape, dtype=float)
    for code, edges in enumerate(edge_table):
        m = offset_codes == code
        if not m.any():
            continue
        e = edges[None, :]
        a = np.abs(lo[..., m][..., None] - e).min(axis=-1)
        b = np.abs(hi[..., m][..., None] - e).min(axis=-1)
        out[..., m] = np.minimum(a, b)
    return out


def obs_gap(xlo, xhi, ylo, yhi, dx_code, dy_code, ex_table, ey_table):
    """min over the two axes. This is the inflating step, mirrored in the null."""
    return np.minimum(
        axis_gap(xlo, xhi, dx_code, ex_table),
        axis_gap(ylo, yhi, dy_code, ey_table),
    )


# =========================================================================
# cluster aggregation
# =========================================================================

def build_padding(cluster_ids):
    """Index map so a per observation array becomes (n_clusters, max_support).

    Padded slots hold -1 and are masked to NaN before the median.
    """
    order = np.argsort(cluster_ids, kind="mergesort")
    sorted_ids = cluster_ids[order]
    uniq, starts, counts = np.unique(
        sorted_ids, return_index=True, return_counts=True
    )
    width = int(counts.max())
    idx = np.full((len(uniq), width), -1, dtype=np.int64)
    for i, (s, c) in enumerate(zip(starts, counts)):
        idx[i, :c] = order[s:s + c]
    return uniq, idx, idx >= 0


def cluster_median(gaps, idx, mask):
    """gaps is (..., n_obs). Returns (..., n_clusters)."""
    g = gaps[..., np.clip(idx, 0, None)]
    g = np.where(mask, g, np.nan)
    return np.nanmedian(g, axis=-1)


# =========================================================================
# main
# =========================================================================

def run():
    print("MATCH_IOU:", MATCH_IOU)
    print("decision rule fixed in advance:")
    print("  PIN_TOL", PIN_TOL, "px   P_CRIT", P_CRIT,
          "  N_NULL", N_NULL, "  seed", NULL_SEED)
    print("  observation gap = min over BOTH axes; null built the same way")
    print("")

    pool = pm.load_pool(OUT_DIR)
    clusters, cluster_of = pm.cluster_across_phases(pool, MATCH_IOU)
    pooled = pm.attach_clusters(pool, cluster_of, clusters).reset_index(drop=True)
    print("")
    print("clusters:", len(clusters), "  observations:", len(pooled))

    xlo = pooled["xmin"].to_numpy(dtype=float)
    xhi = pooled["xmax"].to_numpy(dtype=float)
    ylo = pooled["ymin"].to_numpy(dtype=float)
    yhi = pooled["ymax"].to_numpy(dtype=float)
    w = xhi - xlo
    h = yhi - ylo

    offsets = pm.PHASE_OFFSETS
    ex_table = [edges_for(o) for o in offsets]
    ey_table = [edges_for(o) for o in offsets]
    code = {o: i for i, o in enumerate(offsets)}
    dx_code = np.array([code[int(v)] for v in pooled["dx"]])
    dy_code = np.array([code[int(v)] for v in pooled["dy"]])

    cid = pooled["cluster_id"].to_numpy()
    uniq, idx, mask = build_padding(cid)
    support = clusters.set_index("cluster_id").loc[uniq, "n_phases"].to_numpy()

    # --- observed ---------------------------------------------------------
    g_obs = obs_gap(xlo[None, :], xhi[None, :], ylo[None, :], yhi[None, :],
                    dx_code, dy_code, ex_table, ey_table)
    c_obs = cluster_median(g_obs, idx, mask)[0]
    pinned_obs = c_obs <= PIN_TOL

    # --- null -------------------------------------------------------------
    lo_b = float(pm.CORE_INSET)
    hi_b = float(pm.WIN_SIZE - pm.CORE_INSET)
    span_x = np.maximum(hi_b - w - lo_b, 0.0)
    span_y = np.maximum(hi_b - h - lo_b, 0.0)

    rng = np.random.default_rng(NULL_SEED)
    levels = np.arange(1, pm.N_PHASES + 1)
    null_share = np.zeros((N_NULL, len(levels)))
    null_med = np.zeros(N_NULL)

    done = 0
    while done < N_NULL:
        b = min(BLOCK, N_NULL - done)
        nx = lo_b + rng.random((b, len(w))) * span_x
        ny = lo_b + rng.random((b, len(h))) * span_y
        g = obs_gap(nx, nx + w, ny, ny + h,
                    dx_code, dy_code, ex_table, ey_table)
        cg = cluster_median(g, idx, mask)          # (b, n_clusters)
        p = cg <= PIN_TOL
        null_med[done:done + b] = np.median(cg, axis=1)
        for j, L in enumerate(levels):
            sel = support == L
            null_share[done:done + b, j] = (
                p[:, sel].mean(axis=1) if sel.any() else np.nan
            )
        done += b
        print(f"  null {done}/{N_NULL}", end="\r")
    print(" " * 30, end="\r")

    # --- per level table --------------------------------------------------
    rows = []
    for j, L in enumerate(levels):
        sel = support == L
        n = int(sel.sum())
        if n == 0:
            continue
        obs_share = float(pinned_obs[sel].mean())
        ns = null_share[:, j]
        p_emp = float(np.mean(ns >= obs_share))
        lo95, hi95 = np.quantile(ns, [0.025, 0.975])
        if obs_share > hi95 and p_emp < P_CRIT:
            verdict = "PINNED"
        elif lo95 <= obs_share <= hi95:
            verdict = "not pinned"
        else:
            verdict = "ambiguous"
        rows.append({
            "support": int(L),
            "n_clusters": n,
            "n_pinned": int(pinned_obs[sel].sum()),
            "obs_share_pinned": round(obs_share, 4),
            "median_gap_px": round(float(np.median(c_obs[sel])), 4),
            "null_share_med": round(float(np.median(ns)), 4),
            "null_share_hi95": round(float(hi95), 4),
            "p_empirical": round(p_emp, 5),
            "verdict": verdict,
        })
    by_level = pd.DataFrame(rows)

    print("=" * 78)
    print("PINNED SHARE BY SUPPORT LEVEL")
    print("=" * 78)
    print(by_level.to_string(index=False))
    print("")

    # --- the three populations, never pooled ------------------------------
    def band(name, sel):
        n = int(sel.sum())
        if n == 0:
            print(f"{name:22s} no clusters")
            return None
        return {
            "population": name,
            "n_clusters": n,
            "n_pinned": int(pinned_obs[sel].sum()),
            "share_pinned": round(float(pinned_obs[sel].mean()), 4),
            "median_gap_px": round(float(np.median(c_obs[sel])), 4),
        }

    bands = [b for b in [
        band("support 1", support == 1),
        band("support 2 to 15", (support >= 2) & (support <= 15)),
        band("support 16", support == pm.N_PHASES),
        band("all clusters", support >= 1),
    ] if b]
    print("=" * 78)
    print("BY POPULATION, support 1 kept separate from the 2 to 15 band")
    print("=" * 78)
    print(pd.DataFrame(bands).to_string(index=False))
    print("")

    # --- bridge to the 71 -------------------------------------------------
    crowns = build_crowns(pooled, clusters)
    sens = crowns[crowns["sensitive_axis"].notna()
                  & crowns["support"].isin([4, 8, 12])]
    sel71 = np.isin(uniq, sens["cluster_id"].to_numpy())
    print("=" * 78)
    print("BRIDGE TO check_seam_pinning.py")
    print("=" * 78)
    print("the 71 single axis sensitive clusters, measured HERE with the two")
    print("axis minimum rather than the single sensitive axis:")
    print("  n              :", int(sel71.sum()))
    print("  share pinned   :", round(float(pinned_obs[sel71].mean()), 4))
    print("  median gap px  :", round(float(np.median(c_obs[sel71])), 4))
    print("recorded there with the single axis rule: 0.8873, 63 of 71,")
    print("median gap 0.0 px. The two axis rule can only raise the share,")
    print("never lower it, so a lower number here means something is wrong.")
    print("")

    # --- outputs ----------------------------------------------------------
    pd.DataFrame({
        "cluster_id": uniq,
        "support": support,
        "cluster_gap_px": np.round(c_obs, 4),
        "pinned": pinned_obs,
    }).to_csv(os.path.join(OUT_DIR, "seam_pinning_all_clusters.csv"),
              index=False)
    by_level.to_csv(
        os.path.join(OUT_DIR, "seam_pinning_all_by_support.csv"), index=False)
    print("wrote seam_pinning_all_clusters.csv and "
          "seam_pinning_all_by_support.csv")


if __name__ == "__main__":
    run()
