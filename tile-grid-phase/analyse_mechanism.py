"""
Mechanism tests: is single axis sensitivity caused by tile seams severing
crowns?

Reads the per phase box CSVs. No model, no raster, no inference.

Background
----------
The support histogram is U shaped with a local peak at support 4. Crowns at
support 4 are overwhelmingly structured: found at all four dx offsets for one
dy, or all four dy for one dx. The two classes are roughly balanced, 31 and 40
in the first run, and the axis marginals show no global bias, sign test p
0.313.

State the finding as: each crown is sensitive to exactly one axis, and which
axis varies by crown. NOT as anisotropy. There is no evidence that one axis
matters more overall.

Support 4 crowns also have the highest median aspect of any level, 2.23, and
the smallest median area, 1.67 m2. Small elongated slivers. The hypothesis
this script set out to test was that they are crowns severed by a tile seam.
THAT HYPOTHESIS IS REFUTED. At the surveys where a crown was missed it was
better contained inside a tile, median containment margin 130.3292 px, than at
the samples found, 90.6744 px. Severing predicts the opposite. The slivers are
manufactured by the tiling rather than cut by it: 63 of 71 have a box edge
sitting exactly on a grid boundary. See check_seam_pinning.py for that result.

Terminology follows Miller et al., ICRA 2019, arXiv 1809.06006: sample,
observation, cluster, support. See phase_matching.py.

The three tests
---------------
TASK 1  ELONGATION AXIS VERSUS SENSITIVITY AXIS
        A crown cut by a vertical seam, a seam at constant x, leaves a tall
        thin fragment. Its detectability depends on where the vertical seams
        fall, which dx controls, and not on dy. So it should be detected at
        every dy but only one dx, and its box should be taller than wide.

            sensitive to dx  (all_dy_one_dx)  ->  taller than wide
            sensitive to dy  (all_dx_one_dy)  ->  wider than tall

        If elongation axis predicts sensitivity axis, the seam severing
        mechanism is demonstrated rather than inferred.

        Careful with the class names. all_dx_one_dy means found at ALL four
        dx offsets at a SINGLE dy, so varying dx does not kill it and varying
        dy does. That crown is sensitive to dy.

TASK 2  SEAM PROXIMITY
        A positional test. For each single axis sensitive crown, measure how
        close it sits to a tile seam on its sensitive axis, at the phase where
        it was detected and at the phases where it was not.

        Prediction: at the phases where it was missed, it sits close to a
        seam. At the phase where it was found, it does not.

TASK 3  SPATIAL VARIANCE PER CLUSTER
        Miller et al. extract spatial uncertainty as the total variance of
        bounding box coordinates within a cluster, x and y separately, and
        show it separates spatially accurate from inaccurate observations.

        Because it is separable by axis it gives a second reading on single
        axis sensitivity, using a different quantity. Note it is NOT
        statistically independent: it is computed over the same clusters.

        Prediction, and it runs opposite to first intuition. A crown detected
        at four samples that share one dx has an identical x tiling across all
        four observations, so its x coordinates should be stable. Its dy
        varies, so its y coordinates should move.

            sensitive to dx  (all_dy_one_dx)  ->  var_y > var_x
            sensitive to dy  (all_dx_one_dy)  ->  var_x > var_y

Synthetic control, run before this script was committed
-------------------------------------------------------
Thirty round stable crowns plus twenty four planted slivers, each sliver
placed so it is clear of a seam at exactly one offset of one axis and near a
seam at the other three. Not real data, a control.

    Task 1  recovered the mechanism exactly: 24 of 24 concordant, odds ratio
            infinite, Fisher p 7.4e-07.
    Task 2  recovered it: seam distance 86.4 px at the samples found against
            28.8 px at the samples missed, 24 of 24, sign test p 1.2e-07.
    Task 3  returned null, 45.83 percent concordant, p 1.

The Task 3 null is expected and does NOT validate or invalidate that test. The
planted observations were jittered isotropically at random rather than shifted
by the tiling, so there was no axis structure in the coordinates for the
variance test to find. Task 3 is therefore UNTESTED by the control. Treat its
result on real data as exploratory.

The single observation NaN path was exercised separately and confirmed: a
support 1 cluster yields NaN, a support 2 cluster yields a finite variance.

Outputs (gitignored)
--------------------
mechanism_elongation.csv     per crown, sensitivity axis and elongation axis
mechanism_seam_distance.csv  per crown, seam distances found vs missed
mechanism_spatial_var.csv    per cluster, var_x, var_y, total
mechanism_summary.txt        the printed report

Not run yet.
"""

import os
from math import comb

import numpy as np
import pandas as pd

import phase_matching as pm

MATCH_IOU = 0.5
STRUCTURED_SUPPORTS = [4, 8, 12]   # 4 is the primary, 8 and 12 are secondary

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================================
# exact tests, so scipy is not required
# =========================================================================

def fisher_exact_2x2(a, b, c, d):
    """Two sided Fisher exact test on [[a, b], [c, d]].

    Returns (odds_ratio, p). Sums the probability of every table with the
    same margins whose probability is at or below the observed one.
    """
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def prob(x):
        y = row1 - x
        z = col1 - x
        w = n - x - y - z
        if min(x, y, z, w) < 0:
            return 0.0
        return (comb(row1, x) * comb(n - row1, z)) / comb(n, col1)

    p_obs = prob(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    p = sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * 1.0000001)

    if b * c == 0:
        odds = float("inf") if a * d > 0 else float("nan")
    else:
        odds = (a * d) / (b * c)
    return odds, min(1.0, p)


def sign_test_two_sided(n_pos, n_neg):
    n = n_pos + n_neg
    if n == 0:
        return float("nan")
    k = min(n_pos, n_neg)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


# =========================================================================
# classification, shared with analyse_support.py
# =========================================================================

def classify_pattern(cells):
    cells = list(cells)
    X = {c[0] for c in cells}
    Y = {c[1] for c in cells}
    if len(X) * len(Y) == len(cells):
        if len(Y) == 1 and len(X) == pm.PHASES_PER_AXIS:
            return "all_dx_one_dy"
        if len(X) == 1 and len(Y) == pm.PHASES_PER_AXIS:
            return "all_dy_one_dx"
        return f"block_{len(Y)}dy_x_{len(X)}dx"
    return "scattered"


def sensitive_axis(pattern):
    """Which axis the crown is sensitive to, or None.

    all_dx_one_dy: survives every dx, dies when dy moves -> sensitive to dy
    all_dy_one_dx: survives every dy, dies when dx moves -> sensitive to dx
    """
    if pattern == "all_dx_one_dy":
        return "dy"
    if pattern == "all_dy_one_dx":
        return "dx"
    return None


# =========================================================================
# per crown table
# =========================================================================

def build_crowns(pooled, clusters):
    """One row per cluster, with member geometry and sample cells."""
    w = (pooled["xmax"] - pooled["xmin"]).to_numpy(dtype=float)
    h = (pooled["ymax"] - pooled["ymin"]).to_numpy(dtype=float)
    g = pm.GSD_CM / 100.0

    df = pd.DataFrame({
        "cluster_id": pooled["cluster_id"].to_numpy(),
        "support": pooled["support"].to_numpy(),
        "xmin": pooled["xmin"].to_numpy(dtype=float),
        "ymin": pooled["ymin"].to_numpy(dtype=float),
        "xmax": pooled["xmax"].to_numpy(dtype=float),
        "ymax": pooled["ymax"].to_numpy(dtype=float),
        "cx": pooled["cx"].to_numpy(dtype=float),
        "cy": pooled["cy"].to_numpy(dtype=float),
        "w_px": w,
        "h_px": h,
        "dx": pooled["dx"].to_numpy(dtype=int),
        "dy": pooled["dy"].to_numpy(dtype=int),
    })

    agg = df.groupby("cluster_id").agg(
        support=("support", "first"),
        med_cx=("cx", "median"),
        med_cy=("cy", "median"),
        med_xmin=("xmin", "median"),
        med_ymin=("ymin", "median"),
        med_xmax=("xmax", "median"),
        med_ymax=("ymax", "median"),
        med_w_px=("w_px", "median"),
        med_h_px=("h_px", "median"),
    )
    agg["med_w_m"] = agg["med_w_px"] * g
    agg["med_h_m"] = agg["med_h_px"] * g
    agg["med_aspect"] = (
        np.maximum(agg["med_w_px"], agg["med_h_px"])
        / np.maximum(np.minimum(agg["med_w_px"], agg["med_h_px"]), 1e-9)
    )

    # spatial variance, Miller et al. Sample variance, ddof 1, so a single
    # member cluster gives NaN rather than a fake zero. See report_variance().
    var = df.groupby("cluster_id").agg(
        var_xmin=("xmin", lambda s: s.var(ddof=1)),
        var_xmax=("xmax", lambda s: s.var(ddof=1)),
        var_ymin=("ymin", lambda s: s.var(ddof=1)),
        var_ymax=("ymax", lambda s: s.var(ddof=1)),
    )
    agg["var_x"] = var["var_xmin"] + var["var_xmax"]
    agg["var_y"] = var["var_ymin"] + var["var_ymax"]
    agg["var_total"] = agg["var_x"] + agg["var_y"]

    # sample cells and offsets present
    cells, dxs, dys = {}, {}, {}
    for cid, dx, dy in zip(df["cluster_id"], df["dx"], df["dy"]):
        ix, iy = pm.phase_index(dx, dy)
        cells.setdefault(cid, set()).add((ix, iy))
        dxs.setdefault(cid, set()).add(int(dx))
        dys.setdefault(cid, set()).add(int(dy))

    agg = agg.reset_index()
    agg["cells"] = agg["cluster_id"].map(lambda c: sorted(cells[c]))
    agg["dx_present"] = agg["cluster_id"].map(lambda c: sorted(dxs[c]))
    agg["dy_present"] = agg["cluster_id"].map(lambda c: sorted(dys[c]))
    agg["pattern"] = agg["cells"].map(classify_pattern)
    agg["sensitive_axis"] = agg["pattern"].map(sensitive_axis)
    return agg


# =========================================================================
# TASK 1
# =========================================================================

def elongation_axis(row):
    if row["med_h_px"] > row["med_w_px"]:
        return "taller"
    if row["med_w_px"] > row["med_h_px"]:
        return "wider"
    return "square"


def report_elongation(crowns):
    print("=" * 70)
    print("TASK 1: ELONGATION AXIS VERSUS SENSITIVITY AXIS")
    print("=" * 70)
    print("prediction: sensitive to dx -> taller than wide")
    print("            sensitive to dy -> wider than tall")
    print("geometry is the median box across the cluster's observations")
    print("")

    out = []
    for k in STRUCTURED_SUPPORTS:
        sub = crowns[(crowns["support"] == k)
                     & crowns["sensitive_axis"].notna()].copy()
        label = f"support {k}"
        print(f"--- {label}, {len(sub)} single axis sensitive crowns ---")
        if len(sub) == 0:
            print("  none")
            print("")
            continue

        sub["elongation"] = sub.apply(elongation_axis, axis=1)
        n_square = int((sub["elongation"] == "square").sum())
        if n_square:
            print(f"  {n_square} crowns are exactly square, excluded from the")
            print("  2 by 2 test and reported here rather than dropped quietly")

        t = sub[sub["elongation"] != "square"]
        tab = pd.crosstab(t["sensitive_axis"], t["elongation"])
        tab = tab.reindex(index=["dx", "dy"],
                          columns=["taller", "wider"], fill_value=0)
        print("")
        print("  rows = sensitive axis, columns = elongation")
        print(tab.to_string())

        a = int(tab.loc["dx", "taller"])   # predicted
        b = int(tab.loc["dx", "wider"])
        c = int(tab.loc["dy", "taller"])
        d = int(tab.loc["dy", "wider"])    # predicted
        n = a + b + c + d
        concordant = a + d
        odds, p = fisher_exact_2x2(a, b, c, d)

        print("")
        print(f"  concordant with prediction: {concordant} of {n}"
              + (f" ({100.0 * concordant / n:.2f}%)" if n else ""))
        print(f"  odds ratio                : {odds:.4f}")
        print(f"  Fisher exact, two sided p : {p:.4g}")
        print("")
        print("  Odds ratio above 1 means dx sensitive crowns are taller and")
        print("  dy sensitive crowns are wider, which is the prediction. A")
        print("  ratio near 1 means elongation carries no information about")
        print("  which axis the crown is sensitive to, and the seam severing")
        print("  mechanism is not demonstrated by this test.")
        print("")

        sub["support_level"] = k
        out.append(sub[["cluster_id", "support_level", "pattern",
                        "sensitive_axis", "elongation", "med_w_m", "med_h_m",
                        "med_aspect", "med_cx", "med_cy"]])

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# =========================================================================
# TASK 2
# =========================================================================

def seam_distance(centre, phase):
    """Distance from a coordinate to the nearest tile boundary at a phase."""
    edges = np.array(pm.tile_edges(phase), dtype=float)
    return float(np.min(np.abs(edges - centre)))


def containment_margin(lo, hi, phase):
    """Best slack the box has inside any single tile on this axis.

    For each tile [o, o + PATCH_SIZE), the slack is
    min(lo - o, o + PATCH_SIZE - hi). Positive means the box fits inside that
    tile with that much room. The best over all tiles is returned.

    Negative means NO tile on this axis fully contains the box: it is severed
    wherever it is placed. This is the mechanistically meaningful quantity,
    because with 25 percent overlap a tile boundary is interior to its
    neighbour, so being near a boundary is not by itself harmful.
    """
    best = -np.inf
    for o in pm.scored_tile_origins(phase):
        best = max(best, min(lo - o, o + pm.PATCH_SIZE - hi))
    return float(best)


def report_seam_proximity(crowns):
    print("=" * 70)
    print("TASK 2: SEAM PROXIMITY")
    print("=" * 70)
    print("Two quantities, reported separately.")
    print("")
    print("  seam_dist   distance from the crown centre to the nearest tile")
    print("              boundary on the sensitive axis. This is the measure")
    print("              as specified.")
    print("  contain     best slack the crown's box has inside any single")
    print("              tile on that axis. Negative means no tile contains")
    print("              it. With 25 percent overlap a boundary of one tile")
    print("              sits inside its neighbour, so seam_dist alone can")
    print("              mislead. contain is the sharper test.")
    print("")
    print("The crown has no box at the samples where it was missed, so its")
    print("MEDIAN box across the samples where it was found is evaluated")
    print("against every phase's grid. Geometry held fixed, grid varied.")
    print("")

    sub = crowns[crowns["sensitive_axis"].notna()
                 & crowns["support"].isin(STRUCTURED_SUPPORTS)].copy()
    if len(sub) == 0:
        print("no single axis sensitive crowns")
        return pd.DataFrame()

    rows = []
    for _, r in sub.iterrows():
        axis = r["sensitive_axis"]
        if axis == "dx":
            centre, lo, hi = r["med_cx"], r["med_xmin"], r["med_xmax"]
            found_offsets = set(r["dx_present"])
        else:
            centre, lo, hi = r["med_cy"], r["med_ymin"], r["med_ymax"]
            found_offsets = set(r["dy_present"])
        missed_offsets = [p for p in pm.PHASE_OFFSETS
                          if p not in found_offsets]

        f_seam = [seam_distance(centre, p) for p in sorted(found_offsets)]
        m_seam = [seam_distance(centre, p) for p in missed_offsets]
        f_cont = [containment_margin(lo, hi, p) for p in sorted(found_offsets)]
        m_cont = [containment_margin(lo, hi, p) for p in missed_offsets]

        rows.append({
            "cluster_id": r["cluster_id"],
            "support": r["support"],
            "sensitive_axis": axis,
            "med_w_m": r["med_w_m"],
            "med_h_m": r["med_h_m"],
            "seam_found": float(np.mean(f_seam)),
            "seam_missed": float(np.mean(m_seam)) if m_seam else np.nan,
            "contain_found": float(np.mean(f_cont)),
            "contain_missed": float(np.mean(m_cont)) if m_cont else np.nan,
        })

    t = pd.DataFrame(rows)
    t["seam_diff"] = t["seam_found"] - t["seam_missed"]
    t["contain_diff"] = t["contain_found"] - t["contain_missed"]

    print(f"crowns tested: {len(t)}")
    print("")
    for col, label, direction in [
        ("seam", "seam distance, px", "found should be LARGER"),
        ("contain", "containment margin, px", "found should be LARGER"),
    ]:
        f = t[f"{col}_found"]
        m = t[f"{col}_missed"]
        d = t[f"{col}_diff"].dropna()
        n_pos = int((d > 0).sum())
        n_neg = int((d < 0).sum())
        p = sign_test_two_sided(n_pos, n_neg)
        print(f"--- {label} ({direction}) ---")
        print(f"  at samples found  : median {f.median():8.3f}  "
              f"mean {f.mean():8.3f}")
        print(f"  at samples missed : median {m.median():8.3f}  "
              f"mean {m.mean():8.3f}")
        print(f"  paired difference : median {d.median():8.3f}")
        print(f"  found > missed    : {n_pos} of {n_pos + n_neg} discordant, "
              f"sign test two sided p {p:.4g}")
        print("")

    n_never = int((t["contain_missed"] < 0).sum())
    n_found_cut = int((t["contain_found"] < 0).sum())
    print(f"crowns with negative containment at the missed samples : {n_never}")
    print(f"crowns with negative containment at the found samples  : {n_found_cut}")
    print("")
    print("The mechanism predicts the first number is large and the second is")
    print("small: severed where it was missed, intact where it was found.")
    print("")

    by_axis = t.groupby("sensitive_axis")[
        ["seam_found", "seam_missed", "contain_found", "contain_missed"]
    ].median().round(3)
    print("medians split by sensitive axis:")
    print(by_axis.to_string())
    print("")
    return t


# =========================================================================
# TASK 3
# =========================================================================

def report_variance(crowns):
    print("=" * 70)
    print("TASK 3: SPATIAL VARIANCE PER CLUSTER")
    print("=" * 70)
    print("Miller et al., ICRA 2019, arXiv 1809.06006: spatial uncertainty as")
    print("the total variance of box coordinates within a cluster.")
    print("")
    print("  var_x = Var(xmin) + Var(xmax)")
    print("  var_y = Var(ymin) + Var(ymax)")
    print("  total = var_x + var_y")
    print("")
    print("HANDLING OF SINGLE OBSERVATION CLUSTERS, stated rather than")
    print("silently applied: variance is sample variance with ddof 1, so a")
    print("cluster with one observation gives NaN, NOT zero. Zero would be a")
    print("fabricated claim of perfect agreement from a cluster that has")
    print("nothing to agree with, and it would drag every band median down.")
    print("Those clusters are counted and excluded from variance statistics,")
    print("never folded in. Support 2 clusters have one degree of freedom and")
    print("are noisy, so the by support table is the honest read, not the")
    print("band table.")
    print("")

    n_single = int((crowns["support"] == 1).sum())
    usable = crowns[crowns["support"] >= 2]
    print(f"clusters total                 : {len(crowns)}")
    print(f"single observation, variance NaN: {n_single}")
    print(f"usable for variance            : {len(usable)}")
    assert usable["var_total"].notna().all(), \
        "a multi observation cluster produced NaN variance"
    print("")

    lvl = (usable.groupby("support")
           .agg(n_clusters=("cluster_id", "size"),
                med_var_x=("var_x", "median"),
                med_var_y=("var_y", "median"),
                med_var_total=("var_total", "median"))
           .reindex(range(2, pm.N_PHASES + 1)).round(3))
    print("--- by support level ---")
    print(lvl.to_string())
    print("")

    print("--- monotonicity with support ---")
    for col in ["var_x", "var_y", "var_total"]:
        sub = usable[[col, "support"]].dropna()
        r = spearman(sub["support"], sub[col])
        print(f"  support vs {col:9s}: rho {r:+.4f}")
    print("")
    print("Miller et al. use this to separate spatially accurate from")
    print("inaccurate observations. Here a negative rho would mean")
    print("well supported crowns also agree on where they are.")
    print("")

    # --- the axis test --------------------------------------------------
    print("--- second reading on single axis sensitivity ---")
    print("prediction: sensitive to dx -> var_y > var_x")
    print("            sensitive to dy -> var_x > var_y")
    print("because the sensitive axis is the one held CONSTANT across the")
    print("cluster's samples, so its coordinates should be the stable ones.")
    print("")
    print("This uses the same clusters as Task 1, so it is a different")
    print("quantity but NOT a statistically independent test.")
    print("")

    sub = usable[usable["sensitive_axis"].notna()].copy()
    if len(sub) == 0:
        print("no single axis sensitive clusters with 2 or more observations")
        return usable

    sub["larger_var"] = np.where(sub["var_y"] > sub["var_x"], "var_y",
                                 np.where(sub["var_x"] > sub["var_y"],
                                          "var_x", "equal"))
    t = sub[sub["larger_var"] != "equal"]
    tab = pd.crosstab(t["sensitive_axis"], t["larger_var"])
    tab = tab.reindex(index=["dx", "dy"],
                      columns=["var_x", "var_y"], fill_value=0)
    print("rows = sensitive axis, columns = which variance is larger")
    print(tab.to_string())

    a = int(tab.loc["dx", "var_y"])    # predicted
    b = int(tab.loc["dx", "var_x"])
    c = int(tab.loc["dy", "var_y"])
    d = int(tab.loc["dy", "var_x"])    # predicted
    n = a + b + c + d
    odds, p = fisher_exact_2x2(a, b, c, d)
    print("")
    print(f"  concordant with prediction: {a + d} of {n}"
          + (f" ({100.0 * (a + d) / n:.2f}%)" if n else ""))
    print(f"  odds ratio                : {odds:.4f}")
    print(f"  Fisher exact, two sided p : {p:.4g}")
    print("")
    return usable


def rankdata(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


# =========================================================================
# main
# =========================================================================

def run():
    print("MATCH_IOU:", MATCH_IOU)
    print("")
    pool = pm.load_pool(OUT_DIR)
    clusters, cluster_of = pm.cluster_across_phases(pool, MATCH_IOU)
    pooled = pm.attach_clusters(pool, cluster_of, clusters)
    print("")
    print("clusters:", len(clusters))
    print("")

    crowns = build_crowns(pooled, clusters)

    elong = report_elongation(crowns)
    if len(elong):
        elong.to_csv(os.path.join(OUT_DIR, "mechanism_elongation.csv"),
                     index=False)

    seam = report_seam_proximity(crowns)
    if len(seam):
        seam.to_csv(os.path.join(OUT_DIR, "mechanism_seam_distance.csv"),
                    index=False)

    var = report_variance(crowns)
    crowns.drop(columns=["cells", "dx_present", "dy_present"]).to_csv(
        os.path.join(OUT_DIR, "mechanism_spatial_var.csv"), index=False)


if __name__ == "__main__":
    run()
