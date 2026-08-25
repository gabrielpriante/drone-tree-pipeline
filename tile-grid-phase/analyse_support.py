"""
Support structure analysis.

Reads the per phase box CSVs and phase_summary.csv. No model, no raster, no
inference. Answers four things.

1. BOX GEOMETRY BY SUPPORT
   Width, height, area and aspect ratio per crown, related to support count.

   The hypothesis going in was that unstable detections are systematically
   larger and more malformed than stable ones. HALF OF IT WAS WRONG. First run
   result:

       width  rho +0.3043    stable crowns are LARGER
       area   rho +0.3605    stable crowns are LARGER
       aspect rho -0.4597    UNCONDITIONAL, confounded with seam pinning

   Band medians: singletons 1.59 m wide, 2.16 m2, aspect 1.72, 43.88 percent
   above aspect 2. Support 16: 2.49 m, 6.36 m2, aspect 1.05, 0.00 percent
   above aspect 2.

   So the size hypothesis is refuted. The shape association is UNCONDITIONAL
   and is confounded with seam pinning: the conditional test in
   analyse_geometry_support.py returned r_rb +0.2223, verdict INCONCLUSIVE.
   Unstable detections are SMALL and ELONGATED, but severing is refuted and
   the slivers are manufactured by the tiling, not severed by it.

2. PATTERN OF SUPPORT AT 4, 8 AND 12
   Whether crowns found at exactly 4 phases were found at all four dx offsets
   for a single dy, all four dy for a single dx, a 2 by 2 block, or scattered.
   Compared against an exact null, see below.

3. AXIS MARGINALS
   How many distinct dx and how many distinct dy each crown was found at, as a
   joint 4 by 4 table. A direct test of whether dx and dy contribute equally.

4. PER PHASE COUNT SPREAD
   Core counts across phases, quoted over the 15 four tile phases with phase
   (0, 0) separate, since phase 0 runs 25 tiles per phase against 16 elsewhere
   and is a different tiling regime.

Which box is used for a crown's geometry
----------------------------------------
A crown detected at k phases has k boxes. Geometry is the MEDIAN across those
k member boxes, taken per statistic: median of the member widths, median of
the member areas, median of the member aspect ratios. Not derived from each
other, so the reported aspect is a real member's typical aspect rather than a
ratio of two medians.

Why the median:
  It is the central tendency of how the crown appears across the phases where
  it was found, and it is robust to one phase producing an outlier box.
  It degenerates to the single box at support 1, so the same definition applies
  to every band with no special casing. A seed box rule, for instance, would
  make singletons and multi phase crowns incomparable, because the seed is by
  construction the highest scoring member.

Caveat: a support 1 crown's statistic comes from one box and a support 16
crown's from sixteen, so per crown noise differs by band. Band medians over
many crowns remain comparable. Seed box geometry is reported alongside as a
sensitivity check, and if the two tell different stories, say so rather than
picking one.

Aspect ratio is defined as the long side over the short side, so it is always
at or above 1 and larger means more elongated.

Outputs (gitignored)
--------------------
crown_geometry.csv          one row per crown
geometry_by_band.csv        1, 2 to 15, 16
geometry_by_support.csv     all 16 support levels
support_patterns.csv        pattern classes at support 4, 8, 12, with null
axis_marginals.csv          joint distribution of distinct dx by distinct dy
phase_count_spread.csv      per phase counts, with and without phase 0

Not run yet.
"""

import os
from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd

import phase_matching as pm

MATCH_IOU = 0.5           # primary threshold. Set to 0.3 for the conservative
                          # case, see the README on threshold sensitivity.
SUPPORTS_TO_CLASSIFY = [4, 8, 12]
PHASE_ZERO = "dx000_dy000"

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================================
# small stats helpers, so scipy is not required
# =========================================================================

def rankdata(a):
    """Average ranks, ties shared."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average over tied groups
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
    """Spearman rank correlation. No p value, the sign and size are the point."""
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def sign_test_two_sided(n_pos, n_neg):
    """Exact two sided sign test on discordant pairs."""
    n = n_pos + n_neg
    if n == 0:
        return float("nan")
    k = min(n_pos, n_neg)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


# =========================================================================
# pattern classification
# =========================================================================

def classify_pattern(cells):
    """cells is an iterable of (ix, iy) grid indices, 0 to 3 on each axis.

    Returns one of:
        all_dx_one_dy      every dx offset, at a single dy
        all_dy_one_dx      every dy offset, at a single dx
        block_<a>dy_x_<b>dx   a full rectangular block, a rows by b columns
        scattered          anything else
    """
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


def null_pattern_distribution(k):
    """Exact null: every k subset of the 4 by 4 grid, equally likely.

    Enumerated in full, not sampled. C(16,4)=1820, C(16,8)=12870,
    C(16,12)=1820, all trivially small.

    This is the baseline the observed counts have to beat. Structured patterns
    are rare by chance: at support 4 only 2.42 percent of subsets are a full
    row, column or 2 by 2 block.
    """
    cells = [(x, y) for y in range(pm.PHASES_PER_AXIS)
             for x in range(pm.PHASES_PER_AXIS)]
    counts = Counter(classify_pattern(s) for s in combinations(cells, k))
    total = sum(counts.values())
    return {name: n / total for name, n in counts.items()}, total


# =========================================================================
# per crown geometry
# =========================================================================

def crown_geometry(pooled, clusters):
    """One row per crown. Median across member boxes, plus seed box."""
    w = (pooled["xmax"] - pooled["xmin"]).to_numpy(dtype=float)
    h = (pooled["ymax"] - pooled["ymin"]).to_numpy(dtype=float)
    g = pm.GSD_CM / 100.0

    df = pd.DataFrame({
        "cluster_id": pooled["cluster_id"].to_numpy(),
        "support": pooled["support"].to_numpy(),
        "score": pooled["score"].to_numpy(dtype=float),
        "w_m": w * g,
        "h_m": h * g,
        "area_m2": (w * g) * (h * g),
        "aspect": np.maximum(w, h) / np.maximum(np.minimum(w, h), 1e-9),
        "ix": [pm.phase_index(dx, dy)[0]
               for dx, dy in zip(pooled["dx"], pooled["dy"])],
        "iy": [pm.phase_index(dx, dy)[1]
               for dx, dy in zip(pooled["dx"], pooled["dy"])],
    })

    med = df.groupby("cluster_id").agg(
        support=("support", "first"),
        n_boxes=("score", "size"),
        med_w_m=("w_m", "median"),
        med_h_m=("h_m", "median"),
        med_area_m2=("area_m2", "median"),
        med_aspect=("aspect", "median"),
        n_dx=("ix", "nunique"),
        n_dy=("iy", "nunique"),
    )

    # seed box, highest scoring member, as a sensitivity check
    seed_idx = df.groupby("cluster_id")["score"].idxmax()
    seed = df.loc[seed_idx].set_index("cluster_id")[
        ["w_m", "area_m2", "aspect"]
    ].rename(columns={"w_m": "seed_w_m",
                      "area_m2": "seed_area_m2",
                      "aspect": "seed_aspect"})

    out = med.join(seed).reset_index()

    # cell sets, for pattern classification. Built without groupby.apply so
    # this does not depend on the pandas include_groups behaviour, which
    # changed in 2.2.
    cells = {}
    for cid, ix, iy in zip(df["cluster_id"], df["ix"], df["iy"]):
        cells.setdefault(cid, set()).add((ix, iy))
    out["cells"] = out["cluster_id"].map(lambda c: sorted(cells[c]))

    assert (out["n_boxes"] == out["support"]).all(), \
        "member count does not equal support, clustering invariant broken"
    return out


def band_of(s):
    if s == pm.N_PHASES:
        return "16"
    if s == 1:
        return "1"
    return "2 to 15"


# =========================================================================
# reporting
# =========================================================================

def report_geometry(crowns):
    print("=" * 68)
    print("TASK 1: BOX GEOMETRY BY SUPPORT")
    print("=" * 68)
    print("geometry per crown = median across its member boxes")
    print("aspect = long side / short side, at or above 1")
    print("")

    crowns = crowns.copy()
    crowns["band"] = crowns["support"].map(band_of)

    order = ["1", "2 to 15", "16"]
    band = (crowns.groupby("band")
            .agg(n_crowns=("cluster_id", "size"),
                 med_width_m=("med_w_m", "median"),
                 med_area_m2=("med_area_m2", "median"),
                 med_aspect=("med_aspect", "median"),
                 pct_aspect_gt_2=("med_aspect", lambda s: 100.0 * (s > 2).mean()),
                 seed_med_width_m=("seed_w_m", "median"),
                 seed_med_area_m2=("seed_area_m2", "median"),
                 seed_med_aspect=("seed_aspect", "median"))
            .reindex(order).round(4))
    print("--- by band ---")
    print(band.to_string())
    print("")

    lvl = (crowns.groupby("support")
           .agg(n_crowns=("cluster_id", "size"),
                med_width_m=("med_w_m", "median"),
                med_height_m=("med_h_m", "median"),
                med_area_m2=("med_area_m2", "median"),
                med_aspect=("med_aspect", "median"))
           .reindex(range(1, pm.N_PHASES + 1)).round(4))
    print("--- by support level, all 16 ---")
    print(lvl.to_string())
    print("")

    print("--- monotonicity, Spearman rank correlation with support ---")
    for col, label in [("med_w_m", "width"),
                       ("med_area_m2", "area"),
                       ("med_aspect", "aspect")]:
        r = spearman(crowns["support"], crowns[col])
        print(f"  support vs {label:7s}: rho {r:+.4f}")
    print("")
    print("Sign convention: positive rho means the quantity RISES with")
    print("support, negative means it falls.")
    print("")
    print("RESULT OF THE FIRST RUN, recorded so this text is not misread")
    print("again. The size hypothesis was REFUTED. The shape association is")
    print("UNCONDITIONAL and is confounded with seam pinning, see")
    print("analyse_geometry_support.py: r_rb +0.2223, verdict INCONCLUSIVE.")
    print("  width  rho +0.3043   stable crowns are LARGER, not smaller")
    print("  area   rho +0.3605   stable crowns are LARGER, not smaller")
    print("  aspect rho -0.4597   UNCONDITIONAL, confounded with pinning")
    print("An earlier version of this note claimed negative was the")
    print("predicted direction for all three. That was wrong.")
    print("")
    print("Check the seed box columns against the median columns. If they")
    print("disagree in direction, the band difference depends on which member")
    print("box is used and should be reported as such, not as a fact.")
    print("")
    return band, lvl


def report_patterns(crowns):
    print("=" * 68)
    print("TASK 2: PATTERN OF SUPPORT AT 4, 8 AND 12")
    print("=" * 68)
    rows = []
    for k in SUPPORTS_TO_CLASSIFY:
        sub = crowns[crowns["support"] == k]
        n = len(sub)
        print(f"--- support {k}, {n} crowns ---")
        if n == 0:
            print("  none")
            print("")
            continue
        obs = Counter(classify_pattern(c) for c in sub["cells"])
        null, n_subsets = null_pattern_distribution(k)
        names = sorted(set(obs) | set(null),
                       key=lambda s: -obs.get(s, 0))
        print(f"  {'class':22s} {'observed':>9s} {'obs pct':>9s} "
              f"{'null pct':>9s} {'expected':>9s} {'ratio':>7s}")
        for name in names:
            o = obs.get(name, 0)
            p = null.get(name, 0.0)
            e = p * n
            ratio = (o / e) if e > 0 else float("inf") if o else float("nan")
            print(f"  {name:22s} {o:9d} {100.0 * o / n:8.2f}% "
                  f"{100.0 * p:8.4f}% {e:9.2f} {ratio:7.2f}")
            rows.append({"support": k, "pattern": name, "observed": o,
                         "obs_pct": round(100.0 * o / n, 3),
                         "null_pct": round(100.0 * p, 5),
                         "expected": round(e, 3),
                         "obs_over_expected": round(ratio, 3)
                         if np.isfinite(ratio) else None})
        structured = n - obs.get("scattered", 0)
        null_scattered = null.get("scattered", 0.0)
        print(f"  structured (not scattered): {structured} of {n} "
              f"({100.0 * structured / n:.2f}%), null expects "
              f"{100.0 * (1 - null_scattered):.2f}%")
        print("")
    print("Null is exact, every k subset of the 4 by 4 grid enumerated, not")
    print("sampled. A ratio near 1 means the pattern is what chance gives. A")
    print("large ratio on all_dx_one_dy or all_dy_one_dx is the anisotropy")
    print("signal. A large ratio on a block class is not anisotropy, it is")
    print("spatial coherence in both axes at once.")
    print("")
    return pd.DataFrame(rows)


def report_marginals(crowns):
    print("=" * 68)
    print("TASK 3: AXIS MARGINALS")
    print("=" * 68)
    n_ax = pm.PHASES_PER_AXIS
    joint = pd.crosstab(crowns["n_dx"], crowns["n_dy"], dropna=False)
    joint = joint.reindex(index=range(1, n_ax + 1),
                          columns=range(1, n_ax + 1), fill_value=0)
    joint.index.name = "n_dx"
    joint.columns.name = "n_dy"
    print("joint counts, rows are distinct dx, columns are distinct dy:")
    print(joint.to_string())
    print("")
    print("row totals (distinct dx):", joint.sum(axis=1).to_dict())
    print("col totals (distinct dy):", joint.sum(axis=0).to_dict())
    print("")
    print("mean distinct dx:", round(float(crowns["n_dx"].mean()), 4))
    print("mean distinct dy:", round(float(crowns["n_dy"].mean()), 4))

    n_pos = int((crowns["n_dx"] > crowns["n_dy"]).sum())
    n_neg = int((crowns["n_dx"] < crowns["n_dy"]).sum())
    n_tie = len(crowns) - n_pos - n_neg
    p = sign_test_two_sided(n_pos, n_neg)
    print("")
    print("n_dx > n_dy :", n_pos)
    print("n_dx < n_dy :", n_neg)
    print("equal       :", n_tie)
    print("sign test on discordant pairs, two sided p:",
          f"{p:.3g}" if np.isfinite(p) else "n/a")
    print("")
    print("Equal contribution predicts a symmetric table and roughly equal")
    print("discordant counts. A lopsided table means one axis matters more,")
    print("which would point at something directional in the imagery or the")
    print("model, not at the grid alone.")
    print("")
    return joint


def report_phase_spread():
    print("=" * 68)
    print("TASK 4: PER PHASE COUNT SPREAD, PHASE 0 SEPARATED")
    print("=" * 68)
    path = os.path.join(OUT_DIR, "phase_summary.csv")
    if not os.path.exists(path):
        print("phase_summary.csv not found, skipping")
        return None
    s = pd.read_csv(path)
    print(s[["phase_id", "n_tiles", "n_scored", "n_core"]].to_string(index=False))
    print("")

    zero = s[s["phase_id"] == PHASE_ZERO]
    rest = s[s["phase_id"] != PHASE_ZERO]

    for label, frame in [("all 16 phases", s),
                         ("15 four tile phases", rest)]:
        for col in ["n_scored", "n_core"]:
            v = frame[col]
            print(f"{label:22s} {col:9s}: min {v.min():4d} max {v.max():4d} "
                  f"mean {v.mean():8.2f} sd {v.std():7.3f} "
                  f"cv {v.std() / v.mean():.4f}")
    print("")
    if len(zero):
        z = zero.iloc[0]
        print(f"phase {PHASE_ZERO} quoted separately: "
              f"{int(z['n_tiles'])} tiles, "
              f"{int(z['n_scored'])} scored, {int(z['n_core'])} core")
        for col in ["n_scored", "n_core"]:
            excess = 100.0 * (z[col] / rest[col].mean() - 1.0)
            print(f"  {col}: {excess:+.2f}% against the 15 phase mean")
    print("")
    print("Quote the 15 phase spread as the experiment's spread. Phase 0 runs")
    print("a different tile count and belongs in its own line.")
    print("")
    return s


# =========================================================================
# main
# =========================================================================

def run():
    print("MATCH_IOU:", MATCH_IOU)
    print("")
    pool = pm.load_pool(OUT_DIR)
    clusters, cluster_of = pm.cluster_across_phases(pool, MATCH_IOU)
    pooled = pm.attach_clusters(pool, cluster_of, clusters)
    print("distinct crowns:", len(clusters))
    print("")

    crowns = crown_geometry(pooled, clusters)
    crowns.drop(columns=["cells"]).to_csv(
        os.path.join(OUT_DIR, "crown_geometry.csv"), index=False)

    band, lvl = report_geometry(crowns)
    band.to_csv(os.path.join(OUT_DIR, "geometry_by_band.csv"))
    lvl.to_csv(os.path.join(OUT_DIR, "geometry_by_support.csv"))

    pat = report_patterns(crowns)
    if len(pat):
        pat.to_csv(os.path.join(OUT_DIR, "support_patterns.csv"), index=False)

    joint = report_marginals(crowns)
    joint.to_csv(os.path.join(OUT_DIR, "axis_marginals.csv"))

    spread = report_phase_spread()
    if spread is not None:
        spread.to_csv(os.path.join(OUT_DIR, "phase_count_spread.csv"),
                      index=False)


if __name__ == "__main__":
    run()
