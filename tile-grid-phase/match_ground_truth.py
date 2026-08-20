"""
Match detections against ground truth annotations.

Every choice below was fixed before any number was produced and none of them
changed after results were seen.

WHY CONTAINMENT IS PRIMARY AND IoU IS SECONDARY
-----------------------------------------------
The median annotated tree is 4.47 by 5.52 m. The median detection at working
resolution is 2.10 m wide. A detection half the linear size of its tree tops
out near 0.25 IoU even when perfectly centred, so an IoU 0.5 criterion would
report near total failure as an artefact of the metric rather than a property
of the detector.

So the primary measurement is containment: how many detections fall inside
each annotated tree. IoU runs second, for comparability with other work, and
is labelled as penalising fragmentation by construction.

PRIMARY, detections per annotated tree
--------------------------------------
For each annotated tree, count detections with at least 50 percent of the
DETECTION's own area inside the annotation. Note the direction: it is the
detection's area that must be mostly inside the tree, not the tree's area
covered by the detection. A small fragment sitting wholly inside a large crown
counts. A huge box swallowing the crown does not.

Reported as a distribution, never a mean: 0, 1, 2, 3, 4, 5 or more, plus the
median and the full histogram, plus detections contained in no annotation.

This is the over segmentation measurement.

SECONDARY, one to one IoU
-------------------------
Hungarian assignment at IoU 0.5. Matched, unmatched annotations, unmatched
detections.

DETECTION SETS, reported separately and never pooled
----------------------------------------------------
Primary   dx225_dy075, the 274 core detections from phase_boxes_dx225_dy075.csv
Secondary the 710 cluster union from phase_stability.csv, mean box corners

SCORING SETS, nested, five rows
-------------------------------
1  live, canopy, certain, not edge clipped
2  plus edge clipped
3  plus uncertain
4  plus understory
5  plus snags

Because the export used mutually exclusive labels, these nest exactly onto the
label values: set 1 is the `tree` label minus edge clipped, and each later set
adds one label group back. See ANNOTATION_PROTOCOL.md limitation 7.

THE SUPPORT QUESTION
--------------------
For the 274 detections at dx225_dy075, is support associated with landing
inside an annotated tree? Reported as the contained share for the 115 found by
all sixteen, the 13 found once, and the 146 in between.

THE TRUE POSITIVE RULE, fixed after the distribution was seen
--------------------------------------------------------------
One to one Hungarian assignment at IoU 0.5 is the true positive rule for
reported precision, recall and F1. That is what the literature reports and it
is what makes these numbers comparable to other work.

Reported ALONGSIDE it, clearly separated and never folded into any F1:

    tree level detection rate       share of annotated trees with at least one
                                    detection at 50 percent containment
    detection level containment     share of detections landing inside some
                                    annotation

The standard metric looks catastrophic while the containment view shows the
detector finding trees and splitting them. That tension is not resolved by
picking one. It is the argument.

THE THREE RATES HAVE DIFFERENT DENOMINATORS AND DIFFERENT UNITS. Precision is
per detection under a one to one rule. Recall is per annotation under the same
rule. Tree level detection rate is per annotation under a containment rule.
Detection level containment is per detection under a containment rule. They
must never be arithmetically combined, averaged, or fed into a single summary
score.

ANNOTATION COVERAGE, method recorded because the number is quoted
------------------------------------------------------------------
Coverage is computed by rasterisation, not by summing box areas. Annotations
overlap by design, see protocol 2.3, so summing areas would double count and
could exceed 100 percent.

Method: a 950 by 950 boolean grid in CHIP coordinates, one cell per chip pixel.
Every annotation in the scoring set marks its half open span
[xmin - 25 : xmax - 25] by [ymin - 25 : ymax - 25] as True. Coverage is the
mean of the grid. Overlap therefore counts once.

The number bounds what a false positive claim could ever mean: a detection
outside every annotation may be in genuinely unannotated ground.

SUPPORT ASSOCIATION, computed here rather than by hand
-------------------------------------------------------
Fisher exact two sided, all sixteen against some but not all, and all sixteen
against found once. Plus the Spearman of support against the inside indicator
over all 274 detections. All three on every scoring set, written to
match_metrics.csv so a rerun regenerates them.

Coordinates
-----------
Annotations are in window coordinates after the +25 conversion, the same frame
as phase_boxes_*.csv. One annotation position is ASSERTED against its known
value before any matching runs.
"""

import os
from math import comb

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import phase_matching as pm

MATCH_IOU_CLUSTER = 0.5      # the cross survey clustering, unchanged
PRIMARY_PHASE = "dx225_dy075"
CONTAIN_FRAC = 0.50          # of the DETECTION's own area
IOU_THRESH = 0.50            # secondary, Hungarian
GSD_M = 7.78 / 100.0
CORE_INSET = 25              # chip to window offset, phase_matching.py
CHIP_SIZE = 950              # core_clean.png side, for the coverage raster

# --- the assert. Annotation tree_id 14, the largest box. ----------------
KNOWN_TREE_ID = 14
KNOWN_WINDOW_BOX = (475.0, 577.0, 684.0, 801.0)

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.join(HERE, "ground_truth", "annotations_raw.csv")

SCORING_SETS = [
    ("1 live canopy certain, not edge clipped", {"tree"}, False),
    ("2 plus edge clipped", {"tree"}, True),
    ("3 plus uncertain", {"tree", "uncertain"}, True),
    ("4 plus understory", {"tree", "uncertain", "understory"}, True),
    ("5 plus snags", {"tree", "uncertain", "understory", "snag"}, True),
]


# =========================================================================
# geometry
# =========================================================================

def pair_intersection(a, b):
    """(Na, 4) against (Nb, 4). Returns (Na, Nb) intersection area."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)


def areas(a):
    return (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])


def iou_matrix(a, b):
    inter = pair_intersection(a, b)
    union = areas(a)[:, None] + areas(b)[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def contained_matrix(ann, det):
    """(n_ann, n_det). Fraction of each DETECTION's area inside each tree."""
    inter = pair_intersection(ann, det)
    return inter / np.maximum(areas(det)[None, :], 1e-9)


# =========================================================================
# data
# =========================================================================

def load_annotations():
    gt = pd.read_csv(GT)
    r = gt[gt["tree_id"] == KNOWN_TREE_ID].iloc[0]
    got = (r["xmin"], r["ymin"], r["xmax"], r["ymax"])
    assert got == KNOWN_WINDOW_BOX, (
        f"annotation frame assert FAILED. tree_id {KNOWN_TREE_ID} is at {got}, "
        f"expected {KNOWN_WINDOW_BOX}. Do not trust anything below."
    )
    print(f"annotation frame assert : PASS, tree_id {KNOWN_TREE_ID} at {got}")
    return gt


def load_detections():
    pool = pm.load_pool(HERE, verbose=False)
    clusters, cluster_of = pm.cluster_across_phases(pool, MATCH_IOU_CLUSTER)
    pooled = pm.attach_clusters(pool, cluster_of, clusters)

    primary = pooled[pooled["phase_id"] == PRIMARY_PHASE].reset_index(drop=True)

    stab = pd.read_csv(os.path.join(HERE, "phase_stability.csv"))
    union = pd.DataFrame({
        "xmin": stab["mean_xmin"], "ymin": stab["mean_ymin"],
        "xmax": stab["mean_xmax"], "ymax": stab["mean_ymax"],
        "support": stab["n_phases"],
    })
    return primary, union, clusters


# =========================================================================
# primary
# =========================================================================

def containment_report(ann_df, det, label, det_name):
    a = ann_df[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    d = det[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    C = contained_matrix(a, d) >= CONTAIN_FRAC

    per_tree = C.sum(axis=1)
    per_det = C.sum(axis=0)

    bins = {k: int((per_tree == k).sum()) for k in range(5)}
    bins["5+"] = int((per_tree >= 5).sum())

    print(f"  {label:<42s} n_ann {len(a):4d}   "
          f"0:{bins[0]:4d}  1:{bins[1]:4d}  2:{bins[2]:4d}  3:{bins[3]:4d}  "
          f"4:{bins[4]:4d}  5+:{bins['5+']:4d}   "
          f"median {np.median(per_tree):.1f}   "
          f"det in none {int((per_det == 0).sum()):4d}/{len(d)}   "
          f"det in >1 tree {int((per_det > 1).sum()):4d}")
    return per_tree, per_det, C


def full_histogram(per_tree, title):
    print("")
    print(f"  full histogram, {title}")
    mx = int(per_tree.max())
    for k in range(mx + 1):
        n = int((per_tree == k).sum())
        bar = "#" * int(round(40 * n / max(1, len(per_tree))))
        print(f"    {k:>2d} detections : {n:4d}  {bar}")
    print(f"    total trees    : {len(per_tree)}")
    print(f"    median         : {np.median(per_tree):.1f}")
    print(f"    mean, for reference only : {per_tree.mean():.2f}")


# =========================================================================
# secondary
# =========================================================================

def hungarian_report(ann_df, det, label):
    a = ann_df[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    d = det[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    M = iou_matrix(a, d)
    ri, ci = linear_sum_assignment(-M)
    keep = M[ri, ci] >= IOU_THRESH
    n = int(keep.sum())
    med = float(np.median(M[ri, ci][keep])) if n else float("nan")
    print(f"  {label:<42s} matched {n:4d}   "
          f"unmatched ann {len(a) - n:4d}   unmatched det {len(d) - n:5d}   "
          f"median matched IoU {med:.4f}" if n else
          f"  {label:<42s} matched    0   "
          f"unmatched ann {len(a):4d}   unmatched det {len(d):5d}")
    return n, M


def fisher_exact_2x2(a, b, c, d):
    """Two sided Fisher exact test on [[a, b], [c, d]]."""
    n = a + b + c + d
    if n == 0:
        return float("nan")
    row1, col1 = a + b, a + c

    def prob(x):
        y, z = row1 - x, col1 - x
        w = n - x - y - z
        if min(x, y, z, w) < 0:
            return 0.0
        return comb(row1, x) * comb(n - row1, z) / comb(n, col1)

    po = prob(a)
    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= po * 1.0000001))


def spearman(x, y):
    """Rank correlation. Average ranks on ties."""
    rx = pd.Series(x).rank().to_numpy().astype(float)
    ry = pd.Series(y).rank().to_numpy().astype(float)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den else float("nan")


def coverage_share(ann_df):
    """Share of the 950 by 950 chip covered by at least one annotation.

    Rasterised, not summed. Annotations overlap by design, so summing areas
    would double count. See the module docstring.
    """
    g = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=bool)
    for _, r in ann_df.iterrows():
        x0 = int(r["xmin"] - CORE_INSET)
        y0 = int(r["ymin"] - CORE_INSET)
        x1 = int(r["xmax"] - CORE_INSET)
        y1 = int(r["ymax"] - CORE_INSET)
        g[max(0, y0):min(CHIP_SIZE, y1), max(0, x0):min(CHIP_SIZE, x1)] = True
    return float(g.mean())


def wilson(k, n, z=1.96):
    """Wilson score interval. Sane at small n, unlike the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def metrics_report(sets, det, det_name):
    print("=" * 78)
    print("STANDARD METRICS, true positive rule: Hungarian one to one at IoU "
          f"{IOU_THRESH}")
    print("=" * 78)
    print("THESE THREE RATES HAVE DIFFERENT DENOMINATORS AND DIFFERENT UNITS.")
    print("Precision is per detection under a one to one rule. Recall is per")
    print("annotation under the same rule. Tree level detection rate is per")
    print("annotation under a CONTAINMENT rule. Detection level containment is")
    print("per detection under a containment rule. NEVER combine them")
    print("arithmetically, average them, or fold them into one score.")
    print("")

    a_det = det[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    rows = []
    for name, sub in sets:
        a_ann = sub[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
        M = iou_matrix(a_ann, a_det)
        ri, ci = linear_sum_assignment(-M)
        tp = int((M[ri, ci] >= IOU_THRESH).sum())
        prec = tp / len(a_det)
        rec = tp / len(a_ann)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        C = contained_matrix(a_ann, a_det) >= CONTAIN_FRAC
        tree_rate = float((C.sum(axis=1) > 0).mean())
        det_rate = float((C.sum(axis=0) > 0).mean())

        # --- support association, per scoring set ------------------------
        inside = C.any(axis=0)
        sup = det["support"].to_numpy()
        m16 = sup == pm.N_PHASES
        mmid = (sup > 1) & (sup < pm.N_PHASES)
        m1 = sup == 1
        p_some = fisher_exact_2x2(int(inside[m16].sum()),
                                  int((~inside[m16]).sum()),
                                  int(inside[mmid].sum()),
                                  int((~inside[mmid]).sum()))
        p_once = fisher_exact_2x2(int(inside[m16].sum()),
                                  int((~inside[m16]).sum()),
                                  int(inside[m1].sum()),
                                  int((~inside[m1]).sum()))
        rho = spearman(sup.astype(float), inside.astype(float))

        rows.append({
            "scoring_set": name, "n_ann": len(a_ann), "n_det": len(a_det),
            "tp": tp,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tree_detection_rate": round(tree_rate, 4),
            "detection_containment_rate": round(det_rate, 4),
            "annotation_coverage_share": round(coverage_share(sub), 4),
            "fisher_p_all16_vs_some": round(p_some, 4),
            "fisher_p_all16_vs_once": round(p_once, 4),
            "spearman_support_vs_inside": round(rho, 4),
        })

    print(f"  {'scoring set':<42s} {'n_ann':>5s} {'TP':>4s} "
          f"{'prec':>7s} {'recall':>7s} {'F1':>7s} | "
          f"{'tree rate':>9s} {'det rate':>9s}")
    for r in rows:
        print(f"  {r['scoring_set']:<42s} {r['n_ann']:5d} {r['tp']:4d} "
              f"{r['precision']:7.1%} {r['recall']:7.1%} {r['f1']:7.4f} | "
              f"{r['tree_detection_rate']:9.1%} "
              f"{r['detection_containment_rate']:9.1%}")
    print("")
    print("")
    print(f"  {'scoring set':<42s} {'coverage':>9s} {'p vs some':>10s} "
          f"{'p vs once':>10s} {'spearman':>9s}")
    for r in rows:
        print(f"  {r['scoring_set']:<42s} "
              f"{r['annotation_coverage_share']:9.1%} "
              f"{r['fisher_p_all16_vs_some']:10.4f} "
              f"{r['fisher_p_all16_vs_once']:10.4f} "
              f"{r['spearman_support_vs_inside']:+9.4f}")
    print("")
    print("  Coverage is rasterised onto the 950 grid, so overlap counts once.")
    print("  A detection outside every annotation is not a false positive")
    print("  while this much of the chip is unannotated.")
    print("")
    print("  Reading it: precision and recall describe how often ONE detection")
    print("  stands for ONE tree. Tree rate describes how often a tree was")
    print("  found at all. The gap between recall and tree rate is the")
    print("  fragmentation, and it is the result, not a defect in either")
    print("  measurement.")
    print("")
    return pd.DataFrame(rows)


def support_intervals(sets, det):
    """Wilson intervals on the support bands, to judge whether the null reads
    as noise or as three genuinely equal numbers."""
    print("=" * 78)
    print("SUPPORT BANDS WITH 95 PERCENT WILSON INTERVALS")
    print("=" * 78)
    d = det[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    sup = det["support"].to_numpy()
    bands = [("found by all sixteen", sup == pm.N_PHASES),
             ("found by some, not all", (sup > 1) & (sup < pm.N_PHASES)),
             ("found once", sup == 1)]
    rows = []
    for name, sub in [sets[0], sets[-1]]:
        a = sub[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
        inside = (contained_matrix(a, d) >= CONTAIN_FRAC).any(axis=0)
        print(f"  {name}, n_ann {len(a)}")
        for label, m in bands:
            n = int(m.sum())
            k = int(inside[m].sum())
            lo, hi = wilson(k, n)
            print(f"    {label:<24s} {k:4d}/{n:<4d} {k / n:6.1%}   "
                  f"95% CI [{lo:.1%}, {hi:.1%}]   width {hi - lo:.1%}")
            rows.append({"scoring_set": name, "band": label, "k": k, "n": n,
                         "share": round(k / n, 4),
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
        print("")
    return pd.DataFrame(rows)


# =========================================================================
# main
# =========================================================================

def run():
    print("=" * 78)
    print("SETUP")
    print("=" * 78)
    gt = load_annotations()
    primary, union, clusters = load_detections()
    print(f"primary detections      : {PRIMARY_PHASE}, {len(primary)}")
    print(f"secondary detections    : cluster union, {len(union)}")
    print(f"containment criterion   : >= {CONTAIN_FRAC:.0%} of the "
          f"DETECTION's own area inside the annotation")
    print(f"secondary IoU threshold : {IOU_THRESH}")
    print("")

    med_w = (gt["xmax"] - gt["xmin"]).median() * GSD_M
    med_h = (gt["ymax"] - gt["ymin"]).median() * GSD_M
    med_d = (primary["xmax"] - primary["xmin"]).median() * GSD_M
    print(f"median annotation       : {med_w:.2f} x {med_h:.2f} m")
    print(f"median detection width  : {med_d:.2f} m")
    print("")

    sets = []
    for name, labels, allow_clipped in SCORING_SETS:
        sel = gt["label_name_as_exported"].isin(labels)
        if not allow_clipped:
            sel &= gt["edge_clipped"] == 0
        sets.append((name, gt[sel].reset_index(drop=True)))

    print("scoring sets, nested:")
    for name, sub in sets:
        print(f"  {name:<42s} n = {len(sub)}")
    print("")

    # =====================================================================
    print("=" * 78)
    print("PRIMARY: DETECTIONS PER ANNOTATED TREE, " + PRIMARY_PHASE)
    print("=" * 78)
    per_tree_primary = None
    for name, sub in sets:
        pt, pd_, C = containment_report(sub, primary, name, PRIMARY_PHASE)
        if per_tree_primary is None:
            per_tree_primary, C_primary, sub_primary = pt, C, sub
    full_histogram(per_tree_primary, f"{PRIMARY_PHASE}, scoring set 1")
    print("")

    print("=" * 78)
    print("PRIMARY: DETECTIONS PER ANNOTATED TREE, CLUSTER UNION")
    print("=" * 78)
    per_tree_union = None
    for name, sub in sets:
        pt, pd_, C = containment_report(sub, union, name, "union")
        if per_tree_union is None:
            per_tree_union = pt
    full_histogram(per_tree_union, "cluster union, scoring set 1")
    print("")

    # =====================================================================
    print("=" * 78)
    print("SECONDARY: ONE TO ONE IoU AT " + str(IOU_THRESH))
    print("=" * 78)
    print("Penalises fragmentation by construction. See the module docstring.")
    print("")
    print(f"  {PRIMARY_PHASE}:")
    for name, sub in sets:
        hungarian_report(sub, primary, name)
    print("")
    print("  cluster union:")
    for name, sub in sets:
        hungarian_report(sub, union, name)
    print("")

    # =====================================================================
    print("=" * 78)
    print("THE SUPPORT QUESTION")
    print("=" * 78)
    print("Does support predict landing inside an annotated tree?")
    print(f"Scoring set 1, n = {len(sub_primary)} annotations.")
    print("")
    a = sub_primary[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    d = primary[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    inside = (contained_matrix(a, d) >= CONTAIN_FRAC).any(axis=0)
    sup = primary["support"].to_numpy()

    bands = [
        ("found by all sixteen", sup == pm.N_PHASES),
        ("found by some, not all", (sup > 1) & (sup < pm.N_PHASES)),
        ("found once", sup == 1),
    ]
    print(f"  {'band':<26s} {'n':>5s} {'inside':>7s} {'share':>8s}")
    for label, m in bands:
        n = int(m.sum())
        k = int(inside[m].sum())
        print(f"  {label:<26s} {n:5d} {k:7d} {k / n:8.1%}" if n else
              f"  {label:<26s} {n:5d}       0        n/a")
    print("")
    print("  by exact support level:")
    for L in range(1, pm.N_PHASES + 1):
        m = sup == L
        n = int(m.sum())
        if n == 0:
            continue
        k = int(inside[m].sum())
        print(f"    support {L:>2d} : n {n:4d}   inside {k:4d}   {k / n:6.1%}")
    print("")

    # --- also on the widest scoring set, so the answer is not an artefact
    a5 = sets[-1][1][["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
    inside5 = (contained_matrix(a5, d) >= CONTAIN_FRAC).any(axis=0)
    print(f"  repeated on scoring set 5, all {len(sets[-1][1])} annotations:")
    for label, m in bands:
        n = int(m.sum())
        k = int(inside5[m].sum())
        print(f"    {label:<26s} {n:5d} {k:7d} {k / n:8.1%}")
    print("")

    mrows = metrics_report(sets, primary, PRIMARY_PHASE)
    mrows.to_csv(os.path.join(HERE, "ground_truth", "match_metrics.csv"),
                 index=False)
    irows = support_intervals(sets, primary)
    irows.to_csv(os.path.join(HERE, "ground_truth",
                              "match_support_bands.csv"), index=False)

    # --- outputs ---------------------------------------------------------
    out = sub_primary.copy()
    out["n_detections_contained"] = per_tree_primary
    out.to_csv(os.path.join(HERE, "ground_truth",
                            "match_per_annotation.csv"), index=False)
    dd = primary[["xmin", "ymin", "xmax", "ymax", "score", "support"]].copy()
    dd["inside_any_annotation_set1"] = inside
    dd["inside_any_annotation_set5"] = inside5
    dd.to_csv(os.path.join(HERE, "ground_truth",
                           "match_per_detection.csv"), index=False)
    print("wrote ground_truth/match_per_annotation.csv and "
          "ground_truth/match_per_detection.csv")


if __name__ == "__main__":
    run()
