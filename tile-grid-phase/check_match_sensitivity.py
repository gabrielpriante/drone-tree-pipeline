"""
Match threshold sensitivity.

Reads the per phase box CSVs written by phase_sweep.py and redoes the cross
phase matching only, at MATCH_IOU 0.3, 0.4 and 0.5. No inference, no model
load, no raster read.

The question it answers
----------------------
The sweep at MATCH_IOU 0.5 reported 710 distinct crowns, 115 found in all 16
phases, and 196 found in exactly one. If a crown shifts slightly between
phases and drops below IoU 0.5, one real crown splits into several one phase
clusters, inflating both the 710 and the 196. Loosening the threshold merges
those splits back together.

How to read the result
----------------------
    singletons fall sharply from 0.5 to 0.3, and distinct crowns fall with
    them
        much of the singleton pile was a clustering artefact. The instability
        finding weakens.

    singletons hold roughly steady across all three
        the singletons are boxes with no counterpart in any other phase at any
        reasonable overlap. They are real detections that appear at one grid
        position and nowhere else. The instability finding holds.

    distinct crowns fall but singletons do not
        splitting was happening among the well supported crowns, not the
        singletons. Read the support histogram to see where the mass moved.

Drift guard
-----------
phase_matching.py holds a duplicated copy of phase_sweep.py's clustering. At
MATCH_IOU 0.5 this script must reproduce the recorded 710 / 115 / 196. If it
does not, the two copies have diverged and nothing below is trustworthy.

Outputs (gitignored)
--------------------
match_sensitivity.csv        one row per threshold
match_sensitivity_hist.csv   support histogram, long format, one row per
                             threshold and support level

Not run yet.
"""

import os
import pandas as pd

import phase_matching as pm

MATCH_IOUS = [0.3, 0.4, 0.5]

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def run():
    print("=== match threshold sensitivity ===")
    print("core inset", pm.CORE_INSET, "px, phases", pm.N_PHASES)
    print("")

    pool = pm.load_pool(OUT_DIR)
    print("")

    summary = []
    hists = []

    for thr in MATCH_IOUS:
        clusters, _ = pm.cluster_across_phases(pool, thr)
        hist = pm.support_histogram(clusters)

        n_all = int(clusters["found_in_all"].sum())
        n_one = int((clusters["n_phases"] == 1).sum())
        n_mid = len(clusters) - n_all - n_one

        summary.append({
            "match_iou": thr,
            "distinct_crowns": len(clusters),
            "found_in_all_16": n_all,
            "found_in_exactly_1": n_one,
            "found_in_2_to_15": n_mid,
            "pct_all_16": round(100.0 * n_all / len(clusters), 2),
            "pct_exactly_1": round(100.0 * n_one / len(clusters), 2),
            "mean_support": round(float(clusters["n_phases"].mean()), 3),
            "median_support": float(clusters["n_phases"].median()),
        })

        h = hist.copy()
        h.insert(0, "match_iou", thr)
        hists.append(h)

        print(f"--- MATCH_IOU {thr} ---")
        print("distinct crowns    :", len(clusters))
        print("found in all 16    :", n_all,
              f"({100.0 * n_all / len(clusters):.1f}%)")
        print("found in exactly 1 :", n_one,
              f"({100.0 * n_one / len(clusters):.1f}%)")
        print("found in 2 to 15   :", n_mid,
              f"({100.0 * n_mid / len(clusters):.1f}%)")
        print("")
        print("support histogram:")
        print(hist.to_string(index=False))
        print("")

        # --- drift guard --------------------------------------------------
        if thr == pm.RECORDED["match_iou"]:
            r = pm.RECORDED
            ok = (
                len(clusters) == r["distinct_crowns"]
                and n_all == r["found_in_all"]
                and n_one == r["found_in_one"]
            )
            print("drift guard at MATCH_IOU", thr, ":",
                  "PASS, reproduces the recorded sweep" if ok else "FAIL")
            if not ok:
                print("  recorded :", r["distinct_crowns"], "crowns,",
                      r["found_in_all"], "in all,",
                      r["found_in_one"], "in one")
                print("  here     :", len(clusters), "crowns,",
                      n_all, "in all,", n_one, "in one")
                print("  phase_matching.py has drifted from phase_sweep.py,")
                print("  or the phase CSVs are not the ones that produced the")
                print("  recorded numbers. Resolve before reading the table.")
            print("")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUT_DIR, "match_sensitivity.csv"),
                      index=False)
    pd.concat(hists, ignore_index=True).to_csv(
        os.path.join(OUT_DIR, "match_sensitivity_hist.csv"), index=False
    )

    print("=== side by side ===")
    print(summary_df.to_string(index=False))

    # --- the artefact question ------------------------------------------
    loose = summary_df.iloc[0]
    tight = summary_df.iloc[-1]
    d_crowns = tight["distinct_crowns"] - loose["distinct_crowns"]
    d_single = tight["found_in_exactly_1"] - loose["found_in_exactly_1"]

    print("")
    print("=== reading it ===")
    print(f"going from IoU {tight['match_iou']} to {loose['match_iou']}:")
    print(f"  distinct crowns  {tight['distinct_crowns']} -> "
          f"{loose['distinct_crowns']}  (drop of {d_crowns})")
    print(f"  singletons       {tight['found_in_exactly_1']} -> "
          f"{loose['found_in_exactly_1']}  (drop of {d_single})")
    if d_crowns > 0:
        print(f"  singletons account for "
              f"{100.0 * d_single / d_crowns:.1f}% of the drop")
    print("")
    print("A large singleton drop means much of the pile was one crown split")
    print("across phases by a strict threshold. A small one means the")
    print("singletons are real single phase detections.")


if __name__ == "__main__":
    run()
