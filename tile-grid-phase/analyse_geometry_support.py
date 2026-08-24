"""
Conditional geometry test: does 2D footprint shape differ between stable and
unstable detections AFTER seam artefacts are removed?

The unconditional association is already recorded: rank correlation of aspect
with support -0.4597 (analyse_support.py). That is confounded with seam
pinning, because 136 of the 196 singletons have a box edge sitting on a tile
boundary. This script asks whether the shape difference survives excluding
those.

Design, fixed before any number here was seen
---------------------------------------------
unit          the cross phase cluster. denominator 710.
groups        support 1 unstable, support 16 stable.
              support 2 to 15 is context only and is EXCLUDED from every test.
seam control  by EXCLUSION, not regression. pinned = median edge gap <= 1.0 px.
              that threshold reproduces the documented 136 and 302 counts.
Set A         primary.    unpinned support 1 + all support 16.
Set B         robustness. all support 1 + all support 16.
endpoints     PRIMARY   aspect ratio (largest unconditional correlation)
              SECONDARY log box area, natural log
              designated in advance, not reordered after seeing results.
tests         Mann Whitney U, two sided, on each endpoint in Set A.
effect size   rank biserial r_rb, bootstrap 95 pct CI, 10000 resamples, seed 42.
multiplicity  Holm across the two endpoints.

Sign convention for r_rb
------------------------
Group 1 is support 1, group 2 is support 16, and

    AUC  = U1 / (n1 * n2)          probability a support 1 cluster ranks above
                                   a support 16 cluster
    r_rb = 2 * AUC - 1

so r_rb POSITIVE means support 1 scores HIGHER than support 16. The
unconditional direction is that support 16 has the LOWER aspect, so the
direction that matches is r_rb > 0 on aspect.

Inputs, both already on disk, both 710 rows, joined on cluster_id
-----------------------------------------------------------------
crown_geometry.csv             med_aspect, med_area_m2, per cluster MEDIAN over
                               the cluster's member observations
                               (analyse_support.py lines 205-214).
                               aspect = long side / short side, >= 1 by
                               construction (line 198). area is a bounding box
                               area in m2, w * h, at GSD 7.78 cm.
seam_pinning_all_clusters.csv  cluster_gap_px, per cluster MEDIAN observation
                               edge gap in EXPERIMENT px, and the pinned flag
                               (check_seam_pinning_all.py lines 306-313).

Units are mixed by inheritance: the endpoints are metric, the exclusion
variable is pixel. 1.0 px = 7.78 cm. Not changed here.

Both inputs are already restricted to the 25 px inset core, because the core
filter runs in phase_sweep.py lines 326-334 BEFORE clustering. No further
restriction is applied here.

Outputs
-------
geometry_by_support_tier.csv   one row per tier per view: n, median and IQR of
                               aspect and of log area.
figure_geometry_support.png    two panels, shared y.

Does NOT touch geometry_by_support.csv, which is an existing artefact of
analyse_support.py with a different schema (16 rows, one per support level).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- locked constants ----------------------------------------------------
PIN_TOL = 1.0          # px. primary exclusion threshold.
PIN_TOL_WIDE = 22.29   # px. RB3 sensitivity only. lower bound of null 95 band.
N_BOOT = 10000
SEED = 42
ALPHA = 0.05
RB_IN = 0.30           # |r_rb| needed to RULE IN
CI_OUT = 0.20          # CI must sit inside +/- this to RULE OUT
CALIPER_SD = 0.25      # RB2 caliper, in SD of log area

EXPECT = {
    "n_clusters": 710,
    "set_a_s1": 60,
    "set_a_s16": 115,
    "set_b": 311,
    "pinned_all": 302,
    "pinned_s1": 136,
}


# =========================================================================
# load and join
# =========================================================================

def load():
    geo = pd.read_csv(os.path.join(OUT_DIR, "crown_geometry.csv"))
    pin = pd.read_csv(os.path.join(OUT_DIR, "seam_pinning_all_clusters.csv"))

    if len(geo) != EXPECT["n_clusters"]:
        sys.exit(f"crown_geometry.csv has {len(geo)} rows, expected 710")
    if len(pin) != EXPECT["n_clusters"]:
        sys.exit(f"seam_pinning_all_clusters.csv has {len(pin)} rows, "
                 f"expected 710")
    if set(geo["cluster_id"]) != set(pin["cluster_id"]):
        sys.exit("cluster_id sets differ between the two inputs")

    df = geo.merge(pin, on="cluster_id", suffixes=("", "_pin"),
                   validate="one_to_one")
    if not (df["support"] == df["support_pin"]).all():
        sys.exit("support disagrees between crown_geometry and "
                 "seam_pinning_all_clusters, the join is wrong")
    if len(df) != EXPECT["n_clusters"]:
        sys.exit(f"join produced {len(df)} rows, expected 710")

    # recompute the flag rather than trusting the stored one
    df["pinned_calc"] = df["cluster_gap_px"] <= PIN_TOL
    if not (df["pinned_calc"] == df["pinned"]).all():
        sys.exit("stored pinned flag does not reproduce from "
                 "cluster_gap_px <= 1.0")

    df["log_area"] = np.log(df["med_area_m2"])
    df["aspect"] = df["med_aspect"]

    if not (df["aspect"] >= 1.0).all():
        sys.exit("aspect below 1.0 found, definition assumption broken")
    if not np.isfinite(df["log_area"]).all():
        sys.exit("non finite log area")

    # reconciliation, hard stop on any mismatch
    n_pin_all = int(df["pinned_calc"].sum())
    n_pin_s1 = int(df.loc[df["support"] == 1, "pinned_calc"].sum())
    n_pin_s16 = int(df.loc[df["support"] == 16, "pinned_calc"].sum())
    if (n_pin_all, n_pin_s1, n_pin_s16) != (EXPECT["pinned_all"],
                                            EXPECT["pinned_s1"], 0):
        sys.exit(f"pinned counts {n_pin_all}/{n_pin_s1}/{n_pin_s16} do not "
                 f"match the documented 302/136/0")
    return df


# =========================================================================
# effect size
# =========================================================================

def mwu(x, y):
    """Mann Whitney U two sided. x is group 1 (support 1), y is group 2.

    Returns (U1, p, auc, r_rb). r_rb > 0 means x ranks above y.
    """
    u1, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    auc = u1 / (len(x) * len(y))
    return float(u1), float(p), float(auc), float(2.0 * auc - 1.0)


def rrb_only(x, y):
    u1 = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2.0 * (u1 / (len(x) * len(y))) - 1.0


def boot_ci(x, y, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """Stratified bootstrap percentile CI on r_rb. Resamples within group."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    out = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        xb = x[rng.integers(0, n1, n1)]
        yb = y[rng.integers(0, n2, n2)]
        out[b] = rrb_only(xb, yb)
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def holm(pvals):
    """Holm step down. Returns adjusted p in the input order."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


# =========================================================================
# descriptive table
# =========================================================================

def tier_of(support):
    if support == 1:
        return "1"
    if support == 16:
        return "16"
    return "2-15"


def describe(df):
    """One row per tier per view. n, median and IQR of aspect and log area."""
    rows = []
    views = [
        ("all_clusters", df),
        ("seam_removed", df[~df["pinned_calc"]]),
    ]
    for view_name, sub in views:
        for tier in ["1", "2-15", "16"]:
            s = sub[sub["tier"] == tier]
            row = {"view": view_name, "tier": tier, "n": len(s)}
            for col, lab in [("aspect", "aspect"), ("log_area", "log_area")]:
                v = s[col].to_numpy(dtype=float)
                if len(v) == 0:
                    row[f"{lab}_median"] = np.nan
                    row[f"{lab}_q1"] = np.nan
                    row[f"{lab}_q3"] = np.nan
                    row[f"{lab}_iqr"] = np.nan
                    continue
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                row[f"{lab}_median"] = round(float(med), 4)
                row[f"{lab}_q1"] = round(float(q1), 4)
                row[f"{lab}_q3"] = round(float(q3), 4)
                row[f"{lab}_iqr"] = round(float(q3 - q1), 4)
            rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================
# logistic regression, no statsmodels in this environment
# =========================================================================

def logistic_fit(X, y, names, max_iter=200, tol=1e-10):
    """UNPENALISED logistic regression by IRLS, with Wald standard errors.

    Written out rather than called from sklearn on purpose. sklearn 1.8
    deprecated the penalty argument, and its default path applies L2 shrinkage
    at C=1.0, which moves the coefficients substantially on this data: the
    aspect coefficient reads -3.71 penalised against -13.76 unpenalised. A
    robustness check must not depend on an undeclared prior, so the maximum
    likelihood fit is computed here directly and is reproducible without a
    library version pin.

    Predictors are standardised for the solve and the estimates are
    transformed back to raw units, so the reported coefficients are per metre
    squared of log area, per unit of aspect and per pixel of edge gap.
    """
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    Xs = (X - mu) / sd
    Ds = np.column_stack([np.ones(len(Xs)), Xs])

    beta_s = np.zeros(Ds.shape[1])
    for _ in range(max_iter):
        eta = Ds @ beta_s
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(p * (1.0 - p), 1e-10, None)
        z = eta + (y - p) / W
        info = Ds.T @ (Ds * W[:, None])
        step = np.linalg.solve(info, Ds.T @ (W * z)) - beta_s
        beta_s = beta_s + step
        if np.max(np.abs(step)) < tol:
            break

    p = 1.0 / (1.0 + np.exp(-(Ds @ beta_s)))
    W = np.clip(p * (1.0 - p), 1e-10, None)
    info = Ds.T @ (Ds * W[:, None])
    cov_s = np.linalg.pinv(info)

    # back transform to raw units
    k = X.shape[1]
    T = np.zeros((k + 1, k + 1))
    T[0, 0] = 1.0
    for j in range(k):
        T[0, j + 1] = -mu[j] / sd[j]
        T[j + 1, j + 1] = 1.0 / sd[j]
    beta = T @ beta_s
    cov = T @ cov_s @ T.T
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pv = 2.0 * stats.norm.sf(np.abs(z))

    return pd.DataFrame({
        "term": ["intercept"] + list(names),
        "coef": np.round(beta, 4),
        "se": np.round(se, 4),
        "z": np.round(z, 3),
        "p": pv,
        "odds_ratio": np.round(np.exp(beta), 4),
    })


# =========================================================================
# RB2 matching
# =========================================================================

def size_match(treat_vals, ctrl_vals, caliper, seed=SEED):
    """Greedy nearest neighbour on one covariate, without replacement.

    treat_vals are the support 1 unpinned clusters, ctrl_vals the support 16.
    Returns (treat_idx, ctrl_idx) as positions into the input arrays.
    Processing order is randomised with the given seed so the greedy result
    does not depend on file order.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(treat_vals))
    used = np.zeros(len(ctrl_vals), dtype=bool)
    ti, ci = [], []
    for t in order:
        d = np.abs(ctrl_vals - treat_vals[t])
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= caliper:
            used[j] = True
            ti.append(int(t))
            ci.append(j)
    return np.array(ti, dtype=int), np.array(ci, dtype=int)


# =========================================================================
# main
# =========================================================================

def run():
    pd.set_option("display.width", 200)
    df = load()
    df["tier"] = df["support"].map(tier_of)

    print("=" * 78)
    print("CONDITIONAL GEOMETRY TEST, SHAPE BY SUPPORT WITH SEAM ARTEFACTS")
    print("REMOVED")
    print("=" * 78)
    print("clusters loaded            :", len(df))
    print("pinning threshold, primary :", PIN_TOL, "px")
    print("pinned, all clusters       :", int(df['pinned_calc'].sum()))
    print("bootstrap resamples / seed :", N_BOOT, "/", SEED)
    print("")
    print("SIGN CONVENTION. group 1 is support 1, group 2 is support 16.")
    print("  AUC  = P(a support 1 cluster ranks above a support 16 cluster)")
    print("  r_rb = 2 * AUC - 1,  so r_rb > 0 means support 1 scores HIGHER.")
    print("  unconditional direction: support 16 has the LOWER aspect,")
    print("  therefore the direction that MATCHES is r_rb > 0 on aspect.")
    print("")

    # --- analysis sets ----------------------------------------------------
    s1_unpinned = df[(df["support"] == 1) & (~df["pinned_calc"])]
    s16 = df[df["support"] == 16]
    set_a = pd.concat([s1_unpinned, s16], ignore_index=True)
    set_b = df[df["support"].isin([1, 16])]

    print("-" * 78)
    print("ANALYSIS SETS")
    print("-" * 78)
    print(f"Set A  unpinned support 1 : {len(s1_unpinned):4d}   "
          f"expected {EXPECT['set_a_s1']}   "
          f"{'ok' if len(s1_unpinned) == EXPECT['set_a_s1'] else 'MISMATCH'}")
    print(f"Set A  all support 16     : {len(s16):4d}   "
          f"expected {EXPECT['set_a_s16']}   "
          f"{'ok' if len(s16) == EXPECT['set_a_s16'] else 'MISMATCH'}")
    print(f"Set A  total              : {len(set_a):4d}")
    print(f"Set B  support 1 and 16   : {len(set_b):4d}   "
          f"expected {EXPECT['set_b']}   "
          f"{'ok' if len(set_b) == EXPECT['set_b'] else 'MISMATCH'}")
    if (len(s1_unpinned) != EXPECT["set_a_s1"]
            or len(s16) != EXPECT["set_a_s16"]
            or len(set_b) != EXPECT["set_b"]):
        sys.exit("analysis set sizes do not match the design. stopping.")
    print("")

    # --- descriptive table ------------------------------------------------
    tier_tab = describe(df)
    print("-" * 78)
    print("GEOMETRY BY TIER AND VIEW  -> geometry_by_support_tier.csv")
    print("-" * 78)
    print(tier_tab.to_string(index=False))
    print("")
    tier_tab.to_csv(os.path.join(OUT_DIR, "geometry_by_support_tier.csv"),
                    index=False)

    # --- PRIMARY ----------------------------------------------------------
    print("=" * 78)
    print("PRIMARY TESTS, SET A, MANN WHITNEY U TWO SIDED")
    print("=" * 78)

    endpoints = [("aspect", "aspect ratio", "PRIMARY"),
                 ("log_area", "log box area", "SECONDARY")]
    res = []
    for col, label, rank in endpoints:
        x = s1_unpinned[col].to_numpy(dtype=float)
        y = s16[col].to_numpy(dtype=float)
        u1, p, auc, rrb = mwu(x, y)
        lo, hi = boot_ci(x, y)
        res.append({"col": col, "label": label, "rank": rank,
                    "n1": len(x), "n2": len(y), "U": u1, "p": p,
                    "auc": auc, "r_rb": rrb, "ci_lo": lo, "ci_hi": hi,
                    "med1": float(np.median(x)), "med2": float(np.median(y))})

    adj = holm([r["p"] for r in res])
    for r, a in zip(res, adj):
        r["p_holm"] = float(a)

    for r in res:
        print(f"{r['rank']}  {r['label']}")
        print(f"  n support 1 unpinned : {r['n1']}")
        print(f"  n support 16         : {r['n2']}")
        print(f"  median support 1     : {r['med1']:.4f}")
        print(f"  median support 16    : {r['med2']:.4f}")
        print(f"  U                    : {r['U']:.1f}")
        print(f"  p raw                : {r['p']:.6g}")
        print(f"  p Holm adjusted      : {r['p_holm']:.6g}")
        print(f"  r_rb                 : {r['r_rb']:+.4f}")
        print(f"  r_rb 95 pct CI       : [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
        print(f"  AUC                  : {r['auc']:.4f}")
        print("")

    # --- decision rule, mechanical ---------------------------------------
    a = res[0]
    assert a["col"] == "aspect", "primary endpoint reordered, abort"

    # RB2 is computed first because the RULE IN branch depends on it.
    rb2 = run_rb2(s1_unpinned, s16, quiet=True)

    print("=" * 78)
    print("DECISION RULE, APPLIED MECHANICALLY ON THE PRIMARY ENDPOINT")
    print("=" * 78)

    cond_p = a["p_holm"] < ALPHA
    cond_mag = abs(a["r_rb"]) >= RB_IN
    cond_sign = a["r_rb"] > 0          # support 16 lower aspect
    cond_match = rb2["survives"]
    ci_in_out = (a["ci_lo"] >= -CI_OUT) and (a["ci_hi"] <= CI_OUT)
    ci_spans_rb = (a["ci_lo"] <= -RB_IN) or (a["ci_hi"] >= RB_IN)

    print(f"  Holm adjusted p < 0.05            : {cond_p}"
          f"   (p_holm = {a['p_holm']:.6g})")
    print(f"  |r_rb| >= 0.30                    : {cond_mag}"
          f"   (|r_rb| = {abs(a['r_rb']):.4f})")
    print(f"  sign matches, support 16 lower    : {cond_sign}"
          f"   (r_rb = {a['r_rb']:+.4f})")
    print(f"  survives the size matched check   : {cond_match}"
          f"   (matched r_rb = {rb2['r_rb']:+.4f}, "
          f"n {rb2['n1']} vs {rb2['n2']})")
    print(f"  CI inside +/- 0.20                : {ci_in_out}"
          f"   ([{a['ci_lo']:+.4f}, {a['ci_hi']:+.4f}])")
    print(f"  CI still spans +/- 0.30           : {ci_spans_rb}")
    print("")
    print("  NOTE. The brief did not define 'survives the size matched")
    print("  check' numerically. Operationalised here, and fixed before the")
    print("  numbers were read, as: same sign as the primary AND matched")
    print("  |r_rb| >= 0.30. The matched p value is printed under RB2 so the")
    print("  alternative reading is available.")
    print("")

    if cond_p and cond_mag and cond_sign and cond_match:
        verdict = "RULED IN"
    elif (not cond_p) and ci_in_out:
        verdict = "RULED OUT"
    elif (not cond_p) and ci_spans_rb:
        verdict = "INCONCLUSIVE (UNDERPOWERED)"
    else:
        verdict = "INCONCLUSIVE"

    print("  " + "-" * 60)
    print(f"  VERDICT: {verdict}")
    print("  " + "-" * 60)
    print("")

    # is the verdict sensitive to how the matched check was operationalised?
    alt_match = bool(np.isfinite(rb2["p"]) and rb2["p"] < ALPHA
                     and rb2["r_rb"] > 0)
    alt_in = cond_p and cond_mag and cond_sign and alt_match
    alt_verdict = "RULED IN" if alt_in else (
        "RULED OUT" if ((not cond_p) and ci_in_out) else (
            "INCONCLUSIVE (UNDERPOWERED)" if ((not cond_p) and ci_spans_rb)
            else "INCONCLUSIVE"))
    print(f"  Under the alternative reading of the matched check")
    print(f"  (matched p < 0.05 and same sign, which is "
          f"{alt_match}), the verdict")
    print(f"  is {alt_verdict}. Verdict sensitive to that choice: "
          f"{alt_verdict != verdict}")
    print("")

    # --- power statement, printed regardless ------------------------------
    print("-" * 78)
    print("POWER")
    print("-" * 78)
    print("At 60 versus 115, alpha 0.05 two sided, power is roughly 80 percent")
    print("to detect r_rb of about 0.30 (AUC 0.65). Effects smaller than that")
    print("cannot be ruled out.")
    print("")

    # --- robustness -------------------------------------------------------
    rb1 = run_rb1(set_b)
    run_rb2(s1_unpinned, s16, quiet=False, pre=rb2)
    rb3 = run_rb3(df)

    build_figure(df, a)

    return {"df": df, "res": res, "verdict": verdict, "tier": tier_tab,
            "rb1": rb1, "rb2": rb2, "rb3": rb3,
            "s1_unpinned": s1_unpinned, "s16": s16}


# =========================================================================
# RB1  logistic regression on Set B
# =========================================================================

def run_rb1(set_b):
    print("=" * 78)
    print("RB1  LOGISTIC REGRESSION ON SET B")
    print("=" * 78)
    print("outcome    : support 16 = 1, support 1 = 0")
    print("predictors : log area (m2), aspect, median edge gap (px)")
    print("n          :", len(set_b))
    print("")

    names = ["log_area", "aspect", "cluster_gap_px"]
    X = set_b[names].to_numpy(dtype=float)
    y = (set_b["support"] == 16).to_numpy(dtype=int)
    print("outcome counts: support 16 =", int(y.sum()),
          " support 1 =", int((1 - y).sum()))
    print("")

    tab = logistic_fit(X, y, names)
    tab["p"] = tab["p"].map(lambda v: f"{v:.4g}")
    print(tab.to_string(index=False))
    print("")

    # separation diagnostic. an enormous coefficient with an enormous SE is
    # the signature of quasi complete separation, and means the point
    # estimate is not interpretable however small its p value looks.
    big = tab[(tab["term"] != "intercept") & (tab["coef"].abs() > 10.0)]
    if len(big):
        print("  WARNING. quasi complete separation suspected on: "
              + ", ".join(big["term"]))
        print("  A coefficient this large means the two groups are nearly")
        print("  perfectly split on that predictor, so the magnitude and the")
        print("  odds ratio are artefacts of the fit, not estimates. Read the")
        print("  SIGN only, and do not quote the odds ratio.")
        for term in big["term"]:
            a1 = set_b.loc[set_b["support"] == 1, term]
            a16 = set_b.loc[set_b["support"] == 16, term]
            print(f"    {term}: support 1 spans "
                  f"{a1.min():.4f} to {a1.max():.4f}, support 16 spans "
                  f"{a16.min():.4f} to {a16.max():.4f}")
        print("  The separation is real structure in the data, not a")
        print("  numerical fault.")
        print("")

    rho, prho = stats.spearmanr(set_b["log_area"], set_b["cluster_gap_px"])
    print(f"Spearman, log area vs median edge gap, Set B: "
          f"rho = {rho:+.4f}, p = {prho:.4g}")
    if abs(rho) > 0.6:
        print("")
        print("  WARNING. |rho| exceeds 0.6. The predictors are strongly")
        print("  collinear, the coefficients are unstable, and they should")
        print("  NOT be leaned on. Read this model as a check that the")
        print("  exclusion design was the right call, not as an estimate.")
    else:
        print("  |rho| at or below 0.6, no collinearity warning triggered.")
    print("")
    return {"table": tab, "rho": float(rho)}


# =========================================================================
# RB2  size matched subsample within Set A
# =========================================================================

def run_rb2(s1_unpinned, s16, quiet=False, pre=None):
    if pre is not None:
        out = pre
    else:
        t = s1_unpinned["log_area"].to_numpy(dtype=float)
        c = s16["log_area"].to_numpy(dtype=float)
        pooled_sd = float(np.std(np.concatenate([t, c]), ddof=1))
        caliper = CALIPER_SD * pooled_sd
        ti, ci = size_match(t, c, caliper, seed=SEED)

        xa = s1_unpinned["aspect"].to_numpy(dtype=float)[ti]
        ya = s16["aspect"].to_numpy(dtype=float)[ci]
        if len(ti) >= 3 and len(ci) >= 3:
            u1, p, auc, rrb = mwu(xa, ya)
            lo, hi = boot_ci(xa, ya)
        else:
            u1 = p = auc = rrb = lo = hi = float("nan")

        out = {"n1": len(ti), "n2": len(ci), "caliper": caliper,
               "pooled_sd": pooled_sd, "U": u1, "p": p, "auc": auc,
               "r_rb": rrb, "ci_lo": lo, "ci_hi": hi,
               "med1": float(np.median(xa)) if len(ti) else float("nan"),
               "med2": float(np.median(ya)) if len(ci) else float("nan"),
               "la1": float(np.median(t[ti])) if len(ti) else float("nan"),
               "la2": float(np.median(c[ci])) if len(ci) else float("nan"),
               "unmatched": len(t) - len(ti)}
        out["survives"] = bool(np.isfinite(rrb) and rrb > 0
                               and abs(rrb) >= RB_IN)
    if quiet:
        return out

    print("=" * 78)
    print("RB2  SIZE MATCHED SUBSAMPLE WITHIN SET A")
    print("=" * 78)
    print("matched on log area, greedy nearest neighbour, WITHOUT")
    print("replacement, processing order randomised with seed", SEED)
    print(f"pooled SD of log area in Set A : {out['pooled_sd']:.4f}")
    print(f"caliper, {CALIPER_SD} SD             : {out['caliper']:.4f} "
          f"log units")
    print("")
    print(f"matched n, support 1 unpinned  : {out['n1']}")
    print(f"matched n, support 16          : {out['n2']}")
    print(f"support 1 clusters left unmatched : {out['unmatched']}")
    print(f"median log area, matched s1    : {out['la1']:.4f}")
    print(f"median log area, matched s16   : {out['la2']:.4f}")
    print("")
    print("aspect only, in the matched set:")
    print(f"  median support 1  : {out['med1']:.4f}")
    print(f"  median support 16 : {out['med2']:.4f}")
    print(f"  U                 : {out['U']:.1f}")
    print(f"  p raw             : {out['p']:.6g}   "
          f"(NOT Holm adjusted, this is a robustness check)")
    print(f"  r_rb              : {out['r_rb']:+.4f}")
    print(f"  r_rb 95 pct CI    : [{out['ci_lo']:+.4f}, {out['ci_hi']:+.4f}]")
    print(f"  AUC               : {out['auc']:.4f}")
    print("")
    return out


# =========================================================================
# RB3  sensitivity to the pinning threshold
# =========================================================================

def run_rb3(df):
    print("=" * 78)
    print("RB3  SENSITIVITY TO THE PINNING THRESHOLD, 1.0 px -> 22.29 px")
    print("=" * 78)
    print("SENSITIVITY ONLY. This does NOT override the primary.")
    print("")
    print("The brief says 'the exclusion threshold widened', while Set A is")
    print("defined as 'unpinned support 1 + ALL support 16'. Under a 1.0 px")
    print("threshold those two readings agree, because zero support 16")
    print("clusters are pinned. At 22.29 px they diverge, so BOTH are")
    print("reported and neither is presented as the answer.")
    print("")

    wide = df["cluster_gap_px"] <= PIN_TOL_WIDE
    s1w = df[(df["support"] == 1) & (~wide)]
    s16w = df[(df["support"] == 16) & (~wide)]
    s16all = df[df["support"] == 16]

    for label, g1, g2 in [
        ("(a) exclusion applied to BOTH groups", s1w, s16w),
        ("(b) exclusion applied to support 1 only, support 16 kept whole",
         s1w, s16all),
    ]:
        print(label)
        print(f"  n support 1 unpinned at 22.29 px : {len(g1)}")
        print(f"  n support 16                     : {len(g2)}")
        if len(g1) < 3 or len(g2) < 3:
            print("  too few clusters on one side to test. no r_rb reported.")
            print("")
            continue
        x = g1["aspect"].to_numpy(dtype=float)
        y = g2["aspect"].to_numpy(dtype=float)
        u1, p, auc, rrb = mwu(x, y)
        lo, hi = boot_ci(x, y)
        print(f"  median aspect support 1          : {np.median(x):.4f}")
        print(f"  median aspect support 16         : {np.median(y):.4f}")
        print(f"  p raw                            : {p:.6g}")
        print(f"  r_rb                             : {rrb:+.4f}")
        print(f"  r_rb 95 pct CI                   : "
              f"[{lo:+.4f}, {hi:+.4f}]")
        print(f"  AUC                              : {auc:.4f}")
        print("")

    print(f"for reference, support 16 clusters with gap <= 22.29 px: "
          f"{int((s16all['cluster_gap_px'] <= PIN_TOL_WIDE).sum())} of "
          f"{len(s16all)}")
    print("")
    return {"n_s1": len(s1w), "n_s16_both": len(s16w),
            "n_s16_whole": len(s16all)}


# =========================================================================
# PHASE 2  figure
# =========================================================================

# tier 1 and tier 16 carry the comparison and get the two hues. The 2 to 15
# tier is context only and is deliberately grey. Orange against blue is the
# standard deuteranopia and protanopia safe pair.
COL = {"1": "#C1571C", "2-15": "#9AA0A6", "16": "#1F6FB2"}
TIERS = ["1", "2-15", "16"]


def build_figure(df, primary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(SEED)

    views = [
        ("all clusters", df),
        ("seam artefacts removed", df[~df["pinned_calc"]]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 6.2), sharey=True)
    ymax = float(df["aspect"].max())
    top = ymax + 0.55

    for ax, (title, sub) in zip(axes, views):
        data, ns = [], []
        for tier in TIERS:
            v = sub.loc[sub["tier"] == tier, "aspect"].to_numpy(dtype=float)
            data.append(v)
            ns.append(len(v))

        # jittered points BEHIND the boxes
        for i, (tier, v) in enumerate(zip(TIERS, data), start=1):
            if len(v) == 0:
                continue
            jit = rng.uniform(-0.17, 0.17, size=len(v))
            ax.scatter(np.full(len(v), i) + jit, v,
                       s=11, color=COL[tier], alpha=0.22,
                       linewidths=0, zorder=1)

        bp = ax.boxplot(data, positions=[1, 2, 3], widths=0.46,
                        showfliers=False, patch_artist=True, zorder=3)
        for i, tier in enumerate(TIERS):
            bp["boxes"][i].set(facecolor="none", edgecolor=COL[tier],
                               linewidth=1.6)
            bp["medians"][i].set(color=COL[tier], linewidth=2.4)
            for key in ("whiskers", "caps"):
                for art in bp[key][2 * i:2 * i + 2]:
                    art.set(color=COL[tier], linewidth=1.3)

        for i, (tier, n) in enumerate(zip(TIERS, ns), start=1):
            ax.text(i, top, f"n = {n}", ha="center", va="top",
                    fontsize=10.5,
                    color="#3C4043" if tier != "2-15" else "#80868B")

        ax.set_title(title, fontsize=12.5, pad=12, color="#202124")
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["1", "2 to 15", "16"], fontsize=11)
        ax.set_xlabel("support tier, surveys detecting the cluster",
                      fontsize=10.5, color="#3C4043")
        ax.set_xlim(0.45, 3.55)
        ax.set_ylim(0.95, top + 0.16)
        ax.grid(axis="y", color="#E4E6E8", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#C9CCCF")
        ax.tick_params(colors="#5F6368", labelsize=10)

    axes[0].set_ylabel("aspect ratio, long side / short side", fontsize=11,
                       color="#3C4043")

    # effect size annotation, RIGHT PANEL ONLY
    note = (f"support 1 vs support 16\n"
            f"$r_{{rb}}$ = {primary['r_rb']:+.3f}   "
            f"95% CI [{primary['ci_lo']:+.3f}, {primary['ci_hi']:+.3f}]\n"
            f"AUC = {primary['auc']:.3f}   "
            f"Holm adj. p = {primary['p_holm']:.4f}")
    # placed below the n label band so it cannot collide with it
    axes[1].text(0.97, 0.80, note, transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=9.8, color="#202124",
                 linespacing=1.5,
                 bbox=dict(boxstyle="round,pad=0.55", facecolor="#FFFFFF",
                           edgecolor="#D2D5D8", linewidth=0.9))

    fig.suptitle("Detection footprint shape by cross phase support",
                 fontsize=14, y=0.975, color="#202124")
    fig.text(0.5, 0.015,
             "The 2 to 15 tier is shown in grey for context and is excluded "
             "from every test. Pinned = median edge gap <= 1.0 px.",
             ha="center", fontsize=9, color="#80868B")

    fig.tight_layout(rect=[0, 0.035, 1, 0.945])
    path = os.path.join(OUT_DIR, "figure_geometry_support.png")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    print("=" * 78)
    print("FIGURE")
    print("=" * 78)
    print("wrote figure_geometry_support.png")
    print("panel n values, read off the data, not assumed:")
    for title, sub in views:
        ns = [int((sub["tier"] == t).sum()) for t in TIERS]
        print(f"  {title:24s} : {ns[0]} / {ns[1]} / {ns[2]}")
    print("")


if __name__ == "__main__":
    run()
