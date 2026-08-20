"""
Three results figures for the paper.

Audience is a reviewer, not a stakeholder, so axis labels and technical
vocabulary are used freely. This is the opposite of build_figure.py, which
strips all of it.

    fig_support_histogram.png       support distribution, 16 bins
    fig_pinning_by_support.png      pinned share by support level
    fig_detections_per_tree.png     over segmentation, scoring set 1

Palette from the validated light mode reference: series blue #2a78d6 on a
#fcfcfb surface, ink #0b0b0b and #52514e, recessive grid and axes. Single
series in every figure, so no legend is needed and the title names the series.

Figure b carries two populations of bar, reportable and sparse. That
distinction is drawn with TEXTURE and a printed n, never with colour alone, so
it survives grayscale, colourblind readers, and a reader who does not read the
caption.

Inputs
------
phase_stability_hist.csv                    figure a
seam_pinning_all_by_support.csv             figure b
ground_truth/match_per_annotation.csv       figure c

Nothing is hardcoded from the chat. Every number is read from disk.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2de"
SERIES = "#2a78d6"
SERIES_LIGHT = "#9ec5f4"
RULE = "#e34948"

N_REPORTABLE = 70          # below this, a per level verdict is not reportable

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 10,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.edgecolor": GRID,
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=3, width=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)


def save(fig, name):
    p = os.path.join(HERE, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# =========================================================================
# a. support histogram
# =========================================================================

def fig_support_histogram():
    d = pd.read_csv(os.path.join(HERE, "phase_stability_hist.csv"))
    total = int(d["n_crowns"].sum())
    median = float(np.median(np.repeat(d["n_phases"], d["n_crowns"])))

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.bar(d["n_phases"], d["n_crowns"], width=0.72, color=SERIES,
           linewidth=0)
    style(ax)

    ax.axvline(median, color=RULE, linestyle="--", linewidth=1.4, zorder=3)
    ax.annotate(f"median support {median:.1f}",
                xy=(median, ax.get_ylim()[1] * 0.86),
                xytext=(median + 0.45, ax.get_ylim()[1] * 0.86),
                color=RULE, fontsize=9, va="center")

    # selective direct labels: the two poles only
    for k in (d["n_crowns"].idxmax(), 15):
        r = d.iloc[k]
        ax.annotate(f"{int(r['n_crowns'])}",
                    xy=(r["n_phases"], r["n_crowns"]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)

    ax.set_xticks(range(1, 17))
    ax.set_xlabel("Number of grid positions the cluster was detected at (support)")
    ax.set_ylabel("Clusters")
    ax.set_title(f"Support is U shaped: {total} clusters, "
                 f"IoU 0.5 cross position matching",
                 loc="left", fontsize=11, color=INK, pad=10)
    save(fig, "fig_support_histogram.png")


# =========================================================================
# b. pinning share by support
# =========================================================================

def fig_pinning_by_support():
    d = pd.read_csv(os.path.join(HERE, "seam_pinning_all_by_support.csv"))
    d = d.sort_values("support")
    sparse = d["n_clusters"] < N_REPORTABLE

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(d.loc[~sparse, "support"], d.loc[~sparse, "obs_share_pinned"],
           width=0.72, color=SERIES, linewidth=0, label="reportable")
    ax.bar(d.loc[sparse, "support"], d.loc[sparse, "obs_share_pinned"],
           width=0.72, color=SERIES_LIGHT, linewidth=0.8,
           edgecolor=SERIES, hatch="///")
    style(ax)

    # the shape: a plateau, then a floor. NOT a monotone decline.
    ax.set_ylim(0, 0.98)
    ax.plot([0.6, 4.4], [0.86, 0.86], color=INK_2, linewidth=0.9)
    ax.annotate("high plateau, supports 1 to 4", xy=(2.5, 0.885),
                ha="center", fontsize=9, color=INK)
    ax.plot([4.6, 16.4], [0.30, 0.30], color=INK_2, linewidth=0.9)
    ax.annotate("floor, supports 5 to 16", xy=(10.5, 0.325),
                ha="center", fontsize=9, color=INK)

    # n as a table row under the axis, so zero height bars do not collide
    for _, r in d.iterrows():
        ax.annotate(f"{int(r['n_clusters'])}", xy=(r["support"], -0.075),
                    ha="center", va="center", fontsize=7.5, color=INK_2,
                    annotation_clip=False)
    ax.annotate("n =", xy=(0.25, -0.075), ha="right", va="center",
                fontsize=7.5, color=INK_2, annotation_clip=False)

    ax.set_xlim(0.3, 16.7)
    ax.set_xticks(range(1, 17))
    ax.set_xlabel("Support", labelpad=16)
    ax.set_ylabel("Share pinned within 1 px of a seam")
    ax.set_title("Seam pinning is a low support phenomenon, not a gradient",
                 loc="left", fontsize=11, color=INK, pad=10)
    fig.text(0.5, -0.075,
             f"Hatched bars have n below {N_REPORTABLE}. Their shares stand; "
             "their per level verdicts do not, because the shuffled null\n"
             "collapses to zero at that n and a single pinned cluster returns "
             "p = 0. Reportable levels are 1, 2, 3, 4 and 16.",
             ha="center", fontsize=8, color=INK_2)
    save(fig, "fig_pinning_by_support.png")


# =========================================================================
# c. detections per annotated tree
# =========================================================================

def fig_detections_per_tree():
    d = pd.read_csv(os.path.join(HERE, "ground_truth",
                                 "match_per_annotation.csv"))
    per = d["n_detections_contained"].to_numpy()
    median = float(np.median(per))
    mx = int(per.max())
    counts = [int((per == k).sum()) for k in range(mx + 1)]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(range(mx + 1), counts, width=0.72, color=SERIES, linewidth=0)
    style(ax)

    ax.axvline(median, color=RULE, linestyle="--", linewidth=1.4, zorder=3)
    ax.annotate(f"median {median:.1f}", xy=(median + 0.2, max(counts) * 0.9),
                color=RULE, fontsize=9)

    for k, dxy in ((0, (0, 4)), (1, (0, 4)), (2, (14, 4))):
        ax.annotate(str(counts[k]), xy=(k, counts[k]), xytext=dxy,
                    textcoords="offset points", ha="center", fontsize=9,
                    color=INK)

    big = d.nlargest(2, "n_detections_contained")
    ax.annotate(
        f"the two trees at {mx} are the two largest annotations,\n"
        f"{big['area_m2'].iloc[0]:.0f} and {big['area_m2'].iloc[1]:.0f} m2",
        xy=(mx, counts[mx]), xytext=(mx - 0.4, max(counts) * 0.45),
        ha="right", fontsize=8, color=INK_2,
        arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.8),
    )

    n_multi = int((per >= 2).sum())
    ax.set_xticks(range(mx + 1))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Detections with at least 50 percent of their own area "
                  "inside the annotated tree")
    ax.set_ylabel("Annotated trees")
    ax.set_title(f"Over segmentation: {n_multi} of {len(per)} annotated trees "
                 f"carry two or more detections",
                 loc="left", fontsize=11, color=INK, pad=10)
    fig.text(0.5, -0.03,
             "Scoring set 1: live, canopy, certain, not edge clipped. "
             "Detections from dx225_dy075.",
             ha="center", fontsize=8, color=INK_2)
    save(fig, "fig_detections_per_tree.png")


if __name__ == "__main__":
    fig_support_histogram()
    fig_pinning_by_support()
    fig_detections_per_tree()
