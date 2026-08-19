"""
Load the ground truth annotation export and convert it to the protocol schema.

Conversion only. No matching, no scoring, no comparison against detections.

Input
-----
`ground_truth/labels_myprojectname_20260819055558.csv`, the export exactly as
it came out of the annotation tool, 110 boxes on `core_clean.png`. It
supersedes an earlier 102 box export and is a clean superset: all 102 carried
over unchanged, 8 added, none removed. The earlier export is not used.

    label_name,bbox_x,bbox_y,bbox_width,bbox_height,image_name,image_width,image_height

`bbox_x, bbox_y` is the TOP LEFT corner in CHIP pixels.

Output
------
`ground_truth/annotations_raw.csv` in the schema from section 6 of
ANNOTATION_PROTOCOL.md, plus `width_m` and `height_m`.

Coordinate frame
----------------
Per protocol section 1.2, chip pixel (0, 0) is window pixel (25, 25), because
`core_clean.png` was cropped by `CORE_INSET` = 25 px. So:

    xmin = bbox_x + 25          xmax = xmin + bbox_width
    ymin = bbox_y + 25          ymax = ymin + bbox_height

Window coordinates are the frame used by `phase_boxes_*.csv` and
`phase_stability.csv`. The offset direction is ASSERTED below against a known
box and against the core bounds, not trusted. A silent sign flip here would
look like a systematic 50 px spatial bias in every future match and would be
very hard to diagnose after the fact.

KNOWN LIMITATION OF THE EXPORT, not of the protocol
---------------------------------------------------
The tool exported `label_name` as a single mutually exclusive class taking one
of tree, understory, uncertain, snag. The protocol treats class, layer and
confidence as three independent fields, and the export cannot represent that.

Consequences, which must be carried into any later analysis:

  - A box labelled `uncertain` carries NO information about whether it was
    canopy or understory. It is recorded here as layer = canopy because that
    is the default, not because it was observed to be canopy.
  - A box labelled `understory` carries NO information about whether the
    annotator was confident. It is recorded as confidence = certain by the
    same default.
  - A snag that was also uncertain, or understory, cannot be represented at
    all. Only one label survived.

So `uncertain` is a floor on annotator uncertainty, not a measurement of it,
and the understory and snag counts are floors in the same way. Reporting
"metrics on certain only" per protocol 6.1 remains possible, but the certain
subset is contaminated with boxes that were merely labelled something else.

This is recorded rather than papered over. Fixing it needs either a tool that
exports independent fields, or a second annotation pass adding the missing
flags.

Not part of this script: matching, precision, recall, or anything comparing
these boxes to detections. No ground truth claim is available until that is
done deliberately and separately.
"""

import os

import pandas as pd

# --- geometry, must match phase_matching.py -----------------------------
CORE_INSET = 25          # px, chip to window offset on both axes
WIN_SIZE = 1000          # px, window side at experiment resolution
CHIP_SIZE = 950          # px, core_clean.png side
GSD_CM = 7.78            # cm per px at experiment resolution

EDGE_TOL = 1.0           # px. an edge this close to the chip boundary is clipped
CHIP_MIN = 0.0
CHIP_MAX = 949.0         # last addressable chip pixel

# --- a box we know the answer for, for the direction assert -------------
# The largest annotation in the export, chip top left (450, 552).
KNOWN_CHIP_XY = (450, 552)
KNOWN_WINDOW_XY = (475, 577)

HERE = os.path.dirname(os.path.abspath(__file__))
GT_DIR = os.path.join(HERE, "ground_truth")
RAW = os.path.join(GT_DIR, "labels_myprojectname_20260819055558.csv")
OUT = os.path.join(GT_DIR, "annotations_raw.csv")

LABELS = {"tree", "understory", "uncertain", "snag"}


def load():
    df = pd.read_csv(RAW)

    # --- the export is what we think it is -----------------------------
    unknown = set(df["label_name"]) - LABELS
    assert not unknown, f"unexpected label_name values: {unknown}"
    assert (df["image_name"] == "core_clean.png").all(), \
        "not every row references core_clean.png"
    assert (df["image_width"] == CHIP_SIZE).all(), "unexpected image_width"
    assert (df["image_height"] == CHIP_SIZE).all(), "unexpected image_height"
    assert (df["bbox_width"] > 0).all() and (df["bbox_height"] > 0).all(), \
        "a box has non positive extent"

    # --- chip to window -------------------------------------------------
    out = pd.DataFrame({
        "tree_id": range(1, len(df) + 1),
        "xmin": df["bbox_x"].astype(float) + CORE_INSET,
        "ymin": df["bbox_y"].astype(float) + CORE_INSET,
    })
    out["xmax"] = out["xmin"] + df["bbox_width"].astype(float).to_numpy()
    out["ymax"] = out["ymin"] + df["bbox_height"].astype(float).to_numpy()

    # --- assert the offset DIRECTION, do not trust it -------------------
    # 1. the known box lands where we say it lands
    hit = df.index[(df["bbox_x"] == KNOWN_CHIP_XY[0])
                   & (df["bbox_y"] == KNOWN_CHIP_XY[1])]
    assert len(hit) == 1, \
        f"expected exactly one box at chip {KNOWN_CHIP_XY}, found {len(hit)}"
    r = out.loc[hit[0]]
    assert (r["xmin"], r["ymin"]) == KNOWN_WINDOW_XY, (
        f"known box maps to ({r['xmin']}, {r['ymin']}), "
        f"expected {KNOWN_WINDOW_XY}. The offset sign is wrong."
    )
    # 2. every converted box lies inside the core region. A flipped sign
    #    would put boxes at negative window coordinates, which cannot exist.
    lo, hi = CORE_INSET, WIN_SIZE - CORE_INSET
    assert out[["xmin", "ymin"]].to_numpy().min() >= lo, (
        "a converted box starts before the core boundary. Offset direction "
        "is inverted."
    )
    assert out[["xmax", "ymax"]].to_numpy().max() <= hi, (
        "a converted box ends past the core boundary. Offset direction or "
        "magnitude is wrong."
    )

    # --- protocol fields -------------------------------------------------
    lab = df["label_name"]
    out["class"] = ["snag" if v == "snag" else "live" for v in lab]
    out["layer"] = ["understory" if v == "understory" else "canopy" for v in lab]
    out["confidence"] = ["uncertain" if v == "uncertain" else "certain"
                         for v in lab]

    # edge_clipped is computed in CHIP coordinates, per the instruction
    cx0 = df["bbox_x"].astype(float)
    cy0 = df["bbox_y"].astype(float)
    cx1 = cx0 + df["bbox_width"].astype(float)
    cy1 = cy0 + df["bbox_height"].astype(float)

    def near(v, edge):
        return (v - edge).abs() <= EDGE_TOL

    clipped = (
        near(cx0, CHIP_MIN) | near(cy0, CHIP_MIN)
        | near(cx1, CHIP_MAX) | near(cy1, CHIP_MAX)
    )
    out["edge_clipped"] = clipped.astype(int).to_numpy()

    out["note"] = ""
    out["label_name_as_exported"] = lab.to_numpy()

    g = GSD_CM / 100.0
    out["width_m"] = (out["xmax"] - out["xmin"]) * g
    out["height_m"] = (out["ymax"] - out["ymin"]) * g
    out["area_m2"] = out["width_m"] * out["height_m"]

    return df, out


def report(df, out):
    print("=" * 66)
    print("CONVERSION")
    print("=" * 66)
    print("rows in                  :", len(df))
    print("rows out                 :", len(out))
    print("offset assert            : PASS, chip", KNOWN_CHIP_XY,
          "maps to window", KNOWN_WINDOW_XY)
    print("core bounds assert       : PASS, all boxes inside",
          f"[{CORE_INSET}, {WIN_SIZE - CORE_INSET}]")
    print("window x range           :",
          out["xmin"].min(), "to", out["xmax"].max())
    print("window y range           :",
          out["ymin"].min(), "to", out["ymax"].max())
    print("")

    print("=" * 66)
    print("COUNTS AFTER CONVERSION")
    print("=" * 66)
    print("as exported, label_name:")
    print(df["label_name"].value_counts().to_string())
    print("")
    for col in ("class", "layer", "confidence"):
        print(f"{col}:")
        print(out[col].value_counts().to_string())
        print("")
    print("edge_clipped:")
    print(out["edge_clipped"].value_counts().to_string())
    print("")

    print("=" * 66)
    print("THREE LARGEST BOXES BY AREA")
    print("=" * 66)
    top = out.nlargest(3, "area_m2")
    for _, r in top.iterrows():
        cxy = (int(r["xmin"] - CORE_INSET), int(r["ymin"] - CORE_INSET))
        print(f"  tree_id {int(r['tree_id']):3d}  "
              f"{r['label_name_as_exported']:<11s} "
              f"{r['width_m']:5.2f} x {r['height_m']:5.2f} m  "
              f"({r['area_m2']:6.1f} m2)")
        print(f"            chip top left {cxy}   "
              f"window xmin {r['xmin']:.0f} ymin {r['ymin']:.0f} "
              f"xmax {r['xmax']:.0f} ymax {r['ymax']:.0f}  "
              f"edge_clipped {int(r['edge_clipped'])}")
    print("")

    print("=" * 66)
    print("RAW COUNT COMPARISON, NOT A MATCHING RESULT")
    print("=" * 66)
    print("110 annotations against 274 detections at dx225_dy075.")
    print("")
    print("This is a comparison of two totals and nothing more. No box has")
    print("been matched to any other box. It does not say the detector over")
    print("counts, and it does not say the annotation under counts. Either")
    print("could be true, both could be true in different places, and the")
    print("over segmentation question is exactly what matching will decide.")
    print("Do not quote a ratio from these two numbers.")


def main():
    df, out = load()
    cols = ["tree_id", "xmin", "ymin", "xmax", "ymax", "class", "layer",
            "edge_clipped", "confidence", "note", "label_name_as_exported",
            "width_m", "height_m", "area_m2"]
    out[cols].to_csv(OUT, index=False)
    report(df, out)
    print("")
    print("wrote", os.path.relpath(OUT, HERE))


if __name__ == "__main__":
    main()
