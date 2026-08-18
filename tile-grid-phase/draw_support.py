"""
Two figures of the core region, crowns coloured by support count.

Figure A, per phase
-------------------
Boxes belonging to one phase only, coloured by the support of the cluster each
landed in. One phase rather than all 16, because 16 overlapping copies of every
crown would be unreadable.

    support 16      blue      found at every grid position
    support 2 to 15 amber     found at some positions, not all
    support 1       magenta   found at exactly one position

Default phase is now dx075_dy075, NOT dx000_dy000. Phase 0 runs 25 tiles per
axis pair against 16 at every other phase, so it is a different tiling regime
and its singletons may not be representative. Set PHASE_TO_DRAW back to
dx000_dy000 to compare the two.

Figure B, pooled singletons
---------------------------
Every singleton from all 16 phases on one base image, with support 16 crowns
drawn underneath for context. Figure A shows only the singletons belonging to
one phase, roughly one sixteenth of the population. This shows all of them.

    support 16      blue      cluster mean box, one per crown
    support 1       magenta   every singleton box, from any phase

What to look for in B: are the singletons spread evenly, or do they pile up in
particular parts of the plot. Piling up next to blue boxes suggests boundary
disputes between neighbouring crowns. Piling up in open areas suggests
something else.

Colours are from the IBM colourblind safe palette.

Reads the orthomosaic to rebuild the imagery. Loads no model, runs no
inference.

Outputs (gitignored)
--------------------
support_<phase>_iou<threshold>.png
support_pooled_singletons_iou<threshold>.png

Not run yet.
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image, ImageDraw, ImageFont

import phase_matching as pm

PATH = r"C:\Users\gabpe\Downloads\000103_ortho-dsm-ptcloud.tif"

# --- scored working window, NATIVE px (settled) --------------------------
WIN_COL_OFF_NATIVE = 4820
WIN_ROW_OFF_NATIVE = 5260
WIN_SIZE_NATIVE = 2000
DOWNSAMPLE = 2

# --- what to draw --------------------------------------------------------
# Four tile phase by default. See the module docstring.
PHASE_TO_DRAW = "dx075_dy075"
MATCH_IOU = 0.5          # set to 0.3 to see the same picture after merging
SCALE = 2                # upscale factor, for legible box outlines
BOX_WIDTH = 2            # px before scaling
CONTEXT_WIDTH = 1        # px before scaling, for the blue context boxes in B

# --- colours, IBM colourblind safe --------------------------------------
COLOUR_ALL = (100, 143, 255)      # blue,    support 16
COLOUR_MID = (255, 176, 0)        # amber,   support 2 to 15
COLOUR_ONE = (220, 38, 127)       # magenta, support 1

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
IOU_TAG = str(MATCH_IOU).replace(".", "")


# =========================================================================
# imagery
# =========================================================================

def load_core_image():
    """Rebuild the scored window, then crop to the core region.

    Read is NATIVE, result is EXPERIMENT. Same path as run_gate.py and
    phase_sweep.py: native read, PIL bilinear resize.
    """
    win = Window(
        WIN_COL_OFF_NATIVE, WIN_ROW_OFF_NATIVE,
        WIN_SIZE_NATIVE, WIN_SIZE_NATIVE,
    )
    with rasterio.open(PATH) as src:
        rgb = src.read([1, 2, 3], window=win)
    native = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)
    scored = Image.fromarray(native).resize(
        (pm.WIN_SIZE, pm.WIN_SIZE), Image.BILINEAR
    )
    lo = pm.CORE_INSET
    hi = pm.WIN_SIZE - pm.CORE_INSET
    return scored.crop((lo, lo, hi, hi))


def canvas_from(core_img):
    img = core_img.resize(
        (core_img.width * SCALE, core_img.height * SCALE), Image.LANCZOS
    ).convert("RGB")
    return img, ImageDraw.Draw(img)


# =========================================================================
# drawing helpers
# =========================================================================

def band_of(n_phases):
    if n_phases == pm.N_PHASES:
        return "all", COLOUR_ALL
    if n_phases == 1:
        return "one", COLOUR_ONE
    return "mid", COLOUR_MID


def pick_font(size):
    for candidate in (
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rect(draw, xmin, ymin, xmax, ymax, colour, width):
    """Box in scored window coordinates, drawn in core crop coordinates."""
    draw.rectangle(
        [(xmin - pm.CORE_INSET) * SCALE, (ymin - pm.CORE_INSET) * SCALE,
         (xmax - pm.CORE_INSET) * SCALE, (ymax - pm.CORE_INSET) * SCALE],
        outline=colour, width=width * SCALE,
    )


def draw_legend(draw, title, lines, font, pad=14):
    """Top left legend block. lines is a list of (colour, label)."""
    sw = 11 * SCALE
    line_h = int(getattr(font, "size", 12) * 1.6)
    box_w = 190 * SCALE
    box_h = pad * 2 + line_h * (len(lines) + 1)
    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0))
    y = pad
    draw.text((pad, y), title, fill=(255, 255, 255), font=font)
    y += line_h
    for colour, label in lines:
        draw.rectangle([pad, y + 3, pad + sw, y + 3 + sw],
                       fill=colour, outline=colour)
        draw.text((pad + sw + 10, y), label, fill=(255, 255, 255), font=font)
        y += line_h


# =========================================================================
# figure A, one phase
# =========================================================================

def figure_per_phase(core_img, pooled, font):
    subset = pooled[pooled["phase_id"] == PHASE_TO_DRAW]
    if len(subset) == 0:
        raise SystemExit(
            f"no core boxes for phase {PHASE_TO_DRAW}. "
            "Check PHASE_TO_DRAW against the phase_boxes_*.csv filenames."
        )

    img, draw = canvas_from(core_img)
    counts = {"all": 0, "mid": 0, "one": 0}

    # well supported first, so singletons sit on top and stay visible
    order = {"all": 0, "mid": 1, "one": 2}
    subset = subset.assign(
        _band=[band_of(s)[0] for s in subset["support"]]
    ).sort_values("_band", key=lambda c: c.map(order))

    for _, r in subset.iterrows():
        band, colour = band_of(r["support"])
        counts[band] += 1
        rect(draw, r["xmin"], r["ymin"], r["xmax"], r["ymax"],
             colour, BOX_WIDTH)

    draw_legend(
        draw,
        f"{PHASE_TO_DRAW}   match IoU {MATCH_IOU}",
        [(COLOUR_ALL, f"support 16        {counts['all']:4d}"),
         (COLOUR_MID, f"support 2 to 15   {counts['mid']:4d}"),
         (COLOUR_ONE, f"support 1         {counts['one']:4d}")],
        font,
    )

    out = os.path.join(OUT_DIR, f"support_{PHASE_TO_DRAW}_iou{IOU_TAG}.png")
    img.save(out)
    print("--- figure A, one phase ---")
    print("phase           :", PHASE_TO_DRAW)
    print("boxes drawn     :", len(subset))
    print("  support 16    :", counts["all"])
    print("  support 2to15 :", counts["mid"])
    print("  support 1     :", counts["one"])
    print("wrote", os.path.basename(out))
    print("")
    return counts


# =========================================================================
# figure B, pooled singletons
# =========================================================================

def figure_pooled_singletons(core_img, pooled, clusters, font):
    singles = pooled[pooled["support"] == 1]
    stable = clusters[clusters["n_phases"] == pm.N_PHASES]

    img, draw = canvas_from(core_img)

    # context first, so singletons sit on top
    for _, r in stable.iterrows():
        rect(draw, r["mean_xmin"], r["mean_ymin"],
             r["mean_xmax"], r["mean_ymax"], COLOUR_ALL, CONTEXT_WIDTH)
    for _, r in singles.iterrows():
        rect(draw, r["xmin"], r["ymin"], r["xmax"], r["ymax"],
             COLOUR_ONE, BOX_WIDTH)

    draw_legend(
        draw,
        f"all 16 phases pooled   match IoU {MATCH_IOU}",
        [(COLOUR_ALL, f"support 16 mean   {len(stable):4d}"),
         (COLOUR_ONE, f"support 1 boxes   {len(singles):4d}")],
        font,
    )

    out = os.path.join(OUT_DIR,
                       f"support_pooled_singletons_iou{IOU_TAG}.png")
    img.save(out)

    print("--- figure B, pooled singletons ---")
    print("singleton boxes drawn :", len(singles))
    print("support 16 context    :", len(stable))
    if len(singles):
        per_phase = singles["phase_id"].value_counts().sort_index()
        print("")
        print("singletons per phase:")
        print(per_phase.to_string())
        print("")
        print("min", int(per_phase.min()), "max", int(per_phase.max()),
              "over", len(per_phase), "phases that have any")
    print("wrote", os.path.basename(out))
    print("")


# =========================================================================
# main
# =========================================================================

def run():
    print("match IoU     :", MATCH_IOU)
    print("core region   :", pm.WIN_SIZE - 2 * pm.CORE_INSET, "px square")
    print("")

    pool = pm.load_pool(OUT_DIR)
    clusters, cluster_of = pm.cluster_across_phases(pool, MATCH_IOU)
    pooled = pm.attach_clusters(pool, cluster_of, clusters)
    print("")
    print("distinct crowns:", len(clusters))
    print("")

    core_img = load_core_image()
    font = pick_font(15 * SCALE)

    figure_per_phase(core_img, pooled, font)
    figure_pooled_singletons(core_img, pooled, clusters, font)

    print("What to look for: are the magenta boxes on tree crowns, or on bare")
    print("ground, shadow and rock? Crowns means real detections flickering")
    print("with grid position. Ground means the singletons are junk and the")
    print("finding is about false positives, not instability.")


if __name__ == "__main__":
    run()
