"""
Draw the core region at phase (0, 0) with crowns coloured by support count.

Purpose: the sweep found 196 crowns detected at exactly one grid position out
of 16. Those are either plausible trees that flicker in and out, or junk on
bare ground and shadow. The two are different findings and the paper has to
say which. This is the picture that settles it by eye.

Bands
-----
    support 16      blue      found at every grid position
    support 2 to 15 amber     found at some positions, not all
    support 1       magenta   found at exactly one position

Colours are from the IBM colourblind safe palette.

What it draws
-------------
Boxes belonging to phase dx000_dy000 only, inside the core region, coloured by
the support of the cluster each one landed in. Drawing one phase rather than
all 16 keeps the image readable: 16 overlapping copies of every crown would be
unreadable.

Caveat worth knowing before reading the picture: a support 1 crown drawn here
is a phase (0, 0) box with no counterpart at any other phase. Singletons that
belong to other phases exist too and are NOT drawn, because their boxes are
not in phase (0, 0). Change PHASE_TO_DRAW to look at another phase.

Reads the orthomosaic to rebuild the imagery. Loads no model and runs no
inference.

Output (gitignored)
-------------------
support_<phase>_iou<threshold>.png

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
PHASE_TO_DRAW = "dx000_dy000"
MATCH_IOU = 0.5          # set to 0.3 to see the same picture after merging
SCALE = 2                # upscale factor, for legible box outlines
BOX_WIDTH = 2            # px before scaling

# --- colours, IBM colourblind safe --------------------------------------
COLOUR_ALL = (100, 143, 255)      # blue,    support 16
COLOUR_MID = (255, 176, 0)        # amber,   support 2 to 15
COLOUR_ONE = (220, 38, 127)       # magenta, support 1

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def draw_legend(draw, counts, font, pad=14):
    """Top left legend block with per band counts."""
    lines = [
        (COLOUR_ALL, f"support 16        {counts['all']:4d}"),
        (COLOUR_MID, f"support 2 to 15   {counts['mid']:4d}"),
        (COLOUR_ONE, f"support 1         {counts['one']:4d}"),
    ]
    sw = 22 * SCALE // 2
    line_h = int(font.size * 1.6) if hasattr(font, "size") else 18
    box_w = 340 * SCALE // 2
    box_h = pad * 2 + line_h * (len(lines) + 1)

    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0))
    y = pad
    draw.text((pad, y), f"{PHASE_TO_DRAW}   match IoU {MATCH_IOU}",
              fill=(255, 255, 255), font=font)
    y += line_h
    for colour, label in lines:
        draw.rectangle([pad, y + 3, pad + sw, y + 3 + sw],
                       fill=colour, outline=colour)
        draw.text((pad + sw + 10, y), label, fill=(255, 255, 255), font=font)
        y += line_h


def run():
    print("phase to draw :", PHASE_TO_DRAW)
    print("match IoU     :", MATCH_IOU)
    print("core region   :", pm.WIN_SIZE - 2 * pm.CORE_INSET, "px square")
    print("")

    pool = pm.load_pool(OUT_DIR)
    clusters, cluster_of = pm.cluster_across_phases(pool, MATCH_IOU)
    print("")
    print("distinct crowns:", len(clusters))

    support = clusters["n_phases"].to_numpy()
    pool = pool.copy()
    pool["cluster_id"] = cluster_of
    pool["support"] = support[cluster_of]

    subset = pool[pool["phase_id"] == PHASE_TO_DRAW]
    if len(subset) == 0:
        raise SystemExit(
            f"no core boxes for phase {PHASE_TO_DRAW}. "
            "Check PHASE_TO_DRAW against the phase_boxes_*.csv filenames."
        )
    print("boxes drawn    :", len(subset))

    img = load_core_image()
    img = img.resize((img.width * SCALE, img.height * SCALE), Image.LANCZOS)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    counts = {"all": 0, "mid": 0, "one": 0}
    # draw the well supported first so singletons sit on top and stay visible
    order = {"all": 0, "mid": 1, "one": 2}
    subset = subset.assign(
        _band=[band_of(s)[0] for s in subset["support"]]
    ).sort_values("_band", key=lambda c: c.map(order))

    for _, r in subset.iterrows():
        band, colour = band_of(r["support"])
        counts[band] += 1
        x0 = (r["xmin"] - pm.CORE_INSET) * SCALE
        y0 = (r["ymin"] - pm.CORE_INSET) * SCALE
        x1 = (r["xmax"] - pm.CORE_INSET) * SCALE
        y1 = (r["ymax"] - pm.CORE_INSET) * SCALE
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=BOX_WIDTH * SCALE)

    draw_legend(draw, counts, pick_font(15 * SCALE))

    tag = str(MATCH_IOU).replace(".", "")
    out = os.path.join(OUT_DIR, f"support_{PHASE_TO_DRAW}_iou{tag}.png")
    img.save(out)

    print("")
    print("band counts in this phase:")
    print("  support 16      :", counts["all"])
    print("  support 2 to 15 :", counts["mid"])
    print("  support 1       :", counts["one"])
    print("")
    print("wrote", os.path.basename(out))
    print("")
    print("What to look for: are the magenta boxes on tree crowns, or on bare")
    print("ground, shadow, and rock? Crowns means real detections flickering")
    print("with grid position. Ground means the singletons are junk and the")
    print("finding is about false positives, not instability.")


if __name__ == "__main__":
    run()
