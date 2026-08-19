"""
Build the viewer facing figure.

Audience is a city forestry director with no data science background. The
figure travels without narration, so every word it needs is on it.

Frame: same count, different trees.

Population, and there is only one on this figure
------------------------------------------------
The 274 trees on the list produced by a single survey, dx225_dy075. The pooled
union across all sixteen surveys, the one off count, and any percentage taken
over that union do not appear on the figure or in its caption. Those belong in
the paper.

Selection rule for the survey shown, recorded here and in the README, not on
the figure: it is at the median of the sixteen on BOTH core count and one off
detections. Core count 274 against a 15 position mean of 273.80, and 13 one off
detections against a median of 13, a min of 6 and a max of 17.

Encoding
--------
    solid blue      found by all sixteen surveys
    dashed amber    found by some surveys but not all

Solid against dashed is the grayscale cue, and it does not depend on either
colour surviving a monochrome conversion. It also carries the meaning:
intermittent line, intermittent tree.

Palette is #3987e5 and #c98500 on a #1a1a19 surface. Validated, not eyeballed:
all six checks pass, worst pair CVD deltaE 27.4 protan and 24.3 tritan,
normal vision 30.7, both at or above 3:1 against the surface.

Every box also carries a near black casing under the coloured line, because
the marks sit on a photograph that ranges from sunlit canopy to black shadow,
not on a flat chart surface. That is the one place this figure departs from a
plain chart, and the casing is what makes the palette work over both extremes.

The strip
---------
Sixteen bars, one per survey, from the n_core column of phase_summary.csv,
all sixteen including the 288. Each bar is stacked: a blue base of exactly 115
and an amber top of whatever else that survey found.

The blue base is 115 in every bar by construction, not by coincidence. A tree
found by all sixteen surveys has exactly one box in each of them, so every
survey's list contains those same 115. The flat blue line across the strip is
that fact drawn.

Bars start at zero. Truncating the axis would manufacture variation that is
not there, and the flatness is the point.

Language
--------
No occurrence of tile, phase, IoU, support, coefficient of variation, cluster,
or crown in any viewer facing string. No accuracy, correctness, precision,
recall, or error. There is no ground truth here. The claim is that repeat
surveys of one photograph disagree with each other.

Inputs
------
core_clean.png              950 by 950 clean backdrop
phase_boxes_*.csv           all sixteen, for the clustering
phase_summary.csv           n_core column, for the strip

N and M are derived at run time. Nothing about the counts is hardcoded, so the
text and the boxes cannot drift apart.

Output
------
figure_same_count_different_trees.png
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import phase_matching as pm

# =========================================================================
# CONFIG
# =========================================================================

PHASE_TO_DRAW = "dx225_dy075"
MATCH_IOU = 0.5

SCALE = 2                      # panel upscale
MARGIN = 70

# --- palette, validated on the dark surface -----------------------------
SURFACE = (26, 26, 25)         # #1a1a19
INK = (255, 255, 255)
INK_2 = (195, 194, 183)        # #c3c2b7
INK_MUTED = (125, 124, 116)
STABLE = (57, 135, 229)        # #3987e5
VARIES = (201, 133, 0)         # #c98500
CASING = (12, 12, 11)

BOX_W = 3                      # px before SCALE
CASE_W = BOX_W + 2
DASH_ON, DASH_OFF = 11, 8      # px before SCALE

SCALE_BAR_M = 20

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CHIP = os.path.join(OUT_DIR, "core_clean.png")
OUT = os.path.join(OUT_DIR, "figure_same_count_different_trees.png")

# --- viewer facing strings, locked --------------------------------------
TITLE = "The same photograph, surveyed sixteen times"
DECK = "This survey found {N} trees. {M} of them were found by all sixteen."
LEG_STABLE = "Found by all sixteen surveys, {M}"
LEG_VARIES = "Found by some surveys but not all, {V}"
STRIP_LABEL = ("How many trees each survey found. "
               "Every survey found between {LO} and {HI}.")
CAPTION = ("Sixteen surveys of one photograph. Nothing changed between them "
           "except where the grid of processing squares started. "
           "The count barely moved. Which trees appeared on the list did.")
CONTEXT = "Northern California closed conifer canopy. Not urban street trees."
CREDIT = ("Imagery from Open Forest Observatory Mission 000103, "
          "licensed CC BY 4.0.")


# =========================================================================
# text
# =========================================================================

def font_at(size, bold=False):
    names = ([r"C:\Windows\Fonts\arialbd.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
             if bold else
             [r"C:\Windows\Fonts\arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_h(draw, s, font):
    b = draw.textbbox((0, 0), s, font=font)
    return b[3] - b[1]


PROBE = ImageDraw.Draw(Image.new("RGB", (8, 8)))


def fit_font(s, max_w, start, bold=False, floor=14):
    """Largest size at or below start whose rendering fits max_w."""
    size = int(start)
    while size > floor:
        f = font_at(size, bold)
        if PROBE.textlength(s, font=f) <= max_w:
            return f
        size -= 2
    return font_at(floor, bold)


def wrap(s, font, max_w):
    """Greedy wrap. Never changes the wording, only where it breaks."""
    words, lines, cur = s.split(), [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if PROBE.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# =========================================================================
# marks
# =========================================================================

def dashed_line(draw, p0, p1, colour, width, on, off):
    (x0, y0), (x1, y1) = p0, p1
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length <= 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    t = 0.0
    while t < length:
        t2 = min(t + on, length)
        draw.line([(x0 + ux * t, y0 + uy * t),
                   (x0 + ux * t2, y0 + uy * t2)], fill=colour, width=width)
        t = t2 + off


def dashed_rect(draw, box, colour, width, on, off):
    x0, y0, x1, y1 = box
    for p0, p1 in [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                   ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]:
        dashed_line(draw, p0, p1, colour, width, on, off)


def draw_box(draw, box, colour, dashed):
    """Casing first, then the coloured line, so it reads on any canopy."""
    if dashed:
        dashed_rect(draw, box, CASING, CASE_W * SCALE,
                    DASH_ON * SCALE, DASH_OFF * SCALE)
        dashed_rect(draw, box, colour, BOX_W * SCALE,
                    DASH_ON * SCALE, DASH_OFF * SCALE)
    else:
        draw.rectangle(box, outline=CASING, width=CASE_W * SCALE)
        draw.rectangle(box, outline=colour, width=BOX_W * SCALE)


# =========================================================================
# data
# =========================================================================

def load():
    pool = pm.load_pool(OUT_DIR)
    clusters, cluster_of = pm.cluster_across_phases(pool, MATCH_IOU)
    pooled = pm.attach_clusters(pool, cluster_of, clusters)

    boxes = pooled[pooled["phase_id"] == PHASE_TO_DRAW].copy()
    if len(boxes) == 0:
        raise SystemExit(f"no boxes for {PHASE_TO_DRAW}")
    boxes["stable"] = boxes["support"] == pm.N_PHASES

    summary = pd.read_csv(os.path.join(OUT_DIR, "phase_summary.csv"))
    counts = summary["n_core"].tolist()

    n = len(boxes)
    m = int(boxes["stable"].sum())
    # the flat blue base of the strip is this same number, by construction
    assert m == int(
        (pd.Series(clusters["n_phases"]) == pm.N_PHASES).sum()
    ), "stable count in this survey differs from the all sixteen total"
    assert n == counts[summary["phase_id"].tolist().index(PHASE_TO_DRAW)], \
        "box count disagrees with phase_summary.csv"
    return boxes, counts, n, m


# =========================================================================
# panel
# =========================================================================

def build_panel(boxes):
    chip = Image.open(CHIP).convert("RGB")
    side = chip.width
    img = chip.resize((side * SCALE, side * SCALE), Image.LANCZOS)
    d = ImageDraw.Draw(img)

    # stable first so the varying boxes sit on top and stay legible
    for stable in (True, False):
        sub = boxes[boxes["stable"] == stable]
        colour = STABLE if stable else VARIES
        for _, r in sub.iterrows():
            box = [(r["xmin"] - pm.CORE_INSET) * SCALE,
                   (r["ymin"] - pm.CORE_INSET) * SCALE,
                   (r["xmax"] - pm.CORE_INSET) * SCALE,
                   (r["ymax"] - pm.CORE_INSET) * SCALE]
            draw_box(d, box, colour, dashed=not stable)

    draw_scale_bar(d, img.width, img.height)
    return img


def draw_scale_bar(d, w, h):
    px_per_m = 100.0 / pm.GSD_CM
    bar = SCALE_BAR_M * px_per_m * SCALE
    f = font_at(19 * SCALE, bold=True)
    pad = 26 * SCALE
    y = h - pad
    x1 = w - pad
    x0 = x1 - bar
    d.line([(x0, y), (x1, y)], fill=CASING, width=9 * SCALE)
    d.line([(x0, y), (x1, y)], fill=INK, width=5 * SCALE)
    for x in (x0, x1):
        d.line([(x, y - 9 * SCALE), (x, y + 9 * SCALE)],
               fill=CASING, width=9 * SCALE)
        d.line([(x, y - 9 * SCALE), (x, y + 9 * SCALE)],
               fill=INK, width=5 * SCALE)
    label = f"{SCALE_BAR_M} m"
    tb = d.textbbox((0, 0), label, font=f)
    tx = (x0 + x1) / 2 - (tb[2] - tb[0]) / 2
    ty = y - 16 * SCALE - (tb[3] - tb[1]) * 2
    d.text((tx + 2, ty + 2), label, font=f, fill=CASING)
    d.text((tx, ty), label, font=f, fill=INK)


# =========================================================================
# strip
# =========================================================================

def draw_strip(d, x, y, w, h, counts, m):
    n_bars = len(counts)
    gap = max(6, int(w * 0.012))
    bar_w = (w - gap * (n_bars - 1)) / n_bars
    top = float(max(counts)) * 1.10
    seg_gap = 2 * SCALE            # surface gap between stacked fills
    f_ax = font_at(17 * SCALE)

    base = y + h
    for i, c in enumerate(counts):
        bx = x + i * (bar_w + gap)
        h_all = h * c / top
        h_blue = h * m / top
        # amber top, rounded data end
        d.rounded_rectangle(
            [bx, base - h_all, bx + bar_w, base - h_blue + seg_gap],
            radius=4 * SCALE, fill=VARIES,
        )
        # blue base, anchored to the baseline
        d.rectangle([bx, base - h_blue, bx + bar_w, base], fill=STABLE)

    d.line([(x, base), (x + w, base)], fill=INK_MUTED, width=2)
    for val in (0, 300):
        yy = base - h * val / top
        if val:
            d.line([(x, yy), (x + w, yy)], fill=(58, 58, 55), width=2)
        d.text((x - 14 * SCALE, yy - 10 * SCALE), str(val),
               font=f_ax, fill=INK_MUTED, anchor="ra")


# =========================================================================
# compose
# =========================================================================

def build():
    boxes, counts, N, M = load()
    V = N - M
    print(f"{PHASE_TO_DRAW}: N {N}   found by all sixteen {M}   rest {V}")
    print(f"strip: {len(counts)} bars, min {min(counts)}, max {max(counts)}")

    panel = build_panel(boxes)
    pw, ph = panel.size
    CW = pw                       # content width, everything aligns to it

    deck = DECK.format(N=N, M=M)
    leg_a = LEG_STABLE.format(M=M)
    leg_b = LEG_VARIES.format(V=V)
    strip_lab = STRIP_LABEL.format(LO=min(counts), HI=max(counts))

    f_title = fit_font(TITLE, CW, 84, bold=True)
    f_deck = fit_font(deck, CW, 56)
    f_lab = fit_font(strip_lab, CW, 40)
    f_cap = font_at(42)
    f_fine = font_at(34)

    # legend on one row if it fits, otherwise stacked
    sw = 108                      # swatch length
    pad_sw = 26                   # swatch to label
    pad_item = 96                 # item to item
    f_leg = font_at(40)
    row_w = (sw + pad_sw + PROBE.textlength(leg_a, font=f_leg) + pad_item
             + sw + pad_sw + PROBE.textlength(leg_b, font=f_leg))
    while row_w > CW and f_leg.size > 24:
        f_leg = font_at(f_leg.size - 2)
        row_w = (sw + pad_sw + PROBE.textlength(leg_a, font=f_leg) + pad_item
                 + sw + pad_sw + PROBE.textlength(leg_b, font=f_leg))
    leg_stacked = row_w > CW

    cap_lines = wrap(CAPTION, f_cap, CW)

    h_title = text_h(PROBE, TITLE, f_title)
    h_deck = text_h(PROBE, deck, f_deck)
    h_leg = text_h(PROBE, "Ag", f_leg)
    h_lab = text_h(PROBE, "Ag", f_lab)
    h_cap = text_h(PROBE, "Ag", f_cap)
    h_fine = text_h(PROBE, "Ag", f_fine)

    leg_block = h_leg * (2 if leg_stacked else 1) + (18 if leg_stacked else 0)
    strip_h = 150 * SCALE
    H = int(
        MARGIN + h_title + 26 + h_deck + 34 + leg_block + 30
        + ph + 46
        + h_lab + 22 + strip_h + 56
        + len(cap_lines) * (h_cap + 12) + 22
        + h_fine + 16 + h_fine + MARGIN
    )

    img = Image.new("RGB", (pw + 2 * MARGIN, H), SURFACE)
    d = ImageDraw.Draw(img)
    x, y = MARGIN, MARGIN

    d.text((x, y), TITLE, font=f_title, fill=INK)
    y += h_title + 26
    d.text((x, y), deck, font=f_deck, fill=INK)
    y += h_deck + 34

    # legend. Identity is never colour alone here: solid against dashed.
    def legend_item(lx, ly, colour, label, dashed):
        cy = ly + h_leg // 2
        if dashed:
            dashed_line(d, (lx, cy), (lx + sw, cy), colour,
                        BOX_W * SCALE, DASH_ON * SCALE, DASH_OFF * SCALE)
        else:
            d.line([(lx, cy), (lx + sw, cy)], fill=colour, width=BOX_W * SCALE)
        d.text((lx + sw + pad_sw, ly), label, font=f_leg, fill=INK_2)
        return lx + sw + pad_sw + PROBE.textlength(label, font=f_leg)

    if leg_stacked:
        legend_item(x, y, STABLE, leg_a, False)
        legend_item(x, y + h_leg + 18, VARIES, leg_b, True)
    else:
        endx = legend_item(x, y, STABLE, leg_a, False)
        legend_item(endx + pad_item, y, VARIES, leg_b, True)
    y += leg_block + 30

    img.paste(panel, (x, int(y)))
    y += ph + 46

    d.text((x, y), strip_lab, font=f_lab, fill=INK_2)
    y += h_lab + 22
    draw_strip(d, x + 40 * SCALE, y, pw - 40 * SCALE, strip_h, counts, M)
    y += strip_h + 56

    for line in cap_lines:
        d.text((x, y), line, font=f_cap, fill=INK)
        y += h_cap + 12
    y += 22

    d.text((x, y), CONTEXT, font=f_fine, fill=INK_2)
    y += h_fine + 16
    d.text((x, y), CREDIT, font=f_fine, fill=INK_MUTED)

    img.save(OUT)
    print("wrote", os.path.basename(OUT), img.size)


if __name__ == "__main__":
    build()
