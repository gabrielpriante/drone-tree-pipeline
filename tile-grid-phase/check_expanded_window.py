"""
Check the expanded read region used by phase_sweep.py.

phase_sweep.py needs a margin of one stride (300 px at 7.78 cm, so 600 px
at native resolution) on every side of the working window, filled with REAL
imagery rather than reflected pixels.

This script confirms the expanded region is inside the raster and reports how
much of it carries valid alpha. The margin ring is the part that matters: the
inner working window was already validated when it was chosen. Alpha 0 in the
margin means black pixels, which would be as damaging as synthetic ones.

Read only. Runs no model.
"""

import rasterio
from rasterio.windows import Window
import numpy as np

PATH = r"C:\Users\gabpe\Downloads\000103_ortho-dsm-ptcloud.tif"

# working window, settled
COL_OFF = 4820
ROW_OFF = 5260
SIZE = 2000

# margin, native px. phase_sweep.py pads by one stride (300 px) at the
# experiment resolution of 7.78 cm, which is 600 px at native 3.89 cm.
DOWNSAMPLE = 2
STRIDE = 300
MARGIN = STRIDE * DOWNSAMPLE

EXP_COL_OFF = COL_OFF - MARGIN
EXP_ROW_OFF = ROW_OFF - MARGIN
EXP_SIZE = SIZE + 2 * MARGIN

with rasterio.open(PATH) as src:
    print("raster           :", src.width, "x", src.height)
    print("working window   : col_off", COL_OFF, "row_off", ROW_OFF,
          "size", SIZE)
    print("margin native px :", MARGIN)
    print("expanded window  : col_off", EXP_COL_OFF, "row_off", EXP_ROW_OFF,
          "size", EXP_SIZE)

    in_bounds = (
        EXP_COL_OFF >= 0
        and EXP_ROW_OFF >= 0
        and EXP_COL_OFF + EXP_SIZE <= src.width
        and EXP_ROW_OFF + EXP_SIZE <= src.height
    )
    print("in bounds        :", in_bounds)
    if not in_bounds:
        raise SystemExit("expanded window falls outside the raster")

    win = Window(EXP_COL_OFF, EXP_ROW_OFF, EXP_SIZE, EXP_SIZE)
    alpha = src.read(4, window=win)

valid = alpha > 0
inner = valid[MARGIN:MARGIN + SIZE, MARGIN:MARGIN + SIZE]

ring_total = valid.size - inner.size
ring_valid = valid.sum() - inner.sum()

print("")
print("percent valid, expanded :", round(100.0 * valid.mean(), 2))
print("percent valid, inner    :", round(100.0 * inner.mean(), 2))
print("percent valid, margin   :", round(100.0 * ring_valid / ring_total, 2))

print("")
print("per side percent valid (margin bands):")
print("  top    :", round(100.0 * valid[:MARGIN, :].mean(), 2))
print("  bottom :", round(100.0 * valid[-MARGIN:, :].mean(), 2))
print("  left   :", round(100.0 * valid[:, :MARGIN].mean(), 2))
print("  right  :", round(100.0 * valid[:, -MARGIN:].mean(), 2))

worst = min(
    valid[:MARGIN, :].mean(),
    valid[-MARGIN:, :].mean(),
    valid[:, :MARGIN].mean(),
    valid[:, -MARGIN:].mean(),
)
print("")
print("worst side :", round(100.0 * worst, 2), "percent valid")
print("verdict    :", "OK for real pixel margin" if worst > 0.99
      else "margin has invalid pixels, do not use real pixel padding as is")
