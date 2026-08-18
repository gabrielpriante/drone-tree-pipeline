import rasterio
from rasterio.windows import Window
import numpy as np

path = r"C:\Users\gabpe\Downloads\000103_ortho-dsm-ptcloud.tif"
SIZE = 2000

with rasterio.open(path) as src:
    alpha = src.read(4, out_shape=(src.height // 20, src.width // 20))
    valid = alpha > 0
    ys, xs = np.where(valid)
    cy = int(np.median(ys) * 20)
    cx = int(np.median(xs) * 20)

    col_off = max(0, cx - SIZE // 2)
    row_off = max(0, cy - SIZE // 2)
    win = Window(col_off, row_off, SIZE, SIZE)

    a = src.read(4, window=win)
    pct_valid = 100.0 * (a > 0).sum() / a.size

    print("window col_off :", col_off)
    print("window row_off :", row_off)
    print("window size    :", SIZE)
    print("percent valid  :", round(pct_valid, 2))
