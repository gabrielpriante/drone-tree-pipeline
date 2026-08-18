import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import numpy as np
from PIL import Image
from deepforest import main
import pandas as pd

path = r"C:\Users\gabpe\Downloads\000103_ortho-dsm-ptcloud.tif"

COL_OFF = 4820
ROW_OFF = 5260
SIZE = 2000

win = Window(COL_OFF, ROW_OFF, SIZE, SIZE)

with rasterio.open(path) as src:
    rgb = src.read([1, 2, 3], window=win)

img_native = np.transpose(rgb, (1, 2, 0)).astype(np.uint8)
Image.fromarray(img_native).save("window_native.png")

half = SIZE // 2
img_half = np.array(
    Image.fromarray(img_native).resize((half, half), Image.BILINEAR)
)
Image.fromarray(img_half).save("window_half.png")

model = main.deepforest()
model.load_model("weecology/deepforest-tree")

for label, fname, gsd in [
    ("native 3.89 cm", "window_native.png", 3.89),
    ("downsampled 7.78 cm", "window_half.png", 7.78),
]:
    boxes = model.predict_tile(
        path=fname,
        patch_size=400,
        patch_overlap=0.25,
    )
    if boxes is None or len(boxes) == 0:
        print(label, "-> 0 detections")
        continue

    boxes.to_csv(fname.replace(".png", "_boxes.csv"), index=False)
    w = boxes["xmax"] - boxes["xmin"]
    h = boxes["ymax"] - boxes["ymin"]

    print("")
    print("===", label)
    print("detections      :", len(boxes))
    print("score min/med/max:",
          round(boxes["score"].min(), 3),
          round(boxes["score"].median(), 3),
          round(boxes["score"].max(), 3))
    print("box width px    : median", round(w.median(), 1))
    print("box width m     : median", round(w.median() * gsd / 100, 2))
