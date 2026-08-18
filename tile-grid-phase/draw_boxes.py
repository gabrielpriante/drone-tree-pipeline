import pandas as pd
from PIL import Image, ImageDraw

for fname in ["window_native", "window_half"]:
    try:
        boxes = pd.read_csv(fname + "_boxes.csv")
    except FileNotFoundError:
        print("no boxes for", fname)
        continue
    img = Image.open(fname + ".png").convert("RGB")
    draw = ImageDraw.Draw(img)
    for _, r in boxes.iterrows():
        draw.rectangle([r["xmin"], r["ymin"], r["xmax"], r["ymax"]],
                       outline=(255, 0, 0), width=2)
    img.save(fname + "_drawn.png")
    print("wrote", fname + "_drawn.png", "with", len(boxes), "boxes")
