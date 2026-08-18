import rasterio

path = r"C:\Users\gabpe\Downloads\000103_ortho-dsm-ptcloud.tif"

with rasterio.open(path) as src:
    print("width x height :", src.width, "x", src.height)
    print("bands          :", src.count)
    print("dtype          :", src.dtypes)
    print("CRS            :", src.crs)
    print("nodata         :", src.nodata)
    res_x, res_y = src.res
    print("pixel size     :", round(res_x, 4), "x", round(res_y, 4), "map units")
    print("GSD cm approx  :", round(res_x * 100, 2))
    print("footprint m    :", round(src.width * res_x, 1), "x", round(src.height * res_y, 1))
    print("bounds         :", src.bounds)
