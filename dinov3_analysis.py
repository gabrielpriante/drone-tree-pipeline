#!/usr/bin/env python3
"""
dinov3_analysis.py — DINOv3-powered drone image analysis for tree detection & canopy mapping.

Part of the drone-tree-pipeline repo (https://github.com/gabrielpriante/drone-tree-pipeline).
Standalone script: run it on a folder of nadir DJI drone images and get back:

  1. Canopy Height Maps (CHMv2)        → per-pixel tree height estimates (meters)
  2. Tree detection bounding boxes      → from DINOv3 dense features + segmentation
  3. Dense feature visualizations       → PCA-colored patch maps for QA / clustering

Usage:
    python dinov3_analysis.py --input ground_truth/ --output dinov3_outputs/
    python dinov3_analysis.py --input ground_truth/ --output dinov3_outputs/ --skip-chmv2
    python dinov3_analysis.py --input ground_truth/ --output dinov3_outputs/ --only chmv2

Requirements:
    pip install torch torchvision transformers pillow numpy matplotlib scikit-learn
    pip install geopandas shapely  # optional, for GeoJSON export with GPS coords

Hardware:
    Designed for a desktop GPU with ≥6 GB VRAM (e.g., RTX 4060 / RX 7600).
    Falls back to CPU automatically if CUDA is unavailable.
    MPS (Apple Silicon) is also supported.

Author: Gabriel Priante / Frontline Gig
License: MIT (same as repo)
"""

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ExifTags

# ---------------------------------------------------------------------------
# Lazy imports — we don't want torch import time if the user just wants --help
# ---------------------------------------------------------------------------

def _import_torch():
    import torch
    return torch

def _import_transformers():
    from transformers import AutoImageProcessor, AutoModel, AutoModelForDepthEstimation
    return AutoImageProcessor, AutoModel, AutoModelForDepthEstimation


# ===========================================================================
#  CONFIG
# ===========================================================================

@dataclass
class Config:
    input_dir: str = "ground_truth"
    output_dir: str = "dinov3_outputs"

    # Model choices (Hugging Face model IDs)
    backbone_model: str = "facebook/dinov3-vitl16-pretrain-sat493m"
    chmv2_model: str = "facebook/dinov3-vitl16-chmv2-dpt-head"

    # Processing
    tile_size: int = 512          # crop size for tiling large images
    tile_overlap: int = 64        # overlap in pixels between tiles
    batch_size: int = 4           # tiles per forward pass (tune to your VRAM)
    resize_for_features: int = 518  # DINOv3 ViT uses 518px default

    # Outputs
    run_chmv2: bool = True
    run_detections: bool = True
    run_features: bool = True

    # Detection thresholds (for feature-based detection)
    detection_height_threshold: float = 1.0   # meters — minimum CHM height to count as tree
    min_crown_area_px: int = 100              # minimum connected-component size in pixels

    # Extensions to scan
    image_extensions: tuple = (".jpg", ".jpeg", ".tif", ".tiff", ".png", ".dng")


# ===========================================================================
#  GPS EXTRACTION
# ===========================================================================

def extract_gps(image_path: str) -> dict | None:
    """Pull GPS lat/lon/alt from EXIF. Returns dict or None."""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None

        gps_tag_id = None
        for tag_id, tag_name in ExifTags.TAGS.items():
            if tag_name == "GPSInfo":
                gps_tag_id = tag_id
                break
        if gps_tag_id is None or gps_tag_id not in exif:
            return None

        gps_info = exif[gps_tag_id]
        gps_data = {}
        for key, val in gps_info.items():
            decoded = ExifTags.GPSTAGS.get(key, key)
            gps_data[decoded] = val

        def _to_decimal(dms, ref):
            d, m, s = dms
            dec = float(d) + float(m) / 60.0 + float(s) / 3600.0
            if ref in ("S", "W"):
                dec = -dec
            return dec

        lat = _to_decimal(gps_data["GPSLatitude"], gps_data.get("GPSLatitudeRef", "N"))
        lon = _to_decimal(gps_data["GPSLongitude"], gps_data.get("GPSLongitudeRef", "W"))
        alt = float(gps_data.get("GPSAltitude", 0))

        return {"latitude": lat, "longitude": lon, "altitude_m": alt}
    except Exception:
        return None


# ===========================================================================
#  DEVICE SELECTION
# ===========================================================================

def get_device():
    torch = _import_torch()
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Device: {name} ({vram:.1f} GB VRAM)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("  Device: Apple Silicon (MPS)")
    else:
        dev = torch.device("cpu")
        print("  Device: CPU (this will be slow)")
    return dev


# ===========================================================================
#  MODULE 1: CANOPY HEIGHT MAPS (CHMv2)
# ===========================================================================

def run_chmv2(image_paths: list[str], config: Config, device) -> dict:
    """
    Run CHMv2 canopy height estimation on each image.
    Returns {filename: {"height_map": np.ndarray, "stats": dict, "gps": dict|None}}
    """
    torch = _import_torch()
    AutoImageProcessor, _, AutoModelForDepthEstimation = _import_transformers()
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    print("\n" + "=" * 60)
    print("  MODULE 1: Canopy Height Maps (CHMv2)")
    print("=" * 60)

    print(f"  Loading model: {config.chmv2_model}")
    processor = AutoImageProcessor.from_pretrained(config.chmv2_model)
    model = AutoModelForDepthEstimation.from_pretrained(config.chmv2_model)
    model = model.to(device).eval()
    print("  Model loaded.")

    results = {}
    out_dir = Path(config.output_dir) / "chmv2"
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        fname = Path(img_path).stem
        print(f"\n  Processing: {Path(img_path).name}")
        t0 = time.time()

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # CHMv2 processor handles resizing/normalization
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        # Post-process to original image dimensions
        height_map = processor.post_process_depth_estimation(
            outputs, target_sizes=[(orig_h, orig_w)]
        )[0]["predicted_depth"]

        height_np = height_map.cpu().numpy()

        # Clamp negatives (artifacts)
        height_np = np.clip(height_np, 0, None)

        elapsed = time.time() - t0
        stats = {
            "max_height_m": float(np.max(height_np)),
            "mean_height_m": float(np.mean(height_np[height_np > 0.5])) if np.any(height_np > 0.5) else 0.0,
            "pct_canopy_cover": float(np.mean(height_np > 1.0) * 100),
            "processing_time_s": round(elapsed, 2),
        }
        gps = extract_gps(img_path)

        print(f"    Max height: {stats['max_height_m']:.1f} m")
        print(f"    Mean canopy: {stats['mean_height_m']:.1f} m")
        print(f"    Canopy cover (>1m): {stats['pct_canopy_cover']:.1f}%")
        print(f"    Time: {elapsed:.1f}s")

        # Save height map as numpy and as visualization
        np.save(out_dir / f"{fname}_height.npy", height_np)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].imshow(image)
        axes[0].set_title("Original Drone Image", fontsize=11)
        axes[0].axis("off")

        im = axes[1].imshow(height_np, cmap=colormaps["YlGn"], vmin=0,
                            vmax=max(float(np.percentile(height_np, 99)), 3.0))
        axes[1].set_title("CHMv2 Canopy Height (meters)", fontsize=11)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Height (m)")

        plt.suptitle(f"{Path(img_path).name}", fontsize=10, color="gray")
        plt.tight_layout()
        plt.savefig(out_dir / f"{fname}_chmv2.png", dpi=150, bbox_inches="tight")
        plt.close()

        results[Path(img_path).name] = {
            "height_map": height_np,
            "stats": stats,
            "gps": gps,
        }

    # Clean up GPU memory
    del model, processor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n  CHMv2 complete — outputs in: {out_dir}/")
    return results


# ===========================================================================
#  MODULE 2: TREE DETECTION FROM CHMv2 HEIGHT MAPS
# ===========================================================================

def run_detections(chmv2_results: dict, image_paths: list[str], config: Config) -> dict:
    """
    Derive tree bounding boxes from CHMv2 height maps using connected-component
    analysis on the thresholded canopy height surface.

    Returns {filename: [{"tree_id": str, "bbox": [x,y,w,h], "height_m": float, ...}]}
    """
    from scipy import ndimage
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    print("\n" + "=" * 60)
    print("  MODULE 2: Tree Detections (from CHMv2 height surface)")
    print("=" * 60)

    out_dir = Path(config.output_dir) / "detections"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_detections = {}

    for img_path in image_paths:
        fname = Path(img_path).name
        if fname not in chmv2_results:
            continue

        height_np = chmv2_results[fname]["height_map"]
        gps = chmv2_results[fname]["gps"]
        stem = Path(img_path).stem

        print(f"\n  Detecting trees in: {fname}")

        # Threshold the height map
        canopy_mask = height_np > config.detection_height_threshold

        # Label connected components
        labeled, n_components = ndimage.label(canopy_mask)

        detections = []
        tree_id = 0

        for comp_id in range(1, n_components + 1):
            component = labeled == comp_id
            area_px = int(np.sum(component))

            if area_px < config.min_crown_area_px:
                continue

            # Bounding box
            rows = np.where(component.any(axis=1))[0]
            cols = np.where(component.any(axis=0))[0]
            y_min, y_max = rows[0], rows[-1]
            x_min, x_max = cols[0], cols[-1]

            # Height stats within this component
            comp_heights = height_np[component]

            tree_id += 1
            det = {
                "tree_id": f"DINOV3_{tree_id:04d}",
                "bbox_x": int(x_min),
                "bbox_y": int(y_min),
                "bbox_w": int(x_max - x_min),
                "bbox_h": int(y_max - y_min),
                "area_px": area_px,
                "max_height_m": round(float(np.max(comp_heights)), 2),
                "mean_height_m": round(float(np.mean(comp_heights)), 2),
                "center_x_px": int((x_min + x_max) / 2),
                "center_y_px": int((y_min + y_max) / 2),
            }
            detections.append(det)

        print(f"    Found {len(detections)} tree crowns (threshold={config.detection_height_threshold}m, "
              f"min_area={config.min_crown_area_px}px)")

        # Save detection overlay image
        image = Image.open(img_path).convert("RGB")
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(image)

        for det in detections:
            rect = mpatches.Rectangle(
                (det["bbox_x"], det["bbox_y"]),
                det["bbox_w"], det["bbox_h"],
                linewidth=1.5, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                det["bbox_x"], det["bbox_y"] - 4,
                f'{det["tree_id"]} ({det["max_height_m"]:.1f}m)',
                color="lime", fontsize=6, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6)
            )

        ax.set_title(f"DINOv3 Tree Detections — {fname} ({len(detections)} trees)", fontsize=11)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_detections.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Save detections as JSON
        with open(out_dir / f"{stem}_detections.json", "w") as f:
            json.dump({"image": fname, "gps": gps, "detections": detections}, f, indent=2)

        # Save as CSV for easy import
        if detections:
            import csv
            csv_path = out_dir / f"{stem}_detections.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=detections[0].keys())
                writer.writeheader()
                writer.writerows(detections)

        all_detections[fname] = detections

    # Also create a combined GeoJSON if GPS is available
    _export_geojson(all_detections, chmv2_results, image_paths, out_dir)

    print(f"\n  Detections complete — outputs in: {out_dir}/")
    return all_detections


def _export_geojson(all_detections, chmv2_results, image_paths, out_dir):
    """Attempt to create a combined GeoJSON of all detections with approximate coords."""
    features = []

    for img_path in image_paths:
        fname = Path(img_path).name
        if fname not in chmv2_results or fname not in all_detections:
            continue

        gps = chmv2_results[fname].get("gps")
        if not gps:
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        # Rough meters-per-pixel estimate from altitude
        # DJI Mini 5 Pro: 24mm equiv FOV ≈ 82° → at altitude h,
        # ground coverage ≈ 2 * h * tan(41°) across the wider dimension
        alt = gps.get("altitude_m", 100)
        import math
        ground_width_m = 2 * alt * math.tan(math.radians(41))
        mpp = ground_width_m / max(img_w, img_h)

        # Degrees per meter at this latitude
        lat_deg_per_m = 1.0 / 111320.0
        lon_deg_per_m = 1.0 / (111320.0 * math.cos(math.radians(gps["latitude"])))

        center_lon = gps["longitude"]
        center_lat = gps["latitude"]

        for det in all_detections[fname]:
            # Offset from image center in pixels
            dx_px = det["center_x_px"] - img_w / 2
            dy_px = det["center_y_px"] - img_h / 2

            # Convert to geographic offset (note: y is inverted in image coords)
            det_lon = center_lon + dx_px * mpp * lon_deg_per_m
            det_lat = center_lat - dy_px * mpp * lat_deg_per_m

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(det_lon, 7), round(det_lat, 7)]
                },
                "properties": {
                    "tree_id": det["tree_id"],
                    "source_image": fname,
                    "max_height_m": det["max_height_m"],
                    "mean_height_m": det["mean_height_m"],
                    "crown_area_px": det["area_px"],
                    "bbox_w_px": det["bbox_w"],
                    "bbox_h_px": det["bbox_h"],
                }
            }
            features.append(feature)

    if features:
        geojson = {"type": "FeatureCollection", "features": features}
        geojson_path = out_dir / "all_detections.geojson"
        with open(geojson_path, "w") as f:
            json.dump(geojson, f, indent=2)
        print(f"    GeoJSON exported: {geojson_path} ({len(features)} features)")


# ===========================================================================
#  MODULE 3: DENSE FEATURE VISUALIZATION
# ===========================================================================

def run_feature_viz(image_paths: list[str], config: Config, device) -> None:
    """
    Extract DINOv3 SAT backbone features and visualize them as PCA-colored maps.
    Great for QA, identifying vegetation clusters, and understanding what the model sees.
    """
    torch = _import_torch()
    AutoImageProcessor, AutoModel, _ = _import_transformers()
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("  MODULE 3: Dense Feature Visualization (DINOv3 SAT backbone)")
    print("=" * 60)

    print(f"  Loading backbone: {config.backbone_model}")
    processor = AutoImageProcessor.from_pretrained(config.backbone_model)
    model = AutoModel.from_pretrained(config.backbone_model)
    model = model.to(device).eval()
    print("  Backbone loaded.")

    out_dir = Path(config.output_dir) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_patch_features = []
    image_meta = []

    for img_path in image_paths:
        fname = Path(img_path).name
        stem = Path(img_path).stem
        print(f"\n  Extracting features: {fname}")
        t0 = time.time()

        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

        # last_hidden_state shape: (1, n_patches + 1, embed_dim)
        # First token is CLS, rest are spatial patches
        patch_tokens = outputs.last_hidden_state[0, 1:, :].cpu().numpy()  # (n_patches, embed_dim)

        # Figure out the spatial grid (ViT-L/16 with 518px input → 518/14 = 37 patches per side)
        n_patches = patch_tokens.shape[0]
        grid_size = int(np.sqrt(n_patches))
        if grid_size * grid_size != n_patches:
            # Non-square — try to infer
            grid_size = int(np.round(np.sqrt(n_patches)))

        elapsed = time.time() - t0
        print(f"    Patches: {n_patches} ({grid_size}x{grid_size}), embed_dim: {patch_tokens.shape[1]}")
        print(f"    Time: {elapsed:.1f}s")

        all_patch_features.append(patch_tokens)
        image_meta.append({
            "path": img_path,
            "fname": fname,
            "stem": stem,
            "grid_size": grid_size,
            "n_patches": n_patches,
        })

    # Global PCA across all images (so colors are comparable)
    print("\n  Running PCA across all images...")
    combined = np.vstack(all_patch_features)
    pca = PCA(n_components=3)
    pca.fit(combined)
    print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

    # Generate per-image visualizations
    offset = 0
    for i, meta in enumerate(image_meta):
        n = meta["n_patches"]
        g = meta["grid_size"]

        patch_pca = pca.transform(all_patch_features[i])  # (n_patches, 3)

        # Normalize to [0, 1] for RGB display
        for ch in range(3):
            mn, mx = patch_pca[:, ch].min(), patch_pca[:, ch].max()
            if mx > mn:
                patch_pca[:, ch] = (patch_pca[:, ch] - mn) / (mx - mn)

        # Reshape into spatial grid
        feature_map = patch_pca[:g*g].reshape(g, g, 3)

        # Plot side by side
        image = Image.open(meta["path"]).convert("RGB")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].imshow(image)
        axes[0].set_title("Original Drone Image", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(feature_map, interpolation="nearest")
        axes[1].set_title("DINOv3 SAT Feature Map (PCA → RGB)", fontsize=11)
        axes[1].axis("off")

        plt.suptitle(f"{meta['fname']} — {g}x{g} patches", fontsize=10, color="gray")
        plt.tight_layout()
        plt.savefig(out_dir / f"{meta['stem']}_features.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Also save raw features for downstream use
        np.save(out_dir / f"{meta['stem']}_features.npy", all_patch_features[i])

        offset += n

    # Clean up
    del model, processor
    torch = _import_torch()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n  Feature visualization complete — outputs in: {out_dir}/")


# ===========================================================================
#  SUMMARY REPORT
# ===========================================================================

def write_summary(chmv2_results: dict, detection_results: dict, config: Config):
    """Write a combined summary JSON and print to console."""

    out_dir = Path(config.output_dir)
    summary = {
        "pipeline": "dinov3_analysis.py",
        "backbone": config.backbone_model,
        "chmv2_model": config.chmv2_model,
        "images_processed": len(chmv2_results) if chmv2_results else 0,
        "images": {},
    }

    for fname in (chmv2_results or {}):
        img_summary = {}
        if chmv2_results and fname in chmv2_results:
            img_summary["chmv2_stats"] = chmv2_results[fname]["stats"]
            img_summary["gps"] = chmv2_results[fname]["gps"]
        if detection_results and fname in detection_results:
            img_summary["n_trees_detected"] = len(detection_results[fname])
            if detection_results[fname]:
                heights = [d["max_height_m"] for d in detection_results[fname]]
                img_summary["tallest_tree_m"] = max(heights)
                img_summary["mean_tree_height_m"] = round(sum(heights) / len(heights), 2)
        summary["images"][fname] = img_summary

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for fname, data in summary["images"].items():
        print(f"\n  {fname}:")
        if "chmv2_stats" in data:
            s = data["chmv2_stats"]
            print(f"    Canopy cover: {s['pct_canopy_cover']:.1f}% | Max height: {s['max_height_m']:.1f}m")
        if "n_trees_detected" in data:
            print(f"    Trees detected: {data['n_trees_detected']}")
            if data.get("tallest_tree_m"):
                print(f"    Tallest: {data['tallest_tree_m']:.1f}m | Mean: {data['mean_tree_height_m']:.1f}m")
        if data.get("gps"):
            g = data["gps"]
            print(f"    GPS: {g['latitude']:.6f}, {g['longitude']:.6f} @ {g['altitude_m']:.0f}m AGL")

    print(f"\n  Full summary: {summary_path}")
    return summary


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DINOv3 drone image analysis — canopy height, tree detection, feature maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dinov3_analysis.py --input ground_truth/ --output dinov3_outputs/
  python dinov3_analysis.py --input ground_truth/ --only chmv2
  python dinov3_analysis.py --input ground_truth/ --skip-chmv2 --skip-detections
        """
    )
    parser.add_argument("--input", "-i", default="ground_truth",
                        help="Directory containing drone images (default: ground_truth/)")
    parser.add_argument("--output", "-o", default="dinov3_outputs",
                        help="Output directory (default: dinov3_outputs/)")
    parser.add_argument("--only", choices=["chmv2", "detections", "features"],
                        help="Run only one module")
    parser.add_argument("--skip-chmv2", action="store_true", help="Skip canopy height maps")
    parser.add_argument("--skip-detections", action="store_true", help="Skip tree detections")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature visualization")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for tiled processing")
    parser.add_argument("--height-threshold", type=float, default=1.0,
                        help="Min height (m) to count as tree crown (default: 1.0)")
    parser.add_argument("--min-crown-area", type=int, default=100,
                        help="Min connected-component area in pixels (default: 100)")

    args = parser.parse_args()

    config = Config(
        input_dir=args.input,
        output_dir=args.output,
        batch_size=args.batch_size,
        detection_height_threshold=args.height_threshold,
        min_crown_area_px=args.min_crown_area,
    )

    if args.only:
        config.run_chmv2 = args.only == "chmv2"
        config.run_detections = args.only == "detections"
        config.run_features = args.only == "features"
    else:
        if args.skip_chmv2:
            config.run_chmv2 = False
        if args.skip_detections:
            config.run_detections = False
        if args.skip_features:
            config.run_features = False

    # Detections depend on CHMv2
    if config.run_detections and not config.run_chmv2:
        print("  Note: detections require CHMv2 — enabling CHMv2 automatically.")
        config.run_chmv2 = True

    # Discover images
    input_path = Path(config.input_dir)
    if not input_path.exists():
        print(f"  Error: input directory not found: {input_path}")
        sys.exit(1)

    image_paths = sorted([
        str(p) for p in input_path.iterdir()
        if p.suffix.lower() in config.image_extensions
    ])

    if not image_paths:
        print(f"  Error: no images found in {input_path}")
        sys.exit(1)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  DINOv3 Drone Analysis Pipeline")
    print("=" * 60)
    print(f"  Images: {len(image_paths)} found in {config.input_dir}/")
    print(f"  Output: {config.output_dir}/")
    print(f"  Modules: CHMv2={'ON' if config.run_chmv2 else 'OFF'} | "
          f"Detections={'ON' if config.run_detections else 'OFF'} | "
          f"Features={'ON' if config.run_features else 'OFF'}")

    device = get_device()

    # --- Run modules ---
    chmv2_results = None
    detection_results = None

    if config.run_chmv2:
        chmv2_results = run_chmv2(image_paths, config, device)

    if config.run_detections and chmv2_results:
        detection_results = run_detections(chmv2_results, image_paths, config)

    if config.run_features:
        run_feature_viz(image_paths, config, device)

    # --- Summary ---
    if chmv2_results or detection_results:
        write_summary(chmv2_results, detection_results, config)

    print("\n  Done! All outputs in: " + config.output_dir + "/")


if __name__ == "__main__":
    main()
