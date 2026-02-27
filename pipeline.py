#!/usr/bin/env python3
"""
Tree Inventory Pipeline
=======================
Takes drone orbital flight video/images of a tree and produces
a structured inventory with measurements and analysis.

Usage:
    python pipeline.py --input /path/to/video_or_images --project my_tree_scan

What this script does (in order):
    1. Extracts frames from video (if video input)
    2. Runs OpenDroneMap via Docker to generate point cloud + orthomosaic
    3. Runs DeepForest for tree crown detection
    4. Extracts height data from the canopy height model
    5. Assembles a structured tree inventory CSV
    6. Generates a visual summary map

Requirements:
    - Docker running with opendronemap/odm image pulled
    - Python packages: deepforest, geopandas, rasterio, laspy, numpy, pandas, matplotlib, Pillow
    - ffmpeg installed (for video input)
"""

import argparse
import subprocess
import sys
import os
import shutil
import json
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================
DRONE_PROJECTS_DIR = Path.home() / "drone-projects"
ODM_DOCKER_IMAGE = "opendronemap/odm"

# Frame extraction settings
FRAME_EXTRACTION_FPS = 2  # Extract 2 frames per second from video
# Increase this for more overlap (better reconstruction, more processing time)
# Decrease for faster processing (may reduce quality)

# ODM processing settings
ODM_OPTIONS = [
    "--dsm",                        # Generate Digital Surface Model
    "--dtm",                        # Generate Digital Terrain Model  
    "--pc-quality", "high",         # Point cloud quality (ultra, high, medium, low, lowest)
    "--orthophoto-resolution", "2",  # Orthophoto cm/pixel
    "--mesh-octree-depth", "12",    # Mesh detail level
    "--feature-quality", "high",    # Feature extraction quality
    "--min-num-features", "10000",  # Minimum features to extract per image
]


# ============================================================
# STEP 1: FRAME EXTRACTION (if video input)
# ============================================================
def extract_frames_from_video(video_path, output_dir, fps=FRAME_EXTRACTION_FPS):
    """
    Extract frames from drone orbital flight video.
    
    The DJI Mini 5 Pro records 4K video. We extract frames at a set interval
    to create overlapping images for photogrammetry. 2 FPS is a good starting
    point for orbital flights — adjust based on flight speed.
    
    If the video has an accompanying .SRT file with GPS data, copy that too
    as ODM can use it for georeferencing.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"STEP 1: Extracting frames from video")
    print(f"{'='*60}")
    print(f"  Video: {video_path}")
    print(f"  Output: {output_dir}")
    print(f"  Rate: {fps} frames per second")
    
    # Check if video exists
    if not video_path.exists():
        print(f"  ERROR: Video file not found: {video_path}")
        sys.exit(1)
    
    # Extract frames using ffmpeg
    output_pattern = str(output_dir / "frame_%05d.jpg")
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",           # High quality JPEG (1-31, lower = better)
        "-y",                   # Overwrite existing files
        output_pattern
    ]
    
    print(f"  Running ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: ffmpeg failed:\n{result.stderr[-500:]}")
        sys.exit(1)
    
    frame_count = len(list(output_dir.glob("frame_*.jpg")))
    print(f"  Extracted {frame_count} frames")
    
    # Check for SRT file (GPS subtitle data from DJI)
    srt_path = video_path.with_suffix(".SRT")
    if not srt_path.exists():
        srt_path = video_path.with_suffix(".srt")
    
    if srt_path.exists():
        dest = output_dir / srt_path.name
        shutil.copy2(srt_path, dest)
        print(f"  Copied GPS subtitle file: {srt_path.name}")
    else:
        print(f"  No .SRT file found (GPS data will come from image EXIF if available)")
    
    return frame_count


# ============================================================
# STEP 1B: COPY IMAGES (if image input instead of video)
# ============================================================
def copy_images(input_path, output_dir):
    """
    If the input is a folder of images (JPEGs/TIFFs/DNGs), copy them
    to the project images directory.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"STEP 1: Copying images to project")
    print(f"{'='*60}")
    
    extensions = {".jpg", ".jpeg", ".tif", ".tiff", ".dng", ".png"}
    count = 0
    
    for f in sorted(input_path.iterdir()):
        if f.suffix.lower() in extensions:
            shutil.copy2(f, output_dir / f.name)
            count += 1
    
    # Also copy any SRT files
    for f in input_path.glob("*.SRT"):
        shutil.copy2(f, output_dir / f.name)
    for f in input_path.glob("*.srt"):
        shutil.copy2(f, output_dir / f.name)
    
    print(f"  Copied {count} images to {output_dir}")
    return count


# ============================================================
# STEP 2: RUN OPENDRONEMAP
# ============================================================
def run_odm(project_dir):
    """
    Run OpenDroneMap via Docker to generate:
    - Orthomosaic (odm_orthophoto/odm_orthophoto.tif)
    - Point cloud (odm_georeferencing/odm_georeferenced_model.laz)
    - Digital Surface Model (odm_dem/dsm.tif)
    - Digital Terrain Model (odm_dem/dtm.tif)
    - 3D mesh (odm_texturing/odm_textured_model.obj)
    
    NOTE: On your 8GB M1 MacBook Air, this will work for small datasets
    (~50-100 images). For larger datasets, use WebODM Lightning (cloud).
    """
    project_dir = Path(project_dir)
    
    print(f"\n{'='*60}")
    print(f"STEP 2: Running OpenDroneMap")
    print(f"{'='*60}")
    print(f"  Project: {project_dir}")
    print(f"  This may take 15-60+ minutes depending on image count...")
    print(f"  Your Mac may get warm — that's normal.")
    print()
    
    # Check Docker is running
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ERROR: Docker is not running!")
        print("  Open Docker Desktop and wait for it to start, then try again.")
        sys.exit(1)
    
    # Check images exist
    images_dir = project_dir / "images"
    image_count = len([f for f in images_dir.iterdir() 
                       if f.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff", ".dng", ".png"}])
    
    if image_count == 0:
        print(f"  ERROR: No images found in {images_dir}")
        sys.exit(1)
    
    print(f"  Found {image_count} images to process")
    
    # Warn if dataset is large for 8GB RAM
    if image_count > 100:
        print(f"\n  ⚠️  WARNING: {image_count} images is a lot for 8GB RAM.")
        print(f"  Processing may fail or be very slow.")
        print(f"  Consider using WebODM Lightning (cloud) for datasets this large.")
        print(f"  Continuing anyway...\n")
    
    # Build docker command
    # Mount the project directory so ODM can read images and write outputs
    parent_dir = str(project_dir.parent)
    project_name = project_dir.name
    
    cmd = [
        "docker", "run",
        "-ti", "--rm",
        "-v", f"{parent_dir}:/datasets",
        ODM_DOCKER_IMAGE,
        "--project-path", "/datasets",
        project_name,
    ] + ODM_OPTIONS
    
    print(f"  Running: {' '.join(cmd[:8])}...")
    print(f"  ODM options: {' '.join(ODM_OPTIONS)}")
    print()
    
    # Run ODM
    start_time = datetime.now()
    result = subprocess.run(cmd)
    elapsed = datetime.now() - start_time
    
    if result.returncode != 0:
        print(f"\n  ERROR: ODM processing failed (exit code {result.returncode})")
        print(f"  Check the output above for details.")
        print(f"  Common fixes:")
        print(f"    - Reduce image count")
        print(f"    - Change --pc-quality to 'medium' or 'low'")
        print(f"    - Allocate more RAM to Docker in Docker Desktop settings")
        print(f"    - Use WebODM Lightning for cloud processing")
        sys.exit(1)
    
    print(f"\n  ODM completed in {elapsed}")
    
    # Verify outputs exist
    outputs = {
        "Orthomosaic": project_dir / "odm_orthophoto" / "odm_orthophoto.tif",
        "Point Cloud": project_dir / "odm_georeferencing" / "odm_georeferenced_model.laz",
        "3D Mesh": project_dir / "odm_texturing" / "odm_textured_model.obj",
    }
    
    # DSM and DTM are in odm_dem/ folder
    dsm_path = project_dir / "odm_dem" / "dsm.tif"
    dtm_path = project_dir / "odm_dem" / "dtm.tif"
    if dsm_path.exists():
        outputs["DSM"] = dsm_path
    if dtm_path.exists():
        outputs["DTM"] = dtm_path
    
    print(f"\n  Generated outputs:")
    for name, path in outputs.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"    ✓ {name}: {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"    ✗ {name}: NOT FOUND")
    
    return outputs


# ============================================================
# STEP 3: TREE DETECTION WITH DEEPFOREST
# ============================================================
def detect_trees(orthomosaic_path, output_dir):
    """
    Run DeepForest on the orthomosaic to detect individual tree crowns.
    Returns bounding boxes with confidence scores and GPS coordinates.
    """
    print(f"\n{'='*60}")
    print(f"STEP 3: Detecting trees with DeepForest")
    print(f"{'='*60}")
    
    try:
        from deepforest import main as df_main
        import geopandas as gpd
        import rasterio
        from rasterio.transform import xy
    except ImportError as e:
        print(f"  ERROR: Missing package: {e}")
        print(f"  Run: pip install deepforest geopandas rasterio")
        sys.exit(1)
    
    orthomosaic_path = Path(orthomosaic_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not orthomosaic_path.exists():
        print(f"  ERROR: Orthomosaic not found: {orthomosaic_path}")
        print(f"  If ODM didn't generate one, the orbital flight may not have")
        print(f"  produced enough top-down coverage for an orthomosaic.")
        print(f"  Skipping tree detection — you can still use the point cloud")
        print(f"  with 3DFin/CloudCompare for trunk measurements.")
        return None
    
    print(f"  Loading DeepForest model...")
    model = df_main.deepforest()
    model.use_release()
    
    print(f"  Running prediction on orthomosaic...")
    print(f"  (This may take a few minutes)")
    
    # Predict on tiles (handles large orthomosaics)
    boxes = model.predict_tile(
        raster_path=str(orthomosaic_path),
        patch_size=400,
        patch_overlap=0.05,
        return_plot=False
    )
    
    if boxes is None or len(boxes) == 0:
        print(f"  No trees detected. This could mean:")
        print(f"    - The orthomosaic is mostly side-view (not enough top-down)")
        print(f"    - The resolution needs adjustment")
        print(f"    - Try fine-tuning DeepForest with local training data")
        return None
    
    print(f"  Detected {len(boxes)} trees")
    
    # Convert pixel coordinates to GPS coordinates
    print(f"  Converting to GPS coordinates...")
    with rasterio.open(str(orthomosaic_path)) as src:
        transform = src.transform
        crs = src.crs
        
        # Calculate center points of each bounding box
        boxes["center_x_px"] = (boxes["xmin"] + boxes["xmax"]) / 2
        boxes["center_y_px"] = (boxes["ymin"] + boxes["ymax"]) / 2
        
        # Convert pixel to geographic coordinates
        lons, lats = [], []
        for _, row in boxes.iterrows():
            lon, lat = xy(transform, int(row["center_y_px"]), int(row["center_x_px"]))
            lons.append(lon)
            lats.append(lat)
        
        boxes["longitude"] = lons
        boxes["latitude"] = lats
        
        # Calculate crown dimensions in real-world units
        pixel_size_x = abs(transform[0])  # meters per pixel
        boxes["crown_width_m"] = (boxes["xmax"] - boxes["xmin"]) * pixel_size_x
        boxes["crown_height_m"] = (boxes["ymax"] - boxes["ymin"]) * pixel_size_x
        boxes["crown_area_m2"] = boxes["crown_width_m"] * boxes["crown_height_m"]
    
    # Add tree IDs
    boxes.insert(0, "tree_id", [f"TREE_{i+1:04d}" for i in range(len(boxes))])
    
    # Save results
    csv_path = output_dir / "tree_detections.csv"
    boxes.to_csv(csv_path, index=False)
    print(f"  Saved detections to: {csv_path}")
    
    # Save as GeoJSON for GIS use
    try:
        from shapely.geometry import box as shapely_box
        geometries = [
            shapely_box(row["longitude"] - row["crown_width_m"]/2,
                       row["latitude"] - row["crown_height_m"]/2,
                       row["longitude"] + row["crown_width_m"]/2,
                       row["latitude"] + row["crown_height_m"]/2)
            for _, row in boxes.iterrows()
        ]
        gdf = gpd.GeoDataFrame(boxes, geometry=geometries, crs=crs)
        geojson_path = output_dir / "tree_detections.geojson"
        gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"  Saved GeoJSON to: {geojson_path}")
    except Exception as e:
        print(f"  Warning: Could not create GeoJSON: {e}")
    
    return boxes


# ============================================================
# STEP 4: HEIGHT EXTRACTION FROM CANOPY HEIGHT MODEL
# ============================================================
def extract_heights(tree_detections, dsm_path, dtm_path, output_dir):
    """
    Calculate tree heights by sampling the DSM and DTM at each tree location.
    Tree height = DSM value (top of canopy) - DTM value (ground level)
    """
    print(f"\n{'='*60}")
    print(f"STEP 4: Extracting tree heights")
    print(f"{'='*60}")
    
    import rasterio
    import numpy as np
    
    dsm_path = Path(dsm_path)
    dtm_path = Path(dtm_path)
    output_dir = Path(output_dir)
    
    if not dsm_path.exists() or not dtm_path.exists():
        print(f"  DSM or DTM not found — skipping height extraction.")
        print(f"  Heights require top-down flight coverage to generate elevation models.")
        print(f"  For orbital-only flights, use 3DFin with the point cloud instead.")
        return tree_detections
    
    print(f"  Reading DSM: {dsm_path}")
    print(f"  Reading DTM: {dtm_path}")
    
    with rasterio.open(str(dsm_path)) as dsm_src:
        dsm_data = dsm_src.read(1)
        dsm_transform = dsm_src.transform
        
    with rasterio.open(str(dtm_path)) as dtm_src:
        dtm_data = dtm_src.read(1)
    
    # Calculate Canopy Height Model
    chm = dsm_data - dtm_data
    chm[chm < 0] = 0  # Remove negative values (artifacts)
    
    # Save CHM as a raster
    chm_path = output_dir / "canopy_height_model.tif"
    with rasterio.open(str(dsm_path)) as src:
        profile = src.profile
        with rasterio.open(str(chm_path), "w", **profile) as dst:
            dst.write(chm, 1)
    print(f"  Saved CHM to: {chm_path}")
    
    # Sample heights at each tree location
    if tree_detections is not None and len(tree_detections) > 0:
        heights = []
        for _, row in tree_detections.iterrows():
            try:
                # Convert geographic coords to pixel coords in DSM
                col, row_px = ~dsm_transform * (row["longitude"], row["latitude"])
                col, row_px = int(col), int(row_px)
                
                if 0 <= row_px < chm.shape[0] and 0 <= col < chm.shape[1]:
                    # Sample a small window around the tree center and take the max
                    window_size = 3
                    r_start = max(0, row_px - window_size)
                    r_end = min(chm.shape[0], row_px + window_size + 1)
                    c_start = max(0, col - window_size)
                    c_end = min(chm.shape[1], col + window_size + 1)
                    
                    window = chm[r_start:r_end, c_start:c_end]
                    height = float(np.nanmax(window))
                    heights.append(round(height, 2))
                else:
                    heights.append(None)
            except Exception:
                heights.append(None)
        
        tree_detections["height_m"] = heights
        tree_detections["height_ft"] = [
            round(h * 3.28084, 1) if h is not None else None 
            for h in heights
        ]
        
        valid_heights = [h for h in heights if h is not None and h > 0]
        if valid_heights:
            print(f"  Heights extracted for {len(valid_heights)} trees")
            print(f"  Range: {min(valid_heights):.1f}m - {max(valid_heights):.1f}m")
            print(f"  Average: {sum(valid_heights)/len(valid_heights):.1f}m")
    
    return tree_detections


# ============================================================
# STEP 5: ASSEMBLE FINAL INVENTORY
# ============================================================
def assemble_inventory(tree_detections, project_dir, output_dir):
    """
    Combine all extracted data into a final structured tree inventory.
    """
    print(f"\n{'='*60}")
    print(f"STEP 5: Assembling tree inventory")
    print(f"{'='*60}")
    
    import pandas as pd
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if tree_detections is None or len(tree_detections) == 0:
        print(f"  No tree detections to assemble.")
        print(f"  For orbital flights without top-down coverage:")
        print(f"  → Open the point cloud (.laz) in CloudCompare")
        print(f"  → Run 3DFin plugin for trunk measurements")
        print(f"  → Export measurements to CSV")
        return
    
    # Select and rename columns for the final inventory
    inventory_columns = {
        "tree_id": "Tree ID",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "score": "Detection Confidence",
        "crown_width_m": "Crown Width (m)",
        "crown_height_m": "Crown Length (m)",
        "crown_area_m2": "Crown Area (m²)",
    }
    
    # Add height columns if available
    if "height_m" in tree_detections.columns:
        inventory_columns["height_m"] = "Tree Height (m)"
        inventory_columns["height_ft"] = "Tree Height (ft)"
    
    # Build final dataframe
    available_cols = {k: v for k, v in inventory_columns.items() 
                      if k in tree_detections.columns}
    inventory = tree_detections[list(available_cols.keys())].rename(columns=available_cols)
    
    # Add metadata columns
    inventory["Survey Date"] = datetime.now().strftime("%Y-%m-%d")
    inventory["Health Status"] = "Not Assessed"  # Placeholder for DeepForest alive/dead model
    inventory["Species"] = "Not Identified"       # Placeholder for future species classification
    inventory["Notes"] = ""
    
    # Sort by tree ID
    inventory = inventory.sort_values("Tree ID").reset_index(drop=True)
    
    # Save
    csv_path = output_dir / "tree_inventory.csv"
    inventory.to_csv(csv_path, index=False)
    print(f"  Saved inventory: {csv_path}")
    print(f"  Total trees: {len(inventory)}")
    
    # Print summary
    print(f"\n  INVENTORY SUMMARY")
    print(f"  {'─'*40}")
    print(f"  Trees detected: {len(inventory)}")
    
    if "Tree Height (m)" in inventory.columns:
        heights = inventory["Tree Height (m)"].dropna()
        if len(heights) > 0:
            print(f"  Height range: {heights.min():.1f}m - {heights.max():.1f}m")
            print(f"  Average height: {heights.mean():.1f}m")
    
    crowns = inventory["Crown Area (m²)"].dropna()
    if len(crowns) > 0:
        print(f"  Total canopy area: {crowns.sum():.1f} m²")
        print(f"  Average crown area: {crowns.mean():.1f} m²")
    
    return inventory


# ============================================================
# STEP 6: GENERATE VISUAL SUMMARY
# ============================================================
def generate_summary_map(tree_detections, orthomosaic_path, output_dir):
    """
    Create a visual map showing detected trees overlaid on the orthomosaic.
    """
    print(f"\n{'='*60}")
    print(f"STEP 6: Generating visual summary")
    print(f"{'='*60}")
    
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    
    output_dir = Path(output_dir)
    orthomosaic_path = Path(orthomosaic_path)
    
    if not orthomosaic_path.exists():
        print(f"  No orthomosaic available for map generation.")
        return
    
    if tree_detections is None or len(tree_detections) == 0:
        print(f"  No detections to map.")
        return
    
    try:
        import rasterio
        from rasterio.plot import show
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        with rasterio.open(str(orthomosaic_path)) as src:
            show(src, ax=ax)
        
        # Plot tree detections
        for _, row in tree_detections.iterrows():
            rect = patches.Rectangle(
                (row["xmin"], row["ymin"]),
                row["xmax"] - row["xmin"],
                row["ymax"] - row["ymin"],
                linewidth=1.5,
                edgecolor="lime",
                facecolor="none",
                alpha=0.8
            )
            ax.add_patch(rect)
        
        ax.set_title(f"Tree Detection Results — {len(tree_detections)} trees detected",
                     fontsize=14, fontweight="bold")
        
        map_path = output_dir / "tree_detection_map.png"
        plt.savefig(str(map_path), dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"  Saved map: {map_path}")
        
    except Exception as e:
        print(f"  Warning: Could not generate map: {e}")


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Tree Inventory Pipeline — From drone video to structured data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  
  # Process a video of an orbital flight around a tree
  python pipeline.py --input ~/drone-footage/DJI_0001.MP4 --project my_first_tree

  # Process a folder of images (JPEGs from the drone SD card)
  python pipeline.py --input ~/drone-footage/photos/ --project lot_survey

  # Skip ODM (if you already processed with WebODM Lightning)
  python pipeline.py --input ~/drone-projects/lot_survey --project lot_survey --skip-odm

  # Extract frames only (to check quality before full processing)
  python pipeline.py --input ~/drone-footage/DJI_0001.MP4 --project test --frames-only
        """
    )
    
    parser.add_argument("--input", required=True,
                       help="Path to video file (.mp4/.mov) OR folder of images")
    parser.add_argument("--project", required=True,
                       help="Project name (creates folder in ~/drone-projects/)")
    parser.add_argument("--fps", type=float, default=FRAME_EXTRACTION_FPS,
                       help=f"Frames per second to extract from video (default: {FRAME_EXTRACTION_FPS})")
    parser.add_argument("--skip-odm", action="store_true",
                       help="Skip ODM processing (use if you already have outputs)")
    parser.add_argument("--frames-only", action="store_true",
                       help="Only extract frames from video, don't process")
    parser.add_argument("--cloud", action="store_true",
                       help="Print instructions for cloud processing instead of local ODM")
    
    args = parser.parse_args()
    
    # Setup project directory
    project_dir = DRONE_PROJECTS_DIR / args.project
    images_dir = project_dir / "images"
    output_dir = project_dir / "outputs"
    
    print(f"\n{'#'*60}")
    print(f"  TREE INVENTORY PIPELINE")
    print(f"  Project: {args.project}")
    print(f"  Input: {args.input}")
    print(f"  Directory: {project_dir}")
    print(f"{'#'*60}")
    
    input_path = Path(args.input)
    
    # ---- STEP 1: Get images into project ----
    if input_path.is_file() and input_path.suffix.lower() in {".mp4", ".mov", ".lrv", ".ts"}:
        # Video input — extract frames
        extract_frames_from_video(input_path, images_dir, fps=args.fps)
    elif input_path.is_dir():
        # Check if this is an already-processed ODM project
        if (input_path / "odm_orthophoto").exists():
            print(f"\n  Detected existing ODM outputs in {input_path}")
            project_dir = input_path
            images_dir = project_dir / "images"
            output_dir = project_dir / "outputs"
            args.skip_odm = True
        else:
            # Folder of images
            copy_images(input_path, images_dir)
    else:
        print(f"  ERROR: Input must be a video file or folder of images")
        print(f"  Got: {input_path}")
        sys.exit(1)
    
    if args.frames_only:
        print(f"\n  Frames extracted. Review them in: {images_dir}")
        print(f"  When ready, run again without --frames-only to process.")
        return
    
    # ---- STEP 2: Run ODM ----
    if args.cloud:
        print(f"\n{'='*60}")
        print(f"CLOUD PROCESSING MODE")
        print(f"{'='*60}")
        print(f"  Your images are ready in: {images_dir}")
        print(f"")
        print(f"  To process with WebODM Lightning:")
        print(f"    1. Go to https://webodm.net")
        print(f"    2. Create an account and start a new task")
        print(f"    3. Upload all images from: {images_dir}")
        print(f"    4. Enable DSM and DTM in processing options")
        print(f"    5. Download results and extract to: {project_dir}")
        print(f"    6. Re-run this script with --skip-odm flag")
        print(f"")
        print(f"  Command for after cloud processing:")
        print(f"    python pipeline.py --input {project_dir} --project {args.project} --skip-odm")
        return
    
    if not args.skip_odm:
        odm_outputs = run_odm(project_dir)
    
    # ---- STEP 3: Detect trees ----
    orthomosaic_path = project_dir / "odm_orthophoto" / "odm_orthophoto.tif"
    tree_detections = detect_trees(orthomosaic_path, output_dir)
    
    # ---- STEP 4: Extract heights ----
    dsm_path = project_dir / "odm_dem" / "dsm.tif"
    dtm_path = project_dir / "odm_dem" / "dtm.tif"
    tree_detections = extract_heights(tree_detections, dsm_path, dtm_path, output_dir)
    
    # ---- STEP 5: Assemble inventory ----
    inventory = assemble_inventory(tree_detections, project_dir, output_dir)
    
    # ---- STEP 6: Generate map ----
    generate_summary_map(tree_detections, orthomosaic_path, output_dir)
    
    # ---- DONE ----
    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'#'*60}")
    print(f"")
    print(f"  Your outputs are in: {output_dir}")
    print(f"")
    print(f"  FILES GENERATED:")
    print(f"    tree_inventory.csv         — Structured tree data")
    print(f"    tree_detections.csv        — Raw detection data")
    print(f"    tree_detections.geojson    — For ArcGIS Online / QGIS")
    print(f"    tree_detection_map.png     — Visual summary")
    print(f"    canopy_height_model.tif    — Height raster (if DSM/DTM available)")
    print(f"")
    print(f"  NEXT STEPS:")
    print(f"    1. Open the point cloud in CloudCompare:")
    print(f"       {project_dir / 'odm_georeferencing' / 'odm_georeferenced_model.laz'}")
    print(f"    2. Run 3DFin plugin in CloudCompare for trunk measurements (DBH)")
    print(f"    3. Upload tree_detections.geojson to ArcGIS Online")
    print(f"    4. Review tree_inventory.csv and add species/health notes")
    print(f"")


if __name__ == "__main__":
    main()
