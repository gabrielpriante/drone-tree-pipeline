# drone-tree-pipeline

An end-to-end **drone → 3D → detections → inventory** pipeline for municipal (and utility-adjacent) tree work.

This repository is maintained by **Gabriel Priante** (Applied Data Scientist + Project Manager @ Frontline Gig) as the working home for turning **DJI Mini 5 Pro** field captures into geospatially-referenced tree detections and a clean inventory export.

At a high level:

1. **Capture** an orbital flight video or a folder of images from the drone
2. **Reconstruct** a georeferenced orthomosaic + point cloud with **OpenDroneMap (ODM)**
3. **Detect** individual tree crowns with **DeepForest**
4. **Measure** canopy geometry (and optionally height via DSM/DTM)
5. **Export** a structured inventory (`CSV` + `GeoJSON`) and a quick-look map

---

## What’s in this repo

Top-level contents (as of 2026-03-07):

- `pipeline.py`  Main end-to-end runner script
- `detection-images/`  Workspace folder (repo-managed)
- `detection-data/`  Workspace folder (repo-managed)
- `LICENSE`  Open-source license

> Note: There’s also a `.DS_Store` committed at repo root (macOS Finder metadata).

---

## The workflow (how `pipeline.py` actually runs)

The pipeline is intentionally linear, with each stage producing durable artifacts you can inspect, re-run, or swap out.

### Step 1 - Ingest the flight capture

**Inputs supported:**

- A **video file** (`.mp4`, `.mov`, `.lrv`, `.ts`) from a DJI orbital flight—the script will extract frames via **ffmpeg**
- A **folder of images** (JPEG/TIFF/DNG/PNG) from the drone SD card—the script will copy them into a project workspace

If an accompanying DJI `.SRT` file exists, it’s copied alongside the imagery because ODM can use it to improve georeferencing.

### Step 2 - Photogrammetry with OpenDroneMap (ODM)

ODM runs via Docker (`opendronemap/odm`) and generates core spatial products:

- Orthomosaic: `odm_orthophoto/odm_orthophoto.tif`
- Georeferenced point cloud: `odm_georeferencing/odm_georeferenced_model.laz`
- 3D mesh: `odm_texturing/odm_textured_model.obj`
- Elevation rasters when available: `odm_dem/dsm.tif` and `odm_dem/dtm.tif`

The script uses higher-quality defaults (e.g., high point-cloud quality and ~2 cm/pixel orthophoto) and includes guardrails for running locally on modest hardware (with a suggestion to use **WebODM Lightning** for larger datasets).

#### Cloud option

If you don’t want to run ODM locally, you can run:

```bash
python pipeline.py --input /path/to/video_or_images --project my_project --cloud
```

This prints WebODM Lightning instructions, then you re-run with `--skip-odm` after downloading outputs into the project folder.

### Step 3 - Tree crown detection with DeepForest

DeepForest is applied to the orthomosaic using tiled prediction (so large rasters still work):

- Predictions are saved to: `outputs/tree_detections.csv`
- A GIS-friendly export is attempted: `outputs/tree_detections.geojson`

Each detection is assigned a stable ID like `TREE_0001` and enriched with:

- Confidence score
- Crown width/length/area (meters)
- Center-point coordinates (derived from orthomosaic geotransform)

**Important:** If there’s no orthomosaic (or the orbital flight doesn’t produce enough top-down coverage), the script skips detections and points you to point-cloud-based workflows instead.

### Step 4 - Height extraction (optional)

When ODM produces both a DSM and DTM, the pipeline computes a **Canopy Height Model (CHM)**:

- `outputs/canopy_height_model.tif`

Then it samples the CHM around each detection and appends:

- `height_m` and `height_ft`

If DSM/DTM are missing (common in some orbital-only capture styles), height extraction is skipped and the README recommends trunk/height measurement from the point cloud using external tooling.

### Step 5 - Inventory assembly

The final inventory is saved to:

- `outputs/tree_inventory.csv`

It includes a clean column set appropriate for handoff to ops, GIS, or downstream analytics (IDs, crown metrics, optional height, survey date, and placeholder fields for health/species/notes).

### Step 6 - Quick-look map

A PNG overlay is generated for fast QA and stakeholder previews:

- `outputs/tree_detection_map.png`

---

## Outputs you’ll care about

Inside each project folder (created under `~/drone-projects/<project>/` by default):

- `outputs/tree_detections.csv`  Raw DeepForest boxes + metrics
- `outputs/tree_detections.geojson`  GIS export (QGIS / ArcGIS Online)
- `outputs/tree_inventory.csv`  Clean inventory for reporting/ops
- `outputs/tree_detection_map.png`  QA visualization
- `outputs/canopy_height_model.tif`  Height surface (only if DSM/DTM exist)

---

## Requirements

### System

- **Docker** (for ODM)
- **ffmpeg** (only if using video input)
- A Python 3 environment with geospatial deps

### Python packages

The script imports these at runtime (install however you prefer):

- `deepforest`
- `geopandas`
- `rasterio`
- `numpy`, `pandas`
- `matplotlib`
- `shapely`

---

## Quick start

### 1) Process a DJI video orbital flight

```bash
python pipeline.py --input ~/drone-footage/DJI_0001.MP4 --project my_first_tree
```

### 2) Process a folder of images

```bash
python pipeline.py --input ~/drone-footage/photos/ --project lot_survey
```

### 3) Extract frames only (sanity check before full processing)

```bash
python pipeline.py --input ~/drone-footage/DJI_0001.MP4 --project test --frames-only
```

### 4) Use cloud ODM (WebODM Lightning), then continue locally

```bash
python pipeline.py --input ~/drone-footage/photos/ --project lot_survey --cloud
# ... upload images, download results into the project folder ...
python pipeline.py --input ~/drone-projects/lot_survey --project lot_survey --skip-odm
```

---

## Where 3DFin fits

This repo’s automated path focuses on *crowns + geospatial inventory* via ODM + DeepForest.

For **trunk/DBH and detailed 3D measurements**, the workflow described in `pipeline.py` expects you to take the ODM point cloud and use **CloudCompare + 3DFin** (or equivalent tooling):

- Open: `odm_georeferencing/odm_georeferenced_model.laz`
- Measure trunks/DBH with the 3DFin plugin
- Export measurements and join them back to `tree_inventory.csv` (by location or a matching scheme)

---

## Notes / limitations

- Orbital flights are great for 3D context, but **DeepForest performs best with top-down, canopy-forward imagery**. If detections are low, adjust capture style or fine-tune DeepForest on local data.
- On consumer laptops, ODM quality settings may require reducing image count or using cloud processing for reliability.

---

## License

See `LICENSE`.