# Pipeline Workflow & Execution Guide

## Overview

This guide walks through the complete 7-phase pipeline for creating a labeled traffic density dataset. The pipeline uses polygon filtering and lane-aware traffic classification for accurate labels.

**Dataset Target**: 4,000+ labeled images for binary traffic classification (traffic_present / traffic_absent)

---

## Complete Pipeline

### Phase 1a: Structure Discovery
**Purpose**: Map the Azure container structure without listing all 18M blobs.

```bash
python main.py run --phase 1a
```

**What it does:**
- Uses hierarchical listing with delimiter to discover virtual directories
- Maps cameras, years, months, and days available in Azure
- **Output**: `inventory.json`
- **Time**: 5-15 minutes

---

### Phase 1b: Stratified Sampling
**Purpose**: Select 700 images per camera with stratified distribution.

```bash
python main.py run --phase 1b
```

**What it does:**
- Equal distribution across 5 time buckets × 4 seasons
- Time buckets: Night, Morning, Midday, Afternoon, Evening
- Season buckets: Winter, Spring, Summer, Autumn
- **Output**: `sample_manifest.json`
- **Time**: 10-30 minutes

**Sampling Matrix (per camera):**
```
                Winter  Spring  Summer  Autumn
Night            35      35      35      35    = 140
Morning          35      35      35      35    = 140
Midday           35      35      35      35    = 140
Afternoon        35      35      35      35    = 140
Evening          35      35      35      35    = 140
              = 175    175     175     175    = 700 total
```

---

### Phase 1c: Parallel Download
**Purpose**: Download selected images from Azure to local disk.

```bash
python main.py run --phase 1c --workers 4
```

**What it does:**
- Downloads images in parallel (configurable workers)
- Resumes from `download_progress.json` if interrupted
- **Output**: Downloaded images in `raw/{CAMERA}/`
- **Time**: 1.5-3 hours (depends on network speed)

---

### Phase 2: YOLO Analysis
**Purpose**: Run YOLO on all downloaded images to detect vehicles.

```bash
python main.py run --phase 2 --device mps --confidence 0.25
```

**What it does:**
- Runs YOLO11n on all images
- Filters detections using lane polygons (if available)
- Counts vehicles only inside lane boundaries
- Classifies as `traffic_present` (1+ vehicles) or `traffic_absent` (0 vehicles)
- **Output**: `yolo_results.json`
- **Time**: ~75 minutes with MPS

**Device options:**
- `mps`: Apple Silicon GPU (recommended for M-series Macs)
- `cuda`: NVIDIA GPU
- `cpu`: CPU fallback

---

### Phase 2a: Lane Annotation (Interactive)
**Purpose**: Define lane polygons for each camera to filter YOLO detections.

```bash
python main.py run --phase 2a --resume
```

**What it does:**
- Shows reference image for each camera
- You draw polygon around traffic lanes
- Excludes parked vehicles, staff areas, buildings
- **Output**: `lane_polygons.json`
- **Time**: 2-3 hours (~3-5 min per camera)

**Controls:**
- `A`: Accept auto-suggestion
- `M`: Modify suggestion
- `D`: Draw manual
- `N`: Next camera (save)
- `R`: Reset polygon
- `X`: Toggle empty reference view
- `Q`: Quit and save progress

**Example output:**
```json
{
  "cameras": {
    "BATROVCI_I": {
      "reference_image": "raw/BATROVCI_I/...",
      "lane_count": 4,
      "polygons": [{
        "polygon": [[120, 150], [680, 140], [720, 580], [80, 600]]
      }]
    }
  }
}
```

---

### Phase 3: Binary Selection
**Purpose**: Select 6,000 balanced images (3K traffic_present, 3K traffic_absent).

```bash
python main.py run --phase 3 --target-per-class 3000
```

**What it does:**
- Balances across cameras, time, and seasons
- Prioritizes diversity (different times/dates per camera)
- Uses neighbor expansion to find similar timestamps
- **Output**: `binary_selection.json`
- **Time**: 1-2 minutes

**Selection strategy:**
- Start with seed samples from Phase 1b
- For each seed, find temporal neighbors (±30 minutes)
- Balance across traffic_present and traffic_absent
- Distribute evenly across all cameras

---

### Phase 4: Label Review (Interactive)
**Purpose**: Manual review and correction of binary labels.

```bash
python main.py run --phase 4 --resume
# or
python main.py run --phase 4 --review-all --resume
```

**What it does:**
- Shows images with YOLO predictions
- You confirm or correct labels
- Option to review only borderline cases or all images
- **Output**: `binary_labeled/{traffic_present,traffic_absent,uncertain}/`
- **Time**:
  - Borderline only: ~1 hour (~1,300 images)
  - All images: 4-6 hours (~6,000 images)

**Controls:**
- `1`: Label as traffic_present
- `2`: Label as traffic_absent
- `Enter`: Accept current label
- `U`: Mark as uncertain (exclude)
- `N`: Next image
- `P`: Previous image
- `Q`: Quit and save

**Review strategies:**
- `--borderline-only`: Review only uncertain predictions (faster)
- `--review-all`: Review all 6,000 images (highest quality)

---

### Phase 5: Exclusion Zones (Interactive)
**Purpose**: Mark regions where YOLO consistently fails to detect vehicles.

```bash
python main.py run --phase 5 --resume
```

**What it does:**
- Shows reference image for each camera
- You draw ONE polygon per camera marking blind spots
- These regions will be cropped for additional training
- **Output**: `yolo_failure_regions.json`
- **Time**: 2-3 hours (~3-5 min per camera)

**Controls:**
- Left-click: Add polygon vertex
- Right-click: Remove last vertex
- `N`: Next camera (save)
- `R`: Reset polygon
- `S`: Skip camera (no blind spots)
- `Q`: Quit and save

**Purpose:** These regions will be cropped and labeled separately to train the CNN on areas where YOLO fails.

---

### Phase 6: Crop Failure Regions
**Purpose**: Crop exclusion zones to 128×128 images with label inheritance.

```bash
python main.py run --phase 6 --resume
```

**What it does:**
- Crops exclusion zones from labeled images
- Inherits labels from full image
- Resizes to 128×128 pixels
- Applies polygon mask
- **Output**: `binary_crops/{traffic_present,traffic_absent}/`
- **Time**: 15-20 minutes

---

### Phase 7: Train/Val Split
**Purpose**: Create 80/20 stratified train/validation split.

```bash
python main.py run --phase 7 --train-ratio 0.8
```

**What it does:**
- Combines labeled full images and crops
- Creates stratified split maintaining class balance
- Uses reproducible random seed
- **Output**: `binary_final/train/` and `binary_final/val/`
- **Time**: 1-2 minutes

**Final dataset structure:**
```
binary_final/
├── train/                    # 80%
│   ├── traffic_present/
│   └── traffic_absent/
└── val/                      # 20%
    ├── traffic_present/
    └── traffic_absent/
```

---

## Quick Start Commands

### Run entire pipeline
```bash
python main.py run --all --resume
```

### Run from specific phase
```bash
python main.py run --from-phase 3
```

### Check pipeline status
```bash
python main.py status
```

### Validate phase output
```bash
python main.py validate --phase 4
python main.py validate --all
```

### Reset a phase
```bash
python main.py reset --phase 5
```

---

## Resume & Error Handling

All phases support `--resume` flag to continue from saved progress:

```bash
python main.py run --phase 2 --resume
```

### What gets saved:
- **Phase 1c**: Download progress after each batch
- **Phase 2**: YOLO results after each image
- **Phase 2a**: Lane polygons after each camera
- **Phase 4**: Review progress after every 10 images
- **Phase 5**: Exclusion zones after each camera
- **Phase 6**: Crop progress after each batch

### If a phase crashes:
1. Check logs: `tail -f traffic_dataset/pipeline.log`
2. Re-run with `--resume` flag
3. Progress will be preserved

---

## Expected Timeline

**Non-interactive (automated):**
- Phase 1a: 5-15 min
- Phase 1b: 10-30 min
- Phase 1c: 1.5-3 hours
- Phase 2: ~75 min
- Phase 3: 1-2 min
- Phase 6: 15-20 min
- Phase 7: 1-2 min

**Interactive (requires human):**
- Phase 2a: 2-3 hours (lane annotation)
- Phase 4: 1-6 hours (label review)
- Phase 5: 2-3 hours (exclusion zones)

**Total time:** ~6-12 hours (depends on thoroughness of Phase 4)

---

## Advanced Options

### YOLO Configuration
```bash
# Use different YOLO model
python main.py run --phase 2 --model yolo11m --device mps

# Adjust confidence threshold
python main.py run --phase 2 --confidence 0.3 --device mps
```

### Download Configuration
```bash
# Increase parallel workers
python main.py run --phase 1c --workers 8
```

### Selection Configuration
```bash
# Adjust target images per class
python main.py run --phase 3 --target-per-class 4000
```

### Split Configuration
```bash
# Change train/val ratio
python main.py run --phase 7 --train-ratio 0.85
```

---

## Troubleshooting

### YOLO runs out of memory
```bash
# Use CPU instead of GPU
python main.py run --phase 2 --device cpu
```

### Download is very slow
```bash
# Increase workers (if network bandwidth allows)
python main.py run --phase 1c --workers 8
```

### Want to see detailed progress
```bash
# Tail the pipeline log
tail -f traffic_dataset/pipeline.log
```

### Need to redo a phase
```bash
# Reset the phase (creates backup)
python main.py reset --phase 5

# Re-run the phase
python main.py run --phase 5 --resume
```

---

## Notes

- **Azure credentials required**: Phases 1a, 1b, 1c need Azure access
- **Interactive phases**: 2a, 4, 5 require human input
- **Auto-save**: Interactive phases save progress automatically
- **Resume safe**: All phases can be safely interrupted and resumed
- **Validation**: Use `python main.py validate --all` to check data integrity

---

## Next Steps

After completing the pipeline:
1. Train CNN model on labeled dataset
2. Evaluate on validation set
3. Deploy for real-time traffic classification
4. Iterate and improve based on validation results
