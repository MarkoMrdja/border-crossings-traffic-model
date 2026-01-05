# Pipeline Execution Guide

## Current Status

✅ **Phase 1 Complete**: Images downloaded (22,400 images from 29 cameras)
✅ **Phase 4b Complete**: Lane polygons defined for 28 cameras
🚫 **Excluded**: 3 cameras with corrupted data (VATIN_U, VATIN_I, GOSTUN_I)

---

## Your Workflow Requirements

Based on your description, here's what you want to do:

1. **Phase 2b (Filtered YOLO)**: Run YOLO with lane polygon filtering to auto-label images
2. **(Optional) Phase 2 Review**: Review a subset of auto-labeled images to verify accuracy
3. **Phase 3 (Balanced Selection)**: Select 500 balanced images per camera
4. **Phase 4d (Exclusion Zones)**: Define regions where YOLO fails (blind spots)
5. **Phase 5 (Crop)**: Crop the ROI regions to 64×64 images
6. **Phase 6 (Label Verification)**: Manually verify all cropped labels
7. **Phase 7 (Train/Val Split)**: Create final 80/20 split

---

## Step-by-Step Execution

### Step 0: Clean Up Old Results

First, remove old pipeline results to start fresh:

```bash
# Run cleanup script (creates backup first)
python3 cleanup_for_phase2_rerun.py
```

**What this does:**
- Removes old `balanced_selection.json`, `roi_references.json`, etc.
- Cleans `crops/`, `labeled/`, `final/` directories
- Creates timestamped backup of removed files
- **Preserves** raw images and your lane polygons

---

### Step 1: Run Filtered YOLO Analysis (Phase 2b)

Run YOLO with lane-aware filtering using your polygons:

```bash
# Option A: Using main.py (recommended)
python3 main.py run --phase 2b --device mps --resume

# Option B: Direct execution
python3 dataset_pipeline/phase2b_yolo_filtered.py --device mps --resume
```

**What this does:**
- Loads your lane polygons from `lane_polygons.json`
- Runs YOLO11m on all images
- Filters detections to only count vehicles INSIDE lane polygons
- Detects active lanes dynamically (X-coordinate clustering)
- Classifies traffic using `vehicles_per_lane` metric
- Outputs: `traffic_dataset/yolo_results_filtered.json`
- **Time**: ~75-90 minutes (22,400 images with MPS)

**Excluded cameras:** VATIN_U, VATIN_I, GOSTUN_I will be automatically skipped.

---

### Step 2 (Optional): Review YOLO Labels

If you want to manually review and correct a subset of YOLO auto-labels:

```bash
# Option A: Using main.py
python3 main.py run --phase 2 --resume

# Option B: Direct execution
python3 dataset_pipeline/phase2_review.py --resume
```

**What this does:**
- Shows you images with YOLO bounding boxes
- Displays current traffic classification
- You can override labels (1=empty, 2=light, 3=moderate, 4=heavy)
- Automatically saves progress every 10 images
- Auto-resumes if interrupted
- Outputs: Modified `yolo_results_filtered.json`
- **Time**: 2-3 hours (optional - you can skip borderline cases)

**Controls:**
- `1-4`: Assign traffic level
- `Enter`: Confirm current label
- `B`: Go back
- `S`: Skip to next
- `Q`: Save and quit

**Note:** This is **optional**. Only review if you want to verify borderline classifications.

---

### Step 3: Balanced Selection

Select 500 balanced images per camera (125 per traffic level):

```bash
# Option A: Using main.py
python3 main.py run --phase 3 --resume

# Option B: Direct execution
python3 dataset_pipeline/phase3_balance.py --resume
```

**What this does:**
- Loads YOLO results (filtered version if Phase 2b was run)
- Selects 500 images per camera with balanced traffic levels
- Uses priority-based selection (heavy → moderate → light → empty)
- Outputs: `balanced_selection.json`, `roi_references.json`
- **Time**: <1 minute

**Result:** 14,000 images selected (500 per camera × 28 cameras)

---

### Step 4: Define YOLO Exclusion Zones (Phase 4d)

**Interactive tool** to mark regions where YOLO fails to detect vehicles:

```bash
python3 dataset_pipeline/phase4d_exclusion_zones.py
```

**What this does:**
- Shows reference image for each camera
- You draw polygons around areas where YOLO misses vehicles
- Examples: distant highway sections, tricky angles, glare spots
- Outputs: `yolo_failure_regions.json`
- **Time**: ~1-2 hours (interactive, 28 cameras)

**Controls:**
- Left-click: Add polygon vertex
- Right-click: Complete polygon
- `R`: Reset polygon
- `N`: Next camera
- `S`: Skip camera (no failure regions)
- `Q`: Save and quit

**Purpose:** These regions will be cropped separately for additional training data.

---

### Step 5: Crop ROI Regions

Crop lane regions and failure zones to 64×64 images:

```bash
# Option A: Using main.py
python3 main.py run --phase 5 --resume

# Option B: Direct execution
python3 dataset_pipeline/phase5_crop.py --resume
```

**What this does:**
- Loads lane polygons and balanced selection
- Crops lane regions → saves to `crops/likely_{level}/`
- Crops failure regions → saves to `failure_region_crops/` (if Phase 4d was run)
- Resizes all crops to 64×64 pixels
- Applies polygon mask (black outside ROI)
- Outputs: 14,000 cropped images in `crops/`
- **Time**: 5-10 minutes

---

### Step 6: Manual Label Verification

**Interactive tool** to verify and correct all cropped labels:

```bash
# Option A: Using main.py
python3 main.py run --phase 6 --resume

# Option B: Direct execution
python3 dataset_pipeline/phase6_label_tool.py --resume
```

**What this does:**
- Shows 64×64 crops (scaled up for visibility)
- Displays YOLO-assigned traffic level
- You confirm or override each label
- Priority-based processing (heavy → moderate → light → empty)
- Automatically saves progress every 10 images
- Auto-resumes if interrupted
- Outputs: Images organized in `labeled/{class}/`
- **Time**: 2-3 hours (interactive, 14,000 images)

**Controls:**
- `1-4`: Assign traffic level (1=empty, 2=light, 3=moderate, 4=heavy)
- `Enter`: Confirm current label
- `B`: Go back
- `S`: Skip to next
- `U`: Undo last
- `Q`: Save and quit

**Tip:** This is the critical quality control step. Take your time!

---

### Step 7: Train/Validation Split

Create final 80/20 stratified split:

```bash
# Option A: Using main.py
python3 main.py run --phase 7 --resume

# Option B: Direct execution
python3 dataset_pipeline/phase7_split.py --resume
```

**What this does:**
- Loads labeled images
- Creates 80/20 split with stratification (maintains class balance)
- Uses reproducible random seed
- Outputs:
  - `final/train/{class}/` - 11,200 images (80%)
  - `final/val/{class}/` - 2,800 images (20%)
  - `split_manifest.json` - Split statistics
- **Time**: <1 minute

---

## Quick Command Summary

Run all phases in sequence (with auto-resume):

```bash
# Clean up first
python3 cleanup_for_phase2_rerun.py

# Run pipeline
python3 main.py run --phase 2b --device mps --resume    # YOLO filtering (~75 min)
# (Optional) python3 main.py run --phase 2 --resume     # Review labels (2-3 hours)
python3 main.py run --phase 3 --resume                  # Balanced selection (<1 min)
python3 dataset_pipeline/phase4d_exclusion_zones.py     # Define failure zones (1-2 hours)
python3 main.py run --phase 5 --resume                  # Crop regions (5-10 min)
python3 main.py run --phase 6 --resume                  # Label verification (2-3 hours)
python3 main.py run --phase 7 --resume                  # Train/val split (<1 min)
```

**Total time:** ~4-6 hours (mostly interactive phases 4d and 6)

---

## Validation & Verification

After each phase, you can validate outputs:

```bash
# Validate specific phase
python3 main.py validate --phase 3

# Validate all phases
python3 main.py validate --all

# Check pipeline status
python3 main.py status
```

---

## Expected Output Dataset

**Final dataset structure:**

```
traffic_dataset/
├── final/
│   ├── train/                    # 11,200 images (80%)
│   │   ├── empty/                # ~2,800 images
│   │   ├── light/                # ~2,800 images
│   │   ├── moderate/             # ~2,800 images
│   │   └── heavy/                # ~2,800 images
│   └── val/                      # 2,800 images (20%)
│       ├── empty/                # ~700 images
│       ├── light/                # ~700 images
│       ├── moderate/             # ~700 images
│       └── heavy/                # ~700 images
```

**Dataset stats:**
- **28 cameras** (3 excluded due to corruption)
- **500 images per camera** (balanced across 4 traffic levels)
- **14,000 total labeled images** (28 cameras × 500 images)
- **4 classes**: empty, light, moderate, heavy
- **Image size**: 64×64 pixels
- **Train/Val split**: 80/20 with stratification

---

## Troubleshooting

### If YOLO runs out of memory:
```bash
# Use smaller batch size
python3 main.py run --phase 2b --device mps --resume --batch-size 16
```

### If you want to re-run a specific phase:
```bash
# Reset phase output
python3 main.py reset --phase 5

# Re-run phase
python3 main.py run --phase 5 --resume
```

### If you want to see detailed logs:
```bash
# Check pipeline log
tail -f traffic_dataset/pipeline.log
```

### If a phase crashes:
All phases support `--resume` flag, so just re-run the same command.

---

## Notes

1. **Excluded cameras:** VATIN_U, VATIN_I, GOSTUN_I are automatically skipped in all phases
2. **Lane-aware classification:** Phase 2b uses your polygons to filter YOLO detections
3. **Active lane detection:** Dynamically detects which lanes have traffic
4. **Resume capability:** All phases can be safely interrupted and resumed
5. **Auto-save:** Interactive phases (2, 6) auto-save every 10 images

---

## Next Steps After Dataset Creation

Once you have the final dataset:

1. **Train CNN model** on the labeled 64×64 crops
2. **Evaluate model** on validation set
3. **Deploy model** for real-time traffic classification
4. **Use exclusion zones** to train secondary congestion detection model

Good luck! 🚀
