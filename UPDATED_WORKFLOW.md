# Updated Workflow with Lane Polygons and Lane-Aware Classification

## Overview

The workflow has been updated to use **polygon filtering** and **lane-aware traffic classification**. This ensures accurate traffic density labels by:
1. Only counting vehicles inside the lane boundaries (excludes parked trucks, staff vehicles)
2. Normalizing traffic levels by number of lanes (10 cars on 2 lanes = heavy, but 10 cars on 6 lanes = light)

---

## Complete Workflow

### Step 1: Annotate Lane Polygons (NEW!)

Run the lane annotation tool:

```bash
python -m dataset_pipeline.phase4b_lane_annotation_tool
```

**For each camera:**
1. Tool shows the image with auto-detected polygon
2. **Set number of lanes** using `+` / `-` keys (defaults to 2)
3. Draw/modify the polygon to encompass all traffic lanes
4. Press `[X]` to toggle empty reference view (see clear lane boundaries)
5. Press `[N]` to save and move to next camera

**Output:** `traffic_dataset/lane_polygons.json`
```json
{
  "cameras": {
    "BATROVCI_I": {
      "lane_count": 4,  // NEW! Number of lanes
      "polygons": [{
        "polygon": [[x1,y1], [x2,y2], ...]
      }]
    }
  }
}
```

---

### Step 2: Clean Up Old Results

Delete old YOLO results before re-running:

```bash
python cleanup_for_rerun.py
```

**This removes:**
- Old `yolo_results.json`
- All cropped images (`traffic_dataset/crops/`)
- All labeled images (`traffic_dataset/labeled/`)

**This KEEPS:**
- Raw images
- `lane_polygons.json` (your annotations!)
- All manifests and configurations

---

### Step 3: Re-run YOLO with Polygon Filtering (NEW!)

Run Phase 2b with the improved model:

```bash
python -m dataset_pipeline.phase2b_yolo_filtered
```

**What this does:**
1. **Uses YOLO11m** (medium model - more accurate than nano)
2. **Filters detections** - only counts vehicles inside your polygons
3. **Lane-aware classification** - normalizes by lane count

**Output:** `traffic_dataset/yolo_results_filtered.json`

---

## Lane-Aware Traffic Classification

### Old Method (Phase 2)
Fixed thresholds regardless of lane count:
- Empty: 0-2 vehicles
- Light: 3-6 vehicles
- Moderate: 7-15 vehicles
- Heavy: 16+ vehicles

**Problem:** 10 vehicles on a 6-lane highway is light traffic, but 10 vehicles on a 2-lane road is heavy!

### New Method (Phase 2b)
Normalized by number of lanes:

**Formula:**
```python
vehicles_per_lane = total_vehicles / lane_count

if vehicles_per_lane < 1.0:
    → likely_empty
elif vehicles_per_lane < 2.5:
    → likely_light
elif vehicles_per_lane < 5.0:
    → likely_moderate
else:  # >= 5.0
    → likely_heavy
```

**Examples:**
| Vehicles | Lanes | vehicles_per_lane | Classification |
|----------|-------|-------------------|----------------|
| 10       | 2     | 5.0               | likely_heavy   |
| 10       | 4     | 2.5               | likely_moderate|
| 10       | 6     | 1.67              | likely_light   |
| 2        | 2     | 1.0               | likely_light   |
| 2        | 4     | 0.5               | likely_empty   |

---

## Polygon Filtering Impact

**Before (old Phase 2):** Counts ALL vehicles in image
```
Image shows:
- 15 cars in traffic lanes
- 3 parked trucks on side
- 2 staff vehicles at border building
→ Total: 20 vehicles → "likely_heavy"
```

**After (new Phase 2b):** Counts only vehicles INSIDE polygon
```
Same image:
- 15 cars in traffic lanes (inside polygon)
- 3 parked trucks on side (OUTSIDE polygon - excluded)
- 2 staff vehicles at building (OUTSIDE polygon - excluded)
→ Total: 15 vehicles ÷ 4 lanes = 3.75 vehicles/lane → "likely_moderate"
```

---

## What Gets Saved in lane_polygons.json

```json
{
  "created_at": "2026-01-05T...",
  "mode": "single",
  "cameras": {
    "BATROVCI_I": {
      "reference_image": "raw/BATROVCI_I/...",
      "lane_count": 4,              // Number of lanes you specified
      "polygons": [{
        "id": 0,
        "name": "all_lanes",
        "polygon": [                 // Your drawn coordinates
          [120, 150],
          [680, 140],
          [720, 580],
          [80, 600]
        ],
        "auto_detected": false,
        "user_modified": true,
        "defined_at": "2026-01-05T..."
      }]
    },
    "BATROVCI_U": {
      "lane_count": 6,              // Different camera, different lane count
      "polygons": [...]
    }
    // ... 30 more cameras
  }
}
```

---

## Annotation Tool Controls

**Main Actions:**
- `[A]` - Accept auto-suggestion
- `[M]` - Modify suggestion (enter edit mode)
- `[D]` - Draw manual (start from scratch)
- `[N]` - Next camera (save)
- `[R]` - Reset polygon
- `[S]` - Skip camera
- `[Q]` - Quit (saves progress)

**Lane Count:**
- `[+]` - Increase lane count
- `[-]` - Decrease lane count

**Visualization:**
- `[X]` - Toggle empty reference image
- `[E]` - Toggle edges
- `[V]` - Toggle coverage (shows which YOLO boxes are inside/outside polygon)

**Mouse:**
- Left-click - Add vertex
- Right-click - Remove last vertex

---

## After Phase 2b: Continue Normal Pipeline

Once you have `yolo_results_filtered.json`, continue with the normal pipeline:

### Phase 3: Balanced Selection
```bash
python -m dataset_pipeline.phase3_balance
```
(May need to update to use `yolo_results_filtered.json` instead of `yolo_results.json`)

### Phase 5: Crop Images
```bash
python -m dataset_pipeline.phase5_crop
```
(Already updated to support `lane_polygons.json`)

### Phase 6: Manual Label Review
```bash
python -m dataset_pipeline.phase6_label_tool
```
Review and correct auto-assigned labels

### Phase 7: Train/Val Split
```bash
python -m dataset_pipeline.phase7_split
```

---

## Summary of Changes

**Modified Files:**
1. `phase4b_lane_annotation_tool.py` - Added lane count input
2. `phase5_crop.py` - Already supports `lane_polygons.json`

**New Files:**
1. `phase2b_yolo_filtered.py` - YOLO with polygon filtering + lane-aware classification
2. `cleanup_for_rerun.py` - Delete old results before re-running

**New Data:**
- `lane_polygons.json` - Polygon coordinates + lane count per camera
- `yolo_results_filtered.json` - Filtered YOLO results with lane-aware labels

---

## Quick Start

```bash
# 1. Annotate polygons + set lane counts for all 32 cameras
python -m dataset_pipeline.phase4b_lane_annotation_tool

# 2. Clean up old results
python cleanup_for_rerun.py

# 3. Re-run YOLO with polygon filtering (downloads yolo11m.pt automatically)
python -m dataset_pipeline.phase2b_yolo_filtered

# 4. Continue with Phase 3, 5, 6, 7...
```

---

## Model Comparison

| Model | Size | Accuracy | Speed | Notes |
|-------|------|----------|-------|-------|
| yolo11n (old) | 5.6 MB | Base | Fastest | Used in original Phase 2 |
| yolo11m (new) | ~49 MB | +15% | Medium | Better for border crossings |

The medium model will download automatically on first run (~49 MB).

---

## Benefits

✅ **More accurate labels** - Only counts vehicles in traffic lanes
✅ **Lane-normalized** - Fair comparison across different border crossings
✅ **Better model** - YOLO11m detects vehicles more reliably
✅ **Cleaner dataset** - Excludes parked vehicles, staff, etc.
✅ **Per-camera customization** - Each camera's lane count is considered

Ready to annotate! 🚗📐
