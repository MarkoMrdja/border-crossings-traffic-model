# Lane Polygon Annotation System - Usage Guide

## Overview

A semi-automated lane polygon annotation system for border crossing traffic cameras. This system solves the problem of YOLO detecting irrelevant vehicles (parked trucks, staff vehicles) by defining lane-specific polygons for precise traffic analysis.

## What Was Built

### 1. **Mini Test Dataset** (`phase4c_mini_dataset.py`)
- Created 64 test images (2 per camera: heaviest + emptiest traffic)
- Located in: `traffic_dataset/mini_test_dataset/`
- Purpose: Validate lane polygons work in extreme traffic conditions

### 2. **Lane Detection Algorithm** (`phase4a_lane_detection.py`)
- **Computer Vision Techniques:**
  - Adaptive Canny edge detection
  - Hough line transform (angle-filtered)
  - YOLO box-guided region inference (convex hull)
  - Road contour detection
  - Polygon merging and simplification
  - Confidence scoring (0-1 scale)

- **Features:**
  - Adaptive thresholds based on image brightness
  - Multiple fallback strategies (YOLO → contours → default bottom 60%)
  - Robust across different camera angles and lighting

### 3. **Interactive Annotation Tool** (`phase4b_lane_annotation_tool.py`)
- **Workflow:**
  1. Auto-detection runs for each camera
  2. Displays polygon suggestion with confidence score
  3. User reviews: Accept / Modify / Reject
  4. Saves to `lane_polygons.json`

- **Keyboard Controls:**
  ```
  [A] Accept auto-suggestion
  [M] Modify suggestion (enter edit mode)
  [D] Reject and draw manual
  [E] Toggle edge detection overlay
  [L] Toggle line detection overlay
  [V] Toggle YOLO coverage visualization
  [N] Next camera (save)
  [R] Reset polygon
  [S] Skip camera
  [Q] Quit and save progress

  Mouse:
  - Left-click: Add vertex
  - Right-click: Remove last vertex
  ```

- **Visualization Layers:**
  - Original image
  - YOLO boxes (green)
  - Edge detection overlay (red, semi-transparent)
  - Line detection overlay (blue)
  - Auto-suggested polygon (cyan)
  - User-edited polygon (yellow)
  - YOLO coverage (green/red centroids showing inside/outside polygon)

### 4. **Configuration** (updated `config.py`)
- Added `LANE_DETECTION_CONFIG` with tunable parameters:
  - Edge detection thresholds
  - Line detection parameters
  - Polygon simplification settings
  - Confidence thresholds

### 5. **Phase 5 Integration** (updated `phase5_crop.py`)
- **Backward Compatible:**
  - Supports both `lane_polygons.json` (new) and `roi_config.json` (old)
  - Automatically transforms lane polygon format to ROI format
  - Existing pipeline continues to work

## Usage

### Step 1: Create Mini Test Dataset

```bash
python -m dataset_pipeline.phase4c_mini_dataset
```

**Output:**
- 64 images in `traffic_dataset/mini_test_dataset/`
- Manifest: `mini_test_dataset/mini_dataset_manifest.json`
- 2 images per camera (EMPTY + HEAVY)

**Verification:**
```bash
ls traffic_dataset/mini_test_dataset/*.jpg | wc -l  # Should be 64
```

---

### Step 2: Run Interactive Annotation Tool

```bash
python -m dataset_pipeline.phase4b_lane_annotation_tool
```

**What to Expect:**
1. OpenCV window opens showing first camera
2. Auto-detection runs (may take a few seconds)
3. Cyan polygon appears with confidence score
4. You review and choose an action:
   - Press **[A]** if polygon looks good → Saves and moves to next camera
   - Press **[M]** to edit → Click to add vertices, right-click to remove
   - Press **[D]** to draw from scratch → Clear auto-suggestion, draw manually

**Tips:**
- Toggle **[E]** to see edge detection (helps validate polygon alignment)
- Toggle **[V]** to see YOLO coverage (green = inside polygon, red = outside)
- Press **[Q]** to quit anytime (progress is saved)
- Use **[S]** to skip problematic cameras and come back later

**Resume Annotation:**
```bash
python -m dataset_pipeline.phase4b_lane_annotation_tool --resume
```

**Output:**
- `traffic_dataset/lane_polygons.json` with polygons for all 32 cameras
- Statistics on auto-acceptance rate, modifications, etc.

---

### Step 3: Verify Output

```bash
python -m dataset_pipeline.phase4b_lane_annotation_tool --validate-only
```

**Expected:**
- All 32 cameras have polygons defined
- Each polygon has at least 3 vertices
- JSON structure is valid

**Manual Inspection:**
```bash
# Check the JSON
cat traffic_dataset/lane_polygons.json | jq '.cameras | keys | length'  # Should be 32

# View statistics
cat traffic_dataset/lane_polygons.json | jq '.statistics'
```

---

### Step 4: Test with Phase 5 Cropping

Once you have `lane_polygons.json`, the existing Phase 5 will automatically use it:

```bash
# Phase 5 will automatically detect and use lane_polygons.json
python -m dataset_pipeline.phase5_crop
```

The integration ensures:
- ✅ `lane_polygons.json` is used if it exists
- ✅ Falls back to `roi_config.json` if not
- ✅ Polygons are extracted correctly for cropping

---

## Output Format: `lane_polygons.json`

```json
{
  "created_at": "2026-01-05T...",
  "mode": "single",
  "cameras": {
    "BATROVCI_I": {
      "reference_image": "raw/BATROVCI_I/2024-08-19_09-35-28.jpg",
      "polygons": [
        {
          "id": 0,
          "name": "all_lanes",
          "polygon": [[x1,y1], [x2,y2], ...],
          "auto_detected": true,
          "user_modified": false,
          "confidence": 0.87,
          "defined_at": "2026-01-05T..."
        }
      ]
    }
  },
  "skipped_cameras": [],
  "statistics": {
    "total_cameras": 32,
    "auto_accepted": 18,
    "user_modified": 10,
    "manual_drawn": 3,
    "skipped": 1
  }
}
```

**Key Fields:**
- `auto_detected`: True if polygon was auto-suggested
- `user_modified`: True if user edited the auto-suggestion
- `confidence`: Detection confidence (0-1), only for auto-detected polygons
- `polygon`: List of [x, y] coordinates defining the lane boundary

---

## Confidence Scoring

Auto-detected polygons receive a confidence score (0.0 to 1.0):

**Metrics (weighted):**
1. **YOLO Box Coverage (40%)**: % of detected vehicles inside polygon
2. **Edge Alignment (40%)**: How well polygon edges align with image edges
3. **Simplicity (20%)**: Preference for 4-6 vertices

**Thresholds:**
- **HIGH** (≥0.75): Likely accurate, good for quick acceptance
- **MEDIUM** (0.50-0.74): Review recommended
- **LOW** (<0.50): Needs modification or manual drawing

---

## Troubleshooting

### Auto-detection confidence is consistently low

**Cause:** Algorithm parameters may need tuning for your specific cameras

**Solution:** Edit `dataset_pipeline/config.py`:
```python
LANE_DETECTION_CONFIG = {
    "edge_detection": {
        "lower_percentile": 0.4,  # Try adjusting these
        "upper_percentile": 1.6,
    },
    "polygon_simplification": {
        "epsilon_factor": 0.03,  # Increase for simpler polygons
    }
}
```

### Polygon has too many vertices

**Cause:** Simplification epsilon too small

**Solution:** Increase `epsilon_factor` in config (e.g., 0.03 or 0.04)

### Auto-suggestion misses parts of lanes

**Cause:**
- No YOLO boxes in those areas (empty traffic)
- Contour detection missed road surface

**Solution:**
- Use **[M]** to modify and add missing vertices manually
- Or use **[D]** to draw from scratch

### Can't see polygon clearly

**Solution:**
- Toggle **[E]** to see edge detection overlay
- Toggle **[L]** to see detected lines
- Resize window if needed (it's resizable)

---

## Statistics & Success Metrics

After completing annotation, check statistics:

```bash
cat traffic_dataset/lane_polygons.json | jq '.statistics'
```

**Target Metrics:**
- **Auto-acceptance rate >70%**: System is working well
- **User modifications 20-25%**: Normal fine-tuning
- **Manual drawings <10%**: Algorithm is robust

**Example Output:**
```json
{
  "total_cameras": 32,
  "auto_accepted": 22,       // 69% acceptance rate
  "user_modified": 8,         // 25% needed tweaking
  "manual_drawn": 2,          // 6% drawn manually
  "skipped": 0
}
```

---

## Architecture Decisions

### Single Polygon per Camera (Current)
**Pros:**
- Faster annotation (32 polygons vs. 64-128)
- Simpler tool and workflow
- Sufficient for filtering out parked vehicles/staff
- Can still count total vehicles in traffic region

**Future: Multi-Lane Mode**
The system is designed to support per-lane polygons later:
- Change `mode` from `"single"` to `"multi"` in config
- Each camera can have multiple polygons (one per lane)
- Enables per-lane traffic analysis

### Backward Compatibility
Phase 5 supports both formats transparently:
```
If lane_polygons.json exists:
  → Use new format
Else if roi_config.json exists:
  → Use old format
Else:
  → Error (no ROI defined)
```

---

## File Structure

```
dataset_pipeline/
  phase4a_lane_detection.py       # Auto-detection algorithm
  phase4b_lane_annotation_tool.py # Interactive GUI tool
  phase4c_mini_dataset.py          # Test dataset creator
  config.py                        # Configuration (updated)
  phase5_crop.py                   # Cropping (updated for compatibility)

traffic_dataset/
  mini_test_dataset/               # 64 test images
    mini_dataset_manifest.json
    BATROVCI_I_EMPTY.jpg
    BATROVCI_I_HEAVY.jpg
    ...
  lane_polygons.json               # Output from annotation tool
```

---

## Next Steps

1. **Run the annotation tool** on all 32 cameras
2. **Review statistics** to assess auto-detection quality
3. **Validate output** with `--validate-only`
4. **Test Phase 5** to ensure cropping works with new polygons
5. **Optional:** Manually inspect a few cropped images to verify polygons capture correct lanes

---

## Advanced: Testing on Specific Cameras

You can test the detection algorithm programmatically:

```python
from dataset_pipeline.phase4a_lane_detection import detect_lane_polygon
from dataset_pipeline.utils import load_json
import cv2
import numpy as np

# Load YOLO data
yolo_data = load_json("traffic_dataset/yolo_results.json")
yolo_lookup = {a['local_path']: a for a in yolo_data['analyses']}

# Test on specific camera
test_image = "traffic_dataset/mini_test_dataset/BATROVCI_I_HEAVY.jpg"
original_path = "raw/BATROVCI_I/..."  # Find from manifest
yolo_boxes = yolo_lookup[original_path]['boxes']

# Run detection
results = detect_lane_polygon(test_image, yolo_boxes, mode='single')

# Visualize
img = cv2.imread(test_image)
polygon = np.array(results[0]['polygon'], dtype=np.int32)
cv2.polylines(img, [polygon], True, (255, 255, 0), 3)
cv2.imwrite("test_detection.jpg", img)

print(f"Confidence: {results[0]['confidence']:.3f}")
print(f"Vertices: {len(results[0]['polygon'])}")
```

---

## Questions?

If you encounter issues or need adjustments:
1. Check the troubleshooting section above
2. Review configuration parameters in `config.py`
3. Inspect auto-detection quality on mini test dataset first
4. Adjust parameters and re-run if needed

**Happy annotating!** 🚗📐
