# Border Crossings Traffic Model - Architecture

## Project Overview

This project creates a labeled traffic density dataset from 32 Serbian border crossing cameras, ultimately training a CNN to classify traffic density with >85% accuracy. The system uses a hybrid approach combining YOLO for near-field vehicle detection with a custom CNN for far-field traffic density classification.

### Border Crossings (Serbia)

| Border | Direction | Neighbor | Border | Direction | Neighbor |
|--------|-----------|----------|--------|-----------|----------|
| GRADINA | U/I | Bulgaria | SID | U/I | Croatia |
| VRSKA-CUKA | U/I | Bulgaria | BATROVCI | U/I | Croatia |
| PRESEVO | U/I | N. Macedonia | VATIN | U/I | Romania |
| GOSTUN | U/I | Montenegro | JABUKA | U/I | Romania |
| SPILJANI | U/I | Montenegro | HORGOS | U/I | Hungary |
| TRBUSNICA | U/I | Bosnia | KELEBIJA | U/I | Hungary |
| KOTROMAN | U/I | Bosnia | DJALA | U/I | Hungary |
| MZVORNIK | U/I | Bosnia | RACA | U/I | Bosnia |

**Note**: Direction codes: U (Ulaz/Entry), I (Izlaz/Exit)

---

## The Problem

### YOLO Detection Limitations

YOLO (specifically yolo11n) performs well for vehicles near the camera but fails to detect distant vehicles where:
- Vehicles appear very small (10-20 pixels)
- Image resolution is low
- Multiple vehicles overlap or blend together
- Heavy traffic creates a dense texture rather than distinguishable objects

This means YOLO might report "5 vehicles detected" when there's actually a 2-kilometer queue of hundreds of cars extending into the distance.

### The Solution: Hybrid Approach

Instead of trying to detect individual distant vehicles (which is physically impossible at this resolution), we train a custom CNN to classify **traffic density** in problematic regions:

```
Camera Image
     │
     ├──► YOLO Detection (near-field)
     │    - Precise vehicle count for nearby vehicles
     │    - Works well within ~50m of camera
     │
     └──► Custom CNN (far-field ROI)
          - Classifies traffic density in distant regions
          - Trained specifically for each camera's problem area

Combined Output: "3 vehicles at checkpoint + heavy queue forming"
```

**Binary Classification (Current):**
- `traffic_present`: Vehicles detected in lanes (any congestion level)
- `traffic_absent`: No vehicles or very light traffic

---

## System Requirements

### Dependencies
```
azure-storage-blob
azure-identity
python-dotenv
ultralytics  # YOLO
torch
torchvision
opencv-python
Pillow
numpy
tqdm
```

### Hardware
- MacBook Air M2 (Apple Silicon)
- MPS (Metal Performance Shaders) for YOLO inference
- Fallback to CPU if MPS unavailable

### Azure Authentication
Service principal credentials from environment variables:
```
AZURE_CLIENT_ID
AZURE_TENANT_ID
AZURE_CLIENT_SECRET
AZURE_STORAGE_URL
```

Container: `not-processed-imgs`

---

## Data Collection Infrastructure

### Image Collector Script

Images are collected by a Python script running in Docker that:
1. Reads camera stream URLs from `stream_urls.csv`
2. Captures one frame from each stream using FFmpeg
3. Uploads frames to Azure Blob Storage
4. Runs continuously in a loop

**Storage Path Format:**
```
{BORDER}/{DIRECTION}/{YEAR}/{MONTH}/{DAY}/{HH-MM-SS}.jpg

Examples:
GRADINA/U/2024/07/15/16-20-58.jpg
VATIN/I/2023/12/25/08-30-00.jpg
```

### Data Volume
- **Start date**: December 2023
- **Total images**: ~18 million
- **Per camera**: ~560,000 images average
- **Storage container**: "not-processed-imgs"

### Peak Traffic Periods
- **Summer peak**: August (holiday season)
- **Winter peak**: December 15 - January 15 (Serbian holidays)
- **Weekly pattern**: Heavier on weekends, especially Friday evenings and Sunday afternoons

---

## Directory Structure

```
./traffic_dataset/
├── inventory.json              # Discovered Azure structure
├── sample_manifest.json        # List of images to download
├── yolo_results.json           # YOLO analysis results
├── lane_polygons.json          # Lane polygon definitions per camera
├── binary_selection.json       # Selected images for binary classification
├── binary_review_log.json      # Review statistics
├── pipeline.log                # Pipeline execution logs
│
├── raw/                        # Downloaded original images
│   ├── GRADINA_U/
│   │   ├── 2024-07-15_16-20-58.jpg
│   │   └── ...
│   └── {CAMERA}/
│
├── binary_labeled/             # Human-verified binary labels
│   ├── traffic_present/
│   ├── traffic_absent/
│   └── uncertain/
│
└── binary_final/               # Train/val split
    ├── train/
    │   ├── traffic_present/
    │   └── traffic_absent/
    └── val/
        ├── traffic_present/
        └── traffic_absent/
```

---

## Lane-Aware Classification

### Polygon Filtering

The system uses lane polygons to:
1. **Filter YOLO detections** - Only count vehicles inside lane boundaries (excludes parked trucks, staff vehicles)
2. **Normalize traffic levels** - Account for different lane counts across cameras

**Before polygon filtering:**
```
Image shows:
- 15 cars in traffic lanes
- 3 parked trucks on side
- 2 staff vehicles at border building
→ Total: 20 vehicles → misclassified as heavy
```

**After polygon filtering:**
```
Same image:
- 15 cars in traffic lanes (inside polygon)
- 3 parked trucks (OUTSIDE polygon - excluded)
- 2 staff vehicles (OUTSIDE polygon - excluded)
→ Total: 15 vehicles → correctly classified
```

### Lane Polygons Format

```json
{
  "cameras": {
    "BATROVCI_I": {
      "reference_image": "raw/BATROVCI_I/...",
      "lane_count": 4,
      "polygons": [{
        "id": 0,
        "name": "all_lanes",
        "polygon": [[120, 150], [680, 140], [720, 580], [80, 600]],
        "auto_detected": false,
        "user_modified": true
      }]
    }
  }
}
```

---

## Success Criteria

### Primary Goal
Train a CNN that accurately classifies traffic presence with >85% accuracy on held-out validation set.

### Metrics
- **Traffic present recall** >90% (don't miss congestion)
- **Traffic absent precision** >90% (avoid false alarms)
- **Processing time** <100ms per image on M2 MacBook (real-time capable)

### Secondary Goals
1. Create a reusable dataset of labeled traffic density images
2. Establish a pipeline that can be re-run to expand the dataset
3. Document the process for potential application to new cameras

---

## Technical Constraints

### Time Budget
- Dataset creation: ~6-8 hours (mostly automated)
- Model training: ~1-2 hours
- Iteration and refinement: variable

### Course Context
This project serves as an introduction to deep learning course project. Key learning objectives:
- Understanding CNN architecture and layer types
- Hands-on experience with dataset curation
- Training, validation, and evaluation of neural networks
- Practical application of transfer learning concepts
- Working with real-world messy data
