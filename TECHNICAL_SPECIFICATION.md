# Traffic Density Dataset Pipeline — Technical Specification

## Overview

This document specifies a multi-phase pipeline to create a labeled dataset for training a traffic density classification CNN. The pipeline samples images from Azure Blob Storage, runs YOLO analysis, enables ROI definition, crops regions, and facilitates labeling.

**Target**: 16,000 labeled images (500 per camera × 32 cameras)

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
tqdm  # Progress bars
```

### Hardware
- MacBook Air M2
- Use MPS (Metal Performance Shaders) for YOLO inference when available
- Fallback to CPU if MPS unavailable

### Azure Authentication
Authentication uses service principal credentials from environment variables:
```
AZURE_CLIENT_ID
AZURE_TENANT_ID
AZURE_CLIENT_SECRET
AZURE_STORAGE_URL
```

Container name: `not-processed-imgs`

---

## Directory Structure

All working data stored locally:

```
./traffic_dataset/
├── inventory.json              # Discovered Azure structure
├── sample_manifest.json        # List of images to download
├── download_progress.json      # Resume state for downloads
├── yolo_results.json           # YOLO analysis results
├── roi_config.json             # ROI polygon definitions per camera
├── labeling_progress.json      # Resume state for labeling
│
├── raw/                        # Downloaded original images
│   ├── GRADINA_U/
│   │   ├── 2024-07-15_16-20-58.jpg
│   │   └── ...
│   ├── GRADINA_I/
│   └── {CAMERA}/
│
├── crops/                      # Cropped ROI regions with auto-labels
│   ├── likely_empty/
│   ├── likely_light/
│   ├── likely_moderate/
│   └── likely_heavy/
│
├── labeled/                    # Human-verified labels
│   ├── empty/
│   ├── light/
│   ├── moderate/
│   └── heavy/
│
└── final/                      # Train/val split
    ├── train/
    │   ├── empty/
    │   ├── light/
    │   ├── moderate/
    │   └── heavy/
    └── val/
        ├── empty/
        ├── light/
        ├── moderate/
        └── heavy/
```

---

## Phase 1: Data Discovery & Sampling

### Phase 1a: Structure Discovery

**Purpose**: Map the Azure container structure without listing all 18M blobs.

**Method**: Use hierarchical listing with delimiter to discover virtual directories.

```python
# Pseudocode
def discover_structure(container_client):
    inventory = {}
    
    # Level 1: Border names
    borders = list_prefixes(container_client, delimiter='/')
    # Returns: ['GRADINA/', 'DJALA/', 'KELEBIJA/', ...]
    
    for border in borders:
        inventory[border] = {}
        
        # Level 2: Directions (U/I)
        directions = list_prefixes(container_client, prefix=border, delimiter='/')
        # Returns: ['GRADINA/U/', 'GRADINA/I/']
        
        for direction in directions:
            camera_id = f"{border.strip('/')}{direction.strip('/').split('/')[-1]}"
            inventory[camera_id] = {}
            
            # Level 3: Years
            years = list_prefixes(container_client, prefix=direction, delimiter='/')
            
            for year in years:
                inventory[camera_id][year] = {}
                
                # Level 4: Months
                months = list_prefixes(container_client, prefix=year, delimiter='/')
                
                for month in months:
                    # Level 5: Days
                    days = list_prefixes(container_client, prefix=month, delimiter='/')
                    inventory[camera_id][year][month] = [list of days]
    
    return inventory
```

**Azure API Usage**:
```python
from azure.storage.blob import ContainerClient

def list_prefixes(container_client, prefix='', delimiter='/'):
    """List virtual directories at one level."""
    prefixes = []
    blob_list = container_client.walk_blobs(name_starts_with=prefix, delimiter=delimiter)
    for item in blob_list:
        if hasattr(item, 'prefix'):  # It's a virtual directory
            prefixes.append(item.prefix)
    return prefixes
```

**Output**: `inventory.json`
```json
{
  "GRADINA_U": {
    "2023": {
      "12": ["01", "02", "03", "..."],
    },
    "2024": {
      "1": ["1", "2", "3", "..."],
      "01": ["01", "02", "03", "..."],
      "07": ["01", "02", "15", "..."]
    }
  },
  "GRADINA_I": { ... },
  ...
}
```

**Note on Month Format**: Both padded (`07`) and non-padded (`7`) formats may exist. Store both in inventory and handle accordingly during sampling.

**Estimated Time**: 5-15 minutes

---

### Phase 1b: Sample Selection

**Purpose**: Select 700 images per camera with stratified distribution.

**Strategy**: Equal distribution across time-of-day and season buckets.

**Time Buckets** (5 buckets):
| Bucket | Hours | Images per Camera |
|--------|-------|-------------------|
| Night | 22:00 - 05:59 | 140 |
| Morning | 06:00 - 09:59 | 140 |
| Midday | 10:00 - 13:59 | 140 |
| Afternoon | 14:00 - 17:59 | 140 |
| Evening | 18:00 - 21:59 | 140 |

**Season Buckets** (4 buckets):
| Season | Months | Images per Time Bucket |
|--------|--------|------------------------|
| Winter | 12, 01, 02 | 35 |
| Spring | 03, 04, 05 | 35 |
| Summer | 06, 07, 08 | 35 |
| Autumn | 09, 10, 11 | 35 |

**Sampling Matrix** (per camera):
```
                Winter  Spring  Summer  Autumn  = 140 per row
Night            35      35      35      35
Morning          35      35      35      35
Midday           35      35      35      35
Afternoon        35      35      35      35
Evening          35      35      35      35
              = 175    175     175     175     = 700 total
```

**Sample Selection Algorithm**:
```python
def select_samples(inventory, camera_id, target_per_cell=35):
    samples = []
    
    for season_months in [['12','01','02'], ['03','04','05'], ['06','07','08'], ['09','10','11']]:
        for time_bucket in [(22,6), (6,10), (10,14), (14,18), (18,22)]:
            candidates = find_candidates(inventory, camera_id, season_months, time_bucket)
            
            if len(candidates) >= target_per_cell:
                selected = random.sample(candidates, target_per_cell)
            else:
                selected = candidates  # Take all available
            
            samples.extend(selected)
    
    return samples
```

**Path Construction**:
```python
def construct_blob_path(camera_id, year, month, day, time_str):
    """
    Construct blob path handling padded/non-padded formats.
    
    camera_id: "GRADINA_U"
    Returns: "GRADINA/U/2024/07/15/16-20-58.jpg"
    """
    parts = camera_id.rsplit('_', 1)  # ['GRADINA', 'U']
    border = parts[0]
    direction = parts[1]
    
    return f"{border}/{direction}/{year}/{month}/{day}/{time_str}.jpg"
```

**For listing images in a specific day**:
Since we need to know what times are available, list blobs within each day directory:
```python
def list_images_in_day(container_client, prefix):
    """List actual image files in a day folder."""
    images = []
    blobs = container_client.list_blobs(name_starts_with=prefix)
    for blob in blobs:
        if blob.name.endswith('.jpg'):
            images.append(blob.name)
    return images
```

**Output**: `sample_manifest.json`
```json
{
  "total_samples": 22400,
  "per_camera": 700,
  "samples": [
    {
      "camera_id": "GRADINA_U",
      "blob_path": "GRADINA/U/2024/07/15/16-20-58.jpg",
      "local_path": "raw/GRADINA_U/2024-07-15_16-20-58.jpg",
      "year": "2024",
      "month": "07",
      "day": "15",
      "time": "16-20-58",
      "time_bucket": "afternoon",
      "season": "summer"
    },
    ...
  ]
}
```

**Estimated Time**: 10-30 minutes (needs to list contents of selected day folders)

---

### Phase 1c: Batch Download

**Purpose**: Download all 22,400 selected images to local disk.

**Resume Capability**:
Track progress in `download_progress.json`:
```json
{
  "total": 22400,
  "completed": 15234,
  "failed": [],
  "last_index": 15233,
  "started_at": "2024-07-15T10:30:00",
  "updated_at": "2024-07-15T11:45:00"
}
```

**Download Logic**:
```python
def download_images(container_client, manifest, progress_file):
    progress = load_progress(progress_file)
    start_index = progress.get('last_index', -1) + 1
    
    for i, sample in enumerate(manifest['samples'][start_index:], start=start_index):
        blob_path = sample['blob_path']
        local_path = os.path.join('traffic_dataset', sample['local_path'])
        
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            blob_client = container_client.get_blob_client(blob_path)
            with open(local_path, 'wb') as f:
                data = blob_client.download_blob()
                f.write(data.readall())
            
            # Verify download
            if os.path.getsize(local_path) < 1000:  # Suspiciously small
                raise Exception("Downloaded file too small")
            
            progress['completed'] += 1
            progress['last_index'] = i
            
        except Exception as e:
            progress['failed'].append({'index': i, 'path': blob_path, 'error': str(e)})
        
        # Save progress every 100 images
        if i % 100 == 0:
            save_progress(progress, progress_file)
            print(f"Progress: {progress['completed']}/{progress['total']}")
    
    save_progress(progress, progress_file)
```

**Local Filename Convention**:
Convert blob path to local filename:
```
Blob:  GRADINA/U/2024/07/15/16-20-58.jpg
Local: raw/GRADINA_U/2024-07-15_16-20-58.jpg
```

This preserves all information while flattening to camera-based folders.

**Estimated Time**: 1.5-3 hours (depends on network speed)

---

## Phase 2: YOLO Analysis

**Purpose**: Run YOLO on all downloaded images to count detected vehicles.

### Configuration

```python
YOLO_CONFIG = {
    'model': 'yolo11n',
    'confidence_threshold': 0.25,
    'device': 'mps',  # Apple Silicon GPU, fallback to 'cpu'
    'classes': [2, 3, 5, 7],  # car, motorcycle, bus, truck in COCO
}
```

### Traffic Level Thresholds

```python
TRAFFIC_THRESHOLDS = {
    'empty': (0, 2),      # 0-2 detections
    'light': (3, 6),      # 3-6 detections
    'moderate': (7, 15),  # 7-15 detections
    'heavy': (16, float('inf'))  # 16+ detections
}
```

### Processing Logic

```python
def analyze_with_yolo(image_dir, manifest, output_file):
    from ultralytics import YOLO
    
    # Select device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO('yolo11n.pt')
    
    results = {
        'model': 'yolo11n',
        'device': device,
        'threshold': 0.25,
        'analyses': []
    }
    
    for sample in tqdm(manifest['samples']):
        local_path = os.path.join('traffic_dataset', sample['local_path'])
        
        if not os.path.exists(local_path):
            continue
        
        # Run inference
        detections = model(local_path, conf=0.25, classes=[2,3,5,7], verbose=False)
        
        # Count vehicles
        vehicle_count = len(detections[0].boxes)
        
        # Get bounding boxes for later visualization
        boxes = []
        for box in detections[0].boxes:
            boxes.append({
                'xyxy': box.xyxy[0].tolist(),
                'confidence': float(box.conf[0]),
                'class': int(box.cls[0])
            })
        
        # Determine traffic level
        traffic_level = categorize_traffic(vehicle_count)
        
        results['analyses'].append({
            'camera_id': sample['camera_id'],
            'local_path': sample['local_path'],
            'vehicle_count': vehicle_count,
            'traffic_level': traffic_level,
            'boxes': boxes
        })
    
    save_json(results, output_file)
    return results

def categorize_traffic(count):
    if count <= 2:
        return 'likely_empty'
    elif count <= 6:
        return 'likely_light'
    elif count <= 15:
        return 'likely_moderate'
    else:
        return 'likely_heavy'
```

**Output**: `yolo_results.json`
```json
{
  "model": "yolo11n",
  "device": "mps",
  "threshold": 0.25,
  "analyses": [
    {
      "camera_id": "GRADINA_U",
      "local_path": "raw/GRADINA_U/2024-07-15_16-20-58.jpg",
      "vehicle_count": 14,
      "traffic_level": "likely_moderate",
      "boxes": [
        {"xyxy": [100, 200, 150, 250], "confidence": 0.87, "class": 2},
        ...
      ]
    },
    ...
  ]
}
```

**Estimated Time**: ~75 minutes with MPS acceleration

---

## Phase 3: Traffic-Balanced Selection

**Purpose**: Select final 500 images per camera, balanced across traffic levels.

### Selection Strategy

```python
TARGET_PER_CAMERA = 500
TARGET_PER_TRAFFIC_LEVEL = 125  # 500 / 4 levels

def select_balanced_samples(yolo_results, manifest):
    final_selection = []
    
    # Group by camera
    by_camera = group_by_camera(yolo_results['analyses'])
    
    for camera_id, analyses in by_camera.items():
        camera_selection = []
        
        # Group by traffic level
        by_level = {
            'likely_empty': [],
            'likely_light': [],
            'likely_moderate': [],
            'likely_heavy': []
        }
        
        for analysis in analyses:
            by_level[analysis['traffic_level']].append(analysis)
        
        # Select from each level
        remaining_quota = TARGET_PER_CAMERA
        
        for level in ['likely_heavy', 'likely_moderate', 'likely_light', 'likely_empty']:
            available = by_level[level]
            target = min(TARGET_PER_TRAFFIC_LEVEL, remaining_quota, len(available))
            
            if len(available) <= target:
                selected = available
            else:
                selected = random.sample(available, target)
            
            camera_selection.extend(selected)
            remaining_quota -= len(selected)
        
        # If still under quota, fill from any level
        if len(camera_selection) < TARGET_PER_CAMERA:
            all_remaining = [a for a in analyses if a not in camera_selection]
            needed = TARGET_PER_CAMERA - len(camera_selection)
            if all_remaining:
                camera_selection.extend(random.sample(all_remaining, min(needed, len(all_remaining))))
        
        final_selection.extend(camera_selection[:TARGET_PER_CAMERA])
    
    return final_selection
```

### Identify Best Images for ROI Definition

For each camera, find the image with highest vehicle count (best for seeing where YOLO fails):

```python
def find_roi_reference_images(yolo_results):
    """Find highest-traffic image per camera for ROI definition."""
    reference_images = {}
    
    by_camera = group_by_camera(yolo_results['analyses'])
    
    for camera_id, analyses in by_camera.items():
        # Sort by vehicle count descending
        sorted_analyses = sorted(analyses, key=lambda x: x['vehicle_count'], reverse=True)
        
        # Take top image as reference
        reference_images[camera_id] = {
            'local_path': sorted_analyses[0]['local_path'],
            'vehicle_count': sorted_analyses[0]['vehicle_count'],
            'boxes': sorted_analyses[0]['boxes']
        }
    
    return reference_images
```

**Output**: Updated manifest with final selection and reference images for ROI definition.

---

## Phase 4: ROI Definition

**Purpose**: For each camera, define polygon region where YOLO misses vehicles.

### Tool Requirements

Interactive tool showing:
1. The reference image (highest traffic for that camera)
2. YOLO bounding boxes overlaid (to see where detection works)
3. User draws polygon around area where vehicles are visible but YOLO didn't detect

### Interface Specification

```
┌─────────────────────────────────────────────────────────────┐
│  ROI Definition - GRADINA_U (Camera 1/32)                   │
│  YOLO detected: 14 vehicles                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    [Image with YOLO boxes in GREEN]                         │
│    [User polygon in YELLOW]                                 │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Left-click: Add vertex    Right-click: Remove vertex       │
│  [N] Next camera (save)    [R] Reset polygon                │
│  [S] Skip camera           [Q] Quit (progress saved)        │
└─────────────────────────────────────────────────────────────┘
```

### Controls

| Key | Action |
|-----|--------|
| Left-click | Add vertex to polygon |
| Right-click | Remove last vertex |
| N | Save polygon and move to next camera |
| R | Reset current polygon |
| S | Skip this camera (no ROI defined) |
| Q | Save progress and quit |

### Output: `roi_config.json`

```json
{
  "created_at": "2024-07-15T14:30:00",
  "cameras": {
    "GRADINA_U": {
      "reference_image": "raw/GRADINA_U/2024-07-15_16-20-58.jpg",
      "polygon": [[120, 50], [580, 50], [620, 200], [80, 200]],
      "defined_at": "2024-07-15T14:32:00"
    },
    "GRADINA_I": {
      "reference_image": "raw/GRADINA_I/2024-07-13_09-15-30.jpg",
      "polygon": [[100, 80], [550, 80], [600, 220], [50, 220]],
      "defined_at": "2024-07-15T14:35:00"
    },
    ...
  },
  "skipped_cameras": []
}
```

### Resume Capability

Tool should:
1. Load existing `roi_config.json` if present
2. Skip cameras that already have polygons defined
3. Allow re-defining a polygon if user chooses

**Estimated Time**: ~2.5 hours (5 minutes × 32 cameras)

---

## Phase 5: Batch Crop

**Purpose**: Crop ROI region from all 500 selected images per camera.

### Cropping Logic

```python
CROP_SIZE = 64  # Output size in pixels

def crop_roi(image_path, polygon, output_size=64):
    """
    Crop polygon region from image.
    
    1. Calculate bounding box of polygon
    2. Crop bounding box from image
    3. Create mask for polygon
    4. Apply mask (black outside polygon)
    5. Resize to output_size × output_size
    """
    image = cv2.imread(image_path)
    
    # Get bounding box
    polygon_np = np.array(polygon)
    x_min, y_min = polygon_np.min(axis=0)
    x_max, y_max = polygon_np.max(axis=0)
    
    # Clamp to image bounds
    h, w = image.shape[:2]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    
    # Crop bounding box
    cropped = image[y_min:y_max, x_min:x_max].copy()
    
    # Create and apply mask
    mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
    shifted_polygon = polygon_np - [x_min, y_min]
    cv2.fillPoly(mask, [shifted_polygon.astype(np.int32)], 255)
    
    # Apply mask (black outside polygon)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    cropped = (cropped * mask_3ch).astype(np.uint8)
    
    # Resize
    resized = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_AREA)
    
    return resized
```

### Batch Processing

```python
def batch_crop(selection, roi_config, yolo_results):
    """Crop all selected images and organize by predicted traffic level."""
    
    # Create output directories
    for level in ['likely_empty', 'likely_light', 'likely_moderate', 'likely_heavy']:
        os.makedirs(f'traffic_dataset/crops/{level}', exist_ok=True)
    
    # Build lookup for YOLO results
    yolo_lookup = {a['local_path']: a for a in yolo_results['analyses']}
    
    for item in tqdm(selection):
        camera_id = item['camera_id']
        local_path = item['local_path']
        
        # Skip if no ROI defined for this camera
        if camera_id not in roi_config['cameras']:
            continue
        
        polygon = roi_config['cameras'][camera_id]['polygon']
        full_path = os.path.join('traffic_dataset', local_path)
        
        if not os.path.exists(full_path):
            continue
        
        # Crop
        cropped = crop_roi(full_path, polygon)
        
        # Determine output folder based on YOLO prediction
        traffic_level = yolo_lookup.get(local_path, {}).get('traffic_level', 'likely_empty')
        
        # Generate output filename
        filename = f"{camera_id}_{os.path.basename(local_path)}"
        output_path = f'traffic_dataset/crops/{traffic_level}/{filename}'
        
        cv2.imwrite(output_path, cropped)
```

**Output**: 16,000 cropped images in `crops/likely_{level}/` folders

**Estimated Time**: 5-10 minutes

---

## Phase 6: Labeling & Verification

**Purpose**: Human verification of auto-assigned labels.

### Tool Design

Since images are pre-sorted by predicted level, the labeling tool should:

1. Show images from one predicted category at a time
2. User confirms (Enter) or re-assigns (1-4)
3. Track corrections for analysis

### Interface

```
┌─────────────────────────────────────────────────────────────┐
│  Labeling - Reviewing: likely_moderate                      │
│  Progress: 2,847 / 4,000  (71.2%)                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│    [64×64 crop, scaled up for visibility]                   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Predicted: MODERATE                                        │
│                                                             │
│  [Enter] Confirm    [1] Empty   [2] Light                   │
│  [3] Moderate       [4] Heavy   [S] Skip   [Q] Quit         │
└─────────────────────────────────────────────────────────────┘
```

### Workflow

```python
def labeling_workflow():
    # Process in order: likely_heavy, likely_moderate, likely_light, likely_empty
    # This order means worst mislabels (missed heavy traffic) are caught first
    
    for predicted_level in ['likely_heavy', 'likely_moderate', 'likely_light', 'likely_empty']:
        review_category(predicted_level)
```

### Key Behaviors

1. **Enter key confirms** - Most images will match their prediction, so confirmation should be fastest action
2. **Number keys re-assign** - Press 1-4 to move to different class
3. **Images move to `labeled/` directory** - Organized by confirmed/corrected label
4. **Progress saved continuously** - Can quit and resume anytime

### Output

Images organized in `labeled/` by verified class:
```
labeled/
├── empty/
├── light/
├── moderate/
└── heavy/
```

Plus `labeling_progress.json` tracking:
```json
{
  "total": 16000,
  "confirmed": 12500,
  "corrected": 2800,
  "skipped": 200,
  "remaining": 500,
  "corrections": {
    "likely_moderate_to_heavy": 450,
    "likely_moderate_to_light": 380,
    ...
  }
}
```

**Estimated Time**: 2-3 hours (16,000 images, ~0.5-1 sec average per image with pre-sorting)

---

## Phase 7: Train/Val Split

**Purpose**: Create final dataset split for model training.

### Split Configuration

```python
TRAIN_RATIO = 0.8  # 80% train, 20% validation
```

### Logic

```python
def create_split(labeled_dir, output_dir, train_ratio=0.8):
    for class_name in ['empty', 'light', 'moderate', 'heavy']:
        class_dir = os.path.join(labeled_dir, class_name)
        images = list(Path(class_dir).glob('*.jpg'))
        
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to final directories
        for img in train_images:
            shutil.copy(img, f'{output_dir}/train/{class_name}/{img.name}')
        
        for img in val_images:
            shutil.copy(img, f'{output_dir}/val/{class_name}/{img.name}')
        
        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")
```

### Final Output

```
final/
├── train/
│   ├── empty/     (~3,200 images)
│   ├── light/     (~3,200 images)
│   ├── moderate/  (~3,200 images)
│   └── heavy/     (~3,200 images)
└── val/
    ├── empty/     (~800 images)
    ├── light/     (~800 images)
    ├── moderate/  (~800 images)
    └── heavy/     (~800 images)
```

---

## Script Organization

### Recommended File Structure

```
dataset_pipeline/
├── config.py               # Shared configuration
├── azure_client.py         # Azure authentication and client setup
├── phase1_discover.py      # Structure discovery
├── phase1_sample.py        # Sample selection
├── phase1_download.py      # Batch download with resume
├── phase2_yolo.py          # YOLO analysis
├── phase3_balance.py       # Traffic-balanced selection
├── phase4_roi_tool.py      # Interactive ROI definition
├── phase5_crop.py          # Batch cropping
├── phase6_label_tool.py    # Labeling interface
├── phase7_split.py         # Train/val split
├── main.py                 # Orchestration script
└── utils.py                # Shared utilities
```

### CLI Interface

```bash
# Run full pipeline interactively
python main.py --all

# Run specific phase
python main.py --phase 1
python main.py --phase 2
# etc.

# Check status
python main.py --status

# Resume interrupted phase
python main.py --phase 1 --resume
```

---

## Error Handling & Logging

### Logging Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_dataset/pipeline.log'),
        logging.StreamHandler()
    ]
)
```

### Error Recovery

Each phase should:
1. Save progress to JSON after each significant step
2. On start, check for existing progress file and offer to resume
3. Log all errors with full context
4. Continue processing other items if one fails

---

## Time Estimates Summary

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 1a | Structure discovery | 5-15 min |
| 1b | Sample selection | 10-30 min |
| 1c | Download 22,400 images | 1.5-3 hours |
| 2 | YOLO analysis | ~75 min |
| 3 | Balanced selection | <1 min |
| 4 | ROI definition | ~2.5 hours |
| 5 | Batch crop | 5-10 min |
| 6 | Labeling verification | 2-3 hours |
| 7 | Train/val split | <1 min |
| **Total** | | **~8-10 hours** |

Note: Phases 1c and 4 and 6 require human presence. Other phases can run unattended.
