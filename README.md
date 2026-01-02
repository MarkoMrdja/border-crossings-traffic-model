# Border Crossings Traffic Model - Dataset Pipeline

A multi-phase pipeline for creating a labeled traffic density dataset from 32 Serbian border crossing cameras. This dataset will be used to train a CNN that classifies traffic density in camera-specific ROI regions where YOLO detection fails on distant vehicles.

## 🎯 Project Goal

Train a CNN to classify traffic density with >85% accuracy on held-out validation set, with:
- Heavy traffic recall >90% (don't miss traffic jams)
- Empty road precision >90% (avoid false alarms)
- Processing time <100ms per image on M2 MacBook (real-time capable)

## 📊 Dataset Overview

- **Target**: 16,000 labeled images (500 per camera × 32 cameras)
- **Cameras**: 16 border crossings × 2 directions (U/I) = 32 cameras
- **Source**: Azure Blob Storage (~18M images from Dec 2023 onwards)
- **Classes**: 4 traffic density levels (empty, light, moderate, heavy)
- **Distribution**: Stratified across 5 time buckets × 4 seasons

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
| MZVORNIK | U/I | Bosnia | - | - | - |
| RACA | U/I | Bosnia | - | - | - |

## 🏗️ Project Structure

```
.
├── dataset_pipeline/              # Multi-phase data processing pipeline
│   ├── __init__.py               # Package exports
│   ├── base.py                   # Abstract PipelinePhase base class
│   ├── config.py                 # Configuration (Azure, Pipeline, Constants)
│   ├── azure_client.py           # Azure Blob Storage wrapper
│   ├── utils.py                  # Common utilities (paths, JSON, progress)
│   ├── phase1_discover.py        # Phase 1a: Structure Discovery
│   ├── phase1_sample.py          # Phase 1b: Stratified Sampling
│   ├── phase1_download.py        # Phase 1c: Parallel Download
│   └── phase2_yolo.py            # Phase 2: YOLO Analysis
│
├── traffic_dataset/              # Working directory for all data
│   ├── inventory.json            # Container structure (Phase 1a output)
│   ├── sample_manifest.json      # Selected samples (Phase 1b output)
│   ├── download_progress.json    # Download state (Phase 1c state)
│   ├── yolo_results.json         # Vehicle detections (Phase 2 output)
│   ├── roi_config.json           # ROI polygons (Phase 4 output)
│   ├── labeling_progress.json    # Labeling state (Phase 6 state)
│   ├── raw/                      # Downloaded images (Phase 1c output)
│   ├── crops/                    # 64×64 ROI crops (Phase 5 output)
│   ├── labeled/                  # Human-verified labels (Phase 6 output)
│   └── final/                    # Train/val split (Phase 7 output)
│
├── tests/                        # Unit tests
│   └── test_utils.py            # Utility function tests
│
├── .beads/                       # Issue tracking database
├── PROJECT_CONTEXT.md            # Project overview
├── TECHNICAL_SPECIFICATION.md    # Detailed 7-phase specification
├── CLAUDE.md                     # AI agent instructions
├── requirements.txt              # Python dependencies
└── .env                          # Azure credentials (not in git)
```

## 📋 Pipeline Phases

### ✅ **Phase 1a: Structure Discovery** (`phase1_discover.py`)
**Status**: Implemented
**Purpose**: Map Azure Blob Storage container structure hierarchically
**Time**: 5-15 minutes

Discovers the container structure (borders → directions → years → months → days) without listing all 18M blobs.

```bash
# Run structure discovery
python -m dataset_pipeline.phase1_discover

# Resume from existing inventory
python -m dataset_pipeline.phase1_discover --resume

# Validate only
python -m dataset_pipeline.phase1_discover --validate-only
```

**Output**: `traffic_dataset/inventory.json`

### ✅ **Phase 1b: Stratified Sampling** (`phase1_sample.py`)
**Status**: Implemented
**Purpose**: Select 700 images per camera with stratified distribution
**Time**: 10-30 minutes

Selects images uniformly across:
- **5 time buckets**: Night (22-06), Morning (6-10), Midday (10-14), Afternoon (14-18), Evening (18-22)
- **4 seasons**: Winter (12,01,02), Spring (03-05), Summer (06-08), Autumn (09-11)
- **Target**: 35 images per time-season cell × 20 cells = 700 per camera

```bash
# Run stratified sampling
python -m dataset_pipeline.phase1_sample

# Resume from existing manifest
python -m dataset_pipeline.phase1_sample --resume
```

**Output**: `traffic_dataset/sample_manifest.json` (22,400 samples)

### ✅ **Phase 1c: Parallel Download** (`phase1_download.py`)
**Status**: Implemented
**Purpose**: Download all selected images from Azure
**Time**: 1.5-3 hours

Downloads 22,400 images with resume capability, progress tracking, and error handling.

```bash
# Download images (4 parallel workers by default)
python -m dataset_pipeline.phase1_download

# Resume interrupted download
python -m dataset_pipeline.phase1_download --resume

# Use 8 parallel workers
python -m dataset_pipeline.phase1_download --workers 8
```

**Output**: Images in `traffic_dataset/raw/{CAMERA_ID}/`

### ✅ **Phase 2: YOLO Analysis** (`phase2_yolo.py`)
**Status**: Implemented
**Purpose**: Run YOLO vehicle detection on all images
**Time**: ~75 minutes with MPS

Runs YOLO11n to detect vehicles (cars, motorcycles, buses, trucks) and categorizes traffic:
- **Empty**: 0-2 vehicles
- **Light**: 3-6 vehicles
- **Moderate**: 7-15 vehicles
- **Heavy**: 16+ vehicles

```bash
# Run YOLO analysis (auto-detect MPS/CUDA/CPU)
python -m dataset_pipeline.phase2_yolo

# Resume from existing results
python -m dataset_pipeline.phase2_yolo --resume

# Force CPU device
python -m dataset_pipeline.phase2_yolo --device cpu

# Use different model or confidence
python -m dataset_pipeline.phase2_yolo --model yolo11s --confidence 0.3
```

**Output**: `traffic_dataset/yolo_results.json` with vehicle counts, bounding boxes, and traffic levels

### ✅ **YOLO Verification Tool** (`verify_yolo.py`)
**Status**: Implemented
**Purpose**: Verify YOLO results before proceeding to Phase 3
**Time**: ~2-5 minutes

Analyzes YOLO detection results to ensure data quality with:
- Distribution statistics (overall and per-camera)
- Traffic level balance verification
- Issue detection (insufficient samples, outliers, imbalanced cameras)
- Visualizations (histograms, bar charts, sample images with bounding boxes)
- Comprehensive text and JSON reports

```bash
# Run full verification
python -m dataset_pipeline.verify_yolo

# Customize number of sample images per level
python -m dataset_pipeline.verify_yolo --samples 10

# Custom output directory
python -m dataset_pipeline.verify_yolo --output-dir ./my_verification
```

**Outputs** (saved to `traffic_dataset/yolo_verification/`):
- `distribution_histogram.png` - Vehicle count distribution with threshold lines
- `traffic_levels_by_camera.png` - Stacked bar chart of traffic levels per camera
- `sample_detections_{level}.jpg` - Grid of sample images with bounding boxes (4 files)
- `yolo_verification_report.txt` - Human-readable summary report
- `yolo_verification.json` - Machine-readable report with full statistics

**Quality Checks**:
- ✓ Sufficient samples per traffic level for balanced selection (125+ per camera per level)
- ✓ No extreme outliers (>50 vehicles likely indicates false positives)
- ✓ Reasonable distribution across cameras and traffic levels
- ⚠️ Warns about cameras with imbalanced distributions
- 🔴 Flags critical issues that may prevent Phase 3 from succeeding

### ❌ **Phase 3: Traffic-Balanced Selection**
**Status**: Not implemented
**Purpose**: Select 500 final images per camera, balanced across 4 traffic levels (125 per level)
**Time**: <1 minute

Will select from the 700 samples to ensure balanced representation across traffic density classes.

### ❌ **Phase 4: ROI Definition (Interactive Tool)**
**Status**: Not implemented
**Purpose**: Define polygon regions where YOLO misses vehicles
**Time**: ~2.5 hours (human-in-the-loop)

Interactive OpenCV tool to draw ROI polygons on sample images from each camera.

### ❌ **Phase 5: Batch Crop**
**Status**: Not implemented
**Purpose**: Crop ROI regions and resize to 64×64
**Time**: 5-10 minutes

Crops defined ROI regions from selected images and organizes by predicted traffic level.

### ❌ **Phase 6: Labeling & Verification (Interactive Tool)**
**Status**: Not implemented
**Purpose**: Human verification of auto-assigned labels
**Time**: 2-3 hours (human-in-the-loop)

Interactive tool to verify/correct YOLO-assigned traffic labels.

### ❌ **Phase 7: Train/Val Split**
**Status**: Not implemented
**Purpose**: Create 80/20 train/validation split
**Time**: <1 minute

Splits verified labels into training (12,800) and validation (3,200) sets.

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Azure credentials (service principal with read access to blob storage)
- ~50GB disk space for images

### Installation

```bash
# Clone repository
git clone <repository-url>
cd border-crossings-traffic-model

# Install dependencies
pip install -r requirements.txt

# Configure Azure credentials in .env
cp .env.example .env  # Edit with your credentials
```

**Required environment variables** (`.env`):
```bash
AZURE_CLIENT_ID=your-client-id
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_STORAGE_URL=https://borderimgstorage.blob.core.windows.net/
```

### Running the Pipeline

Execute phases sequentially:

```bash
# Phase 1a: Discover structure (~10 min)
python -m dataset_pipeline.phase1_discover

# Phase 1b: Sample selection (~20 min)
python -m dataset_pipeline.phase1_sample

# Phase 1c: Download images (~2 hours)
python -m dataset_pipeline.phase1_download --workers 4

# Phase 2: YOLO analysis (~75 min on M2 MacBook)
python -m dataset_pipeline.phase2_yolo

# Verify YOLO results (~2-5 min)
python -m dataset_pipeline.verify_yolo

# ... Phases 3-7 to be implemented
```

### Resume Capability

All phases support `--resume` to skip completed work:

```bash
# Resume interrupted download
python -m dataset_pipeline.phase1_download --resume

# Resume YOLO analysis
python -m dataset_pipeline.phase2_yolo --resume
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=dataset_pipeline tests/

# Run specific test file
pytest tests/test_utils.py
```

## 📦 Dependencies

**Core**:
- `azure-storage-blob>=12.19.0` - Azure Blob Storage access
- `azure-identity>=1.15.0` - Azure authentication
- `python-dotenv>=1.0.0` - Environment variable management

**Computer Vision**:
- `ultralytics>=8.0.0` - YOLO models
- `torch>=2.0.0` - PyTorch deep learning
- `torchvision>=0.15.0` - Vision utilities
- `opencv-python>=4.8.0` - Image processing
- `Pillow>=10.0.0` - Image I/O

**Utilities**:
- `numpy>=1.24.0` - Numerical computing
- `tqdm>=4.66.0` - Progress bars

## 🔧 Configuration

### Pipeline Configuration (`dataset_pipeline/config.py`)

```python
from dataset_pipeline.config import PipelineConfig

config = PipelineConfig()
config.base_dir                        # Path("./traffic_dataset")
config.initial_samples_per_camera      # 700
config.final_samples_per_camera        # 500
config.crop_size                       # 64
config.train_ratio                     # 0.8
```

### YOLO Configuration

```python
from dataset_pipeline.config import YOLO_CONFIG

YOLO_CONFIG = {
    "model": "yolo11n",              # Nano model for efficiency
    "confidence_threshold": 0.25,    # Detection confidence
    "device": "mps",                 # Apple Silicon GPU
    "classes": [2, 3, 5, 7],        # car, motorcycle, bus, truck
}
```

### Traffic Level Thresholds

```python
from dataset_pipeline.config import TRAFFIC_THRESHOLDS

TRAFFIC_THRESHOLDS = {
    "empty": (0, 2),           # 0-2 vehicles
    "light": (3, 6),           # 3-6 vehicles
    "moderate": (7, 15),       # 7-15 vehicles
    "heavy": (16, float("inf"))  # 16+ vehicles
}
```

## 📊 Data Distribution

### Temporal Stratification

**Time Buckets** (per camera):

| Bucket | Hours | Images per Camera |
|--------|-------|-------------------|
| Night | 22:00 - 05:59 | 140 |
| Morning | 06:00 - 09:59 | 140 |
| Midday | 10:00 - 13:59 | 140 |
| Afternoon | 14:00 - 17:59 | 140 |
| Evening | 18:00 - 21:59 | 140 |

**Season Buckets** (per time bucket):

| Season | Months | Images per Time Bucket |
|--------|--------|------------------------|
| Winter | 12, 01, 02 | 35 |
| Spring | 03, 04, 05 | 35 |
| Summer | 06, 07, 08 | 35 |
| Autumn | 09, 10, 11 | 35 |

### Traffic Level Balance (Target Final Dataset)

| Class | Images per Camera | Total Images |
|-------|-------------------|--------------|
| Empty | 125 | 4,000 |
| Light | 125 | 4,000 |
| Moderate | 125 | 4,000 |
| Heavy | 125 | 4,000 |
| **Total** | **500** | **16,000** |

### Train/Val Split

- **Training**: 12,800 images (80%)
- **Validation**: 3,200 images (20%)
- Stratified by traffic level and camera

## 🏛️ Architecture

### Base Pipeline Framework

All phases inherit from `PipelinePhase`:

```python
from dataset_pipeline.base import PipelinePhase

class CustomPhase(PipelinePhase):
    def run(self, resume: bool = False):
        """Execute phase logic"""
        pass

    def validate(self) -> bool:
        """Validate phase output"""
        pass
```

**Built-in features**:
- Progress tracking with JSON persistence
- Automatic logging to file and console
- Error handling and timing
- Resume capability
- Validation after execution

### Azure Client Wrapper

```python
from dataset_pipeline.azure_client import AzureBlobClient
from dataset_pipeline.config import AzureConfig

config = AzureConfig.from_env()
client = AzureBlobClient(config)

# List virtual directories
prefixes = client.list_prefixes(prefix="GRADINA/", delimiter="/")

# List blobs
blobs = client.list_blobs(prefix="GRADINA/U/2024/07/15/")

# Download with retry
success = client.download_blob(
    blob_name="GRADINA/U/2024/07/15/16-20-58.jpg",
    local_path=Path("output/image.jpg"),
    max_retries=3
)
```

## 🐛 Issue Tracking

This project uses **bd (beads)** for issue tracking:

```bash
# List ready (unblocked) issues
bd ready

# Show issue details
bd show <issue-id>

# Create new issue
bd create "Issue title" -t bug|feature|task -p 0-4

# Update issue status
bd update <issue-id> --status in_progress

# Close issue
bd close <issue-id> --reason "Done"

# Sync with git
bd sync
```

See `CLAUDE.md` for full bd workflow documentation.

## 📈 Performance

### Estimated Time (Full Pipeline)

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 1a | Structure discovery | 5-15 min |
| 1b | Sample selection | 10-30 min |
| 1c | Download 22,400 images | 1.5-3 hours |
| 2 | YOLO analysis | ~75 min (MPS) |
| 3 | Balanced selection | <1 min |
| 4 | ROI definition (interactive) | ~2.5 hours |
| 5 | Batch crop | 5-10 min |
| 6 | Labeling verification (interactive) | 2-3 hours |
| 7 | Train/val split | <1 min |
| **Total** | | **~8-10 hours** |

**Notes**:
- Phases 1c, 4, and 6 require human presence
- Other phases can run unattended
- Times based on M2 MacBook Air with MPS

### Hardware Acceleration

- **MPS (Metal Performance Shaders)**: Apple Silicon GPU
- **CUDA**: NVIDIA GPU
- **CPU**: Fallback for compatibility

Auto-detected by PyTorch in Phase 2.

## 🤝 Contributing

### Development Setup

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Run linters
black dataset_pipeline/
flake8 dataset_pipeline/

# Type checking
mypy dataset_pipeline/
```

### Adding a New Phase

1. Create `dataset_pipeline/phaseN_name.py`
2. Inherit from `PipelinePhase`
3. Implement `run()` and `validate()` methods
4. Add CLI entry point in `if __name__ == "__main__"`
5. Export in `dataset_pipeline/__init__.py`
6. Add tests in `tests/test_phaseN.py`

Example:

```python
from .base import PipelinePhase
from .config import PipelineConfig

class NewPhase(PipelinePhase):
    def __init__(self, pipeline_config: PipelineConfig):
        super().__init__(
            config=pipeline_config,
            phase_name="new_phase",
            description="Description of what this phase does"
        )

    def run(self, resume: bool = False):
        # Implementation
        return {"result": "data"}

    def validate(self) -> bool:
        # Validation logic
        return True
```

## 📝 License

[License information to be added]

## 👥 Authors

- Marko Mrdja - Initial implementation

## 🔗 References

- [YOLO11 Documentation](https://docs.ultralytics.com/)
- [Azure Blob Storage SDK](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

## 📧 Contact

For questions or issues, please open an issue in the GitHub repository.

---

**Last Updated**: January 2, 2026
**Project Status**: Phases 1-2 implemented (4/7 phases complete)
