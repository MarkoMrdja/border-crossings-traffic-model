# Border Crossings Traffic Model - Dataset Pipeline

A multi-phase pipeline for creating a labeled traffic density dataset from Serbian border crossing cameras. This dataset trains a CNN to classify traffic density in camera-specific regions where YOLO detection fails on distant vehicles.

## 🎯 Project Goal

Train a CNN to classify binary traffic presence (traffic_present / traffic_absent) with >85% accuracy, achieving:
- Traffic present recall >90% (don't miss congestion)
- Traffic absent precision >90% (avoid false alarms)
- Processing time <100ms per image on M2 MacBook (real-time capable)

## 📊 Current Dataset Status

- **Labeled images**: 4,101 images (2,909 absent, 1,056 present, 136 uncertain)
- **Cameras**: 20 active border crossings (16 U/I pairs)
- **Source**: Azure Blob Storage (~18M images from Dec 2023 onwards)
- **Classification**: Binary (traffic_present / traffic_absent)
- **Distribution**: Stratified across time and seasons

## 🚀 Quick Start

### Prerequisites

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure Azure credentials** (create `.env` file):
```bash
AZURE_CLIENT_ID=your_client_id
AZURE_TENANT_ID=your_tenant_id
AZURE_CLIENT_SECRET=your_client_secret
AZURE_STORAGE_URL=your_storage_url
```

### Run the Pipeline

```bash
# Check current status
python main.py status

# Run entire pipeline with resume
python main.py run --all --resume

# Run specific phase
python main.py run --phase 2 --device mps --resume

# Validate outputs
python main.py validate --all
```

## 📋 Pipeline Overview

| Phase | Name | Interactive | Time | Status |
|-------|------|------------|------|--------|
| 1a | Structure Discovery | No | 5-15 min | ✅ Complete |
| 1b | Stratified Sampling | No | 10-30 min | ✅ Complete |
| 1c | Parallel Download | No | 1.5-3 hrs | ✅ Complete |
| 2 | YOLO Analysis | No | ~75 min | ✅ Complete |
| 2a | Lane Annotation | Yes | 2-3 hrs | ✅ Complete |
| 3 | Binary Selection | No | 1-2 min | ✅ Complete |
| 4 | Label Review | Yes | 1-6 hrs | ✅ Complete |
| 5 | Exclusion Zones | Yes | 2-3 hrs | 🔄 Ready |
| 6 | Crop Regions | No | 15-20 min | 🔄 Ready |
| 7 | Train/Val Split | No | 1-2 min | 🔄 Ready |

**Interactive phases** require human input. All phases support `--resume` to continue from saved progress.

## 🏗️ Project Structure

```
.
├── dataset_pipeline/           # Multi-phase pipeline modules
│   ├── phase1_discover.py     # Azure structure discovery
│   ├── phase1_sample.py       # Stratified sampling
│   ├── phase1_download.py     # Parallel download
│   ├── phase2_yolo.py         # YOLO vehicle detection
│   ├── phase2a_lane_annotation.py  # Lane polygon annotation
│   ├── phase3_selection.py    # Binary balanced selection
│   ├── phase4_review.py       # Manual label review
│   ├── phase5_exclusion_zones.py   # YOLO failure regions
│   ├── phase6_crop.py         # ROI cropping
│   └── phase7_split.py        # Train/val split
│
├── traffic_dataset/           # Working data directory
│   ├── raw/                   # Downloaded images
│   ├── binary_labeled/        # Reviewed labels
│   └── binary_final/          # Train/val split
│
├── docs/                      # Documentation
│   ├── ARCHITECTURE.md        # System architecture & design
│   ├── WORKFLOW.md            # Complete pipeline workflow
│   └── guides/
│       └── lane_annotation.md # Lane annotation guide
│
├── main.py                    # Pipeline orchestrator CLI
├── CLAUDE.md                  # AI agent instructions
└── requirements.txt           # Python dependencies
```

## 📚 Documentation

- **[Architecture](docs/ARCHITECTURE.md)**: System design, data structures, and technical specifications
- **[Workflow](docs/WORKFLOW.md)**: Step-by-step pipeline execution guide
- **[Lane Annotation Guide](docs/guides/lane_annotation.md)**: How to annotate lane polygons

## 🔧 Common Commands

### Check Status
```bash
python main.py status              # Show pipeline progress
python main.py validate --all      # Validate all phases
```

### Run Phases
```bash
# YOLO analysis with MPS acceleration
python main.py run --phase 2 --device mps --resume

# Lane annotation (interactive)
python main.py run --phase 2a --resume

# Binary selection
python main.py run --phase 3 --target-per-class 3000

# Label review (interactive, borderline only)
python main.py run --phase 4 --resume

# Label review (interactive, all images)
python main.py run --phase 4 --review-all --resume
```

### Reset & Troubleshooting
```bash
# Reset a phase (creates backup)
python main.py reset --phase 5

# View logs
tail -f traffic_dataset/pipeline.log
```

## 🎓 Course Project

This project serves as an introduction to deep learning course project with learning objectives:
- Understanding CNN architecture and layer types
- Hands-on dataset curation and labeling
- Training, validation, and evaluation of neural networks
- Working with real-world computer vision problems
- Hybrid approach combining traditional CV (YOLO) with custom models

## 🛠️ Technology Stack

- **Python 3.10+**
- **YOLO**: ultralytics (yolo11n for detection)
- **Deep Learning**: PyTorch with MPS (Apple Silicon) support
- **Cloud**: Azure Blob Storage (azure-storage-blob, azure-identity)
- **CV**: OpenCV, Pillow
- **Progress**: tqdm, rich

## 📝 Issue Tracking

This project uses [bd (beads)](https://github.com/yourusername/beads) for issue tracking:

```bash
# View ready work
bd ready

# View all issues
bd list --status open

# Create new issue
bd create "Issue description" -t bug -p 1

# Close issue
bd close bd-123 --reason "Fixed"
```

See [CLAUDE.md](CLAUDE.md) for complete bd workflow documentation.

## 📄 License

[Add your license here]

## 🤝 Contributing

This is a course project. For questions or issues, please use the bd issue tracker:

```bash
bd create "Your question or issue" -t task -p 2
```
