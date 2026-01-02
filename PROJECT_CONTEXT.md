# Traffic Density Classification for Border Crossings

## Project Overview

This project aims to provide real-time and historical traffic density information for Serbian border crossings by analyzing camera footage. The system will help users understand how congested each border crossing is before they travel.

## The Problem

### Current Setup

32 cameras monitor 16 border crossings in Serbia (one camera for entry "Ulaz/U", one for exit "Izlaz/I" at each crossing). These cameras continuously capture images that are stored in Azure Blob Storage.

**Border Crossings:**
| Border | Camera IDs | Country Border |
|--------|------------|----------------|
| DJALA | DJALA_U, DJALA_I | Hungary |
| KELEBIJA | KELEBIJA_U, KELEBIJA_I | Hungary |
| HORGOS | HORGOS_U, HORGOS_I | Hungary |
| JABUKA | JABUKA_U, JABUKA_I | Romania |
| VATIN | VATIN_U, VATIN_I | Romania |
| BATROVCI | BATROVCI_U, BATROVCI_I | Croatia |
| SID | SID_U, SID_I | Croatia |
| RACA | RACA_U, RACA_I | Bosnia |
| MZVORNIK | MZVORNIK_U, MZVORNIK_I | Bosnia |
| KOTROMAN | KOTROMAN_U, KOTROMAN_I | Bosnia |
| TRBUSNICA | TRBUSNICA_U, TRBUSNICA_I | Bosnia |
| GOSTUN | GOSTUN_U, GOSTUN_I | Montenegro |
| SPILJANI | SPILJANI_U, SPILJANI_I | Montenegro |
| GRADINA | GRADINA_U, GRADINA_I | Bulgaria |
| VRSKA-CUKA | VRSKA-CUKA_U, VRSKA-CUKA_I | Bulgaria |
| PRESEVO | PRESEVO_U, PRESEVO_I | North Macedonia |

### YOLO Detection Limitations

We initially used YOLO (specifically yolo11n) to detect vehicles in camera images. While YOLO performs well for vehicles near the camera, it fails to detect vehicles in the distance where:

- Vehicles appear very small (10-20 pixels)
- Image resolution is low
- Multiple vehicles overlap or blend together
- Heavy traffic creates a dense texture rather than distinguishable objects

This means YOLO might report "5 vehicles detected" when there's actually a 2-kilometer queue of hundreds of cars extending into the distance.

### The Solution

Instead of trying to detect individual distant vehicles (which is physically impossible at this resolution), we train a custom CNN to classify **traffic density** in the problematic regions. This reframes an impossible detection problem into a solvable classification problem.

**Classification Categories:**
| Class | Description | Visual Characteristics |
|-------|-------------|----------------------|
| empty | No vehicles in ROI | Just road surface, no vehicle texture |
| light | 1-5 vehicles, no queue | Scattered vehicles, road clearly visible between them |
| moderate | Partial queue forming | Continuous line of vehicles, some road still visible |
| heavy | Full congestion | Dense vehicle texture, no road visible, obvious traffic jam |

## Technical Architecture

### Hybrid Approach

The final system will use both YOLO and the custom CNN:

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

### Why a Custom Model?

1. **Specificity**: The model is trained on actual images from these specific cameras
2. **Simplicity**: 4-class classification is much easier than object detection
3. **Robustness**: Texture classification works even when individual vehicles are indistinguishable
4. **Speed**: Small CNN runs fast, suitable for real-time processing

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

**Note on Path Format:**
- Historical data (before recent fix): months may be non-zero-padded (7 instead of 07)
- Recent data: months are zero-padded (07)
- Time is always zero-padded (HH-MM-SS)
- The sampling script must handle both formats

### Data Volume

- **Start date**: December 2023
- **Current**: Ongoing collection
- **Total images**: ~18 million
- **Per camera**: ~560,000 images average
- **Storage container**: "not-processed-imgs"

### Peak Traffic Periods

Based on travel patterns:
- **Summer peak**: August (holiday season)
- **Winter peak**: December 15 - January 15 (Serbian holidays)
- **Weekly pattern**: Heavier on weekends, especially Friday evenings and Sunday afternoons

## Project Goals

### Primary Goal
Train a CNN that accurately classifies traffic density in camera-specific ROI regions, achieving >85% accuracy on a held-out validation set.

### Secondary Goals
1. Create a reusable dataset of labeled traffic density images
2. Establish a pipeline that can be re-run to expand the dataset
3. Document the process for potential application to new cameras

### Success Criteria
- Model correctly identifies "heavy" traffic with >90% recall (we don't want to miss jams)
- Model correctly identifies "empty" roads with >90% precision (avoid false alarms)
- Processing time <100ms per image on M2 MacBook (real-time capable)

## College Project Context

This project serves as an introduction to deep learning course project. Key learning objectives:
- Understanding CNN architecture and layer types
- Hands-on experience with dataset curation
- Training, validation, and evaluation of neural networks
- Practical application of transfer learning concepts
- Working with real-world messy data

## Technical Constraints

### Hardware
- MacBook Air M2 (Apple Silicon)
- No dedicated NVIDIA GPU
- PyTorch with MPS (Metal Performance Shaders) backend available

### Software
- Python 3.10+
- YOLO: ultralytics library with yolo11n model
- CNN: PyTorch
- Azure: azure-storage-blob, azure-identity libraries
- Image processing: OpenCV, Pillow

### Time Budget
- Dataset creation: ~6-8 hours (mostly automated)
- Model training: ~1-2 hours
- Iteration and refinement: variable
