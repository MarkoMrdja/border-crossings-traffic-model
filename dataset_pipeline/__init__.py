"""
Border Crossings Traffic Dataset Pipeline

Multi-phase pipeline for creating a labeled traffic density dataset from Azure Blob Storage.
"""

__version__ = "0.1.0"

# Import phase classes
from .phase1_discover import StructureDiscoveryPhase
from .phase1_sample import StratifiedSamplingPhase
from .phase1_download import ParallelDownloadPhase
from .phase2_yolo import YOLOAnalysisPhase
from .phase2a_lane_annotation import LaneAnnotationTool
from .phase3_selection import SelectionPhase
from .phase4_review import ReviewTool
from .phase5_exclusion_zones import ExclusionZonesTool
from .phase6_crop import CropPhase
from .phase7_split import TrainValSplitPhase

# Import verification tools
from .verify_yolo import YOLOVerification

__all__ = [
    "StructureDiscoveryPhase",
    "StratifiedSamplingPhase",
    "ParallelDownloadPhase",
    "YOLOAnalysisPhase",
    "LaneAnnotationTool",
    "SelectionPhase",
    "ReviewTool",
    "ExclusionZonesTool",
    "CropPhase",
    "TrainValSplitPhase",
    "YOLOVerification",
]
