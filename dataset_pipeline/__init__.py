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
from .phase2_review import LabelReviewTool
from .phase3_balance import BalancedSelectionPhase
from .phase4_roi_tool import ROIDefinitionTool
from .phase5_crop import CropPhase
from .phase6_label_tool import LabelingTool
from .phase7_split import TrainValSplitPhase

# Import verification tools
from .verify_yolo import YOLOVerification

__all__ = [
    "StructureDiscoveryPhase",
    "StratifiedSamplingPhase",
    "ParallelDownloadPhase",
    "YOLOAnalysisPhase",
    "LabelReviewTool",
    "BalancedSelectionPhase",
    "ROIDefinitionTool",
    "CropPhase",
    "LabelingTool",
    "TrainValSplitPhase",
    "YOLOVerification",
]
