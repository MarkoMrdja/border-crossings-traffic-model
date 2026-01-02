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

__all__ = [
    "StructureDiscoveryPhase",
    "StratifiedSamplingPhase",
    "ParallelDownloadPhase",
    "YOLOAnalysisPhase",
]
