"""
Configuration for Border Crossings Traffic Dataset Pipeline

This module contains all configuration settings for the dataset pipeline,
including Azure authentication, pipeline parameters, and constants.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class AzureConfig:
    """Azure Blob Storage authentication and connection configuration."""

    client_id: str
    tenant_id: str
    client_secret: str
    storage_url: str
    container_name: str = "not-processed-imgs"

    @classmethod
    def from_env(cls) -> "AzureConfig":
        """Create AzureConfig from environment variables."""
        client_id = os.getenv("AZURE_CLIENT_ID")
        tenant_id = os.getenv("AZURE_TENANT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")
        storage_url = os.getenv("AZURE_STORAGE_URL")

        if not all([client_id, tenant_id, client_secret, storage_url]):
            raise ValueError(
                "Missing required Azure environment variables. "
                "Please ensure AZURE_CLIENT_ID, AZURE_TENANT_ID, "
                "AZURE_CLIENT_SECRET, and AZURE_STORAGE_URL are set."
            )

        return cls(
            client_id=client_id,
            tenant_id=tenant_id,
            client_secret=client_secret,
            storage_url=storage_url
        )


@dataclass
class PipelineConfig:
    """Pipeline execution configuration and parameters."""

    # Base directory for all dataset files
    base_dir: Path = Path("./traffic_dataset")

    # Sampling parameters
    initial_samples_per_camera: int = 700
    final_samples_per_camera: int = 500
    target_per_traffic_level: int = 125  # 500 / 4 levels
    samples_per_time_bucket: int = 140
    samples_per_season: int = 35

    # Cropping parameters
    crop_size: int = 64

    # Train/validation split
    train_ratio: float = 0.8

    # Directory paths (relative to base_dir)
    inventory_file: str = "inventory.json"
    sample_manifest_file: str = "sample_manifest.json"
    download_progress_file: str = "download_progress.json"
    yolo_results_file: str = "yolo_results.json"
    roi_config_file: str = "roi_config.json"
    labeling_progress_file: str = "labeling_progress.json"

    raw_dir: str = "raw"
    crops_dir: str = "crops"
    labeled_dir: str = "labeled"
    final_dir: str = "final"

    # Binary classification directories
    binary_crops_dir: str = "binary_crops"
    binary_labeled_dir: str = "binary_labeled"
    binary_final_dir: str = "binary_final"

    def __post_init__(self):
        """Convert string paths to Path objects and ensure base_dir is absolute."""
        self.base_dir = Path(self.base_dir).resolve()

    def get_path(self, relative_path: str) -> Path:
        """Get absolute path for a relative path within base_dir."""
        return self.base_dir / relative_path

    def ensure_directories(self):
        """Create all required directories if they don't exist."""
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create raw directory for downloaded images
        (self.base_dir / self.raw_dir).mkdir(exist_ok=True)

        # Binary classification directories
        for label in BINARY_CLASSIFICATION["classes"]:
            # Binary crops
            (self.base_dir / self.binary_crops_dir / label).mkdir(parents=True, exist_ok=True)
            # Binary labeled (including uncertain)
            (self.base_dir / self.binary_labeled_dir / label).mkdir(parents=True, exist_ok=True)

        # Add uncertain directory for binary labeled
        (self.base_dir / self.binary_labeled_dir / "uncertain").mkdir(parents=True, exist_ok=True)

        # Binary final directories (train/val split)
        for split in ["train", "val"]:
            for label in BINARY_CLASSIFICATION["classes"]:
                (self.base_dir / self.binary_final_dir / split / label).mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONSTANTS
# ============================================================================

# YOLO Configuration
YOLO_CONFIG = {
    "model": "yolo11n",
    "confidence_threshold": 0.25,
    "device": "mps",  # Apple Silicon GPU, fallback to 'cpu'
    "classes": [2, 3],  # car, motorcycle only (exclude bus, truck for accurate passenger vehicle traffic)
}

# Traffic Level Thresholds (vehicle counts)
TRAFFIC_THRESHOLDS = {
    "empty": (0, 2),           # 0-2 detections
    "light": (3, 6),           # 3-6 detections
    "moderate": (7, 15),       # 7-15 detections
    "heavy": (16, float("inf"))  # 16+ detections
}

# Traffic levels (ordered for processing)
TRAFFIC_LEVELS = ["empty", "light", "moderate", "heavy"]

# Binary Classification Configuration
BINARY_CLASSIFICATION = {
    "enabled": True,
    "crop_size": 128,  # Changed from 64 to 128 for better detail in distant regions
    "classes": ["traffic_absent", "traffic_present"],
    "auto_label_confidence_threshold": 0.70,
    "review_sample_size": 2500  # Number of crops to manually review (if needed)
}

# Binary Labeling Rules
BINARY_LABELING_RULES = {
    "traffic_absent": {
        "max_yolo_count": 6,  # Empty/light traffic (0-6 vehicles)
        "confidence": 0.90
    },
    "traffic_present": {
        "min_yolo_count": 7,  # Moderate/heavy traffic (7+ vehicles)
        "confidence": 0.75
    },
    "borderline": {
        "yolo_range": (4, 8),  # 4-8 vehicles = low confidence, flag for review
        "confidence": 0.60
    }
}

# Time Buckets (hour ranges and target samples per camera)
TIME_BUCKETS = {
    "night": {
        "hours": (22, 6),      # 22:00 - 05:59
        "samples": 140
    },
    "morning": {
        "hours": (6, 10),      # 06:00 - 09:59
        "samples": 140
    },
    "midday": {
        "hours": (10, 14),     # 10:00 - 13:59
        "samples": 140
    },
    "afternoon": {
        "hours": (14, 18),     # 14:00 - 17:59
        "samples": 140
    },
    "evening": {
        "hours": (18, 22),     # 18:00 - 21:59
        "samples": 140
    }
}

# Season Buckets (months and target samples per time bucket)
SEASON_BUCKETS = {
    "winter": {
        "months": ["12", "01", "02"],
        "samples": 35
    },
    "spring": {
        "months": ["03", "04", "05"],
        "samples": 35
    },
    "summer": {
        "months": ["06", "07", "08"],
        "samples": 35
    },
    "autumn": {
        "months": ["09", "10", "11"],
        "samples": 35
    }
}

# COCO Class Names (for reference)
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


def categorize_traffic(vehicle_count: int) -> str:
    """
    Categorize traffic level based on vehicle count.

    Args:
        vehicle_count: Number of vehicles detected by YOLO

    Returns:
        Traffic level category with 'likely_' prefix for auto-labeling
    """
    if vehicle_count <= 2:
        return "likely_empty"
    elif vehicle_count <= 6:
        return "likely_light"
    elif vehicle_count <= 15:
        return "likely_moderate"
    else:
        return "likely_heavy"


def get_time_bucket(hour: int) -> str:
    """
    Determine time bucket for a given hour.

    Args:
        hour: Hour of day (0-23)

    Returns:
        Time bucket name
    """
    for bucket_name, bucket_info in TIME_BUCKETS.items():
        start_hour, end_hour = bucket_info["hours"]

        # Handle wrap-around for night bucket (22:00 - 05:59)
        if start_hour > end_hour:
            if hour >= start_hour or hour < end_hour:
                return bucket_name
        else:
            if start_hour <= hour < end_hour:
                return bucket_name

    # Default fallback
    return "unknown"


def get_season(month: str) -> str:
    """
    Determine season for a given month.

    Args:
        month: Month as string (padded or non-padded, e.g., "07" or "7")

    Returns:
        Season name
    """
    # Normalize month to padded format
    month_padded = month.zfill(2)

    for season_name, season_info in SEASON_BUCKETS.items():
        if month_padded in season_info["months"]:
            return season_name

    # Default fallback
    return "unknown"


# Lane Detection Configuration
LANE_DETECTION_CONFIG = {
    "mode": "single",  # or "multi"
    "edge_detection": {
        "method": "canny",
        "adaptive_thresholds": True,
        "lower_percentile": 0.5,
        "upper_percentile": 1.5,
        "lower_threshold": 50,
        "upper_threshold": 150
    },
    "line_detection": {
        "method": "hough",
        "rho": 1,
        "theta_resolution": 180,  # degrees
        "threshold": 30,  # Lowered from 50 to detect more lines
        "min_line_length": 50,  # Lowered from 100 to catch shorter segments
        "max_line_gap": 30,  # Lowered from 50 for better continuity
        "vertical_angle_range": (60, 120),   # degrees
        "horizontal_angle_range": [(0, 30), (150, 180)]
    },
    "polygon_simplification": {
        "epsilon_factor": 0.02,  # % of perimeter
        "min_vertices": 3,
        "max_vertices": 12
    },
    "confidence_thresholds": {
        "high": 0.75,
        "medium": 0.50,
        "low": 0.30
    }
}


# Excluded Cameras (Corrupted Data or Poor Positioning)
# These cameras will be skipped during all pipeline phases
EXCLUDED_CAMERAS = [
    "VATIN_U",       # Corrupted data
    "VATIN_I",       # Corrupted data
    "GOSTUN_I",      # Corrupted data
    "TRBUSNICA_I",   # Poor positioning for visual analysis
    "TRBUSNICA_U",   # Poor positioning for visual analysis
    "PRESEVO_I",     # Poor positioning for visual analysis
    "PRESEVO_U",     # Poor positioning for visual analysis
    "MZVORNIK_I",    # Poor traffic_present ratio (23/216)
    "MZVORNIK_U",    # Poor traffic_present ratio
    "JABUKA_I",      # Poor traffic_present ratio (19/215)
    "JABUKA_U",      # Poor traffic_present ratio
]


def is_camera_excluded(camera_id: str) -> bool:
    """
    Check if a camera should be excluded from processing.

    Args:
        camera_id: Camera identifier (e.g., "VATIN_U")

    Returns:
        True if camera should be excluded, False otherwise
    """
    return camera_id in EXCLUDED_CAMERAS


def filter_excluded_cameras(camera_list: List[str]) -> List[str]:
    """
    Filter out excluded cameras from a list.

    Args:
        camera_list: List of camera identifiers

    Returns:
        Filtered list without excluded cameras
    """
    return [cam for cam in camera_list if not is_camera_excluded(cam)]
