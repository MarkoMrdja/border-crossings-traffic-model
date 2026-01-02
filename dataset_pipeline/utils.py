"""
Utility functions for the Border Crossings Traffic Dataset Pipeline

This module provides common utilities for path manipulation, JSON I/O,
progress tracking, and other helper functions used across the pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# Path Construction and Parsing
# ============================================================================

def parse_camera_id(camera_id: str) -> Tuple[str, str]:
    """
    Parse camera ID into border name and direction.

    Args:
        camera_id: Camera identifier (e.g., "GRADINA_U", "KELEBIJA_I")

    Returns:
        Tuple of (border_name, direction)

    Examples:
        >>> parse_camera_id("GRADINA_U")
        ("GRADINA", "U")
        >>> parse_camera_id("KELEBIJA_I")
        ("KELEBIJA", "I")
    """
    parts = camera_id.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid camera_id format: {camera_id}")

    border_name = parts[0]
    direction = parts[1]

    return border_name, direction


def construct_blob_path(
    camera_id: str,
    year: str,
    month: str,
    day: str,
    time_str: str
) -> str:
    """
    Construct Azure blob path from components.

    Args:
        camera_id: Camera identifier (e.g., "GRADINA_U")
        year: Year as string (e.g., "2024")
        month: Month as string, padded or non-padded (e.g., "07" or "7")
        day: Day as string, padded or non-padded (e.g., "15" or "1")
        time_str: Time string in format "HH-MM-SS" (e.g., "16-20-58")

    Returns:
        Full blob path (e.g., "GRADINA/U/2024/07/15/16-20-58.jpg")

    Examples:
        >>> construct_blob_path("GRADINA_U", "2024", "07", "15", "16-20-58")
        "GRADINA/U/2024/07/15/16-20-58.jpg"
    """
    border_name, direction = parse_camera_id(camera_id)
    return f"{border_name}/{direction}/{year}/{month}/{day}/{time_str}.jpg"


def construct_local_path(
    camera_id: str,
    year: str,
    month: str,
    day: str,
    time_str: str,
    base_subdir: str = "raw"
) -> str:
    """
    Construct local file path from components.

    Flattens the Azure hierarchical structure into camera-based folders
    with timestamp-based filenames.

    Args:
        camera_id: Camera identifier (e.g., "GRADINA_U")
        year: Year as string (e.g., "2024")
        month: Month as string, padded (e.g., "07")
        day: Day as string, padded (e.g., "15")
        time_str: Time string in format "HH-MM-SS" (e.g., "16-20-58")
        base_subdir: Base subdirectory (default: "raw")

    Returns:
        Relative local path (e.g., "raw/GRADINA_U/2024-07-15_16-20-58.jpg")

    Examples:
        >>> construct_local_path("GRADINA_U", "2024", "07", "15", "16-20-58")
        "raw/GRADINA_U/2024-07-15_16-20-58.jpg"
    """
    # Ensure month and day are zero-padded
    month_padded = month.zfill(2)
    day_padded = day.zfill(2)

    filename = f"{year}-{month_padded}-{day_padded}_{time_str}.jpg"
    return f"{base_subdir}/{camera_id}/{filename}"


def parse_blob_name(blob_name: str) -> Optional[Dict[str, str]]:
    """
    Parse Azure blob name into components.

    Args:
        blob_name: Full blob path (e.g., "GRADINA/U/2024/07/15/16-20-58.jpg")

    Returns:
        Dictionary with parsed components, or None if invalid format

    Examples:
        >>> parse_blob_name("GRADINA/U/2024/07/15/16-20-58.jpg")
        {
            'border': 'GRADINA',
            'direction': 'U',
            'camera_id': 'GRADINA_U',
            'year': '2024',
            'month': '07',
            'day': '15',
            'time': '16-20-58'
        }
    """
    try:
        parts = blob_name.split("/")
        if len(parts) != 6 or not parts[5].endswith(".jpg"):
            return None

        border = parts[0]
        direction = parts[1]
        year = parts[2]
        month = parts[3]
        day = parts[4]
        time_str = parts[5].replace(".jpg", "")

        return {
            "border": border,
            "direction": direction,
            "camera_id": f"{border}_{direction}",
            "year": year,
            "month": month,
            "day": day,
            "time": time_str
        }
    except (IndexError, ValueError):
        return None


# ============================================================================
# JSON I/O
# ============================================================================

def load_json(file_path: Path | str, default: Any = None) -> Any:
    """
    Load JSON data from a file with error handling.

    Args:
        file_path: Path to JSON file
        default: Default value to return if file doesn't exist or is invalid

    Returns:
        Parsed JSON data, or default value if loading fails

    Examples:
        >>> data = load_json("config.json", default={})
        >>> inventory = load_json(Path("inventory.json"), default={})
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"JSON file not found: {file_path}")
        return default

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return default
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return default


def save_json(data: Any, file_path: Path | str, indent: int = 2) -> bool:
    """
    Save data to JSON file with pretty printing.

    Args:
        data: Data to serialize to JSON
        file_path: Path to output JSON file
        indent: Number of spaces for indentation (default: 2)

    Returns:
        True if successful, False otherwise

    Examples:
        >>> save_json({"key": "value"}, "output.json")
        True
    """
    file_path = Path(file_path)

    try:
        # Create parent directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        logger.debug(f"Saved JSON to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


# ============================================================================
# Progress Tracking
# ============================================================================

def create_progress_tracker(
    total: int,
    description: str = "Processing"
) -> Dict[str, Any]:
    """
    Create a new progress tracking dictionary.

    Args:
        total: Total number of items to process
        description: Description of the task

    Returns:
        Progress tracker dictionary

    Examples:
        >>> tracker = create_progress_tracker(1000, "Downloading images")
        >>> tracker["completed"]
        0
    """
    return {
        "description": description,
        "total": total,
        "completed": 0,
        "failed": [],
        "last_index": -1,
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }


def update_progress(
    progress_file: Path | str,
    updates: Dict[str, Any],
    auto_save: bool = True
) -> Dict[str, Any]:
    """
    Update progress tracker and optionally save to file.

    Args:
        progress_file: Path to progress JSON file
        updates: Dictionary of updates to apply
        auto_save: Whether to automatically save after update

    Returns:
        Updated progress dictionary

    Examples:
        >>> progress = update_progress(
        ...     "download_progress.json",
        ...     {"completed": 150, "last_index": 149}
        ... )
    """
    progress_file = Path(progress_file)

    # Load existing progress or create new
    progress = load_json(progress_file, default={})

    # Apply updates
    progress.update(updates)
    progress["updated_at"] = datetime.now().isoformat()

    # Save if requested
    if auto_save:
        save_json(progress, progress_file)

    return progress


def log_progress(progress: Dict[str, Any], interval: int = 100) -> None:
    """
    Log progress information if at logging interval.

    Args:
        progress: Progress dictionary
        interval: Log every N items (default: 100)

    Examples:
        >>> progress = {"completed": 500, "total": 1000, "description": "Processing"}
        >>> log_progress(progress)
        # Logs: "Processing: 500/1000 (50.0%)"
    """
    completed = progress.get("completed", 0)
    total = progress.get("total", 0)
    description = progress.get("description", "Progress")

    if completed % interval == 0 or completed == total:
        if total > 0:
            percentage = (completed / total) * 100
            logger.info(f"{description}: {completed}/{total} ({percentage:.1f}%)")
        else:
            logger.info(f"{description}: {completed} completed")


# ============================================================================
# File System Utilities
# ============================================================================

def ensure_directory(path: Path | str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to directory

    Returns:
        Path object for the directory

    Examples:
        >>> ensure_directory("./data/output")
        Path('./data/output')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_mb(file_path: Path | str) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB, or 0.0 if file doesn't exist

    Examples:
        >>> size = get_file_size_mb("large_file.jpg")
        >>> print(f"Size: {size:.2f} MB")
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return 0.0

    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)


def count_files(directory: Path | str, pattern: str = "*") -> int:
    """
    Count files matching a pattern in a directory.

    Args:
        directory: Path to directory
        pattern: Glob pattern (default: "*" for all files)

    Returns:
        Number of matching files

    Examples:
        >>> count_files("./crops/likely_heavy", "*.jpg")
        125
    """
    directory = Path(directory)
    if not directory.exists():
        return 0

    return len(list(directory.glob(pattern)))


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_image_path(file_path: Path | str, min_size_bytes: int = 1000) -> bool:
    """
    Validate that an image file exists and has reasonable size.

    Args:
        file_path: Path to image file
        min_size_bytes: Minimum expected file size in bytes

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_image_path("image.jpg", min_size_bytes=1000)
        True
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"Image file not found: {file_path}")
        return False

    if not file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        logger.warning(f"Invalid image extension: {file_path}")
        return False

    if file_path.stat().st_size < min_size_bytes:
        logger.warning(f"Image file suspiciously small: {file_path}")
        return False

    return True
