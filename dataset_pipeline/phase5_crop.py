"""
Phase 5: Batch Crop

Crops ROI regions from all selected images and resizes to 64x64.
Organizes cropped images by predicted traffic level.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm

from .base import PipelinePhase
from .config import PipelineConfig, TRAFFIC_LEVELS
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class CropPhase(PipelinePhase):
    """
    Phase 5: Batch crop ROI regions.

    For each selected image:
    1. Load the corresponding ROI polygon for that camera
    2. Crop the polygon region from the image
    3. Apply mask (black outside polygon)
    4. Resize to 64x64
    5. Save to crops/likely_{level}/ based on YOLO prediction
    """

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize crop phase.

        Args:
            pipeline_config: Pipeline configuration
        """
        super().__init__(
            config=pipeline_config,
            phase_name="crop",
            description="Cropping ROI regions from images"
        )

        self.balanced_selection_path = self.config.base_dir / "balanced_selection.json"
        self.roi_config_path = self.config.base_dir / self.config.roi_config_file
        self.yolo_results_path = self.config.base_dir / self.config.yolo_results_file
        self.crops_dir = self.config.base_dir / self.config.crops_dir
        self.output_summary_path = self.config.base_dir / "crop_summary.json"

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Execute batch cropping.

        Args:
            resume: If True, skip already cropped images

        Returns:
            Dictionary with cropping results and statistics
        """
        # Load required data files
        self.logger.info("Loading input data...")
        selection = self._load_balanced_selection()
        roi_config = self._load_roi_config()
        yolo_results = self._load_yolo_results()

        # Build lookup for YOLO results
        yolo_lookup = {a['local_path']: a for a in yolo_results.get('analyses', [])}

        # Ensure output directories exist
        self._ensure_crop_directories()

        # Initialize progress tracking
        if resume and self.progress_file.exists():
            self.load_progress()
            processed_files = set(self.progress.get('processed_files', []))
            self.logger.info(f"Resuming: {len(processed_files)} files already processed")
        else:
            processed_files = set()
            self.initialize_progress(
                total=len(selection['selection']),
                processed=0,
                failed=0,
                skipped=0,
                processed_files=[]
            )

        # Statistics
        stats = {
            'total': len(selection['selection']),
            'processed': len(processed_files),
            'failed': 0,
            'skipped': 0,
            'skipped_no_roi': 0,
            'skipped_missing_image': 0,
            'by_traffic_level': {level: 0 for level in TRAFFIC_LEVELS}
        }

        # Process each selected image
        self.logger.info(f"Processing {stats['total']} images...")

        with tqdm(total=stats['total'], initial=len(processed_files), desc="Cropping images") as pbar:
            for item in selection['selection']:
                camera_id = item['camera_id']
                local_path = item['local_path']

                # Skip if already processed
                if local_path in processed_files:
                    pbar.update(1)
                    continue

                # Skip if no ROI defined for this camera
                if camera_id not in roi_config.get('cameras', {}):
                    self.logger.warning(f"No ROI defined for camera {camera_id}, skipping {local_path}")
                    stats['skipped'] += 1
                    stats['skipped_no_roi'] += 1
                    processed_files.add(local_path)
                    pbar.update(1)
                    continue

                # Get full image path
                full_path = self.config.base_dir / local_path
                if not full_path.exists():
                    self.logger.warning(f"Image not found: {full_path}")
                    stats['skipped'] += 1
                    stats['skipped_missing_image'] += 1
                    processed_files.add(local_path)
                    pbar.update(1)
                    continue

                try:
                    # Get polygon for this camera
                    polygon = roi_config['cameras'][camera_id]['polygon']

                    # Crop ROI
                    cropped = self._crop_roi(
                        str(full_path),
                        polygon,
                        output_size=self.config.crop_size
                    )

                    # Determine traffic level from YOLO results
                    yolo_data = yolo_lookup.get(local_path, {})
                    traffic_level = yolo_data.get('traffic_level', 'likely_empty')

                    # Remove 'likely_' prefix if present to get the actual level
                    if traffic_level.startswith('likely_'):
                        level_name = traffic_level.replace('likely_', '')
                    else:
                        level_name = traffic_level

                    # Generate output filename
                    # Format: CAMERA_YYYY-MM-DD_HH-MM-SS.jpg
                    filename = f"{camera_id}_{Path(local_path).name}"
                    output_path = self.crops_dir / traffic_level / filename

                    # Save cropped image
                    cv2.imwrite(str(output_path), cropped)

                    # Update statistics
                    stats['processed'] += 1
                    if level_name in stats['by_traffic_level']:
                        stats['by_traffic_level'][level_name] += 1

                    processed_files.add(local_path)

                except Exception as e:
                    self.logger.error(f"Failed to crop {local_path}: {e}")
                    stats['failed'] += 1

                # Update progress periodically
                if len(processed_files) % 100 == 0:
                    self.progress['processed'] = stats['processed']
                    self.progress['failed'] = stats['failed']
                    self.progress['skipped'] = stats['skipped']
                    self.progress['processed_files'] = list(processed_files)
                    self.save_progress()

                pbar.update(1)

        # Final progress save
        self.progress['processed'] = stats['processed']
        self.progress['failed'] = stats['failed']
        self.progress['skipped'] = stats['skipped']
        self.progress['processed_files'] = list(processed_files)
        self.progress['completed_at'] = datetime.now().isoformat()
        self.save_progress()

        # Save summary
        summary = {
            'completed_at': datetime.now().isoformat(),
            'crop_size': self.config.crop_size,
            'statistics': stats
        }
        save_json(summary, self.output_summary_path)

        self.logger.info(f"Cropping completed:")
        self.logger.info(f"  Processed: {stats['processed']}")
        self.logger.info(f"  Failed: {stats['failed']}")
        self.logger.info(f"  Skipped: {stats['skipped']} (no ROI: {stats['skipped_no_roi']}, missing: {stats['skipped_missing_image']})")
        self.logger.info(f"  By traffic level: {stats['by_traffic_level']}")

        return summary

    def _load_balanced_selection(self) -> Dict[str, Any]:
        """Load balanced selection from Phase 3."""
        if not self.balanced_selection_path.exists():
            raise FileNotFoundError(
                f"Balanced selection not found at {self.balanced_selection_path}. "
                "Run Phase 3 (balanced selection) first."
            )
        return load_json(self.balanced_selection_path)

    def _load_roi_config(self) -> Dict[str, Any]:
        """Load ROI configuration from Phase 4."""
        if not self.roi_config_path.exists():
            raise FileNotFoundError(
                f"ROI configuration not found at {self.roi_config_path}. "
                "Run Phase 4 (ROI definition) first."
            )
        return load_json(self.roi_config_path)

    def _load_yolo_results(self) -> Dict[str, Any]:
        """Load YOLO results from Phase 2."""
        if not self.yolo_results_path.exists():
            raise FileNotFoundError(
                f"YOLO results not found at {self.yolo_results_path}. "
                "Run Phase 2 (YOLO analysis) first."
            )
        return load_json(self.yolo_results_path)

    def _ensure_crop_directories(self):
        """Create crop output directories if they don't exist."""
        for level in TRAFFIC_LEVELS:
            level_dir = self.crops_dir / f"likely_{level}"
            level_dir.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {level_dir}")

    def _crop_roi(
        self,
        image_path: str,
        polygon: List[List[int]],
        output_size: int = 64
    ) -> np.ndarray:
        """
        Crop polygon region from image.

        Process:
        1. Calculate bounding box of polygon
        2. Crop bounding box from image
        3. Create mask for polygon
        4. Apply mask (black outside polygon)
        5. Resize to output_size × output_size

        Args:
            image_path: Path to source image
            polygon: List of [x, y] coordinates defining the polygon
            output_size: Target size for output (default: 64)

        Returns:
            Cropped and resized image as numpy array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert polygon to numpy array
        polygon_np = np.array(polygon, dtype=np.int32)

        # Get bounding box
        x_min, y_min = polygon_np.min(axis=0)
        x_max, y_max = polygon_np.max(axis=0)

        # Clamp to image bounds
        h, w = image.shape[:2]
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(w, int(x_max))
        y_max = min(h, int(y_max))

        # Crop bounding box
        cropped = image[y_min:y_max, x_min:x_max].copy()

        # Create mask for polygon
        mask_h = y_max - y_min
        mask_w = x_max - x_min
        mask = np.zeros((mask_h, mask_w), dtype=np.uint8)

        # Shift polygon coordinates to cropped image space
        shifted_polygon = polygon_np - [x_min, y_min]

        # Fill polygon in mask
        cv2.fillPoly(mask, [shifted_polygon.astype(np.int32)], 255)

        # Apply mask (black outside polygon)
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        cropped = (cropped * mask_3ch).astype(np.uint8)

        # Resize to target size
        resized = cv2.resize(
            cropped,
            (output_size, output_size),
            interpolation=cv2.INTER_AREA
        )

        return resized

    def validate(self) -> bool:
        """
        Validate that cropping completed successfully.

        Checks:
        - Output summary exists
        - Crops directory has images
        - Statistics look reasonable

        Returns:
            True if validation passed, False otherwise
        """
        # Check summary exists
        if not self.output_summary_path.exists():
            self.logger.error("Crop summary file not found")
            return False

        # Load summary
        summary = load_json(self.output_summary_path)
        stats = summary.get('statistics', {})

        # Check that some images were processed
        if stats.get('processed', 0) == 0:
            self.logger.error("No images were processed")
            return False

        # Count actual files in crops directories
        total_files = 0
        for level in TRAFFIC_LEVELS:
            level_dir = self.crops_dir / f"likely_{level}"
            if level_dir.exists():
                files = list(level_dir.glob("*.jpg"))
                total_files += len(files)
                self.logger.info(f"  {level_dir.name}: {len(files)} images")

        # Verify file count matches processed count
        if total_files != stats['processed']:
            self.logger.warning(
                f"File count mismatch: {total_files} files on disk, "
                f"{stats['processed']} in summary"
            )

        self.logger.info(f"Validation passed: {total_files} cropped images found")
        return True


def main():
    """CLI entry point for Phase 5."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 5: Batch crop ROI regions from images"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation, don't process images"
    )

    args = parser.parse_args()

    # Initialize configuration
    config = PipelineConfig()

    # Create phase instance
    phase = CropPhase(config)

    if args.validate_only:
        # Run validation only
        print("Running validation...")
        success = phase.validate()
        exit(0 if success else 1)
    else:
        # Execute phase
        result = phase.execute(resume=args.resume, validate_after=True)

        if result['status'] == 'completed':
            print(f"\n✓ Phase 5 completed successfully in {result['duration_seconds']:.1f} seconds")
            summary = result['data']
            stats = summary['statistics']
            print(f"  Processed: {stats['processed']}")
            print(f"  Failed: {stats['failed']}")
            print(f"  Skipped: {stats['skipped']}")
            print(f"\nCropped images by traffic level:")
            for level, count in stats['by_traffic_level'].items():
                print(f"  {level}: {count}")
        else:
            print(f"\n✗ Phase 5 failed: {result.get('reason', 'unknown error')}")
            exit(1)


if __name__ == "__main__":
    main()
