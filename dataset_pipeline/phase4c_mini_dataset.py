"""
Phase 4c: Mini Test Dataset Creation

Creates a mini dataset with 64 test images (2 per camera: heaviest + emptiest traffic).
This dataset is used to validate lane polygon quality in extreme traffic conditions.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

from .base import PipelinePhase
from .config import PipelineConfig
from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class MiniDatasetPhase(PipelinePhase):
    """
    Phase 4c: Create mini test dataset.

    Selects 2 images per camera (32 cameras = 64 images total):
    - One image with maximum vehicle count (heaviest traffic)
    - One image with minimum vehicle count (emptiest traffic)

    Purpose: Validate lane polygons work in both extreme scenarios.
    """

    def __init__(self, pipeline_config: PipelineConfig):
        """
        Initialize mini dataset creation phase.

        Args:
            pipeline_config: Pipeline configuration
        """
        super().__init__(
            config=pipeline_config,
            phase_name="mini_dataset",
            description="Creating mini test dataset"
        )

        self.yolo_results_path = self.config.base_dir / self.config.yolo_results_file
        self.mini_dataset_dir = self.config.base_dir / "mini_test_dataset"
        self.manifest_path = self.mini_dataset_dir / "mini_dataset_manifest.json"

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Execute mini dataset creation.

        Args:
            resume: If True, skip already copied images

        Returns:
            Dictionary with mini dataset manifest
        """
        # Load YOLO results
        self.logger.info(f"Loading YOLO results from {self.yolo_results_path}")
        if not self.yolo_results_path.exists():
            raise FileNotFoundError(
                f"YOLO results not found at {self.yolo_results_path}. "
                "Run Phase 2 (YOLO analysis) first."
            )

        yolo_data = load_json(self.yolo_results_path)
        analyses = yolo_data.get('analyses', [])

        if not analyses:
            raise ValueError("No analyses found in YOLO results")

        self.logger.info(f"Loaded {len(analyses)} YOLO analyses")

        # Group analyses by camera
        self.logger.info("Grouping analyses by camera...")
        by_camera = defaultdict(list)
        for analysis in analyses:
            camera_id = analysis.get('camera_id')
            if camera_id:
                by_camera[camera_id].append(analysis)

        num_cameras = len(by_camera)
        self.logger.info(f"Found {num_cameras} cameras")

        # Create output directory
        self.mini_dataset_dir.mkdir(parents=True, exist_ok=True)

        # Initialize manifest
        manifest = {
            'created_at': datetime.now().isoformat(),
            'purpose': 'Test lane polygons in extreme traffic conditions',
            'total_images': 0,
            'total_cameras': num_cameras,
            'cameras': {}
        }

        # Select images for each camera
        self.logger.info("Selecting extreme traffic images for each camera...")

        for camera_id, camera_analyses in sorted(by_camera.items()):
            self.logger.info(f"Processing camera: {camera_id} ({len(camera_analyses)} images)")

            # Sort by vehicle count
            sorted_analyses = sorted(camera_analyses, key=lambda x: x['vehicle_count'])

            # Get emptiest (min vehicle count)
            emptiest = sorted_analyses[0]
            emptiest_count = emptiest['vehicle_count']

            # Get heaviest (max vehicle count)
            heaviest = sorted_analyses[-1]
            heaviest_count = heaviest['vehicle_count']

            self.logger.info(
                f"  Emptiest: {emptiest_count} vehicles, "
                f"Heaviest: {heaviest_count} vehicles"
            )

            # Copy images to mini dataset
            emptiest_dest = self._copy_image(
                emptiest,
                camera_id,
                'EMPTY',
                resume
            )

            heaviest_dest = self._copy_image(
                heaviest,
                camera_id,
                'HEAVY',
                resume
            )

            # Add to manifest
            manifest['cameras'][camera_id] = {
                'emptiest': {
                    'local_path': str(emptiest_dest.relative_to(self.config.base_dir)) if emptiest_dest else None,
                    'vehicle_count': emptiest_count,
                    'traffic_level': emptiest['traffic_level'],
                    'original_path': emptiest['local_path']
                },
                'heaviest': {
                    'local_path': str(heaviest_dest.relative_to(self.config.base_dir)) if heaviest_dest else None,
                    'vehicle_count': heaviest_count,
                    'traffic_level': heaviest['traffic_level'],
                    'original_path': heaviest['local_path']
                }
            }

            manifest['total_images'] += 2

        # Save manifest
        self.logger.info(f"Saving manifest to {self.manifest_path}")
        save_json(manifest, self.manifest_path)

        # Log summary
        self._log_summary(manifest)

        return manifest

    def _copy_image(
        self,
        analysis: Dict[str, Any],
        camera_id: str,
        label: str,
        resume: bool
    ) -> Path:
        """
        Copy image to mini dataset directory.

        Args:
            analysis: YOLO analysis containing local_path
            camera_id: Camera identifier
            label: Label for filename (e.g., 'EMPTY' or 'HEAVY')
            resume: If True, skip if file already exists

        Returns:
            Destination path
        """
        # Get source path
        source_path = self.config.base_dir / analysis['local_path']

        if not source_path.exists():
            self.logger.warning(f"Source image not found: {source_path}")
            return None

        # Generate destination filename
        dest_filename = f"{camera_id}_{label}.jpg"
        dest_path = self.mini_dataset_dir / dest_filename

        # Skip if already exists and resuming
        if resume and dest_path.exists():
            self.logger.debug(f"Skipping (already exists): {dest_filename}")
            return dest_path

        # Copy file
        try:
            shutil.copy2(source_path, dest_path)
            self.logger.debug(f"Copied: {dest_filename}")
            return dest_path
        except Exception as e:
            self.logger.error(f"Failed to copy {source_path} to {dest_path}: {e}")
            return None

    def _log_summary(self, manifest: Dict[str, Any]):
        """
        Log summary of mini dataset creation.

        Args:
            manifest: Mini dataset manifest
        """
        self.logger.info("=" * 60)
        self.logger.info("MINI DATASET CREATION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total cameras: {manifest['total_cameras']}")
        self.logger.info(f"Total images: {manifest['total_images']}")
        self.logger.info(f"Output directory: {self.mini_dataset_dir}")
        self.logger.info(f"Manifest: {self.manifest_path}")

        # Count by traffic level
        empty_counts = []
        heavy_counts = []

        for camera_id, camera_data in manifest['cameras'].items():
            empty_counts.append(camera_data['emptiest']['vehicle_count'])
            heavy_counts.append(camera_data['heaviest']['vehicle_count'])

        self.logger.info(f"\nVehicle count ranges:")
        self.logger.info(f"  Emptiest images: {min(empty_counts)} - {max(empty_counts)} vehicles")
        self.logger.info(f"  Heaviest images: {min(heavy_counts)} - {max(heavy_counts)} vehicles")
        self.logger.info("=" * 60)

    def validate(self) -> bool:
        """
        Validate that mini dataset was created successfully.

        Checks:
        - Manifest exists
        - Directory contains 64 images (2 per camera)
        - All referenced images exist

        Returns:
            True if validation passed, False otherwise
        """
        # Check manifest exists
        if not self.manifest_path.exists():
            self.logger.error(f"Manifest not found: {self.manifest_path}")
            return False

        # Load manifest
        try:
            manifest = load_json(self.manifest_path)
        except Exception as e:
            self.logger.error(f"Failed to load manifest: {e}")
            return False

        # Check expected structure
        if 'cameras' not in manifest:
            self.logger.error("Manifest missing 'cameras' field")
            return False

        # Count images in directory
        image_files = list(self.mini_dataset_dir.glob("*.jpg"))
        num_files = len(image_files)

        expected_images = manifest['total_cameras'] * 2

        if num_files != expected_images:
            self.logger.error(
                f"Image count mismatch: expected {expected_images}, found {num_files}"
            )
            return False

        # Verify all referenced images exist
        missing_images = []
        for camera_id, camera_data in manifest['cameras'].items():
            for key in ['emptiest', 'heaviest']:
                local_path = camera_data[key]['local_path']
                if local_path:
                    full_path = self.config.base_dir / local_path
                    if not full_path.exists():
                        missing_images.append(local_path)

        if missing_images:
            self.logger.error(f"Missing {len(missing_images)} referenced images:")
            for path in missing_images[:5]:  # Show first 5
                self.logger.error(f"  {path}")
            return False

        self.logger.info(
            f"Validation passed: {num_files} images for {manifest['total_cameras']} cameras"
        )
        return True


def main():
    """CLI entry point for Phase 4c."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 4c: Create mini test dataset for lane polygon validation"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images that are already copied"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation, don't create dataset"
    )

    args = parser.parse_args()

    # Initialize configuration
    config = PipelineConfig()

    # Create phase instance
    phase = MiniDatasetPhase(config)

    if args.validate_only:
        # Run validation only
        print("Running validation...")
        success = phase.validate()
        if success:
            print("✓ Validation passed")
            exit(0)
        else:
            print("✗ Validation failed")
            exit(1)
    else:
        # Execute phase
        result = phase.execute(resume=args.resume, validate_after=True)

        if result['status'] == 'completed':
            print(f"\n✓ Phase 4c completed successfully in {result['duration_seconds']:.1f} seconds")
            manifest = result['data']
            print(f"  Total cameras: {manifest['total_cameras']}")
            print(f"  Total images: {manifest['total_images']}")
            print(f"  Output: {config.base_dir / 'mini_test_dataset'}")
            exit(0)
        else:
            print(f"\n✗ Phase 4c failed: {result.get('reason', 'unknown error')}")
            exit(1)


if __name__ == "__main__":
    main()
